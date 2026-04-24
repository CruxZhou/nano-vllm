import os
import time
import argparse
import numpy as np
from random import randint, seed
from tqdm.auto import tqdm

from nanovllm import LLM, SamplingParams


def percentile(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


class RequestMetrics:
    def __init__(self, request_id, input_len, submission_time):
        self.request_id = request_id
        self.input_len = input_len
        self.submission_time = submission_time
        self.first_token_time = None
        self.completion_time = None
        self.output_len = 0

    def record_first_token(self, t=None):
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter() if t is None else t

    def record_completion(self, output_ids):
        self.completion_time = time.perf_counter()
        self.output_len = len(output_ids)

    @property
    def ttft(self):
        if self.first_token_time is None:
            return float("nan")
        return self.first_token_time - self.submission_time

    @property
    def latency(self):
        if self.completion_time is None:
            return float("nan")
        return self.completion_time - self.submission_time

    @property
    def tpot(self):
        if self.first_token_time is None or self.completion_time is None:
            return float("nan")
        if self.output_len <= 1:
            return float("nan")
        return (self.completion_time - self.first_token_time) / (self.output_len - 1)


def add_request(engine, prompt, sampling_params):
    """
    适配 nano-vllm 当前结构。
    如果 LLM 直接有 add_request，就用 LLM.add_request。
    否则用 scheduler.add + Sequence 手动加。
    """
    if hasattr(engine, "add_request"):
        engine.add_request(prompt, sampling_params)
        return engine.scheduler.waiting[-1]

    from nanovllm.engine.sequence import Sequence

    seq = Sequence(prompt, sampling_params)
    engine.scheduler.add(seq)
    return seq


def warmup(engine):
    engine.generate(["Benchmark: "], SamplingParams())
    if hasattr(engine, "last_ttfts"):
        engine.last_ttfts.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--num-requests", type=int, default=1024)
    parser.add_argument("--request-rate", type=float, default=8.0)
    parser.add_argument("--random-input-len", type=int, default=512)
    parser.add_argument("--random-output-len", type=int, default=100)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--chunked-prefill", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    seed(args.seed)
    np.random.seed(args.seed)

    print("========== Serving Benchmark Config ==========")
    print(f"model: {args.model}")
    print(f"num_requests: {args.num_requests}")
    print(f"request_rate: {args.request_rate} req/s")
    print(f"random_input_len: {args.random_input_len}")
    print(f"random_output_len: {args.random_output_len}")
    print(f"max_num_batched_tokens: {args.max_num_batched_tokens}")
    print(f"max_num_seqs: {args.max_num_seqs}")
    print(f"chunked_prefill: {args.chunked_prefill}")
    print("=============================================")

    engine = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=max(4096, args.random_input_len + args.random_output_len + 16),
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        chunked_prefill=args.chunked_prefill,
    )

    warmup(engine)

    prompts = [
        [randint(0, 10000) for _ in range(args.random_input_len)]
        for _ in range(args.num_requests)
    ]

    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=args.random_output_len,
        )
        for _ in range(args.num_requests)
    ]

    request_intervals = np.random.exponential(
        1.0 / args.request_rate,
        args.num_requests,
    )
    arrival_times = np.cumsum(request_intervals)

    metrics = {}
    requests_sent = 0
    completed = 0
    start_time = time.perf_counter()

    with tqdm(total=args.num_requests, desc="Processing Requests") as pbar:
        while completed < args.num_requests:
            now = time.perf_counter()
            elapsed = now - start_time

            while (
                requests_sent < args.num_requests
                and elapsed >= arrival_times[requests_sent]
            ):
                seq = add_request(
                    engine,
                    prompts[requests_sent],
                    sampling_params[requests_sent],
                )

                seq_id = seq.seq_id
                submit_time = start_time + arrival_times[requests_sent]
                metrics[seq_id] = RequestMetrics(
                    request_id=seq_id,
                    input_len=len(prompts[requests_sent]),
                    submission_time=submit_time,
                )

                requests_sent += 1

            if not engine.is_finished():
                outputs, _ = engine.step()

                # outputs 兼容两种格式：
                # old: (seq_id, output_ids)
                # new: (seq_id, output_ids, ttft)
                for item in outputs:
                    if len(item) == 3:
                        seq_id, output_ids, engine_ttft = item
                    else:
                        seq_id, output_ids = item
                        engine_ttft = None

                    if seq_id not in metrics:
                        continue

                    m = metrics[seq_id]

                    if engine_ttft is not None:
                        m.first_token_time = m.submission_time + engine_ttft
                    else:
                        m.record_first_token()

                    m.record_completion(output_ids)
                    completed += 1

                    latencies = [
                        x.latency for x in metrics.values()
                        if not np.isnan(x.latency)
                    ]
                    pbar.set_postfix({
                        "sent": requests_sent,
                        "avg_lat": f"{np.mean(latencies):.2f}s" if latencies else "nan",
                    })
                    pbar.update(1)
            else:
                time.sleep(0.001)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    ttfts = [m.ttft for m in metrics.values() if not np.isnan(m.ttft)]
    tpots = [m.tpot for m in metrics.values() if not np.isnan(m.tpot)]
    latencies = [m.latency for m in metrics.values() if not np.isnan(m.latency)]

    total_input_tokens = sum(m.input_len for m in metrics.values())
    total_output_tokens = sum(m.output_len for m in metrics.values())
    total_tokens = total_input_tokens + total_output_tokens

    print("\n========== Serving Benchmark Result ==========")
    print(f"Total time: {total_time:.2f} s")
    print(f"Requests sent: {requests_sent}")
    print(f"Requests completed: {completed}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Throughput: {total_tokens / total_time:.2f} tok/s")
    print(f"Output throughput: {total_output_tokens / total_time:.2f} tok/s")

    print(
        "TTFT(ms) | "
        f"avg={np.mean(ttfts) * 1000:.2f}, "
        f"p50={percentile(ttfts, 50) * 1000:.2f}, "
        f"p90={percentile(ttfts, 90) * 1000:.2f}, "
        f"p95={percentile(ttfts, 95) * 1000:.2f}, "
        f"p99={percentile(ttfts, 99) * 1000:.2f}"
    )

    print(
        "TPOT(ms) | "
        f"avg={np.mean(tpots) * 1000:.2f}, "
        f"p50={percentile(tpots, 50) * 1000:.2f}, "
        f"p90={percentile(tpots, 90) * 1000:.2f}, "
        f"p95={percentile(tpots, 95) * 1000:.2f}, "
        f"p99={percentile(tpots, 99) * 1000:.2f}"
    )

    print(
        "Latency(s) | "
        f"avg={np.mean(latencies):.2f}, "
        f"p50={percentile(latencies, 50):.2f}, "
        f"p90={percentile(latencies, 90):.2f}, "
        f"p95={percentile(latencies, 95):.2f}, "
        f"p99={percentile(latencies, 99):.2f}"
    )
    print("==============================================")
    

if __name__ == "__main__":
    main()
