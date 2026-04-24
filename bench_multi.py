import os
import time
import argparse
import numpy as np
from random import randint, seed
from tqdm.auto import tqdm
from nanovllm import LLM, SamplingParams


seed(100)
np.random.seed(100)


def percentile(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] * (c - k) + values[c] * (k - f)


class RequestMetrics:
    def __init__(self, request_id, input_len):
        self.request_id = request_id
        self.input_len = input_len
        self.submission_time = -1.0
        self.first_token_time = -1.0
        self.completion_time = -1.0
        self.output_len = -1

    def record_first_token(self, timestamp=None):
        if self.first_token_time < 0:
            self.first_token_time = time.perf_counter() if timestamp is None else timestamp

    def record_completion(self, output_ids, timestamp=None):
        self.completion_time = time.perf_counter() if timestamp is None else timestamp
        self.output_len = len(output_ids)

    @property
    def ttft(self):
        if self.first_token_time < 0 or self.submission_time < 0:
            return float("nan")
        return self.first_token_time - self.submission_time

    @property
    def tpot(self):
        if self.output_len > 1 and self.first_token_time >= 0 and self.completion_time >= 0:
            return (self.completion_time - self.first_token_time) / (self.output_len - 1)
        return float("nan")

    @property
    def latency(self):
        if self.completion_time < 0 or self.submission_time < 0:
            return float("nan")
        return self.completion_time - self.submission_time


def parse_finished_output(item):
    """
    兼容不同版本的 engine.step() 返回格式。

    支持：
    1. (seq_id, output_ids)
    2. (seq_id, output_ids, extra)
    """
    if not isinstance(item, (tuple, list)):
        raise TypeError(f"Unexpected finished output type: {type(item)}")

    if len(item) < 2:
        raise ValueError(f"Unexpected finished output format: {item}")

    seq_id = item[0]
    output_ids = item[1]
    extra = item[2:] if len(item) > 2 else None

    return seq_id, output_ids, extra


def warm_up(engine, input_len, output_len, num_warmup=8):
    prompts = [
        [randint(0, 10000) for _ in range(input_len)]
        for _ in range(num_warmup)
    ]

    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=output_len,
        )
        for _ in range(num_warmup)
    ]

    engine.generate(prompts, sampling_params, use_tqdm=False)


def main():
    parser = argparse.ArgumentParser(
        description="Online serving benchmark for nano-vllm without chunked prefill."
    )

    parser.add_argument("--model", type=str, default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--num-requests", type=int, default=1024)
    parser.add_argument("--request-rate", type=float, default=10.0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--random-input-len", type=int, default=128)
    parser.add_argument("--random-output-len", type=int, default=100)

    # 关键修改：默认不要设成 4096，否则和 max_num_batched_tokens=1024 冲突。
    parser.add_argument("--max-model-len", type=int, default=512)

    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--no-warmup", action="store_true", default=False)

    args = parser.parse_args()
    model_path = os.path.expanduser(args.model)

    if args.random_input_len + args.random_output_len > args.max_model_len:
        raise ValueError(
            f"max_model_len too small: {args.max_model_len}. "
            f"It should be >= random_input_len + random_output_len "
            f"= {args.random_input_len + args.random_output_len}."
        )

    if args.max_num_batched_tokens < args.max_model_len:
        raise ValueError(
            f"max_num_batched_tokens must be >= max_model_len, "
            f"but got max_num_batched_tokens={args.max_num_batched_tokens}, "
            f"max_model_len={args.max_model_len}."
        )

    print("\n--- Running nano-vllm online benchmark without chunked prefill ---")
    print(f"Model: {model_path}")
    print(f"Requests: {args.num_requests}")
    print(f"Request rate: {args.request_rate} req/s")
    print(f"Input length: {args.random_input_len}")
    print(f"Output length: {args.random_output_len}")
    print(f"Max model len: {args.max_model_len}")
    print(f"Max batched tokens: {args.max_num_batched_tokens}")
    print(f"Max seqs: {args.max_num_seqs}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Enforce eager: {args.enforce_eager}")
    print("Chunked prefill: False")
    print("-----------------------------------------------\n")

    llm = LLM(
        model_path,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    engine = llm

    if not args.no_warmup:
        warm_up(
            engine,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_warmup=8,
        )

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
    completed_latencies = []

    start_time = time.perf_counter()

    with tqdm(total=args.num_requests, desc="Processing Requests") as pbar:
        while requests_sent < args.num_requests or not engine.is_finished():
            now = time.perf_counter()

            while (
                requests_sent < args.num_requests
                and now - start_time >= arrival_times[requests_sent]
            ):
                prompt = prompts[requests_sent]
                sp = sampling_params[requests_sent]

                engine.add_request(prompt, sp)

                new_seq = engine.scheduler.waiting[-1]
                seq_id = new_seq.seq_id

                req_metrics = RequestMetrics(seq_id, len(prompt))

                # 为了和 chunked-prefill 版本公平对比，用计划到达时间。
                req_metrics.submission_time = start_time + arrival_times[requests_sent]

                metrics[seq_id] = req_metrics
                requests_sent += 1

                # 更新 now，避免一次循环中时间长期不变。
                now = time.perf_counter()

            if not engine.is_finished():
                step_start = time.perf_counter()
                finished_outputs, _ = engine.step()
                step_end = time.perf_counter()

                # 记录 first token time。
                #
                # 注意：
                # 这里仍然是近似 TTFT，不是真正的 token flush 时间。
                # 但相比完成时兜底记录 first_token，它至少不会让 TPOT 变成 0。
                running_seqs = list(engine.scheduler.running)
                for seq in running_seqs:
                    if seq.seq_id not in metrics:
                        continue

                    if metrics[seq.seq_id].first_token_time >= 0:
                        continue

                    if hasattr(seq, "num_cached_tokens") and hasattr(seq, "num_prompt_tokens"):
                        if seq.num_cached_tokens == seq.num_prompt_tokens:
                            metrics[seq.seq_id].record_first_token(step_end)

                for item in finished_outputs:
                    seq_id, output_ids, extra = parse_finished_output(item)

                    if seq_id not in metrics:
                        continue

                    # 重要：
                    # 不要在这里强行 record_first_token。
                    # 否则 first_token_time 和 completion_time 会几乎一样，TPOT 会变成 0。
                    metrics[seq_id].record_completion(output_ids)

                    completed_latencies.append(metrics[seq_id].latency)
                    avg_latency = float(np.mean(completed_latencies))
                    pbar.set_postfix({"Avg Latency": f"{avg_latency:.2f}s"})
                    pbar.update(1)
            else:
                time.sleep(0.01)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    completed_metrics = [
        m for m in metrics.values()
        if m.completion_time >= 0
    ]

    ttfts = [
        m.ttft for m in completed_metrics
        if not np.isnan(m.ttft)
    ]

    tpots = [
        m.tpot for m in completed_metrics
        if not np.isnan(m.tpot)
    ]

    latencies = [
        m.latency for m in completed_metrics
        if not np.isnan(m.latency)
    ]

    total_input_tokens = sum(m.input_len for m in completed_metrics)
    total_output_tokens = sum(
        m.output_len for m in completed_metrics
        if m.output_len > 0
    )

    total_tokens = total_input_tokens + total_output_tokens
    throughput = total_tokens / total_time if total_time > 0 else float("nan")
    output_throughput = total_output_tokens / total_time if total_time > 0 else float("nan")

    avg_ttft = np.mean(ttfts) if ttfts else float("nan")
    avg_tpot = np.mean(tpots) if tpots else float("nan")
    avg_latency = np.mean(latencies) if latencies else float("nan")

    print("\n--- Benchmark Results: No Chunked Prefill ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests sent: {requests_sent}")
    print(f"Requests completed: {len(completed_metrics)}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total throughput: {throughput:.2f} tokens/s")
    print(f"Output throughput: {output_throughput:.2f} tokens/s")

    print(f"Average TTFT: {avg_ttft * 1000:.2f} ms")
    print(f"Average TPOT: {avg_tpot * 1000:.2f} ms")
    print(f"Average latency: {avg_latency:.2f} s")

    if ttfts:
        ttfts_ms = [x * 1000.0 for x in ttfts]
        print(
            "TTFT(ms) | "
            f"p50={percentile(ttfts_ms, 50):.2f}, "
            f"p90={percentile(ttfts_ms, 90):.2f}, "
            f"p95={percentile(ttfts_ms, 95):.2f}, "
            f"p99={percentile(ttfts_ms, 99):.2f}"
        )
    else:
        print("TTFT(ms) | no valid data")

    if tpots:
        tpots_ms = [x * 1000.0 for x in tpots]
        print(
            "TPOT(ms) | "
            f"p50={percentile(tpots_ms, 50):.2f}, "
            f"p90={percentile(tpots_ms, 90):.2f}, "
            f"p95={percentile(tpots_ms, 95):.2f}, "
            f"p99={percentile(tpots_ms, 99):.2f}"
        )
    else:
        print("TPOT(ms) | no valid data")

    if latencies:
        print(
            "Latency(s) | "
            f"p50={percentile(latencies, 50):.2f}, "
            f"p90={percentile(latencies, 90):.2f}, "
            f"p95={percentile(latencies, 95):.2f}, "
            f"p99={percentile(latencies, 99):.2f}"
        )

    print("---------------------------------------------\n")


if __name__ == "__main__":
    main()
