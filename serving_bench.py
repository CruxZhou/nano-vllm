import time
import numpy as np
import argparse
from random import randint, seed
from tqdm.auto import tqdm
from nanovllm import LLM, SamplingParams


seed(100)
np.random.seed(100)


class RequestMetrics:
    """Stores metrics for a single request."""
    def __init__(self, request_id, input_len):
        self.request_id = request_id
        self.input_len = input_len
        self.submission_time = -1.0
        self.first_token_time = -1.0
        self.completion_time = -1.0
        self.output_len = -1

    def record_first_token(self, ts=None):
        if self.first_token_time == -1:
            self.first_token_time = time.perf_counter() if ts is None else ts

    def record_completion(self, output_ids, ts=None):
        self.completion_time = time.perf_counter() if ts is None else ts
        self.output_len = len(output_ids)

    @property
    def ttft(self):
        if self.submission_time == -1 or self.first_token_time == -1:
            return float("nan")
        return self.first_token_time - self.submission_time

    @property
    def tpot(self):
        if self.output_len > 1 and self.first_token_time != -1 and self.completion_time != -1:
            return (self.completion_time - self.first_token_time) / (self.output_len - 1)
        return float("nan")

    @property
    def latency(self):
        if self.submission_time == -1 or self.completion_time == -1:
            return float("nan")
        return self.completion_time - self.submission_time


def warm_up(engine, args):
    prompts = [
        [randint(0, 10000) for _ in range(args.random_input_len)]
        for _ in range(50)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=args.random_output_len,
        )
        for _ in range(len(prompts))
    ]
    engine.generate(prompts, sampling_params)


def parse_finished_outputs(finished_outputs):
    """
    兼容两种格式：
    1. [(seq_id, output_ids)]
    2. [(seq_id, output_ids, ttft)]
    返回统一格式: [(seq_id, output_ids, ttft_or_none)]
    """
    parsed = []
    for item in finished_outputs:
        if isinstance(item, (tuple, list)):
            if len(item) >= 3:
                seq_id, output_ids, ttft = item[0], item[1], item[2]
            elif len(item) == 2:
                seq_id, output_ids = item
                ttft = None
            else:
                raise ValueError(f"Unexpected finished output format: {item}")
        else:
            raise TypeError(f"Unexpected finished output type: {type(item)}")
        parsed.append((seq_id, output_ids, ttft))
    return parsed


def percentile(values, p):
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def main():
    parser = argparse.ArgumentParser(description="Serving benchmark for nano-vllm.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--request-rate", type=float, default=8.0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--random-input-len", type=int, default=128)
    parser.add_argument("--random-output-len", type=int, default=128)
    parser.add_argument("--chunked-prefill", action="store_true", default=False)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    args = parser.parse_args()

    print(
        f"\n--- Running benchmark with "
        f"--num-requests {args.num_requests} "
        f"--request-rate {args.request_rate} "
        f"--chunked-prefill {args.chunked_prefill} ---"
    )

    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=\,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        chunked_prefill=args.chunked_prefill,
    )
    engine = llm

    warm_up(engine, args)

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

    request_intervals = np.random.exponential(1.0 / args.request_rate, args.num_requests)
    arrival_times = np.cumsum(request_intervals)
    print(arrival_times)

    metrics = {}
    requests_sent = 0
    start_time = time.perf_counter()
    completed_latencies = []

    with tqdm(total=args.num_requests, desc="Processing Requests") as pbar:
        while requests_sent < args.num_requests or not engine.is_finished():
            current_time = time.perf_counter()

            while (
                requests_sent < args.num_requests
                and current_time - start_time >= arrival_times[requests_sent]
            ):
                prompt = prompts[requests_sent]
                sp = sampling_params[requests_sent]

                engine.add_request(prompt, sp)

                # 真实入队后的 seq
                new_seq = engine.scheduler.waiting[-1]
                seq_id = new_seq.seq_id

                req_metrics = RequestMetrics(seq_id, len(prompt))

                # 用真实 arrival_time，而不是计划到达时间
                if hasattr(new_seq, "arrival_time"):
                    req_metrics.submission_time = new_seq.arrival_time
                else:
                    req_metrics.submission_time = time.perf_counter()

                metrics[seq_id] = req_metrics
                requests_sent += 1

                current_time = time.perf_counter()

            if not engine.is_finished():
                step_time = time.perf_counter()
                finished_outputs, _ = engine.step()
                finished_outputs = parse_finished_outputs(finished_outputs)

                # 兼容没有 ttft 的版本：尽量在“prefill 刚结束”时记 first token 时间
                all_processed_seqs = list(engine.scheduler.running)
                for seq in all_processed_seqs:
                    if seq.seq_id not in metrics:
                        continue
                    if metrics[seq.seq_id].first_token_time != -1:
                        continue

                    # 老版本兼容逻辑
                    if hasattr(seq, "num_cached_tokens") and hasattr(seq, "num_prompt_tokens"):
                        if seq.num_cached_tokens == seq.num_prompt_tokens:
                            metrics[seq.seq_id].record_first_token(step_time)

                for seq_id, output_ids, ttft in finished_outputs:
                    if seq_id not in metrics:
                        continue

                    # 优先用引擎给的 ttft
                    if metrics[seq_id].first_token_time == -1:
                        if ttft is not None and not np.isnan(ttft):
                            metrics[seq_id].record_first_token(
                                metrics[seq_id].submission_time + ttft
                            )
                        else:
                            # 至少保证 completion 前 first_token_time 已赋值
                            metrics[seq_id].record_first_token(step_time)

                    metrics[seq_id].record_completion(output_ids)

                    completed_latencies.append(metrics[seq_id].latency)
                    avg_latency_so_far = float(np.mean(completed_latencies))
                    pbar.set_postfix({"Avg Latency": f"{avg_latency_so_far:.2f}s"})
                    pbar.update(1)
            else:
                time.sleep(0.001)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    total_input_tokens = sum(m.input_len for m in metrics.values())
    total_output_tokens = sum(m.output_len for m in metrics.values() if m.output_len != -1)

    ttfts = [m.ttft for m in metrics.values() if not np.isnan(m.ttft)]
    tpots = [m.tpot for m in metrics.values() if not np.isnan(m.tpot)]
    latencies = [m.latency for m in metrics.values() if not np.isnan(m.latency)]

    avg_ttft = float(np.mean(ttfts)) if ttfts else float("nan")
    avg_tpot = float(np.mean(tpots)) if tpots else float("nan")
    avg_latency = float(np.mean(latencies)) if latencies else float("nan")
    throughput = (total_input_tokens + total_output_tokens) / total_time

    print("--- Benchmark Results ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests sent: {requests_sent}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Throughput: {throughput:.2f} tokens/s")

    print(f"Average TTFT: {avg_ttft * 1000:.2f} ms")
    print(f"P50 TTFT: {percentile(ttfts, 50) * 1000:.2f} ms")
    print(f"P95 TTFT: {percentile(ttfts, 95) * 1000:.2f} ms")
    print(f"P99 TTFT: {percentile(ttfts, 99) * 1000:.2f} ms")

    print(f"Average TPOT: {avg_tpot * 1000:.2f} ms")
    print(f"P50 TPOT: {percentile(tpots, 50) * 1000:.2f} ms")
    print(f"P95 TPOT: {percentile(tpots, 95) * 1000:.2f} ms")
    print(f"P99 TPOT: {percentile(tpots, 99) * 1000:.2f} ms")

    print(f"Average latency: {avg_latency:.2f} s")
    print(f"P50 latency: {percentile(latencies, 50):.2f} s")
    print(f"P95 latency: {percentile(latencies, 95):.2f} s")
    print(f"P99 latency: {percentile(latencies, 99):.2f} s")
    print("-------------------------\n")


if __name__ == "__main__":
    main()
