import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams

def percentile(sorted_list, p):
    if not sorted_list:
        return 0.0
    k = (len(sorted_list) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_list) - 1)
    if f == c:
        return sorted_list[f]
    return sorted_list[f] * (c - k) + sorted_list[c] * (k - f)


def print_ttft_stats(ttfts):
    if not ttfts:
        print("TTFT: no data")
        return

    ttfts_ms = sorted(x * 1000.0 for x in ttfts)
    avg = sum(ttfts_ms) / len(ttfts_ms)
    print(
        "TTFT(ms) | "
        f"avg={avg:.2f}, "
        f"p50={percentile(ttfts_ms, 50):.2f}, "
        f"p90={percentile(ttfts_ms, 90):.2f}, "
        f"p95={percentile(ttfts_ms, 95):.2f}, "
        f"p99={percentile(ttfts_ms, 99):.2f}"
    )

def main():
    seed(0)
    num_seqs = 1024
    max_input_len = 128
    max_ouput_len = 100

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(128, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
    print_ttft_stats(llm.last_ttfts)

if __name__ == "__main__":
    main()
