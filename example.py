import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

import argparse

def main(args):
    path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=args.enforce_eager, tensor_parallel_size=args.tensor_parallel_size)

    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens) #采样参数
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100", 
    ] # 同时处理多个prompt而非逐条推理
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params) # 最外层API入口

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="nano vllm")
    parse.add_argument("--model_path", type=str, default="~/huggingface/Qwen3-0.6B/", help="Path to the model")
    parse.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    parse.add_argument("--enforce-eager", type=bool, default=True)
    parse.add_argument("--temperature", type=float, default=0.6)
    parse.add_argument("--max-tokens", type=int, default=256)
    
    args=parse.parse_args()
    
    main(args)
