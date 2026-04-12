import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")

        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.last_ttfts = []

        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        seq.arrival_time = perf_counter()
        self.scheduler.add(seq)
        return seq.seq_id

    def _step_internal(self):
        """
        内部版本：
        返回 scheduler.postprocess 的原始格式，通常是
        [(seq_id, token_ids, ttft), ...]
        """
        seqs = self.scheduler.schedule()
        token_ids, seq_need_compute_logits = self.model_runner.call("run", seqs)
        outputs = self.scheduler.postprocess(seqs, token_ids, seq_need_compute_logits)

        # 这里保持你原来的统计逻辑，不额外改行为
        num_total_tokens = sum(len(seq) for seq in seqs if seq.is_finished)
        return outputs, num_total_tokens

    def step(self, return_ttft: bool = False):
        """
        对外默认兼容 serving_bench.py：
        - return_ttft=False: 返回 [(seq_id, token_ids), ...]
        - return_ttft=True:  返回 [(seq_id, token_ids, ttft), ...]
        """
        outputs, num_total_tokens = self._step_internal()

        if return_ttft:
            return outputs, num_total_tokens

        stripped_outputs = []
        for item in outputs:
            if isinstance(item, (tuple, list)):
                if len(item) >= 2:
                    seq_id = item[0]
                    token_ids = item[1]
                    stripped_outputs.append((seq_id, token_ids))
                else:
                    raise ValueError(f"Unexpected output format from scheduler.postprocess: {item}")
            else:
                raise TypeError(f"Unexpected output type from scheduler.postprocess: {type(item)}")

        return stripped_outputs, num_total_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        self.last_ttfts = []
        outputs = {}

        prefill_throughput = 0.0
        decode_throughput = 0.0

        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step(return_ttft=True)

            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / max(perf_counter() - t, 1e-8)
                else:
                    decode_throughput = -num_tokens / max(perf_counter() - t, 1e-8)

                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })

            for seq_id, token_ids, ttft in output:
                outputs[seq_id] = token_ids
                if ttft is not None:
                    self.last_ttfts.append(ttft)
                if use_tqdm:
                    pbar.update(1)

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]

        if use_tqdm:
            pbar.close()

        return outputs
