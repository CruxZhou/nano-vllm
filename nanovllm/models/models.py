from nanovllm.models.qwen2 import Qwen2ForCausalLM
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.llama import LlamaForCausalLM


model_dict={
    "llama":LlamaForCausalLM,
    "qwen2":Qwen2ForCausalLM,
    "qwen3":Qwen3ForCausalLM,
}