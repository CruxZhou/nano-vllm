import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    mask = slot != -1

    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    cache_offsets = slot * D + tl.arange(0, D)

    key = tl.load(key_ptr + key_offsets, mask=mask, other=0.0)
    value = tl.load(value_ptr + value_offsets, mask=mask, other=0.0)

    tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, value, mask=mask)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    if key.numel() == 0:
        return

    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    store_kvcache_kernel[(N,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D,
    )


def _use_decode_fast_path(context, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> bool:
    if k_cache.numel() == 0 or v_cache.numel() == 0:
        return False
    if context.block_tables is None:
        return False
    if context.cu_seqlens_q is None or context.context_lens is None:
        return False
    if context.max_seqlen_q != 1:
        return False
    if q.ndim != 3:
        return False

    batch_size = context.context_lens.numel()
    return q.size(0) == batch_size


def _flash_attn_with_kvcache_compat(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context,
    softmax_scale: float,
):
    # q: [batch, num_heads, head_dim] -> [batch, 1, num_heads, head_dim]
    q = q.unsqueeze(1).contiguous()
    cache_seqlens = context.context_lens.contiguous()
    block_tables = context.block_tables.contiguous()

    # 兼容不同 flash-attn 版本的参数名 / 调用方式
    try:
        out = flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_tables,
            softmax_scale=softmax_scale,
            causal=True,
        )
    except TypeError:
        try:
            out = flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                cache_seqlens=cache_seqlens,
                block_tables=block_tables,
                softmax_scale=softmax_scale,
                causal=True,
            )
        except TypeError:
            try:
                out = flash_attn_with_kvcache(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    cache_seqlens=cache_seqlens,
                    block_table=block_tables,
                    softmax_scale=softmax_scale,
                    causal=True,
                )
            except TypeError:
                out = flash_attn_with_kvcache(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    cache_seqlens=cache_seqlens,
                    block_tables=block_tables,
                    softmax_scale=softmax_scale,
                    causal=True,
                )

    return out.squeeze(1)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # decode fast path:
        # - paged KV cache 在用
        # - 每条序列本轮 q_len == 1
        if _use_decode_fast_path(context, q, k_cache, v_cache):
            return _flash_attn_with_kvcache_compat(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                context=context,
                softmax_scale=self.scale,
            )

        # chunked prefill / normal prefill / mixed paged path
        if context.block_tables is not None:
            k, v = k_cache, v_cache

        o = flash_attn_varlen_func(
            q,
            k,
            v,
            max_seqlen_q=context.max_seqlen_q,
            cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_k=context.max_seqlen_k,
            cu_seqlens_k=context.cu_seqlens_k,
            softmax_scale=self.scale,
            causal=True,
            block_table=context.block_tables,
        )
        return o
