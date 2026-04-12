import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int, #词表里的token数
        embedding_dim: int, #每个token embedding的维度
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0 #要求词表大小能够整除并行数
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        #上面算当前卡负责多少token和token id范围
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y #就是把不属于本卡的那些位置乘成 0
            # unsqueeze:在某个位置插入一个长度为 1 的维度
            # 假设：mask.shape = [3] y.shape = [3, 4]
            # 先把mask变成3*1 和3*4的y乘会广播成3*4的
            dist.all_reduce(y) #利用torch的通信原语做的归约（逐元素求和）
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context() #拿prefill/decode的状态和切片索引
        last_indices = context.cu_seqlens_q[1:] - 1
        if context.seq_need_compute_logits.numel():
            last_indices = last_indices[context.seq_need_compute_logits]
        x = x[last_indices].contiguous()
       
        logits = F.linear(x, self.weight) #x乘以weight的转置，此处bias为空不用写出来
        if self.tp_size > 1: #如果多卡，那只有一部分的weight（比如本来应该是15w多，四个卡每个卡只有4w多）
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0) #放到rank0 gpu上，不是求和是拼接
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
#Embedding 阶段 输入是 token id，输出是 embedding 向量
#Head 阶段 输入是 hidden state，输出是整个词表上的分数 每张卡算的是 vocab 的不同 slice