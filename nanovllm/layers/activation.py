import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1) #按照最后一个维度做了一个二分的分块（即按列切两块）
        return F.silu(x) * y
