import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_  # noqa: F401


def to_2tuple(x):
    return x if isinstance(x, (tuple, list)) else (x, x)


class DropPath(nn.Module):
    """Stochastic depth. drop_path_rate is 0.0 for TinyViT-5M, and this model is
    only ever run in eval mode here, so this reduces to identity."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self):
        return f"drop_prob={self.drop_prob}"
    