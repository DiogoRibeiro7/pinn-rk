from __future__ import annotations

from typing import List
import torch
from torch import nn, Tensor

class MLP(nn.Module):
    """
    (x,t) -> u(x,t) with boundary conditioning Φ(x)=x(1-x) for homogeneous Dirichlet on [0,1].
    """
    def __init__(self, in_dim: int = 2, width: int = 128, depth: int = 4, activation: str = "tanh") -> None:
        super().__init__()
        if in_dim != 2:
            raise ValueError("MLP expects in_dim=2 (x,t).")
        act_cls = nn.Tanh if activation == "tanh" else nn.SiLU
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), act_cls()]
            d = width
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if not (x.shape == t.shape and x.ndim == 2 and x.shape[1] == 1):
            raise ValueError("x and t must be [B,1] tensors with identical shapes.")
        xt = torch.cat([x, t], dim=1)
        g = self.net(xt)
        phi = x * (1.0 - x)  # boundary factor
        return phi * g
