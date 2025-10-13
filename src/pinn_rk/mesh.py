from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass(frozen=True)
class TimeMesh:
    """
    Time partition 0 = t0 < ... < tN = T and step sizes k_n.
    """
    nodes: Tensor  # [N+1]
    steps: Tensor  # [N]

    @staticmethod
    def uniform(T: float, N: int, device: torch.device = torch.device("cpu")) -> "TimeMesh":
        if not (T > 0 and N >= 1):
            raise ValueError("T must be > 0 and N >= 1.")
        nodes = torch.linspace(0.0, T, N + 1, device=device, dtype=torch.float64)
        steps = nodes[1:] - nodes[:-1]
        return TimeMesh(nodes=nodes, steps=steps)
