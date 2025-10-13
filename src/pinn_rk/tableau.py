from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor


@dataclass(frozen=True)
class ButcherTableau:
    """Butcher tableau for a q-stage RK method."""
    A: Tensor  # shape [q, q]
    b: Tensor  # shape [q]
    c: Tensor  # shape [q]

    def validate(self) -> None:
        q = self.c.numel()
        if self.A.shape != (q, q):
            raise ValueError("A must be square (q×q).")
        if self.b.shape != (q,):
            raise ValueError("b must have length q.")
        if not (torch.all(self.c >= 0) and torch.all(self.c <= 1)):
            raise ValueError("Runge–Kutta nodes c must lie in [0,1].")

def butcher_gauss_legendre_q2(device: torch.device = torch.device("cpu")) -> ButcherTableau:
    sqrt3: Final[float] = math.sqrt(3.0)
    c = torch.tensor([(0.5 - sqrt3 / 6.0), (0.5 + sqrt3 / 6.0)], dtype=torch.float64, device=device)
    A = torch.tensor([[1/4, 1/4 - sqrt3/6],
                      [1/4 + sqrt3/6, 1/4]], dtype=torch.float64, device=device)
    b = torch.tensor([0.5, 0.5], dtype=torch.float64, device=device)
    T = ButcherTableau(A=A, b=b, c=c)
    T.validate()
    return T

def butcher_radau_iia_q2(device: torch.device = torch.device("cpu")) -> ButcherTableau:
    c = torch.tensor([1/3, 1.0], dtype=torch.float64, device=device)
    A = torch.tensor([[5/12, -1/12],
                      [3/4,   1/4]], dtype=torch.float64, device=device)
    b = torch.tensor([3/4, 1/4], dtype=torch.float64, device=device)
    T = ButcherTableau(A=A, b=b, c=c)
    T.validate()
    return T

def butcher_lobatto_iiia_q2(device: torch.device = torch.device("cpu")) -> ButcherTableau:
    c = torch.tensor([0.0, 1.0], dtype=torch.float64, device=device)
    A = torch.tensor([[0.0, 0.0],
                      [1/2, 1/2]], dtype=torch.float64, device=device)
    b = torch.tensor([1/2, 1/2], dtype=torch.float64, device=device)
    T = ButcherTableau(A=A, b=b, c=c)
    T.validate()
    return T
