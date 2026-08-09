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
    A = torch.tensor(
        [[1 / 4, 1 / 4 - sqrt3 / 6], [1 / 4 + sqrt3 / 6, 1 / 4]], dtype=torch.float64, device=device
    )
    b = torch.tensor([0.5, 0.5], dtype=torch.float64, device=device)
    T = ButcherTableau(A=A, b=b, c=c)
    T.validate()
    return T


def butcher_radau_iia_q2(device: torch.device = torch.device("cpu")) -> ButcherTableau:
    c = torch.tensor([1 / 3, 1.0], dtype=torch.float64, device=device)
    A = torch.tensor([[5 / 12, -1 / 12], [3 / 4, 1 / 4]], dtype=torch.float64, device=device)
    b = torch.tensor([3 / 4, 1 / 4], dtype=torch.float64, device=device)
    T = ButcherTableau(A=A, b=b, c=c)
    T.validate()
    return T


def butcher_lobatto_iiia_q2(device: torch.device = torch.device("cpu")) -> ButcherTableau:
    c = torch.tensor([0.0, 1.0], dtype=torch.float64, device=device)
    A = torch.tensor([[0.0, 0.0], [1 / 2, 1 / 2]], dtype=torch.float64, device=device)
    b = torch.tensor([1 / 2, 1 / 2], dtype=torch.float64, device=device)
    T = ButcherTableau(A=A, b=b, c=c)
    T.validate()
    return T


# --- Three-stage families -----------------------------------------------------------
#
# All three are collocation methods, so A and b are determined by the nodes c via
# a_ij = ∫_0^{c_i} L_j(s) ds and b_j = ∫_0^1 L_j(s) ds. The coefficients below are the
# closed forms; tests/test_tableau_order.py re-derives them from c by integrating the
# Lagrange basis and checks the two agree, so a transcription slip cannot go unnoticed.
#
# Raising q lifts the stage order from 2 to 3, which matters because the stage residual
# is what caps the accuracy of the "rk" objective.


def butcher_gauss_legendre_q3(device: torch.device = torch.device("cpu")) -> ButcherTableau:
    """Gauss-Legendre, 3 stages, classical order 6, A-stable and symplectic."""
    s: Final[float] = math.sqrt(15.0)
    c = torch.tensor([0.5 - s / 10.0, 0.5, 0.5 + s / 10.0], dtype=torch.float64, device=device)
    A = torch.tensor(
        [
            [5 / 36, 2 / 9 - s / 15, 5 / 36 - s / 30],
            [5 / 36 + s / 24, 2 / 9, 5 / 36 - s / 24],
            [5 / 36 + s / 30, 2 / 9 + s / 15, 5 / 36],
        ],
        dtype=torch.float64,
        device=device,
    )
    b = torch.tensor([5 / 18, 4 / 9, 5 / 18], dtype=torch.float64, device=device)
    T = ButcherTableau(A=A, b=b, c=c)
    T.validate()
    return T


def butcher_radau_iia_q3(device: torch.device = torch.device("cpu")) -> ButcherTableau:
    """Radau IIA, 3 stages, classical order 5, L-stable and stiffly accurate."""
    s: Final[float] = math.sqrt(6.0)
    c = torch.tensor([(4 - s) / 10.0, (4 + s) / 10.0, 1.0], dtype=torch.float64, device=device)
    A = torch.tensor(
        [
            [(88 - 7 * s) / 360, (296 - 169 * s) / 1800, (-2 + 3 * s) / 225],
            [(296 + 169 * s) / 1800, (88 + 7 * s) / 360, (-2 - 3 * s) / 225],
            [(16 - s) / 36, (16 + s) / 36, 1 / 9],
        ],
        dtype=torch.float64,
        device=device,
    )
    b = torch.tensor([(16 - s) / 36, (16 + s) / 36, 1 / 9], dtype=torch.float64, device=device)
    T = ButcherTableau(A=A, b=b, c=c)
    T.validate()
    return T


def butcher_lobatto_iiia_q3(device: torch.device = torch.device("cpu")) -> ButcherTableau:
    """Lobatto IIIA, 3 stages, classical order 4 (Simpson's rule), A-stable."""
    c = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64, device=device)
    A = torch.tensor(
        [[0.0, 0.0, 0.0], [5 / 24, 1 / 3, -1 / 24], [1 / 6, 2 / 3, 1 / 6]],
        dtype=torch.float64,
        device=device,
    )
    b = torch.tensor([1 / 6, 2 / 3, 1 / 6], dtype=torch.float64, device=device)
    T = ButcherTableau(A=A, b=b, c=c)
    T.validate()
    return T
