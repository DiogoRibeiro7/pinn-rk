from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
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


def _lagrange_integral(c: np.ndarray, j: int, upper: float) -> float:
    """∫_0^upper L_j(s) ds, by exact integration of the Lagrange basis polynomial."""
    poly = np.poly1d([1.0])
    denom = 1.0
    for m in range(len(c)):
        if m != j:
            poly = poly * np.poly1d([1.0, -c[m]])
            denom *= c[j] - c[m]
    anti = poly.integ()
    return float((anti(upper) - anti(0.0)) / denom)


def collocation_tableau(
    nodes: Sequence[float] | Tensor, device: torch.device = torch.device("cpu")
) -> ButcherTableau:
    """
    Build the collocation tableau determined by its nodes.

    For a collocation method the Butcher coefficients are not free: given distinct nodes
    c, they are fixed by

        a_ij = ∫_0^{c_i} L_j(s) ds,      b_j = ∫_0^1 L_j(s) ds

    with L_j the Lagrange basis on c. Gauss-Legendre, Radau IIA and Lobatto IIIA are all
    collocation families, differing only in where the nodes sit, so this reconstructs any
    of them from the nodes alone.

    Useful directly for adding a new family, and used internally for q=4, where the A
    matrices have no workable closed form and writing out 16 coefficients apiece would be
    transcription risk with no benefit.

    Parameters
    ----------
    nodes : sequence or Tensor of q distinct values in [0,1]
    device : torch device for the resulting tensors

    Returns
    -------
    ButcherTableau, already validated.
    """
    c_np = np.asarray(
        nodes.detach().cpu().numpy() if isinstance(nodes, Tensor) else nodes, dtype=np.float64
    )
    if c_np.ndim != 1 or c_np.size < 1:
        raise ValueError("nodes must be a non-empty 1D sequence.")
    if len(np.unique(np.round(c_np, 14))) != c_np.size:
        raise ValueError("collocation nodes must be pairwise distinct.")

    q = c_np.size
    A_np = np.array(
        [[_lagrange_integral(c_np, j, float(c_np[i])) for j in range(q)] for i in range(q)]
    )
    b_np = np.array([_lagrange_integral(c_np, j, 1.0) for j in range(q)])

    T = ButcherTableau(
        A=torch.tensor(A_np, dtype=torch.float64, device=device),
        b=torch.tensor(b_np, dtype=torch.float64, device=device),
        c=torch.tensor(c_np, dtype=torch.float64, device=device),
    )
    T.validate()
    return T


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


# --- Four-stage families ------------------------------------------------------------
#
# At q=4 the A matrices have no workable closed form, so these give the nodes only and
# let `collocation_tableau` derive the 16 coefficients apiece. The order conditions are
# checked in tests/test_tableau_order.py, which is what actually pins the accuracy: for
# these tableaux the derivation cannot serve as an independent check, since it is also
# what built them.
#
# Stage order reaches 4 here, so the ceiling on the "rk" objective moves to O(k^4).


def butcher_gauss_legendre_q4(device: torch.device = torch.device("cpu")) -> ButcherTableau:
    """Gauss-Legendre, 4 stages, classical order 8, A-stable and symplectic."""
    g: Final[float] = math.sqrt(6.0 / 5.0)
    outer: Final[float] = math.sqrt((3 + 2 * g) / 7)
    inner: Final[float] = math.sqrt((3 - 2 * g) / 7)
    nodes = [(1 - outer) / 2, (1 - inner) / 2, (1 + inner) / 2, (1 + outer) / 2]
    return collocation_tableau(nodes, device=device)


def butcher_radau_iia_q4(device: torch.device = torch.device("cpu")) -> ButcherTableau:
    """
    Radau IIA, 4 stages, classical order 7, L-stable and stiffly accurate.

    The interior nodes are roots of the third derivative of x^3 (x-1)^4 and have no
    closed form worth writing down, so they are given here to full double precision. To
    regenerate them, take that polynomial, differentiate three times, solve, and polish
    with Newton; the final node is exactly 1 by definition of Radau IIA.
    """
    nodes = [
        0.08858795951270396,
        0.40946686444073477,
        0.7876594617608462,
        1.0,
    ]
    return collocation_tableau(nodes, device=device)


def butcher_lobatto_iiia_q4(device: torch.device = torch.device("cpu")) -> ButcherTableau:
    """Lobatto IIIA, 4 stages, classical order 6, A-stable and stiffly accurate."""
    r5: Final[float] = math.sqrt(5.0)
    nodes = [0.0, (1 - 1 / r5) / 2, (1 + 1 / r5) / 2, 1.0]
    return collocation_tableau(nodes, device=device)
