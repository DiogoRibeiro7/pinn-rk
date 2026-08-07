"""
Consistency of the polynomial time reconstruction used by the RK-PINN residual.

For the manufactured solution u(x,t) = sin(pi x) exp(-pi^2 t) the heat residual
r = u_t - u_xx vanishes identically. Feeding that exact solution through the
residual assembly therefore isolates the consistency error of the time
reconstruction: whatever is left is truncation error, and since training
minimises the residual, it sets a floor on attainable accuracy.

These tests pin the observed convergence orders so a regression in the
reconstruction shows up as a failing order rather than a slightly worse loss.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from pinn_rk.interpolants import differentiation_matrix
from pinn_rk.operators import Laplacian1D
from pinn_rk.tableau import (
    butcher_gauss_legendre_q2,
    butcher_lobatto_iiia_q2,
    butcher_radau_iia_q2,
)

DEV = torch.device("cpu")


def _u_exact(x: Tensor, t: Tensor) -> Tensor:
    return torch.sin(math.pi * x) * torch.exp(-(math.pi**2) * t)


def _residual_on_exact(tab_fn, k: float, extend: bool, t0: float = 0.0, nx: int = 64) -> float:
    """Max |u_t + L u| over the stage nodes of one slab, for the exact solution."""
    torch.set_default_dtype(torch.float64)
    bt = tab_fn(DEV)
    q = bt.c.numel()
    t_stage = t0 + bt.c.to(DEV) * k
    x = torch.linspace(0.05, 0.95, nx, dtype=torch.float64).unsqueeze(1)

    Lu = []
    for i in range(q):
        xi = x.clone().requires_grad_(True)
        ti = torch.full_like(xi, float(t_stage[i]))
        Lu.append(Laplacian1D()(xi, _u_exact(xi, ti)).detach())
    Lu_s = torch.stack(Lu, dim=1)

    use_extend = extend and not bool(
        torch.isclose(t_stage, torch.tensor(t0, dtype=torch.float64)).any()
    )
    nodes = torch.cat([torch.tensor([t0], dtype=torch.float64), t_stage]) if use_extend else t_stage
    U = torch.stack([_u_exact(x, torch.full_like(x, float(tv))) for tv in nodes], dim=1)
    D = differentiation_matrix(nodes)
    if use_extend:
        D = D[1:]
    u_t = torch.einsum("ij,bjk->bik", D, U)

    return float((u_t + Lu_s).abs().max().item())


def _observed_order(tab_fn, extend: bool) -> float:
    """Least-squares slope of log(residual) against log(1/k) over a k-refinement."""
    ks = [0.1 / n for n in (10, 20, 40, 80)]
    errs = [_residual_on_exact(tab_fn, k, extend) for k in ks]
    logs_k = [math.log(k) for k in ks]
    logs_e = [math.log(e) for e in errs]
    n = len(ks)
    mk = sum(logs_k) / n
    me = sum(logs_e) / n
    num = sum((a - mk) * (b - me) for a, b in zip(logs_k, logs_e, strict=True))
    den = sum((a - mk) ** 2 for a in logs_k)
    return num / den


TABLEAUX = [butcher_radau_iia_q2, butcher_gauss_legendre_q2]


@pytest.mark.parametrize("tab_fn", TABLEAUX)
def test_stage_node_reconstruction_is_first_order(tab_fn) -> None:
    """Interpolating only the q=2 stage values gives a linear u_hat: O(k)."""
    assert _observed_order(tab_fn, extend=False) == pytest.approx(1.0, abs=0.15)


@pytest.mark.parametrize("tab_fn", TABLEAUX)
def test_extended_reconstruction_is_second_order(tab_fn) -> None:
    """Adding the slab start raises u_hat to degree 2: O(k^2)."""
    assert _observed_order(tab_fn, extend=True) == pytest.approx(2.0, abs=0.15)


@pytest.mark.parametrize("tab_fn", TABLEAUX)
def test_extending_reduces_the_consistency_error(tab_fn) -> None:
    """At the slab size used by the shipped example, extending is far more accurate."""
    k = 0.1 / 20
    assert _residual_on_exact(tab_fn, k, extend=True) < 0.1 * _residual_on_exact(
        tab_fn, k, extend=False
    )


def test_lobatto_cannot_extend_and_stays_first_order() -> None:
    """
    Lobatto IIIA has c_1 = 0, so the slab start is already a stage node.

    The stencil cannot be extended without duplicating a node, so this tableau is
    held to first order and both settings must agree exactly.
    """
    k = 0.1 / 20
    same = _residual_on_exact(butcher_lobatto_iiia_q2, k, extend=False)
    extended = _residual_on_exact(butcher_lobatto_iiia_q2, k, extend=True)
    assert same == pytest.approx(extended, rel=1e-12)
    assert _observed_order(butcher_lobatto_iiia_q2, extend=True) == pytest.approx(1.0, abs=0.15)
