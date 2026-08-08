"""
Convergence orders of the Runge-Kutta collocation residual.

Writing the semi-discrete problem as u' = f - L u, the method is defined by

    U_i     = u_n + k sum_j a_ij F_j     (stage equations, using A)
    u_{n+1} = u_n + k sum_i b_i F_i      (update equation, using b)

Feeding the exact solution u = sin(pi x) exp(-pi^2 t) through these leaves only
the local truncation error of the tableau. The stage residual then converges at
the method's stage order, and the update residual at its classical order.

The classical orders are what make the choice of tableau meaningful, and they are
the property the pre-0.3 interpolant residual could not express -- it ignored A,
so every tableau behaved identically. These tests pin the orders so that losing
that property fails loudly.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from pinn_rk.config import RkPinnConfig
from pinn_rk.loss import RkPinnLoss
from pinn_rk.mesh import TimeMesh
from pinn_rk.operators import Laplacian1D
from pinn_rk.tableau import (
    butcher_gauss_legendre_q2,
    butcher_lobatto_iiia_q2,
    butcher_radau_iia_q2,
)

DEV = torch.device("cpu")


class ExactHeatSolution(nn.Module):
    """u(x,t) = sin(pi x) exp(-pi^2 t), which solves u_t - u_xx = 0 exactly."""

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.sin(math.pi * x) * torch.exp(-(math.pi**2) * t)


def _zero_rhs(x: Tensor, t: Tensor) -> Tensor:
    return torch.zeros_like(x)


def _residual_norms(tab_fn, n_slabs: int, nx: int = 48) -> tuple[float, float]:
    """Max |r_stage| and |r_step| on the first slab, for the exact solution."""
    torch.set_default_dtype(torch.float64)
    cfg = RkPinnConfig(
        tableau=tab_fn(DEV),
        time_mesh=TimeMesh.uniform(T=0.1, N=n_slabs, device=DEV),
        residual="rk",
        n_x_train=nx,
        device=DEV,
    )
    loss_fn = RkPinnLoss(model=ExactHeatSolution(), Lop=Laplacian1D(), f_rhs=_zero_rhs, cfg=cfg)
    x = torch.linspace(0.05, 0.95, nx, dtype=torch.float64).unsqueeze(1)
    r_stage, r_step = loss_fn.rk_residuals(x, 0, cfg.time_mesh.steps[0], loss_fn._stage_times(0))
    return float(r_stage.detach().abs().max()), float(r_step.detach().abs().max())


def _order(errs: list[float], ks: list[float]) -> float:
    """Least-squares slope of log(error) against log(k)."""
    le, lk = [math.log(e) for e in errs], [math.log(k) for k in ks]
    n = len(ks)
    mk, me = sum(lk) / n, sum(le) / n
    num = sum((a - mk) * (b - me) for a, b in zip(lk, le, strict=True))
    return num / sum((a - mk) ** 2 for a in lk)


NS = [5, 10, 20, 40]
KS = [0.1 / n for n in NS]

# tableau -> classical order p
CLASSICAL_ORDER = {
    butcher_gauss_legendre_q2: 4,
    butcher_radau_iia_q2: 3,
    butcher_lobatto_iiia_q2: 2,
}


@pytest.mark.parametrize("tab_fn", list(CLASSICAL_ORDER))
def test_stage_residual_has_stage_order_two(tab_fn) -> None:
    """All three q=2 collocation tableaux have stage order 2."""
    errs = [_residual_norms(tab_fn, n)[0] for n in NS]
    assert _order(errs, KS) == pytest.approx(2.0, abs=0.2)


@pytest.mark.parametrize(("tab_fn", "p"), list(CLASSICAL_ORDER.items()))
def test_update_residual_has_the_tableau_classical_order(tab_fn, p: int) -> None:
    """
    The update residual converges at the tableau's classical order: 4, 3, 2.

    This is the property that makes the choice of tableau matter. It is only
    reachable because the residual uses the Butcher matrix A.
    """
    errs = [_residual_norms(tab_fn, n)[1] for n in NS]
    assert _order(errs, KS) == pytest.approx(float(p), abs=0.25)


def test_gauss_update_residual_is_far_below_radau_at_equal_cost() -> None:
    """
    Both tableaux are 2-stage, so this is a like-for-like comparison.

    Gauss q=2 is order 4 against Radau IIA's 3, and at the slab size used by the
    shipped example that is worth more than two orders of magnitude.
    """
    gauss = _residual_norms(butcher_gauss_legendre_q2, 20)[1]
    radau = _residual_norms(butcher_radau_iia_q2, 20)[1]
    assert gauss < radau / 100.0


def test_lobatto_is_stiffly_accurate() -> None:
    """
    Lobatto IIIA q=2 has b equal to the last row of A.

    The update equation therefore coincides with the final stage equation, so the
    two residuals must agree to machine precision.
    """
    stage, step = _residual_norms(butcher_lobatto_iiia_q2, 20)
    assert stage == pytest.approx(step, rel=1e-10)


def test_rk_residual_beats_the_interpolant_form() -> None:
    """
    Same tableau, same slab size: the RK update residual is far more consistent.

    The interpolant form ignores A and reaches only order 2, whichever tableau is
    supplied; the RK form reaches order 4 for Gauss.
    """
    torch.set_default_dtype(torch.float64)
    nx = 48
    x = torch.linspace(0.05, 0.95, nx, dtype=torch.float64).unsqueeze(1)

    def interpolant_residual() -> float:
        cfg = RkPinnConfig(
            tableau=butcher_gauss_legendre_q2(DEV),
            time_mesh=TimeMesh.uniform(T=0.1, N=20, device=DEV),
            residual="interpolant",
            q_aux="extend",
            n_x_train=nx,
            device=DEV,
        )
        fn = RkPinnLoss(model=ExactHeatSolution(), Lop=Laplacian1D(), f_rhs=_zero_rhs, cfg=cfg)
        t_stage = fn._stage_times(0)
        U, LU, F = fn._stage_values(x, t_stage)
        nodes, extended = fn._interp_nodes(t_stage, cfg.time_mesh.nodes[0])
        from pinn_rk.interpolants import differentiation_matrix

        if extended:
            u0 = fn._eval_model(x, torch.full_like(x, float(cfg.time_mesh.nodes[0].item())))
            U = torch.cat([u0.unsqueeze(1), U], dim=1)
        D = differentiation_matrix(nodes)
        if extended:
            D = D[1:]
        return float((torch.einsum("ij,bjk->bik", D, U) + LU - F).detach().abs().max())

    assert _residual_norms(butcher_gauss_legendre_q2, 20)[1] < interpolant_residual() / 100.0
