from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from pinn_rk.config import RkPinnConfig
from pinn_rk.interpolants import differentiation_matrix
from pinn_rk.loss import RkPinnLoss
from pinn_rk.mesh import TimeMesh
from pinn_rk.model import MLP
from pinn_rk.operators import Laplacian1D
from pinn_rk.tableau import butcher_lobatto_iiia_q2, butcher_radau_iia_q2


@torch.no_grad()
def exact_u(x: Tensor, t: Tensor) -> Tensor:
    return torch.sin(math.pi * x) * torch.exp(-(math.pi**2) * t)


@torch.no_grad()
def exact_f(x: Tensor, t: Tensor) -> Tensor:
    return torch.zeros_like(x)


def make_init_data(n0: int, device: torch.device) -> tuple[Tensor, Tensor]:
    x0 = torch.linspace(1e-6, 1 - 1e-6, n0, device=device, dtype=torch.float64).unsqueeze(1)
    t0 = torch.zeros_like(x0)
    return x0, exact_u(x0, t0)


def test_loss_decreases_a_few_steps() -> None:
    device = torch.device("cpu")
    torch.set_default_dtype(torch.float64)

    bt = butcher_radau_iia_q2(device)
    mesh = TimeMesh.uniform(T=0.05, N=5, device=device)
    model = MLP().to(device)
    L = Laplacian1D()
    x0, u0 = make_init_data(64, device)

    cfg = RkPinnConfig(tableau=bt, time_mesh=mesh, n_x_train=64, device=device, init_data=(x0, u0))
    loss_fn = RkPinnLoss(model=model, Lop=L, f_rhs=exact_f, cfg=cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    # measure first loss
    loss0 = float(loss_fn().item())
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        opt.step()
    loss1 = float(loss_fn().item())

    assert loss1 < loss0, f"Loss did not decrease: {loss0:.3e} -> {loss1:.3e}"


def _build(tableau_fn, q_aux: str) -> RkPinnLoss:
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")
    torch.manual_seed(0)
    x0, u0 = make_init_data(32, device)
    cfg = RkPinnConfig(
        tableau=tableau_fn(device),
        time_mesh=TimeMesh.uniform(T=0.05, N=3, device=device),
        n_x_train=32,
        device=device,
        init_data=(x0, u0),
        q_aux=q_aux,  # type: ignore[arg-type]
    )
    return RkPinnLoss(
        model=MLP(dtype=torch.float64).to(device),
        Lop=Laplacian1D(),
        f_rhs=exact_f,
        cfg=cfg,
    ).to(device)


@pytest.mark.parametrize("q_aux", ["same", "extend"])
def test_loss_is_finite_for_both_reconstruction_stencils(q_aux: str) -> None:
    loss = _build(butcher_radau_iia_q2, q_aux)()
    assert torch.isfinite(loss), f"non-finite loss for q_aux={q_aux}"
    assert loss.item() > 0.0


def test_extend_is_skipped_when_a_stage_node_sits_at_the_slab_start() -> None:
    """
    Lobatto IIIA has c_1 = 0, so prepending t_n would duplicate a node.

    A duplicated node makes the barycentric weights infinite, so the stencil must
    stay unextended and the loss must remain finite.
    """
    loss_fn = _build(butcher_lobatto_iiia_q2, "extend")
    t_stage = loss_fn._stage_times(0)
    nodes, extended = loss_fn._interp_nodes(t_stage, loss_fn.cfg.time_mesh.nodes[0])

    assert not extended
    assert nodes.numel() == t_stage.numel()
    assert torch.isfinite(loss_fn())


def test_extend_adds_the_slab_start_for_tableaux_that_need_it() -> None:
    """Radau IIA has c_1 = 1/3, so t_n is genuinely a new node."""
    loss_fn = _build(butcher_radau_iia_q2, "extend")
    t_n = loss_fn.cfg.time_mesh.nodes[0]
    t_stage = loss_fn._stage_times(0)
    nodes, extended = loss_fn._interp_nodes(t_stage, t_n)

    assert extended
    assert nodes.numel() == t_stage.numel() + 1
    assert torch.isclose(nodes[0], t_n)


def test_stage_derivative_is_exact_for_a_linear_in_time_field() -> None:
    """
    The q=2 reconstruction is linear in t, so it differentiates a linear field exactly.

    Using u(x,t) = x(1-x)(1 + 3t), the interpolant derivative at both stage nodes
    must equal 3*x(1-x) to machine precision -- something the previous
    finite-difference path only achieved approximately.
    """
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")

    class LinearInTime(torch.nn.Module):
        def forward(self, x: Tensor, t: Tensor) -> Tensor:
            return x * (1.0 - x) * (1.0 + 3.0 * t)

    cfg = RkPinnConfig(
        tableau=butcher_radau_iia_q2(device),
        time_mesh=TimeMesh.uniform(T=0.1, N=2, device=device),
        n_x_train=16,
        device=device,
    )
    loss_fn = RkPinnLoss(model=LinearInTime(), Lop=Laplacian1D(), f_rhs=exact_f, cfg=cfg)

    x = torch.linspace(0.1, 0.9, 16, dtype=torch.float64).unsqueeze(1)
    t_stage = loss_fn._stage_times(0)
    U = torch.stack(
        [loss_fn.model(x, torch.full_like(x, float(t_stage[i].item()))) for i in range(2)],
        dim=1,
    )
    D = differentiation_matrix(t_stage)
    u_t = torch.einsum("ij,bjk->bik", D, U)

    expected = 3.0 * x * (1.0 - x)  # [B,1]
    for i in range(2):
        assert torch.allclose(u_t[:, i, :], expected, atol=1e-12)
