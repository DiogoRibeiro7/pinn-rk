from __future__ import annotations

import math
import torch
from torch import nn, Tensor

from pinn_rk.config import RkPinnConfig
from pinn_rk.mesh import TimeMesh
from pinn_rk.model import MLP
from pinn_rk.operators import Laplacian1D
from pinn_rk.tableau import butcher_radau_iia_q2
from pinn_rk.loss import RkPinnLoss


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
