from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

from pinn_rk import (
    MLP,
    Laplacian1D,
    RkPinnConfig,
    RkPinnLoss,
    TimeMesh,
    butcher_gauss_legendre_q2,
    butcher_lobatto_iiia_q2,
    butcher_radau_iia_q2,
)

# --- Exact manufactured solution for validation ---


def exact_u(x: Tensor, t: Tensor) -> Tensor:
    return torch.sin(math.pi * x) * torch.exp(-(math.pi**2) * t)


def exact_f(x: Tensor, t: Tensor) -> Tensor:
    return torch.zeros_like(x)


def make_init_data(n0: int, device: torch.device) -> tuple[Tensor, Tensor]:
    x0 = torch.linspace(1e-6, 1 - 1e-6, n0, device=device, dtype=torch.float64).unsqueeze(1)
    t0 = torch.zeros_like(x0)
    return x0, exact_u(x0, t0)


# --- Training and evaluation utilities ---


def train_heat_equation(
    method: Literal["gauss2", "radau2", "lobatto2"] = "radau2",
    T: float = 0.1,
    N: int = 20,
    n_x_train: int = 256,
    steps: int = 5000,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    torch.set_default_dtype(torch.float64)

    if method == "gauss2":
        bt = butcher_gauss_legendre_q2(device)
    elif method == "radau2":
        bt = butcher_radau_iia_q2(device)
    elif method == "lobatto2":
        bt = butcher_lobatto_iiia_q2(device)
    else:
        raise ValueError("Unknown RK method.")

    mesh = TimeMesh.uniform(T=T, N=N, device=device)
    model = MLP(in_dim=2, width=128, depth=4, activation="tanh", dtype=torch.float64).to(device)
    L = Laplacian1D()

    x0, u0 = make_init_data(n0=128, device=device)

    cfg = RkPinnConfig(
        tableau=bt,
        time_mesh=mesh,
        n_x_train=n_x_train,
        device=device,
        dtype=torch.float64,
        init_data=(x0, u0),
    )
    loss_fn = RkPinnLoss(model=model, Lop=L, f_rhs=exact_f, cfg=cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for it in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {it}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if it % 500 == 0:
            print(f"[{it:5d}] loss = {loss.item():.6e}")

    return model


def l2_error(
    model: nn.Module, T: float, nx: int = 1001, device: torch.device = torch.device("cpu")
) -> float:
    x = torch.linspace(0, 1, nx, device=device, dtype=torch.float64).unsqueeze(1)
    t = torch.full_like(x, T)
    with torch.no_grad():
        u = model(x, t)
        ue = exact_u(x, t)
    err = torch.sqrt(torch.mean((u - ue) ** 2)).item()
    return err


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_heat_equation(
        method="radau2", T=0.1, N=20, n_x_train=512, steps=4000, lr=2e-3, device=dev
    )
    errT = l2_error(model, T=0.1, nx=2001, device=dev)
    print(f"L2 error at T=0.1: {errT:.3e}")
