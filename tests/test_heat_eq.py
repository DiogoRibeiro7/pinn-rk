from __future__ import annotations
import torch
from pinn_rk import train_heat_equation, l2_error

def test_train_and_error():
    device = torch.device("cpu")
    model = train_heat_equation(method="radau2", T=0.05, N=5, n_x_train=64, steps=200, lr=1e-2, device=device)
    err = l2_error(model, T=0.05, nx=201, device=device)
    # Loose bound for CI; just sanity:
    assert err < 0.25
