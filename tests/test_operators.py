from __future__ import annotations

import torch
from torch import Tensor
from pinn_rk.model import MLP
from pinn_rk.operators import Laplacian1D


def test_laplacian_shapes_and_autograd() -> None:
    device = torch.device("cpu")
    x = torch.linspace(1e-6, 1 - 1e-6, 32, dtype=torch.float64, device=device).unsqueeze(1)
    t = torch.zeros_like(x)

    model = MLP().to(device)
    u = model(x.requires_grad_(True), t.requires_grad_(True))

    L = Laplacian1D()
    Lu = L(x, u)
    assert Lu.shape == u.shape
    # Backprop sanity: sum(Lu) should produce grads on parameters
    loss = Lu.sum()
    loss.backward()
    # At least one parameter should have grad
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad
