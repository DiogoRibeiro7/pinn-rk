from __future__ import annotations

import torch

from pinn_rk.model import MLP


def test_boundary_condition_is_satisfied() -> None:
    device = torch.device("cpu")
    model = MLP(dtype=torch.float64).to(device)
    x0 = torch.zeros(8, 1, dtype=torch.float64)         # x = 0
    x1 = torch.ones(8, 1, dtype=torch.float64)          # x = 1
    t  = torch.linspace(0, 0.1, 8, dtype=torch.float64).unsqueeze(1)

    with torch.no_grad():
        u0 = model(x0, t)
        u1 = model(x1, t)
    # BC enforced by phi(x) = x(1-x) → 0 at boundaries
    assert torch.allclose(u0, torch.zeros_like(u0), atol=1e-12)
    assert torch.allclose(u1, torch.zeros_like(u1), atol=1e-12)
