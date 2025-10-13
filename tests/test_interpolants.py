from __future__ import annotations

import torch

from pinn_rk.interpolants import barycentric_weights, lagrange_eval


def test_lagrange_partition_of_unity() -> None:
    nodes = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    w = barycentric_weights(nodes)
    t = torch.linspace(-0.2, 1.2, 25, dtype=torch.float64)  # include outside range
    L = lagrange_eval(t, nodes, w)  # [25, 3]
    s = L.sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-12)


def test_lagrange_exact_at_nodes() -> None:
    nodes = torch.tensor([0.0, 0.3, 0.8], dtype=torch.float64)
    w = barycentric_weights(nodes)
    for i in range(len(nodes)):
        Li = lagrange_eval(nodes[i], nodes, w).squeeze(0)  # [3]
        target = torch.zeros_like(nodes)
        target[i] = 1.0
        assert torch.allclose(Li, target, atol=1e-12)
