from __future__ import annotations

import pytest
import torch

from pinn_rk.interpolants import (
    barycentric_weights,
    differentiation_matrix,
    lagrange_eval,
)


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


@pytest.mark.parametrize("q", [2, 3, 4, 5])
def test_differentiation_matrix_exact_on_polynomials(q: int) -> None:
    """D reproduces p'(t_i) exactly for every polynomial of degree <= q-1."""
    torch.manual_seed(q)
    nodes = torch.sort(torch.rand(q, dtype=torch.float64)).values
    D = differentiation_matrix(nodes)
    for deg in range(q):
        u = nodes**deg
        expected = torch.zeros_like(nodes) if deg == 0 else deg * nodes ** (deg - 1)
        assert torch.allclose(D @ u, expected, atol=1e-10)


def test_differentiation_matrix_rows_sum_to_zero() -> None:
    """Differentiating a constant gives zero, so every row must sum to zero."""
    nodes = torch.tensor([0.0, 0.21, 0.55, 1.0], dtype=torch.float64)
    D = differentiation_matrix(nodes)
    assert torch.allclose(D.sum(dim=1), torch.zeros(4, dtype=torch.float64), atol=1e-12)


def test_differentiation_matrix_two_nodes_is_secant_slope() -> None:
    """With two nodes the interpolant is linear, so both rows give the secant."""
    nodes = torch.tensor([0.0, 1.0], dtype=torch.float64)
    D = differentiation_matrix(nodes)
    expected = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float64)
    assert torch.allclose(D, expected, atol=1e-12)


def test_differentiation_matrix_accepts_precomputed_weights() -> None:
    nodes = torch.tensor([0.1, 0.4, 0.9], dtype=torch.float64)
    w = barycentric_weights(nodes)
    assert torch.allclose(differentiation_matrix(nodes, w), differentiation_matrix(nodes))


def test_differentiation_matrix_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        differentiation_matrix(torch.zeros(2, 2, dtype=torch.float64))
    with pytest.raises(ValueError):
        differentiation_matrix(torch.tensor([0.5], dtype=torch.float64))
    with pytest.raises(ValueError):
        nodes = torch.tensor([0.0, 1.0], dtype=torch.float64)
        differentiation_matrix(nodes, torch.ones(3, dtype=torch.float64))
