from __future__ import annotations

import torch
from torch import Tensor


def barycentric_weights(nodes: Tensor) -> Tensor:
    """
    Compute first-form barycentric weights for Lagrange interpolation.
    Complexity O(q^2), stable for small q (as here).
    """
    if nodes.ndim != 1:
        raise ValueError("nodes must be a 1D tensor.")
    q = nodes.numel()
    w = torch.ones(q, dtype=torch.float64, device=nodes.device)
    for j in range(q):
        for m in range(q):
            if m != j:
                w[j] /= nodes[j] - nodes[m]
    return w


def lagrange_eval(t: Tensor, nodes: Tensor, w: Tensor) -> Tensor:
    """
    Evaluate Lagrange basis at t using first-form barycentric formula.

    Parameters
    ----------
    t : Tensor [...], evaluation points
    nodes : Tensor [q], interpolation nodes
    w : Tensor [q], barycentric weights

    Returns
    -------
    Tensor [..., q] : basis weights ℓ_j(t)
    """
    if nodes.ndim != 1 or w.ndim != 1 or nodes.numel() != w.numel():
        raise ValueError("nodes and w must be 1D with the same length.")
    t = t.unsqueeze(-1)  # [..., 1]
    diff = t - nodes  # [..., q]
    near = torch.isclose(diff, torch.zeros_like(diff), atol=1e-14, rtol=0.0)
    if near.any():
        idx = near.float().argmax(dim=-1)
        L = torch.zeros(*t.shape[:-1], nodes.numel(), dtype=torch.float64, device=t.device)
        L.scatter_(-1, idx.unsqueeze(-1), 1.0)
        return L
    num = w / diff  # [..., q]
    denom = num.sum(dim=-1, keepdim=True)
    return num / denom


def differentiation_matrix(nodes: Tensor, w: Tensor | None = None) -> Tensor:
    """
    Build the barycentric differentiation matrix for the given interpolation nodes.

    Returns ``D`` with ``D[i, j] = ℓ_j'(t_i)``, so that for values ``u`` sampled at
    ``nodes``, ``D @ u`` is the derivative of the interpolating polynomial evaluated
    at those same nodes. This is exact for polynomials of degree ≤ q-1.

    Off-diagonal entries follow the standard barycentric identity
    ``D[i, j] = (w_j / w_i) / (t_i - t_j)``; the diagonal is set by the negative
    row sum, which enforces exactness on constants.

    Parameters
    ----------
    nodes : Tensor [q], interpolation nodes, pairwise distinct
    w : Tensor [q], optional precomputed barycentric weights

    Returns
    -------
    Tensor [q, q] : differentiation matrix
    """
    if nodes.ndim != 1:
        raise ValueError("nodes must be a 1D tensor.")
    q = nodes.numel()
    if q < 2:
        raise ValueError("differentiation requires at least two nodes.")
    if w is None:
        w = barycentric_weights(nodes)
    elif w.ndim != 1 or w.numel() != q:
        raise ValueError("w must be 1D with the same length as nodes.")

    diff = nodes.unsqueeze(1) - nodes.unsqueeze(0)  # [q,q], diff[i,j] = t_i - t_j
    eye = torch.eye(q, dtype=torch.bool, device=nodes.device)
    # Avoid dividing by the zero diagonal; those entries are overwritten below.
    safe = torch.where(eye, torch.ones_like(diff), diff)
    D = (w.unsqueeze(0) / w.unsqueeze(1)) / safe  # [i,j] = (w_j / w_i) / (t_i - t_j)
    D = D.masked_fill(eye, 0.0)
    return D + torch.diag_embed(-D.sum(dim=1))
