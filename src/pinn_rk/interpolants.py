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
                w[j] /= (nodes[j] - nodes[m])
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
