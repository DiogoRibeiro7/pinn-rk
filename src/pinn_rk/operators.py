from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class EllipticOperator(Protocol):
    def __call__(self, x: Tensor, u: Tensor) -> Tensor: ...
    def requires_hessian(self) -> bool: ...


class Laplacian1D(EllipticOperator):
    """L u = -u_xx on a 1D domain."""

    def __call__(self, x: Tensor, u: Tensor) -> Tensor:
        assert x.ndim == u.ndim == 2 and x.shape == u.shape
        grad = torch.autograd.grad(
            u, x, torch.ones_like(u), create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        d2 = torch.autograd.grad(
            grad, x, torch.ones_like(grad), create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        return -d2

    def requires_hessian(self) -> bool:
        return True


class LaplacianND(EllipticOperator):
    """
    L u = -Δu = -Σ_i ∂²u/∂x_i² on a d-dimensional domain.

    Each second derivative is taken separately rather than by summing the diagonal of a
    full Hessian: only the d diagonal entries are needed, and forming the whole matrix
    would cost d times as much for values that are then discarded.

    Equivalent to `Laplacian1D` when d = 1, and agrees with it to machine precision.
    """

    def __call__(self, x: Tensor, u: Tensor) -> Tensor:
        if x.ndim != 2 or u.ndim != 2 or u.shape[1] != 1 or u.shape[0] != x.shape[0]:
            raise ValueError(
                f"expected x [B,d] and u [B,1] with matching batch; got "
                f"{tuple(x.shape)} and {tuple(u.shape)}."
            )
        grad = torch.autograd.grad(
            u, x, torch.ones_like(u), create_graph=True, retain_graph=True, only_inputs=True
        )[0]  # [B,d]

        laplacian = torch.zeros_like(u)
        for i in range(x.shape[1]):
            gi = grad[:, i : i + 1]
            d2i = torch.autograd.grad(
                gi, x, torch.ones_like(gi), create_graph=True, retain_graph=True, only_inputs=True
            )[0][:, i : i + 1]
            laplacian = laplacian + d2i
        return -laplacian

    def requires_hessian(self) -> bool:
        return True
