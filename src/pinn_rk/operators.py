from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class EllipticOperator(Protocol):
    def __call__(self, x: Tensor, u: Tensor) -> Tensor: ...
    def requires_hessian(self) -> bool: ...


class Laplacian1D(EllipticOperator):
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
