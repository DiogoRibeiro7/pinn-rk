from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor
from torch import device as _device

from .mesh import TimeMesh
from .tableau import ButcherTableau

SpatialSampler = Callable[[int, _device], Tensor]


@dataclass
class RkPinnConfig:
    """
    Configuration for RK-PINN loss assembly.

    Attributes
    ----------
    tableau:
        Runge-Kutta method encoded as a Butcher tableau.
    time_mesh:
        Time discretization [0,T] -> {t_n}.
    q_aux:
        Node set carrying the polynomial time reconstruction on each slab.
        "same" interpolates the q stage values, so û has degree q-1.
        "extend" prepends the slab start t_n, raising û to degree q at the cost of
        one extra network evaluation per slab. It is ignored for tableaux whose
        first stage already sits at t_n (Lobatto IIIA has c_1 = 0), since that
        would duplicate an interpolation node.

        Defaults to "extend": measured against the manufactured solution, the
        residual's consistency error is O(k) for "same" but O(k^2) for "extend",
        roughly sixty times smaller at the slab size used by the shipped example.
        See tests/test_time_reconstruction.py, which pins both orders.
    spatial_sampler:
        Callable that returns x-samples inside the spatial domain Ω for a given batch size.
    n_x_train:
        Number of spatial samples per time slab.
    device:
        Torch device for tensors and model.
    dtype:
        Torch dtype used across computations.
    init_data:
        Optional tuple (x0, u0(x0)) to impose initial H¹ seminorm penalty.
    """

    tableau: ButcherTableau
    time_mesh: TimeMesh
    q_aux: Literal["same", "extend"] = "extend"
    spatial_sampler: SpatialSampler | None = None
    n_x_train: int = 256
    device: _device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float64
    init_data: tuple[Tensor, Tensor] | None = None
