from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple
import torch
from torch import Tensor, device as _device

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
        Runge–Kutta method encoded as a Butcher tableau.
    time_mesh:
        Time discretization [0,T] -> {t_n}.
    q_aux:
        Whether to extend interpolation nodes beyond stage nodes. Currently "same".
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
    q_aux: Literal["same", "extend"] = "same"
    spatial_sampler: Optional[SpatialSampler] = None
    n_x_train: int = 256
    device: _device = torch.device("cpu")
    dtype: torch.dtype = torch.float64
    init_data: Optional[Tuple[Tensor, Tensor]] = None
