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
    residual:
        Which residual the loss imposes.

        "rk" (default) uses the full Butcher tableau: the stage equations
        U_i = u_n + k Σ_j a_ij F_j bring A into the loss, and the update equation
        u_{n+1} = u_n + k Σ_i b_i F_i carries the tableau's classical order. This
        is the only setting under which the choice of tableau affects accuracy —
        measured on the manufactured solution, the update residual converges at
        order 4 for Gauss q=2, 3 for Radau IIA q=2 and 2 for Lobatto IIIA q=2.

        "interpolant" is the pre-0.3 behaviour: ∂ₜû is taken from the polynomial
        reconstruction through the stage values and the residual is u_t + L u - f.
        It ignores A entirely, so every tableau behaves the same and accuracy is
        governed by the degree of û. Retained for comparison and for the
        convergence study.

    q_aux:
        Node set carrying the polynomial time reconstruction on each slab.
        Applies only when ``residual="interpolant"``; it is ignored otherwise.
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
    ic_weight:
        Multiplier on the initial-condition penalty, relative to the PDE residual.

        The two terms are otherwise summed unweighted, and their balance is not
        stable during training: on the shipped heat-equation example the penalty
        accounts for essentially all of the loss at initialisation (5.1 against a
        residual of 1.7e-4) but only about a fifth of it after a few hundred steps.
        Training therefore begins by fitting the initial condition almost
        exclusively. Lower this to let the PDE residual dominate sooner; set it to
        0.0 to drop the penalty entirely and measure the residual alone.
    """

    tableau: ButcherTableau
    time_mesh: TimeMesh
    residual: Literal["rk", "interpolant"] = "rk"
    q_aux: Literal["same", "extend"] = "extend"
    spatial_sampler: SpatialSampler | None = None
    n_x_train: int = 256
    device: _device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float64
    init_data: tuple[Tensor, Tensor] | None = None
    ic_weight: float = 1.0
