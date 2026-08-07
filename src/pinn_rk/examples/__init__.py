"""Reference examples built on the :mod:`pinn_rk` core.

These modules are shipped with the package so the documented workflows can be
run directly after installation. They are illustrative rather than part of the
stable API: signatures here may change between minor releases.
"""

from __future__ import annotations

from .train_heat_equation import exact_f as exact_f
from .train_heat_equation import exact_u as exact_u
from .train_heat_equation import l2_error as l2_error
from .train_heat_equation import make_init_data as make_init_data
from .train_heat_equation import train_heat_equation as train_heat_equation

__all__ = [
    "exact_f",
    "exact_u",
    "l2_error",
    "make_init_data",
    "train_heat_equation",
]
