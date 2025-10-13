from __future__ import annotations

import torch

from pinn_rk.tableau import (
    ButcherTableau,
    butcher_gauss_legendre_q2,
    butcher_lobatto_iiia_q2,
    butcher_radau_iia_q2,
)


def test_butcher_shapes_and_bounds() -> None:
    for factory in (butcher_gauss_legendre_q2, butcher_radau_iia_q2, butcher_lobatto_iiia_q2):
        T = factory()
        q = T.c.numel()
        assert T.A.shape == (q, q)
        assert T.b.shape == (q,)
        assert torch.all(T.c >= 0) and torch.all(T.c <= 1)


def test_butcher_validation_raises() -> None:
    q = 2
    A = torch.zeros(q, q, dtype=torch.float64)
    b = torch.zeros(q, dtype=torch.float64)
    c = torch.tensor([-0.1, 1.1], dtype=torch.float64)  # invalid
    T = ButcherTableau(A=A, b=b, c=c)
    try:
        T.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for c outside [0,1].")
