"""
Algebraic verification of every shipped Butcher tableau.

All three families are *collocation* methods, which means A and b are not free: they
are forced by the nodes,

    a_ij = int_0^{c_i} L_j(s) ds,      b_j = int_0^1 L_j(s) ds

with L_j the Lagrange basis on c. Writing coefficients out by hand is therefore
transcription risk with no upside, so these tests re-derive them from the nodes alone
and compare. A mistyped digit fails here rather than quietly degrading the method.

The order conditions are checked directly as well:

    B(p):  sum_i b_i c_i^{k-1}     = 1/k        for k = 1..p   -> classical order
    C(s):  sum_j a_ij c_j^{k-1}    = c_i^k / k  for k = 1..s   -> stage order
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pinn_rk.tableau import (
    ButcherTableau,
    butcher_gauss_legendre_q2,
    butcher_gauss_legendre_q3,
    butcher_lobatto_iiia_q2,
    butcher_lobatto_iiia_q3,
    butcher_radau_iia_q2,
    butcher_radau_iia_q3,
)

# factory -> (stages q, classical order p, stiffly accurate)
TABLEAUX = {
    butcher_gauss_legendre_q2: (2, 4, False),
    butcher_radau_iia_q2: (2, 3, True),
    butcher_lobatto_iiia_q2: (2, 2, True),
    butcher_gauss_legendre_q3: (3, 6, False),
    butcher_radau_iia_q3: (3, 5, True),
    butcher_lobatto_iiia_q3: (3, 4, True),
}
IDS = [f.__name__.replace("butcher_", "") for f in TABLEAUX]


def _lagrange_integral(c: np.ndarray, j: int, upper: float) -> float:
    """int_0^upper L_j(s) ds, by exact polynomial integration."""
    poly = np.poly1d([1.0])
    denom = 1.0
    for m in range(len(c)):
        if m != j:
            poly = poly * np.poly1d([1.0, -c[m]])
            denom *= c[j] - c[m]
    anti = poly.integ()
    return float((anti(upper) - anti(0.0)) / denom)


def _derive_from_nodes(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = len(c)
    A = np.array([[_lagrange_integral(c, j, c[i]) for j in range(q)] for i in range(q)])
    b = np.array([_lagrange_integral(c, j, 1.0) for j in range(q)])
    return A, b


def _numpy(t: ButcherTableau) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return t.A.numpy(), t.b.numpy(), t.c.numpy()


@pytest.mark.parametrize("factory", TABLEAUX, ids=IDS)
def test_coefficients_match_the_collocation_derivation(factory) -> None:
    """The hard-coded A and b must equal what the nodes force them to be."""
    A, b, c = _numpy(factory())
    A_derived, b_derived = _derive_from_nodes(c)
    assert np.max(np.abs(A - A_derived)) < 1e-14
    assert np.max(np.abs(b - b_derived)) < 1e-14


@pytest.mark.parametrize("factory", TABLEAUX, ids=IDS)
def test_row_sums_equal_the_nodes(factory) -> None:
    """Consistency: each stage must be exact on the constant function."""
    A, _, c = _numpy(factory())
    assert np.max(np.abs(A.sum(axis=1) - c)) < 1e-14


@pytest.mark.parametrize("factory", TABLEAUX, ids=IDS)
def test_weights_sum_to_one(factory) -> None:
    _, b, _ = _numpy(factory())
    assert float(b.sum()) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("factory", TABLEAUX, ids=IDS)
def test_classical_order_conditions(factory) -> None:
    """B(p) holds to the advertised order p, and fails at p+1."""
    _, (_, p, _) = factory, TABLEAUX[factory]
    _, b, c = _numpy(factory())
    for k in range(1, p + 1):
        assert float(np.sum(b * c ** (k - 1))) == pytest.approx(1.0 / k, abs=1e-13), (
            f"B({k}) should hold"
        )
    k = p + 1
    assert abs(float(np.sum(b * c ** (k - 1))) - 1.0 / k) > 1e-10, (
        f"B({k}) unexpectedly holds; order is higher than advertised"
    )


@pytest.mark.parametrize("factory", TABLEAUX, ids=IDS)
def test_stage_order_equals_the_number_of_stages(factory) -> None:
    """Collocation methods have stage order q, which caps the RK stage residual."""
    q = TABLEAUX[factory][0]
    A, _, c = _numpy(factory())
    for k in range(1, q + 1):
        assert np.max(np.abs(A @ c ** (k - 1) - c**k / k)) < 1e-13, f"C({k}) should hold"


@pytest.mark.parametrize("factory", TABLEAUX, ids=IDS)
def test_stiff_accuracy_flag(factory) -> None:
    """Stiffly accurate means b equals the final row of A."""
    expected = TABLEAUX[factory][2]
    A, b, _ = _numpy(factory())
    assert bool(np.max(np.abs(A[-1] - b)) < 1e-14) is expected


@pytest.mark.parametrize("factory", TABLEAUX, ids=IDS)
def test_nodes_are_distinct_and_in_range(factory) -> None:
    """Duplicate nodes would make the barycentric weights infinite."""
    _, _, c = _numpy(factory())
    assert np.all(c >= 0.0) and np.all(c <= 1.0)
    assert len(np.unique(np.round(c, 14))) == len(c)


@pytest.mark.parametrize("factory", TABLEAUX, ids=IDS)
def test_shapes_and_dtype(factory) -> None:
    t = factory()
    q = TABLEAUX[factory][0]
    assert t.A.shape == (q, q)
    assert t.b.shape == (q,)
    assert t.c.shape == (q,)
    assert t.A.dtype == torch.float64


def test_three_stage_families_raise_the_stage_order() -> None:
    """
    The point of q=3: stage order rises from 2 to 3.

    The stage residual carries the stage order and dominates the RK objective, so this
    is the lever that lifts the O(k^2) ceiling the two-stage tableaux impose.
    """
    for two, three in (
        (butcher_gauss_legendre_q2, butcher_gauss_legendre_q3),
        (butcher_radau_iia_q2, butcher_radau_iia_q3),
        (butcher_lobatto_iiia_q2, butcher_lobatto_iiia_q3),
    ):
        assert TABLEAUX[two][0] == 2
        assert TABLEAUX[three][0] == 3
        assert TABLEAUX[three][1] > TABLEAUX[two][1]
