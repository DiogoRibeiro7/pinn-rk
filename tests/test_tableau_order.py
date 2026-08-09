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
    butcher_gauss_legendre_q4,
    butcher_lobatto_iiia_q2,
    butcher_lobatto_iiia_q3,
    butcher_lobatto_iiia_q4,
    butcher_radau_iia_q2,
    butcher_radau_iia_q3,
    butcher_radau_iia_q4,
    collocation_tableau,
)

# factory -> (stages q, classical order p, stiffly accurate)
TABLEAUX = {
    butcher_gauss_legendre_q2: (2, 4, False),
    butcher_radau_iia_q2: (2, 3, True),
    butcher_lobatto_iiia_q2: (2, 2, True),
    butcher_gauss_legendre_q3: (3, 6, False),
    butcher_radau_iia_q3: (3, 5, True),
    butcher_lobatto_iiia_q3: (3, 4, True),
    butcher_gauss_legendre_q4: (4, 8, False),
    butcher_radau_iia_q4: (4, 7, True),
    butcher_lobatto_iiia_q4: (4, 6, True),
}
IDS = [f.__name__.replace("butcher_", "") for f in TABLEAUX]

# The q=2 and q=3 tableaux carry hard-coded closed-form coefficients, so re-deriving them
# from their nodes is a genuine independent check. The q=4 tableaux are *built* by
# `collocation_tableau`, so that same comparison would only test the derivation against
# itself. For those the verification weight is carried by the order conditions, which are
# independent mathematical facts about the tableau rather than restatements of how it was
# constructed -- in particular `test_classical_order_conditions`, which also asserts that
# B(p+1) fails and therefore pins the order from above as well as below.
HARDCODED = {f: v for f, v in TABLEAUX.items() if v[0] in (2, 3)}
HARDCODED_IDS = [f.__name__.replace("butcher_", "") for f in HARDCODED]


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


@pytest.mark.parametrize("factory", HARDCODED, ids=HARDCODED_IDS)
def test_coefficients_match_the_collocation_derivation(factory) -> None:
    """
    The hard-coded A and b must equal what the nodes force them to be.

    Only the q=2 and q=3 tableaux are covered: the q=4 ones are produced by the very
    derivation used here, so including them would compare the code to itself.
    """
    A, b, c = _numpy(factory())
    A_derived, b_derived = _derive_from_nodes(c)
    assert np.max(np.abs(A - A_derived)) < 1e-14
    assert np.max(np.abs(b - b_derived)) < 1e-14


def test_collocation_builder_reproduces_a_known_tableau() -> None:
    """
    `collocation_tableau` is checked against a tableau whose coefficients are known.

    Nodes [0, 1/2, 1] must give Simpson's rule, b = [1/6, 2/3, 1/6], and the whole
    Lobatto IIIA q=3 tableau. This is what makes the builder trustworthy for q=4, where
    no closed form is available to compare against.
    """
    built = collocation_tableau([0.0, 0.5, 1.0])
    known = butcher_lobatto_iiia_q3()
    assert np.max(np.abs(built.A.numpy() - known.A.numpy())) < 1e-14
    assert np.max(np.abs(built.b.numpy() - known.b.numpy())) < 1e-14
    assert np.max(np.abs(built.c.numpy() - known.c.numpy())) < 1e-14


def test_collocation_builder_rejects_duplicate_nodes() -> None:
    """Repeated nodes make the Lagrange basis undefined; fail rather than divide by zero."""
    with pytest.raises(ValueError, match="distinct"):
        collocation_tableau([0.0, 0.5, 0.5])


def test_collocation_builder_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        collocation_tableau([])


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


@pytest.mark.parametrize(
    "family",
    [
        (butcher_gauss_legendre_q2, butcher_gauss_legendre_q3, butcher_gauss_legendre_q4),
        (butcher_radau_iia_q2, butcher_radau_iia_q3, butcher_radau_iia_q4),
        (butcher_lobatto_iiia_q2, butcher_lobatto_iiia_q3, butcher_lobatto_iiia_q4),
    ],
    ids=["gauss", "radau_iia", "lobatto_iiia"],
)
def test_stage_and_classical_order_rise_with_the_stage_count(family) -> None:
    """
    Within each family, both orders increase strictly with q.

    Stage order is the one that matters here: it carries the stage residual, which
    dominates the RK objective, so it -- not the classical order -- is what caps the
    attainable accuracy. Each added stage lifts that ceiling by one power of k.
    """
    stages = [TABLEAUX[f][0] for f in family]
    orders = [TABLEAUX[f][1] for f in family]
    assert stages == [2, 3, 4]
    assert orders[0] < orders[1] < orders[2]
