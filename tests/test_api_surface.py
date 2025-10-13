from __future__ import annotations

import inspect
import pinn_rk


def test_public_api_has_expected_symbols() -> None:
    expected = {
        "RkPinnConfig",
        "ButcherTableau",
        "butcher_gauss_legendre_q2",
        "butcher_radau_iia_q2",
        "butcher_lobatto_iiia_q2",
        "TimeMesh",
        "barycentric_weights",
        "lagrange_eval",
        "EllipticOperator",
        "Laplacian1D",
        "MLP",
        "RkPinnLoss",
        "__version__",
    }
    exported = {name for name, _ in inspect.getmembers(pinn_rk)}
    missing = expected - exported
    assert not missing, f"Missing from public API: {missing}"
