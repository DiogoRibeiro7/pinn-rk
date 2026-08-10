from __future__ import annotations

import torch

from pinn_rk.examples.train_heat_equation import l2_error, train_heat_equation


def test_train_and_error_small_run() -> None:
    """
    A short end-to-end training run reaches a sane error.

    Seeded deliberately. Both the network initialisation and the spatial sampler are
    random, and nothing else in this path fixes a seed, so the run was previously
    non-deterministic and the assertion held only by luck. It eventually rolled badly
    and failed on macOS with 0.356 against this same 0.35 bound.

    Measured across eight seeds the run spans 0.015 to 0.112, median 0.034, and none of
    them come close to the bound. Seed 0 gives about 0.026, so the assertion now has
    more than a factor of ten of headroom rather than effectively none.

    The 0.356 that failed in CI is therefore a tail event, not the typical case: a poor
    initialisation sometimes fails to converge within 200 steps. Pinning the seed keeps
    this test measuring the training path rather than sampling that tail.
    """
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = train_heat_equation(
        method="radau2",
        T=0.05,
        N=5,
        n_x_train=64,
        steps=200,  # short run for CI
        lr=1e-2,
        device=device,
    )
    err = l2_error(model, T=0.05, nx=201, device=device)
    # Sanity bound. Seeded, this run gives ~0.03; the bound is deliberately loose so
    # small cross-platform floating-point differences cannot flip it.
    assert err < 0.35, f"Final L2 error too large: {err:.3f}"
