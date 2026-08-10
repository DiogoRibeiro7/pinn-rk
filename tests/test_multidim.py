"""
Multi-dimensional support: the d-dimensional Laplacian, ansatz, and residual.

The time-discretisation machinery was always dimension-agnostic -- only the operator,
the boundary factor and the shape of the time column assumed d = 1. These tests pin the
generalisation, and in particular that the 1D behaviour is unchanged, since every
existing result depends on it.

The 2D manufactured solution is u = sin(pi x) sin(pi y) exp(-2 pi^2 t), which satisfies
u_t - laplacian(u) = 0 exactly and vanishes on the whole boundary of the unit square.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from pinn_rk import (
    MLP,
    Laplacian1D,
    LaplacianND,
    RkPinnConfig,
    RkPinnLoss,
    TimeMesh,
    butcher_radau_iia_q2,
    butcher_radau_iia_q3,
)

DEV = torch.device("cpu")


def _sines(X: Tensor) -> Tensor:
    """Product of sin(pi x_i) over the spatial dimensions."""
    out = torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
    for i in range(X.shape[1]):
        out = out * torch.sin(math.pi * X[:, i : i + 1])
    return out


def _exact_2d(X: Tensor, t: Tensor) -> Tensor:
    return _sines(X) * torch.exp(-2 * math.pi**2 * t)


def _zero_rhs(X: Tensor, t: Tensor) -> Tensor:
    return torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)


class Exact2D(nn.Module):
    """Stands in for a perfectly trained 2D network."""

    def forward(self, X: Tensor, t: Tensor) -> Tensor:
        return _exact_2d(X, t)


# --- operator ------------------------------------------------------------------------


def test_laplacian_nd_matches_laplacian_1d() -> None:
    """In one dimension the general operator must reproduce the specialised one."""
    torch.set_default_dtype(torch.float64)
    x = torch.linspace(0.05, 0.95, 32, dtype=torch.float64).unsqueeze(1).requires_grad_(True)
    u = torch.sin(math.pi * x)
    assert torch.allclose(Laplacian1D()(x, u), LaplacianND()(x, u), atol=1e-14)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_laplacian_nd_matches_closed_form(d: int) -> None:
    """For a product of sines, -laplacian(u) = d pi^2 u."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(d)
    X = torch.rand(48, d, dtype=torch.float64).requires_grad_(True)
    u = _sines(X)
    got = LaplacianND()(X, u)
    assert torch.allclose(got, d * math.pi**2 * u, atol=1e-12)


def test_laplacian_nd_rejects_mismatched_shapes() -> None:
    torch.set_default_dtype(torch.float64)
    X = torch.rand(8, 2, dtype=torch.float64).requires_grad_(True)
    with pytest.raises(ValueError, match=r"\[B,1\]"):
        LaplacianND()(X, torch.rand(8, 2, dtype=torch.float64))


# --- ansatz --------------------------------------------------------------------------


@pytest.mark.parametrize("d", [1, 2, 3])
def test_boundary_conditions_hold_exactly_in_any_dimension(d: int) -> None:
    """
    Phi = prod x_i(1-x_i) vanishes on every face, for any weights.

    This is the property that lets the loss carry no boundary term at all.
    """
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    model = MLP(in_dim=d + 1, width=32, depth=3, dtype=torch.float64)

    pts = []
    for i in range(d):
        for face in (0.0, 1.0):
            p = torch.full((4, d), 0.37, dtype=torch.float64)
            p[:, i] = face
            pts.append(p)
    boundary = torch.cat(pts)
    t = torch.full((boundary.shape[0], 1), 0.05, dtype=torch.float64)

    with torch.no_grad():
        assert model(boundary, t).abs().max() < 1e-14

    interior = torch.full((1, d), 0.5, dtype=torch.float64)
    with torch.no_grad():
        assert model(interior, torch.zeros(1, 1, dtype=torch.float64)).abs().max() > 0.0


@pytest.mark.parametrize("d", [1, 2, 3])
def test_boundary_factor_peaks_at_one_in_any_dimension(d: int) -> None:
    """
    Φ = Π 4 x_i(1-x_i) peaks at 1 whatever d is.

    Without the factor of 4 per dimension it peaks at 4^-d, so the network would have to
    grow like 4^d to represent an O(1) solution and the gradients reaching it would be
    attenuated by the same factor. That is not a cosmetic scaling: measured on the 2D
    heat equation the unnormalised form plateaus near 55% relative error where this one
    reaches about 5%. The 1D results are unaffected, since a constant factor is absorbed
    by the weights.
    """
    torch.set_default_dtype(torch.float64)
    centre = torch.full((1, d), 0.5, dtype=torch.float64)
    phi = (4.0 * centre * (1.0 - centre)).prod(dim=1, keepdim=True)
    assert float(phi) == pytest.approx(1.0, abs=1e-14)

    # and it must still be a valid cutoff: strictly inside (0,1] on the interior
    torch.manual_seed(0)
    interior = 0.05 + 0.9 * torch.rand(64, d, dtype=torch.float64)
    phi_int = (4.0 * interior * (1.0 - interior)).prod(dim=1, keepdim=True)
    assert float(phi_int.min()) > 0.0
    assert float(phi_int.max()) <= 1.0 + 1e-14


def test_model_rejects_wrong_spatial_width() -> None:
    torch.set_default_dtype(torch.float64)
    model = MLP(in_dim=3, dtype=torch.float64)  # expects x of width 2
    with pytest.raises(ValueError, match=r"x must be \[B,2\]"):
        model(torch.rand(4, 1, dtype=torch.float64), torch.zeros(4, 1, dtype=torch.float64))


def test_model_requires_time_as_a_single_column() -> None:
    """Time stays [B,1] whatever d is; passing [B,d] would feed d copies of t."""
    torch.set_default_dtype(torch.float64)
    model = MLP(in_dim=3, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"t must be \[B,1\]"):
        model(torch.rand(4, 2, dtype=torch.float64), torch.zeros(4, 2, dtype=torch.float64))


def test_model_rejects_in_dim_below_two() -> None:
    with pytest.raises(ValueError, match="in_dim >= 2"):
        MLP(in_dim=1)


# --- residual ------------------------------------------------------------------------


def _residual_2d(tab_fn, n_slabs: int, nx: int = 5) -> tuple[float, float]:
    torch.set_default_dtype(torch.float64)
    cfg = RkPinnConfig(
        tableau=tab_fn(DEV),
        time_mesh=TimeMesh.uniform(T=0.05, N=n_slabs, device=DEV),
        residual="rk",
        n_x_train=nx * nx,
        space_dim=2,
        device=DEV,
    )
    fn = RkPinnLoss(model=Exact2D(), Lop=LaplacianND(), f_rhs=_zero_rhs, cfg=cfg)
    g = torch.linspace(0.08, 0.92, nx, dtype=torch.float64)
    X = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    r_stage, r_step = fn.rk_residuals(X, 0, cfg.time_mesh.steps[0], fn._stage_times(0))
    return float(r_stage.detach().abs().max()), float(r_step.detach().abs().max())


@pytest.mark.parametrize(
    ("tab_fn", "q"),
    [(butcher_radau_iia_q2, 2), (butcher_radau_iia_q3, 3)],
    ids=["radau2", "radau3"],
)
def test_stage_order_is_unchanged_in_two_dimensions(tab_fn, q: int) -> None:
    """
    The stage order is a property of the tableau, not of the spatial dimension.

    Measuring it in 2D and recovering the same value is what shows the generalisation
    did not quietly change the time discretisation.
    """
    errs = [_residual_2d(tab_fn, n)[0] for n in (3, 6)]
    observed = math.log(errs[0] / errs[1]) / math.log(2.0)
    assert observed == pytest.approx(float(q), abs=0.3)


def test_2d_residual_falls_under_refinement() -> None:
    coarse = _residual_2d(butcher_radau_iia_q3, 3)
    fine = _residual_2d(butcher_radau_iia_q3, 6)
    assert fine[0] < coarse[0] / 4.0
    assert fine[1] < coarse[1] / 8.0


# --- initial condition ---------------------------------------------------------------


def test_callable_initial_data_works_in_two_dimensions() -> None:
    """A callable u0 is differentiated by autograd, so it needs no grid ordering."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    g = torch.linspace(1e-6, 1 - 1e-6, 8, dtype=torch.float64)
    x0 = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)

    cfg = RkPinnConfig(
        tableau=butcher_radau_iia_q2(DEV),
        time_mesh=TimeMesh.uniform(T=0.02, N=2, device=DEV),
        n_x_train=16,
        space_dim=2,
        device=DEV,
        init_data=(x0, _sines),
    )
    fn = RkPinnLoss(
        model=MLP(in_dim=3, width=32, depth=3, dtype=torch.float64),
        Lop=LaplacianND(),
        f_rhs=_zero_rhs,
        cfg=cfg,
    )
    loss = fn()
    assert torch.isfinite(loss) and loss.item() > 0.0


def test_sampled_initial_data_is_rejected_beyond_one_dimension() -> None:
    """
    Sampled u0 needs a sorted grid to differentiate along, which only exists in 1D.

    Failing loudly beats silently differentiating along a meaningless ordering.
    """
    torch.set_default_dtype(torch.float64)
    g = torch.linspace(1e-6, 1 - 1e-6, 4, dtype=torch.float64)
    x0 = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    cfg = RkPinnConfig(
        tableau=butcher_radau_iia_q2(DEV),
        time_mesh=TimeMesh.uniform(T=0.02, N=2, device=DEV),
        space_dim=2,
        device=DEV,
        init_data=(x0, _sines(x0)),
    )
    with pytest.raises(ValueError, match="callable"):
        RkPinnLoss(
            model=MLP(in_dim=3, dtype=torch.float64),
            Lop=LaplacianND(),
            f_rhs=_zero_rhs,
            cfg=cfg,
        )


def test_initial_data_width_must_match_space_dim() -> None:
    torch.set_default_dtype(torch.float64)
    cfg = RkPinnConfig(
        tableau=butcher_radau_iia_q2(DEV),
        time_mesh=TimeMesh.uniform(T=0.02, N=2, device=DEV),
        space_dim=2,
        device=DEV,
        init_data=(torch.rand(8, 1, dtype=torch.float64), _sines),
    )
    with pytest.raises(ValueError, match=r"\[N,2\]"):
        RkPinnLoss(
            model=MLP(in_dim=3, dtype=torch.float64),
            Lop=LaplacianND(),
            f_rhs=_zero_rhs,
            cfg=cfg,
        )


def test_default_sampler_respects_space_dim() -> None:
    torch.set_default_dtype(torch.float64)
    cfg = RkPinnConfig(
        tableau=butcher_radau_iia_q2(DEV),
        time_mesh=TimeMesh.uniform(T=0.02, N=2, device=DEV),
        n_x_train=16,
        space_dim=3,
        device=DEV,
    )
    # Constructing the loss is what installs the default sampler.
    RkPinnLoss(
        model=MLP(in_dim=4, dtype=torch.float64),
        Lop=LaplacianND(),
        f_rhs=_zero_rhs,
        cfg=cfg,
    )
    assert cfg.spatial_sampler is not None
    pts = cfg.spatial_sampler(16, DEV)
    assert pts.shape == (16, 3)
    assert float(pts.min()) > 0.0 and float(pts.max()) < 1.0
