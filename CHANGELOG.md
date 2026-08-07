# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Zenodo DOIs, now that the archive exists. The README badge and BibTeX entry, and
  `CITATION.cff`, carry the concept DOI `10.5281/zenodo.21839391`, which always
  resolves to the latest version; the per-version DOIs are recorded alongside it.
- A documented release checklist in `CONTRIBUTING.md`, covering the four places the
  version has to agree, why `.zenodo.json` deliberately carries no version field,
  and the fact that publishing a release mints a DOI and is not reversible.

### Removed

- `.github/workflows/release.yml`. Releases are cut manually. The workflow ran
  `semantic-release publish` on every push to `main` and failed on each one:
  python-semantic-release 9.x moves versioning and tagging into
  `semantic-release version`, and the required `[tool.semantic_release]`
  configuration was never present. Both releases to date were tagged by hand, so
  removing it drops nothing that worked.

## [0.2.0] - 2026-08-07

Wires the polynomial time reconstruction into the residual, so the loss now
computes what the documented formulation always described. Users upgrading from
0.1.0 should expect different loss values and different trained weights: the
residual is measured against the interpolant û rather than against a finite
difference of the network.

### Added

- `differentiation_matrix(nodes, w=None)` in `pinn_rk.interpolants`, exported from
  the package root. Returns `D[i, j] = L_j'(t_i)` from the barycentric weights,
  exact for polynomials of degree ≤ q-1.
- `RkPinnConfig.q_aux` is now honoured rather than ignored. `"same"` reconstructs
  û through the q stage values (degree q-1); `"extend"` also uses the slab start
  t_n (degree q, one extra network evaluation per slab). Tableaux whose first
  stage already sits at t_n, such as Lobatto IIIA with c_1 = 0, are left
  unextended, since a duplicated node makes the barycentric weights infinite.
- `tests/test_time_reconstruction.py`, which measures the residual's consistency
  error against the manufactured solution and pins the observed convergence
  orders, so a regression shows up as a failing order rather than a slightly
  worse loss.

### Changed

- ∂ₜû at the stage nodes is now taken analytically from the polynomial time
  reconstruction instead of by symmetric finite differences of the network. This
  is what the README's formulation always described: the residual is measured
  against the interpolant û, not against the raw network.
- The barycentric interpolation utilities are now actually used by the loss. They
  previously backed only a dead expression whose result was discarded.
- `RkPinnConfig.q_aux` now defaults to `"extend"` rather than `"same"`. Measured
  on the manufactured solution, the residual's consistency error is O(k) for
  `"same"` and O(k²) for `"extend"` — at the slab size used by the shipped
  example, 1.6e-1 against 2.6e-3. The previous value was never actually read, so
  no existing behaviour changes.

### Removed

- The finite-difference time-derivative path, along with the `eps = 1e-6 * k_n`
  step size it depended on. It cost 2q network evaluations per slab and made the
  residual sensitive to the choice of eps.

## [0.1.0] - 2026-08-07

First published release. The package was developed on an unreleased `develop`
branch and never tagged, so this entry covers both the initial feature set and
the correctness fixes made before publication.

### Fixed

Each of the following was present throughout development and is fixed here, in
the first release to reach users:

- `RkPinnLoss.forward` raised `RuntimeError` on every call: the interpolant
  `torch.einsum` used `1` as a subscript, which is not a valid subscript label.
  The core loss could not be evaluated at all.
- The H¹ initial-condition penalty called `torch.autograd.grad` on `u0`, which is
  supplied as sampled values and carries no autograd history. The target
  derivative ∂ₓu₀ is now computed numerically on the `x0` grid.
- `MLP` built float32 layers while the loss fed float64 inputs, so constructing a
  model without first calling `torch.set_default_dtype(torch.float64)` failed with
  a dtype mismatch.
- README rendered raw LaTeX as literal text, and its package layout described a
  `rk_pinn.py` module that does not exist.
- CI never ran: the `test` job requested Poetry-based dependency caching before
  installing Poetry, and no workflow triggered on the actual default branch.

### Added

- Initial release of pinn-rk
- Runge-Kutta Physics-Informed Neural Networks with time-discrete losses
- Support for three RK schemes:

  - Gauss-Legendre q=2 (order 4, A-stable)
  - Radau IIA q=2 (order 3, L-stable)
  - Lobatto IIIA q=2 (trapezoidal rule, A-stable)

- Core components:

  - `ButcherTableau`: RK method representation
  - `TimeMesh`: Time discretization
  - `MLP`: Boundary-conditioned neural network ansatz
  - `Laplacian1D`: 1D Laplacian operator
  - `RkPinnLoss`: Time-discrete RK-PINN loss function
  - `RkPinnConfig`: Configuration dataclass

- Barycentric Lagrange interpolation utilities (`barycentric_weights`,
  `lagrange_eval`). Note that the time-discrete loss currently approximates ∂ₜû by
  finite differences and does not yet use them for polynomial reconstruction.
- Boundary conditioning via multiplicative factor Φ(x) = x(1-x)
- Optional H¹ seminorm initial condition penalty
- Complete example: 1D heat equation with manufactured solution
- Comprehensive test suite with >85% coverage
- Type hints throughout (mypy strict mode)
- Code quality checks (ruff, mypy, bandit)
- Pre-commit hooks for automated checks
- GitHub Actions CI across Python 3.10, 3.11, 3.12
- Cross-platform testing (Linux, macOS, Windows)
- MIT License
- Detailed README with installation and usage instructions
- ROADMAP for future development
- `dtype` parameter on `MLP`, so the network's precision is explicit rather than
  inherited from the global `torch.set_default_dtype` state
- `pinn_rk.examples` as a regular package, making
  `from pinn_rk.examples import train_heat_equation, l2_error` valid
- Validation of `init_data`: `x0` must be a `[N,1]` tensor sorted in strictly
  increasing order, which the initial-condition penalty relies on
- `.zenodo.json` for archival metadata, and `.gitignore`

### Changed

- Default branch renamed from `develop` to `main`, which is what the CI and
  release workflows were already configured to trigger on.

### Documentation

- README with quick start guide
- ROADMAP with phased development plan
- Inline NumPy-style docstrings
- Type annotations for all public APIs

### Testing

- Unit tests for all core components
- Integration tests for training workflow
- Property-based tests for interpolation
- Boundary condition validation tests
- Operator gradient tests
- Loss sanity checks

### Infrastructure

- Poetry for dependency management
- Pre-commit hooks for code quality
- GitHub Actions CI/CD
- Dependabot for dependency updates
- Ruff for linting
- Mypy for type checking
- Pytest for testing with coverage reporting
- Bandit for security scanning

## Release Notes

### Version 0.1.0

This is the first public release of **pinn-rk**, a PyTorch library for solving time-dependent PDEs using Runge-Kutta Physics-Informed Neural Networks.

**Key Features:**

- Time-discrete loss formulation using RK collocation
- Multiple high-order RK schemes (Gauss, Radau, Lobatto)
- Exact boundary condition enforcement through neural ansatz
- Modular design for easy extension
- Type-safe, well-tested implementation

**Example Use Case:**

```python
from pinn_rk.examples import train_heat_equation, l2_error

model = train_heat_equation(
    method="radau2",
    T=0.1,
    N=20,
    n_x_train=256,
    steps=1000,
    lr=2e-3,
)
print(f"L2 error: {l2_error(model, T=0.1):.3e}")
```

**Limitations:**

- Currently supports 1D spatial domains only
- Homogeneous Dirichlet boundary conditions only
- Limited to linear parabolic PDEs in examples
- Time derivative approximated via finite differences

**Future Plans:** See [ROADMAP.md](./ROADMAP.md) for planned features including:

- Higher-order RK methods (q ≥ 3)
- 2D/3D operators
- Analytic time derivative computation
- Additional boundary condition types
- Adaptive time stepping

--------------------------------------------------------------------------------

## How to Use This Changelog

### For Users

Check this file to see what's new in each release, including:

- New features you can use
- Bug fixes
- Breaking changes that might affect your code
- Deprecation notices

### For Contributors

When making changes:

1. Add your changes under the `[Unreleased]` section
2. Use the appropriate category (Added, Changed, Fixed, etc.)
3. Write user-facing descriptions, not technical implementation details
4. Link to relevant issues/PRs using #123 syntax
5. Before release, move unreleased changes to a new version section

### Categories Explained

- **Added**: New features or functionality
- **Changed**: Changes to existing functionality
- **Deprecated**: Features that will be removed in future versions
- **Removed**: Features that have been removed
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes

--------------------------------------------------------------------------------

[0.1.0]: https://github.com/DiogoRibeiro7/pinn-rk/releases/tag/v0.1.0
[0.2.0]: https://github.com/DiogoRibeiro7/pinn-rk/compare/v0.1.0...v0.2.0
[unreleased]: https://github.com/DiogoRibeiro7/pinn-rk/compare/v0.2.0...HEAD
