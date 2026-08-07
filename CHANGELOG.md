# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `dtype` parameter on `MLP`, so the network's precision is explicit instead of
  inherited from the global `torch.set_default_dtype` state
- `pinn_rk.examples` is now a regular package, making
  `from pinn_rk.examples import train_heat_equation, l2_error` valid
- Validation of `init_data`: `x0` must be a `[N,1]` tensor sorted in strictly
  increasing order, which the initial-condition penalty relies on
- `.zenodo.json` for archival metadata on release
- `.gitignore`

### Changed

- Renamed the default branch from `develop` to `main`. This also activates the CI
  and release workflows, which were already configured to trigger on `main` and so
  had never run.

### Deprecated

- Nothing yet

### Removed

- Nothing yet

### Fixed

- `RkPinnLoss.forward` raised `RuntimeError` on every call: the interpolant
  `torch.einsum` used `1` as a subscript, which is not a valid subscript label.
  The core loss could not be evaluated at all.
- The H¹ initial-condition penalty called `torch.autograd.grad` on `u0`, which is
  supplied as sampled values and carries no autograd history. The target
  derivative ∂ₓu₀ is now computed numerically on the `x0` grid.
- `MLP` built float32 layers while the loss fed float64 inputs, so constructing a
  model without first calling `torch.set_default_dtype(torch.float64)` failed with
  a dtype mismatch.
- Repository, documentation, and badge URLs pointed at a repository name that does
  not exist (`diogoribeiro7/pinn-rk` rather than `DiogoRibeiro7/rk-pinns`), so every
  badge and link was broken.
- README rendered raw LaTeX as literal text, and its package layout described a
  `rk_pinn.py` module that does not exist.

### Security

- Nothing yet

## [0.1.0] - 2025-01-XX

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

- Barycentric Lagrange interpolation for polynomial reconstruction
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

[0.1.0]: https://github.com/DiogoRibeiro7/rk-pinns/releases/tag/v0.1.0
[unreleased]: https://github.com/DiogoRibeiro7/rk-pinns/compare/v0.1.0...HEAD
