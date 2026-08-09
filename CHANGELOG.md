# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Four-stage tableaux**: `butcher_gauss_legendre_q4` (classical order 8),
  `butcher_radau_iia_q4` (order 7) and `butcher_lobatto_iiia_q4` (order 6).

  Measured stage order now rises 1.9 → 2.9 → 3.9 across q=2, 3, 4. Since the stage
  residual dominates the objective, that is the ceiling moving from O(k²) to O(k⁴).

- `collocation_tableau(nodes)`, which derives a Butcher tableau from its nodes alone.
  For a collocation method the coefficients are forced: `a_ij = ∫₀^{c_i} L_j`,
  `b_j = ∫₀¹ L_j`. Gauss-Legendre, Radau IIA and Lobatto IIIA are all collocation
  families, so this reconstructs any of them, and adding a new one now needs only its
  nodes. Used internally for q=4, where the A matrices have no workable closed form and
  writing out 16 coefficients apiece would be transcription risk with no benefit.

- Tests covering the q=4 tableaux and the builder, including one asserting that Gauss
  q=4's update residual **stalls at the float64 floor**. It is order 8, and
  `(u_{n+1} - u_n)/k` carries rounding of about ε/k — roughly 2.6e-14 at k=1/120 — so
  the true residual falls beneath it and refining the mesh stops helping. Pinning that
  behaviour keeps the flat region from being read as a convergence failure.

### Changed

- `train_heat_equation` and `examples/04_convergence_study.py` accept the q=4 methods.
- The order-condition tests distinguish hard-coded tableaux from derived ones. Comparing
  a derived tableau against the derivation that produced it proves nothing, so for q=4
  the verification rests on the order conditions instead — including a check that B(p+1)
  *fails*, which pins the order from above as well as below.

## [0.4.0] - 2026-08-09

Raises the accuracy ceiling with three-stage tableaux, fills in the two example scripts
that had been documented but never written, and gets the documentation site building and
deployed for the first time.

### Added

- **Three-stage tableaux**: `butcher_gauss_legendre_q3` (classical order 6),
  `butcher_radau_iia_q3` (order 5) and `butcher_lobatto_iiia_q3` (order 4), exported from
  the package root and selectable through `train_heat_equation(method=...)`.

  These lift the accuracy ceiling. The stage residual converges at the *stage order*,
  which for collocation methods equals the stage count, and it dominates the objective —
  so the two-stage families capped the whole scheme at O(k²) regardless of their
  classical order. Measured stage order rises from 1.91-1.94 to 2.91-2.92, and Gauss's
  stage residual falls by roughly 190x at the same slab size.

- `tests/test_tableau_order.py`, verifying every shipped tableau algebraically. All three
  families are collocation methods, so A and b are forced by the nodes; the test
  re-derives them by integrating the Lagrange basis and compares, then checks the order
  conditions B(p) and C(q), the row sums, stiff accuracy, and node distinctness. A
  mistyped coefficient fails here rather than silently degrading the method.

- `examples/03_custom_operator.py` — a reaction-diffusion operator implementing
  `EllipticOperator`, with a non-zero right-hand side from a manufactured solution, and a
  validation step that checks the operator against its closed form before training.

- `examples/04_convergence_study.py` — sweeps `N`, fits log-log rates, and reports
  consistency error and trained L2 error side by side while keeping them clearly
  distinct. Supports `--train`, `--save-plots`, `--output-dir` and `--export-csv`, the
  flags the examples README had long documented without them existing.

- A documentation site that actually builds: API reference pages generated with
  mkdocstrings, an installation guide, a quick start, and a page deriving the RK-PINN
  formulation. `docs/javascripts/mathjax.js` was missing, so no LaTeX in the docs would
  have rendered.

- `examples/notebooks/visualization.ipynb` — training history shown with its stochastic
  noise rather than smoothed, solution evolution against the exact solution, a
  space-time error heat map, an interactive Plotly time slider, and a same-seed
  comparison of `residual="rk"` against `residual="interpolant"`.
- `examples/notebooks/benchmarking.ipynb` — cost per step split into forward and
  backward, scaling in slabs and spatial batch, autograd-graph memory, a CPU/GPU (or
  thread-count) comparison, and the accuracy-per-unit-cost study showing all three
  2-stage tableaux recovering classical orders 4, 3 and 2.
- `tests/test_notebooks.py`, asserting the committed notebooks parse, store no error
  outputs, and were genuinely executed rather than authored by hand. It reads the
  notebooks as plain JSON so it runs in CI, where the optional `examples` dependency
  group is not installed.

### Fixed

- `examples/README.md` documented `03_custom_operator.py`, `04_convergence_study.py`,
  and both notebooks, none of which existed. The notebooks now exist; the two scripts
  are labelled as not yet written instead of being advertised as available.
- Removed a malformed YAML front-matter block at the top of `examples/README.md`, which
  rendered as raw text and duplicated the description of example 02.
- Replaced the "Convergence Rates" table in `examples/README.md`. Its figures were
  invented — a suspiciously smooth 0.91/0.92/0.92 — and are now the measured
  consistency orders, labelled as such rather than implied to be trained-model results.
- The documented `--save-plots`, `--output-dir` and `--export-csv` flags belonged to a
  script that did not exist. `04_convergence_study.py` now implements them.
- `mkdocs build` failed outright: `pymdownx.emoji` was configured with quoted strings
  where Markdown calls the values, aborting with `TypeError: unsupported callable`. The
  nav also listed 24 pages of which only one existed. Both fixed, so the site builds
  under `--strict`.
- `mkdocs.yml` loaded `polyfill.io`, a domain that changed hands in 2024 and served
  malicious code. Removed; MathJax 3 needs no polyfill in supported browsers.
- `docs/index.md` linked to two pages that were never written.
- `.coverage`, a binary SQLite artifact, was tracked in git. Untracked; it was already
  listed in `.gitignore`.

## [0.3.0] - 2026-08-08

Brings the Butcher matrix `A` into the loss, so the residual is the Runge-Kutta
method rather than a reconstruction that merely borrowed its nodes and weights.
Each tableau now recovers its own classical order; previously all three behaved
identically because `A` was validated and never read.

Users upgrading from 0.2.0 should expect different loss values and different
trained weights, and should read the note in the README on what the consistency
measurements do and do not imply about training.

### Added

- `RkPinnConfig.residual`, selecting which residual the loss imposes.
  `"rk"` (new default) uses the **full Butcher tableau**: the stage equations
  `U_i = u_n + k Σ_j a_ij F_j` bring the coupling matrix `A` into the loss for the
  first time, and the update equation `u_{n+1} = u_n + k Σ_i b_i F_i` carries the
  tableau's classical order. `"interpolant"` is the previous behaviour, retained
  for comparison.
- `RkPinnLoss.rk_residuals(...)`, returning the stage and update residuals
  separately. They converge at different rates and only the update residual
  reflects the classical order, so they are measured independently.
- `residual` argument on `train_heat_equation`.
- `tests/test_rk_order.py`, pinning the measured convergence orders per tableau.
- `RkPinnConfig.ic_weight`, scaling the initial-condition penalty against the PDE
  residual. The two were summed unweighted and their balance drifts sharply during
  training: on the shipped example the penalty is essentially the whole loss at
  initialisation (5.1 against a residual of 1.7e-4) and about a fifth of it after a
  few hundred steps, so training begins by fitting the initial condition almost
  exclusively. `ic_weight=0.0` drops the penalty and measures the residual alone.
- Zenodo DOIs, now that the archive exists. The README badge and BibTeX entry, and
  `CITATION.cff`, carry the concept DOI `10.5281/zenodo.21839391`, which always
  resolves to the latest version; the per-version DOIs are recorded alongside it.
- A documented release checklist in `CONTRIBUTING.md`, covering the four places the
  version has to agree, why `.zenodo.json` deliberately carries no version field,
  and the fact that publishing a release mints a DOI and is not reversible.

### Changed

- **The choice of Butcher tableau now affects accuracy.** Measured on the
  manufactured solution, the update residual converges at each method's classical
  order — 4 for Gauss q=2, 3 for Radau IIA q=2, 2 for Lobatto IIIA q=2 — against
  a uniform order 2 previously. At the slab size used by the shipped example the
  Gauss update residual is 1.3e-8, roughly 10^5 times smaller than the
  interpolant form's 1.8e-3. Under the old residual all three tableaux behaved
  identically, because `A` was never read.
- `RkPinnConfig` gained a field ahead of `q_aux`, so any caller passing
  configuration positionally past `time_mesh` must move to keyword arguments.
  Everything in this repository already used keywords.
- `q_aux` now applies only when `residual="interpolant"`, and is ignored otherwise.

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
[0.3.0]: https://github.com/DiogoRibeiro7/pinn-rk/compare/v0.2.0...v0.3.0
[0.4.0]: https://github.com/DiogoRibeiro7/pinn-rk/compare/v0.3.0...v0.4.0
[unreleased]: https://github.com/DiogoRibeiro7/pinn-rk/compare/v0.4.0...HEAD
