# pinn-rk • Roadmap

A pragmatic, incremental plan to evolve **pinn-rk** into a robust, research‑grade library for time‑discrete Runge–Kutta PINNs.

---

## Where things stand

0.6.0 is on PyPI, archived on Zenodo under concept DOI
[10.5281/zenodo.21839391](https://doi.org/10.5281/zenodo.21839391), documented at
<https://diogoribeiro7.github.io/pinn-rk/>, and green on CI across three operating
systems and three Python versions.

| Phase | State |
| --- | --- |
| 0 Foundations | done |
| 1 Numerical breadth | done except samplers and switchable BCs |
| 2 Convergence & stability | convergence harness only |
| 3 Performance | not started |
| 4 Documentation site | done except two guide pages |
| 5 cG/dG in time | not started |
| 6 Packaging & releases | done |

The honest summary of the numerics: the tableau now genuinely drives accuracy, and
measured stage order tracks the stage count up to q=4, where double precision runs
out. What is missing is breadth of evidence rather than machinery — manufactured
solutions, error norms and ablations are all still open.

---

## Phase 0 — Foundations (complete)

**Goal:** Solid engineering baseline.

* [x] **Module split completed** and public API re‑exports.
* [x] **Tests**: unit, E2E smoke, API surface, convergence orders. Coverage is 93%
  against a target of 85%.
* [x] **CI**: ruff, mypy, pytest on 3.10–3.12 across Linux, macOS and Windows.
  Poetry-based dependency caching was **removed deliberately**: `setup-python`'s
  `cache: "poetry"` needs Poetry on PATH before that step runs, and installing it
  first is unreliable on macOS, where `setup-python` replaces the Python that Poetry
  was installed into. The job already caches `.venv` keyed on `poetry.lock`, so the
  extra cache bought nothing and broke one matrix entry.
* [x] **Repo hygiene**: issue/PR templates, dependabot, CODEOWNERS, `.gitignore`.
  Dependabot alerts are at zero.
* [ ] **Branch protection** on `main`: require a pull request and a green
  **All Checks Passed** before merge. Every change since 0.1.0 has gone through a PR
  with CI green, but nothing enforces it — a direct push to `main` would succeed
  today.

**Deliverables**

* [x] Passing CI on main
* [x] Coverage badge (Codecov), reporting 93%. Uploads are tokenless, which Codecov
  warns about on a protected branch; it works, but a `CODECOV_TOKEN` would make it
  robust against their rate limits.
* [x] CONTRIBUTING.md, SECURITY.md

**Acceptance**

* [x] Green CI, coverage ≥ 85%, pre‑commit passes locally.

---

## Phase 1 — Numerical breadth

**Goal:** Expand RK methods and core numerics.

* [x] **Higher‑order RK**: Gauss, Radau IIA and Lobatto IIIA at q=3 and q=4 — classical
  orders up to 8. Each added stage lifts the accuracy ceiling, because the stage
  residual converges at the stage order (= stage count) and dominates the objective:
  measured stage order rises ~1.9 → ~2.9 → ~3.9. `collocation_tableau` builds any
  collocation family from its nodes, so a new one needs only the nodes.

  Beyond q=4 the residual outruns double precision — Gauss q=4 is already order 8, and
  `(u_{n+1} - u_n)/k` carries rounding of about ε/k, which the true residual falls
  below. Going higher needs a reformulation of the residual, not more nodes.
* [x] **Analytic time derivative** of the Lagrange interpolant (\hat u_t) (replace finite‑diff)
  — done in v0.2.0, and retained as `residual="interpolant"`.
* [x] **Full Butcher tableau in the residual** — the stage equations bring `A` into
  the loss, giving measured orders of 4 (Gauss q=2), 3 (Radau IIA q=2) and
  2 (Lobatto IIIA q=2). Previously `A` was validated but never read, so the choice
  of tableau had no effect on accuracy.
* [ ] **Sampling strategies**: uniform | Sobol | Halton; stratified in time.
* [x] **Operators**: `LaplacianND` covers any number of spatial dimensions, verified in
  1D, 2D and 3D, and the ansatz, sampler and residual work on the unit cube `[0,1]^d`.
  The stage orders measured in 2D match 1D exactly, so the time discretisation is
  unchanged. Rectangular (non-unit) domains are still open: the boundary factor and the
  default sampler both assume `[0,1]^d`.
* [ ] **Boundary handling options**: hard BC via (\phi(x)) vs soft penalty; switchable.

**Deliverables**

* [x] `tableau.py` with q=3,4 factories, plus `collocation_tableau` for any others
* [x] `interpolants.py` gains the barycentric basis and `differentiation_matrix`
* [ ] `samplers.py` with Sobol/Halton
* [x] New tests (properties + convergence checks)

**Acceptance**

* [x] Interpolation tests: partition of unity, node exactness, derivative agreement
* [x] Tiny training runs pass with q=3 on CI

---

## Phase 2 — Convergence, stability & MR‑aligned design

**Goal:** Evidence of correctness and stable training design.

* [x] **Convergence harness**: `examples/04_convergence_study.py` sweeps N across any
  subset of the nine tableaux, fits log-log rates, and supports `--train`,
  `--save-plots` and `--export-csv`. It reports consistency error and trained L2
  separately, since the first bounds the second but does not predict it.
* [ ] **Manufactured solutions**: 1D/2D with non‑zero (f), time‑varying BCs.
* [ ] **Error norms**: L2 and H1 utilities; optional final‑time H1 penalty.
* [ ] **Ablations**: BC enforcement (hard vs soft), sampler choice, RK scheme.

**Deliverables**

* [x] Convergence scripts + results. These landed as
  `examples/04_convergence_study.py` and `examples/notebooks/benchmarking.ipynb`
  rather than a separate `bench/` directory, which would have duplicated the
  examples for no gain.
* [ ] `pinn_rk/metrics.py` with L2/H1 estimators

**Acceptance**

* Plots showing expected rate trends vs RK order (qualitative)
* Reproducible runs via documented seed & config

---

## Phase 3 — Performance engineering

**Goal:** Faster training without compromising clarity.

* [ ] **`torch.compile`** flag (AOTAutograd) + fallbacks.
* [ ] **AMP** (mixed precision) on CUDA; autocast context in training helpers.
* [ ] **Profiler recipe**: `torch.profiler` traces + README guidance.
* [ ] Micro‑optimisations: reduce graph breaks, reuse buffers, JIT small kernels if helpful.

**Deliverables**

* `training.py` with compile/AMP toggles
* `docs/performance.md` with profiler screenshots

**Acceptance**

* Documented wall‑clock improvements on reference run (≥ 1.3× on CUDA in AMP mode)

---

## Phase 4 — Documentation site

**Goal:** Publish high‑quality docs.

* [x] **MkDocs Material** site, live at
  <https://diogoribeiro7.github.io/pinn-rk/>:

  * Getting started: installation and quick start
  * Theory: the RK-PINN formulation and what the measured orders do and do not mean
  * API reference via mkdocstrings for all seven modules
* [x] **Examples gallery**: two notebooks committed with executed outputs, so the
  plots and error tables render on GitHub without being run.
* [ ] A 2D tutorial page. No longer blocked: `LaplacianND` landed in 0.6.0 and
  `examples/05_2d_heat_equation.py` solves the 2D heat equation end to end. What is
  missing is only the prose.
* [ ] A guide page on BC strategies, which needs the soft-BC option from Phase 1
  first — with only the hard constraint implemented there is nothing to compare.
* [ ] **Links** to paper(s) and comparisons

**Deliverables**

* [x] `docs/` + `mkdocs.yml`, a CI job that builds the site on every pull request,
  and a Pages deploy. The build step was added after a release shipped with docs that
  did not build; CI now catches that before merge.

**Acceptance**

* [x] Clean site build; internal links validated; examples runnable

---

## Phase 5 — Extended time discretisations (optional)

**Goal:** Generalise time‑discrete PINN beyond RK.

* [ ] **cG/dG in time** under the same pointwise form
* [ ] Quadrature & projection backends for cG/dG
* [ ] Unified loss interface: `backend = {"rk","cg","dg"}`

**Deliverables**

* `timebackends/` with rk/cg/dg implementations
* Examples comparing RK vs cG/dG

**Acceptance**

* Smoke tests for cg/dg; convergence harness includes cg/dg sweeps

---

## Phase 6 — Packaging & releases

**Goal:** Stable releases with changelog and tags.

* [x] **Manual release process**, documented in [CONTRIBUTING.md](./CONTRIBUTING.md#releasing).
  Semantic-release was tried and dropped: it ran on every push to `main` and failed
  on each one, and tagging is infrequent enough that the automation cost more than
  it saved. Revisit only if release frequency rises.
* [x] **Zenodo archiving**: releases are archived automatically and minted a DOI
  (concept DOI `10.5281/zenodo.21839391`)
* [x] **Wheels** build + `pip install pinn-rk` — published as 0.6.0. Uploads run from
  CI over OIDC trusted publishing, with no API token anywhere, and the build fails if
  the pyproject version disagrees with the release tag.
* [x] **Badges**: PyPI version, Python versions and downloads, all resolving now that
  the package exists.
* [ ] Expose the `examples` and `docs` groups as pip extras, so
  `pip install "pinn-rk[examples]"` works. They are Poetry groups today, which pip
  cannot see.

**Deliverables**

* Documented, repeatable manual release checklist
* CI green on the tagged commit before any release is published

**Acceptance**

* [x] `v0.1.0` through `v0.6.0` released, each archived by Zenodo and citable by its
  own version DOI under the concept DOI above

---

## Stretch goals

* [ ] Domain‑specific samplers (e.g., boundary‑biased)
* [ ] Adaptive time partition (local error indicator)
* [ ] PDE suites: advection‑diffusion, reaction‑diffusion, Burgers (viscous)
* [ ] Optional JAX backend for research comparisons

---

## Issue seeds (copy/paste to GitHub)

Still open:

* feat: Sobol/Halton sampler; config hook and tests
* feat: soft boundary penalty as an alternative to the hard ansatz, switchable
* feat: manufactured solutions with non-zero source and time-varying BCs
* feat: rectangular domains — the boundary factor and sampler both assume `[0,1]^d`
* feat: expose the `examples` and `docs` groups as pip extras
* test: property‑based tests for interpolation & derivatives (Hypothesis)
* perf: torch.compile + AMP toggle and benchmarks
* docs: 2D tutorial page; BC strategies guide
* chore: branch protection on `main`

Closed:

* ~~feat: add Radau IIA q=3 tableau + unit tests~~ — v0.4.0, alongside q=4 in v0.5.0
* ~~feat: implement analytic Lagrange derivative; replace FD in loss~~ — v0.2.0
* ~~feat: Laplacian2D/3D~~ — v0.6.0, as `LaplacianND` for arbitrary `d`
* ~~docs: MkDocs site with tutorials and API~~ — v0.4.0, deployed to Pages
* ~~bench: convergence harness and plots~~ — v0.4.0
* ~~release: semantic‑release setup and first tagged release~~ — released
  manually from v0.1.0 to v0.6.0; semantic‑release dropped, see Phase 6

---

## Versioning plan

Where the releases actually went (details in [CHANGELOG.md](./CHANGELOG.md)):

* **v0.1.0** – Foundations: module split, tests, CI, first DOI
* **v0.2.0** – Analytic Lagrange time derivative in place of finite differences
* **v0.3.0** – The stage equations, which bring the Butcher `A` matrix into the loss
* **v0.4.0** – q=3 tableaux, convergence harness, notebooks, docs site
* **v0.5.0** – q=4 tableaux and `collocation_tableau` (v0.5.1 reconciled the docs
  with what the code does)
* **v0.6.0** – Multi-dimensional domains, and publication to PyPI

Ahead, subject to change:

* **v0.7.x** – Samplers, switchable BCs, manufactured solutions (rest of Phases 1–2)
* **v0.8.x** – Performance: `torch.compile`, AMP, profiling (Phase 3)
* **v0.9.x** – cG/dG optional backends, broader PDE gallery (Phase 5)
* **v1.0.0** – Stable API, documented guarantees, benchmarks

The API has been additive since 0.1.0: every release added configuration or tableaux
without breaking a documented call. `1.0.0` is the point at which that becomes a
promise rather than an observation.
