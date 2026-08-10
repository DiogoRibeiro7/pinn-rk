# pinn-rk • Roadmap

A pragmatic, incremental plan to evolve **pinn-rk** into a robust, research‑grade library for time‑discrete Runge–Kutta PINNs.

---

## Phase 0 — Foundations (Now)

**Goal:** Solid engineering baseline.

* [x] **Module split completed** and public API re‑exports.
* [x] **Tests**: unit, E2E smoke, API surface, convergence orders. Coverage is 91%
  against a target of 85%.
* [x] **CI**: ruff, mypy, pytest on 3.10–3.12 across Linux, macOS and Windows.
  Poetry-based dependency caching was **removed deliberately**: `setup-python`'s
  `cache: "poetry"` needs Poetry on PATH before that step runs, and installing it
  first is unreliable on macOS, where `setup-python` replaces the Python that Poetry
  was installed into. The job already caches `.venv` keyed on `poetry.lock`, so the
  extra cache bought nothing and broke one matrix entry.
* [x] **Repo hygiene**: issue/PR templates, dependabot, CODEOWNERS, `.gitignore`.

**Deliverables**

* Passing CI on main
* Coverage badge (Codecov)
* CONTRIBUTING.md, SECURITY.md

**Acceptance**

* Green CI, coverage ≥ 85%, pre‑commit passes locally.

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

* `tableau.py` with q=3,4 factories
* `interpolants.py` gains `lagrange_basis_and_derivative`
* `samplers.py` (optional) with Sobol/Halton
* New tests (properties + convergence checks)

**Acceptance**

* Interpolation tests: partition of unity, node exactness, derivative agreement
* Tiny training runs pass with q=3 on CI

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

* `bench/` with scripts + results; `notebooks/benchmarks.ipynb`
* `pinn_rk/metrics.py` with L2/H1 estimators

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
* [ ] Guide pages on BC strategies, and a 2D tutorial (blocked on 2D operators)
* [ ] **Links** to paper(s) and comparisons

**Deliverables**

* `docs/` + `mkdocs.yml`; CI job to build; optional Pages deploy

**Acceptance**

* Clean site build; internal links validated; examples runnable

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

* [x] `v0.1.0` and `v0.2.0` released, archived, and citable by DOI

---

## Stretch goals

* [ ] Domain‑specific samplers (e.g., boundary‑biased)
* [ ] Adaptive time partition (local error indicator)
* [ ] PDE suites: advection‑diffusion, reaction‑diffusion, Burgers (viscous)
* [ ] Optional JAX backend for research comparisons

---

## Issue seeds (copy/paste to GitHub)

* feat: add Radau IIA q=3 tableau + unit tests
* ~~feat: implement analytic Lagrange derivative; replace FD in loss~~ — done in v0.2.0
* feat: Sobol sampler; config hook and tests
* feat: Laplacian2D/3D + manufactured solutions
* test: property‑based tests for interpolation & derivatives (Hypothesis)
* perf: torch.compile + AMP toggle and benchmarks
* docs: MkDocs site with tutorials and API
* bench: convergence harness and plots
* ~~release: semantic‑release setup and first tagged release~~ — released manually
  as v0.1.0 and v0.2.0; semantic‑release dropped, see Phase 6

---

## Versioning plan

* **v0.1.x** – Foundations + q=2 RK + tests/CI
* **v0.2.x** – Phase 1–3 features, docs draft
* **v0.3.x** – cG/dG optional backends, broader PDE gallery
* **v1.0.0** – Stable API, documented guarantees, benchmarks
