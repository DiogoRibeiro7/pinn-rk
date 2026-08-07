# pinn-rk • Roadmap

A pragmatic, incremental plan to evolve **pinn-rk** into a robust, research‑grade library for time‑discrete Runge–Kutta PINNs.

---

## Phase 0 — Foundations (Now)

**Goal:** Solid engineering baseline.

* [ ] **Module split completed** (✅ code done) and public API re‑exports.
* [ ] **Tests**: unit, E2E smoke, API surface; coverage target ≥ 85%.
* [ ] **CI**: ruff, mypy, pytest on 3.10–3.12 and 3 OSs; cache Poetry.
* [ ] **Repo hygiene**: issue/PR templates, dependabot, CODEOWNERS.

**Deliverables**

* Passing CI on main
* Coverage badge (Codecov)
* CONTRIBUTING.md, SECURITY.md

**Acceptance**

* Green CI, coverage ≥ 85%, pre‑commit passes locally.

---

## Phase 1 — Numerical breadth

**Goal:** Expand RK methods and core numerics.

* [ ] **Higher‑order RK**: Gauss, Radau IIA, Lobatto for q=3,4.
* [ ] **Analytic time derivative** of the Lagrange interpolant (\hat u_t) (replace finite‑diff).
* [ ] **Sampling strategies**: uniform | Sobol | Halton; stratified in time.
* [ ] **Operators**: Laplacian2D/3D; rectangular domains.
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

* [ ] **Convergence harness**: sweep (N, steps) and RK (Gauss/Radau/Lobatto); save CSV + plots.
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

* [ ] **MkDocs Material** site with:

  * Guide: concepts, RK loss construction, BC strategies
  * Tutorials: 1D heat, 2D Poisson‑type, custom RHS/BCs
  * API reference via mkdocstrings
* [ ] **Examples gallery** (plots, error tables)
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
* [ ] **Wheels** build + `python -m pip install pinn-rk` sanity
* [ ] **Badges**: PyPI, version, downloads

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
