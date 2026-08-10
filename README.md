# pinn-rk

<p align="center">
  <a href="https://github.com/DiogoRibeiro7/pinn-rk/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/DiogoRibeiro7/pinn-rk/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="https://codecov.io/gh/DiogoRibeiro7/pinn-rk"><img alt="Coverage" src="https://codecov.io/gh/DiogoRibeiro7/pinn-rk/branch/main/graph/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue">
  <a href="https://github.com/DiogoRibeiro7/pinn-rk/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://doi.org/10.5281/zenodo.21839391"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.21839391.svg"></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Code style: ruff" src="https://img.shields.io/badge/code%20style-ruff-000000.svg"></a>
</p>

Runge–Kutta Physics‑Informed Neural Networks (PINNs) with **time‑discrete losses** in PyTorch. Ships Gauss–Legendre, Radau IIA and Lobatto IIIA at 2, 3 and 4 stages — classical orders up to 8 — with a boundary‑conditioned neural ansatz, d‑dimensional operators, and end‑to‑end examples for the 1D and 2D heat equations.

> See **[ROADMAP.md](./ROADMAP.md)** for milestones and planned features.

---

## Key features

* **Time‑discrete residual** built from Runge–Kutta collocation: residuals evaluated at stage nodes and integrated with RK weights.
* **General RK backend** via `ButcherTableau` — nine tableaux included, and `collocation_tableau(nodes)` derives a new collocation family from its nodes alone.
* **Boundary conditioning** through a multiplicative factor $\Phi(x)$ to satisfy homogeneous Dirichlet BCs exactly.
* **Modular PDE operators** with autograd‑based derivatives: `Laplacian1D`, and `LaplacianND` for any number of spatial dimensions.
* **1D, 2D and 3D domains** on the unit cube, with exact homogeneous Dirichlet conditions on every face.
* **Practical implementation**: type hints, ruff/mypy clean, tests, and GitHub Actions CI.

---

## Installation (development)

```bash
# Using Poetry (recommended)
poetry install
pre-commit install
```

> Requires Python 3.10–3.12. Install a CPU or CUDA build of PyTorch appropriate for your environment.

---

## Quick start

Train on the 1D heat equation $u_t - u_{xx} = 0$ on $(0,1)$ with homogeneous Dirichlet BCs and initial condition $u_0(x) = \sin(\pi x)$:

```python
from pinn_rk.examples.train_heat_equation import train_heat_equation, l2_error
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = train_heat_equation(
    method="radau3",   # gauss2|radau2|lobatto2 · gauss3|radau3|lobatto3 · gauss4|radau4|lobatto4
    T=0.1,
    N=20,
    n_x_train=256,
    steps=1000,
    lr=2e-3,
    device=device,
)
print("L2(T=0.1) =", l2_error(model, T=0.1, nx=1001, device=device))
```

Expected output (ballpark): `L2(T=0.1) ~ 1e-2 … 1e-1` depending on training steps and hardware.

---

## Concept overview

We consider linear parabolic PDEs of the form

$$
u_t + \mathcal{L}u = f \quad \text{in } \Omega \times (0,T], \qquad u = 0 \text{ on } \partial\Omega, \qquad u(\cdot, 0) = u_0 .
$$

The time interval is partitioned into slabs $J_n = [t_n, t_{n+1}]$ with step $k_n$. For a $q$-stage RK method with nodes $c_i$, we form **stage times** $t_{n,i} = t_n + c_i k_n$ and evaluate the network $u_\theta(x,t)$ and the operator $\mathcal{L}u_\theta$ at these times. The **discrete RK residual** is accumulated using the quadrature weights $b_i$:

$$
\int_{J_n} \lVert r(t) \rVert^2 \, dt \;\approx\; k_n \sum_{i=1}^{q} b_i \, \lVert r(t_{n,i}) \rVert^2 , \qquad r := \partial_t \hat{u} + \Pi_{q-1}(\mathcal{L}\hat{u}) - \tilde{\Pi}_{q-1} f .
$$

Here $\hat{u}$ is a polynomial time interpolant on each slab, and $\Pi_{q-1}$, $\tilde{\Pi}_{q-1}$ are degree $q-1$ projections realized at the collocation nodes.

### The residual

Writing the semi-discrete problem as $u' = f - \mathcal{L}u =: F$, the Runge–Kutta method is *defined* by its stage and update equations:

$$
U_i = u_n + k\sum_j a_{ij} F_j, \qquad u_{n+1} = u_n + k\sum_i b_i F_i .
$$

`pinn-rk` imposes both directly on the network (`residual="rk"`, the default), dividing by $k$ so they carry the units of a time derivative:

$$
r_i = \frac{U_i - u_n}{k} - \sum_j a_{ij} F_j , \qquad r_{\text{step}} = \frac{u_{n+1} - u_n}{k} - \sum_i b_i F_i .
$$

This uses the **full** Butcher tableau — including the coupling matrix $A$ — so the scheme inherits the tableau's own accuracy. Feeding the manufactured solution $u = \sin(\pi x)e^{-\pi^2 t}$ through the residual leaves only local truncation error:

| tableau | stages | stage order | update order | classical $p$ |
| --- | --- | --- | --- | --- |
| `lobatto2` | 2 | $1.91$ | $1.91$ | 2 |
| `radau2` | 2 | $1.92$ | $2.91$ | 3 |
| `gauss2` | 2 | $1.94$ | $3.91$ | 4 |
| `lobatto3` | 3 | $2.92$ | $3.91$ | 4 |
| `radau3` | 3 | $2.92$ | $4.91$ | 5 |
| `gauss3` | 3 | $2.91$ | $5.91$ | 6 |
| `lobatto4` | 4 | $3.88$ | $5.85$ | 6 |
| `radau4` | 4 | $3.84$ | $6.84$ | 7 |
| `gauss4` | 4 | $3.86$ | — | 8 |

Gauss q=4 is order 8, which is past what double precision can measure here. The update residual is $(u_{n+1}-u_n)/k$, so rounding in $u$ contributes about $\epsilon/k$ — roughly $2.6\times10^{-14}$ at $k=1/120$ — and the true residual falls beneath it. On coarse slabs, where it is still visible, the observed rate is $\approx 7.7$; refining further makes the measured value stall rather than fall, which `tests/test_rk_order.py` asserts explicitly, so the flat region is not mistaken for a convergence failure.

The update residual recovers each method's **classical order**, which is what makes the choice of tableau meaningful: Gauss and Radau cost the same stages, and Gauss is orders of magnitude more consistent at the same slab size.

The stage residual converges at the **stage order**, which for collocation methods equals the number of stages $q$. Since the objective sums both, **the stage order sets the ceiling** — which is why the stage count matters more than the classical order alone suggests: each added stage lifts that ceiling by one power of $k$, from $\mathcal{O}(k^2)$ at $q=2$ to $\mathcal{O}(k^4)$ at $q=4$. Orders are pinned by `tests/test_rk_order.py`, and the tableaux themselves are re-derived from their nodes and checked against the order conditions in `tests/test_tableau_order.py`.

Lobatto IIIA is *stiffly accurate* — its $b$ equals the last row of $A$ — so its update and final stage residuals coincide exactly.

### The interpolant residual (`residual="interpolant"`)

The pre-0.3 formulation is retained for comparison. It reconstructs $\hat{u}$ as a polynomial in time through the stage values, differentiates it analytically via the barycentric differentiation matrix $D_{ij} = L_j'(t_i)$, and imposes $u_t + \mathcal{L}u - f$. It uses only $c$ and $b$, **ignoring $A$**, so every tableau behaves identically and accuracy follows the degree of $\hat{u}$ rather than the order of the method:

| stencil (`q_aux`) | $\deg\hat{u}$ | max residual at $k=5\times10^{-3}$ | observed order |
| --- | --- | --- | --- |
| `"same"` — stage nodes only | $q-1$ | $1.6\times10^{-1}$ | $\mathcal{O}(k)$ |
| `"extend"` — plus slab start | $q$ | $2.6\times10^{-3}$ | $\mathcal{O}(k^2)$ |

For Gauss the RK form is roughly $10^5$ times more consistent than this at the same slab size. `q_aux` applies only to this setting and is ignored under `residual="rk"`.

> **What consistency does and does not tell you.** The tables above measure *truncation error* — the residual left on the exact solution — which bounds the best a perfectly trained network could do. It does not predict optimisation behaviour. In short single-seed training runs on the heat-equation example the two forms trade places depending on the tableau and the step budget, and the loss trace is non-monotonic because the spatial sampler redraws each step. Treat the training comparison as unresolved: a fair answer needs several seeds and converged runs. `residual` exists so the comparison can be made rather than assumed.

### Balancing the two loss terms

The objective sums the PDE residual and the initial-condition penalty. Their relative size is **not** stable: on the shipped example the penalty is essentially the entire loss at initialisation ($5.1$ against a residual of $1.7\times10^{-4}$) and roughly a fifth of it after a few hundred steps. Training therefore starts by fitting the initial condition almost exclusively. `RkPinnConfig.ic_weight` scales the penalty so this can be controlled; `ic_weight=0.0` drops it entirely and measures the residual alone.

---

## Package layout

```plaintext
src/pinn_rk/
├─ tableau.py       # ButcherTableau + Gauss/Radau/Lobatto factories
├─ mesh.py          # TimeMesh: partition of [0,T] into slabs
├─ interpolants.py  # barycentric weights, Lagrange evaluation
├─ model.py         # MLP with boundary-conditioned ansatz
├─ operators.py     # elliptic operators (e.g., Laplacian1D)
├─ config.py        # RkPinnConfig
├─ loss.py          # RkPinnLoss: the time-discrete RK objective
├─ examples/        # reference training routines (heat equation)
├─ __init__.py      # public API
└─ __about__.py     # version
```

---

## API reference (essentials)

### `ButcherTableau`

**Purpose.** Encodes a Runge–Kutta method.

* **Fields:** `A: Tensor [q,q]`, `b: Tensor [q]`, `c: Tensor [q]`.
* **Factories:**

  * `butcher_gauss_legendre_q2()` – order 4, 2 stages
  * `butcher_radau_iia_q2()` – order 3, 2 stages
  * `butcher_lobatto_iiia_q2()` – trapezoidal rule, 2 stages

### `TimeMesh`

**Purpose.** Uniform or user‑defined partition of $[0,T]$.

* `TimeMesh.uniform(T: float, N: int, device) -> TimeMesh`
* Fields: `nodes: Tensor [N+1]`, `steps: Tensor [N]`.

### `MLP`

**Purpose.** Network $g_\theta(x,t)$ used inside the boundary‑conditioned ansatz
$u_\theta(x,t) = \Phi(x) \, g_\theta(x,t)$ with $\Phi(x) = \prod_i 4x_i(1-x_i)$ on $[0,1]^d$.

The factor of 4 per dimension normalises the peak to 1. Without it $\Phi$ peaks at $4^{-d}$, so the network must grow like $4^d$ to represent an $\mathcal{O}(1)$ solution and the gradients reaching it are attenuated by the same factor. On the 2D heat equation the unnormalised form plateaus near 55% relative error where this one reaches 0.4%.

* `MLP(in_dim=2, width=128, depth=4, activation="tanh")`

### `RkPinnConfig`

**Purpose.** Configuration for assembling the time‑discrete loss.

* Key fields: `tableau`, `time_mesh`, `n_x_train`, `spatial_sampler`, `init_data`.
* `residual`: `"rk"` (default) imposes the stage and update equations of the full Butcher tableau; `"interpolant"` selects the pre‑0.3 reconstruction‑derivative form.
* `q_aux`: `"same"` or `"extend"` — reconstruction stencil, used only when `residual="interpolant"`.

### `RkPinnLoss`

**Purpose.** Computes the RK‑PINN time‑discrete objective over all slabs.

* `RkPinnLoss(model, Lop, f_rhs, cfg)`
* Call to compute scalar loss: `loss = loss_fn()`

### Interpolants

**Purpose.** Barycentric Lagrange machinery backing the time reconstruction $\hat{u}$.

* `barycentric_weights(nodes) -> Tensor [q]`
* `lagrange_eval(t, nodes, w) -> Tensor [..., q]` – basis values $L_j(t)$, for evaluating $\hat{u}$ away from the nodes.
* `differentiation_matrix(nodes, w=None) -> Tensor [q,q]` – $D_{ij} = L_j'(t_i)$, exact for polynomials of degree $\le q-1$.

$\partial_t\hat{u}$ at the stage nodes is `D @ U`, taken analytically from the interpolant. The reconstruction stencil is selected by `RkPinnConfig.q_aux`: `"same"` interpolates the $q$ stage values ($\hat{u}$ of degree $q-1$), `"extend"` also uses the slab start $t_n$ (degree $q$, one extra network evaluation per slab).

### Utilities

* `train_heat_equation(...) -> nn.Module` – reference training routine.
* `l2_error(model, T, nx=1001, device) -> float` – $L^2$ error at final time.

---

## Choosing the RK scheme

Tableaux with the same stage count cost the same per slab, so within a row of this table accuracy is close to free. Under `residual="rk"` the trade‑off is the classical one between order and stability:

| scheme | stages | classical order | stability | notes |
| --- | --- | --- | --- | --- |
| `lobatto2` (Lobatto IIIA) | 2 | 2 | A‑stable | trapezoidal rule; symmetric, stiffly accurate |
| `radau2` (Radau IIA) | 2 | **3** | **L‑stable** | damps stiff transients |
| `gauss2` (Gauss–Legendre) | 2 | **4** | A‑stable | most accurate per stage; symplectic |
| `lobatto3` | 3 | 4 | A‑stable | Simpson's rule; stiffly accurate |
| `radau3` | 3 | **5** | **L‑stable** | robust default when stiff |
| `gauss3` | 3 | **6** | A‑stable | most accurate per stage; symplectic |
| `lobatto4` | 4 | 6 | A‑stable | stiffly accurate |
| `radau4` | 4 | **7** | **L‑stable** | highest order with L‑stability |
| `gauss4` | 4 | **8** | A‑stable | highest order shipped; symplectic |

Prefer Gauss for accuracy on smooth problems and Radau IIA when the operator is stiff — A‑stability alone does not damp the stiffest modes, which is why Radau IIA remains the robust choice despite the lower order.

Because the **stage order** equals the stage count and caps the objective, moving from `q=2` to `q=3` raises the ceiling for every family, not just the classical order.

Switch via the `method` argument in `train_heat_equation`.

---

## Extending the library

1. **Add RK variants.** Pass the nodes to `collocation_tableau(c)` and it derives `A` and `b` for you — for a collocation method they are forced by the nodes. Gauss, Radau IIA and Lobatto IIIA ship at q=2, 3 and 4. No other code changes required.
2. **New PDE operators.** Create a class implementing the `EllipticOperator` protocol and supply it to `RkPinnLoss`.
3. **Initial/boundary data.** Replace `init_data` and/or change the boundary factor $\Phi$ for different domains/BCs.
4. **Right‑hand side.** Provide a custom `f_rhs(x,t)` callable.

---

## Reproducibility & testing

* Unit tests live in `tests/` and cover training sanity and error bounds.
* Set environment variables as needed for deterministic PyTorch runs (note some CUDA ops are non‑deterministic).
* CI runs `ruff`, `mypy`, and `pytest` on Linux, macOS, and Windows across multiple Python versions.

---

## Examples and notebooks

Runnable scripts live in [`examples/`](./examples), and two notebooks in
[`examples/notebooks/`](./examples/notebooks) are committed **with their outputs**, so they
render on GitHub without being run:

* [`visualization.ipynb`](./examples/notebooks/visualization.ipynb) — training history, solution evolution against the exact solution, a space‑time error map, an interactive time slider, and `residual="rk"` vs `"interpolant"` on one seed.
* [`benchmarking.ipynb`](./examples/notebooks/benchmarking.ipynb) — cost per step, scaling in `N` and `n_x_train`, memory, CPU/GPU, and the accuracy‑per‑unit‑cost study reproducing the convergence orders above.

```bash
poetry install --with examples
poetry run jupyter lab examples/notebooks
```

Every figure and number in them was produced by executing the notebook against the
current code. `tests/test_notebooks.py` asserts they parse, store no error outputs, and
were genuinely executed rather than authored by hand.

---

## Benchmarks (indicative)

Training the heat‑equation example for ~1k–5k steps typically reaches $L^2$ errors between `1e-2` and `1e-1` at `T=0.1`, depending on the RK scheme and batch sizes. Use `radau2` for stability and increase `N` and training `steps` for tighter accuracy.

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

Key points:

- Follow code style (ruff, mypy)
- Add tests for new features
- Update documentation and CHANGELOG
- Use conventional commit messages

---

## Citation

If you use this software in your research, please cite it:

```bibtex
@software{ribeiro_pinn_rk,
  author    = {Ribeiro, Diogo},
  title     = {pinn-rk: Runge-Kutta Physics-Informed Neural Networks},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21839391},
  url       = {https://doi.org/10.5281/zenodo.21839391},
  version   = {0.2.0}
}
```

The DOI above is the **concept DOI**: it always resolves to the latest version. To cite a specific release, use its own DOI instead — `10.5281/zenodo.21871378` for v0.5.1, `10.5281/zenodo.21865023` for v0.5.0, `10.5281/zenodo.21860298` for v0.4.0, `10.5281/zenodo.21850047` for v0.3.0, `10.5281/zenodo.21843920` for v0.2.0, `10.5281/zenodo.21839392` for v0.1.0.

Or see [CITATION.cff](./CITATION.cff) for the full citation information.

---

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

---

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for planned features including:

- Higher-order RK methods (q≥3)
- 2D/3D operators
- Adaptive time stepping
- Additional boundary conditions
- Documentation website

---

## Acknowledgments

This work builds upon research in Physics-Informed Neural Networks and time-stepping methods for PDEs.

---

## Support

- 📖 [Documentation](https://github.com/DiogoRibeiro7/pinn-rk)
- 🐛 [Issue Tracker](https://github.com/DiogoRibeiro7/pinn-rk/issues)
- 💬 [Discussions](https://github.com/DiogoRibeiro7/pinn-rk/discussions)
- 📧 Contact: [dfr@esmad.ipp.pt](mailto:dfr@esmad.ipp.pt)
