# pinn-rk

<p align="center">
  <a href="https://github.com/DiogoRibeiro7/pinn-rk/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/DiogoRibeiro7/pinn-rk/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="https://codecov.io/gh/DiogoRibeiro7/pinn-rk"><img alt="Coverage" src="https://codecov.io/gh/DiogoRibeiro7/pinn-rk/branch/main/graph/badge.svg"></a>
  <a href="https://pypi.org/project/pinn-rk/"><img alt="PyPI" src="https://img.shields.io/pypi/v/pinn-rk"></a>
  <a href="https://pypi.org/project/pinn-rk/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/pinn-rk"></a>
  <a href="https://github.com/DiogoRibeiro7/pinn-rk/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://doi.org/10.5281/zenodo.21839391"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.21839391.svg"></a>
  <a href="https://github.com/psf/black"><img alt="Code style: ruff" src="https://img.shields.io/badge/code%20style-ruff-000000.svg"></a>
  <a href="https://github.com/DiogoRibeiro7/pinn-rk"><img alt="Downloads" src="https://img.shields.io/pypi/dm/pinn-rk"></a>
</p>

Runge–Kutta Physics‑Informed Neural Networks (PINNs) with **time‑discrete losses** in PyTorch. Supports Gauss, Radau IIA, and Lobatto IIIA Runge–Kutta schemes via Butcher tableaux, with boundary-conditioned neural ansatz and an end‑to‑end example for the 1D heat equation.

> See **[ROADMAP.md](./ROADMAP.md)** for milestones and planned features.

---

## Key features

* **Time‑discrete residual** built from Runge–Kutta collocation: residuals evaluated at stage nodes and integrated with RK weights.
* **General RK backend** via `ButcherTableau` (Gauss/Radau/Lobatto included; easily extensible).
* **Boundary conditioning** through a multiplicative factor $\Phi(x)$ to satisfy homogeneous Dirichlet BCs exactly.
* **Modular PDE operators** (e.g., `Laplacian1D`) with autograd‑based derivatives.
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
    method="radau2",   # "gauss2" | "radau2" | "lobatto2"
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

| tableau | $r_{\text{step}}$ at $k=5\times10^{-3}$ | observed order | classical order $p$ |
| --- | --- | --- | --- |
| `gauss2` | $1.3\times10^{-8}$ | $3.96$ | 4 |
| `radau2` | $5.3\times10^{-6}$ | $2.96$ | 3 |
| `lobatto2` | $2.0\times10^{-3}$ | $1.96$ | 2 |

The stage residual converges at the **stage order** ($2$ for all three q=2 collocation tableaux); the update residual recovers each method's **classical order**. This is what makes the choice of tableau meaningful: Gauss and Radau cost the same two stages, and Gauss is two orders of magnitude more consistent at the same slab size. Orders are pinned by `tests/test_rk_order.py`.

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
$u_\theta(x,t) = \Phi(x) \, g_\theta(x,t)$ with $\Phi(x) = x(1-x)$.

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

All three are 2‑stage, so they cost the same per slab. Under `residual="rk"` they differ in accuracy, and the trade‑off is the classical one between order and stability:

| scheme | classical order | stability | notes |
| --- | --- | --- | --- |
| `gauss2` (Gauss–Legendre) | **4** | A‑stable | most accurate per stage; symplectic |
| `radau2` (Radau IIA) | **3** | **L‑stable** | damps stiff transients; safest default |
| `lobatto2` (Lobatto IIIA) | **2** | A‑stable | trapezoidal rule; symmetric, stiffly accurate |

Prefer `gauss2` for accuracy on smooth problems and `radau2` when the operator is stiff — A‑stability alone does not damp the stiffest modes, which is why Radau IIA remains the robust choice despite the lower order.

Switch via the `method` argument in `train_heat_equation`.

---

## Extending the library

1. **Add RK variants.** Implement additional `ButcherTableau` factories (e.g., Gauss q=3, Radau IIA q=3). No other code changes required.
2. **New PDE operators.** Create a class implementing the `EllipticOperator` protocol and supply it to `RkPinnLoss`.
3. **Initial/boundary data.** Replace `init_data` and/or change the boundary factor $\Phi$ for different domains/BCs.
4. **Right‑hand side.** Provide a custom `f_rhs(x,t)` callable.

---

## Reproducibility & testing

* Unit tests live in `tests/` and cover training sanity and error bounds.
* Set environment variables as needed for deterministic PyTorch runs (note some CUDA ops are non‑deterministic).
* CI runs `ruff`, `mypy`, and `pytest` on Linux, macOS, and Windows across multiple Python versions.

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

The DOI above is the **concept DOI**: it always resolves to the latest version. To cite a specific release, use its own DOI instead — `10.5281/zenodo.21850047` for v0.3.0, `10.5281/zenodo.21843920` for v0.2.0, `10.5281/zenodo.21839392` for v0.1.0.

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
