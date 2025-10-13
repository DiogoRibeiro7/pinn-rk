# pinn-rk

Runge–Kutta Physics‑Informed Neural Networks (PINNs) with **time‑discrete losses** in PyTorch. Supports Gauss, Radau IIA, and Lobatto IIIA Runge–Kutta schemes via Butcher tableaux, with boundary-conditioned neural ansatz and an end‑to‑end example for the 1D heat equation.

<p align="left">
  <a href="#"><img alt="CI" src="https://img.shields.io/badge/CI-GitHub_Actions-blue"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-3776AB"></a>
  <a href="#"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

---

## Key features

* **Time‑discrete residual** built from Runge–Kutta collocation: residuals evaluated at stage nodes and integrated with RK weights.
* **General RK backend** via `ButcherTableau` (Gauss/Radau/Lobatto included; easily extensible).
* **Boundary conditioning** through a multiplicative factor (\Phi(x)) to satisfy homogeneous Dirichlet BCs exactly.
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

Train on the 1D heat equation (u_t - u_{xx} = 0) on ((0,1)) with homogeneous Dirichlet BCs and initial condition (u_0(x) = \sin(\pi x)):

```python
from pinn_rk import train_heat_equation, l2_error
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

[ u_t + \mathcal{L} u = f \quad \text{in } \Omega\times(0,T], \qquad u=0 \text{ on } \partial\Omega, \qquad u(\cdot,0)=u_0. ]

The time interval is partitioned into slabs (J_n=[t_n,t_{n+1}]) with step (k_n). For a (q)-stage RK method with nodes (c_i), we form **stage times** (t_{n,i}=t_n + c_i k_n) and evaluate the network (u_\theta(x,t)) and operator (\mathcal{L} u_\theta) at these times. The **discrete RK residual** is accumulated using the quadrature weights (b_i):

[ \int_{J_n} | r(t) |^2 dt ;\approx; k_n \sum_{i=1}^q b_i, | r(t_{n,i}) |^2,\quad r := \partial_t \hat u + \Pi_{q-1}(\mathcal{L}\hat u) - \tilde\Pi_{q-1} f. ]

Here, (\hat u) is a polynomial time interpolant on each slab and (\Pi_{q-1}, \tilde\Pi_{q-1}) are degree (q-1) projections realized at collocation nodes.

---

## Package layout

```
src/pinn_rk/
├─ rk_pinn.py       # core loss, RK tableaux, model, training utilities
├─ operators.py     # elliptic operators (e.g., Laplacian1D)
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

**Purpose.** Uniform or user‑defined partition of ([0,T]).

* `TimeMesh.uniform(T: float, N: int, device) -> TimeMesh`
* Fields: `nodes: Tensor [N+1]`, `steps: Tensor [N]`.

### `MLP`

**Purpose.** Network (g_\theta(x,t)) used inside the boundary‑conditioned ansatz
(u_\theta(x,t) = \Phi(x), g_\theta(x,t)) with (\Phi(x)=x(1-x)).

* `MLP(in_dim=2, width=128, depth=4, activation="tanh")`

### `RkPinnConfig`

**Purpose.** Configuration for assembling the time‑discrete loss.

* Key fields: `tableau`, `time_mesh`, `n_x_train`, `spatial_sampler`, `init_data`.

### `RkPinnLoss`

**Purpose.** Computes the RK‑PINN time‑discrete objective over all slabs.

* `RkPinnLoss(model, Lop, f_rhs, cfg)`
* Call to compute scalar loss: `loss = loss_fn()`

### Utilities

* `train_heat_equation(...) -> nn.Module` – reference training routine.
* `l2_error(model, T, nx=1001, device) -> float` – (L^2) error at final time.

---

## Choosing the RK scheme

* **`gauss2`** (Gauss–Legendre): higher order, A‑stable; good accuracy per stage.
* **`radau2`** (Radau IIA): L‑stable; robust for stiff operators (recommended default).
* **`lobatto2`** (Lobatto IIIA): trapezoidal rule; symmetric, A‑stable.

Switch via the `method` argument in `train_heat_equation`.

---

## Extending the library

1. **Add RK variants.** Implement additional `ButcherTableau` factories (e.g., Gauss q=3, Radau IIA q=3). No other code changes required.
2. **New PDE operators.** Create a class implementing the `EllipticOperator` protocol and supply it to `RkPinnLoss`.
3. **Initial/boundary data.** Replace `init_data` and/or change the boundary factor (\Phi) for different domains/BCs.
4. **Right‑hand side.** Provide a custom `f_rhs(x,t)` callable.

---

## Reproducibility & testing

* Unit tests live in `tests/` and cover training sanity and error bounds.
* Set environment variables as needed for deterministic PyTorch runs (note some CUDA ops are non‑deterministic).
* CI runs `ruff`, `mypy`, and `pytest` on Linux, macOS, and Windows across multiple Python versions.

---

## Benchmarks (indicative)

Training the heat‑equation example for ~1k–5k steps typically reaches (L^2) errors between `1e-2` and `1e-1` at `T=0.1`, depending on the RK scheme and batch sizes. Use `radau2` for stability and increase `N` and training `steps` for tighter accuracy.

---

## Roadmap

* [ ] Higher‑order Gauss/Radau/Lobatto (q≥3)
* [ ] Analytic time‑derivative of the interpolant (replace finite differences)
* [ ] 2D/3D operators and manufactured‑solution suites
* [ ] Optional cG/dG time discretizations under the same pointwise form
* [ ] Example notebooks and visualization utilities

---

## Contributing

Contributions are welcome! Please open an issue or a pull request. The project follows Conventional Commits and includes ruff/mypy/pytest checks via pre‑commit and CI.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
