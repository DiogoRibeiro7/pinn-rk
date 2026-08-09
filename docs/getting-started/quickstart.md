# Quick Start

## The shortest path

Train on the 1D heat equation and measure the error against the exact solution:

```python
import torch
from pinn_rk.examples.train_heat_equation import train_heat_equation, l2_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = train_heat_equation(
    method="radau3",   # gauss2 | radau2 | lobatto2 | gauss3 | radau3 | lobatto3
    T=0.1,
    N=20,
    n_x_train=256,
    steps=1000,
    lr=2e-3,
    device=device,
)
print("L2(T=0.1) =", l2_error(model, T=0.1, nx=1001, device=device))
```

## Assembling it yourself

The helper above hides four objects. Building them directly is what you would do for a
new PDE:

```python
import math
import torch
from pinn_rk import (
    MLP, Laplacian1D, RkPinnConfig, RkPinnLoss, TimeMesh, butcher_radau_iia_q3,
)

torch.set_default_dtype(torch.float64)
device = torch.device("cpu")

def exact(x, t):
    return torch.sin(math.pi * x) * torch.exp(-(math.pi ** 2) * t)

def rhs(x, t):
    return torch.zeros_like(x)

# Initial data on a sorted grid: the H1 penalty differentiates u0 along it.
x0 = torch.linspace(1e-6, 1 - 1e-6, 128, device=device).unsqueeze(1)

cfg = RkPinnConfig(
    tableau=butcher_radau_iia_q3(device),
    time_mesh=TimeMesh.uniform(T=0.1, N=8, device=device),
    residual="rk",
    n_x_train=96,
    device=device,
    init_data=(x0, exact(x0, torch.zeros_like(x0))),
)

model = MLP(width=96, depth=4, dtype=torch.float64).to(device)
loss_fn = RkPinnLoss(model=model, Lop=Laplacian1D(), f_rhs=rhs, cfg=cfg).to(device)

opt = torch.optim.Adam(model.parameters(), lr=5e-3)
for step in range(1, 801):
    opt.zero_grad(set_to_none=True)
    loss = loss_fn()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
```

Boundary conditions need no term in the loss. `MLP` multiplies its output by
`Phi(x) = x(1-x)`, so homogeneous Dirichlet conditions hold exactly by construction and
boundary error never has to be traded against the residual.

## Choosing a tableau

All tableaux with the same stage count cost the same per slab, so accuracy is close to
free. Under `residual="rk"` the classical order is what you actually get:

| method | stages | classical order | stability |
| --- | --- | --- | --- |
| `lobatto2` | 2 | 2 | A-stable, stiffly accurate |
| `radau2` | 2 | 3 | **L-stable** |
| `gauss2` | 2 | 4 | A-stable, symplectic |
| `lobatto3` | 3 | 4 | A-stable, stiffly accurate |
| `radau3` | 3 | 5 | **L-stable** |
| `gauss3` | 3 | 6 | A-stable, symplectic |

Prefer Gauss for accuracy on smooth problems and Radau IIA when the operator is stiff:
A-stability alone does not damp the stiffest modes, which is why Radau remains the robust
default despite the lower order.

!!! note "Stage order is the real ceiling"
    The stage residual converges at the **stage order**, which equals the number of
    stages, and it dominates the objective. So `q=3` improves matters more than the
    classical order alone suggests: it lifts the ceiling from `O(k^2)` to `O(k^3)`.

## Verifying, not guessing

Because the exact solution makes the residual vanish, feeding it in isolates truncation
error. That is how the shipped orders are measured rather than asserted:

```bash
python examples/04_convergence_study.py --methods radau2 radau3
```

```text
=== radau3  (q=3, classical order p=5) ===
   N         k    stage res   rate   update res   rate
   3   0.03333    3.414e-04   -      4.463e-07   -
   5   0.02000    7.830e-05   2.88    3.711e-08   4.87
   8   0.01250    1.977e-05   2.93    3.676e-09   4.92
  12   0.00833    5.970e-06   2.95    4.943e-10   4.95
```

!!! warning "Consistency is not the same as trained accuracy"
    These rates bound what a *perfectly trained* network could reach. They do not predict
    optimisation: trained `L2` error is stochastic, since the spatial sampler redraws
    every step, and is limited by the optimiser rather than the discretisation alone.
    Pass `--train` to see both side by side.

## Next steps

- [`examples/notebooks/visualization.ipynb`](https://github.com/DiogoRibeiro7/pinn-rk/blob/main/examples/notebooks/visualization.ipynb) — what the trained solution looks like and where the error concentrates
- [`examples/notebooks/benchmarking.ipynb`](https://github.com/DiogoRibeiro7/pinn-rk/blob/main/examples/notebooks/benchmarking.ipynb) — cost, scaling, and accuracy per unit of work
- [`examples/03_custom_operator.py`](https://github.com/DiogoRibeiro7/pinn-rk/blob/main/examples/03_custom_operator.py) — implementing `EllipticOperator` for a new PDE
- [Loss functions](../api/loss.md) — the residual forms and `ic_weight`
