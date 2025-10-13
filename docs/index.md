# Welcome to pinn-rk

**Runge-Kutta Physics-Informed Neural Networks in PyTorch**

!!! info "Latest Version" Current version: **0.1.0** | [Release Notes](changelog.md) | [GitHub](https://github.com/diogoribeiro7/pinn-rk)

## What is pinn-rk?

pinn-rk is a PyTorch library for solving time-dependent partial differential equations (PDEs) using Physics-Informed Neural Networks (PINNs) with Runge-Kutta time discretization. Unlike standard PINNs that use continuous time formulations, pinn-rk employs a **time-discrete** approach based on high-order Runge-Kutta methods.

### Key Features

- :material-clock-fast: **Time-discrete formulation** using Runge-Kutta collocation
- :material-function: **Multiple RK schemes**: Gauss-Legendre, Radau IIA, Lobatto IIIA
- :material-vector-polyline: **Exact boundary conditions** via multiplicative neural ansatz
- :material-cube-outline: **Modular design** for easy extension
- :material-code-tags: **Type-safe** with full type hints and mypy checking
- :material-test-tube: **Well-tested** with >85% code coverage

--------------------------------------------------------------------------------

## Quick Example

Solve the 1D heat equation in just a few lines:

```python
from pinn_rk.examples.train_heat_equation import train_heat_equation, l2_error
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train model
model = train_heat_equation(
    method="radau2",  # Radau IIA (L-stable)
    T=0.1,            # Final time
    N=20,             # Time steps
    steps=1000,       # Training iterations
    device=device,
)

# Evaluate
print(f"L2 error: {l2_error(model, T=0.1):.3e}")
```

**Output:**

```
[  500] loss = 2.451e-03
[ 1000] loss = 8.327e-04
L2 error: 3.42e-02
```

--------------------------------------------------------------------------------

## Why Time-Discrete?

Traditional PINNs evaluate the PDE residual continuously in time, which can lead to:

- Difficulty enforcing initial conditions
- Training instability for stiff problems
- Poor temporal resolution

**Time-discrete PINNs** address these issues by:

1. Partitioning time into intervals (slabs)
2. Using high-order RK methods for temporal discretization
3. Enforcing residuals at collocation points
4. Providing L-stability for stiff equations (Radau IIA)

--------------------------------------------------------------------------------

## Installation

### Using Poetry (Recommended)

```bash
git clone https://github.com/diogoribeiro7/pinn-rk.git
cd pinn-rk
poetry install
```

### Using pip

```bash
pip install pinn-rk
```

**Requirements:**

- Python 3.10, 3.11, or 3.12
- PyTorch ≥ 2.3.0
- NumPy ≥ 2.0.0

--------------------------------------------------------------------------------

## Getting Started

### 1\. Choose Your RK Method

```python
from pinn_rk import (
    butcher_gauss_legendre_q2,  # Order 4, A-stable
    butcher_radau_iia_q2,       # Order 3, L-stable ⭐
    butcher_lobatto_iiia_q2,    # Order 2, A-stable
)

# Recommended for most problems
tableau = butcher_radau_iia_q2(device)
```

### 2\. Set Up Time Discretization

```python
from pinn_rk import TimeMesh

# Uniform partition: 0 = t_0 < t_1 < ... < t_N = T
mesh = TimeMesh.uniform(T=0.1, N=20, device=device)
```

### 3\. Define Your Model

```python
from pinn_rk import MLP

# Boundary-conditioned neural network
# Automatically satisfies u(0,t) = u(1,t) = 0
model = MLP(in_dim=2, width=128, depth=4).to(device)
```

### 4\. Configure and Train

```python
from pinn_rk import RkPinnConfig, RkPinnLoss, Laplacian1D

config = RkPinnConfig(
    tableau=tableau,
    time_mesh=mesh,
    n_x_train=256,
    device=device,
)

loss_fn = RkPinnLoss(
    model=model,
    Lop=Laplacian1D(),
    f_rhs=lambda x, t: torch.zeros_like(x),
    cfg=config,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(1000):
    optimizer.zero_grad()
    loss = loss_fn()
    loss.backward()
    optimizer.step()
```

--------------------------------------------------------------------------------

## Use Cases

### Research Applications

- **Parabolic PDEs**: Heat equation, diffusion, reaction-diffusion
- **Stiff problems**: L-stable Radau methods excel here
- **Time-dependent boundary conditions**: Easy to incorporate
- **Inverse problems**: Parameter estimation in PDEs

### Learning Scientific ML

- **Pedagogical tool**: Understand time discretization in PINNs
- **Method comparison**: Compare RK schemes empirically
- **Extensible framework**: Add custom operators and PDEs

--------------------------------------------------------------------------------

## Comparison with Standard PINNs

Feature             | Standard PINN      | Time-Discrete PINN (pinn-rk)
------------------- | ------------------ | ----------------------------
Time treatment      | Continuous         | Discrete (RK slabs)
Initial conditions  | Soft penalty       | Can use H¹ seminorm
Stiff problems      | Challenging        | L-stable methods available
Temporal accuracy   | Depends on network | Controlled by RK order
Boundary conditions | Often soft         | Can be exact (ansatz)

--------------------------------------------------------------------------------

## Examples Gallery

### 1D Heat Equation

```python
from pinn_rk.examples import train_heat_equation

model = train_heat_equation(method="radau2", T=0.1, N=20)
```

### Method Comparison

```python
from pinn_rk.examples import compare_methods

results = compare_methods(
    methods=["gauss2", "radau2", "lobatto2"],
    steps=1000
)
```

### Custom PDE

```python
class MyOperator:
    def __call__(self, x, u):
        # Your custom operator
        pass
```

See [Examples](examples/overview.md) for more.

--------------------------------------------------------------------------------

## Architecture

```
┌─────────────┐
│   Model     │ u(x,t) = φ(x) · g_θ(x,t)
│   (MLP)     │ φ(x) = x(1-x) enforces BCs
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Operator   │ L[u] = -∂²u/∂x²
│ (Laplacian) │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  RK Loss    │ Σ k_n Σ b_i |r(t_{n,i})|²
│  Function   │
└─────────────┘
```

--------------------------------------------------------------------------------

## Performance

Typical results on 1D heat equation (T=0.1, N=20, 1000 steps):

Method         | L² Error | Training Time | Stability
-------------- | -------- | ------------- | ----------
Gauss-Legendre | 2.8e-2   | 45s           | A-stable
Radau IIA      | 3.4e-2   | 48s           | L-stable ⭐
Lobatto IIIA   | 4.1e-2   | 42s           | A-stable

_Tested on NVIDIA RTX 4090_

--------------------------------------------------------------------------------

## Documentation

- [Getting Started](getting-started/installation.md) - Installation and first steps
- [Theory](theory/mathematical-background.md) - Mathematical background
- [User Guide](guide/configuration.md) - Detailed usage
- [API Reference](api/config.md) - Complete API documentation
- [Examples](examples/overview.md) - Practical examples

--------------------------------------------------------------------------------

## Community

- :fontawesome-brands-github: [GitHub Repository](https://github.com/diogoribeiro7/pinn-rk)
- :material-bug: [Issue Tracker](https://github.com/diogoribeiro7/pinn-rk/issues)
- :material-forum: [Discussions](https://github.com/diogoribeiro7/pinn-rk/discussions)
- :material-book: [Contributing Guide](contributing.md)

--------------------------------------------------------------------------------

## Citation

If you use pinn-rk in your research, please cite:

```bibtex
@software{ribeiro2025pinnrk,
  author = {Ribeiro, Diogo},
  title = {pinn-rk: Runge-Kutta Physics-Informed Neural Networks},
  year = {2025},
  url = {https://github.com/diogoribeiro7/pinn-rk},
  version = {0.1.0}
}
```

--------------------------------------------------------------------------------

## License

pinn-rk is released under the [MIT License](https://github.com/diogoribeiro7/pinn-rk/blob/main/LICENSE).

--------------------------------------------------------------------------------

## Roadmap

Current focus areas:

- ✅ Core RK-PINN implementation
- ✅ Multiple RK schemes (q=2)
- ✅ 1D spatial operators
- 🔄 Higher-order methods (q≥3)
- 🔄 2D/3D operators
- 📋 Adaptive time stepping
- 📋 Documentation website

See [ROADMAP.md](roadmap.md) for detailed plans.

--------------------------------------------------------------------------------

## Quick Links

<div class="grid cards" markdown="">

-   :material-clock-fast:{ .lg .middle } <strong>Getting Started</strong>

    ---

    Install pinn-rk and run your first example

    <a href="getting-started/quickstart.md">:octicons-arrow-right-24: Quick Start</a>

-   :material-book-open-variant:{ .lg .middle } <strong>User Guide</strong>

    ---

    Learn how to configure and train models

    <a href="guide/configuration.md">:octicons-arrow-right-24: User Guide</a>

-   :material-code-braces:{ .lg .middle } <strong>API Reference</strong>

    ---

    Detailed documentation of all classes and functions

    <a href="api/config.md">:octicons-arrow-right-24: API Docs</a>

-   :material-flask:{ .lg .middle } <strong>Examples</strong>

    ---

    Practical examples and tutorials

    <a href="examples/overview.md">:octicons-arrow-right-24: Examples</a></div>

--------------------------------------------------------------------------------

!!! tip "Need Help?"

```
- Check the [FAQ](getting-started/faq.md)
- Browse [Examples](examples/overview.md)
- Ask in [Discussions](https://github.com/diogoribeiro7/pinn-rk/discussions)
- Report bugs in [Issues](https://github.com/diogoribeiro7/pinn-rk/issues)
```
