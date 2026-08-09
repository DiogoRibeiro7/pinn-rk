# Examples

This directory contains practical examples demonstrating how to use pinn-rk for solving PDEs.

## Quick Start Examples

### 1\. Basic Heat Equation (`01_basic_heat_equation.py`)

The simplest example - solve the 1D heat equation with default settings.

```python
python examples/01_basic_heat_equation.py
```

**What it demonstrates:**

- Setting up a basic PINN problem
- Training with Radau IIA method
- Computing L2 error against exact solution
- Basic visualization

**Expected runtime:** ~1-2 minutes on CPU

### 2\. Comparing RK schemes (`02_different_rk_methods.py`)

Compare Gauss, Radau IIA and Lobatto IIIA on the same problem.

```bash
python examples/02_different_rk_methods.py
```

**What it demonstrates:**

- Training with different RK schemes
- Accuracy comparison at identical cost — all three are 2-stage
- Timing per scheme

**Expected runtime:** ~3-5 minutes on CPU

--------------------------------------------------------------------------------

### 3\. Custom Operator (`03_custom_operator.py`)

Implement a custom PDE operator for a reaction-diffusion equation.

```bash
python examples/03_custom_operator.py
```

**What it demonstrates:**

- Implementing the `EllipticOperator` protocol
- Adding a reaction term: `L u = -D u_xx + k u`
- A **non-zero** right-hand side, derived from a manufactured solution
- Validating the operator against its closed form *before* training on it

That last point is the one worth copying. A miswritten operator produces a PINN that
trains happily to the wrong answer, and that is expensive to diagnose later; comparing
`L u` to its analytic value takes microseconds and catches sign and factor errors
immediately.

**Expected runtime:** ~2-3 minutes on CPU

--------------------------------------------------------------------------------

### 4\. Convergence Study (`04_convergence_study.py`)

Sweep over the number of time slabs and measure observed convergence rates.

```bash
python examples/04_convergence_study.py                        # consistency only, seconds
python examples/04_convergence_study.py --train                # adds training, minutes
python examples/04_convergence_study.py --save-plots --export-csv results.csv
```

**What it demonstrates:**

- Sweeping over `N` and fitting log-log rates
- Keeping **consistency error** and **trained L2 error** apart — they measure different
  things and need not agree
- Plot generation and CSV export

Consistency error is deterministic and isolates the time discretisation; trained error is
stochastic and optimisation-limited. Reporting only the first would overstate the method;
reporting only the second would hide where the error comes from.

**Expected runtime:** seconds without `--train`, ~10-15 minutes with it

--------------------------------------------------------------------------------

## Jupyter Notebooks

Both notebooks are committed **with their outputs**, so they render on GitHub without
being run. Every figure and number in them was produced by executing the notebook
against the current code, not written by hand.

**Requirements:**

```bash
poetry install --with examples     # jupyter, matplotlib, plotly
poetry run jupyter lab examples/notebooks
```

### Visualization (`notebooks/visualization.ipynb`)

Trains an RK-PINN on the heat equation and inspects what it learned.

**Contents:**

- Training loss history, shown with its stochastic noise rather than smoothed
- Solution evolution against the exact solution
- Space-time error heat map, and how the error behaves in time
- Interactive time slider (Plotly — requires running locally; GitHub does not execute JS)
- `residual="rk"` vs `residual="interpolant"` on the same seed

**Expected runtime:** ~1-2 minutes on CPU

--------------------------------------------------------------------------------

### Benchmarking (`notebooks/benchmarking.ipynb`)

What a loss evaluation costs, how it scales, and what accuracy each tableau buys.

**Contents:**

- Cost per step, split into forward and backward
- Scaling in time slabs `N` and spatial batch `n_x_train`
- Autograd-graph memory, and CUDA peak allocation where available
- CPU vs GPU, or a CPU thread-count sweep when no GPU is present
- **Accuracy per unit cost:** all three tableaux are 2-stage, yet recover classical
  orders 4, 3 and 2 — measured, not asserted

**Expected runtime:** ~1-2 minutes on CPU

--------------------------------------------------------------------------------

## Running Examples

### Basic Usage

```bash
# From repository root
python -m examples.01_basic_heat_equation

# Or with Poetry
poetry run python examples/01_basic_heat_equation.py
```

### With GPU

```bash
# Most examples auto-detect CUDA
python examples/01_basic_heat_equation.py

# Force CPU
python examples/01_basic_heat_equation.py --device cpu

# Specific GPU
CUDA_VISIBLE_DEVICES=0 python examples/01_basic_heat_equation.py
```

### Saving Results

```bash
# Save trained model
python examples/01_basic_heat_equation.py --save-model ./model.pth

# Full flag list for any example
python examples/01_basic_heat_equation.py --help

# Convergence plots and CSV export
python examples/04_convergence_study.py --save-plots --output-dir ./results
python examples/04_convergence_study.py --export-csv ./data.csv
```

--------------------------------------------------------------------------------

## Example Structure

Each example follows this structure:

```python
"""
Brief description of what the example demonstrates.
"""

import torch
from pinn_rk import ...

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure problem
    # ...

    # Train
    # ...

    # Evaluate and visualize
    # ...

if __name__ == "__main__":
    main()
```

--------------------------------------------------------------------------------

## Expected Results

### Heat equation (Radau IIA, `T=0.1`, `N=8`, 400 steps)

Ballpark from `notebooks/visualization.ipynb`; the loss is stochastic, so a single value
is not meaningful on its own:

```plaintext
L2 error at t=0    : ~3e-02
L2 error at t=0.10 : ~2e-02
```

Absolute error falls with time because the solution itself decays by
$e^{-\pi^2 T}\approx 0.37$; relative error grows. Expect run-to-run variation — the
spatial sampler redraws every step.

### Convergence of the residual

These are **consistency** orders: the residual left on the exact solution, which
isolates truncation error. Measured by `tests/test_rk_order.py` and reproduced in
`notebooks/benchmarking.ipynb`:

| tableau | stages | stage order | update order | classical order |
| --- | --- | --- | --- | --- |
| `lobatto2` | 2 | 1.91 | 1.91 | 2 |
| `radau2` | 2 | 1.92 | 2.91 | 3 |
| `gauss2` | 2 | 1.94 | 3.91 | 4 |
| `lobatto3` | 3 | 2.92 | 3.91 | 4 |
| `radau3` | 3 | 2.92 | 4.91 | 5 |
| `gauss3` | 3 | 2.91 | 5.91 | 6 |

The **stage order** equals the number of stages and dominates the objective, so it is
what caps accuracy. Moving to `q=3` lifts that ceiling from `O(k^2)` to `O(k^3)`.

This bounds what a perfectly trained network could achieve; it is **not** a trained-model
convergence study. A rate table for trained L2 error against `N` needs several seeds and
converged runs; `04_convergence_study.py --train` produces one, and prints a reminder
that a single seed cannot establish a trend.

--------------------------------------------------------------------------------

## Troubleshooting

### Common Issues

**1\. Loss not decreasing:**

- Reduce learning rate (try `lr=1e-4`)
- Increase training steps
- Check initial condition penalty weight
- Try different RK method (Radau is most stable)

**2\. NaN/Inf errors:**

- Reduce learning rate
- Add gradient clipping (enabled by default)
- Check boundary condition implementation
- Reduce time step size (increase N)

**3\. High L2 error:**

- Increase training steps
- Increase network size (`width=256, depth=6`)
- Increase spatial sampling (`n_x_train=512`)
- Increase temporal resolution (larger N)

**4\. Memory errors:**

- Reduce batch size (`n_x_train`)
- Use gradient checkpointing
- Process time slabs sequentially
- Use mixed precision training

--------------------------------------------------------------------------------

## Creating Custom Examples

Template for new examples:

```python
"""
Example: [Brief description]

This example demonstrates:
- Feature 1
- Feature 2
- Feature 3
"""

import torch
from pinn_rk import (
    ButcherTableau,
    RkPinnConfig,
    TimeMesh,
    MLP,
    RkPinnLoss,
    butcher_radau_iia_q2,
)

# Define your PDE
def exact_solution(x, t):
    """Known solution for validation."""
    pass

def source_term(x, t):
    """Right-hand side f(x,t)."""
    pass

# Implement custom operator if needed
class MyOperator:
    def __call__(self, x, u):
        """Apply operator L to u."""
        pass

    def requires_hessian(self):
        return True  # or False

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration
    T = 1.0
    N = 20
    n_x_train = 256

    # Setup
    bt = butcher_radau_iia_q2(device)
    mesh = TimeMesh.uniform(T=T, N=N, device=device)
    model = MLP().to(device)
    operator = MyOperator()

    cfg = RkPinnConfig(
        tableau=bt,
        time_mesh=mesh,
        n_x_train=n_x_train,
        device=device,
    )

    loss_fn = RkPinnLoss(model, operator, source_term, cfg)

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(1000):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"Step {step+1}: loss = {loss.item():.3e}")

    # Evaluation
    print("Training complete!")

if __name__ == "__main__":
    main()
```

--------------------------------------------------------------------------------

## Contributing Examples

To contribute a new example:

1. **Follow the naming convention:** `XX_descriptive_name.py`
2. **Include docstring** at the top explaining what it demonstrates
3. **Add comments** for complex sections
4. **Keep it focused** on one or two concepts
5. **Test it works** on CPU and GPU
6. **Update this README** with a description
7. **Add expected outputs** and runtime estimates

See [CONTRIBUTING.md](../CONTRIBUTING.md) for more details.

--------------------------------------------------------------------------------

## Additional Resources

- **Documentation:** See main README.md
- **Theory:** See ROADMAP.md for mathematical background
- **API Reference:** See docstrings in source code
- **Issues:** Report problems on GitHub

--------------------------------------------------------------------------------

## Performance Tips

### For Faster Training

1. **Use GPU:** `device = torch.device("cuda")`
2. **Increase batch size:** `n_x_train=512` (if memory allows)
3. **Use compiled mode:** `torch.compile(model)` (PyTorch 2.0+)
4. **Reduce validation frequency:** Check error every N steps

### For Better Accuracy

1. **More time steps:** Increase `N`
2. **Longer training:** More optimization steps
3. **Larger network:** `width=256, depth=6`
4. **Better optimizer:** Try LBFGS for final refinement
5. **Curriculum learning:** Start with coarse mesh, refine gradually

### For Large Problems

1. **Gradient checkpointing:** Trade compute for memory
2. **Mixed precision:** Use `torch.cuda.amp`
3. **Sequential processing:** Process time slabs one at a time
4. **Distributed training:** Multi-GPU setup (future feature)

--------------------------------------------------------------------------------

## Citation

If you use these examples in your research, please cite the pinn-rk library. See [CITATION.cff](../CITATION.cff).

--------------------------------------------------------------------------------

Last updated: August 2026 (pinn-rk v0.3.0)
