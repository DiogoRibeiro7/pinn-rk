---
>-
  Compare Gauss, Radau, and Lobatto methods on the same problem.

  ```python python examples/02_different_rk_methods.py ```

  **What it demonstrates:** - Training with different RK schemes - Convergence
  comparison - Stability analysis - Performance benchmarking

  **Expected runtime:** ~3-5 minutes on CPU
---

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

### 3\. Custom Operator (`03_custom_operator.py`)

Implement a custom PDE operator for a reaction-diffusion equation.

```python
python examples/03_custom_operator.py
```

**What it demonstrates:**

- Implementing `EllipticOperator` protocol
- Adding reaction terms
- Custom right-hand side functions
- Operator validation

**Expected runtime:** ~2-3 minutes on CPU

--------------------------------------------------------------------------------

### 4\. Convergence Study (`04_convergence_study.py`)

Systematic study of convergence with respect to time steps and training.

```python
python examples/04_convergence_study.py --save-plots
```

**What it demonstrates:**

- Sweeping over N (number of time steps)
- Measuring convergence rates
- Generating publication-ready plots
- Error analysis

**Expected runtime:** ~10-15 minutes on CPU

--------------------------------------------------------------------------------

## Jupyter Notebooks

### Visualization Notebook (`notebooks/visualization.ipynb`)

Interactive notebook for visualizing PINN solutions.

**Contents:**

- Solution evolution over time
- Error distribution plots
- Training loss curves
- Interactive parameter exploration

**Requirements:**

```bash
pip install jupyter matplotlib plotly
```

--------------------------------------------------------------------------------

### Benchmarking Notebook (`notebooks/benchmarking.ipynb`)

Comprehensive benchmarking across different configurations.

**Contents:**

- Performance profiling
- Memory usage analysis
- GPU vs CPU comparison
- Scaling studies

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
# Save plots
python examples/04_convergence_study.py --save-plots --output-dir ./results

# Save trained model
python examples/01_basic_heat_equation.py --save-model ./model.pth

# Export data
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

### Heat Equation (Radau IIA, N=20, 1000 steps)

```plaintext
[  500] loss = 2.451e-03
[ 1000] loss = 8.327e-04
L2 error at T=0.1: 3.42e-02
```

### Convergence Rates

N (steps) | L2 Error | Rate
--------- | -------- | ----
5         | 1.2e-01  | -
10        | 6.4e-02  | 0.91
20        | 3.4e-02  | 0.92
40        | 1.8e-02  | 0.92

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

Last updated: October 2025
