# Installation

## Requirements

- Python 3.10–3.12
- PyTorch 2.x — install a CPU or CUDA build appropriate for your machine

`pinn-rk` runs in `float64` throughout. The residual divides differences of network
outputs by the slab size `k`, which amplifies rounding error by `1/k`; in `float32` that
noise can swamp the quantity being minimised.

## From source

```bash
git clone https://github.com/DiogoRibeiro7/pinn-rk.git
cd pinn-rk
poetry install
```

That installs the runtime dependencies (`numpy`, `torch`) and the development group
(`pytest`, `ruff`, `mypy`, `bandit`, `pre-commit`).

## Optional dependency groups

Both are optional and off by default, so a plain install stays small.

```bash
poetry install --with examples   # matplotlib, jupyter, plotly -> notebooks and plots
poetry install --with docs       # mkdocs and friends -> this site
```

`--with examples` is what you need for `examples/notebooks/` and for the `--save-plots`
flag on the convergence study.

## Verifying the install

```bash
poetry run pytest                      # full suite
poetry run python -c "import pinn_rk; print(pinn_rk.__version__)"
```

A quick end-to-end check that exercises the actual numerics rather than just the import:

```bash
poetry run python examples/04_convergence_study.py --methods radau3
```

This runs in seconds and prints measured convergence rates. If the fitted orders come out
near the theoretical values, the installation is sound.

## Development setup

```bash
poetry install
poetry run pre-commit install
```

Before opening a pull request:

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src
poetry run pytest
```

CI runs exactly these on Linux, macOS and Windows across Python 3.10–3.12.

## A note on Windows

Installing PyTorch into a deeply nested virtualenv path can fail with `WinError 206`
(“filename or extension is too long”) while unpacking its bundled third-party licence
tree. This is a `MAX_PATH` limitation rather than a packaging fault. Either enable long
paths in Windows, or place the virtualenv nearer the drive root:

```bash
poetry config virtualenvs.in-project true
```
