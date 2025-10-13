# pinn-rk

Runge–Kutta Physics-Informed Neural Networks (PINNs) with **time-discrete losses** (Gauss/Radau/Lobatto) in PyTorch.

## Features
- General RK via Butcher tableau (q=2 Gauss, Radau IIA, Lobatto IIIA included)
- Time-discrete residual with collocation projection
- Dirichlet BC embedding via boundary factor
- Example: 1D heat equation with known solution
- Type hints, mypy, ruff, tests, CI

## Install (dev)
```bash
poetry install
pre-commit install
