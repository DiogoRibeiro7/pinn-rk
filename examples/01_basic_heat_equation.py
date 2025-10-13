"""
Example 1: Basic Heat Equation

This example demonstrates the simplest use case of pinn-rk:
- Solving the 1D heat equation with homogeneous Dirichlet BCs
- Using the Radau IIA method
- Training and evaluating the model
- Computing L2 error against exact solution

PDE: u_t - u_xx = 0, x in (0,1), t in (0, T]
BC:  u(0,t) = u(1,t) = 0
IC:  u(x,0) = sin(π x)

Exact solution: u(x,t) = sin(π x) exp(-π² t)
"""

from __future__ import annotations

import argparse
import math
import torch
from torch import nn, Tensor

from pinn_rk import (
    RkPinnConfig,
    TimeMesh,
    MLP,
    Laplacian1D,
    RkPinnLoss,
    butcher_radau_iia_q2,
)


def exact_solution(x: Tensor, t: Tensor) -> Tensor:
    """Exact solution for validation."""
    return torch.sin(math.pi * x) * torch.exp(-(math.pi**2) * t)


def source_term(x: Tensor, t: Tensor) -> Tensor:
    """Source term f(x,t) = 0 for homogeneous heat equation."""
    return torch.zeros_like(x)


def make_initial_data(n_points: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Create initial condition data for H^1 penalty."""
    x0 = torch.linspace(1e-6, 1 - 1e-6, n_points, device=device, dtype=torch.float64)
    x0 = x0.unsqueeze(1)
    t0 = torch.zeros_like(x0)
    u0 = exact_solution(x0, t0)
    return x0, u0


def compute_l2_error(
    model: nn.Module, T: float, n_points: int, device: torch.device
) -> float:
    """Compute L2 error at final time."""
    x = torch.linspace(0, 1, n_points, device=device, dtype=torch.float64)
    x = x.unsqueeze(1)
    t = torch.full_like(x, T)
    
    with torch.no_grad():
        u_pred = model(x, t)
        u_exact = exact_solution(x, t)
        error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2))
    
    return float(error.item())


def train_model(
    T: float = 0.1,
    N: int = 20,
    n_x_train: int = 256,
    steps: int = 1000,
    lr: float = 2e-3,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> nn.Module:
    """Train the PINN model."""
    # Set default dtype
    torch.set_default_dtype(torch.float64)
    
    # Setup RK method and time mesh
    tableau = butcher_radau_iia_q2(device)
    mesh = TimeMesh.uniform(T=T, N=N, device=device)
    
    # Create model and operator
    model = MLP(in_dim=2, width=128, depth=4, activation="tanh").to(device)
    operator = Laplacian1D()
    
    # Initial condition for H^1 penalty
    x0, u0 = make_initial_data(n_points=128, device=device)
    
    # Configure loss
    config = RkPinnConfig(
        tableau=tableau,
        time_mesh=mesh,
        n_x_train=n_x_train,
        device=device,
        dtype=torch.float64,
        init_data=(x0, u0),
    )
    
    loss_fn = RkPinnLoss(model=model, Lop=operator, f_rhs=source_term, cfg=config)
    loss_fn = loss_fn.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    if verbose:
        print(f"Training on {device}...")
        print(f"Configuration: T={T}, N={N}, n_x_train={n_x_train}, steps={steps}")
        print("-" * 60)
    
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if verbose and step % 100 == 0:
            print(f"Step {step:5d}: loss = {loss.item():.6e}")
    
    if verbose:
        print("-" * 60)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train PINN on 1D heat equation")
    parser.add_argument("--T", type=float, default=0.1, help="Final time")
    parser.add_argument("--N", type=int, default=20, help="Number of time steps")
    parser.add_argument("--n-x-train", type=int, default=256, help="Spatial batch size")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--save-model", type=str, default=None, help="Path to save model")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Train model
    model = train_model(
        T=args.T,
        N=args.N,
        n_x_train=args.n_x_train,
        steps=args.steps,
        lr=args.lr,
        device=device,
        verbose=not args.quiet,
    )
    
    # Evaluate
    error = compute_l2_error(model, T=args.T, n_points=1001, device=device)
    
    if not args.quiet:
        print(f"\nResults:")
        print(f"L2 error at T={args.T}: {error:.3e}")
        
        # Classification of result quality
        if error < 0.01:
            quality = "Excellent"
        elif error < 0.05:
            quality = "Good"
        elif error < 0.1:
            quality = "Acceptable"
        else:
            quality = "Poor (consider more training steps)"
        print(f"Quality: {quality}")
    
    # Save model if requested
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        if not args.quiet:
            print(f"\nModel saved to: {args.save_model}")
    
    return model, error


if __name__ == "__main__":
    main()
