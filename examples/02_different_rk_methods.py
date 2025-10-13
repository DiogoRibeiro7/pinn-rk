"""
Example 2: Comparing Different Runge-Kutta Methods

This example compares three RK schemes on the same heat equation:
- Gauss-Legendre (order 4, A-stable)
- Radau IIA (order 3, L-stable)
- Lobatto IIIA (order 2, A-stable)

Demonstrates:
- Using different RK tableaux
- Comparing accuracy and convergence
- Performance benchmarking
"""

from __future__ import annotations

import argparse
import math
import time

import torch
from torch import Tensor, nn

from pinn_rk import (
    MLP,
    Laplacian1D,
    RkPinnConfig,
    RkPinnLoss,
    TimeMesh,
    butcher_gauss_legendre_q2,
    butcher_lobatto_iiia_q2,
    butcher_radau_iia_q2,
)


def exact_solution(x: Tensor, t: Tensor) -> Tensor:
    """Exact solution for validation."""
    return torch.sin(math.pi * x) * torch.exp(-(math.pi**2) * t)


def source_term(x: Tensor, t: Tensor) -> Tensor:
    """Source term f(x,t) = 0."""
    return torch.zeros_like(x)


def make_initial_data(n_points: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Create initial condition data."""
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


def train_with_method(
    method_name: str,
    T: float,
    N: int,
    n_x_train: int,
    steps: int,
    lr: float,
    device: torch.device,
) -> tuple[nn.Module, float, float]:
    """Train model with specified RK method and return model, error, and training time."""
    torch.set_default_dtype(torch.float64)
    
    # Select RK tableau
    if method_name == "gauss2":
        tableau = butcher_gauss_legendre_q2(device)
    elif method_name == "radau2":
        tableau = butcher_radau_iia_q2(device)
    elif method_name == "lobatto2":
        tableau = butcher_lobatto_iiia_q2(device)
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    # Setup
    mesh = TimeMesh.uniform(T=T, N=N, device=device)
    model = MLP(in_dim=2, width=128, depth=4, activation="tanh").to(device)
    operator = Laplacian1D()
    x0, u0 = make_initial_data(n_points=128, device=device)
    
    config = RkPinnConfig(
        tableau=tableau,
        time_mesh=mesh,
        n_x_train=n_x_train,
        device=device,
        dtype=torch.float64,
        init_data=(x0, u0),
    )
    
    loss_fn = RkPinnLoss(model=model, Lop=operator, f_rhs=source_term, cfg=config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training with timing
    start_time = time.time()
    
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    training_time = time.time() - start_time
    
    # Evaluate
    error = compute_l2_error(model, T=T, n_points=1001, device=device)
    
    return model, error, training_time


def compare_methods(
    methods: list[str],
    T: float,
    N: int,
    n_x_train: int,
    steps: int,
    lr: float,
    device: torch.device,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Compare multiple RK methods."""
    results = {}
    
    method_names = {
        "gauss2": "Gauss-Legendre q=2 (order 4)",
        "radau2": "Radau IIA q=2 (order 3)",
        "lobatto2": "Lobatto IIIA q=2 (order 2)",
    }
    
    if verbose:
        print("=" * 70)
        print("Comparing RK Methods on Heat Equation")
        print(f"Configuration: T={T}, N={N}, steps={steps}, device={device}")
        print("=" * 70)
        print()
    
    for method in methods:
        if verbose:
            print(f"Training with {method_names[method]}...")
        
        model, error, train_time = train_with_method(
            method, T, N, n_x_train, steps, lr, device
        )
        
        results[method] = {
            "error": error,
            "time": train_time,
            "model": model,
        }
        
        if verbose:
            print(f"  L2 error: {error:.6e}")
            print(f"  Training time: {train_time:.2f}s")
            print()
    
    return results


def print_comparison_table(results: dict[str, dict[str, float]]) -> None:
    """Print formatted comparison table."""
    print("=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Method':<25} {'L2 Error':>15} {'Time (s)':>12} {'Quality':>15}")
    print("-" * 70)
    
    method_names = {
        "gauss2": "Gauss-Legendre",
        "radau2": "Radau IIA",
        "lobatto2": "Lobatto IIIA",
    }
    
    for method, data in results.items():
        error = data["error"]
        train_time = data["time"]
        
        # Quality assessment
        if error < 0.01:
            quality = "Excellent"
        elif error < 0.05:
            quality = "Good"
        elif error < 0.1:
            quality = "Acceptable"
        else:
            quality = "Poor"
        
        print(f"{method_names[method]:<25} {error:>15.6e} {train_time:>12.2f} {quality:>15}")
    
    print("=" * 70)
    
    # Find best method
    best_accuracy = min(results.items(), key=lambda x: x[1]["error"])
    best_speed = min(results.items(), key=lambda x: x[1]["time"])
    
    print(f"\nBest accuracy: {method_names[best_accuracy[0]]} "
          f"(error = {best_accuracy[1]['error']:.6e})")
    print(f"Fastest: {method_names[best_speed[0]]} "
          f"(time = {best_speed[1]['time']:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description="Compare RK methods on heat equation")
    parser.add_argument("--methods", nargs="+", default=["gauss2", "radau2", "lobatto2"],
                        choices=["gauss2", "radau2", "lobatto2"],
                        help="Methods to compare")
    parser.add_argument("--T", type=float, default=0.1, help="Final time")
    parser.add_argument("--N", type=int, default=20, help="Number of time steps")
    parser.add_argument("--n-x-train", type=int, default=256, help="Spatial batch size")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Compare methods
    results = compare_methods(
        methods=args.methods,
        T=args.T,
        N=args.N,
        n_x_train=args.n_x_train,
        steps=args.steps,
        lr=args.lr,
        device=device,
        verbose=not args.quiet,
    )
    
    # Print comparison
    if not args.quiet:
        print_comparison_table(results)
        
        print("\nRecommendations:")
        print("- Radau IIA: Best balance of stability and accuracy (L-stable)")
        print("- Gauss-Legendre: Highest order, best for smooth solutions")
        print("- Lobatto IIIA: Simplest (trapezoidal), good for testing")


if __name__ == "__main__":
    main()
