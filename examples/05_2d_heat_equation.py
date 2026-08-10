"""
Example 5: 2D Heat Equation

This example demonstrates:
- Solving a PDE on a two-dimensional domain
- `LaplacianND`, which works in any number of spatial dimensions
- Exact boundary conditions on all four edges, with no boundary term in the loss
- Supplying the initial condition as a **callable**, which is what makes the H1
  penalty work beyond one dimension

PDE: u_t - (u_xx + u_yy) = 0,   (x,y) in (0,1)^2,  t in (0,T]
BC:  u = 0 on the whole boundary
IC:  u(x,y,0) = sin(pi x) sin(pi y)

Exact solution: u(x,y,t) = sin(pi x) sin(pi y) exp(-2 pi^2 t)

Note the decay rate. In 1D the mode decays like exp(-pi^2 t); in 2D the Laplacian
contributes pi^2 per dimension, so it decays twice as fast. Time intervals that were
reasonable in 1D leave almost nothing to fit here, which is why T defaults to 0.02.

Expected runtime: ~2-4 minutes on CPU.
"""

from __future__ import annotations

import argparse
import math

import torch
from torch import Tensor, nn

from pinn_rk import (
    MLP,
    LaplacianND,
    RkPinnConfig,
    RkPinnLoss,
    TimeMesh,
    butcher_radau_iia_q3,
)


def exact_solution(X: Tensor, t: Tensor) -> Tensor:
    """u(x,y,t) = sin(pi x) sin(pi y) exp(-2 pi^2 t)."""
    return (
        torch.sin(math.pi * X[:, :1])
        * torch.sin(math.pi * X[:, 1:2])
        * torch.exp(-2 * math.pi**2 * t)
    )


def initial_condition(X: Tensor) -> Tensor:
    """u0(x,y) = sin(pi x) sin(pi y), as a callable so autograd can differentiate it."""
    return torch.sin(math.pi * X[:, :1]) * torch.sin(math.pi * X[:, 1:2])


def source_term(X: Tensor, t: Tensor) -> Tensor:
    """Homogeneous: the manufactured solution already satisfies the equation."""
    return torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)


def grid(n: int, device: torch.device, lo: float = 0.0, hi: float = 1.0) -> Tensor:
    """A regular [n*n, 2] grid over the square."""
    g = torch.linspace(lo, hi, n, device=device, dtype=torch.float64)
    return torch.stack(torch.meshgrid(g, g, indexing="ij"), dim=-1).reshape(-1, 2)


def l2_error(model: nn.Module, t_val: float, n: int, device: torch.device) -> tuple[float, float]:
    """Absolute and relative L2 error over the square at a fixed time."""
    X = grid(n, device)
    t = torch.full((X.shape[0], 1), t_val, device=device, dtype=torch.float64)
    with torch.no_grad():
        ref = exact_solution(X, t)
        err = float((model(X, t) - ref).pow(2).mean().sqrt())
        norm = float(ref.pow(2).mean().sqrt())
    return err, (err / norm if norm > 0 else float("nan"))


def main() -> None:
    parser = argparse.ArgumentParser(description="2D heat equation with an RK-PINN")
    parser.add_argument("--T", type=float, default=0.02, help="Final time")
    parser.add_argument("--N", type=int, default=4, help="Number of time slabs")
    parser.add_argument("--n-x-train", type=int, default=256, help="Spatial samples per slab")
    parser.add_argument("--steps", type=int, default=1500, help="Training steps")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--width", type=int, default=64, help="Network width")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    # Initial data as a callable. With sampled values the target derivative would have to
    # be taken along a sorted grid, and no such ordering exists in two dimensions.
    x0 = grid(24, device, lo=1e-6, hi=1 - 1e-6)

    cfg = RkPinnConfig(
        tableau=butcher_radau_iia_q3(device),
        time_mesh=TimeMesh.uniform(T=args.T, N=args.N, device=device),
        residual="rk",
        n_x_train=args.n_x_train,
        space_dim=2,
        device=device,
        init_data=(x0, initial_condition),
    )
    # in_dim = 3: two spatial coordinates plus time.
    model = MLP(in_dim=3, width=args.width, depth=4, dtype=torch.float64).to(device)
    loss_fn = RkPinnLoss(model=model, Lop=LaplacianND(), f_rhs=source_term, cfg=cfg).to(device)

    if not args.quiet:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"2D heat equation on (0,1)^2, T={args.T}, N={args.N}, {n_params:,} parameters")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if not args.quiet and step % 250 == 0:
            print(f"[{step:5d}] loss = {loss.item():.4e}")

    if not args.quiet:
        print()
        for t_val in (0.0, args.T):
            err, rel = l2_error(model, t_val, 81, device)
            print(f"t={t_val:.3f}   L2 error = {err:.4e}   relative = {rel:.2%}")

        # The ansatz makes this exact, not approximate.
        edges = torch.tensor(
            [[0.0, 0.3], [1.0, 0.3], [0.3, 0.0], [0.3, 1.0]],
            device=device,
            dtype=torch.float64,
        )
        t_edge = torch.full((edges.shape[0], 1), args.T, device=device, dtype=torch.float64)
        with torch.no_grad():
            print(f"max |u| on the boundary = {float(model(edges, t_edge).abs().max()):.2e}")


if __name__ == "__main__":
    main()
