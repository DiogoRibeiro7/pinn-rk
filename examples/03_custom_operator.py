"""
Example 3: Custom PDE Operator

This example demonstrates:
- Implementing the `EllipticOperator` protocol for a new PDE
- Adding a reaction term to the diffusion operator
- Supplying a non-zero right-hand side f(x, t)
- Validating an operator against a manufactured solution before training on it

PDE: u_t - D u_xx + k u = f,   x in (0,1), t in (0, T]
BC:  u(0,t) = u(1,t) = 0
IC:  u(x,0) = sin(π x)

Written in pinn-rk's convention u_t + L u = f, the reaction-diffusion operator is

    L u = -D u_xx + k u

so `Laplacian1D` is the special case D = 1, k = 0.

Manufactured solution
---------------------
We *choose* u(x,t) = sin(π x) exp(-α t) and derive the f that makes it exact:

    u_t   = -α u
    u_xx  = -π² u
    f     = u_t + L u = (-α + D π² + k) u

Choosing α = D π² + k gives f ≡ 0, but that only re-tests the homogeneous case. Taking
α freely leaves a non-zero f and genuinely exercises the right-hand side path.
"""

from __future__ import annotations

import argparse
import math

import torch
from torch import Tensor, nn

from pinn_rk import (
    MLP,
    RkPinnConfig,
    RkPinnLoss,
    TimeMesh,
    butcher_radau_iia_q3,
)

# --- Problem parameters -------------------------------------------------------------

DIFFUSIVITY = 0.5  # D
REACTION = 2.0  # k
DECAY = 3.0  # alpha, chosen independently so that f is not identically zero


class ReactionDiffusion1D:
    """
    L u = -D u_xx + k u, implementing the `EllipticOperator` protocol.

    Only two methods are required. `requires_hessian` tells the loss whether second
    derivatives in x are needed, so it can enable the higher-order autograd graph; a
    purely first-order operator would return False and train more cheaply.
    """

    def __init__(self, diffusivity: float = DIFFUSIVITY, reaction: float = REACTION) -> None:
        self.diffusivity = diffusivity
        self.reaction = reaction

    def __call__(self, x: Tensor, u: Tensor) -> Tensor:
        assert x.ndim == u.ndim == 2 and x.shape == u.shape
        grad = torch.autograd.grad(
            u, x, torch.ones_like(u), create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        u_xx = torch.autograd.grad(
            grad, x, torch.ones_like(grad), create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        return -self.diffusivity * u_xx + self.reaction * u

    def requires_hessian(self) -> bool:
        return True


def exact_solution(x: Tensor, t: Tensor) -> Tensor:
    """u(x,t) = sin(π x) exp(-α t)."""
    return torch.sin(math.pi * x) * torch.exp(-DECAY * t)


def source_term(x: Tensor, t: Tensor) -> Tensor:
    """f = (-α + D π² + k) u, the forcing that makes `exact_solution` exact."""
    coeff = -DECAY + DIFFUSIVITY * math.pi**2 + REACTION
    return coeff * exact_solution(x, t)


# --- Operator validation ------------------------------------------------------------


def validate_operator(device: torch.device, nx: int = 256, tol: float = 1e-10) -> float:
    """
    Check the operator against the manufactured solution before training on it.

    A miswritten operator produces a PINN that trains happily to the wrong answer, and
    that failure is expensive to diagnose later. Here L u is compared to its closed
    form, -D π² u + k u, which is cheap and catches sign and factor errors immediately.
    """
    x = torch.linspace(0.05, 0.95, nx, device=device, dtype=torch.float64).unsqueeze(1)
    x = x.requires_grad_(True)
    t = torch.full_like(x, 0.1)

    u = exact_solution(x, t)
    computed = ReactionDiffusion1D()(x, u)
    expected = (DIFFUSIVITY * math.pi**2 + REACTION) * u

    err = float((computed - expected).abs().max().item())
    if err > tol:
        raise AssertionError(f"operator disagrees with its closed form: max error {err:.3e}")
    return err


# --- Training -----------------------------------------------------------------------


def l2_error(model: nn.Module, T: float, nx: int, device: torch.device) -> float:
    x = torch.linspace(0, 1, nx, device=device, dtype=torch.float64).unsqueeze(1)
    t = torch.full_like(x, T)
    with torch.no_grad():
        return float(torch.sqrt(torch.mean((model(x, t) - exact_solution(x, t)) ** 2)).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Reaction-diffusion with a custom operator")
    parser.add_argument("--T", type=float, default=0.1, help="Final time")
    parser.add_argument("--N", type=int, default=8, help="Number of time slabs")
    parser.add_argument("--n-x-train", type=int, default=128, help="Spatial batch size")
    parser.add_argument("--steps", type=int, default=800, help="Training steps")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
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

    err = validate_operator(device)
    if not args.quiet:
        coeff = -DECAY + DIFFUSIVITY * math.pi**2 + REACTION
        print(f"operator validated against closed form (max error {err:.2e})")
        print(f"D = {DIFFUSIVITY}, k = {REACTION}, alpha = {DECAY}")
        print(f"source coefficient = {coeff:.6f} (non-zero, so f is exercised)\n")

    x0 = torch.linspace(1e-6, 1 - 1e-6, 128, device=device, dtype=torch.float64).unsqueeze(1)
    u0 = exact_solution(x0, torch.zeros_like(x0))

    cfg = RkPinnConfig(
        tableau=butcher_radau_iia_q3(device),
        time_mesh=TimeMesh.uniform(T=args.T, N=args.N, device=device),
        residual="rk",
        n_x_train=args.n_x_train,
        device=device,
        init_data=(x0, u0),
    )
    model = MLP(width=96, depth=4, dtype=torch.float64).to(device)
    loss_fn = RkPinnLoss(model=model, Lop=ReactionDiffusion1D(), f_rhs=source_term, cfg=cfg).to(
        device
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if not args.quiet and step % 200 == 0:
            print(f"[{step:5d}] loss = {loss.item():.4e}")

    err_final = l2_error(model, args.T, nx=401, device=device)
    if not args.quiet:
        print(f"\nL2 error at T={args.T}: {err_final:.4e}")


if __name__ == "__main__":
    main()
