"""
Example 4: Convergence Study

This example demonstrates:
- Sweeping over N, the number of time slabs
- Measuring observed convergence rates and comparing them to theory
- Generating plots
- Exporting results to CSV

Two quantities are measured, and keeping them apart is the whole point of this script.

**Consistency error** is the residual left on the *exact* solution. It is deterministic,
cheap, and isolates the time discretisation: it converges at the tableau's own order and
bounds what a perfectly trained network could achieve.

**Trained L2 error** is what an actual optimisation run reaches. It is stochastic -- the
spatial sampler redraws every step -- and is limited by optimisation, not only by the
discretisation. It does not have to follow the consistency rate, and generally will not
until training is converged.

Reporting only the first would overstate the method; reporting only the second would
hide where the error comes from.

Usage
-----
    python examples/04_convergence_study.py                       # consistency only, seconds
    python examples/04_convergence_study.py --train               # adds training, minutes
    python examples/04_convergence_study.py --save-plots --export-csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from torch import Tensor, nn

from pinn_rk import (
    MLP,
    Laplacian1D,
    RkPinnConfig,
    RkPinnLoss,
    TimeMesh,
    butcher_gauss_legendre_q2,
    butcher_gauss_legendre_q3,
    butcher_lobatto_iiia_q2,
    butcher_lobatto_iiia_q3,
    butcher_radau_iia_q2,
    butcher_radau_iia_q3,
)

# name -> (factory, stages q, classical order p)
METHODS = {
    "gauss2": (butcher_gauss_legendre_q2, 2, 4),
    "radau2": (butcher_radau_iia_q2, 2, 3),
    "lobatto2": (butcher_lobatto_iiia_q2, 2, 2),
    "gauss3": (butcher_gauss_legendre_q3, 3, 6),
    "radau3": (butcher_radau_iia_q3, 3, 5),
    "lobatto3": (butcher_lobatto_iiia_q3, 3, 4),
}

T_FINAL = 0.1


def exact_solution(x: Tensor, t: Tensor) -> Tensor:
    return torch.sin(math.pi * x) * torch.exp(-(math.pi**2) * t)


def zero_source(x: Tensor, t: Tensor) -> Tensor:
    return torch.zeros_like(x)


class ExactSolution(nn.Module):
    """Stands in for a perfectly trained network, to isolate truncation error."""

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return exact_solution(x, t)


# --- Consistency ---------------------------------------------------------------------


def consistency(method: str, N: int, device: torch.device, nx: int = 48) -> tuple[float, float]:
    """Max |stage residual| and |update residual| on the exact solution, first slab."""
    factory, _, _ = METHODS[method]
    cfg = RkPinnConfig(
        tableau=factory(device),
        time_mesh=TimeMesh.uniform(T=T_FINAL, N=N, device=device),
        residual="rk",
        n_x_train=nx,
        device=device,
    )
    loss_fn = RkPinnLoss(model=ExactSolution(), Lop=Laplacian1D(), f_rhs=zero_source, cfg=cfg)
    x = torch.linspace(0.05, 0.95, nx, device=device, dtype=torch.float64).unsqueeze(1)
    r_stage, r_step = loss_fn.rk_residuals(x, 0, cfg.time_mesh.steps[0], loss_fn._stage_times(0))
    return float(r_stage.detach().abs().max()), float(r_step.detach().abs().max())


# --- Training ------------------------------------------------------------------------


def train_once(method: str, N: int, device: torch.device, steps: int, seed: int) -> float:
    factory, _, _ = METHODS[method]
    torch.manual_seed(seed)
    x0 = torch.linspace(1e-6, 1 - 1e-6, 128, device=device, dtype=torch.float64).unsqueeze(1)
    cfg = RkPinnConfig(
        tableau=factory(device),
        time_mesh=TimeMesh.uniform(T=T_FINAL, N=N, device=device),
        residual="rk",
        n_x_train=96,
        device=device,
        init_data=(x0, exact_solution(x0, torch.zeros_like(x0))),
    )
    model = MLP(width=96, depth=4, dtype=torch.float64).to(device)
    loss_fn = RkPinnLoss(model=model, Lop=Laplacian1D(), f_rhs=zero_source, cfg=cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

    x = torch.linspace(0, 1, 401, device=device, dtype=torch.float64).unsqueeze(1)
    t = torch.full_like(x, T_FINAL)
    with torch.no_grad():
        return float(torch.sqrt(torch.mean((model(x, t) - exact_solution(x, t)) ** 2)).item())


# --- Rates ---------------------------------------------------------------------------


def pairwise_rates(ks: list[float], errs: list[float]) -> list[float | None]:
    """log-log slope between consecutive refinements; None where it is undefined."""
    rates: list[float | None] = [None]
    for i in range(1, len(ks)):
        if errs[i] > 0 and errs[i - 1] > 0:
            rates.append(math.log(errs[i - 1] / errs[i]) / math.log(ks[i - 1] / ks[i]))
        else:
            rates.append(None)
    return rates


def fitted_order(ks: list[float], errs: list[float]) -> float:
    lk = [math.log(k) for k in ks]
    le = [math.log(max(e, 1e-300)) for e in errs]
    n = len(ks)
    mk, me = sum(lk) / n, sum(le) / n
    num = sum((a - mk) * (b - me) for a, b in zip(lk, le, strict=True))
    return num / sum((a - mk) ** 2 for a in lk)


# --- Reporting -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Convergence study over time slabs")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["radau2", "radau3", "gauss2", "gauss3"],
        choices=sorted(METHODS),
        help="Tableaux to sweep",
    )
    parser.add_argument("--slabs", nargs="+", type=int, default=[3, 5, 8, 12], help="Values of N")
    parser.add_argument("--train", action="store_true", help="Also run training (slow)")
    parser.add_argument("--steps", type=int, default=600, help="Training steps per point")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the training runs")
    parser.add_argument("--save-plots", action="store_true", help="Write convergence plots")
    parser.add_argument("--output-dir", type=str, default=".", help="Where to write plots")
    parser.add_argument("--export-csv", type=str, default=None, help="Write results to CSV")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    ks = [T_FINAL / n for n in args.slabs]
    rows: list[dict[str, object]] = []

    for method in args.methods:
        _, q, p = METHODS[method]
        stage = [consistency(method, n, device)[0] for n in args.slabs]
        step = [consistency(method, n, device)[1] for n in args.slabs]
        trained = (
            [train_once(method, n, device, args.steps, args.seed) for n in args.slabs]
            if args.train
            else [float("nan")] * len(args.slabs)
        )

        r_stage, r_step = pairwise_rates(ks, stage), pairwise_rates(ks, step)

        print(f"\n=== {method}  (q={q}, classical order p={p}) ===")
        header = f"{'N':>4} {'k':>9} {'stage res':>12} {'rate':>6} {'update res':>12} {'rate':>6}"
        if args.train:
            header += f" {'trained L2':>12}"
        print(header)
        print("-" * len(header))
        for i, n in enumerate(args.slabs):
            line = (
                f"{n:>4} {ks[i]:9.5f} {stage[i]:12.3e} "
                f"{('  -  ' if r_stage[i] is None else f'{r_stage[i]:6.2f}')} "
                f"{step[i]:12.3e} "
                f"{('  -  ' if r_step[i] is None else f'{r_step[i]:6.2f}')}"
            )
            if args.train:
                line += f" {trained[i]:12.3e}"
            print(line)
            rows.append(
                {
                    "method": method,
                    "q": q,
                    "classical_order": p,
                    "N": n,
                    "k": ks[i],
                    "stage_residual": stage[i],
                    "update_residual": step[i],
                    "trained_l2": trained[i],
                }
            )

        print(f"  fitted stage order  = {fitted_order(ks, stage):.2f}   (expected {q})")
        print(f"  fitted update order = {fitted_order(ks, step):.2f}   (expected {p})")

    if args.train:
        print(
            "\nNote: trained L2 is stochastic and optimisation-limited. It is not expected\n"
            "to follow the consistency rates, and a single seed cannot establish a trend."
        )

    if args.export_csv:
        path = Path(args.export_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {path} ({len(rows)} rows)")

    if args.save_plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise SystemExit(
                "--save-plots needs matplotlib: poetry install --with examples"
            ) from None

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
        for method in args.methods:
            sel = [r for r in rows if r["method"] == method]
            kk = [r["k"] for r in sel]
            a1.loglog(kk, [r["stage_residual"] for r in sel], "o-", label=method)
            a2.loglog(kk, [r["update_residual"] for r in sel], "o-", label=method)
        for ax, title in ((a1, "Stage residual"), (a2, "Update residual")):
            ax.set_xlabel("slab size k")
            ax.set_ylabel("max residual on exact solution")
            ax.set_title(title)
            ax.grid(alpha=0.3, which="both")
            ax.legend()
        plt.tight_layout()
        target = out_dir / "convergence.png"
        fig.savefig(target, dpi=130)
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
