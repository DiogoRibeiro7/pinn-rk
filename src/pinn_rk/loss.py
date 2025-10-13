from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn

from .config import RkPinnConfig
from .interpolants import barycentric_weights, lagrange_eval
from .operators import EllipticOperator


class RkPinnLoss(nn.Module):
    """
    Time-discrete RK-PINN objective:
      Sum over slabs J_n of k_n * Σ_i b_i || r(t_{n,i}) ||^2 + optional H¹ seminorm IC penalty.
    Residual r = d/dt û + Π_{q-1}(L û) - Π̃_{q-1} f, evaluated at collocation nodes.
    """

    def __init__(
        self,
        model: nn.Module,
        Lop: EllipticOperator,
        f_rhs: Callable[[Tensor, Tensor], Tensor],  # f(x,t)
        cfg: RkPinnConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.L = Lop
        self.f_rhs = f_rhs
        self.cfg = cfg
        self.to(cfg.device, dtype=cfg.dtype)

        if cfg.spatial_sampler is None:
            self.cfg.spatial_sampler = self._default_uniform_sampler

        self._cache: dict[int, dict[str, Tensor]] = {}

    @staticmethod
    def _default_uniform_sampler(n: int, device: torch.device) -> Tensor:
        # Uniform in (0,1); avoid exact boundaries as BCs are embedded in the ansatz.
        x = torch.rand(n, 1, device=device, dtype=torch.float64)
        return (1e-6) + (1 - 2e-6) * x

    def _stage_times(self, n: int) -> Tensor:
        t_n = self.cfg.time_mesh.nodes[n]
        k_n = self.cfg.time_mesh.steps[n]
        c = self.cfg.tableau.c.to(self.cfg.device, dtype=self.cfg.dtype)
        return t_n + c * k_n

    def _eval_model(self, x: Tensor, t: Tensor) -> Tensor:
        # enable higher-order grads if required by L
        x.requires_grad_(self.L.requires_hessian())
        t.requires_grad_(self.L.requires_hessian())
        return cast(Tensor, self.model(x, t))

    def forward(self) -> Tensor:
        device = self.cfg.device
        dtype = self.cfg.dtype
        bt = self.cfg.tableau
        times = self.cfg.time_mesh
        q = bt.c.numel()

        total = torch.zeros((), device=device, dtype=dtype)

        for n in range(times.steps.numel()):
            k_n = times.steps[n]
            assert self.cfg.spatial_sampler is not None
            x = self.cfg.spatial_sampler(self.cfg.n_x_train, device)  # [B,1]

            # Stage times and barycentric weights on J_n
            t_stage = self._stage_times(n)  # [q]
            w_nodes = barycentric_weights(t_stage)

            # Evaluate at stage nodes
            u_stage: list[Tensor] = []
            Lu_stage: list[Tensor] = []
            f_stage: list[Tensor] = []

            for i in range(q):
                ti = torch.full_like(x, fill_value=t_stage[i].item())
                ui = self._eval_model(x, ti)  # [B,1]
                u_stage.append(ui)
                Lu_stage.append(self.L(x, ui))  # [B,1]
                f_stage.append(self.f_rhs(x, ti))

            U = torch.stack(u_stage, dim=1)  # [B,q,1]
            LU = torch.stack(Lu_stage, dim=1)  # [B,q,1]
            F_stack = torch.stack(f_stage, dim=1)  # [B,q,1]

            # Interpolant û at collocation nodes (not used currently but computed for completeness)
            t_eval = t_stage.view(1, q).repeat(self.cfg.n_x_train, 1)  # [B,q]
            L_eval = lagrange_eval(t_eval, t_stage, w_nodes)  # [B,q,q]
            _ = torch.einsum("bij, bj1 -> bi1", L_eval, U)  # [B,q,1]

            # Approximate û_t via symmetric finite differences around each stage time
            eps = 1e-6 * float(k_n.item())
            u_t_list: list[Tensor] = []
            for i in range(q):
                ti_val = float(t_stage[i].item())
                t_left = torch.full_like(x, ti_val - eps)
                t_right = torch.full_like(x, ti_val + eps)
                ui_r = self._eval_model(x, t_right)
                ui_l = self._eval_model(x, t_left)
                u_t_list.append((ui_r - ui_l) / (2.0 * eps))
            u_t_eval = torch.stack(u_t_list, dim=1)  # [B,q,1]

            # In collocation form, Π_{q-1} evaluations at nodes equal values there
            Lhat_proj = LU
            f_proj = F_stack

            res = u_t_eval + Lhat_proj - f_proj  # [B,q,1]
            b = bt.b.view(1, q, 1).to(device=device, dtype=dtype)  # [1,q,1]

            # integrate over time slab with RK weights (mean over x)
            sq = (res**2).mean(dim=0, keepdim=False)  # [q,1]
            slab = k_n * torch.sum(b * sq)
            total = total + slab

        # Initial condition H¹ seminorm penalty if provided
        if self.cfg.init_data is not None:
            x0, u0 = self.cfg.init_data
            x0 = x0.to(device=device, dtype=dtype)
            u0 = u0.to(device=device, dtype=dtype)
            t0 = torch.zeros_like(x0)
            u_init = self._eval_model(x0, t0)
            grad_u = torch.autograd.grad(
                u_init,
                x0,
                grad_outputs=torch.ones_like(u_init),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grad_u0 = torch.autograd.grad(
                u0,
                x0,
                grad_outputs=torch.ones_like(u0),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            total = total + torch.nn.functional.mse_loss(grad_u, grad_u0)

        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite loss encountered.")
        return total
