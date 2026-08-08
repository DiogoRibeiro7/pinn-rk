from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn

from .config import RkPinnConfig
from .interpolants import differentiation_matrix
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

        if cfg.ic_weight < 0.0:
            raise ValueError("ic_weight must be non-negative.")

        if cfg.init_data is not None:
            # The H¹ penalty differentiates the sampled u0 along this grid, which is
            # only meaningful if the samples are ordered.
            x0_init = cfg.init_data[0]
            if x0_init.ndim != 2 or x0_init.shape[1] != 1:
                raise ValueError("init_data x0 must be a [N,1] tensor.")
            if not bool(torch.all(x0_init[1:, 0] > x0_init[:-1, 0])):
                raise ValueError(
                    "init_data x0 must be sorted in strictly increasing order; the "
                    "initial-condition penalty differentiates u0 on this grid."
                )

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

    def _interp_nodes(self, t_stage: Tensor, t_n: Tensor) -> tuple[Tensor, bool]:
        """
        Nodes carrying the polynomial time reconstruction on a slab.

        With ``q_aux="same"`` the reconstruction interpolates the q stage values, so
        û has degree q-1. With ``q_aux="extend"`` the slab start t_n joins them,
        raising û to degree q at the cost of one extra network evaluation. Tableaux
        whose first node already sits at t_n (Lobatto IIIA has c_1 = 0) would
        duplicate a node and produce infinite barycentric weights, so for those the
        stencil is left unextended.

        Returns the nodes and whether t_n was prepended.
        """
        if self.cfg.q_aux == "extend" and not bool(
            torch.isclose(t_stage, t_n, atol=1e-14, rtol=0.0).any()
        ):
            return torch.cat([t_n.reshape(1), t_stage]), True
        return t_stage, False

    def _stage_values(self, x: Tensor, t_stage: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate the network and the PDE operator at the stage nodes.

        Returns ``(U, LU, F_rhs)``, each ``[B, q, 1]``.
        """
        u_stage: list[Tensor] = []
        Lu_stage: list[Tensor] = []
        f_stage: list[Tensor] = []
        for i in range(t_stage.numel()):
            ti = torch.full_like(x, fill_value=t_stage[i].item())
            ui = self._eval_model(x, ti)  # [B,1]
            u_stage.append(ui)
            Lu_stage.append(self.L(x, ui))  # [B,1]
            f_stage.append(self.f_rhs(x, ti))
        return (
            torch.stack(u_stage, dim=1),
            torch.stack(Lu_stage, dim=1),
            torch.stack(f_stage, dim=1),
        )

    def _slab_loss_rk(self, x: Tensor, n: int, k_n: Tensor, t_stage: Tensor) -> Tensor:
        """
        Runge-Kutta collocation residual, built from the full Butcher tableau.

        Writing the semi-discrete problem as u' = f - L u =: F, the method is defined
        by the stage equations and the update equation

            U_i     = u_n + k Σ_j a_ij F_j,
            u_{n+1} = u_n + k Σ_i b_i F_i,

        which are imposed here as residuals on the network. Dividing by k gives them
        the units of a time derivative, so they stay comparable across slab sizes:

            r_i    = (U_i - u_n)/k     - Σ_j a_ij F_j,
            r_step = (u_{n+1} - u_n)/k - Σ_i b_i F_i.

        Unlike differentiating an interpolant, this uses A, and so reproduces the
        tableau's own accuracy: the stage residual carries the method's stage order
        and the update residual its classical order (4 for Gauss q=2, 3 for Radau
        IIA q=2, 2 for Lobatto IIIA q=2).
        """
        bt = self.cfg.tableau
        q = bt.c.numel()
        b = bt.b.to(device=self.cfg.device, dtype=self.cfg.dtype)  # [q]

        r_stage, r_step = self.rk_residuals(x, n, k_n, t_stage)
        sq_stage = (r_stage**2).mean(dim=0)  # [q,1]
        sq_step = (r_step**2).mean(dim=0)  # [1]
        return k_n * (torch.sum(b.view(q, 1) * sq_stage) + torch.sum(sq_step))

    def rk_residuals(
        self, x: Tensor, n: int, k_n: Tensor, t_stage: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Stage and update residuals of the Runge-Kutta collocation form on slab ``n``.

        Exposed separately from the loss so the two can be measured independently:
        they converge at different rates, and only the update residual reflects the
        tableau's classical order.

        Returns ``(r_stage, r_step)`` with shapes ``[B, q, 1]`` and ``[B, 1]``.
        """
        device, dtype = self.cfg.device, self.cfg.dtype
        bt = self.cfg.tableau
        A = bt.A.to(device=device, dtype=dtype)  # [q,q]
        b = bt.b.to(device=device, dtype=dtype)  # [q]
        times = self.cfg.time_mesh

        u_n = self._eval_model(x, torch.full_like(x, float(times.nodes[n].item())))  # [B,1]
        U, LU, F_rhs = self._stage_values(x, t_stage)
        F = F_rhs - LU  # u' = f - L u, evaluated at the stage nodes  [B,q,1]

        # Stage equations: these are what bring A into the loss.
        r_stage = (U - u_n.unsqueeze(1)) / k_n - torch.einsum("ij,bjk->bik", A, F)  # [B,q,1]

        # Update equation, carrying the tableau's classical order.
        u_next = self._eval_model(x, torch.full_like(x, float(times.nodes[n + 1].item())))
        r_step = (u_next - u_n) / k_n - torch.einsum("i,bik->bk", b, F)  # [B,1]
        return r_stage, r_step

    def _slab_loss_interpolant(self, x: Tensor, n: int, k_n: Tensor, t_stage: Tensor) -> Tensor:
        """
        Residual measured against the polynomial time reconstruction.

        ∂ₜû at the stage nodes is taken analytically from û(t) = Σ_j L_j(t) U_j, so
        the network is never differentiated in t. This uses only the nodes c and the
        quadrature weights b; accuracy follows the degree of û rather than the order
        of the tableau. See ``q_aux`` for the choice of reconstruction stencil.
        """
        device, dtype = self.cfg.device, self.cfg.dtype
        bt = self.cfg.tableau
        q = bt.c.numel()
        times = self.cfg.time_mesh

        U, LU, F_stack = self._stage_values(x, t_stage)

        interp_t, extended = self._interp_nodes(t_stage, times.nodes[n])
        if extended:
            t_start = torch.full_like(x, float(times.nodes[n].item()))
            u_start = self._eval_model(x, t_start)  # [B,1]
            U_interp = torch.cat([u_start.unsqueeze(1), U], dim=1)  # [B,q+1,1]
        else:
            U_interp = U  # [B,q,1]

        D = differentiation_matrix(interp_t)  # [m,m], m = q or q+1
        if extended:
            D = D[1:]  # keep only the rows evaluating at stage nodes
        u_t_eval = torch.einsum("ij,bjk->bik", D, U_interp)  # [B,q,1]

        # In collocation form, Π_{q-1} evaluations at nodes equal values there
        res = u_t_eval + LU - F_stack  # [B,q,1]
        b = bt.b.view(1, q, 1).to(device=device, dtype=dtype)  # [1,q,1]

        # integrate over time slab with RK weights (mean over x)
        sq = (res**2).mean(dim=0, keepdim=False)  # [q,1]
        return k_n * torch.sum(b * sq)

    def forward(self) -> Tensor:
        device = self.cfg.device
        dtype = self.cfg.dtype
        times = self.cfg.time_mesh

        total = torch.zeros((), device=device, dtype=dtype)

        for n in range(times.steps.numel()):
            k_n = times.steps[n]
            assert self.cfg.spatial_sampler is not None
            x = self.cfg.spatial_sampler(self.cfg.n_x_train, device)  # [B,1]
            t_stage = self._stage_times(n)  # [q]

            if self.cfg.residual == "rk":
                total = total + self._slab_loss_rk(x, n, k_n, t_stage)
            else:
                total = total + self._slab_loss_interpolant(x, n, k_n, t_stage)

        # Initial condition H¹ seminorm penalty if provided
        if self.cfg.init_data is not None and self.cfg.ic_weight != 0.0:
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
            # u0 is supplied as sampled values and carries no autograd history, so the
            # target derivative ∂ₓu₀ is taken numerically on the x0 grid rather than by
            # autograd. Constant w.r.t. θ, hence detached.
            x0_grid = x0.detach().squeeze(1)
            u0_grid = u0.detach().squeeze(1)
            grad_u0 = torch.gradient(u0_grid, spacing=(x0_grid,))[0].unsqueeze(1)
            total = total + self.cfg.ic_weight * torch.nn.functional.mse_loss(grad_u, grad_u0)

        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite loss encountered.")
        return total
