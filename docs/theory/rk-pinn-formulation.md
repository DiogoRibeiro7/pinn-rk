# RK-PINN Formulation

## The problem

We consider linear parabolic problems

$$
u_t + \mathcal{L}u = f \quad \text{in } \Omega\times(0,T], \qquad
u = 0 \text{ on } \partial\Omega, \qquad u(\cdot,0) = u_0 ,
$$

with $\mathcal{L}$ elliptic. Writing the semi-discrete problem as an evolution equation,

$$
u' = f - \mathcal{L}u =: F ,
$$

puts it in the form a Runge–Kutta method expects.

## Time slabs and stage nodes

The interval is partitioned into slabs $J_n = [t_n, t_{n+1}]$ of size $k_n$. A $q$-stage
tableau with nodes $c_i$ defines **stage times** $t_{n,i} = t_n + c_i k_n$. The network
$u_\theta(x,t)$ is evaluated there, giving stage values $U_i$ and stage slopes
$F_i = f(t_{n,i}) - \mathcal{L}u_\theta(t_{n,i})$.

## The residual

A Runge–Kutta method is *defined* by two relations — the stage equations and the update
equation:

$$
U_i = u_n + k\sum_j a_{ij}F_j , \qquad u_{n+1} = u_n + k\sum_i b_i F_i .
$$

`pinn-rk` imposes both directly on the network, divided by $k$ so they carry the units of
a time derivative and stay comparable across slab sizes:

$$
r_i = \frac{U_i - u_n}{k} - \sum_j a_{ij}F_j , \qquad
r_{\text{step}} = \frac{u_{n+1} - u_n}{k} - \sum_i b_i F_i .
$$

The slab contribution is these residuals in a weighted least-squares sense,

$$
k_n\Big(\textstyle\sum_i b_i \lVert r_i\rVert^2 + \lVert r_{\text{step}}\rVert^2\Big),
$$

summed over slabs. Crucially this uses the **full** tableau, including the coupling
matrix $A$. The network is never differentiated in $t$.

## What that buys

Because the exact solution satisfies the PDE, feeding it through the residual leaves only
local truncation error. Measured on $u = \sin(\pi x)e^{-\pi^2 t}$:

| tableau | stages | stage residual order | update residual order | classical $p$ |
| --- | --- | --- | --- | --- |
| `lobatto2` | 2 | 1.91 | 1.91 | 2 |
| `radau2` | 2 | 1.92 | 2.91 | 3 |
| `gauss2` | 2 | 1.94 | 3.91 | 4 |
| `lobatto3` | 3 | 2.92 | 3.91 | 4 |
| `radau3` | 3 | 2.92 | 4.91 | 5 |
| `gauss3` | 3 | 2.91 | 5.91 | 6 |
| `lobatto4` | 4 | 3.88 | 5.85 | 6 |
| `radau4` | 4 | 3.84 | 6.84 | 7 |
| `gauss4` | 4 | 3.86 | not measurable | 8 |

Two patterns, both of them consequences of the theory rather than coincidences:

- The **update residual** converges at the tableau's classical order $p$. This is what
  makes the choice of tableau meaningful.
- The **stage residual** converges at the **stage order**, which for collocation methods
  equals the number of stages $q$.

Since the objective sums both, the stage term dominates and **the stage order sets the
ceiling**. That is why the stage count matters more than the classical order alone
suggests: each added stage lifts the ceiling by one power of $k$, from
$\mathcal{O}(k^2)$ at $q=2$ to $\mathcal{O}(k^4)$ at $q=4$.

!!! note "Gauss q=4 outruns double precision"
    Its update residual is order 8, and $(u_{n+1}-u_n)/k$ carries rounding of about
    $\epsilon/k$ &mdash; near $2.6\times10^{-14}$ at $k=1/120$. The true residual falls
    below that, so refining the mesh makes the measured value stall instead of fall.
    On coarse slabs, where it is still visible, the observed rate is $\approx 7.7$.

`tests/test_rk_order.py` pins these orders, so a regression fails as a wrong convergence
rate rather than as a slightly worse loss.

## The alternative: differentiating the interpolant

`residual="interpolant"` is the earlier formulation, kept for comparison. It builds a
polynomial $\hat{u}$ in time through the stage values, differentiates it analytically via
the barycentric differentiation matrix $D_{ij} = L_j'(t_i)$, and imposes
$u_t + \mathcal{L}u - f$ at the nodes.

It uses only $c$ and $b$, **ignoring $A$**. Every tableau therefore behaves identically,
and accuracy follows the degree of $\hat{u}$:

| stencil (`q_aux`) | $\deg\hat u$ | order |
| --- | --- | --- |
| `"same"` — stage nodes only | $q-1$ | $\mathcal{O}(k)$ |
| `"extend"` — plus the slab start | $q$ | $\mathcal{O}(k^2)$ |

For Gauss $q=2$ the RK form is roughly $10^5$ times more consistent at the same slab
size.

!!! warning "Consistency bounds accuracy; it does not predict training"
    Everything above measures truncation error, which bounds what a perfectly trained
    network could achieve. It says nothing about whether the optimiser gets there. In
    short single-seed runs the two residual forms trade places depending on tableau and
    step budget, and the loss trace is non-monotonic because the spatial sampler redraws
    each step. `residual` exists so the comparison can be made rather than assumed.

## Boundary and initial conditions

Boundary conditions are enforced **structurally**, not by penalty: the ansatz

$$
u_\theta(x,t) = \Phi(x)\,g_\theta(x,t), \qquad \Phi(x) = x(1-x)
$$

vanishes at $x\in\{0,1\}$ for any $g_\theta$. Boundary error is therefore identically
zero and never competes with the residual.

The initial condition is a penalty, matching $\partial_x u_\theta(\cdot,0)$ against
$\partial_x u_0$ in an $H^1$ seminorm. Note this constrains the *derivative*, not the
values. Its weight relative to the residual is not stable during training — on the
shipped example it is essentially the whole loss at initialisation and about a fifth of
it a few hundred steps later — so `RkPinnConfig.ic_weight` exists to control the balance.

## Known limitations

- Linear parabolic problems on 1D spatial domains, homogeneous Dirichlet conditions.
- The stage order caps accuracy at $\mathcal{O}(k^q)$; raising $q$ is the lever.
- The initial-condition penalty is an $H^1$ seminorm, so it does not pin $u(\cdot,0)$
  pointwise.

## References

- Raissi, Perdikaris & Karniadakis (2019), *Physics-informed neural networks*, JCP 378.
- Hairer & Wanner (1996), *Solving Ordinary Differential Equations II*, Springer.
- Butcher (2016), *Numerical Methods for Ordinary Differential Equations*, Wiley.
- Berrut & Trefethen (2004), *Barycentric Lagrange interpolation*, SIAM Review 46(3).
