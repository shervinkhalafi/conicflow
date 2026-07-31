"""
Receding-Horizon Path-Space Control for Flow Matching
=====================================================

Implements the softened path-space MPC variant with an INEQUALITY constraint
h(x) <= 0 enforced via a Lagrangian dual term and projected dual ascent:

    min_{x_{1:H}}  sum_{l=0}^{H-1} (1/(2*lam*dt)) * ||x_{l+1} - x_l - dt*v(t_{j+l}, x_l)||^2
                   - r(x_H) + <mu, h(z)>
    s.t. x_0 = x_cur

where z is either x_H (constraint_on="horizon") or the flow-map prediction of
the true endpoint X(t_{j+H}, 1; x_H) (constraint_on="endpoint").

After each inner solve, mu is updated via projected dual ascent:
    mu <- max(0, mu + lr_dual * h(z))

mu is a per-sample tensor of shape (B, C), one multiplier per particle per
constraint dimension. C == 1 for a scalar constraint.

You must supply:
    v_theta(t, x)      -> velocity,          shape (B, D)
    r(x)               -> terminal reward,   shape (B,)   (differentiable)
    h(x)               -> constraint tensor, shape (B, C) (optional; h(x) <= 0)
                          Use stack_constraints() to compose multiple scalar
                          constraint functions into one (B, C) callable.
    endpoint_map(t, x) -> flow-map to t=1,   shape (B, D) (required when
                          constraint_on="endpoint")
"""

import torch
from typing import Callable, List, Optional


# ── Helper ────────────────────────────────────────────────────────────────────

def stack_constraints(fns: List[Callable]) -> Callable:
    """Combine a list of scalar constraint functions into one (B, C) callable.

    Each function must accept x: (B, D) and return a 1-D tensor of shape (B,)
    or a column tensor of shape (B, 1).  The result is stacked column-wise so
    the combined output has shape (B, C) where C = len(fns).

    Example
    -------
    h = stack_constraints([
        lambda x: x[:, 0] - 3.0,            # x_0 <= 3
        lambda x: -reward(x) + threshold,   # reward >= threshold
    ])
    """
    def combined(x: torch.Tensor) -> torch.Tensor:
        cols = []
        for fn in fns:
            v = fn(x)
            if v.dim() == 1:
                v = v.unsqueeze(-1)   # (B,) -> (B, 1)
            cols.append(v)
        return torch.cat(cols, dim=-1)  # (B, C)
    return combined


# ── Core algorithm ────────────────────────────────────────────────────────────

@torch.no_grad()
def _rollout_uncontrolled(
    v_theta: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    t0_idx: int,
    H: int,
    dt: float,
    t_grid: torch.Tensor,
) -> torch.Tensor:
    """Warm start: roll the uncontrolled flow forward H steps.

    Returns x_bar of shape (H, B, D) — the states x_1, ..., x_H
    (x_0 is the fixed current state and is not included).
    """
    xs = []
    x = x0
    for ell in range(H):
        t = t_grid[t0_idx + ell]
        x = x + dt * v_theta(t, x)
        xs.append(x)
    return torch.stack(xs, dim=0)


def _horizon_objective(
    x_traj: torch.Tensor,               # (H, B, D)
    x_cur: torch.Tensor,                # (B, D), fixed
    v_theta: Callable,
    r: Callable,
    h: Optional[Callable],
    t0_idx: int,
    dt: float,
    lam: float,
    mu: Optional[torch.Tensor],         # (B, C) Lagrange multipliers, or None
    t_grid: torch.Tensor,
    constraint_on: str,                 # "horizon" | "endpoint"
    endpoint_map: Optional[Callable],
    t_horizon: Optional[torch.Tensor],  # scalar tensor for the horizon end-time
) -> torch.Tensor:
    """Lagrangian path-space objective, summed over the batch (scalar)."""
    H = x_traj.shape[0]
    x_full = torch.cat([x_cur.unsqueeze(0), x_traj], dim=0)  # (H+1, B, D)

    # Control-energy term
    energy = x_full.new_zeros(x_full.shape[1])  # (B,)
    for ell in range(H):
        t = t_grid[t0_idx + ell]
        resid = x_full[ell + 1] - x_full[ell] - dt * v_theta(t, x_full[ell])
        energy = energy + resid.pow(2).sum(dim=-1) / (2.0 * lam * dt)

    x_H = x_full[-1]
    obj = energy - r(x_H)  # (B,)

    # Inequality constraint term: <mu, h(z)>  (mu detached; grad flows through h)
    if h is not None and mu is not None:
        if constraint_on == "endpoint" and endpoint_map is not None:
            z = endpoint_map(t_horizon.expand(x_H.shape[0]), x_H)
        else:
            z = x_H
        hx = h(z)                           # (B, C)
        if hx.dim() == 1:
            hx = hx.unsqueeze(-1)           # (B,) -> (B, 1)
        obj = obj + (mu.detach() * hx).sum(dim=-1)

    return obj.sum()


def solve_horizon_problem(
    x_cur: torch.Tensor,
    v_theta: Callable,
    r: Callable,
    h: Optional[Callable],
    t0_idx: int,
    H: int,
    dt: float,
    lam: float,
    mu: Optional[torch.Tensor],
    t_grid: torch.Tensor,
    constraint_on: str,
    endpoint_map: Optional[Callable],
    optimizer: str = "lbfgs",
    n_iters: int = 50,
    lr: float = 0.1,
):
    """Solve the finite-horizon path-space problem.

    Returns:
        x1_star: (B, D)  first step of the optimal trajectory
        xH_star: (B, D)  last  step of the optimal trajectory (for dual update)
    """
    x_init = _rollout_uncontrolled(v_theta, x_cur, t0_idx, H, dt, t_grid)
    x_traj = x_init.clone().requires_grad_(True)
    x_cur_fixed = x_cur.detach()
    t_horizon = t_grid[t0_idx + H]

    kw = dict(
        x_cur=x_cur_fixed, v_theta=v_theta, r=r, h=h,
        t0_idx=t0_idx, dt=dt, lam=lam, mu=mu, t_grid=t_grid,
        constraint_on=constraint_on, endpoint_map=endpoint_map,
        t_horizon=t_horizon,
    )

    if optimizer.lower() == "lbfgs":
        opt = torch.optim.LBFGS(
            [x_traj], lr=lr, max_iter=n_iters,
            tolerance_grad=1e-8, tolerance_change=1e-10,
            line_search_fn="strong_wolfe",
        )
        def closure():
            opt.zero_grad()
            loss = _horizon_objective(x_traj, **kw)
            loss.backward()
            return loss
        opt.step(closure)
    else:  # adam
        opt = torch.optim.Adam([x_traj], lr=lr)
        for _ in range(n_iters):
            opt.zero_grad()
            _horizon_objective(x_traj, **kw).backward()
            opt.step()

    return x_traj[0].detach(), x_traj[-1].detach()  # x_1^*, x_H^*


def receding_horizon_flow_control(
    v_theta: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    r: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    N: int = 100,
    H: int = 10,
    lam: float = 0.1,
    mu_init: float = 0.0,
    h: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    constraint_on: str = "horizon",
    endpoint_map: Optional[Callable] = None,
    lr_dual: float = 0.0,
    optimizer: str = "lbfgs",
    n_iters: int = 50,
    lr: float = 0.1,
    verbose: bool = False,
    device: Optional[torch.device] = None,
):
    """
    Receding-Horizon Path-Space Control with inequality constraint h(x) <= 0.

    Args:
        v_theta:       velocity field  v_theta(t, x) -> (B, D).
        r:             terminal reward  r(x) -> (B,).
        x0:            initial samples from p_0, shape (B, D).
        N:             number of time steps (dt = 1/N).
        H:             planning horizon (clipped near t=1).
        lam:           control weight lambda > 0.
        mu_init:       initial scalar value for all Lagrange multipliers.
        h:             inequality constraint  h(x) -> (B, C),  h(x) <= 0.
                       For a single scalar constraint pass a function returning
                       (B,) or (B, 1).  For multiple constraints use
                       stack_constraints([fn1, fn2, ...]) to get (B, C).
                       Pass None to run unconstrained.
        constraint_on: where to evaluate h for the Lagrangian and dual update:
                       "horizon"  — evaluate at x_H  (default).
                       "endpoint" — evaluate at the flow-map prediction
                                    X(t_{j+H}, 1; x_H) of the true endpoint.
        endpoint_map:  Callable(t_batch, x) -> (B, D).  Required when
                       constraint_on="endpoint".
        lr_dual:       projected dual-ascent step size (>= 0).
                       0 keeps mu frozen at mu_init (fixed Lagrangian penalty).
        optimizer:     "lbfgs" or "adam" for the inner solve.
        n_iters:       inner-solver iterations per MPC step.
        lr:            inner-solver learning rate.
        verbose:       print per-step diagnostics.
        device:        torch device (defaults to x0.device).

    Returns:
        x_N:      final controlled samples,       shape (B, D).
        traj:     full controlled trajectory,     shape (N+1, B, D).
        controls: applied residual controls u_j*, shape (N, B, D).
        mu_final: final per-sample multipliers,   shape (B, C), or None.
    """
    if device is None:
        device = x0.device
    x0 = x0.to(device)
    dt = 1.0 / N
    t_grid = torch.linspace(0.0, 1.0, N + 1, device=device, dtype=x0.dtype)

    x = x0.detach().clone()
    traj = [x.clone()]
    controls = []

    # Initialise per-sample Lagrange multipliers mu_t: (B, C)
    mu_t = None
    if h is not None:
        with torch.no_grad():
            h0 = h(x0)
            if h0.dim() == 1:
                h0 = h0.unsqueeze(-1)
            C = h0.shape[-1]
        mu_t = torch.full(
            (x0.shape[0], C), float(mu_init), dtype=x0.dtype, device=device
        )

    for j in range(N):
        H_j = min(H, N - j)

        x1_star, xH_star = solve_horizon_problem(
            x, v_theta, r, h, t0_idx=j, H=H_j, dt=dt,
            lam=lam, mu=mu_t, t_grid=t_grid,
            constraint_on=constraint_on, endpoint_map=endpoint_map,
            optimizer=optimizer, n_iters=n_iters, lr=lr,
        )

        # ── Projected dual ascent ─────────────────────────────────────────────
        if h is not None and lr_dual > 0.0:
            with torch.no_grad():
                if constraint_on == "endpoint" and endpoint_map is not None:
                    t_h = t_grid[j + H_j]
                    z = endpoint_map(t_h.expand(x.shape[0]), xH_star)
                else:
                    z = xH_star
                hx = h(z)
                if hx.dim() == 1:
                    hx = hx.unsqueeze(-1)
                mu_t = (mu_t + lr_dual * hx).clamp_min(0.0)

        # ── Apply first MPC step ──────────────────────────────────────────────
        with torch.no_grad():
            v_j = v_theta(t_grid[j], x)
            u_star = (x1_star - x) / dt - v_j
            x = x + dt * (v_j + u_star)

        traj.append(x.clone())
        controls.append(u_star.clone())

        if verbose:
            mu_mean = mu_t.mean().item() if mu_t is not None else 0.0
            print(
                f"step {j+1:4d}/{N}  "
                f"|u*| mean = {u_star.norm(dim=-1).mean():.4e}  "
                f"mu mean = {mu_mean:.4e}"
            )

    return x, torch.stack(traj, dim=0), torch.stack(controls, dim=0), mu_t


# ─────────────────────────────────────────────────────────────────────────────
# Demo — multi-dimensional constraint example
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(0)
B, D = 256, 2
N_demo = 20

v_theta    = model.calc_b
mpc_device = next(model.parameters()).device

# Two inequality constraints composed into a single (B, 2) callable:
#   c1: x_0 <= -0.5   (first coordinate below -0.5)
#   c2: upper-moon reward >= 0.9
h = stack_constraints([
    lambda x: x[:, 0] - (-0.5),               # c1(x) <= 0
    lambda x: -upper_moon_reward(x) + 0.9,    # c2(x) <= 0
])

def r(x):
    return torch.zeros(x.shape[0], device=x.device)

# endpoint_map: model.forward(s, t, x) = X(s,t,x); use t=1 to predict x_N.
def endpoint_map(t_batch, x):
    ones = torch.ones_like(t_batch)
    return model(t_batch, ones, x)

x0 = torch.randn(B, D, device=mpc_device)

# Uncontrolled rollout for comparison.
dt_demo     = 1.0 / N_demo
t_grid_demo = torch.linspace(0.0, 1.0, N_demo + 1, device=mpc_device, dtype=x0.dtype)
with torch.no_grad():
    x_unc = x0.clone()
    traj_unc_list = [x_unc.clone()]
    for j in range(N_demo):
        x_unc = x_unc + dt_demo * v_theta(t_grid_demo[j], x_unc)
        traj_unc_list.append(x_unc.clone())
traj_unc = torch.stack(traj_unc_list, dim=0)  # (N+1, B, D)

xN, traj, controls, mu_final = receding_horizon_flow_control(
    v_theta, r, x0,
    N=N_demo, H=4,
    lam=2.0,
    mu_init=0.0,
    h=h,
    constraint_on="horizon",   # switch to "endpoint" to use X(x_H)
    endpoint_map=endpoint_map,
    lr_dual=1.0,
    optimizer="lbfgs", n_iters=30, lr=0.5,
    verbose=True, device=mpc_device,
)

print("\nfinal mean (controlled):  ", xN.mean(dim=0).tolist())
print("final mean (uncontrolled):", x_unc.mean(dim=0).tolist())
if mu_final is not None:
    # mu_final: (B, C) — report per-constraint stats
    for c in range(mu_final.shape[1]):
        col = mu_final[:, c]
        print(f"  mu[{c}]  mean={col.mean().item():.4f}  max={col.max().item():.4f}")

# ── MPC visualisation ────────────────────────────────────────────────────────
mpc_metrics = visualize_control_results(
    traj_ctrl=traj,
    traj_unc=traj_unc,
    x0=x0,
    xN_ctrl=xN,
    xN_unc=x_unc,
    controls=controls,
    r=r,
    bg_pts=sample_target(4000),
    method_name="MPC",
    controlled_label="Controlled (MPC)",
    lam=2.0,
    controls_are_velocity=True,
    n_particles=256,
    show_snaps=True,
    snap_times=[0.0, 0.25, 0.5, 0.75, 1.0],
    lim=3.5,
    device=mpc_device,
)
