"""
Constrained Control for Flow Matching
=====================================

Two samplers live here:

1. ``receding_horizon_flow_control`` — softened path-space MPC (soft constraints
   via a Lagrangian dual term).
2. ``hardflow_sample`` — HardFlow, which at each step solves a terminal subproblem
   on the posterior-mean estimate.  With ``h`` supplied it enforces h(x) <= 0 exactly
   via augmented Lagrangian projection; with ``h=None`` it reduces to unconstrained
   proximal reward steering (minimise C(x) + weight||x - x_ref||^2).

Receding-Horizon Path-Space Control
-----------------------------------

Implements the softened path-space MPC variant with an INEQUALITY constraint
h(x) <= 0 enforced via a Lagrangian dual term and projected dual ascent:

    min_{x_{1:H}}  sum_{l=0}^{H-1} (1/(2*lam*dt)) * ||x_{l+1} - x_l - dt*v(t_{j+l}, x_l)||^2
                   - r(z) + <mu, h(z)>
    s.t. x_0 = x_cur

where z is the evaluation point for BOTH the reward and the constraint:

    constraint_on="horizon"  -> z = x_H
        Reward and constraint are evaluated at the planning horizon state.
        No extra network pass beyond the trajectory rollout.

    constraint_on="endpoint" -> z = X(t_{j+H}, 1; x_H)  via endpoint_map
        Reward and constraint are both evaluated at the flow-map prediction of
        the true endpoint t=1.  This adds a backprop through endpoint_map on
        every inner-solver step (even when h is None / mu=0).

After each inner solve, mu is updated via projected dual ascent:
    mu <- max(0, mu + lr_dual * h(z))

mu is a per-sample tensor of shape (B, C), one multiplier per particle per
constraint dimension.

You must supply:
    v_theta(t, x)      -> velocity,          shape (B, D)
    r(x)               -> terminal reward,   shape (B,)   (differentiable)
    h(x)               -> constraint vector, shape (B, C) (optional; h(x) <= 0)
    endpoint_map(t, x) -> flow-map to t=1,   shape (B, D) (required when
                          constraint_on="endpoint")
"""

import torch
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


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
    x_traj: torch.Tensor,                  # (H, B, D)
    x_cur: torch.Tensor,                   # (B, D), fixed
    v_theta: Callable,
    r: Callable,
    h: Optional[Callable],
    t0_idx: int,
    dt: float,
    lam: float,
    mu: Optional[torch.Tensor],            # (B, C) Lagrange multipliers, or None
    t_grid: torch.Tensor,
    constraint_on: str,                    # "horizon" | "endpoint"
    endpoint_map: Optional[Callable],
    t_horizon: Optional[torch.Tensor],     # scalar tensor for the horizon end-time
    penalty: str = "dual",                 # "dual" | "alm"
    rho: Optional[float] = None,           # ALM penalty parameter (penalty="alm")
    h_scale: Optional[torch.Tensor] = None,  # (C,) per-constraint normaliser
) -> torch.Tensor:
    """Lagrangian path-space objective, summed over the batch (scalar).

    Both the reward r(z) and the constraint term are evaluated at the same
    point z, determined by constraint_on:
        "horizon"  -> z = x_H         (no extra network call)
        "endpoint" -> z = endpoint_map(t_horizon, x_H)  (backprop through flow map)

    The constraint term itself depends on ``penalty``:
        "dual" — plain Lagrangian  <mu, h(z)>  (linear in h).
        "alm"  — Powell-Hestenes-Rockafellar augmented Lagrangian
                 (||max(0, mu + rho*h)||^2 - ||mu||^2) / (2*rho), the same
                 form HardFlow's terminal subproblem uses. The quadratic term
                 grows with rho, so feasibility is enforced far more strongly
                 (and far less sensitively to multiplier tuning) than the
                 linear dual term.
    """
    H = x_traj.shape[0]
    x_full = torch.cat([x_cur.unsqueeze(0), x_traj], dim=0)  # (H+1, B, D)

    # Control-energy term
    energy = x_full.new_zeros(x_full.shape[1])  # (B,)
    for ell in range(H):
        t = t_grid[t0_idx + ell]
        resid = x_full[ell + 1] - x_full[ell] - dt * v_theta(t, x_full[ell])
        energy = energy + resid.pow(2).sum(dim=-1) / (2.0 * lam * dt)

    x_H = x_full[-1]

    # Evaluation point for both r and h.
    if constraint_on == "endpoint" and endpoint_map is not None:
        z = endpoint_map(t_horizon.expand(x_H.shape[0]), x_H)
    else:
        z = x_H

    obj = energy - r(z)  # (B,)

    # Inequality constraint term.
    if h is not None and mu is not None:
        hx = h(z)                               # (B, C)
        if hx.dim() == 1:
            hx = hx.unsqueeze(-1)
        if h_scale is not None:
            # Work with the normalised constraint h_c / s_c throughout, so the
            # whole Lagrangian is invariant to per-constraint rescaling.
            hx = hx / h_scale
        mu_d = mu.detach()
        if penalty == "alm" and rho is not None:
            slack = torch.clamp(mu_d + rho * hx, min=0.0)
            obj = obj + (slack.pow(2) - mu_d.pow(2)).sum(dim=-1) / (2.0 * rho)
        else:
            obj = obj + (mu_d * hx).sum(dim=-1)

    return obj.sum()


def _shift_warm_start(
    x_traj_prev: torch.Tensor,
    H_new: int,
    v_theta: Callable,
    t0_idx_new: int,
    dt: float,
    t_grid: torch.Tensor,
) -> torch.Tensor:
    """Shift the previous MPC step's converged trajectory by one index to seed
    a warm start for the new step's horizon.

    ``x_traj_prev`` is the (H_prev, B, D) trajectory just solved for the step
    that already had its first state applied — dropping that first state
    (index 0) gives the remaining ``H_prev - 1`` states, which line up with
    the new step's horizon starting one index later. If the new horizon is
    longer (the usual case, when it hasn't started shrinking near t=1), the
    tail is extended by rolling the uncontrolled flow forward from the last
    known state.

    Returns:
        (H_new, B, D) warm-start trajectory.
    """
    shifted = x_traj_prev[1:]
    n_have = shifted.shape[0]
    if n_have >= H_new:
        return shifted[:H_new].clone()

    n_extra = H_new - n_have
    x = shifted[-1] if n_have > 0 else x_traj_prev[0]
    tail = []
    with torch.no_grad():
        for k in range(n_extra):
            t = t_grid[t0_idx_new + n_have + k]
            x = x + dt * v_theta(t, x)
            tail.append(x.clone())
    tail = torch.stack(tail, dim=0)
    return torch.cat([shifted, tail], dim=0) if n_have > 0 else tail


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
    x_init: Optional[torch.Tensor] = None,
    penalty: str = "dual",
    rho: Optional[float] = None,
    h_scale: Optional[torch.Tensor] = None,
):
    """Solve the finite-horizon path-space problem for a fixed mu.

    Args:
        x_init: optional (H, B, D) warm-start trajectory (e.g. the previous
            primal-dual round's solution, when called from a multi-step outer
            loop). Defaults to an uncontrolled rollout from x_cur.
        penalty: "dual" (plain Lagrangian) or "alm" (augmented Lagrangian).
        rho: ALM penalty parameter; required when penalty="alm".

    Returns:
        x1_star:  (B, D)     first step of the optimal trajectory
        xH_star:  (B, D)     last  step of the optimal trajectory (for dual update)
        x_traj:   (H, B, D)  full optimal trajectory (for warm-starting the next
                  primal solve when mu is updated and re-solved in place)
    """
    if x_init is None:
        x_init = _rollout_uncontrolled(v_theta, x_cur, t0_idx, H, dt, t_grid)
    x_traj = x_init.clone().requires_grad_(True)
    x_cur_fixed = x_cur.detach()
    t_horizon = t_grid[t0_idx + H]

    kw = dict(
        x_cur=x_cur_fixed, v_theta=v_theta, r=r, h=h,
        t0_idx=t0_idx, dt=dt, lam=lam, mu=mu, t_grid=t_grid,
        constraint_on=constraint_on, endpoint_map=endpoint_map,
        t_horizon=t_horizon, penalty=penalty, rho=rho, h_scale=h_scale,
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

    return x_traj[0].detach(), x_traj[-1].detach(), x_traj.detach()  # x_1^*, x_H^*, full traj


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
    n_dual_steps: int = 1,
    warm_start_across_steps: bool = False,
    commit_final_window: bool = False,
    final_window_dual_mult: int = 1,
    constraint_penalty: str = "dual",
    rho: float = 1.0,
    rho_scale: float = 2.0,
    dual_adaptive: bool = False,
    dual_beta: float = 0.9,
    dual_eps: float = 1e-8,
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
                       Pass None to run unconstrained.
        constraint_on: evaluation point for BOTH r(.) and h(.) in the
                       Lagrangian:
                       "horizon"  — use x_H (default).  No extra network pass.
                       "endpoint" — use endpoint_map(t_{j+H}, x_H), the
                                    flow-map prediction of the true endpoint
                                    t=1.  Adds a backprop through endpoint_map
                                    on every inner-solver step, even when h is
                                    None or mu=0.
        endpoint_map:  Callable(t_batch, x) -> (B, D).  Required (and must be
                       differentiable) when constraint_on="endpoint".
        lr_dual:       projected dual-ascent step size (>= 0), used only when
                       constraint_penalty="dual".
                       0 keeps mu frozen at mu_init (fixed Lagrangian penalty).
        optimizer:     "lbfgs" or "adam" for the inner solve.
        n_iters:       inner-solver iterations per primal-dual round.
        lr:            inner-solver learning rate.
        n_dual_steps:  number of primal-dual rounds solved *in place* on the
                       same horizon subproblem before applying the first MPC
                       step, mirroring HardFlow's augmented-Lagrangian outer
                       loop: solve the primal for `n_iters` iterations at the
                       current mu, update mu via projected dual ascent, then
                       re-solve the (warm-started) primal at the new mu, and
                       repeat. 1 (default) reproduces the original single
                       primal-solve-then-dual-update-once-per-step behaviour.
                       Only the last round's h(.) / mu are recorded into
                       h_history / mu_history, so their shapes are unchanged.
        warm_start_across_steps: if True, seed each MPC step's first
                       primal-dual round from the *previous* step's converged
                       trajectory, shifted by one index (dropping the just-
                       applied first state) instead of a fresh uncontrolled
                       rollout. False (default) reproduces the original
                       from-scratch-every-step behaviour.
        commit_final_window: if True, at the first step j where the horizon
                       reaches exactly t=1 (j = max(0, N - H), the largest
                       window whose far edge is the true endpoint), solve
                       that window once and commit to its ENTIRE resulting
                       trajectory instead of applying only the first step and
                       continuing to re-plan with a shrinking horizon. At
                       that window x_H corresponds to t=1 exactly, and
                       endpoint_map(t=1, x_H) == x_H by construction (zero
                       time delta), so under BOTH constraint_on modes the
                       evaluated point z is provably x_N itself rather than a
                       lookahead prediction — closing the surrogate-vs-
                       realized gap for this final stretch the same way
                       HardFlow's terminal correction does. False (default)
                       keeps receding all the way to the end as before.
        final_window_dual_mult: multiplier k applied to n_dual_steps only for
                       the committed final window (ignored if
                       commit_final_window is False): that window is solved
                       with n_dual_steps * k primal-dual rounds. k=1
                       (default) uses the same round count as every other
                       step; k>1 spends extra rounds specifically on this
                       one true terminal correction.
        constraint_penalty: how h(x) <= 0 is enforced inside each horizon
                       subproblem.
                       "dual" (default) — plain Lagrangian term <mu, h(z)>
                            with projected dual ascent
                            mu <- max(0, mu + lr_dual * h(z)) between
                            primal-dual rounds. mu persists across MPC steps.
                            Convergence depends entirely on lr_dual being
                            well tuned, and can overshoot/oscillate when many
                            rounds are run.
                       "alm" — Powell-Hestenes-Rockafellar augmented
                            Lagrangian, the same mechanism HardFlow's
                            terminal subproblem uses: the quadratic penalty
                            (||max(0, mu + rho*h)||^2 - ||mu||^2) / (2*rho)
                            with multiplier update
                            mu <- max(0, mu + rho * h(z)) and a geometrically
                            growing rho between rounds. mu and rho are RESET
                            at the start of every MPC step (mu to mu_init,
                            rho to `rho`), making each step's constrained
                            subproblem self-contained — mirroring HardFlow,
                            and avoiding the blow-up that a persistent mu
                            would cause once rho has grown large. lr_dual is
                            ignored in this mode.
        rho:           initial ALM penalty parameter (constraint_penalty="alm").
        rho_scale:     multiplicative growth of rho per primal-dual round
                       (constraint_penalty="alm").
        dual_adaptive: RMSProp-normalise the projected dual-ascent step,
                       per constraint (constraint_penalty="dual" only):

                           v_c   <- beta*v_c + (1-beta)*h_c^2
                           mu_c  <- max(0, mu_c + lr_dual * h_c / (sqrt(v_c) + eps))

                       (with Adam-style bias correction on v). Plain dual
                       ascent grows mu_c at a rate proportional to |h_c|, so
                       under a shared lr_dual a constraint whose typical
                       violation is small gets proportionally less dual
                       pressure and stalls, while a large-violation
                       constraint saturates. The normalised step is exactly
                       invariant to per-constraint rescaling h_c -> k_c*h_c,
                       so every constraint makes comparable progress without
                       any offline calibration or knowledge of the thresholds.
        dual_beta:     EMA decay for the dual second-moment estimate.
        dual_eps:      numerical floor in the normalised dual step.
        verbose:       print per-step diagnostics.
        device:        torch device (defaults to x0.device).

    Returns:
        x_N:        final controlled samples,               shape (B, D).
        traj:       full controlled trajectory,             shape (N+1, B, D).
        controls:   applied residual controls u_j*,         shape (N, B, D).
        mu_final:   final per-sample multipliers,           shape (B, C), or None.
        mu_history: per-step multipliers after dual update, shape (N, B, C), or None.
        h_history:  per-step constraint values h(x_H*),    shape (N, B, C), or None.
    """
    if device is None:
        device = x0.device
    x0 = x0.to(device)
    dt = 1.0 / N
    t_grid = torch.linspace(0.0, 1.0, N + 1, device=device, dtype=x0.dtype)

    x = x0.detach().clone()
    traj = [x.clone()]
    controls = []
    mu_history = []
    h_history = []

    mu_t = None
    if h is not None:
        with torch.no_grad():
            h0 = h(x0)
            C = h0.shape[-1] if h0.dim() > 1 else 1
        mu_t = torch.full(
            (x0.shape[0], C), float(mu_init), dtype=x0.dtype, device=device
        )

    prev_traj = None   # previous MPC step's converged trajectory, for warm_start_across_steps
    j_commit = max(0, N - H) if commit_final_window else None

    # Running per-constraint scale s_c for dual_adaptive. The objective and the
    # dual step both use h_c / s_c, so the whole Lagrangian — and hence the
    # trajectory — is invariant to h_c -> k_c * h_c.
    _dual_state = {"v": None, "t": 0}

    def _h_scale() -> Optional[torch.Tensor]:
        """Current (C,) normaliser, or None before the first observation."""
        if not dual_adaptive or _dual_state["v"] is None:
            return None
        v_hat = _dual_state["v"] / (1.0 - dual_beta ** _dual_state["t"])
        return v_hat.sqrt() + dual_eps

    def _dual_update(mu: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        """Projected dual ascent, on the scale-normalised constraint if enabled."""
        if not dual_adaptive:
            return (mu + lr_dual * hx).clamp_min(0.0)
        # Update the per-constraint scale from this step's observations first.
        v_obs = hx.pow(2).mean(dim=0)                     # (C,)
        if _dual_state["v"] is None:
            _dual_state["v"] = torch.zeros_like(v_obs)
        _dual_state["t"] += 1
        _dual_state["v"] = dual_beta * _dual_state["v"] + (1.0 - dual_beta) * v_obs
        return (mu + lr_dual * hx / _h_scale()).clamp_min(0.0)

    for j in range(N):
        H_j = min(H, N - j)

        # ── Final-window commit: this window's far edge is exactly t=1, so
        # z (under either constraint_on mode) is provably x_N, not a
        # lookahead surrogate. Solve once (optionally with more rounds via
        # final_window_dual_mult) and apply the WHOLE resulting trajectory,
        # then stop receding — there is nothing left to re-plan. ────────────
        if commit_final_window and j == j_commit:
            n_rounds = (n_dual_steps * final_window_dual_mult) if h is not None else 1
            if warm_start_across_steps and prev_traj is not None:
                x_warm = _shift_warm_start(prev_traj, H_j, v_theta, j, dt, t_grid)
            else:
                x_warm = None
            hx = None
            rho_j = rho
            if constraint_penalty == "alm" and mu_t is not None:
                mu_t = torch.full_like(mu_t, float(mu_init))
            for pd_round in range(n_rounds):
                x1_star, xH_star, x_warm = solve_horizon_problem(
                    x, v_theta, r, h, t0_idx=j, H=H_j, dt=dt,
                    lam=lam, mu=mu_t, t_grid=t_grid,
                    constraint_on=constraint_on, endpoint_map=endpoint_map,
                    optimizer=optimizer, n_iters=n_iters, lr=lr,
                    x_init=x_warm,
                    penalty=constraint_penalty, rho=rho_j, h_scale=_h_scale(),
                )
                if h is None:
                    continue
                with torch.no_grad():
                    if constraint_on == "endpoint" and endpoint_map is not None:
                        t_h = t_grid[j + H_j]
                        z = endpoint_map(t_h.expand(x.shape[0]), xH_star)
                    else:
                        z = xH_star
                    hx = h(z)
                    if hx.dim() == 1:
                        hx = hx.unsqueeze(-1)
                    if constraint_penalty == "alm":
                        mu_t = torch.clamp(mu_t + rho_j * hx, min=0.0)
                        rho_j *= rho_scale
                    elif lr_dual > 0.0:
                        mu_t = _dual_update(mu_t, hx)

            full_traj = x_warm   # (H_j, B, D); full_traj[-1] == xH_star == x_N exactly
            with torch.no_grad():
                prev = x
                for i in range(H_j):
                    v_i = v_theta(t_grid[j + i], prev)
                    u_i = (full_traj[i] - prev) / dt - v_i
                    controls.append(u_i.clone())
                    prev = full_traj[i]
                    traj.append(prev.clone())
                    if h is not None:
                        h_history.append(hx.clone())
                    if mu_t is not None:
                        mu_history.append(mu_t.clone())
                x = prev

            if verbose:
                mu_mean = mu_t.mean().item() if mu_t is not None else 0.0
                hx_max = hx.max().item() if hx is not None else float("nan")
                print(
                    f"steps {j+1:4d}-{N:<4d}  (final commit, {n_rounds} rounds)  "
                    f"mu mean = {mu_mean:.4e}  max h = {hx_max:+.4e}"
                )
            break

        # ── Primal-dual loop: re-solve the SAME horizon subproblem up to
        # n_dual_steps times, updating mu via projected dual ascent between
        # rounds and warm-starting each primal re-solve from the previous
        # round's trajectory (mirrors HardFlow's augmented-Lagrangian outer
        # loop, but applied to this fixed-horizon subproblem). Only the last
        # round's h(.) is recorded, so h_history/mu_history keep one entry
        # per MPC step regardless of n_dual_steps. With h=None there is
        # nothing to update between rounds, so only one primal solve runs. ──
        if warm_start_across_steps and prev_traj is not None:
            x_warm = _shift_warm_start(prev_traj, H_j, v_theta, j, dt, t_grid)
        else:
            x_warm = None
        n_rounds = n_dual_steps if h is not None else 1
        rho_j = rho
        if constraint_penalty == "alm" and mu_t is not None:
            mu_t = torch.full_like(mu_t, float(mu_init))
        for pd_round in range(n_rounds):
            x1_star, xH_star, x_warm = solve_horizon_problem(
                x, v_theta, r, h, t0_idx=j, H=H_j, dt=dt,
                lam=lam, mu=mu_t, t_grid=t_grid,
                constraint_on=constraint_on, endpoint_map=endpoint_map,
                optimizer=optimizer, n_iters=n_iters, lr=lr,
                x_init=x_warm,
                penalty=constraint_penalty, rho=rho_j, h_scale=_h_scale(),
            )

            if h is None:
                continue

            with torch.no_grad():
                if constraint_on == "endpoint" and endpoint_map is not None:
                    t_h = t_grid[j + H_j]
                    z = endpoint_map(t_h.expand(x.shape[0]), xH_star)
                else:
                    z = xH_star
                hx = h(z)
                if hx.dim() == 1:
                    hx = hx.unsqueeze(-1)
                if constraint_penalty == "alm":
                    mu_t = torch.clamp(mu_t + rho_j * hx, min=0.0)
                    rho_j *= rho_scale
                elif lr_dual > 0.0:
                    mu_t = _dual_update(mu_t, hx)

        if h is not None:
            h_history.append(hx.clone())

        if mu_t is not None:
            mu_history.append(mu_t.clone())

        prev_traj = x_warm   # this step's converged (H_j, B, D) trajectory

        # ── Apply first MPC step ──────────────────────────────────────────────
        with torch.no_grad():
            v_j = v_theta(t_grid[j], x)
            u_star = (x1_star - x) / dt - v_j
            x = x + dt * (v_j + u_star)

        traj.append(x.clone())
        controls.append(u_star.clone())

        if verbose:
            mu_mean = mu_t.mean().item() if mu_t is not None else 0.0
            hx_max  = h_history[-1].max().item() if h_history else float("nan")
            print(
                f"step {j+1:4d}/{N}  "
                f"|u*| mean = {u_star.norm(dim=-1).mean():.4e}  "
                f"mu mean = {mu_mean:.4e}  "
                f"max h = {hx_max:+.4e}"
            )

    mu_hist = torch.stack(mu_history, dim=0) if mu_history else None
    h_hist  = torch.stack(h_history,  dim=0) if h_history  else None
    return x, torch.stack(traj, dim=0), torch.stack(controls, dim=0), mu_t, mu_hist, h_hist


# ─────────────────────────────────────────────────────────────────────────────
# HardFlow: hard-constrained sampling for flow-matching models
# ─────────────────────────────────────────────────────────────────────────────


def _autograd_dot(fn: Callable, t: torch.Tensor) -> torch.Tensor:
    """d/dt fn(t) by autograd, for a scheduler that only supplies alpha/beta."""
    t_ = t.detach().clone().requires_grad_(True)
    return torch.autograd.grad(fn(t_).sum(), t_)[0]


@dataclass
class AffineScheduler:
    """Affine interpolant  x_t = alpha_t * x_1 + beta_t * x_0.

    ``alpha_dot`` / ``beta_dot`` are optional; when omitted they are obtained by
    autograd, so any differentiable alpha/beta pair works.
    """

    alpha: Callable[[torch.Tensor], torch.Tensor]
    beta: Callable[[torch.Tensor], torch.Tensor]
    alpha_dot: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    beta_dot: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def coeffs(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (alpha_t, beta_t, alpha_dot_t, beta_dot_t)."""
        a = self.alpha(t)
        b = self.beta(t)
        ad = self.alpha_dot(t) if self.alpha_dot is not None else _autograd_dot(self.alpha, t)
        bd = self.beta_dot(t) if self.beta_dot is not None else _autograd_dot(self.beta, t)
        return a, b, ad, bd


# CondOT / linear path: x_t = t * x_1 + (1 - t) * x_0.
LINEAR_SCHEDULER = AffineScheduler(
    alpha=lambda t: t,
    beta=lambda t: 1.0 - t,
    alpha_dot=lambda t: torch.ones_like(t),
    beta_dot=lambda t: -torch.ones_like(t),
)


def _solve_proximal_subproblem(
    x_ref: torch.Tensor,
    C: Optional[Callable[[torch.Tensor], torch.Tensor]],
    weight: torch.Tensor,
    *,
    optimizer: str = "lbfgs",
    n_iters: int = 100,
    lr: float = 0.1,
    x_init: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Solve  min_x  C(x) + weight * ||x - x_ref||^2  (no constraints).

    Used by HardFlow when ``h=None`` — reward-only proximal steering toward the
    flow's posterior-mean terminal estimate.
    """
    x_ref = x_ref.detach()
    start = x_ref if x_init is None else x_init.detach()
    x = start.clone().requires_grad_(True)

    def _objective() -> torch.Tensor:
        obj = weight * (x - x_ref).pow(2).sum(dim=-1)
        if C is not None:
            obj = obj + C(x)
        return obj.sum()

    if optimizer.lower() == "lbfgs":
        opt = torch.optim.LBFGS(
            [x], lr=lr, max_iter=n_iters,
            tolerance_grad=1e-7, tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )
        def closure():
            opt.zero_grad()
            loss = _objective()
            loss.backward()
            return loss
        opt.step(closure)
    else:
        opt = torch.optim.Adam([x], lr=lr)
        for _ in range(n_iters):
            opt.zero_grad()
            _objective().backward()
            opt.step()

    return x.detach()


def solve_constrained_subproblem(
    x_ref: torch.Tensor,
    C: Optional[Callable[[torch.Tensor], torch.Tensor]],
    h: Optional[Callable[[torch.Tensor], torch.Tensor]],
    weight: torch.Tensor,
    *,
    optimizer: str = "lbfgs",
    n_outer: int = 5,
    n_inner: int = 20,
    lr: float = 0.1,
    rho: float = 10.0,
    rho_scale: float = 2.0,
    x_init: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Solve  min_x  C(x) + weight * ||x - x_ref||^2   s.t.  h(x) <= 0.

    When ``h is None`` the constraint is dropped and the problem reduces to
    unconstrained proximal optimisation (see ``_solve_proximal_subproblem``).

    Uses a Powell-Hestenes-Rockafellar augmented Lagrangian with projected dual
    ascent on the multipliers and a geometrically increasing penalty rho.

    Args:
        x_ref:      (B, D) unconstrained target (the posterior-mean estimate).
        C:          cost  C(x) -> (B,);  None for a pure projection / no reward term.
        h:          constraint  h(x) -> (B, C) with  h(x) <= 0;  None to skip
                    constraints entirely (unconstrained proximal mode).
        weight:     scalar (or broadcastable) proximal weight.
        optimizer:  "lbfgs" (default, recommended for nonlinear constraints) or
                    "adam".  LBFGS uses line search and retains curvature info
                    across inner iterations; Adam can struggle when the penalty
                    gradient overwhelms the proximal term at high rho.
        n_outer:    dual/penalty updates (outer ALM iterations).
        n_inner:    inner solver steps per outer iteration.
                    For "lbfgs": max_iter passed to LBFGS (each may call
                    the closure multiple times for line search).
                    For "adam": number of Adam gradient steps.
        lr:         inner solver learning rate / step size.
        rho:        initial penalty parameter.
        rho_scale:  multiplicative growth of rho per outer iteration.
        x_init:     (B, D) warm start; defaults to x_ref.

    Returns:
        x_star: (B, D) detached solution.
    """
    if h is None:
        return _solve_proximal_subproblem(
            x_ref, C, weight,
            optimizer=optimizer,
            n_iters=n_outer * n_inner,
            lr=lr,
            x_init=x_init,
        )

    x_ref = x_ref.detach()
    start = x_ref if x_init is None else x_init.detach()
    x = start.clone().requires_grad_(True)

    with torch.no_grad():
        h0 = h(x_ref)
        n_con = h0.shape[-1] if h0.dim() > 1 else 1
    mu = x_ref.new_zeros(x_ref.shape[0], n_con)

    def _objective() -> torch.Tensor:
        obj = weight * (x - x_ref).pow(2).sum(dim=-1)
        if C is not None:
            obj = obj + C(x)
        hx = h(x)
        if hx.dim() == 1:
            hx = hx.unsqueeze(-1)
        slack = torch.clamp(mu + rho * hx, min=0.0)
        obj = obj + (slack.pow(2) - mu.pow(2)).sum(dim=-1) / (2.0 * rho)
        return obj.sum()

    for _ in range(n_outer):
        if optimizer.lower() == "lbfgs":
            opt = torch.optim.LBFGS(
                [x], lr=lr, max_iter=n_inner,
                tolerance_grad=1e-7, tolerance_change=1e-9,
                line_search_fn="strong_wolfe",
            )
            def closure():
                opt.zero_grad()
                loss = _objective()
                loss.backward()
                return loss
            opt.step(closure)
        else:
            opt = torch.optim.Adam([x], lr=lr)
            for _ in range(n_inner):
                opt.zero_grad()
                _objective().backward()
                opt.step()

        with torch.no_grad():
            hx = h(x)
            if hx.dim() == 1:
                hx = hx.unsqueeze(-1)
            mu = torch.clamp(mu + rho * hx, min=0.0)
            rho *= rho_scale

    return x.detach()


def hardflow_sample(
    v_theta: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    h: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    C: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    N: int = 100,
    lam: float = 1.0,
    scheduler: Optional[AffineScheduler] = None,
    t_grid: Optional[torch.Tensor] = None,
    solver: Callable[..., torch.Tensor] = solve_constrained_subproblem,
    solver_kwargs: Optional[dict] = None,
    warm_start: bool = True,
    verbose: bool = False,
    device: Optional[torch.device] = None,
):
    """HardFlow: flow-guided sampling with a terminal subproblem at each step.

    At each Euler step the uncorrected state x_bar_{i+1} is inverted through the
    affine interpolant to a posterior-mean terminal estimate x_bar_N, that
    estimate is corrected by a terminal subproblem (trading off the cost C), and
    the corrected terminal state is mapped back to x_{i+1} while reusing the
    implied noise estimate:

        Delta_i     = t_{i+1} - t_i
        x_bar_{i+1} = x_i + Delta_i * v(t_i, x_i)
        x_bar_N     = (bd * x_bar_{i+1} - b * v(t_{i+1}, x_bar_{i+1})) / (a*bd - ad*b)
        x_hat_N*    = argmin C(x) + a^2 / (2 Delta_i * lam) * ||x - x_bar_N||^2
                      [s.t. h(x) <= 0   when h is supplied]
        eps_hat     = (-ad * x_bar_{i+1} + a * v(t_{i+1}, x_bar_{i+1})) / (a*bd - ad*b)
        x_{i+1}     = a * x_hat_N* + b * eps_hat

    with (a, b, ad, bd) = (alpha, beta, alpha_dot, beta_dot) evaluated at t_{i+1}.
    When ``h`` is supplied and alpha_1 = 1, beta_1 = 0, the returned sample
    equals the last subproblem solution and satisfies h(x_N) <= 0.
    When ``h is None``, the subproblem is unconstrained proximal steering — pass
    ``C=lambda x: -r(x)`` to maximise a reward r.

    Args:
        v_theta:       velocity field  v_theta(t, x) -> (B, D).
        x0:            initial samples from p_0, shape (B, D).
        h:             inequality constraint  h(x) -> (B, C),  h(x) <= 0.
                       Pass None for unconstrained reward-only mode (requires C).
        C:             optional terminal cost  C(x) -> (B,); None for a pure
                       feasibility projection (only meaningful when h is set).
                       To maximise a reward r, pass ``C=lambda x: -r(x)``.
        N:             number of discretization steps (ignored if t_grid given).
        lam:           control weight lambda > 0, same convention as
                       ``receding_horizon_flow_control`` and ``fmrg_j``: larger
                       values give the solver more freedom to deviate from the
                       uncontrolled flow (higher reward, more control effort).
                       Internally lam_oc = 1 / lam, so small lam keeps the
                       sample close to the unconstrained flow.
        scheduler:     AffineScheduler; defaults to LINEAR_SCHEDULER (CondOT).
        t_grid:        optional (N+1,) time grid; defaults to linspace(0, 1, N+1).
        solver:        solve_constrained_subproblem-compatible callable
                       ``solver(x_ref, C, h, weight, **solver_kwargs)``.
        solver_kwargs: extra keyword arguments forwarded to ``solver``.
        warm_start:    pass the previous subproblem solution to the solver.
        verbose:       print per-step diagnostics.
        device:        torch device (defaults to x0.device).

    Returns:
        x_N:      final constrained samples,        shape (B, D).
        traj:     full trajectory,                  shape (N+1, B, D).
        controls: correction velocities             shape (N, B, D),
                  u_i = (x_{i+1} - x_bar_{i+1}) / Delta_i.
        h_hist:   constraint values at x_hat_N*, shape (N, B, C), or None if h
                  was not supplied.
    """
    if device is None:
        device = x0.device
    if scheduler is None:
        scheduler = LINEAR_SCHEDULER
    if solver_kwargs is None:
        solver_kwargs = {}

    x0 = x0.to(device)
    if t_grid is None:
        t_grid = torch.linspace(0.0, 1.0, N + 1, device=device, dtype=x0.dtype)
    else:
        t_grid = t_grid.to(device=device, dtype=x0.dtype)
    N = t_grid.shape[0] - 1

    x = x0.detach().clone()
    traj = [x.clone()]
    controls = []
    h_hist = []
    xN_prev = None

    for i in range(N):
        dt = (t_grid[i + 1] - t_grid[i]).item()
        t_next = t_grid[i + 1]

        with torch.no_grad():
            x_bar_next = x + dt * v_theta(t_grid[i], x)
            v_next = v_theta(t_next, x_bar_next)

        a, b, ad, bd = scheduler.coeffs(t_next)
        denom = a * bd - ad * b
        # A degenerate scheduler can send denom to 0; clamp its magnitude, keep its sign.
        sign = torch.where(denom < 0, -torch.ones_like(denom), torch.ones_like(denom))
        denom = torch.where(denom.abs() < 1e-8, sign * 1e-8, denom)

        # Posterior-mean terminal estimate and the implied noise estimate.
        x_bar_N = (bd * x_bar_next - b * v_next) / denom
        eps_hat = (-ad * x_bar_next + a * v_next) / denom

        weight = a.pow(2) / (2.0 * dt * lam)
        xN_star = solver(
            x_bar_N,
            C,
            h,
            weight,
            x_init=xN_prev if warm_start else None,
            **solver_kwargs,
        )
        if warm_start:
            xN_prev = xN_star

        with torch.no_grad():
            x = a * xN_star + b * eps_hat
            u_i = (x - x_bar_next) / dt
            if h is not None:
                hx = h(xN_star)
                if hx.dim() == 1:
                    hx = hx.unsqueeze(-1)
                h_hist.append(hx.clone())

        traj.append(x.clone())
        controls.append(u_i.clone())

        if verbose:
            msg = (
                f"step {i+1:4d}/{N}  t={t_next.item():.3f}  "
                f"|u| mean = {u_i.norm(dim=-1).mean():.4e}"
            )
            if h is not None:
                msg += (
                    f"  max h = {hx.max().item():+.4e}  "
                    f"viol frac = {(hx > 0).any(dim=-1).float().mean().item():.3f}"
                )
            print(msg)

    h_out = torch.stack(h_hist, dim=0) if h_hist else None
    return (
        x,
        torch.stack(traj, dim=0),
        torch.stack(controls, dim=0),
        h_out,
    )

from typing import Union, Callable, Optional

"""
FMRG  (Flow Map Reward Guidance)
=================================

Supports two variants selected by the ``variant`` argument of ``fmrg_j``:

FMRG-J  (Jacobian, variant="J")  — Algorithm 5
  At each step k:
    1. x ← X_{t_k, t_{k+1}}(x)             [flow-map advance]
    2. For j = 0, …, n_opt − 1:
         u  =  J_X(x)^T ∇r( X_{t_{k+1},1}(x) )   [chain-rule via backprop]
         x  ←  x  +  (Δt / n_opt) * λ * u

FMRG-E  (Euclidean, variant="E")  — Algorithm 6
  Same as J, except the guidance signal is:
         u  =  ∇r( X_{t_{k+1},1}(x) )             [stop-gradient at X]
  — avoids backpropagation through the network entirely.

Signature of r:   r(x: Tensor[B, D]) -> Tensor[B]
Signature of lam: float  or  Callable(t, x) -> Tensor[B] or scalar
"""




def fmrg_j(
    model,
    r: Callable,
    x0: torch.Tensor,
    *,
    N: int = 20,
    n_iters: int = 1,
    lam: Union[float, Callable] = 1.0,
    variant: str = "J",
    verbose: bool = False,
    device: Optional[torch.device] = None,
):
    """
    FMRG guided sampling via the flow map (Alg. 5 / 6 in the paper).

    Two gradient variants are supported via the ``variant`` argument:

    ``"J"``  — **FMRG-J (Jacobian)**:
        Backpropagates through the full-horizon map X_{t,1} to obtain the
        chain-rule gradient:

            u = ∇_x [ r( X_{t+1,1}(x) ) ] = J_X(x)^T ∇r( X_{t+1,1}(x) )

        This is the more expressive signal but requires one backward pass
        through the network per guidance step.

    ``"E"``  — **FMRG-E (Euclidean)**:
        Evaluates the full-horizon map with ``torch.no_grad()``, then
        differentiates only through the reward at the mapped point:

            u = ∇r( X_{t+1,1}(x) )     [no backprop through X]

        Avoids the memory and compute cost of the network Jacobian, at the
        cost of ignoring the geometry of X.

    Note: unlike ``receding_horizon_flow_control`` and ``hardflow_sample``,
    the first argument is the flow-map *model* (not a velocity callable)
    because FMRG requires both X_{s,t} (incremental step) and X_{t,1}
    (full-horizon guidance signal).

    Args:
        model:    FlowMapMLP with forward(s, t, x) -> X_{s,t}(x), shape (B, D).
        r:        differentiable reward  r(x) -> (B,).
        x0:       initial noise, shape (B, D), drawn from the base distribution.
        N:        number of time steps (dt = 1/N).
        n_iters:  guidance steps applied after each flow-map advance.
        lam:      guidance strength — scalar float or callable lam(t, x) -> Tensor.
        variant:  ``"J"`` (Jacobian, default) or ``"E"`` (Euclidean).
        verbose:  print per-step reward diagnostics.
        device:   target device (defaults to x0.device).

    Returns:
        x_N:      final guided samples,           shape (B, D).
        traj:     full trajectory,                shape (N+1, B, D).
        controls: guidance displacement per step, shape (N, B, D).
    """
    if variant not in ("J", "E"):
        raise ValueError(f"variant must be 'J' or 'E', got {variant!r}")
    if device is None:
        device = x0.device
    x = x0.clone().to(device)
    dt = 1.0 / N
    ts = torch.linspace(0.0, 1.0, N + 1, device=device, dtype=x.dtype)
    t_one = ts[-1].unsqueeze(0)   # shape (1,) — reused as t=1

    B = x.shape[0]
    traj     = [x.clone()]
    controls = []

    for k in range(N):
        s = ts[k]
        t = ts[k + 1]

        # ── Step 1: flow-map advance  x ← X_{t_k, t_{k+1}}(x) ───────────────
        with torch.no_grad():
            x = model(s.expand(B), t.expand(B), x)

        # ── Step 2: n_iters guidance steps ───────────────────────────────────
        x_before = x.detach().clone()
        for _ in range(n_iters):
            if variant == "J":
                # FMRG-J: u = J_X(x)^T ∇r(X_{t+1,1}(x)) via backprop through X
                x_in = x.detach().requires_grad_(True)
                X1 = model(t.expand(B), t_one.expand(B), x_in)    # (B, D)
                u  = torch.autograd.grad(r(X1).sum(), x_in)[0]    # (B, D)
            else:
                # FMRG-E: u = ∇r(X_{t+1,1}(x)) — stop gradient at X
                with torch.no_grad():
                    X1 = model(t.expand(B), t_one.expand(B), x)   # (B, D)
                X1_g = X1.detach().requires_grad_(True)
                u = torch.autograd.grad(r(X1_g).sum(), X1_g)[0]   # (B, D)

            with torch.no_grad():
                strength = lam(t, x) if callable(lam) else lam
                x = x.detach() + (dt / n_iters) * strength * u.detach()

        controls.append(x.detach() - x_before)
        traj.append(x.clone())

        if verbose:
            with torch.no_grad():
                r_mean = r(x).mean().item()
            print(f"step {k+1:4d}/{N}  t={t.item():.3f}  avg r = {r_mean:.4e}  [{variant}]")

    return x.detach(), torch.stack(traj, dim=0), torch.stack(controls, dim=0)


