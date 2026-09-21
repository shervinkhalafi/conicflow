"""
image_mpc_edited_sung.py — Receding-Horizon Path-Space MPC for FLUX FlowMap images
===================================================================================

DERIVED COPY of Shervin's `/home/shervink/flow_matching/image_mpc.py`
(sungcho, 2026-08-26). Per plan Step 5 ("Primal-dual alternation"): that
step needs to add an `n_alt` parameter to `receding_horizon_flow_control`,
which means editing the body of an existing function in a file this
project's standing rule ("do not touch image_mpc.py") treats as
Shervin's/read-only. Rather than edit his file in place, this is a full
copy under a different name, kept in this directory.

DIVERGENCES FROM THE SOURCE (keep this list current):
  1. (2026-08-26) `receding_horizon_flow_control` gained the `n_alt` loop and
     `skip_first_primal_step`, plus one new module-level validation helper.
  2. (2026-08-26) `_horizon_objective`: constraint evaluated with grad=True at
     the endpoint (see the BUG FIX comment in that function).
  3. (2026-08-29) NEW function `hardflow_image` — a port of HardFlow
     (Li, Alim & Azizan, TPAMI 2026) as a third control type alongside MPC and
     FMRG, for baseline comparison. Not present upstream in any form. See
     `260829-hardflow-check.md` for the code/paper audit it is based on and
     for the deliberate deviations from the authors' released implementation.

Every other function is byte-identical to the source at copy time.

WHY A FULL COPY, NOT A DIFF/PATCH FILE: `run_sweep.py` imports several
functions from this module (`make_flux_wrappers`, `fmrg_j_image`,
`trajectory_rewards`, `tune_sampler_speed`) that this change does not
touch; importing "receding_horizon_flow_control from here, everything else
from image_mpc.py" would reintroduce exactly the two-diverged-copies
problem plan Step 1 spent a session undoing, just at import-statement
granularity. A single self-contained module avoids that.

KEEPING THIS IN SYNC: if Shervin's `image_mpc.py` changes upstream, this
file's untouched functions must be re-diffed against it (the same
inventory-and-diff discipline Step 1 used for the two `run_sweep.py`
copies) — this docstring is the marker for future sessions to check that.
As of this copy: `image_mpc.py`, mtime/md5 not independently re-verified
here (see the session log entry for this change for the actual values
recorded at copy time).

Two sections:
  1. Core MPC engine  (generic, works for any state shape)
  2. FLUX image wrappers + demo  (assumes a loaded FluxFlowMapSampler)

Time conventions
----------------
  MPC  : s ∈ [0, 1]   s=0 is noise, s=1 is clean data
  FLUX : t ∈ [0, 1]   t=1 is noise, t=0 is clean data
  Mapping : t_flux = 1 - s_mpc
  Velocity: dx/ds = -dx/dt_flux  →  v_mpc(s,z) = -v_flux(1-s, z)

Usage in notebook
-----------------
  After running cells 0-2 (imports + model load):

      from image_mpc_edited_sung import (
          receding_horizon_flow_control,
          make_flux_wrappers,
          run_image_mpc_demo,
      )
      v_theta, r_fn, endpoint_map = make_flux_wrappers(sampler, PROMPT, guidance_scale=3.5)
      run_image_mpc_demo(sampler, PROMPT)
"""

from __future__ import annotations

import gc
import torch
import torch.nn as nn
from typing import Callable, Optional

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm


def _flat_norm_mean(t: torch.Tensor) -> float:
    """Mean L2 norm over batch for arbitrary-shaped (B, *) tensors."""
    return t.flatten(1).norm(dim=1).mean().item()


def _reward_mean(r: Callable, x: torch.Tensor) -> float:
    with torch.no_grad():
        return r(x).mean().item()


def _maybe_free_memory(low_memory: bool = True) -> None:
    """Run gc/empty_cache only in memory-constrained mode."""
    if not low_memory:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _grad_checkpointing_enabled(module) -> bool:
    if hasattr(module, "is_gradient_checkpointing"):
        return bool(module.is_gradient_checkpointing)
    return bool(getattr(module, "gradient_checkpointing", False))


def _pbar_postfix(
    outer: int,
    outer_total: int,
    reward: Optional[float],
    control: float,
    mu: Optional[float] = None,
) -> dict:
    # reward=None omits the "r" field (2026-09-16). Producing that number costs
    # a VAE decode + a reward-model forward, and under constraint_on="endpoint"
    # a no-grad endpoint rollout on top -- see `pbar_rewards` on
    # receding_horizon_flow_control for why it is now off by default.
    d: dict = {"step": f"{outer}/{outer_total}"}
    if reward is not None:
        d["r"] = f"{reward:.4f}"
    d["|u|"] = f"{control:.4e}"
    if mu is not None:
        d["μ"] = f"{mu:.4e}"
    return d


# ═════════════════════════════════════════════════════════════════════════════
# Section 1 — Core MPC engine
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _rollout_uncontrolled(
    v_theta: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    t0_idx: int,
    H: int,
    dt: float,
    t_grid: torch.Tensor,
) -> torch.Tensor:
    """Warm-start rollout: roll the uncontrolled flow forward H steps.

    Returns x_bar of shape (H, B, *) — states x_1, ..., x_H.
    """
    xs = []
    x = x0
    for ell in range(H):
        t = t_grid[t0_idx + ell]
        x = x + dt * v_theta(t, x)
        xs.append(x)
    return torch.stack(xs, dim=0)


@torch.no_grad()
def _shift_warmstart(
    prev_traj: torch.Tensor,
    v_theta: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_cur: torch.Tensor,
    t0_idx: int,
    H: int,
    dt: float,
    t_grid: torch.Tensor,
) -> torch.Tensor:
    """Shift the previous window's solution into a warm start for the next window.

    The window solved at MPC step ``t0_idx - 1`` produced ``prev_traj`` covering
    times t_{t0_idx}, ..., t_{t0_idx + H_prev - 1}.  The state at t_{t0_idx} has
    now been consumed as the current state, so dropping it leaves an optimized
    guess for t_{t0_idx+1}, ..., t_{t0_idx + H_prev - 1} — exactly the head of
    the new window.  Any remaining tail is filled by rolling the uncontrolled
    flow forward from the last reused state.

    Args:
        prev_traj: (H_prev, B, *) previous window's optimized states.
        v_theta:   velocity field, used only to extend the tail.
        x_cur:     current state at t_{t0_idx} (fallback when nothing is reusable).
        t0_idx:    index of the current state on t_grid.
        H:         length of the new window.
        dt:        step size.
        t_grid:    time grid.

    Returns:
        (H, B, *) warm start covering t_{t0_idx+1}, ..., t_{t0_idx + H}.
    """
    shifted = prev_traj[1:]                        # states from t_{t0_idx+1} onward
    n_reuse = min(shifted.shape[0], H)

    parts = []
    if n_reuse > 0:
        parts.append(shifted[:n_reuse])
        x_last = shifted[n_reuse - 1]              # state at t_{t0_idx + n_reuse}
    else:
        x_last = x_cur                             # nothing reusable: roll from x_cur

    n_extra = H - n_reuse
    if n_extra > 0:
        parts.append(
            _rollout_uncontrolled(
                v_theta, x_last, t0_idx + n_reuse, n_extra, dt, t_grid
            )
        )

    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


def _horizon_objective(
    x_traj: torch.Tensor,           # (H, B, *)
    x_cur: torch.Tensor,            # (B, *), fixed
    v_theta: Callable,
    r: Callable,
    h: Optional[Callable],
    t0_idx: int,
    dt: float,
    lam: float,
    mu: Optional[torch.Tensor],     # (B, C) Lagrange multipliers, or None
    t_grid: torch.Tensor,
    constraint_on: str,             # "horizon" | "endpoint"
    endpoint_map: Optional[Callable],
    t_horizon: Optional[torch.Tensor],
    v_cur: Optional[torch.Tensor] = None,
    drift_detach: bool = False,
    h_scale: Optional[torch.Tensor] = None,   # (C,) per-constraint normaliser
) -> torch.Tensor:
    """Lagrangian path-space objective, summed over the batch (scalar).

    Works for arbitrary state shapes (flat (B,D) or image latents (B,C,H,W)).

    Args:
        v_cur:        cached v_theta(t_{t0_idx}, x_cur).  The ell=0 residual
                      evaluates the drift at the fixed current state, so it is
                      identical across inner iterations and contributes no
                      gradient — passing it in avoids one transformer call per
                      iteration.  Mathematically exact.
        drift_detach: evaluate the drift for ell>0 under no_grad, treating it as
                      locally constant.  Drops the J_v^T term from the gradient
                      (an approximation) but removes every transformer backward
                      pass, leaving only the reward path to differentiate.
    """
    H = x_traj.shape[0]
    x_full = torch.cat([x_cur.unsqueeze(0), x_traj], dim=0)  # (H+1, B, *)

    # Control-energy: flatten non-batch dims so sum gives (B,) for any shape
    energy = x_full.new_zeros(x_full.shape[1])  # (B,)
    for ell in range(H):
        t = t_grid[t0_idx + ell]
        if ell == 0 and v_cur is not None:
            v = v_cur                       # x_full[0] is x_cur: fixed, no grad
        elif drift_detach:
            with torch.no_grad():
                v = v_theta(t, x_full[ell].detach())
        else:
            v = v_theta(t, x_full[ell])
        resid = x_full[ell + 1] - x_full[ell] - dt * v
        energy = energy + resid.pow(2).flatten(1).sum(1) / (2.0 * lam * dt)

    x_H = x_full[-1]

    # Resolve the point at which both reward and constraints are evaluated.
    # "horizon"  → x_H (end of the planning window).
    # "endpoint" → flow-map prediction of the final image latent from x_H.
    #   reward uses grad=True so gradients flow back through the rollout;
    #   constraint uses grad=False (detached) since it only needs the value.
    if constraint_on == "endpoint" and endpoint_map is not None:
        # BUG FIX 2026-08-26 (sungcho). Was:
        #     z_r = endpoint_map(..., x_H, grad=True)
        #     z_h = endpoint_map(..., x_H, grad=False)   # <-- detached
        # The constraint was evaluated at the endpoint (correct, and what was
        # agreed) but DETACHED there, so h(z_h) was a constant w.r.t. x_traj
        # and `mu * hx` contributed exactly zero gradient. In a primal-dual
        # method that makes the dual variable decorative: mu climbs via dual
        # ascent and pushes on nothing, the primal never sees the constraint,
        # and with reward_model="none" the objective reduces to control energy
        # whose minimizer is the uncontrolled trajectory. Verified with
        # torch.autograd.grad: gradients at mu=0 vs mu=50 were bit-identical
        # under "endpoint" (max|diff| 0.000e+00) while differing by 5.0e+01
        # under "horizon".
        #
        # WHERE h is evaluated (the endpoint) and WHETHER it is differentiable
        # there are independent axes; the agreement was about the former, and
        # grad=False silently broke the latter.
        #
        # The two calls above were identical apart from `grad`, so the fix is
        # to compute ONE differentiable rollout and share it -- strictly less
        # forward work than before (one rollout, not two), at the cost of
        # letting backward flow through the constraint path, which is exactly
        # the coupling a primal-dual method requires.
        z_r = z_h = endpoint_map(t_horizon.expand(x_H.shape[0]), x_H, grad=True)
    else:
        z_r = z_h = x_H

    obj = energy - r(z_r)  # (B,)

    # Lagrangian penalty for inequality constraint h(z) <= 0
    if h is not None and mu is not None:
        hx = h(z_h)
        if hx.dim() == 1:
            hx = hx.unsqueeze(-1)
        if h_scale is not None:
            # Work with the normalised constraint h_c / s_c here as well as in
            # the dual step, so that mu_c * h_c -- and hence the whole
            # Lagrangian -- is invariant to per-constraint rescaling
            # h_c -> k_c * h_c. Normalising only the dual step leaves mu_c
            # invariant and the product still scaling by k_c.
            hx = hx / h_scale
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
    x_init: Optional[torch.Tensor] = None,
    return_full: bool = False,
    drift_detach: bool = False,
    v_cur: Optional[torch.Tensor] = None,
    pbar: Optional[tqdm] = None,
    pbar_rewards: bool = False,
    outer_step: Optional[int] = None,
    outer_total: Optional[int] = None,
    low_memory: bool = True,
    h_scale: Optional[torch.Tensor] = None,
):
    """Solve the finite-horizon path-space problem.

    Args:
        x_init:       optional (H, B, *) warm-start trajectory.  Defaults to the
                      uncontrolled rollout from x_cur.  Pass the shifted solution
                      of the previous window to warm start the solve.
        return_full:  also return the full optimized window (needed to warm start
                      the next window).
        drift_detach: treat the drift as locally constant, removing all
                      transformer backward passes (approximate but much faster).
        v_cur:        precomputed v_theta(t_{t0_idx}, x_cur).  The caller usually
                      needs this tensor anyway to apply the MPC step, so passing
                      it in avoids recomputing it here.

    Returns:
        x1_star: (B, *)  first step of the optimal trajectory
        xH_star: (B, *)  last  step of the optimal trajectory (for dual update)
        x_full:  (H, B, *) full optimized window — only when return_full=True
    """
    if x_init is None:
        x_init = _rollout_uncontrolled(v_theta, x_cur, t0_idx, H, dt, t_grid)
    elif x_init.shape[0] != H:
        raise ValueError(
            f"x_init has {x_init.shape[0]} steps but the window needs {H}."
        )
    x_traj = x_init.detach().clone().requires_grad_(True)
    x_cur_fixed = x_cur.detach()
    t_horizon = t_grid[t0_idx + H]

    # The ell=0 drift is evaluated at the fixed current state, so it is constant
    # across inner iterations. Computing it once removes one transformer
    # forward+backward per iteration (a 1/H saving) with no approximation.
    if v_cur is None:
        with torch.no_grad():
            v_cur = v_theta(t_grid[t0_idx], x_cur_fixed)

    kw = dict(
        x_cur=x_cur_fixed, v_theta=v_theta, r=r, h=h,
        t0_idx=t0_idx, dt=dt, lam=lam, mu=mu, t_grid=t_grid,
        constraint_on=constraint_on, endpoint_map=endpoint_map,
        t_horizon=t_horizon,
        v_cur=v_cur, drift_detach=drift_detach,
        h_scale=h_scale,
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
        if pbar is not None:
            with torch.no_grad():
                resid0 = x_traj[0] - x_cur_fixed - dt * v_cur
                pbar.set_postfix(_pbar_postfix(
                    outer_step or 0, outer_total or 0,
                    _reward_mean(r, x_traj[-1]) if pbar_rewards else None,
                    _flat_norm_mean(resid0),
                ))
            pbar.update(n_iters)
    else:  # adam
        opt = torch.optim.Adam([x_traj], lr=lr)
        for _ in range(n_iters):
            opt.zero_grad(set_to_none=True)
            loss = _horizon_objective(x_traj, **kw)
            loss.backward()
            opt.step()
            if pbar is not None:
                with torch.no_grad():
                    resid0 = x_traj[0] - x_cur_fixed - dt * v_cur
                    pbar.set_postfix(_pbar_postfix(
                        outer_step or 0, outer_total or 0,
                        # A VAE decode + a reward-model forward per inner
                        # iteration, for a display field. Off unless asked.
                        _reward_mean(r, x_traj[-1]) if pbar_rewards else None,
                        _flat_norm_mean(resid0),
                    ))
                pbar.update(1)
            del loss

    x1_star, xH_star = x_traj[0].detach(), x_traj[-1].detach()
    x_full = x_traj.detach().clone() if return_full else None
    del x_traj, x_init, opt, v_cur
    _maybe_free_memory(low_memory)
    if return_full:
        return x1_star, xH_star, x_full
    return x1_star, xH_star


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
    dual_adaptive: bool = False,
    dual_beta: float = 0.9,
    dual_eps: float = 1e-8,
    n_alt: int = 1,
    skip_first_primal_step: bool = False,
    optimizer: str = "lbfgs",
    n_iters: int = 50,
    lr: float = 0.1,
    warmstart: bool = False,
    drift_detach: bool = False,
    verbose: bool = False,
    show_progress: bool = True,
    pbar_rewards: bool = False,
    low_memory: bool = True,
    device: Optional[torch.device] = None,
    return_dual_h: bool = False,
):
    """Receding-Horizon Path-Space Control with inequality constraint h(x) <= 0.

    Args:
        v_theta:       velocity field  v_theta(t, x) -> (B, *).
        r:             terminal reward  r(x) -> (B,).  Must be differentiable.
        x0:            initial samples from p_0, shape (B, *).
        N:             number of time steps (dt = 1/N).
        H:             planning horizon (clipped near t=1).
        lam:           control weight lambda > 0.
        mu_init:       initial scalar value for all Lagrange multipliers.
        h:             inequality constraint  h(x) -> (B, C),  h(x) <= 0.
                       Pass None to run unconstrained.
        constraint_on: "horizon"  — evaluate r and h at x_H.
                       "endpoint" — evaluate r and h at endpoint_map(t_H, x_H).
        endpoint_map:  Callable(t_batch, x) -> (B, *).  Required when
                       constraint_on="endpoint".
        lr_dual:       projected dual-ascent step size (>= 0).
        dual_adaptive: normalise the projected dual step per constraint by a
                       running RMS scale s_c, and use the SAME normalised
                       constraint h_c / s_c in the Lagrangian:

                           v_c  <- beta*v_c + (1-beta)*mean_i[h_c(x_i)^2]
                           s_c  =  sqrt(v_c_bias_corrected) + eps
                           mu_c <- max(0, mu_c + lr_dual * h_c / s_c)

                       Plain dual ascent grows mu_c at a rate proportional to
                       |h_c|, so under a shared lr_dual a constraint whose
                       typical violation is small gets proportionally less
                       dual pressure and stalls while a large-violation one
                       saturates. Normalising equalises that pressure without
                       any offline calibration or knowledge of the thresholds.
                       Scaling the constraint in the objective as well (rather
                       than only the dual step) is what makes the Lagrangian
                       exactly invariant to h_c -> k_c * h_c.

                       s_c is updated on EVERY dual update, so with n_alt > 1
                       it advances n_alt times per outer step.
                       False (default) reproduces plain dual ascent exactly.
        dual_beta:     EMA decay for the dual second-moment estimate.
        dual_eps:      numerical floor in the normalised dual step.
        n_alt:         number of primal-solve <-> dual-update alternations
                       PER FIXED WINDOW, before the window shifts (plan Step
                       5). Added in this derived copy; NOT present in
                       Shervin's original `image_mpc.py`. Default 1
                       reproduces the original function exactly (a single
                       primal solve, a single dual update, then shift) --
                       verified bit-for-bit on a fixed seed, see the
                       session log entry for this change. n_alt>1 re-solves
                       the SAME window's primal subproblem, warm-started
                       from the previous alternation's own solution (not the
                       uncontrolled rollout), between each dual update --
                       the "substantive" version pre-meeting notes
                       distinguish from the useless one (repeating the dual
                       step alone is algebraically identical to scaling
                       lr_dual by n_alt and adds nothing). Requires h is not
                       None and lr_dual > 0 -- raises otherwise, since
                       alternating a primal-only re-solve with no dual
                       update in between changes nothing about the result
                       while costing n_alt x the compute. Asserts h(z)
                       actually changes between successive dual updates
                       within a window; raises if it doesn't, per the plan's
                       own acceptance criterion for this step. mu_hist still
                       records exactly one multiplier snapshot per OUTER
                       step j (the state after the last of that step's
                       n_alt dual updates) -- unchanged shape (N, B, C) --
                       so every downstream consumer of mu_hist (run_sweep.py's
                       CSV/mu_trace/plot code) needs no changes regardless
                       of n_alt.
        skip_first_primal_step: plan Step 7, option (b) ("skip the first
                       primal step... do one dual update, then start").
                       Added in this derived copy; NOT present in Shervin's
                       original `image_mpc.py`. Addresses the same first-step
                       degeneracy option (a) (a nonzero `mu_init`) addresses:
                       with `reward_model="none"` the MPC warm start is
                       dynamics-exact (energy == 0) and, at `mu_init=0.0`,
                       the constraint term is also 0, so step j=0's primal
                       objective and gradient are both exactly zero and that
                       inner solve accomplishes nothing. Rather than eating
                       a wasted primal solve, this option performs ONE
                       projected dual-ascent update -- using h evaluated at
                       the UNCONTROLLED rollout's predicted window endpoint
                       (no primal optimization involved, since none has
                       happened yet) -- BEFORE the outer loop starts, so
                       j=0's primal solve inherits an already-informative
                       mu instead of `mu_init` verbatim. Composable with a
                       nonzero `mu_init`: the pre-loop update adds
                       `lr_dual * h(uncontrolled x_H0)` on top of whatever
                       `mu_init` supplies, clamped to be nonnegative exactly
                       like every other dual update in this function (plan
                       Step 7's own comparison grid crosses both options,
                       so they must be combinable, not mutually exclusive).
                       Requires h is not None and lr_dual > 0 -- raises
                       otherwise, mirroring n_alt's guard, since "one dual
                       update" with the dual step disabled is a no-op.
                       Does NOT add an entry to `mu_hist`: mu_hist stays
                       (N, B, C), one snapshot per OUTER step, matching
                       n_alt's own contract -- the pre-loop update only
                       changes what `mu_t` already IS by the time step j=0's
                       own snapshot is taken, so no downstream consumer
                       needs to change.
        optimizer:     "lbfgs" or "adam" for the inner solve.
        n_iters:       inner-solver iterations per MPC step.
        lr:            inner-solver learning rate.
        warmstart:     reuse the previous window's solution as the initial guess.
                       The window at step j covers t_{j+1}..t_{j+H}; the previous
                       window already optimized t_{j+1}..t_{j+H-1}, so those
                       states are shifted in and only the final state is filled
                       by an uncontrolled rollout.  Usually converges in far
                       fewer inner iterations than a cold rollout start.
        drift_detach:  treat the drift as locally constant inside the inner
                       solve.  Removes every transformer backward pass, so only
                       the reward path is differentiated.  Roughly halves the
                       cost again at the price of dropping the J_v^T term from
                       the gradient.
        verbose:       print per-step diagnostics.
        show_progress: show a tqdm bar of length N * n_iters with outer step,
                       control norm, and (only when pbar_rewards) reward.
        pbar_rewards:  compute the reward displayed in the progress bar.
                       Default False because it is NOT free: per outer step a
                       no-grad endpoint rollout (N-j-1 transformer forwards
                       under constraint_on="endpoint") plus a decode and a
                       reward-model pass, and another decode + reward pass per
                       inner iteration -- ~4% of a run at N=8, for a field
                       nobody reads inside a sweep. Separate from
                       show_progress because tqdm(disable=True) still returns a
                       live object, so the `if pbar is not None` guards never
                       suppressed this work (measured 2026-09-16: 232 s with
                       show_progress True and False alike).
        low_memory:    if False, skip per-step gc/empty_cache (faster when VRAM
                       headroom is available; pair with ``--fast true``).
        device:        torch device (defaults to x0.device).

    Returns:
        x_N:      final controlled samples,          shape (B, *).
        traj:     full trajectory,                   shape (N+1, B, *).
        controls: applied controls u_j*,             shape (N, B, *).
        mu_final: final per-sample multipliers,      shape (B, C), or None.
        mu_hist:  per-step multipliers after each
                  dual-ascent update,                shape (N, B, C), or None.
    """
    if n_alt < 1:
        raise ValueError(f"n_alt must be >= 1, got {n_alt}.")
    if n_alt > 1 and not (h is not None and lr_dual > 0.0):
        raise ValueError(
            f"n_alt={n_alt} requires both h is not None and lr_dual > 0.0 "
            f"(got h={'set' if h is not None else None}, lr_dual={lr_dual}). "
            f"Without an active dual update, re-solving the same primal "
            f"window n_alt times converges to the same optimum every time "
            f"-- n_alt-x the compute for no different result. Not a case to "
            f"silently allow."
        )
    if skip_first_primal_step and not (h is not None and lr_dual > 0.0):
        raise ValueError(
            f"skip_first_primal_step=True requires both h is not None and "
            f"lr_dual > 0.0 (got h={'set' if h is not None else None}, "
            f"lr_dual={lr_dual}). The option's whole effect is one extra "
            f"projected dual-ascent update before the loop starts -- with "
            f"no active dual step there is nothing for it to do. Not a "
            f"case to silently allow."
        )
    if device is None:
        device = x0.device
    x0 = x0.to(device)
    dt = 1.0 / N
    t_grid = torch.linspace(0.0, 1.0, N + 1, device=device, dtype=x0.dtype)

    x = x0.detach().clone()
    traj = [x.clone()]
    controls = []
    mu_history: list = []
    h_dual_history: list = []   # the h each dual update actually consumed

    mu_t = None
    if h is not None:
        with torch.no_grad():
            h0 = h(x0)
            C = h0.shape[-1] if h0.dim() > 1 else 1
        mu_t = torch.full(
            (x0.shape[0], C), float(mu_init), dtype=x0.dtype, device=device
        )

    # Running per-constraint scale s_c for dual_adaptive (see the docstring).
    # Shared by the dual step and the Lagrangian, which is what makes the whole
    # objective invariant to per-constraint rescaling h_c -> k_c * h_c.
    _dual_state: dict = {"v": None, "t": 0}

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
        # Refresh the per-constraint scale from this step's observations first.
        v_obs = hx.pow(2).mean(dim=0)                    # (C,)
        if _dual_state["v"] is None:
            _dual_state["v"] = torch.zeros_like(v_obs)
        _dual_state["t"] += 1
        _dual_state["v"] = dual_beta * _dual_state["v"] + (1.0 - dual_beta) * v_obs
        return (mu + lr_dual * hx / _h_scale()).clamp_min(0.0)

    # Plan Step 7, option (b): one projected dual-ascent update BEFORE the
    # outer loop starts, using h at the UNCONTROLLED rollout's predicted
    # window endpoint (no primal solve has happened yet, so there is no
    # optimized x_H to use instead -- the uncontrolled one is the only
    # information available at this point). This is the same dual-update
    # formula the per-step loop below uses, just evaluated once, early,
    # against x0 rather than against x (the loop's running state hasn't
    # been created yet either).
    if skip_first_primal_step:
        H_0 = min(H, N)
        with torch.no_grad():
            x_bar_0 = _rollout_uncontrolled(
                v_theta, x0, t0_idx=0, H=H_0, dt=dt, t_grid=t_grid
            )
            xH_uncontrolled = x_bar_0[-1]
            if constraint_on == "endpoint" and endpoint_map is not None:
                t_h0 = t_grid[H_0]
                z0_h = endpoint_map(t_h0.expand(x0.shape[0]), xH_uncontrolled)
            else:
                z0_h = xH_uncontrolled
            hx0 = h(z0_h)
            if hx0.dim() == 1:
                hx0 = hx0.unsqueeze(-1)
            mu_t = _dual_update(mu_t, hx0)
        del x_bar_0, xH_uncontrolled, z0_h, hx0

    prev_window = None   # previous window's optimized states (warmstart only)

    pbar = tqdm(
        total=N * n_iters * n_alt, desc="MPC", disable=not show_progress,
    )

    for j in range(N):
        H_j = min(H, N - j)

        x_init_j = None
        if warmstart and prev_window is not None:
            x_init_j = _shift_warmstart(
                prev_window, v_theta, x, t0_idx=j, H=H_j, dt=dt, t_grid=t_grid
            )

        # Drift at the current state: needed both by the inner objective (ell=0)
        # and to apply the MPC step below, so evaluate it a single time.
        with torch.no_grad():
            v_j = v_theta(t_grid[j], x)

        # n_alt primal<->dual alternations within this ONE fixed window
        # (plan Step 5). n_alt=1 (default) executes this loop body exactly
        # once, with the exact same solve_horizon_problem call, the exact
        # same dual-update block, and the exact same return_full condition
        # (`warmstart`) as Shervin's original -- verified bit-for-bit
        # identical to the un-modified function on a fixed seed at n_alt=1.
        need_full = warmstart or n_alt > 1
        x_full_j = None
        h_prev_for_assert = None
        h_dual_step = None
        for alt in range(n_alt):
            solved = solve_horizon_problem(
                x, v_theta, r, h, t0_idx=j, H=H_j, dt=dt,
                lam=lam, mu=mu_t, t_grid=t_grid,
                constraint_on=constraint_on, endpoint_map=endpoint_map,
                optimizer=optimizer, n_iters=n_iters, lr=lr,
                x_init=x_init_j, return_full=need_full,
                drift_detach=drift_detach, v_cur=v_j,
                pbar=pbar, pbar_rewards=pbar_rewards, outer_step=j + 1, outer_total=N,
                low_memory=low_memory, h_scale=_h_scale(),
            )
            if need_full:
                x1_star, xH_star, x_full_j = solved
            else:
                x1_star, xH_star = solved
            del solved

            if n_alt > 1 and alt < n_alt - 1:
                # Warm-start the NEXT alternation's primal re-solve from
                # THIS alternation's own optimum (same window), not from a
                # cold uncontrolled rollout -- this is what makes each
                # successive dual update see a genuinely re-optimized u,
                # per Step 5's "substantive" vs. "useless" distinction.
                x_init_j = x_full_j

            # Projected dual ascent (identical block to the un-modified
            # function; now runs up to n_alt times per outer step instead
            # of exactly once).
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

                    # Step 5's own acceptance criterion: if h does not
                    # change between successive dual updates within this
                    # window, the primal is not actually being re-solved --
                    # the alternation has degenerated into repeating the
                    # dual step with the primal frozen, which is
                    # algebraically identical to scaling lr_dual by n_alt
                    # (adds nothing; the plan explicitly calls this the
                    # "useless" version). Fail loudly rather than silently
                    # returning a result indistinguishable from that case.
                    if n_alt > 1 and alt > 0 and h_prev_for_assert is not None:
                        if torch.allclose(hx, h_prev_for_assert, atol=1e-6, rtol=1e-5):
                            raise RuntimeError(
                                f"n_alt={n_alt}: h(z) did not change between "
                                f"successive dual updates (outer step {j}, "
                                f"alternation {alt - 1} -> {alt}). The primal "
                                f"subproblem is not actually being re-solved "
                                f"between alternations -- see this function's "
                                f"n_alt docstring."
                            )
                    h_prev_for_assert = hx.clone()

                    mu_t = _dual_update(mu_t, hx)
                    # The EXACT h the dual consumed at this outer step. Recorded
                    # because run_sweep.py otherwise has to reconstruct it from
                    # the realized state, which is a DIFFERENT quantity: the dual
                    # reads h at the optimizer's horizon endpoint (H_j steps
                    # ahead of the realized state), so the reconstruction lags by
                    # ~H and makes mu look like it falls while h is still
                    # positive. See run_sweep.py's "Caveat that must not be
                    # glossed" comment, which this makes unnecessary.
                    h_dual_step = hx.clone()

        if warmstart:
            prev_window = x_full_j
        del x_init_j, x_full_j, h_prev_for_assert

        # Exactly one mu snapshot per OUTER step (the state after that
        # step's last dual update) -- mu_hist's shape (N, B, C) is
        # unchanged by n_alt, so no downstream consumer needs to change.
        if mu_t is not None:
            mu_history.append(mu_t.clone())
            h_dual_history.append(
                h_dual_step if h_dual_step is not None
                else torch.full_like(mu_t, float("nan"))
            )

        # Apply first MPC step, reusing the drift computed above
        with torch.no_grad():
            u_star = (x1_star - x) / dt - v_j
            x = x + dt * (v_j + u_star)

        traj.append(x.clone())
        controls.append(u_star.clone())

        if pbar is not None:
            mu_mean_val = mu_t.mean().item() if mu_t is not None else None
            # Not free (2026-09-16): under constraint_on="endpoint" this is a
            # full no-grad endpoint rollout -- N-j-1 transformer forwards --
            # plus a decode and a reward-model pass, every outer step, to fill
            # one progress-bar field. Gated on pbar_rewards, NOT on
            # `pbar is not None`, because tqdm(disable=True) is still an object.
            r_disp = None
            if pbar_rewards:
                if constraint_on == "endpoint" and endpoint_map is not None:
                    with torch.no_grad():
                        x_disp = endpoint_map(t_grid[j + 1].expand(x.shape[0]), x)
                else:
                    x_disp = x
                r_disp = _reward_mean(r, x_disp)
            pbar.set_postfix(_pbar_postfix(
                j + 1, N, r_disp, _flat_norm_mean(u_star), mu_mean_val,
            ))

        if verbose:
            mu_mean = mu_t.mean().item() if mu_t is not None else 0.0
            u_mean = u_star.flatten(1).norm(dim=1).mean().item()
            print(
                f"step {j+1:4d}/{N}  "
                f"|u*| mean = {u_mean:.4e}  "
                f"mu mean = {mu_mean:.4e}"
            )

        del x1_star, xH_star, v_j, u_star
        _maybe_free_memory(low_memory)

    if pbar is not None:
        pbar.close()

    mu_hist = torch.stack(mu_history, dim=0) if mu_history else None
    h_dual_hist = torch.stack(h_dual_history, dim=0) if h_dual_history else None
    out = (x, torch.stack(traj, dim=0), torch.stack(controls, dim=0), mu_t, mu_hist)
    # Opt-in 6th element so every existing 5-tuple call site is untouched.
    return out + (h_dual_hist,) if return_dual_h else out


# ═════════════════════════════════════════════════════════════════════════════
# Section 2 — FLUX image wrappers
# ═════════════════════════════════════════════════════════════════════════════

def tune_sampler_speed(
    sampler,
    fast: bool = True,
    verbose: bool = True,
    *,
    control_type: Optional[str] = None,
    drift_detach: Optional[bool] = None,
):
    """Trade the sampler's memory-saving settings for speed.

    ``FluxFlowMapSampler`` enables gradient checkpointing on the transformer and
    slicing + tiling on the VAE to minimise VRAM.

    Safety rule for transformer gradient checkpointing:
      ``fast=True`` disables checkpointing ONLY when it is safe to do so, i.e.
      when the inner solve does NOT backprop through the transformer:
        - drift_detach=True  (MPC): drifts computed under no_grad → safe
        - control_type="none":      no backward at all → safe
        - FMRG:                     always backprops through transformer → UNSAFE
        - drift_detach=False (MPC): same as FMRG → UNSAFE

      Disabling checkpointing when the transformer backward is live requires
      storing H-1 full activation graphs simultaneously.  For FLUX (12+ GB
      weights) this causes hard OOM / system crash with H >= 2.

    VAE tiling/slicing is always safe to disable (affects decode speed only).

    Note that weight quantization (``quantize_mode``) is fixed at load time and
    cannot be toggled here.  Load with ``quantize_mode="none"`` for maximum speed.
    """
    tf, vae = sampler.pipeline.transformer, sampler.pipeline.vae

    # Decide whether disabling transformer checkpointing is memory-safe.
    # Conservative default: keep checkpointing unless we KNOW backward is absent.
    transformer_backward_needed = True
    if drift_detach is True:
        transformer_backward_needed = False   # MPC no-grad drifts
    elif control_type == "none":
        transformer_backward_needed = False   # uncontrolled rollout only
    # fmrg, drift_detach=False, or unknown → keep checkpointing

    disable_ckpt = fast and not transformer_backward_needed

    if disable_ckpt:
        if hasattr(tf, "disable_gradient_checkpointing"):
            tf.disable_gradient_checkpointing()
    else:
        if hasattr(tf, "enable_gradient_checkpointing"):
            tf.enable_gradient_checkpointing()

    # VAE tiling/slicing: always safe to toggle
    if fast:
        if hasattr(vae, "disable_tiling"):
            vae.disable_tiling()
        if hasattr(vae, "disable_slicing"):
            vae.disable_slicing()
    else:
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()

    if verbose:
        ckpt = _grad_checkpointing_enabled(tf)
        tiling = getattr(vae, "use_tiling", "n/a")
        slicing = getattr(vae, "use_slicing", "n/a")
        print(
            f"sampler speed mode: fast={fast}  "
            f"(transformer grad-checkpointing={ckpt}, "
            f"vae_tiling={tiling}, vae_slicing={slicing}, "
            f"quantize_mode={getattr(sampler, 'quantize_mode', 'n/a')})"
        )
        if fast and control_type == "mpc" and drift_detach is True:
            print(
                "  note: MPC+drift_detach=True: transformer checkpointing is not "
                "on the hot path — speedup is from VAE tiling only."
            )
        elif fast and transformer_backward_needed:
            print(
                "  note: transformer grad-checkpointing kept ON because backward "
                "passes through the transformer are needed (fmrg or drift_detach=False). "
                "Disabling it would OOM.  Use drift_detach=True for a safe --fast speedup."
            )
    return sampler


def make_flux_wrappers(
    sampler,
    prompt: str,
    guidance_scale: float = 3.5,
    n_mpc_steps: int = 4,
    sample_mode: str = "flow_map2",
):
    """Build (v_theta, r, endpoint_map) callables for use with receding_horizon_flow_control.

    The MPC uses progress time s ∈ [0,1] (s=0 noise, s=1 data).
    FLUX uses t ∈ [0,1] (t=1 noise, t=0 data).  Mapping: t = 1 - s.

    Args:
        sampler:        loaded FluxFlowMapSampler instance.
        prompt:         text prompt used for guidance (must match sampler.cached_*).
        guidance_scale: CFG scale passed to predict_vector.
        n_mpc_steps:    N used in the outer MPC loop (needed for step size and endpoint_map).
        sample_mode:    "flow_map2" (canonical, t_next = t_cur - dt)  ← default
                        "flow_map1" (shortcut, t_next = 0)
                        "flow_matching" (plain ODE, t_next = t_cur)

    Returns:
        v_theta_mpc:   (s: tensor, z: tensor) -> tensor  velocity in MPC convention
        r_image:       (z: tensor) -> (B,)               differentiable reward
        endpoint_map:  (s: tensor, z: tensor) -> tensor  predicted final latent
    """
    pe  = sampler.cached_prompt_embeds
    ppe = sampler.cached_pooled_prompt_embeds
    dt_flux = 1.0 / n_mpc_steps   # FLUX-time step size

    def _t_next_for_mode(t_flux: float) -> float:
        """Return t_next to pass to predict_vector based on the sample mode."""
        if sample_mode == "flow_map2":
            return max(0.0, t_flux - dt_flux)   # next FLUX timestep
        elif sample_mode == "flow_map1":
            return 0.0                           # predict all the way to t=0
        else:  # flow_matching
            return t_flux                        # standard ODE (t_next = t_cur)

    def v_theta_mpc(s_mpc: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """FLUX velocity in MPC convention: v_mpc = -v_flux(1-s, z)."""
        t_flux = 1.0 - s_mpc.item()
        t_next = _t_next_for_mode(t_flux)
        ctx = torch.enable_grad() if z.requires_grad else torch.no_grad()
        with ctx:
            u = sampler.predict_vector(
                z, t_flux,
                prompt_embeds=pe,
                pooled_prompt_embeds=ppe,
                guidance_scale=guidance_scale,
                t_next=t_next,
            )
        return -u

    def r_image(z: torch.Tensor) -> torch.Tensor:
        """Default reward: mean pixel brightness of the decoded image.

        Replace with any differentiable function of the decoded image,
        e.g. HPSv2, CLIP similarity, or a custom aesthetic scorer.

        Returns shape (B,).
        """
        img = sampler.decode(z)              # (B, 3, H, W) in [-1, 1], grads flow
        return img.mean(dim=(1, 2, 3))

    def endpoint_map(s_mpc: torch.Tensor, z: torch.Tensor, grad: bool = False) -> torch.Tensor:
        """Predict the final latent by rolling the uncontrolled FLUX flow to t=0.

        Args:
            grad: if True, keep gradients flowing through the rollout so that
                  the result can be used as a differentiable reward target.
                  If False (default), the rollout runs under no_grad and the
                  input is detached — suitable for constraint evaluation and
                  dual-ascent updates where no backward is needed.
        """
        t_start = 1.0 - s_mpc.item()
        n_remain = max(1, round(t_start * n_mpc_steps))
        ts = torch.linspace(t_start, 0.0, n_remain + 1, device=z.device, dtype=z.dtype)
        z_run = z if grad else z.detach().clone()
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            for i in range(n_remain):
                t_c = ts[i].item()
                t_n = _t_next_for_mode(t_c)
                dt_r = (ts[i] - ts[i + 1]).item()
                u = sampler.predict_vector(
                    z_run, t_c,
                    prompt_embeds=pe,
                    pooled_prompt_embeds=ppe,
                    guidance_scale=guidance_scale,
                    t_next=t_n,
                )
                z_run = z_run - dt_r * u
        return z_run

    return v_theta_mpc, r_image, endpoint_map


def run_image_mpc_demo(
    sampler,
    prompt: str,
    height: int = 256,
    width: int = 256,
    n_mpc_steps: int = 4,
    h_horizon: int = 2,
    lam: float = 1.0,
    n_inner: int = 20,
    lr_inner: float = 1e-3,
    guidance_scale: float = 3.5,
    seed: int = 42,
    sample_mode: str = "flow_map2",
    warmstart: bool = False,
):
    """Run a side-by-side uncontrolled vs MPC-controlled FLUX image demo.

    Prints per-step control norms and shows a matplotlib figure.

    Args:
        sampler:         loaded FluxFlowMapSampler.
        prompt:          text prompt (must match sampler's cached embeddings).
        height/width:    output image resolution.
        n_mpc_steps:     N — total denoising steps.
        h_horizon:       H — MPC planning horizon.
        lam:             control weight lambda.
        n_inner:         inner Adam iterations per MPC step.
        lr_inner:        inner Adam learning rate.
        guidance_scale:  CFG guidance scale.
        seed:            random seed for the initial noise.
    """
    import matplotlib.pyplot as plt

    device = next(sampler.pipeline.transformer.parameters()).device
    dtype  = next(sampler.pipeline.transformer.parameters()).dtype

    v_theta_mpc, r_image, endpoint_map = make_flux_wrappers(
        sampler, prompt, guidance_scale=guidance_scale,
        n_mpc_steps=n_mpc_steps, sample_mode=sample_mode,
    )

    # Initial noise latent
    torch.manual_seed(seed)
    in_ch   = sampler.pipeline.transformer.config.in_channels // 4
    vscale  = sampler.vae_scale_factor
    H_lat, W_lat = height // vscale, width // vscale
    z0 = torch.randn(1, in_ch, H_lat, W_lat, device=device, dtype=dtype)

    # Uncontrolled baseline using the same sample_mode
    dt_flux = 1.0 / n_mpc_steps
    print(f"Running uncontrolled baseline ({sample_mode})...")
    with torch.no_grad():
        z_unc = z0.clone()
        ts_unc = torch.linspace(1.0, 0.0, n_mpc_steps + 1, device=device, dtype=dtype)
        for j in range(n_mpc_steps):
            t_c = ts_unc[j].item()
            if sample_mode == "flow_map2":
                t_n = max(0.0, t_c - dt_flux)
            elif sample_mode == "flow_map1":
                t_n = 0.0
            else:
                t_n = t_c
            dt_unc = (ts_unc[j] - ts_unc[j + 1]).item()
            u = sampler.predict_vector(
                z_unc, t_c,
                prompt_embeds=sampler.cached_prompt_embeds,
                pooled_prompt_embeds=sampler.cached_pooled_prompt_embeds,
                guidance_scale=guidance_scale,
                t_next=t_n,
            )
            z_unc = z_unc - dt_unc * u
    img_unc = sampler.pipeline.image_processor.postprocess(
        sampler.decode_no_grad(z_unc).float().detach().cpu(), output_type="pil"
    )[0]

    # MPC-controlled run
    print("Running MPC...")
    z_mpc, traj_mpc, controls_mpc, mu_final, _ = receding_horizon_flow_control(
        v_theta   = v_theta_mpc,
        r         = r_image,
        x0        = z0,
        N         = n_mpc_steps,
        H         = h_horizon,
        lam       = lam,
        optimizer = "adam",
        n_iters   = n_inner,
        lr        = lr_inner,
        warmstart = warmstart,
        verbose   = True,
        device    = device,
    )
    img_mpc = sampler.pipeline.image_processor.postprocess(
        sampler.decode_no_grad(z_mpc).float().detach().cpu(), output_type="pil"
    )[0]

    # Side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_unc); axes[0].axis("off"); axes[0].set_title("Uncontrolled")
    axes[1].imshow(img_mpc); axes[1].axis("off")
    axes[1].set_title(f"MPC  H={h_horizon}  λ={lam}  N_inner={n_inner}")
    plt.suptitle(prompt[:70], fontsize=10)
    plt.tight_layout()
    plt.show()

    ctrl_norms = controls_mpc.flatten(2).norm(dim=2)   # (N, B)
    print(f"\nMean control ‖u‖ per step: {ctrl_norms.mean(dim=1).tolist()}")
    return img_unc, img_mpc, traj_mpc, controls_mpc


def fmrg_j_image(
    sampler,
    r: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    *,
    N: int = 4,
    n_opt: int = 1,
    lam: float = 1.0,
    guidance_scale: float = 3.5,
    sample_mode: str = "flow_map2",
    return_traj: bool = False,
    return_controls: bool = False,
    verbose: bool = False,
    show_progress: bool = True,
    low_memory: bool = True,
) -> tuple:
    """FMRG-J for FLUX image latents.

    Mirrors the 2-D ``fmrg_j`` in flow_map_tests.ipynb, adapted for FLUX:

    At each denoising step k (FLUX time t_k → t_{k+1}):
      1. Advance the latent via one Euler step of the FLUX velocity.
      2. Apply ``n_opt`` Jacobian guidance steps:
             endpoint  ≈  z - t_{k+1} * v_flux(z, t_{k+1} → 0)
             u  =  ∇_z r(endpoint)          (backprop through endpoint prediction)
             z  ←  z + (dt / n_opt) * lam * u

    Time convention: FLUX t = 1 (noise) → 0 (data); ``sample_mode`` controls
    which second timestep is passed to ``predict_vector`` for the advance step
    (same as ``make_flux_wrappers``).

    Args:
        sampler:        loaded FluxFlowMapSampler.
        r:              differentiable reward  r(z) -> (B,).
        z0:             initial noise latent, shape (B, C, H_l, W_l).
        N:              number of denoising steps.
        n_opt:          Jacobian guidance steps per denoising step.
        lam:            guidance strength (scalar).
        guidance_scale: CFG scale for ``predict_vector``.
        sample_mode:    "flow_map2" | "flow_map1" | "flow_matching".
        return_traj:    if True, also return (N+1, B, C, H_l, W_l) trajectory.
        return_controls: if True, also return (N, B, C, H_l, W_l) controls.
        verbose:        print per-step diagnostics.
        show_progress:  show a tqdm bar of length N * n_opt with outer step,
                        reward, and guidance norm.

    Returns:
        z_N:      final latent, shape (B, C, H_l, W_l).
        traj:     (optional) trajectory tensor, shape (N+1, B, C, H_l, W_l).
        controls: (optional) guidance displacement per step, same shape as traj[1:].
    """
    pe  = sampler.cached_prompt_embeds
    ppe = sampler.cached_pooled_prompt_embeds
    dt_flux = 1.0 / N
    device = z0.device
    dtype  = z0.dtype

    def _t_next_advance(t_cur: float) -> float:
        """t_next for the advance step (same logic as make_flux_wrappers)."""
        if sample_mode == "flow_map2":
            return max(0.0, t_cur - dt_flux)
        elif sample_mode == "flow_map1":
            return 0.0
        else:
            return t_cur

    ts_flux = torch.linspace(1.0, 0.0, N + 1, device=device, dtype=dtype)

    z = z0.clone()
    traj_list     = [z.detach().cpu()] if return_traj else None
    controls_list = [] if return_controls else None

    pbar = tqdm(
        total=N * n_opt, desc="FMRG", disable=not show_progress,
    )

    for k in range(N):
        t_cur  = ts_flux[k].item()
        t_next = ts_flux[k + 1].item()
        dt     = t_cur - t_next      # positive scalar

        # ── Step 1: advance via FLUX velocity ────────────────────────────────
        with torch.no_grad():
            t_cond = _t_next_advance(t_cur)
            v_adv = sampler.predict_vector(
                z, t_cur,
                prompt_embeds=pe, pooled_prompt_embeds=ppe,
                guidance_scale=guidance_scale, t_next=t_cond,
            )
            z = z - dt * v_adv
        del v_adv

        # ── Step 2: n_opt Jacobian guidance steps ────────────────────────────
        z_before = z.detach().clone()
        for _ in range(n_opt):
            z_in = z.detach().requires_grad_(True)

            # Endpoint prediction: one-step flow-map1 from t_next to 0
            # X_{t_next → 0}(z) ≈ z - t_next * v_flux(z, t_next → 0)
            v_ep = sampler.predict_vector(
                z_in, t_next,
                prompt_embeds=pe, pooled_prompt_embeds=ppe,
                guidance_scale=guidance_scale, t_next=0.0,
            )
            endpoint = z_in - t_next * v_ep   # (B, C, H_l, W_l)

            # u = J_X^T ∇r  via autograd
            # Use .backward() instead of autograd.grad(): the latter is
            # incompatible with use_reentrant=True gradient checkpointing
            # (the diffusers default), which raises RuntimeError at backward time.
            obj = r(endpoint).sum()
            obj.backward()
            u = z_in.grad.clone()

            with torch.no_grad():
                z = z.detach() + (dt / n_opt) * lam * u
                if pbar is not None:
                    pbar.set_postfix(_pbar_postfix(
                        k + 1, N, _reward_mean(r, endpoint), _flat_norm_mean(u),
                    ))
                    pbar.update(1)

            del z_in, v_ep, endpoint, obj, u

        if return_controls:
            controls_list.append((z.detach() - z_before).cpu())
        del z_before

        if verbose:
            with torch.no_grad():
                ctrl_norm = (z - z0).flatten(1).norm(dim=1).mean().item()
            print(f"step {k+1:3d}/{N}  |guidance| mean = {ctrl_norm:.4e}")

        if return_traj:
            traj_list.append(z.detach().cpu())

        _maybe_free_memory(low_memory)

    if pbar is not None:
        pbar.close()

    z_final = z.detach()

    if return_traj and return_controls:
        return z_final, torch.stack(traj_list, dim=0), torch.stack(controls_list, dim=0)
    if return_traj:
        return z_final, torch.stack(traj_list, dim=0)
    if return_controls:
        return z_final, torch.stack(controls_list, dim=0)
    return z_final


def hardflow_image(
    sampler,
    r: Optional[Callable[[torch.Tensor], torch.Tensor]],
    z0: torch.Tensor,
    *,
    N: int = 100,
    n_inner: int = 40,
    lr: float = 2.5,
    h: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    mu_init: float = 0.0,
    lr_dual: Optional[float] = None,
    rho: float = 5.0,
    rho_mult: float = 2.0,
    rho_grow_tol: float = 0.02,
    dual_every: int = 8,
    constraint_margin: Optional[float] = None,
    activation_frac: float = 0.0,
    hardflow_reg: float = 0.0,
    guidance_scale: float = 3.5,
    sample_mode: str = "flow_map2",
    endpoint_mode: str = "flow_map1",
    endpoint_map: Optional[Callable] = None,
    verbose: bool = False,
    show_progress: bool = True,
    low_memory: bool = True,
    dual_probe: Optional[list] = None,
    return_dual_h: bool = False,
):
    """HardFlow (Li, Alim & Azizan, TPAMI 2026) ported to FLUX latents.

    Port of ``hardflow_optimization`` from the paper's released code
    (github.com/azizanlab/HardFlow, branch ``image``, commit ``016adb2``,
    ``utils/methods.py:117-207``).  See ``260829-hardflow-check.md`` for the
    full code/paper audit this port is based on.

    Method (their notation on the left, FLUX convention on the right; their
    t runs 1=data->0=noise, ours runs 1=noise->0=data, so t_hf = 1 - t_flux):

        posterior mean   x + (1-t_hf) v      ->   z - t_flux * v
        map-back factor  alpha = t_hf        ->   alpha = 1 - t_flux

    At each sampling step the decision variable is the *predicted terminal
    latent* ``x_hat``, not the state.  One velocity evaluation per step is
    made under ``no_grad``; the inner loop then differentiates only ``r`` and
    ``h`` w.r.t. ``x_hat``.  The transformer is never differentiated -- this
    is the method's whole efficiency claim, and it is the structural
    difference from ``receding_horizon_flow_control``, whose inner loop
    backprops through ``endpoint_map``'s full ODE rollout.

    Terminal feasibility: at the final step ``t_next == 0`` exactly, so
    ``alpha == 1`` and the output is ``z_next_ref + (x_hat - z_pred_ref)``.
    In ``flow_map1``/``flow_map2`` modes the advance and endpoint calls both
    target t=0 at that step, so ``z_next_ref`` and ``z_pred_ref`` coincide up
    to the bf16 grid and ``z_N`` is ``x_hat`` to within that -- whatever the
    last inner solve produced is what comes out.  (Their code lands on
    ``alpha = 1.00001`` because of an epsilon in their time grid; ours hits 1
    on the nose.)  This is the mechanism behind their Proposition 1 -- though
    note that with an augmented-Lagrangian inner solve rather than a
    constrained one, feasibility is empirical, not guaranteed, in their
    implementation and in this port alike.

    DELIBERATE DEVIATIONS from their released code -- each one is a place
    their implementation departs from their own paper, so reproducing it
    verbatim would import a bug rather than a method:

      * Regularizer sign.  Their code ADDS the deviation penalty to a
        maximized objective (``methods.py:177``), which rewards drifting away
        from the nominal terminal -- the opposite of their Eq. (11).  Here it
        is subtracted, i.e. the paper's intent.
      * Regularizer scale.  Theirs uses ``.mean()`` over 3*256*256 elements,
        making the term ~1e-9 and numerically inert (their effective
        lambda_oc is 0).  Here it is a per-sample sum of squares, matching
        Eq. (11).  ``hardflow_reg`` therefore does NOT carry over from their
        ``regularization=0.0005``; it defaults to 0.0, which reproduces their
        *effective* behaviour rather than their nominal setting.
    The regularizer sign is the ONLY place this port knowingly departs from
    the authors' released code.  Decision of 2026-08-31 (Sung): the meeting
    that day settled on "keep their implementation the same", but that
    instruction was given over a summary list and the sign flip is an
    unambiguous bug -- their own Appendix B.A describes the term as
    "penalizing large control effort", and the identical quantity is
    correctly signed as a penalty in this module's own MPC objective
    (``_horizon_objective``'s control-energy term; the two are the same
    thing up to the alpha^2 change of coordinates).  Reproducing it verbatim
    would mean shipping a baseline whose regularizer pushes samples off
    distribution.  THIS MUST BE REPORTED TO SHERVIN, not left implicit.

    Everything else follows their released code, per that same decision:

      * Constraint margin.  Theirs hard-codes ``constraint_margin=0.02``,
        tightening every constraint by 0.02 while reporting against the
        untightened threshold.  Undocumented in the paper, and present in all
        four of their branches.  THIS FUNCTION's sentinel default stays 0.02,
        because this function's job is to document what they do.  run_sweep.py's
        ``RunConfig.hardflow_margin`` is **0.0 as of 2026-09-10** (09-10 call
        decision) and is always passed explicitly, so the two never conflict --
        the 0.02 here only applies to direct callers of this function.  Runs
        before 2026-09-10 used 0.02 and are not margin-comparable with runs
        after it.  It is exposed rather than hard-coded so it can be swept and
        reported; do not let it stay implicit in any results table.
      * Dual semantics.  Multipliers reset to ``mu_init`` at every sampling
        step rather than persisting, as in theirs.
      * Endpoint estimator.  Left as their single-call posterior mean; see
        ``endpoint_mode`` below.

    NOTE ON ``hardflow_reg`` vs ``lam``: opposite conventions.  This module's
    MPC ``lam`` DIVIDES the control-energy term, so larger ``lam`` means
    weaker regularization.  ``hardflow_reg`` MULTIPLIES, so larger means
    stronger.

    They ARE the same physical quantity, and the conversion (derived
    2026-09-09; an earlier note in run_sweep.py claimed ``lam <-> 1/reg``,
    wrong by a factor of ``2*dt``) is::

        hardflow_reg = 1 / (2 * lam * dt) = N / (2 * lam)

    because the applied displacement is ``D = alpha*(x_hat - z_pred_ref)``
    with stored control ``u = D/dt``, so this function's
    ``reg * alpha^2 * dev`` equals ``reg * dt^2 * ||u||^2`` while MPC's term
    is ``dt * ||u||^2 / (2*lam)``.  It DEPENDS ON N -- a reg matched at one
    step count is not matched at another.  Exact per step only: MPC sums over
    an H-step window that is re-solved and mostly discarded, so this matches
    the marginal cost of control effort, not the total.

    This function's default stays 0.0, which reproduces the authors' EFFECTIVE
    behaviour (their nominal 0.0005 is ``.mean()``-scaled to ~1e-9, i.e. inert).
    run_sweep.py's RunConfig default is 5e-4 as of 2026-09-09 and is always
    passed explicitly, so the two do not conflict -- this default only applies
    to direct callers of this function.

    Args:
        sampler:        loaded FluxFlowMapSampler.
        r:              differentiable reward r(z) -> (B,), or None to steer
                        purely from constraints.
        z0:             initial noise latent, (B, C, H_l, W_l).
        N:              number of denoising steps.
        n_inner:        inner gradient-ascent steps per denoising step
                        (their 40).
        lr:             inner step size (their 2.5 -- calibrated against a
                        pixel-space model whose velocity is internally
                        rescaled by t*999, so it does NOT transfer; tune it).
        h:              inequality constraint h(z) -> (B, C), h <= 0 feasible.
        mu_init:        value the multipliers are (re)set to at the start of
                        every sampling step.  Their code resets to 0 each
                        step; duals do not persist across steps.
        lr_dual:        dual step size.  Defaults to ``rho`` (their scheme
                        uses rho for both the penalty weight and the dual
                        step); pass a value to decouple them.
        rho:            augmented-Lagrangian quadratic penalty weight.
        rho_mult:       factor rho grows by when the violation exceeds
                        ``rho_grow_tol`` at a dual update.
        rho_grow_tol:   violation above which rho grows.
        dual_every:     inner steps between dual updates (their 8).
        constraint_margin: tightens every constraint by this amount.  None
                        (default) resolves to their hard-coded 0.02 when ``h``
                        is given and to 0.0 when it is not; see the note above.
        activation_frac: fraction of sampling steps to skip before enabling
                        the constraint.  Their code hard-codes 0.5 -- the
                        posterior-mean terminal estimate is unreliable early,
                        so evaluating h against it does more harm than good.
                        Exposed as a knob per the 2026-08-31 decision, and
                        defaulted to 0.0 (control from step 0) for parity with
                        our MPC setting.  Set 0.5 to reproduce their schedule.
                        Sweep it -- at small N the skip costs a large fraction
                        of the available control steps.
        hardflow_reg:   weight on alpha^2 * ||x_hat - x_bar_N||^2.  See above.
        sample_mode:    "flow_map2" | "flow_map1" | "flow_matching" for the
                        advance step (as in ``make_flux_wrappers``).
        endpoint_mode:  how the terminal estimate x_bar_N is computed.  This
                        is a LOAD-BEARING choice, not a tuning knob -- see the
                        note below.
                          "flow_map1"     (default) one call with t_next=0:
                                          the LoRA's learned direct jump to
                                          data.  One NFE.
                          "flow_matching" one call with t_next=t_cur: the
                                          instantaneous velocity, i.e. the
                                          plain Euler extrapolation their
                                          rectified-flow model provides.  The
                                          stricter reading of their spec.
                        Note ``sample_mode`` is deliberately NOT reused here:
                        with sample_mode="flow_map2" (our RunConfig default)
                        there is no single-call endpoint at all -- flow_map2
                        is a one-step-ahead map, so reaching t=0 needs a
                        rollout.  HardFlow's premise is one NFE per step, so
                        its endpoint must be a single call regardless of how
                        the advance step is taken.
        endpoint_map:   optional Callable(s_mpc, z, grad=False) -> (B, *) with
                        ``make_flux_wrappers``' signature (s_mpc = 1 - t_flux).
                        When given, it REPLACES the single-call estimator and
                        ``endpoint_mode`` is ignored.  This exists for one
                        purpose: the ablation that isolates the algorithm from
                        the estimator.  MPC's endpoint is a full differentiable
                        rollout while HardFlow's is a one-call posterior mean,
                        so a head-to-head confounds "which control algorithm"
                        with "which endpoint estimate".  Passing MPC's
                        ``endpoint_map`` here runs HardFlow on MPC's estimator
                        and separates the two.  It is NOT the default: doing
                        this per inner step would reintroduce the transformer
                        rollout HardFlow exists to avoid, so it is only
                        affordable here because HardFlow evaluates the endpoint
                        once per OUTER step, not once per inner step.

    Returns:
        z_N:      final controlled latent,            (B, C, H_l, W_l).
        traj:     full trajectory,                    (N+1, B, C, H_l, W_l).
        controls: per-step displacement from nominal, (N, B, C, H_l, W_l).
        mu_final: final multipliers,                  (B, C), or None.
        mu_hist:  multipliers after each step's last
                  dual update,                        (N, B, C), or None.

        Matches ``receding_horizon_flow_control``'s contract so run_sweep's
        constraint bookkeeping (mu_trace CSV, constraint plots) works unchanged.
    """
    if r is None and h is None:
        raise ValueError(
            "hardflow_image: both r and h are None -- there is nothing to "
            "optimize, and every inner step would be a no-op returning the "
            "uncontrolled rollout. Pass a reward, a constraint, or both."
        )
    if not 0.0 <= activation_frac < 1.0:
        raise ValueError(
            f"activation_frac must be in [0, 1), got {activation_frac}. "
            f"At 1.0 the constraint never activates."
        )
    if n_inner < 1:
        raise ValueError(f"n_inner must be >= 1, got {n_inner}.")
    # Sentinel default: 0.02 (the authors' hard-coded value) when there is a
    # constraint to tighten, 0.0 when there is not. A plain 0.02 default would
    # make every unconstrained run trip the guard below, which exists to catch
    # a caller who set a margin believing it does something -- not to reject
    # the default on a run that simply has no constraint.
    if constraint_margin is None:
        constraint_margin = 0.02 if h is not None else 0.0
    elif h is None and constraint_margin != 0.0:
        raise ValueError(
            f"constraint_margin={constraint_margin} was given explicitly with "
            f"h=None. The margin only tightens a constraint; with no "
            f"constraint it silently does nothing."
        )

    # Explicit gradient ascent on the quadratic deviation penalty evolves the
    # error as e <- e * (1 - 2*lr*hardflow_reg*alpha^2), which converges only
    # while lr*hardflow_reg*alpha^2 < 1. alpha reaches 1 at the final step, so
    # lr*hardflow_reg >= 1 guarantees divergence there (verified: lr=0.5 with
    # reg=100 amplifies ~99x per inner step, reaching inf within one run).
    # Raised rather than clamped, per the loud-fail rule -- and rather than
    # left to surface as a NaN after a GPU-hour.
    if hardflow_reg != 0.0 and lr * hardflow_reg >= 1.0:
        raise ValueError(
            f"lr={lr} * hardflow_reg={hardflow_reg} = {lr * hardflow_reg} >= 1: "
            f"the inner ascent on the deviation penalty is unstable and will "
            f"diverge to inf/NaN by the final step (alpha -> 1). Reduce either. "
            f"With the authors' lr=2.5 this bounds hardflow_reg < 0.4."
        )

    if lr_dual is None:
        lr_dual = rho

    pe = sampler.cached_prompt_embeds
    ppe = sampler.cached_pooled_prompt_embeds
    device = z0.device
    dtype = z0.dtype
    B = z0.shape[0]

    # Same grid construction as fmrg_j_image. dt is read OFF this grid rather
    # than computed as 1/N: in bf16 the linspace spacing is silently
    # non-uniform, and stepping one grid while regularizing against another
    # differentiates a different map than the one being integrated.
    ts_flux = torch.linspace(1.0, 0.0, N + 1, device=device, dtype=dtype)
    dt_flux = 1.0 / N  # nominal spacing, only for _t_next_advance's lookahead

    def _t_next_advance(t_cur: float) -> float:
        if sample_mode == "flow_map2":
            return max(0.0, t_cur - dt_flux)
        elif sample_mode == "flow_map1":
            return 0.0
        elif sample_mode == "flow_matching":
            return t_cur
        raise ValueError(f"Unsupported sample_mode: {sample_mode}")

    def _t_next_endpoint(t_cur: float) -> float:
        if endpoint_mode == "flow_map1":
            return 0.0
        elif endpoint_mode == "flow_matching":
            return t_cur
        raise ValueError(f"Unsupported endpoint_mode: {endpoint_mode}")

    def _v(z, t_cur: float, t_next: float):
        return sampler.predict_vector(
            z, t_cur,
            prompt_embeds=pe, pooled_prompt_embeds=ppe,
            guidance_scale=guidance_scale, t_next=t_next,
        )

    activation_step = int(activation_frac * N)

    C = None
    mu_t = None
    if h is not None:
        with torch.no_grad():
            h0 = h(z0)
            C = h0.shape[-1] if h0.dim() > 1 else 1
        del h0

    z = z0.detach().clone()
    traj = [z.clone()]
    controls = []
    mu_history: list = []
    h_dual_history: list = []   # the h each per-step ALM actually consumed

    pbar = tqdm(total=N * n_inner, desc="HardFlow", disable=not show_progress)

    for k in range(N):
        t_cur = ts_flux[k].item()
        t_next = ts_flux[k + 1].item()
        dt = t_cur - t_next
        alpha = 1.0 - t_next  # = t_hf at the next step

        # One velocity evaluation per step, no gradients. This is the only
        # place the transformer is touched.
        with torch.no_grad():
            v_adv = _v(z, t_cur, _t_next_advance(t_cur))
            z_next_ref = z - dt * v_adv                      # nominal x_bar_{i+1}
            del v_adv
            if endpoint_map is not None:
                # Ablation path: MPC's rollout estimator instead of the
                # one-call posterior mean. Affordable only because this runs
                # once per OUTER step, not once per inner step.
                s_mpc = torch.full((B,), 1.0 - t_cur, device=device, dtype=dtype)
                z_pred_ref = endpoint_map(s_mpc, z, grad=False)
                del s_mpc
            else:
                v_ep = _v(z, t_cur, _t_next_endpoint(t_cur))
                z_pred_ref = z - t_cur * v_ep                # nominal x_bar_N
                del v_ep

        use_constraint = h is not None and k >= activation_step

        mu_step = None
        rho_step = rho
        if h is not None:
            mu_step = torch.full((B, C), float(mu_init), dtype=dtype, device=device)

        # With no reward and the constraint not yet active there is nothing in
        # the objective: the gradient would be identically zero (and .backward()
        # would raise, since obj has no graph). Skip the inner solve and take
        # the nominal step. This is the `--reward none` + early-step case, which
        # is a normal configuration for us, not an edge case.
        inner_active = (r is not None) or use_constraint

        # Inner solve: optimize the predicted terminal latent directly.
        x_hat = z_pred_ref.detach().clone()
        hx_last = None
        r_last = float("nan")
        for j in range(n_inner if inner_active else 0):
            x_hat = x_hat.detach().requires_grad_(True)

            obj = z_pred_ref.new_zeros(B)
            if r is not None:
                r_val = r(x_hat)
                obj = obj + r_val
                r_last = r_val.detach().mean().item()
                del r_val

            hx = None
            if use_constraint:
                hx = h(x_hat)
                if hx.dim() == 1:
                    hx = hx.unsqueeze(-1)
                hx = hx + constraint_margin
                viol = hx.clamp_min(0.0)
                obj = obj - (mu_step * hx).sum(-1) - 0.5 * rho_step * viol.pow(2).sum(-1)

            if hardflow_reg != 0.0:
                dev = (x_hat - z_pred_ref).flatten(1).pow(2).sum(1)
                obj = obj - hardflow_reg * (alpha ** 2) * dev

            obj_sum = obj.sum()
            if not obj_sum.requires_grad:
                # Nothing in the objective actually depends on x_hat, so the
                # gradient is identically zero and every remaining inner step
                # is a no-op. This is reachable and normal, not defensive
                # padding: run_sweep's reward="none" returns a closure that
                # emits `z.new_zeros(...)` with no graph, so any
                # activation_frac > 0 leaves the early steps with a
                # gradient-free objective. Take the nominal step rather than
                # letting .backward() raise "does not require grad".
                del obj, obj_sum
                if pbar is not None:
                    pbar.update(n_inner - j)
                break

            # .backward() rather than autograd.grad(): the latter is
            # incompatible with use_reentrant=True gradient checkpointing
            # (the diffusers default), same as in fmrg_j_image.
            obj_sum.backward()
            grad = x_hat.grad
            with torch.no_grad():
                x_hat = x_hat.detach() + lr * grad          # ascent
            del grad, obj, obj_sum

            if use_constraint:
                hx_last = hx.detach()
                if (j + 1) % dual_every == 0:
                    with torch.no_grad():
                        hx_cur = h(x_hat)
                        if hx_cur.dim() == 1:
                            hx_cur = hx_cur.unsqueeze(-1)
                        hx_cur = hx_cur + constraint_margin
                        mu_step = (mu_step + lr_dual * hx_cur).clamp_min(0.0)
                        if (hx_cur > rho_grow_tol).any():
                            rho_step = rho_step * max(1.0, rho_mult)
                        hx_last = hx_cur
                        # Diagnostic hook (2026-09-07). mu_history records only
                        # the multipliers after the LAST dual update of each
                        # sampling step, so a multiplier that rises and is then
                        # driven back to zero within the step is indistinguishable
                        # from one that never left zero -- which is exactly the
                        # ambiguity in the 20260904_231018_lpr2c grid, where
                        # HardFlow's mu_1 reads 0.000 at every step. Default None
                        # means no allocation and no behaviour change.
                        if dual_probe is not None:
                            dual_probe.append({
                                "step": k, "inner_j": j, "alpha": float(alpha),
                                "rho": float(rho_step),
                                # h AFTER the margin tightening, i.e. the value
                                # the dual update actually consumed.
                                "h": hx_cur.mean(0).detach().cpu().tolist(),
                                "mu": mu_step.mean(0).detach().cpu().tolist(),
                            })
                del hx

            if pbar is not None:
                with torch.no_grad():
                    # r_last is reused from the objective above rather than
                    # re-evaluated: r costs a VAE decode + reward forward, and
                    # recomputing it here would double the inner-loop cost.
                    pbar.set_postfix(_pbar_postfix(
                        k + 1, N, r_last,
                        _flat_norm_mean(x_hat - z_pred_ref),
                    ))
                pbar.update(1)

        if pbar is not None and not inner_active:
            pbar.update(n_inner)

        # Map the optimized terminal back to a state update.
        with torch.no_grad():
            x_hat = x_hat.detach()
            z = z_next_ref + alpha * (x_hat - z_pred_ref)
            # Store the control in the same units as
            # receding_horizon_flow_control's u_star = (x1_star - x)/dt - v,
            # i.e. a velocity-like perturbation, so ctrl_energy is comparable
            # across control types in summary.csv.
            controls.append(((z - z_next_ref) / dt).detach().clone())

        if mu_step is not None:
            mu_history.append(mu_step.detach().clone())
            # The h this step's ALM actually consumed, at HardFlow's own
            # decision variable x_hat. Recorded for the same reason as the
            # MPC's h_dual_history (see receding_horizon_flow_control): without
            # it, run_sweep can only reconstruct h at the REALIZED latent
            # post-hoc, which is a different quantity, and the MPC-vs-HardFlow
            # constraint panel has no comparable series for this arm.
            #
            # Costs nothing: hx_last is already computed by the inner loop and
            # was being dropped at the `del` below. NaN when the constraint is
            # not yet active (activation_frac > 0), never 0.0, which would read
            # as "the ALM saw h = 0".
            #
            # MARGIN IS REMOVED before recording. The inner loop works on the
            # TIGHTENED h (`hx = h(x_hat) + constraint_margin`, HardFlow's
            # hard-coded 0.02), because that is what their ALM optimizes. But
            # the MPC arm has no margin, so recording the tightened value would
            # put a constant +margin offset into every cross-arm constraint
            # plot and make HardFlow look 0.02 worse than it is. What is stored
            # here is therefore the TRUE h at x_hat; the multiplier in
            # mu_history is still the one driven by the tightened value.
            h_dual_history.append(
                (hx_last.detach() - constraint_margin).clone()
                if hx_last is not None
                else torch.full_like(mu_step, float("nan"))
            )
            mu_t = mu_step

        traj.append(z.detach().clone())

        if verbose:
            with torch.no_grad():
                msg = (f"step {k+1:3d}/{N}  alpha={alpha:.3f}  "
                       f"|dx_hat|={_flat_norm_mean(x_hat - z_pred_ref):.4e}")
                if hx_last is not None:
                    msg += f"  h_max={hx_last.max().item():+.4e}"
                print(msg)

        del x_hat, z_pred_ref, z_next_ref, hx_last
        _maybe_free_memory(low_memory)

    if pbar is not None:
        pbar.close()

    mu_hist = torch.stack(mu_history, dim=0) if mu_history else None
    h_dual_hist = torch.stack(h_dual_history, dim=0) if h_dual_history else None
    out = (
        z.detach(),
        torch.stack(traj, dim=0),
        torch.stack(controls, dim=0),
        mu_t,
        mu_hist,
    )
    # Opt-in 6th element so every existing 5-tuple call site is untouched --
    # same contract as receding_horizon_flow_control's return_dual_h.
    return out + (h_dual_hist,) if return_dual_h else out


def trajectory_rewards(
    traj: torch.Tensor,
    r: Callable[[torch.Tensor], torch.Tensor],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Evaluate reward r along a stored trajectory.

    Trajectory entries may live on CPU (e.g. from ``fmrg_j_image`` with
    ``return_traj=True``) while r typically needs GPU tensors.

    Args:
        traj:   (T, B, ...) latent trajectory on CPU or GPU.
        r:      reward callable, r(z) -> (B,).
        device: target device for evaluation (inferred from traj if on GPU).
        dtype:  target dtype (defaults to traj dtype).

    Returns:
        Tensor of shape (T,) on CPU with mean reward per time step.
    """
    rs = []
    for t in range(traj.shape[0]):
        z = traj[t]
        if device is None and z.is_cuda:
            dev, dt = z.device, z.dtype
        else:
            dev = device if device is not None else torch.device("cpu")
            dt = dtype if dtype is not None else z.dtype
        if z.device != dev or (dtype is not None and z.dtype != dt):
            z = z.to(device=dev, dtype=dt)
        with torch.no_grad():
            rs.append(r(z).mean().detach().cpu())
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.stack(rs)


# ─────────────────────────────────────────────────────────────────────────────
# Notebook snippet (paste into a cell after loading the sampler)
# ─────────────────────────────────────────────────────────────────────────────
# from image_mpc_edited_sung import run_image_mpc_demo, make_flux_wrappers
#
# img_unc, img_mpc, traj, controls = run_image_mpc_demo(
#     sampler, PROMPT,
#     height=256, width=256,
#     n_mpc_steps=4, h_horizon=2, lam=1.0, n_inner=20,
# )
#
# -- or build wrappers manually and call receding_horizon_flow_control --
# v_theta, r, endpoint_map = make_flux_wrappers(sampler, PROMPT)
# z_mpc, traj, controls, mu, mu_hist = receding_horizon_flow_control(
#     v_theta, r, z0, N=4, H=2, lam=1.0,
#     optimizer="adam", n_iters=20, lr=1e-3, verbose=True,
# )
#
# -- n_alt (Step 5, this copy only): alternate primal/dual n_alt times per
#    fixed window before it shifts --
# z_mpc, traj, controls, mu, mu_hist = receding_horizon_flow_control(
#     v_theta, r, z0, N=10, H=4, lam=1.0, h=h_fn, mu_init=0.5, lr_dual=0.05,
#     n_alt=2, optimizer="adam", n_iters=20, lr=1e-3, verbose=True,
# )
