"""
image_mpc.py — Receding-Horizon Path-Space MPC for FLUX FlowMap images
=======================================================================

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

      from image_mpc import (
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
    reward: float,
    control: float,
    mu: Optional[float] = None,
) -> dict:
    d: dict = {
        "step": f"{outer}/{outer_total}",
        "r": f"{reward:.4f}",
        "|u|": f"{control:.4e}",
    }
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
        z_r = endpoint_map(t_horizon.expand(x_H.shape[0]), x_H, grad=True)
        z_h = endpoint_map(t_horizon.expand(x_H.shape[0]), x_H, grad=False)
    else:
        z_r = z_h = x_H

    obj = energy - r(z_r)  # (B,)

    # Lagrangian penalty for inequality constraint h(z) <= 0
    if h is not None and mu is not None:
        hx = h(z_h)
        if hx.dim() == 1:
            hx = hx.unsqueeze(-1)
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
    outer_step: Optional[int] = None,
    outer_total: Optional[int] = None,
    low_memory: bool = True,
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
                    _reward_mean(r, x_traj[-1]), _flat_norm_mean(resid0),
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
                        _reward_mean(r, x_traj[-1]), _flat_norm_mean(resid0),
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
    optimizer: str = "lbfgs",
    n_iters: int = 50,
    lr: float = 0.1,
    warmstart: bool = False,
    drift_detach: bool = False,
    verbose: bool = False,
    show_progress: bool = True,
    low_memory: bool = True,
    device: Optional[torch.device] = None,
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
                       reward, and control norm.
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
    if device is None:
        device = x0.device
    x0 = x0.to(device)
    dt = 1.0 / N
    t_grid = torch.linspace(0.0, 1.0, N + 1, device=device, dtype=x0.dtype)

    x = x0.detach().clone()
    traj = [x.clone()]
    controls = []
    mu_history: list = []

    mu_t = None
    if h is not None:
        with torch.no_grad():
            h0 = h(x0)
            C = h0.shape[-1] if h0.dim() > 1 else 1
        mu_t = torch.full(
            (x0.shape[0], C), float(mu_init), dtype=x0.dtype, device=device
        )

    prev_window = None   # previous window's optimized states (warmstart only)

    pbar = tqdm(
        total=N * n_iters, desc="MPC", disable=not show_progress,
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

        solved = solve_horizon_problem(
            x, v_theta, r, h, t0_idx=j, H=H_j, dt=dt,
            lam=lam, mu=mu_t, t_grid=t_grid,
            constraint_on=constraint_on, endpoint_map=endpoint_map,
            optimizer=optimizer, n_iters=n_iters, lr=lr,
            x_init=x_init_j, return_full=warmstart,
            drift_detach=drift_detach, v_cur=v_j,
            pbar=pbar, outer_step=j + 1, outer_total=N,
            low_memory=low_memory,
        )
        if warmstart:
            x1_star, xH_star, prev_window = solved
        else:
            x1_star, xH_star = solved
        del solved, x_init_j

        # Projected dual ascent
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

        if mu_t is not None:
            mu_history.append(mu_t.clone())

        # Apply first MPC step, reusing the drift computed above
        with torch.no_grad():
            u_star = (x1_star - x) / dt - v_j
            x = x + dt * (v_j + u_star)

        traj.append(x.clone())
        controls.append(u_star.clone())

        if pbar is not None:
            mu_mean_val = mu_t.mean().item() if mu_t is not None else None
            if constraint_on == "endpoint" and endpoint_map is not None:
                with torch.no_grad():
                    x_disp = endpoint_map(t_grid[j + 1].expand(x.shape[0]), x)
            else:
                x_disp = x
            pbar.set_postfix(_pbar_postfix(
                j + 1, N, _reward_mean(r, x_disp), _flat_norm_mean(u_star), mu_mean_val,
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
    return x, torch.stack(traj, dim=0), torch.stack(controls, dim=0), mu_t, mu_hist


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
# from image_mpc import run_image_mpc_demo, make_flux_wrappers
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
