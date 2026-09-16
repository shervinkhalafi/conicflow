#!/usr/bin/env python3
"""
run_sweep.py — Sweep MPC, HardFlow, and FMRG controllers on two-moons.

Iterates over every combination of the supplied hyperparameter lists,
runs each config, logs metrics to a CSV, and saves all plots as a single
PNG per run.

Usage examples
--------------
# MPC: sweep H, constraint_on, lam, n_iters, and inner lr (single line)
python run_sweep.py --ckpt /path/to/model.pkl --methods mpc --N 16 --H 4 8 16 --constraint_on horizon endpoint --lam 0.1 0.5 1.0 2.0 --mpc_n_iters 10 20 50 --mpc_optimizer adam --mpc_lr 0.01 0.05 --output_dir results/mpc_sweep

# HardFlow: sweep lam and outer iterations
python run_sweep.py --ckpt /path/to/model.pkl \\
    --methods hardflow \\
    --lam 0.5 1.0 2.0 \\
    --hf_n_outer 5 10

# FMRG (J and E variants): sweep guidance iterations and strength
python run_sweep.py --ckpt /path/to/model.pkl \\
    --methods fmrg \\
    --lam 1.0 2.0 5.0 \\
    --fmrg_n_iters 1 3 5 \\
    --fmrg_variant J E

# All three together
python run_sweep.py --ckpt /path/to/model.pkl \\
    --methods mpc hardflow fmrg \\
    --H 5 10 --constraint_on endpoint --lam 2.0 \\
    --fmrg_n_iters 3 \\
    --output_dir results/my_sweep

# Dry-run: print all configs without running
python run_sweep.py --ckpt /path/to/model.pkl --dry_run

# Pin to a single GPU (overrides --device)
python run_sweep.py --ckpt /path/to/model.pkl --gpu 1 --methods mpc --dry_run
"""

import argparse
import csv
import datetime
import itertools
import sys
import textwrap
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ── local imports (same directory as this script) ────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sampling_utils import sample_two_moons
from controllers import (
    receding_horizon_flow_control,
    fmrg_j,
    hardflow_sample,
    solve_constrained_subproblem,
)
from visualization_utils import visualize_control_results


# =============================================================================
# Model (FlowMapMLP) — identical to lowdim_comparisons.ipynb cell 3
# =============================================================================

class FlowMapMLP(nn.Module):
    """Flow-map model: X(s, t, x) = x + (t-s) * phi(s, t, x/rescale)."""

    def __init__(
        self,
        n_hidden: int = 3,
        n_neurons: int = 256,
        input_dim: int = 4,
        output_dim: int = 2,
        rescale: float = 1.0,
    ):
        super().__init__()
        self.rescale = rescale
        layers: list = [nn.Linear(input_dim, n_neurons), nn.GELU()]
        for _ in range(n_hidden):
            layers += [nn.Linear(n_neurons, n_neurons), nn.GELU()]
        layers += [nn.Linear(n_neurons, output_dim)]
        self.phi_mlp = nn.Sequential(*layers)

    def calc_b(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Velocity b(t, x) = phi(t, t, x) — diagonal of the flow map."""
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        st = torch.stack([t, t], dim=-1)
        inp = torch.cat([st, x / self.rescale], dim=-1)
        return self.rescale * self.phi_mlp(inp)

    def forward(
        self, s: torch.Tensor, t: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Full flow map X(s, t, x)."""
        if s.dim() == 0:
            s = s.expand(x.shape[0])
            t = t.expand(x.shape[0])
        st = torch.stack([s, t], dim=-1)
        inp = torch.cat([st, x / self.rescale], dim=-1)
        phi = self.rescale * self.phi_mlp(inp)
        return x + (t - s).unsqueeze(-1) * phi


def _flax_to_pt(flax_dense: dict, pt_linear: nn.Linear) -> None:
    W = torch.from_numpy(np.array(flax_dense["kernel"]))
    b = torch.from_numpy(np.array(flax_dense["bias"]))
    pt_linear.weight.data.copy_(W.T)
    pt_linear.bias.data.copy_(b)


def load_model(ckpt_path: str, device: torch.device) -> FlowMapMLP:
    """Load a FlowMapMLP from a JAX/Flax msgpack .pkl checkpoint."""
    from flax.serialization import msgpack_restore

    print(f"Loading checkpoint: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        raw = msgpack_restore(f.read())

    rescale = float(np.std(sample_two_moons(500_000)))
    print(f"  RESCALE = {rescale:.4f}  (std of two-moons training data)")

    ema_key = list(raw["ema_params"].keys())[0]
    phi = raw["ema_params"][ema_key]["params"]["flow_map"]["phi_mlp"]
    dense_names = sorted(phi.keys())                       # Dense_0, Dense_1, ...
    n_dense = len(dense_names)
    n_hidden = n_dense - 2                                 # first + n_hidden hidden + output
    n_neurons = int(phi[dense_names[0]]["bias"].shape[-1])
    input_dim = int(phi[dense_names[0]]["kernel"].shape[0])
    output_dim = int(phi[dense_names[-1]]["bias"].shape[-1])

    model = FlowMapMLP(
        n_hidden=n_hidden,
        n_neurons=n_neurons,
        input_dim=input_dim,
        output_dim=output_dim,
        rescale=rescale,
    ).to(device)

    dense_to_seq = {name: 2 * k for k, name in enumerate(dense_names)}
    for dname, seq_idx in dense_to_seq.items():
        _flax_to_pt(phi[dname], model.phi_mlp[seq_idx])

    model.eval()
    step = int(raw.get("step", -1))
    print(f"  Loaded step={step}  arch: {n_hidden} hidden × {n_neurons}  in={input_dim}  out={output_dim}")
    return model


# =============================================================================
# Reward and constraint (hardcoded to match lowdim_comparisons.ipynb)
# =============================================================================

_N_CON = 5   # number of constraint components reported per-column
_REWARD_SIGMA  = 0.35   # RBF bandwidth (data coordinates)
_N_REF         = 400    # reference points on the upper moon
# 5-constraint problem: two disk containments, two horizontal bounds, one
# slanted half-plane. All five verified non-redundant (each removes 9-48% of
# the region the other four allow). All are unit-gradient signed distances.
_C1 = torch.tensor([-1.10, 0.55]);  _R1 = 0.55
_C2 = torch.tensor([-1.50, 0.40]);  _R2 = 0.50
_Y_HI = 0.55
_Y_LO = 0.24
_N5   = torch.tensor([0.53, -0.85]) / torch.tensor([0.53, -0.85]).norm()
_D5   = -0.86

_OFFTOP_CENTER = torch.tensor([-0.1, 1.75])  # ball centre, off the upper-moon's top
_OFFTOP_SIGMA  = 0.5                          # ball bandwidth


def _build_reward_fn(device: torch.device, reward_type: str = "upper_moon") -> Callable:
    """Build the terminal reward r(x) -> (B,).

    Args:
        device:      torch device for any precomputed reference tensors.
        reward_type: "upper_moon"   — RBF envelope over the upper moon (default).
                     "linear"       — r(x) = x[:, 1], unbounded in y.
                     "off_top_ball" — single Gaussian bump centred off the
                                      upper moon's top, entirely off the data
                                      manifold; a harder target than
                                      "upper_moon" for gradient-only guidance.
                     "none"         — r(x) = 0 for all x; use with a constraint
                                      to isolate pure feasibility-seeking
                                      behaviour with no reward pressure.
    """
    if reward_type == "upper_moon":
        return _build_upper_moon_reward_fn(device)
    if reward_type == "linear":
        def reward(x: torch.Tensor) -> torch.Tensor:
            return x[:, 1]
        return reward
    if reward_type == "off_top_ball":
        return _build_off_top_ball_reward_fn(device)
    if reward_type == "none":
        def reward(x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return reward
    raise ValueError(
        f"Unknown reward_type: {reward_type!r}  "
        "(expected 'upper_moon', 'linear', 'off_top_ball', or 'none')"
    )


def _build_upper_moon_reward_fn(device: torch.device) -> Callable:
    """Upper-moon RBF reward: r(x) ≈ 1 on upper moon, → 0 elsewhere."""
    all_pts, labels = sample_two_moons(50_000, return_labels=True)
    upper = torch.tensor(all_pts[labels == 0], dtype=torch.float32)
    perm = torch.randperm(upper.shape[0])[:_N_REF]
    ref = upper[perm].to(device)

    def reward(x: torch.Tensor) -> torch.Tensor:
        r = ref.to(x.device, x.dtype)
        d2 = ((x[:, None] - r[None]) ** 2).sum(-1)   # (B, N_REF)
        return torch.exp(-d2.min(1).values / (2.0 * _REWARD_SIGMA ** 2))

    return reward


def _build_off_top_ball_reward_fn(device: torch.device) -> Callable:
    """Single Gaussian bump off the top of the upper moon (see reward_offtop_ball_v3.png)."""
    center = _OFFTOP_CENTER.to(device)

    def reward(x: torch.Tensor) -> torch.Tensor:
        c = center.to(x.device, x.dtype)
        d2 = ((x - c) ** 2).sum(-1)
        return torch.exp(-d2 / (2.0 * _OFFTOP_SIGMA ** 2))

    return reward


def _build_constraint_fn(device: torch.device) -> Callable:
    """h(x) <= 0 encodes five constraints:
        h1: ||x - c1|| <= 0.55            (disk containment)
        h2: ||x - c2|| <= 0.50            (disk containment)
        h3: x_2 <= 0.55                   (upper bound)
        h4: x_2 >= 0.24                   (lower bound)
        h5: 0.53*x_1 - 0.85*x_2 <= -0.86  (slanted half-plane)
    Feasible set = lens of the two disks, clipped by the band and half-plane.
    """
    def h(x: torch.Tensor) -> torch.Tensor:
        c1 = _C1.to(x.device, x.dtype); c2 = _C2.to(x.device, x.dtype)
        n5 = _N5.to(x.device, x.dtype)
        return torch.stack([
            torch.linalg.norm(x - c1, dim=-1) - _R1,
            torch.linalg.norm(x - c2, dim=-1) - _R2,
            x[:, 1] - _Y_HI,
            _Y_LO - x[:, 1],
            x @ n5 - _D5,
        ], dim=-1)

    return h


# =============================================================================
# RunConfig
# =============================================================================

@dataclass
class RunConfig:
    method: str        = "mpc"      # "mpc" | "fmrg"
    N: int             = 20         # Euler discretization steps

    # ── shared ────────────────────────────────────────────────────────────────
    lam: float         = 2.0        # λ — MPC control weight / HardFlow λ_oc / FMRG guidance
    batch_size: int    = 512
    seed: int          = 42
    constraint_margin: float = 0.0  # tighten to h(x) <= -margin for CONTROL only
                                    # (metrics are always scored against h(x) <= 0)

    # ── MPC ───────────────────────────────────────────────────────────────────
    H: int             = 10         # planning horizon (steps)
    constraint_on: str = "horizon"  # "horizon" | "endpoint"
    lr_dual: float     = 0.1        # dual-ascent step size (0 → frozen multiplier)
    mu_init: float     = 0.0        # initial Lagrange multiplier value
    mpc_n_iters: int   = 50         # inner-solver iterations per primal-dual round
    mpc_optimizer: str = "adam"     # "adam" | "lbfgs"
    mpc_lr: float      = 0.05       # inner-solver learning rate
    mpc_n_dual_steps: int = 1       # primal-dual rounds per MPC step (in-place re-solve)
    mpc_warm_start_across_steps: bool = False  # shift previous step's trajectory as warm start
    mpc_commit_final_window: bool = False      # commit whole final-window solve instead of receding
    mpc_final_window_dual_mult: int = 1        # k: n_dual_steps multiplier for that final window
    mpc_constraint_penalty: str = "dual"       # "dual" (plain Lagrangian) | "alm"
    mpc_dual_adaptive: bool = False # scale-normalised (per-constraint) dual step
    mpc_rho: float     = 1.0        # initial ALM penalty (mpc_constraint_penalty="alm")
    mpc_rho_scale: float = 2.0      # ALM penalty growth per primal-dual round

    # ── HardFlow ──────────────────────────────────────────────────────────────
    hf_optimizer: str  = "lbfgs"    # "lbfgs" | "adam"
    hf_n_outer: int    = 5          # augmented-Lagrangian outer iterations
    hf_n_inner: int    = 20         # inner solver iterations per outer step
    hf_lr: float       = 0.1        # inner solver learning rate
    hf_rho: float      = 1.0        # initial ALM penalty ρ
    hf_rho_scale: float = 2.0       # penalty scaling per outer step
    hf_warm_start: bool = True       # warm-start inner solver from previous step

    # ── FMRG ──────────────────────────────────────────────────────────────────
    fmrg_n_iters: int  = 3          # guidance steps per Euler step
    fmrg_variant: str  = "J"        # "J" (Jacobian) | "E" (Euclidean)


def _fmt_num(v: float) -> str:
    """Compact numeric string for titles and filenames."""
    if v == int(v):
        return str(int(v))
    return f"{v:g}"


def _fmt_num_fname(v: float) -> str:
    """Filesystem-safe numeric string (0.1 → 0p1)."""
    return _fmt_num(v).replace(".", "p").replace("-", "m")


def _config_plot_title(cfg: RunConfig) -> str:
    """Human-readable title listing swept / method-specific hyperparameters."""
    parts = [cfg.method.upper(), f"N={cfg.N}", f"λ={_fmt_num(cfg.lam)}"]
    if cfg.method == "mpc":
        parts += [
            f"H={cfg.H}",
            f"constraint={cfg.constraint_on}",
            f"opt={cfg.mpc_optimizer}",
            f"n_iters={cfg.mpc_n_iters}",
            f"lr={_fmt_num(cfg.mpc_lr)}",
            f"n_dual={cfg.mpc_n_dual_steps}",
            f"warm_start_across_steps={cfg.mpc_warm_start_across_steps}",
        ]
        if cfg.mpc_commit_final_window:
            parts.append(f"commit_final_window(k={cfg.mpc_final_window_dual_mult})")
        if cfg.mpc_constraint_penalty == "alm":
            parts.append(
                f"penalty=alm(ρ={_fmt_num(cfg.mpc_rho)},ρ_scale={_fmt_num(cfg.mpc_rho_scale)})"
            )
        else:
            parts.append(
                f"penalty=dual(lr_dual={_fmt_num(cfg.lr_dual)}"
                + (", adaptive" if cfg.mpc_dual_adaptive else "") + ")"
            )
    elif cfg.method == "hardflow":
        parts += [
            f"opt={cfg.hf_optimizer}",
            f"n_outer={cfg.hf_n_outer}",
            f"n_inner={cfg.hf_n_inner}",
            f"lr={_fmt_num(cfg.hf_lr)}",
            f"ρ={_fmt_num(cfg.hf_rho)}",
            f"ρ_scale={_fmt_num(cfg.hf_rho_scale)}",
            f"warm_start={cfg.hf_warm_start}",
        ]
    elif cfg.method == "fmrg":
        parts += [
            f"n_iters={cfg.fmrg_n_iters}",
            f"variant={cfg.fmrg_variant}",
        ]
    return "  |  ".join(parts)


def _config_plot_stem(cfg: RunConfig) -> str:
    """Filesystem-safe filename stem encoding the run configuration."""
    lam = _fmt_num_fname(cfg.lam)
    if cfg.method == "mpc":
        if cfg.mpc_constraint_penalty == "alm":
            penalty_tag = f"alm_rho{_fmt_num_fname(cfg.mpc_rho)}_rhoScale{_fmt_num_fname(cfg.mpc_rho_scale)}"
        else:
            penalty_tag = f"dual_lrDual{_fmt_num_fname(cfg.lr_dual)}"
            if cfg.mpc_dual_adaptive:
                penalty_tag += "_adaptive"
        ws_tag = "ws" if cfg.mpc_warm_start_across_steps else "nows"
        commit_tag = f"commitK{cfg.mpc_final_window_dual_mult}" if cfg.mpc_commit_final_window else "nocommit"
        return (
            f"mpc_lam{lam}_H{cfg.H}_{cfg.constraint_on}"
            f"_{cfg.mpc_optimizer}_ni{cfg.mpc_n_iters}"
            f"_lr{_fmt_num_fname(cfg.mpc_lr)}"
            f"_nd{cfg.mpc_n_dual_steps}"
            f"_{penalty_tag}_{ws_tag}_{commit_tag}"
        )
    if cfg.method == "hardflow":
        ws = "ws" if cfg.hf_warm_start else "nows"
        return (
            f"hardflow_lam{lam}_{cfg.hf_optimizer}"
            f"_outer{cfg.hf_n_outer}_inner{cfg.hf_n_inner}_{ws}"
        )
    if cfg.method == "fmrg":
        return f"fmrg_lam{lam}_n{cfg.fmrg_n_iters}_{cfg.fmrg_variant}"
    return f"{cfg.method}_lam{lam}"


# =============================================================================
# CSV schema
# =============================================================================

CSV_KEYS: List[str] = [
    # identifiers
    "run_id", "timestamp",
    # config — method + shared
    "method", "N", "lam", "batch_size", "seed", "constraint_margin",
    # config — MPC
    "H", "constraint_on", "lr_dual", "mu_init",
    "mpc_n_iters", "mpc_optimizer", "mpc_lr", "mpc_n_dual_steps",
    "mpc_warm_start_across_steps",
    "mpc_commit_final_window", "mpc_final_window_dual_mult",
    "mpc_constraint_penalty", "mpc_rho", "mpc_rho_scale", "mpc_dual_adaptive",
    # config — HardFlow
    "hf_optimizer", "hf_n_outer", "hf_n_inner",
    "hf_lr", "hf_rho", "hf_rho_scale", "hf_warm_start",
    # config — FMRG
    "fmrg_n_iters", "fmrg_variant",
    # metrics — reward
    "avg_r_unc", "avg_r_ctrl", "reward_gain",
    # metrics — control cost
    "ctrl_cost_raw", "ctrl_cost_obj",
    # metrics — combined objective  J = Σ‖Δx‖²/(2λ) - r(x_T)
    "objective_J",
    # metrics — constraint violations at t = 1
    "viol_frac_final",   # fraction of particles violating any constraint
    "max_h_final",       # max h value over all particles and constraints
    "max_avg_viol",
    *[f"viol_frac_h{k+1}" for k in range(_N_CON)],
    *[f"max_h{k+1}_final" for k in range(_N_CON)],
    *[f"avg_viol_h{k+1}"  for k in range(_N_CON)],
    *[f"avg_h{k+1}"       for k in range(_N_CON)],
    # bookkeeping
    "ctrl_wall_time_s",
    "plot_path",
]


# =============================================================================
# Core: run one config
# =============================================================================

def _endpoint_map(model: FlowMapMLP) -> Callable:
    def _fn(t_batch: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(t_batch)
        return model(t_batch, ones, x)
    return _fn


def _compute_control_metrics(
    controls: torch.Tensor,
    lam: float,
    controls_are_velocity: bool,
) -> Tuple[float, float]:
    """Return (ctrl_cost_raw, ctrl_cost_obj) from a control sequence."""
    N_steps = controls.shape[0]
    dt = 1.0 / N_steps
    dx = (dt * controls) if controls_are_velocity else controls
    dx_sq_sum = dx.pow(2).sum(dim=-1).sum(dim=0)   # (B,)
    ctrl_cost_raw = float(dx_sq_sum.mean().item())
    ctrl_cost_obj = float((dx_sq_sum / (2.0 * lam)).mean().item())
    return ctrl_cost_raw, ctrl_cost_obj


def _h_violation_metrics(h_final: Optional[np.ndarray]) -> Dict:
    """Per-component and worst-case constraint-violation summary stats.

    avg_viol_h{c} is the mean over particles of max(h_c, 0) — the average
    violation MAGNITUDE for constraint c (0 for already-feasible particles).
    max_avg_viol is the max of avg_viol_h{c} over c: the worst-behaved
    constraint component ON AVERAGE, as opposed to max_h_final which is the
    single worst (particle, component) pair.
    avg_h{c} is the raw (unclipped, signed) mean of h_c over particles — the
    average TERMINAL VALUE of constraint c, negative when feasible on
    average, unlike avg_viol_h{c} which floors feasible particles at 0.
    """
    if h_final is None:
        out = dict(viol_frac_final=float("nan"), max_h_final=float("nan"),
                   max_avg_viol=float("nan"))
        for k in range(_N_CON):
            out[f"viol_frac_h{k+1}"] = float("nan")
            out[f"max_h{k+1}_final"] = float("nan")
            out[f"avg_viol_h{k+1}"]  = float("nan")
            out[f"avg_h{k+1}"]       = float("nan")
        return out
    C = h_final.shape[1]
    h_pos = np.clip(h_final, 0.0, None)
    out = dict(
        viol_frac_final=float((h_final > 0).any(axis=1).mean()),
        max_h_final=float(h_final.max()),
        max_avg_viol=float(h_pos.mean(axis=0).max()),
    )
    for k in range(_N_CON):
        if k < C:
            out[f"viol_frac_h{k+1}"] = float((h_final[:, k] > 0).mean())
            out[f"max_h{k+1}_final"] = float(h_final[:, k].max())
            out[f"avg_viol_h{k+1}"]  = float(h_pos[:, k].mean())
            out[f"avg_h{k+1}"]       = float(h_final[:, k].mean())
        else:
            for pre in ("viol_frac_h", "max_h", "avg_viol_h", "avg_h"):
                key = f"max_h{k+1}_final" if pre == "max_h" else f"{pre}{k+1}"
                out[key] = float("nan")
    return out


def _compute_run_metrics(
    xN: torch.Tensor,
    x_unc: torch.Tensor,
    controls: torch.Tensor,
    r: Callable,
    h: Optional[Callable],
    lam: float,
    controls_are_velocity: bool,
) -> Dict:
    """Lightweight metrics path used when --no_plots is set."""
    with torch.no_grad():
        avg_r_unc = float(r(x_unc).mean().item())
        avg_r_ctrl = float(r(xN).mean().item())
        ctrl_cost_raw, ctrl_cost_obj = _compute_control_metrics(
            controls, lam, controls_are_velocity,
        )
        h_final = h(xN).cpu().numpy() if h is not None else None

    objective_J = ctrl_cost_obj - avg_r_ctrl
    reward_gain = avg_r_ctrl - avg_r_unc

    return {
        "avg_r_unc": avg_r_unc,
        "avg_r_ctrl": avg_r_ctrl,
        "reward_gain": reward_gain,
        "ctrl_cost_raw": ctrl_cost_raw,
        "ctrl_cost_obj": ctrl_cost_obj,
        "objective_J": objective_J,
        **_h_violation_metrics(h_final),
    }


def run_one(
    cfg: RunConfig,
    model: FlowMapMLP,
    x0: torch.Tensor,
    traj_unc: torch.Tensor,
    x_unc: torch.Tensor,
    r: Callable,
    h: Optional[Callable],
    device: torch.device,
    run_id: str,
    plots_dir: Path,
    *,
    no_plots: bool = False,
) -> Dict:
    """Run a single controller config; return metrics dict.

    ``h`` is the TRUE constraint and is what every reported metric is scored
    against. When cfg.constraint_margin > 0 the CONTROLLER is instead handed a
    tightened h(x) + margin (i.e. it must reach h(x) <= -margin), leaving a
    feasibility buffer to absorb plan-vs-realization error. The margin never
    enters the evaluation, so results stay comparable across margins.
    """

    v_theta = model.calc_b

    if h is not None and cfg.constraint_margin != 0.0:
        _margin = float(cfg.constraint_margin)
        _h_true = h

        def h_ctrl(x: torch.Tensor) -> torch.Tensor:
            return _h_true(x) + _margin
    else:
        h_ctrl = h

    # ── run controller ────────────────────────────────────────────────────────
    _t0 = time.perf_counter()
    if cfg.method == "mpc":
        xN, traj_ctrl, controls, _, mu_hist, h_hist = receding_horizon_flow_control(
            v_theta, r, x0,
            N=cfg.N, H=cfg.H, lam=cfg.lam,
            mu_init=cfg.mu_init, h=h_ctrl,
            constraint_on=cfg.constraint_on,
            endpoint_map=_endpoint_map(model),
            lr_dual=cfg.lr_dual,
            optimizer=cfg.mpc_optimizer,
            n_iters=cfg.mpc_n_iters,
            lr=cfg.mpc_lr,
            n_dual_steps=cfg.mpc_n_dual_steps,
            warm_start_across_steps=cfg.mpc_warm_start_across_steps,
            commit_final_window=cfg.mpc_commit_final_window,
            final_window_dual_mult=cfg.mpc_final_window_dual_mult,
            constraint_penalty=cfg.mpc_constraint_penalty,
            dual_adaptive=cfg.mpc_dual_adaptive,
            rho=cfg.mpc_rho,
            rho_scale=cfg.mpc_rho_scale,
            verbose=False,
            device=device,
        )
        controls_are_velocity = True
        mu_labels = [r"$h_1: \|x-c_1\|\leq R_1$", r"$h_2: \|x-c_2\|\leq R_2$",
                     r"$h_3: x_2\leq 0.55$", r"$h_4: x_2\geq 0.24$",
                     r"$h_5:$ slanted"]
        h_labels  = [r"$h_1: \|x-c_1\|\leq R_1$", r"$h_2: \|x-c_2\|\leq R_2$",
                     r"$h_3: x_2\leq 0.55$", r"$h_4: x_2\geq 0.24$",
                     r"$h_5:$ slanted"]

    elif cfg.method == "hardflow":
        xN, traj_ctrl, controls, h_hist = hardflow_sample(
            v_theta, x0, h_ctrl,
            C=lambda x: -r(x),           # maximise r via feasibility-cost trade-off
            N=cfg.N,
            lam=cfg.lam,
            solver=solve_constrained_subproblem,
            solver_kwargs=dict(
                optimizer=cfg.hf_optimizer,
                n_outer=cfg.hf_n_outer,
                n_inner=cfg.hf_n_inner,
                lr=cfg.hf_lr,
                rho=cfg.hf_rho,
                rho_scale=cfg.hf_rho_scale,
            ),
            warm_start=cfg.hf_warm_start,
            verbose=False,
            device=device,
        )
        mu_hist   = None
        mu_labels = None
        h_labels  = [r"$h_1: \|x-c_1\|\leq R_1$", r"$h_2: \|x-c_2\|\leq R_2$",
                     r"$h_3: x_2\leq 0.55$", r"$h_4: x_2\geq 0.24$",
                     r"$h_5:$ slanted"]
        controls_are_velocity = True

    elif cfg.method == "fmrg":
        xN, traj_ctrl, controls = fmrg_j(
            model, r, x0,
            N=cfg.N, n_iters=cfg.fmrg_n_iters, lam=cfg.lam,
            variant=cfg.fmrg_variant,
            verbose=False, device=device,
        )
        mu_hist  = None
        h_hist   = None
        mu_labels = None
        h_labels  = None
        controls_are_velocity = False

    else:
        raise ValueError(f"Unknown method: {cfg.method!r}")

    ctrl_wall_time_s = time.perf_counter() - _t0

    if no_plots:
        metrics = _compute_run_metrics(
            xN, x_unc, controls, r, h, cfg.lam, controls_are_velocity,
        )
        metrics["ctrl_wall_time_s"] = round(ctrl_wall_time_s, 3)
        metrics["plot_path"] = ""
        return metrics

    plot_title = _config_plot_title(cfg)

    # ── visualize and compute metrics ─────────────────────────────────────────
    avg_r_unc, avg_r_ctrl, ctrl_cost_raw, ctrl_cost_obj, figs = visualize_control_results(
        traj_ctrl=traj_ctrl,
        traj_unc=traj_unc,
        x0=x0,
        xN_ctrl=xN,
        xN_unc=x_unc,
        controls=controls,
        mu_history=mu_hist,
        mu_labels=mu_labels,
        h_history=h_hist,
        h_labels=h_labels,
        h=h,
        r=r,
        show_reward_over_time=True,
        method_name=cfg.method.upper(),
        controlled_label=f"Controlled ({cfg.method.upper()})",
        lam=cfg.lam,
        controls_are_velocity=controls_are_velocity,
        show_snaps=True,
        lim=3.5,
        device=device,
        show=False,  # never pop a GUI window in sweep mode
    )

    # ── constraint violations at terminal state ───────────────────────────────
    if h is not None:
        with torch.no_grad():
            h_final = h(xN).cpu().numpy()   # (B, C)
    else:
        h_final = None
    viol_metrics = _h_violation_metrics(h_final)

    # ── objective J = energy/(2λ) - r(x_T) ───────────────────────────────────
    if ctrl_cost_obj is not None and avg_r_ctrl is not None:
        objective_J = ctrl_cost_obj - avg_r_ctrl
    else:
        objective_J = float("nan")

    reward_gain = (
        avg_r_ctrl - avg_r_unc
        if (avg_r_ctrl is not None and avg_r_unc is not None)
        else float("nan")
    )

    # ── save all figures as one PNG ───────────────────────────────────────────
    plot_stem = _config_plot_stem(cfg)
    plot_path = plots_dir / f"{plot_stem}.png"
    if plot_path.exists():
        plot_path = plots_dir / f"{plot_stem}_{run_id[-6:]}.png"
    _save_figs_as_png(figs, plot_path, title=plot_title)

    return {
        "avg_r_unc":       avg_r_unc,
        "avg_r_ctrl":      avg_r_ctrl,
        "reward_gain":     reward_gain,
        "ctrl_cost_raw":   ctrl_cost_raw,
        "ctrl_cost_obj":   ctrl_cost_obj,
        "objective_J":     objective_J,
        **viol_metrics,
        "ctrl_wall_time_s": round(ctrl_wall_time_s, 3),
        "plot_path":       str(plot_path),
    }


# =============================================================================
# Plot stitching
# =============================================================================

def _save_figs_as_png(
    figs: list,
    save_path: Path,
    title: str = "",
    dpi: int = 100,
) -> None:
    """Render each figure to a bitmap and stack them vertically into one PNG."""
    if not figs:
        return

    images: List[np.ndarray] = []
    for fig in figs:
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        buf.seek(0)
        img = plt.imread(buf)   # (H, W, 4) float32 RGBA in [0, 1]
        images.append(img)
        plt.close(fig)

    # Pad all images to the same width
    max_w = max(img.shape[1] for img in images)
    padded: List[np.ndarray] = []
    for img in images:
        gap = max_w - img.shape[1]
        if gap > 0:
            pad = np.ones((img.shape[0], gap, img.shape[2]), dtype=img.dtype)
            img = np.concatenate([img, pad], axis=1)
        padded.append(img)

    combined = np.concatenate(padded, axis=0)   # (total_H, max_W, 4)

    title_lines = 0
    if title:
        title_lines = len(textwrap.wrap(title, width=110))
    title_rows = 0.22 * max(title_lines, 1) if title else 0.0

    fig_h = combined.shape[0] / dpi + title_rows
    fig_out = plt.figure(figsize=(max_w / dpi, fig_h), dpi=dpi)

    img_bottom = 0.0
    img_height = combined.shape[0] / (fig_h * dpi)
    if title:
        wrapped = "\n".join(textwrap.wrap(title, width=110))
        fig_out.text(0.5, 1.0 - 0.02 / fig_h, wrapped,
                     ha="center", va="top", fontsize=9, family="monospace")
        img_bottom = title_rows / fig_h
        img_height = combined.shape[0] / (fig_h * dpi) * (1 - img_bottom)

    ax = fig_out.add_axes([0, img_bottom, 1, img_height])
    ax.imshow(combined, interpolation="nearest", aspect="auto")
    ax.axis("off")
    fig_out.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig_out)


# =============================================================================
# CSV helpers
# =============================================================================

def _append_csv(csv_path: Path, row: Dict) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_KEYS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_KEYS})


# =============================================================================
# Sweep config builder
# =============================================================================

def _build_configs(args) -> List[RunConfig]:
    configs: List[RunConfig] = []

    common = dict(N=args.N, batch_size=args.batch_size, seed=args.seed)

    if "mpc" in args.methods:
        for H, con, lam, lr_d, n_iters, mpc_lr, n_dual, fw_mult, margin, mu0 in itertools.product(
            args.H, args.constraint_on, args.lam, args.lr_dual,
            args.mpc_n_iters, args.mpc_lr, args.mpc_n_dual_steps,
            args.mpc_final_window_dual_mult, args.constraint_margin,
            args.mu_init,
        ):
            configs.append(RunConfig(
                method="mpc",
                H=H, constraint_on=con, lam=lam, lr_dual=lr_d,
                mu_init=mu0,
                mpc_n_iters=n_iters,
                mpc_optimizer=args.mpc_optimizer,
                mpc_lr=mpc_lr,
                mpc_n_dual_steps=n_dual,
                mpc_warm_start_across_steps=args.mpc_warm_start_across_steps,
                mpc_commit_final_window=args.mpc_commit_final_window,
                mpc_final_window_dual_mult=fw_mult,
                mpc_constraint_penalty=args.mpc_constraint_penalty,
                mpc_dual_adaptive=args.mpc_dual_adaptive,
                mpc_rho=args.mpc_rho,
                mpc_rho_scale=args.mpc_rho_scale,
                constraint_margin=margin,
                **common,
            ))

    if "hardflow" in args.methods:
        for lam, n_outer, n_inner, hf_lr, margin in itertools.product(
            args.lam, args.hf_n_outer, args.hf_n_inner, args.hf_lr,
            args.constraint_margin,
        ):
            configs.append(RunConfig(
                method="hardflow",
                lam=lam,
                hf_optimizer=args.hf_optimizer,
                hf_n_outer=n_outer,
                hf_n_inner=n_inner,
                hf_lr=hf_lr,
                hf_rho=args.hf_rho,
                hf_rho_scale=args.hf_rho_scale,
                hf_warm_start=not args.hf_no_warm_start,
                constraint_margin=margin,
                **common,
            ))

    if "fmrg" in args.methods:
        for lam, n_it, var in itertools.product(
            args.lam, args.fmrg_n_iters, args.fmrg_variant
        ):
            configs.append(RunConfig(
                method="fmrg",
                lam=lam,
                fmrg_n_iters=n_it,
                fmrg_variant=var,
                **common,
            ))

    return configs


# =============================================================================
# Uncontrolled rollout (computed once, shared across all configs)
# =============================================================================

def _rollout_uncontrolled(
    model: FlowMapMLP,
    x0: torch.Tensor,
    N: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (traj_unc, x_unc) where traj_unc has shape (N+1, B, D)."""
    dt = 1.0 / N
    t_grid = torch.linspace(0.0, 1.0, N + 1, device=device, dtype=x0.dtype)
    with torch.no_grad():
        x = x0.clone()
        traj = [x.clone()]
        for j in range(N):
            x = x + dt * model.calc_b(t_grid[j], x)
            traj.append(x.clone())
    return torch.stack(traj, dim=0), x


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── model / device ────────────────────────────────────────────────────────
    p.add_argument("--ckpt", required=True,
                   help="Path to FlowMapMLP .pkl checkpoint (JAX/Flax format).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Torch device string (default: cuda if available). "
                        "Ignored when --gpu is set.")
    p.add_argument("--gpu", type=int, default=None,
                   help="Pin the sweep to one CUDA device by index (e.g. 0). "
                        "Sets device to cuda:<index> and overrides --device.")

    # ── output ────────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", default="results",
                   help="Directory for CSV and plots sub-folder.")
    p.add_argument("--csv_name", default="sweep_results.csv",
                   help="Name of the results CSV file inside output_dir.")

    # ── methods ───────────────────────────────────────────────────────────────
    p.add_argument("--methods", nargs="+", default=["mpc", "fmrg"],
                   choices=["mpc", "hardflow", "fmrg"],
                   help="Controller(s) to include in the sweep.")

    # ── shared sweep axes ─────────────────────────────────────────────────────
    p.add_argument("--N", type=int, default=20,
                   help="Number of Euler steps (fixed for all configs).")
    p.add_argument("--lam", nargs="+", type=float, default=[2.0],
                   help="Control/guidance weight λ (MPC, HardFlow λ_oc, FMRG).")
    p.add_argument("--batch_size", type=int, default=512,
                   help="Batch size (same x0 reused across all configs).")
    p.add_argument("--seed", type=int, default=42)

    # ── MPC-specific sweep axes ───────────────────────────────────────────────
    p.add_argument("--H", nargs="+", type=int, default=[10],
                   help="MPC planning horizon(s) to sweep.")
    p.add_argument("--constraint_on", nargs="+",
                   default=["endpoint"], choices=["horizon", "endpoint"],
                   help="Where to evaluate h for the MPC Lagrangian.")
    p.add_argument("--lr_dual", nargs="+", type=float, default=[0.1],
                   help="Dual-ascent step size(s) for MPC.")
    p.add_argument("--mu_init", nargs="+", type=float, default=[0.0],
                   help="Initial Lagrange multiplier value.")
    p.add_argument("--mpc_n_iters", nargs="+", type=int, default=[50],
                   help="MPC inner-solver iteration(s) per step to sweep.")
    p.add_argument("--mpc_optimizer", default="adam",
                   choices=["adam", "lbfgs"],
                   help="Inner optimizer for the MPC horizon problem.")
    p.add_argument("--mpc_lr", nargs="+", type=float, default=[0.05],
                   help="MPC inner-solver learning rate(s) to sweep.")
    p.add_argument("--mpc_n_dual_steps", nargs="+", type=int, default=[1],
                   help="Primal-dual round(s) per MPC step to sweep: re-solves "
                        "the same horizon subproblem in place, updating mu "
                        "between rounds and warm-starting each re-solve "
                        "(mirrors HardFlow's augmented-Lagrangian outer loop). "
                        "1 (default) reproduces the original single-solve "
                        "behaviour.")
    p.add_argument("--mpc_warm_start_across_steps", action="store_true",
                   help="Seed each MPC step's first primal-dual round from the "
                        "PREVIOUS step's converged trajectory, shifted by one "
                        "index, instead of a fresh uncontrolled rollout. Off "
                        "by default (matches original from-scratch behaviour).")
    p.add_argument("--mpc_commit_final_window", action="store_true",
                   help="At the first step whose horizon reaches exactly t=1 "
                        "(j = max(0, N-H)), solve that window once and commit "
                        "to its ENTIRE trajectory instead of applying only the "
                        "first step and continuing to recede. At that window "
                        "the evaluated constraint point is provably x_N itself "
                        "(not a lookahead prediction) under both constraint_on "
                        "modes, closing the surrogate-vs-realized gap for this "
                        "final stretch. Off by default.")
    p.add_argument("--mpc_constraint_penalty", default="dual",
                   choices=["dual", "alm"],
                   help="How MPC enforces h(x) <= 0 inside each horizon "
                        "subproblem: 'dual' (default, plain Lagrangian "
                        "<mu,h> with projected dual ascent at lr_dual, mu "
                        "persisting across steps) or 'alm' (augmented "
                        "Lagrangian with growing rho, mu/rho reset per MPC "
                        "step — the same mechanism HardFlow uses; lr_dual is "
                        "ignored).")
    p.add_argument("--mpc_dual_adaptive", action="store_true",
                   help="Normalise the projected dual step per constraint by a "
                        "running RMS scale (and use the same normalised "
                        "constraint in the Lagrangian), so every constraint gets "
                        "comparable dual pressure regardless of its typical "
                        "violation magnitude. Threshold-free and self-calibrating; "
                        "dual penalty only.")
    p.add_argument("--mpc_rho", type=float, default=1.0,
                   help="Initial ALM penalty rho (--mpc_constraint_penalty alm).")
    p.add_argument("--mpc_rho_scale", type=float, default=2.0,
                   help="ALM penalty growth per primal-dual round "
                        "(--mpc_constraint_penalty alm).")
    p.add_argument("--mpc_final_window_dual_mult", nargs="+", type=int, default=[1],
                   help="Multiplier(s) k on n_dual_steps to sweep, used ONLY "
                        "for the committed final window (ignored unless "
                        "--mpc_commit_final_window is set): that window runs "
                        "n_dual_steps * k primal-dual rounds. k=1 (default) "
                        "uses the same round count as every other step.")

    # ── HardFlow-specific sweep axes ──────────────────────────────────────────
    p.add_argument("--hf_optimizer", default="lbfgs", choices=["lbfgs", "adam"],
                   help="HardFlow inner solver optimizer.")
    p.add_argument("--hf_n_outer", nargs="+", type=int, default=[5],
                   help="HardFlow augmented-Lagrangian outer iterations to sweep.")
    p.add_argument("--hf_n_inner", nargs="+", type=int, default=[20],
                   help="HardFlow inner solver iteration(s) per outer step to sweep.")
    p.add_argument("--hf_lr", nargs="+", type=float, default=[0.1],
                   help="HardFlow inner solver learning rate(s) to sweep.")
    p.add_argument("--hf_rho", type=float, default=1.0,
                   help="HardFlow initial ALM penalty ρ.")
    p.add_argument("--hf_rho_scale", type=float, default=2.0,
                   help="HardFlow penalty scaling per outer step.")
    p.add_argument("--hf_no_warm_start", action="store_true",
                   help="Disable warm-starting the HardFlow inner solver.")

    # ── FMRG-specific sweep axes ──────────────────────────────────────────────
    p.add_argument("--fmrg_n_iters", nargs="+", type=int, default=[3],
                   help="FMRG guidance steps per Euler step.")
    p.add_argument("--fmrg_variant", nargs="+", default=["J"],
                   choices=["J", "E"],
                   help="FMRG gradient variant: J (Jacobian, backprop through X) "
                        "or E (Euclidean, stop-gradient at X). Default: J.")

    # ── misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--no_plots", action="store_true",
                   help="Skip matplotlib plots and PNG saving; log metrics only. "
                        "Much faster for large sweeps.")
    p.add_argument("--reward_type", default="upper_moon",
                   choices=["upper_moon", "linear", "off_top_ball", "none"],
                   help="Terminal reward: upper_moon (RBF on upper moon, default), "
                        "linear (r(x) = x[:, 1]), off_top_ball (Gaussian bump "
                        "off the upper moon's top, off the data manifold), or "
                        "none (r(x) = 0; pure constraint-seeking, no reward).")
    p.add_argument("--constraint_margin", nargs="+", type=float, default=[0.0],
                   help="Constraint-tightening margin(s) eps to sweep: the "
                        "CONTROLLER is required to reach h(x) <= -eps instead "
                        "of h(x) <= 0, leaving a buffer to absorb plan-vs-"
                        "realization error (standard robust-MPC tightening). "
                        "All reported metrics are still scored against the "
                        "TRUE h(x) <= 0, so runs stay comparable across eps.")
    p.add_argument("--no_constraint", action="store_true",
                   help="Disable the forbidden-region constraint h(x) <= 0; "
                        "runs a pure reward-only comparison (MPC and HardFlow "
                        "fall back to unconstrained proximal reward steering, "
                        "matching FMRG). Constraint-violation columns are left "
                        "blank in the CSV and constraint/multiplier plots are "
                        "skipped.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print all configs and exit without running anything.")

    return p


# =============================================================================
# main
# =============================================================================

def _resolve_device(args: argparse.Namespace) -> torch.device:
    """Return the torch device for this sweep."""
    if args.gpu is not None:
        if not torch.cuda.is_available():
            raise SystemExit(f"--gpu {args.gpu} requested but CUDA is not available")
        n = torch.cuda.device_count()
        if args.gpu < 0 or args.gpu >= n:
            raise SystemExit(
                f"--gpu {args.gpu} out of range: found {n} CUDA device(s) "
                f"(valid indices: 0..{n - 1})"
            )
        return torch.device(f"cuda:{args.gpu}")
    return torch.device(args.device)


def main() -> None:
    args = _build_parser().parse_args()

    device    = _resolve_device(args)
    out_dir   = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / args.csv_name

    # ── build all configs ─────────────────────────────────────────────────────
    configs = _build_configs(args)
    print(f"Total configurations to run: {len(configs)}")
    col_w = max(len(str(c)) for c in configs) if configs else 80
    for i, cfg in enumerate(configs):
        print(f"  [{i+1:3d}]  {cfg}")

    if args.dry_run:
        print("\n--dry_run set; exiting without running.")
        return

    # ── load model ────────────────────────────────────────────────────────────
    model = load_model(args.ckpt, device)

    # ── build reward and constraint ───────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    r = _build_reward_fn(device, args.reward_type)
    h = None if args.no_constraint else _build_constraint_fn(device)

    # ── shared initial samples and uncontrolled baseline ─────────────────────
    x0 = torch.randn(args.batch_size, 2, device=device) * float(model.rescale)
    traj_unc, x_unc = _rollout_uncontrolled(model, x0, args.N, device)
    with torch.no_grad():
        avg_r_baseline = r(x_unc).mean().item()
    reward_desc = "r(x) = x[:, 1]" if args.reward_type == "linear" else "upper-moon RBF"
    print(f"\nDevice: {device}")
    print(f"Reward: {args.reward_type}  ({reward_desc})")
    print(f"Uncontrolled baseline  avg r = {avg_r_baseline:.4f}")
    print(f"Results → {csv_path}")
    print(f"Plots   → {plots_dir}/\n")

    # ── sweep ─────────────────────────────────────────────────────────────────
    n_ok = n_err = 0
    for i, cfg in enumerate(configs):
        run_id    = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.datetime.now().isoformat()
        print(f"[{i+1}/{len(configs)}]  {cfg.method.upper()}  run_id={run_id}")

        try:
            metrics = run_one(
                cfg, model,
                x0.clone(),        # fresh copy so the model can't mutate shared state
                traj_unc, x_unc,
                r, h, device,
                run_id, plots_dir,
                no_plots=args.no_plots,
            )
            row = {"run_id": run_id, "timestamp": timestamp,
                   **asdict(cfg), **metrics}
            _append_csv(csv_path, row)

            print(
                f"  avg_r_ctrl={metrics['avg_r_ctrl']:.4f}  "
                f"gain={metrics['reward_gain']:+.4f}  "
                f"J={metrics['objective_J']:.4f}  "
                f"viol={metrics['viol_frac_final']:.3f}  "
                f"max_h={metrics['max_h_final']:+.4f}"
            )
            n_ok += 1

        except Exception as exc:
            print(f"  [ERROR] {exc}")
            traceback.print_exc()
            n_err += 1

    print(f"\nDone.  {n_ok} succeeded, {n_err} failed.")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
