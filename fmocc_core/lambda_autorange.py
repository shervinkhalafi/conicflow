#!/usr/bin/env python3
"""
lambda_autorange.py -- pick a lambda grid automatically, per method, for a
given (reward, prompt, edit_prompt), so a big unattended sweep does not need
its lambda range hand-tuned for every reward/prompt combination.

WHY THIS EXISTS (2026-09-16). `lam` means a different thing in each method --
a Lagrange weight the optimiser solves against (MPC), the same quantity via
reg = N/(2*lam) (HardFlow), and an open-loop gain on the reward gradient
(FMRG) -- and its useful range also moves with the reward's scale (switching
LPIPS -> ImageReward de-regularises by ~200x at fixed lam). A fixed grid such
as {0.1 .. 10000} therefore spends most of its points where nothing happens:
for ImageReward, MPC is already 79% saturated at lam=1 and fully saturated by
lam=10, so 6 of the 8 suite points are on the ceiling.

WHAT IT DOES. For each method it measures the response curve delta_r(lam) --
which has the same shape for every method and reward: ~0 where the effort
penalty dominates, a transition, then a plateau at the reward's attainable
ceiling -- and places the grid across the transition:

  0. Probes (seconds): uncontrolled rollout -> r_unc; reward ceiling from
     reward_ranges.REWARD_OBSERVED_RANGE -> delta_r_max; gradient norm
     ||grad_z r|| at the uncontrolled endpoint -> a scale-aware prior lam0.
  1. Bracket: evaluate lam0, then step x10 upward until the curve is flat
     against the ceiling (or energy explodes for no reward), and x10 downward
     until delta_r is inside the noise. Failures (e.g. HardFlow's lr*reg >= 1
     floor) are recorded and treated as the admissible boundary.
  2. Refine: fit a logistic in log10(lam) to the points and put the final grid
     at equally spaced fractions of the fitted rise, so every grid point buys
     a distinct point on the energy/reward frontier. The refined points are
     evaluated too, so the output is the whole measured curve.

It runs in ONE process -- FLUX loads once, the reward model loads once per
(reward, edit_prompt) -- by calling run_sweep._run_one directly, so every
guard in that path (locked defaults, plausibility ranges, non-finite checks)
still applies. Range-finding uses a cheap proxy N (default 4); the recommended
grid is for the sweep you then run at --target_N. The transition in lam is
approximately N-invariant for MPC (lam is per-step normalised via dt) but
HardFlow's admissible floor lam >= N*lr/2 is not, so the floor is reported for
the target N, not the proxy.

Usage:
  python lambda_autorange.py --reward imagereward \\
      --prompt "a portrait of a man" --edit_prompt "a smiling face" \\
      --methods mpc hardflow fmrg --probe_N 4 --target_N 10 --n_grid 6 \\
      --flux_device cuda:0 --reward_device cuda:0 --out_dir results/autorange/imagereward

Outputs in --out_dir: autorange.json (everything, machine-readable),
evaluations.csv (one row per lambda evaluated), curves.png, and a
grids.txt with ready-to-paste --sweep_lam lines per method.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_sweep as rs                                   # noqa: E402
from reward_ranges import REWARD_OBSERVED_RANGE           # noqa: E402

LAM_ABS_MIN, LAM_ABS_MAX = 1e-4, 1e7    # search never leaves this (log-)box


# ═════════════════════════════════════════════════════════════════════════════
# Reward-model memoisation: _run_one rebuilds the reward model every call and
# moves it to CPU at the end. Building is the dominant per-lambda overhead for
# the cheap proxy configs, so cache one object per (reward, edit_prompt, dtype)
# and move it back to the reward device on reuse (a few hundred ms).
# ═════════════════════════════════════════════════════════════════════════════
_REWARD_CACHE: Dict[tuple, tuple] = {}
_orig_build_reward_fn = rs._build_reward_fn


def _memo_build_reward_fn(sampler, cfg, reward_device, dtype, reward_model="clip",
                          ref_holder=None, decode_fn=None, drop_text_tower=False,
                          reward_dtype=torch.float32):
    # lpips needs a per-run ref_holder that is populated after the uncontrolled
    # rollout; its closure is cheap to rebuild, so never cache it.
    if reward_model in ("lpips", "none"):
        return _orig_build_reward_fn(sampler, cfg, reward_device, dtype, reward_model=reward_model,
                                     ref_holder=ref_holder, decode_fn=decode_fn,
                                     drop_text_tower=drop_text_tower, reward_dtype=reward_dtype)
    key = (reward_model, cfg.edit_prompt or cfg.prompt, str(reward_device), str(reward_dtype),
           bool(drop_text_tower), tuple(cfg.style_ref_paths or ()))
    if key not in _REWARD_CACHE:
        _REWARD_CACHE[key] = _orig_build_reward_fn(
            sampler, cfg, reward_device, dtype, reward_model=reward_model, ref_holder=ref_holder,
            decode_fn=decode_fn, drop_text_tower=drop_text_tower, reward_dtype=reward_dtype)
    r_image, obj = _REWARD_CACHE[key]
    if obj is not None and hasattr(obj, "to"):
        obj.to(reward_device)          # _run_one parked it on CPU at the end of the last run
    return r_image, obj


rs._build_reward_fn = _memo_build_reward_fn


# ═════════════════════════════════════════════════════════════════════════════
# Probes
# ═════════════════════════════════════════════════════════════════════════════
def probe_uncontrolled(sampler, cfg, device, dtype, reward_model, reward_device, reward_dtype):
    """r_unc, ||grad_z r|| at the uncontrolled endpoint, and the endpoint latent norm."""
    torch.manual_seed(cfg.seed)
    in_ch = sampler.pipeline.transformer.config.in_channels // 4
    vscale = sampler.vae_scale_factor
    z0 = torch.randn(1, in_ch, cfg.height // vscale, cfg.width // vscale, device=device, dtype=dtype)
    z_unc, _ = rs._run_uncontrolled(sampler, z0, cfg, device, dtype, log=lambda *a, **k: None)
    r_image, _obj = rs._build_reward_fn(sampler, cfg, reward_device, dtype, reward_model=reward_model,
                                        ref_holder={}, drop_text_tower=True, reward_dtype=reward_dtype)
    z = z_unc.detach().clone().requires_grad_(True)
    r = r_image(z)
    (g,) = torch.autograd.grad(r.sum(), z)
    out = dict(r_unc=float(r.mean().item()),
               grad_norm=float(g.float().flatten().norm().item()),
               z_norm=float(z_unc.float().flatten().norm().item()),
               n_latent=int(z_unc.numel()))
    del z, r, g, z_unc, z0
    torch.cuda.empty_cache()
    return out


def scale_prior(method: str, pr: dict, dr_max: float, N: int) -> float:
    """A scale-aware starting lambda. It only has to land within a decade or so;
    bracketing corrects it, and the upward direction is the cheap one to be
    wrong in (saturated runs are the fast ones for HardFlow/FMRG).

    lam0 = c * dr_max / ||grad_z r||.  Rationale: the reward has to be moved by
    dr_max, and the gradient norm at the uncontrolled endpoint sets how much
    latent displacement that costs; every method's lambda multiplies (MPC/HF via
    the energy trade-off J = E/(2 lam) - r, FMRG as a gain on grad r) a
    displacement that is linear in lam to first order. A 1/||grad r||^2 law was
    tried first and was 3 decades low on ImageReward (||grad r|| ~ 12); the
    measured knees -- MPC ~0.3 on ImageReward (||grad r|| 12.3, dr_max 3.3) and
    ~30 on CLIP (0.19, 0.65) -- sit within a decade of the 1/||grad r|| form
    with c ~ 1, which is what is used. FMRG's knee was ~2x MPC's on both.
    """
    g = max(pr["grad_norm"], 1e-12)
    c = {"mpc": 1.0, "hardflow": 1.0, "fmrg": 2.0}[method]
    lam = c * dr_max / g
    return float(min(max(lam, LAM_ABS_MIN), LAM_ABS_MAX))


def hardflow_floor(N: int, lr: float) -> float:
    """hardflow_image raises when lr*reg >= 1 with reg = N/(2*lam)  =>  lam >= N*lr/2."""
    return N * lr / 2.0


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation of one lambda
# ═════════════════════════════════════════════════════════════════════════════
def make_cfg(method: str, a, lam: float) -> rs.RunConfig:
    common = dict(prompt=a.prompt, edit_prompt=a.edit_prompt, N=a.probe_N, seed=a.seed,
                  lam=float(lam), sample_mode=a.sample_mode, height=a.height, width=a.width)
    if method == "mpc":
        cfg = rs.RunConfig(control_type="mpc", H=a.mpc_H, n_inner=a.mpc_n_inner, lr_inner=a.mpc_lr_inner,
                           endpoint_surrogate=a.endpoint_surrogate, **common)
    elif method == "hardflow":
        cfg = rs.RunConfig(control_type="hardflow", H=2, n_inner=a.hardflow_n_inner,
                           lr_inner=a.hardflow_lr_inner, **common)
    elif method == "fmrg":
        cfg = rs.RunConfig(control_type="fmrg", n_inner=a.fmrg_n_opt, **common)
    else:
        raise ValueError(method)
    rs._resolve_hardflow_reg(cfg)
    return cfg


class Evaluator:
    def __init__(self, sampler, device, reward_device, dtype, a):
        self.sampler, self.device, self.reward_device, self.dtype, self.a = sampler, device, reward_device, dtype, a
        self.reward_dtype = rs._REWARD_DTYPES[a.reward_dtype]
        self.rows: List[dict] = []

    def __call__(self, method: str, lam: float) -> dict:
        cfg = make_cfg(method, self.a, lam)
        t0 = time.perf_counter()
        try:
            res = rs._run_one(self.sampler, cfg, self.device, self.reward_device, self.dtype, quiet=True,
                              fast=self.a.fast, low_memory=not self.a.fast, reward_model=self.a.reward,
                              reward_dtype=self.reward_dtype, pbar_rewards=False)
            row = dict(method=method, lam=float(lam), ok=bool(res["run_ok"]),
                       r_unc=res["r_unc"], r_ctrl=res["r_ctrl"], dr=res["delta_r"], E=res["ctrl_energy"],
                       J=res["objective"], t_ctrl=res["time_ctrl_s"], err=res.get("error_msg", ""))
        except Exception as e:  # e.g. HardFlow's lr*reg >= 1 guard, OOM
            row = dict(method=method, lam=float(lam), ok=False, r_unc=float("nan"), r_ctrl=float("nan"),
                       dr=float("nan"), E=float("nan"), J=float("nan"), t_ctrl=float("nan"),
                       err=f"{type(e).__name__}: {str(e)[:160]}")
            torch.cuda.empty_cache()
        row["wall_s"] = time.perf_counter() - t0
        self.rows.append(row)
        flag = "" if row["ok"] else f"   FAILED: {row['err'][:70]}"
        print(f"    {method:8} lam={lam:<10.4g} dr={row['dr']:8.4f}  E={row['E']:9.4f}  J={row['J']:9.4f}"
              f"  t={row['t_ctrl']:6.1f}s{flag}", flush=True)
        return row


# ═════════════════════════════════════════════════════════════════════════════
# The search
# ═════════════════════════════════════════════════════════════════════════════
def _valid(points: Dict[float, dict]) -> List[dict]:
    return sorted((p for p in points.values() if p["ok"] and math.isfinite(p["dr"])), key=lambda p: p["lam"])


def _saturated(vp: List[dict], ceiling: float, tol: float) -> bool:
    """Top of the bracket is flat. Judged on the OBSERVED rise, not the
    theoretical ceiling: the reward's documented maximum is what the model can
    reach on some image, not what this prompt pair can reach (CLIP's observed
    max 0.80 gave a ceiling of 0.65 here while the real plateau was 0.24, and a
    ceiling-only rule then walked lambda up to 3.5e6 looking for it)."""
    if len(vp) < 2:
        return False
    a, b = vp[-2], vp[-1]
    drs = [p["dr"] for p in vp]
    rise = b["dr"] - min(drs)                     # how far the curve has climbed so far
    gain = b["dr"] - a["dr"]                      # what the last decade bought
    # "risen" guards against calling two ~0 points saturated. 2*tol rather than a
    # larger multiple: HardFlow's admissible floor can sit partway up the curve, so
    # its whole visible rise may be small (CLIP: 0.08 with tol 0.033 -- a 4*tol
    # guard walked it to lam=5e6 before stopping, 2*tol stops at the plateau).
    risen = rise >= 2.0 * tol
    flat = risen and gain < tol
    near_ceiling = b["dr"] >= 0.9 * ceiling
    energy_waste = risen and b["E"] > 3.0 * max(a["E"], 1e-9) and gain < 2.0 * tol
    falling = risen and b["dr"] < max(drs) - tol  # overshoot: the image is being destroyed
    return flat or near_ceiling or energy_waste or falling


def _inactive(vp: List[dict], tol: float) -> bool:
    return len(vp) >= 1 and vp[0]["dr"] <= tol


def bracket(method: str, ev: Evaluator, lam0: float, ceiling: float, tol: float,
            lam_min_adm: float, max_evals: int) -> Dict[float, dict]:
    points: Dict[float, dict] = {}
    n = 0

    def run(lam):
        nonlocal n
        lam = float(f"{lam:.3g}")
        if lam in points:
            return points[lam]
        n += 1
        points[lam] = ev(method, lam)
        return points[lam]

    lam0 = max(lam0, lam_min_adm)
    run(lam0)
    # ── upward: until saturated ──────────────────────────────────────────
    lam = lam0
    while n < max_evals and lam * 10 <= LAM_ABS_MAX:
        vp = _valid(points)
        if _saturated(vp, ceiling, tol):
            break
        lam *= 10
        run(lam)
    # ── downward: until inactive (or admissible floor) ───────────────────
    lam = lam0
    while n < max_evals and lam / 10 >= max(LAM_ABS_MIN, lam_min_adm):
        vp = _valid(points)
        if _inactive(vp, tol):
            break
        lam /= 10
        run(lam)
    return points


def fit_logistic(vp: List[dict], ceiling: float):
    """delta_r(x) = A / (1 + exp(-k (x - m))),  x = log10(lam).  Grid search on (m, k),
    A solved in closed form; numpy only (no scipy dependency)."""
    x = np.array([math.log10(p["lam"]) for p in vp]); y = np.array([p["dr"] for p in vp])
    if len(x) < 3 or np.ptp(y) < 1e-9:
        return None
    best = None
    for m in np.linspace(x.min() - 1, x.max() + 1, 121):
        for k in np.linspace(0.3, 6.0, 58):
            s = 1.0 / (1.0 + np.exp(-k * (x - m)))
            A = float(np.dot(s, y) / max(np.dot(s, s), 1e-12))
            A = min(max(A, 0.0), 1.5 * max(ceiling, y.max()))
            err = float(np.sum((A * s - y) ** 2))
            if best is None or err < best[0]:
                best = (err, m, k, A)
    err, m, k, A = best
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    return dict(m=float(m), k=float(k), A=float(A), r2=1.0 - err / ss_tot)


def quantile_grid(fit, n_grid: int, lo: float, hi: float) -> List[float]:
    """lambda values where the fitted curve reaches equally spaced fractions of its rise."""
    qs = np.linspace(0.08, 0.92, n_grid)
    xs = [fit["m"] + math.log(q / (1 - q)) / fit["k"] for q in qs]
    lams = [10 ** x for x in xs]
    lams = [min(max(l, lo), hi) for l in lams]
    return sorted({float(f"{l:.2g}") for l in lams})


def loguniform_grid(n_grid: int, lo: float, hi: float) -> List[float]:
    return sorted({float(f"{10 ** x:.2g}") for x in np.linspace(math.log10(lo), math.log10(hi), n_grid)})


def autorange(method: str, ev: Evaluator, a, pr: dict, ceiling: float, lam_floor_probe: float,
              lam_floor_target: float) -> dict:
    tol = max(a.tol * ceiling, a.tol_abs)
    lam0 = scale_prior(method, pr, ceiling, a.probe_N)
    print(f"\n== {method}: prior lam0={lam0:.3g}  ceiling(dr_max)={ceiling:.3f}  tol={tol:.3f}"
          f"{f'  admissible floor lam>={lam_floor_probe:.3g} (probe N), >={lam_floor_target:.3g} (target N)' if lam_floor_probe else ''}")
    t0 = time.perf_counter()
    pts = bracket(method, ev, lam0, ceiling, tol, lam_floor_probe, a.max_evals)
    vp = _valid(pts)
    out = dict(method=method, prior_lam0=lam0, tol=tol, ceiling=ceiling,
               lam_floor_probe_N=lam_floor_probe, lam_floor_target_N=lam_floor_target)
    if len(vp) < 2:
        out.update(status="insufficient", grid=[], points=list(pts.values()))
        return out
    # active region: from the last point that is still inactive (or the lowest) to the first saturated
    lo_pts = [p for p in vp if p["dr"] <= tol]
    lo = lo_pts[-1]["lam"] if lo_pts else vp[0]["lam"]
    sat_idx = next((i for i in range(1, len(vp)) if _saturated(vp[:i + 1], ceiling, tol)), len(vp) - 1)
    hi = vp[sat_idx]["lam"]
    if hi <= lo:
        lo, hi = vp[0]["lam"], vp[-1]["lam"]
    fit = fit_logistic(vp, ceiling)
    grid = quantile_grid(fit, a.n_grid, lo, hi) if fit and fit["r2"] > 0.5 else loguniform_grid(a.n_grid, lo, hi)
    # quantiles collapsed (flat or nearly flat curve) or produced near-duplicates
    # after rounding: spread log-uniformly over the active region instead.
    if len(grid) < max(3, a.n_grid // 2) or any(b / a_ < 1.25 for a_, b in zip(grid, grid[1:])):
        grid = loguniform_grid(a.n_grid, lo, hi)
    # HardFlow special case: if the lowest ADMISSIBLE point is already on the
    # plateau, the transition lies below the lr*reg >= 1 floor and no grid above
    # it can measure it. Say so instead of handing back six plateau points.
    dr_top = max(p["dr"] for p in vp)
    floor_saturated = bool(lam_floor_probe) and any(not p["ok"] for p in pts.values()) \
        and vp[0]["dr"] >= dr_top - tol and dr_top >= 4.0 * tol
    # ── refine: evaluate the proposed grid points we have not seen yet ──
    if a.evaluate_grid:
        for lam in grid:
            if float(f"{lam:.3g}") not in pts and len(pts) < a.max_evals + a.n_grid:
                pts[float(f"{lam:.3g}")] = ev(method, lam)
        vp = _valid(pts)
        fit = fit_logistic(vp, ceiling) or fit
    # HardFlow: the target-N floor may cut the grid; report rather than silently shift
    clipped = [l for l in grid if l < lam_floor_target] if lam_floor_target else []
    grid_target = [l for l in grid if l >= lam_floor_target] if lam_floor_target else grid
    out.update(status="saturated_at_admissible_floor" if floor_saturated else "ok",
               dr_plateau_observed=dr_top, active_lo=lo, active_hi=hi, fit=fit, grid=grid,
               grid_for_target_N=grid_target, clipped_by_target_floor=clipped,
               points=sorted(pts.values(), key=lambda p: p["lam"]),
               n_evals=len(pts), wall_s=time.perf_counter() - t0)
    if floor_saturated:
        lr = a.hardflow_lr_inner
        out["note"] = (f"every admissible lambda (>= {lam_floor_probe:g} at N={a.probe_N}, >= {lam_floor_target:g} "
                       f"at N={a.target_N}) is already on the plateau (dr {vp[0]['dr']:.3f} of {dr_top:.3f}); the "
                       f"transition is below HardFlow's lr*reg >= 1 floor. The floor is N*lr/2, so lower "
                       f"--hardflow_lr_inner (e.g. {lr/10:g} moves it to {hardflow_floor(a.target_N, lr/10):g}) "
                       f"or accept that HardFlow only has a saturated regime for this reward.")
    return out


# ═════════════════════════════════════════════════════════════════════════════
def plot(results: List[dict], out: Path, reward: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), squeeze=False)
    for ax, r in zip(axes[0], results):
        ok = [p for p in r["points"] if p["ok"]]
        bad = [p for p in r["points"] if not p["ok"]]
        ax.semilogx([p["lam"] for p in ok], [p["dr"] for p in ok], "o", label="measured")
        for p in bad:
            ax.axvline(p["lam"], color="red", alpha=.25, ls=":")
        if r.get("fit"):
            xs = np.logspace(math.log10(min(p["lam"] for p in r["points"])) - .5,
                             math.log10(max(p["lam"] for p in r["points"])) + .5, 200)
            f = r["fit"]; ys = f["A"] / (1 + np.exp(-f["k"] * (np.log10(xs) - f["m"])))
            ax.semilogx(xs, ys, "-", alpha=.6, label=f"logistic fit (R²={f['r2']:.2f})")
        for l in r.get("grid", []):
            ax.axvline(l, color="green", alpha=.5)
        ax.axhline(r["ceiling"], color="gray", ls="--", alpha=.5, label="ceiling est.")
        if r.get("lam_floor_target_N"):
            ax.axvline(r["lam_floor_target_N"], color="orange", ls="--", label="HardFlow floor @ target N")
        ax.set_title(f"{r['method']}  ({reward})"); ax.set_xlabel("lambda"); ax.set_ylabel("Δr")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "curves.png", dpi=110); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reward", required=True, choices=rs.REWARD_CHOICES)
    p.add_argument("--prompt", required=True)
    p.add_argument("--edit_prompt", default=None)
    p.add_argument("--methods", nargs="+", default=["mpc", "hardflow", "fmrg"], choices=["mpc", "hardflow", "fmrg"])
    p.add_argument("--probe_N", type=int, default=4, help="cheap N used for range finding")
    p.add_argument("--target_N", type=int, default=10, help="N of the sweep you will run with the grid")
    p.add_argument("--n_grid", type=int, default=6, help="lambda points in the recommended grid")
    p.add_argument("--max_evals", type=int, default=10, help="bracketing budget per method (refinement adds n_grid)")
    p.add_argument("--tol", type=float, default=0.05, help="flatness tolerance as a fraction of the reward ceiling")
    p.add_argument("--tol_abs", type=float, default=0.0, help="absolute floor for the tolerance")
    p.add_argument("--evaluate_grid", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_mode", default="flow_map2")
    p.add_argument("--height", type=int, default=256); p.add_argument("--width", type=int, default=256)
    p.add_argument("--mpc_H", type=int, default=2); p.add_argument("--mpc_n_inner", type=int, default=6)
    p.add_argument("--mpc_lr_inner", type=float, default=1e-2)
    p.add_argument("--endpoint_surrogate", default="jump", choices=["rollout", "jump"],
                   help="MPC surrogate for range finding; jump is ~3x cheaper and the transition matches")
    p.add_argument("--hardflow_n_inner", type=int, default=40); p.add_argument("--hardflow_lr_inner", type=float, default=2.5)
    p.add_argument("--fmrg_n_opt", type=int, default=3)
    p.add_argument("--reward_dtype", default="fp32", choices=list(rs._REWARD_DTYPES))
    p.add_argument("--fast", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--flux_device", default="cuda:0"); p.add_argument("--reward_device", default="cuda:0")
    p.add_argument("--quantize", default="int8"); p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--lora_path", default=None)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    flux_device, reward_device = rs._resolve_devices(a.flux_device, a.reward_device)
    lora_path = a.lora_path or os.path.join(rs.FMRG_DIR, "checkpoints", "flux-flowmap-lora")
    print(f"Loading FLUX ({a.quantize}) on {flux_device}; reward {a.reward} on {reward_device} ...", flush=True)
    sampler = rs.FluxFlowMapSampler(model_id=a.model_id, lora_path=lora_path, device=str(flux_device),
                                    dtype=torch.bfloat16, prompt=a.prompt, quantize_mode=a.quantize)
    device = next(sampler.pipeline.transformer.parameters()).device
    dtype = next(sampler.pipeline.transformer.parameters()).dtype
    for mod in (sampler.pipeline.transformer, sampler.pipeline.vae):      # same freeze as run_sweep.main
        for prm in mod.parameters():
            if prm.requires_grad:
                prm.requires_grad_(False)

    # ── probes ────────────────────────────────────────────────────────────
    probe_cfg = make_cfg("mpc", a, 1.0)
    pr = probe_uncontrolled(sampler, probe_cfg, device, dtype, a.reward, reward_device, rs._REWARD_DTYPES[a.reward_dtype])
    r_lo, r_hi = REWARD_OBSERVED_RANGE[a.reward]
    ceiling = max(r_hi - pr["r_unc"], 1e-6)
    print(f"probe: r_unc={pr['r_unc']:.4f}  observed reward max={r_hi}  -> dr ceiling est={ceiling:.3f}   "
          f"||grad r||={pr['grad_norm']:.4g}  ||z||={pr['z_norm']:.3g} ({pr['n_latent']} elems)", flush=True)

    ev = Evaluator(sampler, device, reward_device, dtype, a)
    results = []
    for m in a.methods:
        fl_probe = hardflow_floor(a.probe_N, a.hardflow_lr_inner) if m == "hardflow" else 0.0
        fl_target = hardflow_floor(a.target_N, a.hardflow_lr_inner) if m == "hardflow" else 0.0
        results.append(autorange(m, ev, a, pr, ceiling, fl_probe, fl_target))

    # ── outputs ───────────────────────────────────────────────────────────
    with open(out / "evaluations.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ev.rows[0].keys())); w.writeheader(); w.writerows(ev.rows)
    summary = dict(reward=a.reward, prompt=a.prompt, edit_prompt=a.edit_prompt, probe=pr, ceiling=ceiling,
                   probe_N=a.probe_N, target_N=a.target_N, args=vars(a), results=results)
    (out / "autorange.json").write_text(json.dumps(summary, indent=1, default=float))
    lines = []
    for r in results:
        if r["status"] == "insufficient":
            lines.append(f"# {r['method']}: {r['status']}"); continue
        if r["status"] != "ok":
            lines.append(f"# {r['method']}: {r['status']} -- {r.get('note','')}")
        g = " ".join(f"{l:g}" for l in r["grid_for_target_N"])
        note = (f"   # {len(r['clipped_by_target_floor'])} point(s) below HardFlow floor lam>={r['lam_floor_target_N']:g} at N={a.target_N} dropped"
                if r["clipped_by_target_floor"] else "")
        lines.append(f"{r['method']:8} --sweep_lam {g}{note}")
        lines.append(f"{'':8} # active region lam in [{r['active_lo']:g}, {r['active_hi']:g}], "
                     f"{r['n_evals']} evals, {r['wall_s']:.0f}s"
                     + (f", logistic R2={r['fit']['r2']:.2f}" if r.get('fit') else ""))
    (out / "grids.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    try:
        plot(results, out, a.reward)
    except Exception as e:  # plotting must never fail the run
        print(f"(plot skipped: {e})")
    print(f"\nwrote {out}/autorange.json, evaluations.csv, grids.txt, curves.png")


if __name__ == "__main__":
    main()
