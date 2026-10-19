#!/usr/bin/env python3
"""
lambda_autorange.py -- pick a lambda grid automatically, per method, for a
given (reward, prompt, edit_prompt), so a big unattended sweep does not need
its lambda range hand-tuned for every reward/prompt combination.

WHY THIS EXISTS (2026-09-16). `lam` means a different thing in each method --
a Lagrange weight the optimiser solves against (ConicFlow), the same quantity via
reg = N/(2*lam) (HardFlow), and an open-loop gain on the reward gradient
(FMRG) -- and its useful range also moves with the reward's scale (switching
LPIPS -> ImageReward de-regularises by ~200x at fixed lam). A fixed grid such
as {0.1 .. 10000} therefore spends most of its points where nothing happens:
for ImageReward, ConicFlow is already 79% saturated at lam=1 and fully saturated by
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

HOW FAR UP (--high_stop, 2026-09-23). The default stops at the plateau, which is
where the reward stops paying -- NOT where the edit stops getting stronger, since
energy keeps climbing past it. `--high_stop break` instead walks up until the
IMAGE breaks (pixel-side statistics against the uncontrolled sample, see the
PERMISSIVE HIGH END block), records the plateau onset as `lam_knee` on the way
past, and extends the grid above it. Use it when the sweep wants the aggressive
end of the frontier; measured on pickscore, ConicFlow at lam=280 gives delta_r=13.0 on
a clean image while the default grid for that cell topped out at lam=21.

It runs in ONE process -- FLUX loads once, the reward model loads once per
(reward, edit_prompt) -- by calling run_sweep._run_one directly, so every
guard in that path (locked defaults, plausibility ranges, non-finite checks)
still applies. Range-finding uses a cheap proxy N (default 4); the recommended
grid is for the sweep you then run at --target_N. The transition in lam is
approximately N-invariant for ConicFlow (lam is per-step normalised via dt) but
HardFlow's admissible floor lam >= N*lr/2 is not, so the floor is reported for
the target N, not the proxy.

Usage:
  python lambda_autorange.py --reward imagereward \\
      --prompt "a portrait of a man" --edit_prompt "a smiling face" \\
      --methods conicflow hardflow fmrg --probe_N 4 --target_N 10 --n_grid 6 \\
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
SPIKE_DECADES = 3.0                     # energy spike threshold; see mark_suspects

# ─────────────────────────────────────────────────────────────────────────────
# PERMISSIVE HIGH END (2026-09-23): keep lambda increasing until the images
# break, so as to reach less conservative edits.
#
# The default ("plateau") behaviour stops the upward walk as soon as the reward
# curve flattens, and then places the grid at 8-92% of the fitted rise. Both
# were measured to be conservative by 1-2 decades:
#
#   * 24 of the 51 method-searches on disk stopped on `near_ceiling`, i.e.
#     against REWARD_OBSERVED_RANGE, whose pickscore entry (18.0, 27.0) is
#     labelled "MEASURED but WEAK: n=4 only" in reward_ranges.py. ConicFlow on
#     pickscore/"a portrait of a man" reaches r_ctrl ~= 32.8 at lam=280, so that
#     ceiling is simply too low and the stop fires while the curve is climbing.
#   * The 92% quantile caps placement well below the bracketed region: on that
#     same cell the emitted ConicFlow grid topped out at lam=21 (delta_r ~3.8) while
#     the active region ran to 279 and lam=280 measured delta_r=13.0 with a
#     clean image (results/auto/pickscore/.../conicflow_seed42, lam0.95..280).
#
# In "break" mode the plateau tests become recorded MARKERS (lam_knee) instead
# of stops, and the walk continues until the IMAGE breaks or the reward
# collapses. Reward alone cannot be the break signal: on pickscore HardFlow
# scores 11.7-15.8 on visibly destroyed images, i.e. the scorer rewards the
# wreckage, which is why the test below is pixel-side.
#
# THRESHOLD PROVENANCE, read before trusting them. Calibrated on ONE cell
# (pickscore, "a portrait of a man" -> "a smiling face", seed 42, N=8) where the
# FMRG arm is visibly clean at lam=130 and visibly destroyed at lam=280:
#
#   metric                       conicflow 280  hf 280  fmrg 130  fmrg 280  threshold
#   clipped-pixel excess vs unc     0.000   0.000     0.038     0.206      0.10
#   total-variation ratio            1.12    1.11      1.35      1.52      1.45
#   pixel-std ratio                  1.03    1.00      1.50      1.95      1.80
#   relative pixel L2                0.26    0.26      0.74      1.31      1.00
#
# So they separate "destroyed" from "large but clean" on n=1, and nothing more.
# They are a CATASTROPHE stop, not a quality judgment -- they do not catch
# HardFlow's crunchy-texture degradation at lam=280, which is visible to the eye
# but moves none of them. That is why break mode also writes the decoded image
# for every lambda it evaluates (--save_images): the operator decides where
# "too far" is by looking, and these numbers only stop the search running away.
# ─────────────────────────────────────────────────────────────────────────────
BREAK_CLIP_EXCESS = 0.10                # fraction of channel values at 0/255, above uncontrolled
BREAK_TV_RATIO    = 1.45                # total variation vs uncontrolled
BREAK_STD_RATIO   = 1.80                # pixel std vs uncontrolled
BREAK_PIX_L2      = 1.00                # ||img_ctrl - img_unc|| / ||img_unc||
BREAK_DR_DROP     = 0.50                # reward collapse, as a fraction of the observed rise
CEILING_STOP_FRAC = 0.90                # near_ceiling stop; >1 disables it


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
    latent displacement that costs; every method's lambda multiplies (ConicFlow/HF via
    the energy trade-off J = E/(2 lam) - r, FMRG as a gain on grad r) a
    displacement that is linear in lam to first order. A 1/||grad r||^2 law was
    tried first and was 3 decades low on ImageReward (||grad r|| ~ 12); the
    measured knees -- ConicFlow ~0.3 on ImageReward (||grad r|| 12.3, dr_max 3.3) and
    ~30 on CLIP (0.19, 0.65) -- sit within a decade of the 1/||grad r|| form
    with c ~ 1, which is what is used. FMRG's knee was ~2x ConicFlow's on both.
    """
    g = max(pr["grad_norm"], 1e-12)
    c = {"conicflow": 1.0, "hardflow": 1.0, "fmrg": 2.0}[method]
    lam = c * dr_max / g
    return float(min(max(lam, LAM_ABS_MIN), LAM_ABS_MAX))


def hardflow_floor(N: int, lr: float) -> float:
    """hardflow_image raises when lr*reg >= 1 with reg = N/(2*lam)  =>  lam >= N*lr/2."""
    return N * lr / 2.0


# ═════════════════════════════════════════════════════════════════════════════
# Image-side break statistics (2026-09-23) -- see the PERMISSIVE HIGH END block.
# _run_one already returns both decoded PIL images (`_img_ctrl`, `_img_unc`), so
# every one of these is free: no extra NFE, no extra model, no decode.
# ═════════════════════════════════════════════════════════════════════════════
IMG_STAT_KEYS = ("pix_l2", "tv_ratio", "std_ratio", "clip_excess")


def _nan_img_stats() -> Dict[str, float]:
    return {k: float("nan") for k in IMG_STAT_KEYS}


def image_break_stats(img_ctrl, img_unc) -> Dict[str, float]:
    """Four cheap off-manifold statistics of the controlled image vs the
    uncontrolled one at the same seed. All are ratios or relative quantities, so
    they are comparable across prompts (an intrinsically dark or busy
    uncontrolled image shifts the numerator and the denominator together)."""
    if img_ctrl is None or img_unc is None:
        return _nan_img_stats()
    a = np.asarray(img_ctrl.convert("RGB"), dtype=np.float64)
    b = np.asarray(img_unc.convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        return _nan_img_stats()

    def tv(x):    # mean absolute gradient, both axes
        return 0.5 * (np.abs(np.diff(x, axis=0)).mean() + np.abs(np.diff(x, axis=1)).mean())

    # `clip_excess` is measured AGAINST the uncontrolled image rather than
    # absolutely: the uncontrolled portrait cell already has 0.2% of its channel
    # values pinned at 0 (a dark background), and an absolute threshold would
    # charge that to the controller.
    clipped = lambda x: float((((x <= 2.0) | (x >= 253.0)).mean()))
    return dict(pix_l2=float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-9)),
                tv_ratio=float(tv(a) / max(tv(b), 1e-9)),
                std_ratio=float(a.std() / max(b.std(), 1e-9)),
                clip_excess=float(clipped(a) - clipped(b)))


def broken_reasons(row: dict, a) -> List[str]:
    """Which break thresholds this evaluation trips. Any one is enough; they are
    deliberately loose (see the provenance table) so that a large-but-clean edit
    does not stop the walk."""
    out = []
    if not math.isfinite(row.get("pix_l2", float("nan"))):
        return out                      # no image (failed run) -- not evidence of breaking
    if row["clip_excess"] > a.break_clip_excess:
        out.append(f"clip_excess={row['clip_excess']:.3f}>{a.break_clip_excess:g}")
    if row["tv_ratio"] > a.break_tv_ratio:
        out.append(f"tv_ratio={row['tv_ratio']:.2f}>{a.break_tv_ratio:g}")
    if row["std_ratio"] > a.break_std_ratio:
        out.append(f"std_ratio={row['std_ratio']:.2f}>{a.break_std_ratio:g}")
    if row["pix_l2"] > a.break_pix_l2:
        out.append(f"pix_l2={row['pix_l2']:.2f}>{a.break_pix_l2:g}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation of one lambda
# ═════════════════════════════════════════════════════════════════════════════
def make_cfg(method: str, a, lam: float) -> rs.RunConfig:
    common = dict(prompt=a.prompt, edit_prompt=a.edit_prompt, N=a.probe_N, seed=a.seed,
                  lam=float(lam), sample_mode=a.sample_mode, height=a.height, width=a.width)
    if method == "conicflow":
        cfg = rs.RunConfig(control_type="conicflow", H=a.conicflow_H, n_inner=a.conicflow_n_inner, lr_inner=a.conicflow_lr_inner,
                           endpoint_surrogate=a.endpoint_surrogate, **common)
    elif method == "hardflow":
        cfg = rs.RunConfig(control_type="hardflow", H=2, n_inner=a.hardflow_n_inner,
                           lr_inner=a.hardflow_lr_inner, hardflow_lr_alpha=a.hardflow_lr_alpha,
                           **common)
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
        self.img_dir = Path(a.out_dir) / "images" if a.save_images else None
        if self.img_dir:
            self.img_dir.mkdir(parents=True, exist_ok=True)

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
            # (2026-09-23) Image-side break statistics, and optionally the image
            # itself. Both come from _run_one's already-decoded PIL outputs; a
            # failed run returns img_ctrl = img_unc as a placeholder, so guard on
            # run_ok rather than on the images being present.
            row.update(image_break_stats(res.get("_img_ctrl"), res.get("_img_unc"))
                       if row["ok"] else _nan_img_stats())
            if self.img_dir is not None and row["ok"] and res.get("_img_ctrl") is not None:
                res["_img_ctrl"].save(self.img_dir / f"{method}__lam{lam:.4g}__ctrl.png")
                unc = self.img_dir / "uncontrolled.png"
                if not unc.exists() and res.get("_img_unc") is not None:
                    res["_img_unc"].save(unc)
        except Exception as e:  # e.g. HardFlow's lr*reg >= 1 guard, OOM
            row = dict(method=method, lam=float(lam), ok=False, r_unc=float("nan"), r_ctrl=float("nan"),
                       dr=float("nan"), E=float("nan"), J=float("nan"), t_ctrl=float("nan"),
                       err=f"{type(e).__name__}: {str(e)[:160]}")
            row.update(_nan_img_stats())
            torch.cuda.empty_cache()
        row["suspect"] = False      # set later by mark_suspects; keeps the CSV column stable
        row["broken"] = ""          # set by the upward stop in break mode; ditto
        row["wall_s"] = time.perf_counter() - t0
        self.rows.append(row)
        flag = "" if row["ok"] else f"   FAILED: {row['err'][:70]}"
        img = ("" if not math.isfinite(row["pix_l2"]) else
               f"  L2={row['pix_l2']:.2f} TV={row['tv_ratio']:.2f} sd={row['std_ratio']:.2f}"
               f" clip={row['clip_excess']:+.3f}")
        print(f"    {method:8} lam={lam:<10.4g} dr={row['dr']:8.4f}  E={row['E']:9.4f}  J={row['J']:9.4f}"
              f"  t={row['t_ctrl']:6.1f}s{img}{flag}", flush=True)
        return row


# ═════════════════════════════════════════════════════════════════════════════
# The search
# ═════════════════════════════════════════════════════════════════════════════
def _valid(points: Dict[float, dict], include_suspect: bool = False) -> List[dict]:
    return sorted((p for p in points.values()
                   if p["ok"] and math.isfinite(p["dr"]) and (include_suspect or not p.get("suspect"))),
                  key=lambda p: p["lam"])


def mark_suspects(points: Dict[float, dict], spike_decades: float = SPIKE_DECADES) -> List[dict]:
    """Flag control-energy SPIKES in place, and return the surviving points.

    WHY (2026-09-17). HardFlow's explicit fixed-step inner ascent can diverge at
    a single lambda and still return a finite number: on ImageReward at lam=27
    it produced ctrl_energy = 1.15e18 between neighbours at 2.9 and 11.8, with
    delta_r falling to -0.83. _run_one's guards only catch non-finite values, so
    that point survives into the curve, where it (a) trips the `falling` and
    `energy_waste` stopping tests and can truncate the bracket, and (b) drags
    the logistic fit, the active region and therefore the grid with it. One
    unstable solve is not a feature of the response curve.

    A SPIKE, not a threshold. Energy legitimately grows by many decades across a
    sweep -- FMRG's top is ~1e3 above its bottom and is real -- so a point is
    only suspect when it stands above BOTH of its lambda-neighbours by
    `spike_decades`, i.e. the curve comes back down afterwards. That is what
    separates one bad solve from the genuine overshoot regime, where energy goes
    up and STAYS up (FMRG on CLIP past lam~3500: energy 1e9 with delta_r
    falling, which is a real finding and must not be filtered away).

    Endpoints are never flagged: with no point beyond it there is no evidence
    the energy came back down. A spike that is still the last point therefore
    stays visible to the stopping tests until bracketing or refinement puts a
    point above it.

    Suspect points stay in `points`, in evaluations.csv and in the plots -- they
    are marked, not deleted, so a run that hit an instability is never silently
    prettier than it was.
    """
    vp = _valid(points, include_suspect=True)
    for p in vp:
        p["suspect"] = False
    if len(vp) < 3:
        return vp
    le = [math.log10(max(p["E"], 1e-300)) if math.isfinite(p["E"]) else float("inf") for p in vp]
    for i in range(1, len(vp) - 1):
        if le[i] - max(le[i - 1], le[i + 1]) > spike_decades:
            vp[i]["suspect"] = True
    return [p for p in vp if not p["suspect"]]


def saturation_reasons(vp: List[dict], ceiling: float, tol: float,
                       ceiling_frac: float = CEILING_STOP_FRAC) -> List[str]:
    """Which plateau tests fire at the top of the bracket. Judged on the OBSERVED
    rise, not the theoretical ceiling: the reward's documented maximum is what the
    model can reach on some image, not what this prompt pair can reach (CLIP's
    observed max 0.80 gave a ceiling of 0.65 here while the real plateau was 0.24,
    and a ceiling-only rule then walked lambda up to 3.5e6 looking for it)."""
    if len(vp) < 2:
        return []
    a, b = vp[-2], vp[-1]
    drs = [p["dr"] for p in vp]
    # (2026-09-23) `rise` was `b["dr"] - min(drs)`, i.e. the LAST point minus the
    # curve's minimum. On a collapsing curve the last point IS the minimum, so
    # rise was identically 0, `risen` was False, and flat/energy_waste/falling --
    # every test that needs it -- were gated off exactly in the overshoot case
    # `falling` was written for. Measured consequence: FMRG on
    # pickscore/"A Japanese garden at dawn" peaked at delta_r=4.19 (lam=25), fell
    # to -6.08, and the upward walk ran to LAM_ABS_MAX instead of stopping at
    # lam=160; 60 of 529 evaluations across the 17 cells on disk were spent above
    # the peak on a falling curve, and 6 searches reported an active region out to
    # >=1e5, which then dragged their shared grid (see docs/lambda_autorange.md).
    # The rise of a CURVE is max-min, which is what this now computes.
    rise = max(drs) - min(drs)
    gain = b["dr"] - a["dr"]                      # what the last decade bought
    # "risen" guards against calling two ~0 points saturated. 2*tol rather than a
    # larger multiple: HardFlow's admissible floor can sit partway up the curve, so
    # its whole visible rise may be small (CLIP: 0.08 with tol 0.033 -- a 4*tol
    # guard walked it to lam=5e6 before stopping, 2*tol stops at the plateau).
    risen = rise >= 2.0 * tol
    out = []
    if risen and gain < tol:
        out.append("flat")
    # ceiling_frac > 1 disables this test. It is the single most common stop --
    # 24 of 51 searches on disk -- and it fires against REWARD_OBSERVED_RANGE,
    # which for pickscore is an n=4 measurement; see the PERMISSIVE HIGH END block.
    if b["dr"] >= ceiling_frac * ceiling:
        out.append("near_ceiling")
    if risen and b["E"] > 3.0 * max(a["E"], 1e-9) and gain < 2.0 * tol:
        out.append("energy_waste")
    if risen and b["dr"] < max(drs) - tol:        # overshoot: reward is going back down
        out.append("falling")
    return out


def _saturated(vp: List[dict], ceiling: float, tol: float,
               ceiling_frac: float = CEILING_STOP_FRAC) -> bool:
    return bool(saturation_reasons(vp, ceiling, tol, ceiling_frac))


def break_stop_reasons(vp: List[dict], tol: float, a) -> List[str]:
    """The upward stop in --high_stop break: keep climbing until the IMAGE breaks
    or the reward collapses outright. The plateau tests are not consulted -- a
    flat reward curve is precisely the regime this mode is meant to keep walking
    through, because energy (hence how far the edit has moved) keeps rising
    there even when the scorer has stopped paying for it.

    The reward-collapse test requires a drop of `--break_dr_drop` x the observed
    rise, not the `tol` that `falling` uses. FMRG's response bifurcates near its
    knee (a 0.06 change in lam flipped delta_r between 0.09 and 2.94 on
    ImageReward), so a tol-sized dip is noise, not a collapse."""
    if not vp:
        return []
    b = vp[-1]
    drs = [p["dr"] for p in vp]
    out = list(broken_reasons(b, a))
    rise = max(drs) - min(drs)
    if rise >= 2.0 * tol and (max(drs) - b["dr"]) >= a.break_dr_drop * rise:
        out.append(f"dr_collapsed({b['dr']:.2f} from peak {max(drs):.2f})")
    return out


def _inactive(vp: List[dict], tol: float) -> bool:
    return len(vp) >= 1 and vp[0]["dr"] <= tol


def bracket(method: str, ev: Evaluator, lam0: float, ceiling: float, tol: float,
            lam_min_adm: float, max_evals: int, a,
            spike_decades: float = SPIKE_DECADES) -> tuple:
    """Returns (points, info). `info` carries what ended the upward walk and, in
    break mode, where the plateau began (`lam_knee`) -- which is the value the
    default mode would have stopped at, kept so the two modes stay comparable."""
    points: Dict[float, dict] = {}
    info: Dict[str, object] = dict(up_stop=[], up_stop_lam=None, lam_knee=None, knee_reasons=[])
    n = 0
    break_mode = a.high_stop == "break"
    # In break mode the upward step is configurable: x10 walks past the plateau
    # fast but lands the last point up to a decade beyond where the image broke,
    # which is a whole decade of grid the sweep then cannot use. x sqrt(10) halves
    # that at the cost of ~2 extra evaluations per decade.
    step = a.up_step if break_mode else 10.0

    def run(lam):
        nonlocal n
        lam = float(f"{lam:.3g}")
        if lam in points:
            return points[lam]
        n += 1
        points[lam] = ev(method, lam)
        # Re-flag after every evaluation so the stopping tests below never see a
        # spike -- a spurious `falling` would cut the bracket short.
        mark_suspects(points, spike_decades)
        return points[lam]

    lam0 = max(lam0, lam_min_adm)
    run(lam0)
    # ── upward: until saturated (default) or until it breaks (break mode) ─
    lam = lam0
    while n < max_evals and lam * step <= LAM_ABS_MAX:
        vp = _valid(points)
        # The knee is deliberately judged at the DEFAULT ceiling fraction even in
        # break mode (where the ceiling stop is off): its job is to record where
        # the plateau-mode run would have stopped, so the two modes stay
        # comparable on the same cell.
        knee = saturation_reasons(vp, ceiling, tol,
                                  CEILING_STOP_FRAC if break_mode else a.ceiling_stop_frac)
        if knee and info["lam_knee"] is None and vp:
            # Record the plateau onset whether or not we stop on it.
            info["lam_knee"] = vp[-1]["lam"]
            info["knee_reasons"] = knee
        stop = break_stop_reasons(vp, tol, a) if break_mode else knee
        if stop:
            info["up_stop"], info["up_stop_lam"] = stop, (vp[-1]["lam"] if vp else None)
            if break_mode and vp:
                vp[-1]["broken"] = ";".join(stop)
            break
        lam *= step
        run(lam)
    if not info["up_stop"]:
        info["up_stop"] = ["budget_or_box"]
        info["up_stop_lam"] = max((p["lam"] for p in _valid(points)), default=None)
    # ── downward: until inactive (or admissible floor) ───────────────────
    lam = lam0
    while n < max_evals and lam / 10 >= max(LAM_ABS_MIN, lam_min_adm):
        vp = _valid(points)
        if _inactive(vp, tol):
            break
        lam /= 10
        run(lam)
    return points, info


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


def quantile_grid(fit, n_grid: int, lo: float, hi: float,
                  q_lo: float = 0.08, q_hi: float = 0.92) -> List[float]:
    """lambda values where the fitted curve reaches equally spaced fractions of its rise.

    q_hi is the top of that range. At 0.92 the emitted grid stops where the fitted
    rise is 92% done, which on ConicFlow/pickscore cells sat 1-2 decades below the
    measured active region -- grid top lam=21 against an active region to 279,
    where the sweep later measured delta_r=13.0 on an intact image.

    RAISING q_hi IS THE WRONG FIX for that (tried 2026-09-23, 0.995): the offset
    from the fitted midpoint is ln(q/(1-q))/k decades, so the top quantile shoots
    up the tail -- on clip/cat (k=2.2) it jumped from 660 to 13000 and left a
    2-decade hole between it and the next point down. Use --n_high, which fills
    the range from the top quantile to `hi` uniformly in log lambda, and leave
    q_hi where it is."""
    qs = np.linspace(q_lo, q_hi, n_grid)
    xs = [fit["m"] + math.log(q / (1 - q)) / fit["k"] for q in qs]
    lams = [10 ** x for x in xs]
    lams = [min(max(l, lo), hi) for l in lams]
    return sorted({float(f"{l:.2g}") for l in lams})


def extend_high(grid: List[float], hi: float, n_high: int) -> List[float]:
    """Resolve the top of the range: make sure `hi` is on the grid and put
    n_high log-spaced points between it and the rest of the grid.

    Needed because above the knee the logistic is flat, so no quantile of the FIT
    can place a point there -- the information at the top of the curve is in the
    energy and in the image, not in the reward. This is what actually delivers
    "keep going until it breaks" as grid points rather than as a wider bracket.

    The anchor is the highest grid point that is not already at the top, because
    a high --q_hi tends to clip its last quantile to `hi` and leave one huge gap
    below it (measured: 18 -> 280, a factor 15). Filling that gap is the point;
    appending above `hi` is not, so the result never exceeds `hi`."""
    if n_high <= 0 or not grid:
        return grid
    g = sorted(set(grid))
    below = [l for l in g if l < hi / 1.25]
    if not below:
        return g                      # the grid already sits at the top of the range
    xs = np.linspace(math.log10(max(below)), math.log10(hi), n_high + 2)[1:]   # drop the anchor, keep hi
    return sorted(set(g) | {float(f"{10 ** x:.2g}") for x in xs})


def shared_grid(results: List[dict], n_max: int, min_ratio: float) -> dict:
    """One lambda grid for all methods, so runs can be compared at the SAME lambda.

    WHY THIS IS A LEGITIMATE COMPARISON. ConicFlow's per-step linearised problem
    min ||resid||^2/(2 lam dt) - <grad r, resid> has the closed form
    resid* = lam*dt*grad r; FMRG's update, n_opt steps of (dt/n_opt)*lam*grad r,
    is exactly that per step; HardFlow's reg = N/(2 lam) is matched per step by
    the derivation in RunConfig. So at a shared lam all three are solving the
    same first-order problem, and the CSV's `objective` column
    (E/(2 lam) - r_ctrl) scores each of them against it. Where they differ is
    feedback: ConicFlow re-solves, FMRG is open-loop and overshoots at high lam.

    CONSTRUCTION. Take the union of every method's quantile grid (each already
    spans 8-92% of that method's own rise), merge neighbours closer than
    `min_ratio`, then if more than n_max points remain drop, one at a time, the
    point whose removal least widens the largest log-gap. This keeps every
    method's transition resolved rather than spreading points log-uniformly over
    a union region one method may not use.
    """
    pts = sorted({float(f"{l:.2g}") for r in results if r.get("status") != "insufficient" for l in r.get("grid", [])})
    merged: List[float] = []
    for l in pts:
        if merged and l / merged[-1] < min_ratio:
            merged[-1] = float(f"{math.sqrt(merged[-1] * l):.2g}")     # replace the pair by their log-midpoint
        else:
            merged.append(l)
    while len(merged) > n_max:
        gaps = [math.log10(b / a_) for a_, b in zip(merged, merged[1:])]
        # removing interior point i merges gaps i-1 and i; pick the smallest merged gap
        cand = min(range(1, len(merged) - 1), key=lambda i: gaps[i - 1] + gaps[i])
        merged.pop(cand)
    coverage = {}
    for r in results:
        if r.get("status") == "insufficient":
            continue
        inside = [l for l in merged if r["active_lo"] <= l <= r["active_hi"]]
        coverage[r["method"]] = dict(n_inside=len(inside), active=[r["active_lo"], r["active_hi"]])
    return dict(grid=merged, coverage=coverage)


def loguniform_grid(n_grid: int, lo: float, hi: float) -> List[float]:
    return sorted({float(f"{10 ** x:.2g}") for x in np.linspace(math.log10(lo), math.log10(hi), n_grid)})


def autorange(method: str, ev: Evaluator, a, pr: dict, ceiling: float, lam_floor_probe: float,
              lam_floor_target: float) -> dict:
    tol = max(a.tol * ceiling, a.tol_abs)
    lam0 = scale_prior(method, pr, ceiling, a.probe_N)
    print(f"\n== {method}: prior lam0={lam0:.3g}  ceiling(dr_max)={ceiling:.3f}  tol={tol:.3f}"
          f"{f'  admissible floor lam>={lam_floor_probe:.3g} (probe N), >={lam_floor_target:.3g} (target N)' if lam_floor_probe else ''}")
    t0 = time.perf_counter()
    pts, binfo = bracket(method, ev, lam0, ceiling, tol, lam_floor_probe, a.max_evals, a, a.spike_decades)
    # Final pass over the whole set: a point that was the last one during
    # bracketing (and so unflaggable) may now have a neighbour above it.
    mark_suspects(pts, a.spike_decades)
    vp = _valid(pts)
    out = dict(method=method, prior_lam0=lam0, tol=tol, ceiling=ceiling,
               lam_floor_probe_N=lam_floor_probe, lam_floor_target_N=lam_floor_target,
               high_stop=a.high_stop, up_stop=binfo["up_stop"], up_stop_lam=binfo["up_stop_lam"],
               lam_knee=binfo["lam_knee"], knee_reasons=binfo["knee_reasons"])
    if len(vp) < 2:
        out.update(status="insufficient", grid=[], points=list(pts.values()))
        return out
    # active region: from the last point that is still inactive (or the lowest) to the first saturated
    lo_pts = [p for p in vp if p["dr"] <= tol]
    lo = lo_pts[-1]["lam"] if lo_pts else vp[0]["lam"]
    # (2026-09-23) A point whose image broke is a real measurement -- it stays in
    # `points`, in the CSV and in the plot -- but it must not shape the grid: its
    # reward has usually collapsed, which drags the logistic, and no grid should
    # ship the run that destroyed the image. Same treatment as energy spikes.
    vp_fit = [p for p in vp if not p.get("broken")] or vp
    sat_idx = next((i for i in range(1, len(vp_fit))
                    if _saturated(vp_fit[:i + 1], ceiling, tol, a.ceiling_stop_frac)), len(vp_fit) - 1)
    hi = vp_fit[sat_idx]["lam"]
    # In break mode the active region runs to the EDGE -- the highest lambda that
    # is neither broken nor collapsed -- rather than to the plateau onset, since
    # walking past the plateau is the whole point of the mode.
    lam_edge = None
    if a.high_stop == "break":
        lam_edge = vp_fit[-1]["lam"]
        hi = lam_edge
    if hi <= lo:
        lo, hi = vp_fit[0]["lam"], vp_fit[-1]["lam"]
    fit = fit_logistic(vp_fit, ceiling)
    grid = (quantile_grid(fit, a.n_grid, lo, hi, q_hi=a.q_hi) if fit and fit["r2"] > 0.5
            else loguniform_grid(a.n_grid, lo, hi))
    # quantiles collapsed (flat or nearly flat curve) or produced near-duplicates
    # after rounding: spread log-uniformly over the active region instead.
    if len(grid) < max(3, a.n_grid // 2) or any(b / a_ < 1.25 for a_, b in zip(grid, grid[1:])):
        grid = loguniform_grid(a.n_grid, lo, hi)
    # (2026-09-23) Extend the grid above the knee, up to the top of the active
    # region (= the break edge in break mode). Adds points, so the grid can be
    # n_grid + n_high long; that is deliberate -- these are the aggressive
    # operating points, and dropping a transition point to pay for one would
    # defeat the quantile placement.
    grid = extend_high(grid, hi, a.n_high)
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
        # (2026-09-23) Refined points get the break test too. Only the bracket's
        # stop sets `broken` otherwise, so without this a grid point that broke
        # the image would be reported as an ordinary operating point.
        if a.high_stop == "break":
            for p in pts.values():
                if p["ok"] and not p.get("broken"):
                    rs_ = broken_reasons(p, a)
                    if rs_:
                        p["broken"] = ";".join(rs_)
            # and then they leave the grid: the whole grid is one lambda coarser
            # than the walk, so a point can only be tested after it is proposed.
            grid = [l for l in grid if not pts.get(float(f"{l:.3g}"), {}).get("broken")]
        mark_suspects(pts, a.spike_decades)
        vp = _valid(pts)
        # refinement can add points above the knee in break mode; keep the fit on
        # the unbroken ones, as above
        fit = fit_logistic([p for p in vp if not p.get("broken")] or vp, ceiling) or fit
    # HardFlow: the target-N floor may cut the grid; report rather than silently shift
    clipped = [l for l in grid if l < lam_floor_target] if lam_floor_target else []
    grid_target = [l for l in grid if l >= lam_floor_target] if lam_floor_target else grid
    suspects = sorted(p["lam"] for p in pts.values() if p.get("suspect"))
    if suspects:
        print(f"    [suspect] energy spikes at lam {', '.join(f'{l:g}' for l in suspects)} "
              f"(>{a.spike_decades:g} decades above both neighbours) -- excluded from the fit, "
              f"the active region and the grid; still in evaluations.csv")
    broken_lams = sorted(p["lam"] for p in pts.values() if p.get("broken"))
    if a.high_stop == "break":
        _fmt = lambda v: "none" if v is None else f"{v:g}"
        print(f"    [high end] stopped on {';'.join(binfo['up_stop'])} at lam={_fmt(binfo['up_stop_lam'])}"
              f"   plateau began at lam={_fmt(binfo['lam_knee'])}"
              f" ({','.join(binfo['knee_reasons']) or '-'})   grid edge lam={_fmt(lam_edge)}")
    out.update(suspect_lams=suspects, lam_edge=lam_edge, broken_lams=broken_lams,
               status="saturated_at_admissible_floor" if floor_saturated else "ok",
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
        # (2026-09-23) knee vs break: in break mode the grid deliberately runs
        # past the plateau, so the plot has to show both or it looks like a bug.
        if r.get("lam_knee"):
            ax.axvline(r["lam_knee"], color="tab:blue", ls="-.", alpha=.7, label="plateau onset")
        for l in r.get("broken_lams", []):
            ax.axvline(l, color="black", ls="--", alpha=.8, label="image broke")
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
    p.add_argument("--methods", nargs="+", default=["conicflow", "hardflow", "fmrg"], choices=["conicflow", "hardflow", "fmrg"])
    p.add_argument("--probe_N", type=int, default=4, help="cheap N used for range finding")
    p.add_argument("--target_N", type=int, default=10, help="N of the sweep you will run with the grid")
    p.add_argument("--n_grid", type=int, default=6, help="lambda points in the recommended grid")
    p.add_argument("--max_evals", type=int, default=None,
                   help="bracketing budget per method (refinement adds n_grid). "
                        "Default 10, or 14 with --high_stop break, which walks further up.")
    # ── permissive high end (2026-09-23) ──────────────────────────────────
    # One switch, --high_stop break, with the other knobs resolved from it below
    # so the mode cannot be half-applied. Every default with high_stop=plateau is
    # the pre-2026-09-23 value, so the old behaviour is bit-for-bit reachable.
    p.add_argument("--high_stop", choices=["plateau", "break"], default="plateau",
                   help="where the upward walk stops. plateau (default): as soon as the reward "
                        "curve flattens. break: keep climbing until the IMAGE breaks or the reward "
                        "collapses, recording the plateau onset as lam_knee on the way past. Use "
                        "break when you want the aggressive end of the frontier; the plateau is "
                        "where reward stops paying, not where the edit stops getting stronger.")
    p.add_argument("--ceiling_stop_frac", type=float, default=None,
                   help="stop when delta_r reaches this fraction of the ceiling estimate. Default "
                        "0.9 (plateau) / disabled (break). The ceiling comes from "
                        "REWARD_OBSERVED_RANGE, which for pickscore is an n=4 measurement that "
                        "ConicFlow overshoots, so this is the first thing to loosen.")
    p.add_argument("--q_hi", type=float, default=None,
                   help="top logistic quantile for grid placement. Default 0.92 (plateau) / 0.995 "
                        "(break); the offset from the fitted midpoint is log10(q/(1-q))/k decades.")
    p.add_argument("--n_high", type=int, default=None,
                   help="extra log-spaced grid points between the top quantile and the top of the "
                        "active region. Default 0 (plateau) / 2 (break). Above the knee the fitted "
                        "logistic is flat, so no quantile can place a point there.")
    p.add_argument("--up_step", type=float, default=None,
                   help="multiplicative step of the upward walk. Default 10 (plateau) / 3.1623 "
                        "(break), which halves how far past the break point the last eval lands.")
    p.add_argument("--break_clip_excess", type=float, default=BREAK_CLIP_EXCESS)
    p.add_argument("--break_tv_ratio", type=float, default=BREAK_TV_RATIO)
    p.add_argument("--break_std_ratio", type=float, default=BREAK_STD_RATIO)
    p.add_argument("--break_pix_l2", type=float, default=BREAK_PIX_L2)
    p.add_argument("--break_dr_drop", type=float, default=BREAK_DR_DROP,
                   help="reward collapse, as a fraction of the observed rise; larger than "
                        "`falling`'s tol because FMRG's response bifurcates near its knee")
    p.add_argument("--save_images", type=lambda x: x.lower() != "false", default=None,
                   help="write the decoded image for every lambda to <out_dir>/images/. Default "
                        "off (plateau) / on (break): the break thresholds are calibrated on one "
                        "cell, so the operator should be able to look.")
    p.add_argument("--tol", type=float, default=0.05, help="flatness tolerance as a fraction of the reward ceiling")
    p.add_argument("--tol_abs", type=float, default=0.0, help="absolute floor for the tolerance")
    p.add_argument("--spike_decades", type=float, default=SPIKE_DECADES,
                   help="a point whose control energy stands this many decades above BOTH its "
                        "lambda-neighbours is a diverged solve, not curve shape: it is marked "
                        "suspect and excluded from the fit, the active region and the grid "
                        "(still recorded). 0 disables.")
    p.add_argument("--evaluate_grid", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_mode", default="flow_map2")
    p.add_argument("--height", type=int, default=256); p.add_argument("--width", type=int, default=256)
    p.add_argument("--conicflow_H", type=int, default=2); p.add_argument("--conicflow_n_inner", type=int, default=6)
    p.add_argument("--conicflow_lr_inner", type=float, default=1e-2)
    p.add_argument("--endpoint_surrogate", default="jump", choices=["rollout", "jump"],
                   help="ConicFlow surrogate for range finding; jump is ~3x cheaper and the transition matches")
    p.add_argument("--hardflow_n_inner", type=int, default=40); p.add_argument("--hardflow_lr_inner", type=float, default=2.5)
    p.add_argument("--hardflow_lr_alpha", type=float, default=None,
                   help="lift HardFlow's lr*reg >= 1 floor: below it lr = ALPHA/reg (see run_sweep.py). "
                        "Without this HardFlow's transition is unreachable for high-gradient rewards.")
    p.add_argument("--n_grid_shared", type=int, default=8, help="points in the shared (all-method) grid")
    p.add_argument("--shared_min_ratio", type=float, default=1.4,
                   help="neighbouring shared-grid points closer than this ratio are merged")
    p.add_argument("--fmrg_n_opt", type=int, default=3)
    p.add_argument("--reward_dtype", default="fp32", choices=list(rs._REWARD_DTYPES))
    p.add_argument("--fast", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--flux_device", default="cuda:0"); p.add_argument("--reward_device", default="cuda:0")
    p.add_argument("--quantize", default="int8"); p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--lora_path", default=None)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()
    # (2026-09-23) Resolve the mode-dependent defaults in ONE place, and print
    # them, so a run's high-end behaviour is never half-configured and the json's
    # `args` block records what actually applied rather than a pile of Nones.
    _MODE_DEFAULTS = {"plateau": dict(max_evals=10, ceiling_stop_frac=CEILING_STOP_FRAC, q_hi=0.92,
                                      n_high=0, up_step=10.0, save_images=False),
                      # q_hi stays at 0.92 in break mode on purpose -- see
                      # quantile_grid: the top end is opened up with n_high, not
                      # by pushing one quantile out onto the logistic's tail.
                      "break":   dict(max_evals=14, ceiling_stop_frac=1e9, q_hi=0.92,
                                      n_high=2, up_step=math.sqrt(10.0), save_images=True)}
    for k, v in _MODE_DEFAULTS[a.high_stop].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if not 0.5 < a.q_hi < 1.0:
        raise ValueError(f"--q_hi must be in (0.5, 1.0), got {a.q_hi}")
    print(f"high end: --high_stop {a.high_stop}  ceiling_stop_frac="
          f"{'disabled' if a.ceiling_stop_frac > 1 else f'{a.ceiling_stop_frac:g}'}  q_hi={a.q_hi:g}  "
          f"n_high={a.n_high}  up_step={a.up_step:.4g}  max_evals={a.max_evals}  "
          f"save_images={a.save_images}", flush=True)

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
    probe_cfg = make_cfg("conicflow", a, 1.0)
    pr = probe_uncontrolled(sampler, probe_cfg, device, dtype, a.reward, reward_device, rs._REWARD_DTYPES[a.reward_dtype])
    r_lo, r_hi = REWARD_OBSERVED_RANGE[a.reward]
    ceiling = max(r_hi - pr["r_unc"], 1e-6)
    print(f"probe: r_unc={pr['r_unc']:.4f}  observed reward max={r_hi}  -> dr ceiling est={ceiling:.3f}   "
          f"||grad r||={pr['grad_norm']:.4g}  ||z||={pr['z_norm']:.3g} ({pr['n_latent']} elems)", flush=True)

    ev = Evaluator(sampler, device, reward_device, dtype, a)
    results = []
    for m in a.methods:
        # With --hardflow_lr_alpha the lr*reg >= 1 floor is lifted (run_sweep pins
        # lr*reg at alpha below it), so there is no admissible boundary to report.
        lifted = m == "hardflow" and a.hardflow_lr_alpha is not None
        fl_probe = hardflow_floor(a.probe_N, a.hardflow_lr_inner) if (m == "hardflow" and not lifted) else 0.0
        fl_target = hardflow_floor(a.target_N, a.hardflow_lr_inner) if (m == "hardflow" and not lifted) else 0.0
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
        if r.get("suspect_lams"):
            lines.append(f"{'':8} # EXCLUDED energy spikes at lam "
                         f"{' '.join(f'{l:g}' for l in r['suspect_lams'])} (diverged solves)")
        lines.append(f"{'':8} # active region lam in [{r['active_lo']:g}, {r['active_hi']:g}], "
                     f"{r['n_evals']} evals, {r['wall_s']:.0f}s"
                     + (f", logistic R2={r['fit']['r2']:.2f}" if r.get('fit') else ""))
        # (2026-09-23) In break mode say, per method, where the plateau began and
        # what ended the walk -- the whole point of the mode is that the grid now
        # extends past the knee, so the knee has to stay visible.
        if r.get("high_stop") == "break":
            knee = "none" if r.get("lam_knee") is None else f"{r['lam_knee']:g}"
            edge = "none" if r.get("lam_edge") is None else f"{r['lam_edge']:g}"
            lines.append(f"{'':8} # plateau began at lam={knee}"
                         f" ({','.join(r.get('knee_reasons') or []) or '-'}), walked to lam="
                         f"{r['up_stop_lam']:g}, stopped on {';'.join(r['up_stop'])}; "
                         f"grid edge lam={edge}")
            if r.get("broken_lams"):
                lines.append(f"{'':8} # BROKEN (excluded from the grid) at lam "
                             f"{' '.join(f'{l:g}' for l in r['broken_lams'])} "
                             f"-- look at images/ before trusting the top of this grid")
    sh = shared_grid(results, a.n_grid_shared, a.shared_min_ratio)
    summary["shared"] = sh
    lines.append("")
    lines.append(f"shared   --sweep_lam {' '.join(f'{l:g}' for l in sh['grid'])}")
    for m, c in sh["coverage"].items():
        flag = "" if c["n_inside"] >= 3 else "   # UNDER-RESOLVED: raise --n_grid_shared"
        lines.append(f"{'':8} # {m}: {c['n_inside']} of {len(sh['grid'])} shared points inside its active "
                     f"region [{c['active'][0]:g}, {c['active'][1]:g}]{flag}")
    (out / "autorange.json").write_text(json.dumps(summary, indent=1, default=float))
    (out / "grids.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    try:
        plot(results, out, a.reward)
    except Exception as e:  # plotting must never fail the run
        print(f"(plot skipped: {e})")
    print(f"\nwrote {out}/autorange.json, evaluations.csv, grids.txt, curves.png")


if __name__ == "__main__":
    main()
