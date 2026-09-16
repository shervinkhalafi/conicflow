"""
run_sweep.py — Controlled vs uncontrolled FLUX image generation sweep
======================================================================

Shervin: Thew unconstrained sweeps that ran with this work well.

Runs every combination in SWEEP_CONFIGS, saves images + a CSV summary.

Usage
-----
    # inside the flowmaps conda env, with the sampler already loadable:
    python run_sweep.py [--out_dir results] [--seed 42] [--quiet]

Requires
--------
  - sampler loaded via FluxFlowMapSampler (configured below, or pass --hf_token)
  - FMRG repo on PYTHONPATH (set FMRG_DIR below or in the environment)
  - CLIP model cached at HF_HOME (auto-downloaded on first run)

Sweep dimensions
----------------
  prompt, N (total steps), H (horizon), control_type, lam, sample_mode
  Add/remove entries in SWEEP_CONFIGS to taste.
"""

import argparse
import csv
import gc
import itertools
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import torch
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt

# ── path setup ───────────────────────────────────────────────────────────────
# This is a standalone copy (sungcho) — image_mpc.py / fluxfm_sampler_reward.py
# are NOT duplicated here, they're imported read-only from Shervin's original
# checkout. Override FLOW_MATCHING_DIR / FMRG_DIR env vars once you have your
# own copies of those modules.
FLOW_MATCHING_DIR = os.environ.get("FLOW_MATCHING_DIR", "/home/shervink/flow_matching")
FMRG_DIR = os.environ.get("FMRG_DIR", os.path.join(FLOW_MATCHING_DIR, "fmrg"))
sys.path.insert(0, FMRG_DIR)
sys.path.insert(0, FLOW_MATCHING_DIR)           # image_mpc.py lives here (read-only import)
sys.path.insert(0, os.path.dirname(__file__))

from image_mpc import (
    make_flux_wrappers,
    receding_horizon_flow_control,
    fmrg_j_image,
    trajectory_rewards,
    tune_sampler_speed,
)
from fluxfm_sampler_reward import (
    FluxFlowMapSampler,
    CLIPReward,
    HPSReward,
    ImageRewardReward,
    PickScoreReward,
    create_clip_preprocess,
    HF_CACHE,
)

try:
    from transformers import AutoProcessor
except ImportError:
    raise SystemExit("transformers not installed. Run: pip install transformers")

REWARD_CHOICES = ["clip", "hpsv2", "pickscore", "imagereward"]


# ═════════════════════════════════════════════════════════════════════════════
# Sweep configuration — edit this block
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RunConfig:
    prompt:       str   = "a cat wearing sunglasses on a beach"     # image-generation prompt
    edit_prompt:  Optional[str] = None  # reward-model target prompt; falls back to `prompt` if None
    N:            int   = 4       # total denoising steps
    H:            int   = 2       # MPC planning horizon (ignored for fmrg)
    control_type: str   = "mpc"   # "mpc" | "fmrg" | "none"
    lam:          float = 10.0    # control / guidance strength
    n_inner:      int   = 5       # inner iterations (MPC) or n_opt (FMRG)
    lr_inner:     float = 1e-2    # inner Adam lr (MPC only)
    sample_mode:  str   = "flow_map2"
    guidance_scale: float = 3.5
    height:       int   = 256
    width:        int   = 256
    warmstart:    bool  = True    # MPC only
    drift_detach: bool  = True    # MPC only (approx but fast)
    constraint_on: str  = "horizon"  # MPC only; "horizon" | "endpoint" -- no-op
                                      # while h=None (no constraint fn wired up),
                                      # recorded for provenance only
    # populated at runtime:
    seed:         int   = 42
    out_dir:      str   = "results"


SWEEP_CONFIGS: List[RunConfig] = [
    # ── prompt sweep ────────────────────────────────────────────────────────
    RunConfig(prompt="a sunset over the ocean",         N=4, H=2, control_type="mpc",  lam=50.0),
    RunConfig(prompt="a sunset over the ocean",         N=4, H=2, control_type="fmrg", lam=5.0),
    RunConfig(prompt="a sunset over the ocean",         N=4, H=2, control_type="none"),
    # ── horizon sweep (mpc) ─────────────────────────────────────────────────
    RunConfig(prompt="a cat wearing sunglasses",        N=4, H=1, control_type="mpc",  lam=50.0),
    RunConfig(prompt="a cat wearing sunglasses",        N=4, H=2, control_type="mpc",  lam=50.0),
    RunConfig(prompt="a cat wearing sunglasses",        N=4, H=3, control_type="mpc",  lam=50.0),
    # ── N sweep ─────────────────────────────────────────────────────────────
    RunConfig(prompt="a cat wearing sunglasses",        N=3, H=2, control_type="mpc",  lam=50.0),
    RunConfig(prompt="a cat wearing sunglasses",        N=6, H=2, control_type="mpc",  lam=50.0),
    # ── lambda sweep ────────────────────────────────────────────────────────
    RunConfig(prompt="a cat wearing sunglasses",        N=4, H=2, control_type="mpc",  lam=5.0),
    RunConfig(prompt="a cat wearing sunglasses",        N=4, H=2, control_type="mpc",  lam=200.0),
    # ── sample mode comparison ───────────────────────────────────────────────
    RunConfig(prompt="a cat wearing sunglasses",        N=4, H=2, control_type="mpc",  lam=50.0, sample_mode="flow_map1"),
    # ── edit_prompt example: generate with one prompt, reward against another ──
    RunConfig(prompt="a portrait of a man",             edit_prompt="a smiling face",
              N=4, H=2, control_type="mpc",  lam=50.0),
    RunConfig(prompt="a portrait of a man",             edit_prompt="a smiling face",
              N=4, H=2, control_type="fmrg", lam=5.0),
]


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _free(*tensors):
    for t in tensors:
        del t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_uncontrolled(sampler, z0, cfg: RunConfig, device, dtype):
    """Roll the uncontrolled FLUX flow and return (z_final, traj)."""
    dt_flux = 1.0 / cfg.N
    ts = torch.linspace(1.0, 0.0, cfg.N + 1, device=device, dtype=dtype)
    z = z0.clone()
    traj = [z.clone()]
    with torch.no_grad():
        for j in range(cfg.N):
            t_c = ts[j].item()
            if cfg.sample_mode == "flow_map2":
                t_n = max(0.0, t_c - dt_flux)
            elif cfg.sample_mode == "flow_map1":
                t_n = 0.0
            else:
                t_n = t_c
            dt = (ts[j] - ts[j + 1]).item()
            u = sampler.predict_vector(
                z, t_c,
                prompt_embeds=sampler.cached_prompt_embeds,
                pooled_prompt_embeds=sampler.cached_pooled_prompt_embeds,
                guidance_scale=cfg.guidance_scale,
                t_next=t_n,
            )
            z = z - dt * u
            traj.append(z.clone())
    return z, torch.stack(traj, dim=0)


def _ctrl_norms_from_traj(
    traj: torch.Tensor,
    v_theta_mpc,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """Compute per-step control inputs from the trajectory using the dynamics.

    Uses the definition   u_t = x_{t+1} - x_t - dt * b_t(x_t)
    (i.e. the residual between the actual step and the uncontrolled drift).

    This is independent of how the control was parameterized internally (MPC
    stores u/dt, FMRG stores the guidance displacement), so norms are
    comparable across methods.

    Requires N transformer fwd passes (no_grad).

    Returns:
        ctrl_norms:  (N,) numpy array — mean ‖u_t‖ over batch per step
        ctrl_energy: scalar — Σ_t ‖u_t‖² · dt
    """
    dt = 1.0 / N
    t_grid = torch.linspace(0.0, 1.0, N + 1)
    norms = []
    with torch.no_grad():
        for t_idx in range(N):
            x_t   = traj[t_idx    ].to(device=device, dtype=dtype)
            x_next = traj[t_idx + 1].to(device=device, dtype=dtype)
            b_t   = v_theta_mpc(t_grid[t_idx].to(device), x_t)
            u_t   = x_next - x_t - dt * b_t          # displacement residual
            norms.append(u_t.flatten(1).norm(dim=1).mean().item())
    import numpy as np
    norms = np.array(norms, dtype=np.float32)
    ctrl_energy = float((norms ** 2 * dt).sum())
    return norms, ctrl_energy


def _decode_pil(sampler, z):
    img = sampler.decode_no_grad(z).float().detach().cpu()
    return sampler.pipeline.image_processor.postprocess(img, output_type="pil")[0]


def _resolve_devices(flux_device: str, reward_device: str) -> tuple[torch.device, torch.device]:
    """Resolve FLUX and reward-model devices, with sensible multi-GPU defaults."""
    flux_dev = torch.device(flux_device)
    if reward_device != "auto":
        return flux_dev, torch.device(reward_device)

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return flux_dev, torch.device("cpu")

    flux_idx = flux_dev.index if flux_dev.type == "cuda" else 0
    for idx in range(torch.cuda.device_count()):
        if idx != flux_idx:
            return flux_dev, torch.device(f"cuda:{idx}")
    return flux_dev, torch.device("cpu")


def _build_reward_fn(sampler, cfg, reward_device, dtype, reward_model: str = "clip"):
    """Return ``(r_image, reward_obj)`` for the given config and reward model.

    All reward models run on ``reward_device``.  VAE decode stays on the FLUX
    device; only the decoded image is moved for scoring.

    Supported ``reward_model`` values:
        "clip"        — CLIP ViT-H-14 cosine similarity (default, fast)
        "hpsv2"       — HPSv2 aesthetic/human-preference score
        "pickscore"   — PickScore human-preference model
        "imagereward" — ImageReward (requires ``ImageReward`` package)
    """
    reward_model = reward_model.lower()
    preprocess = create_clip_preprocess(224)
    # Reward is scored against edit_prompt when set, else falls back to the
    # generation prompt (e.g. cfg.prompt="a portrait of a man",
    # cfg.edit_prompt="a smiling face" edits toward the CLIP-smiling direction
    # while still generating a man's portrait as the starting image).
    prompt = cfg.edit_prompt if cfg.edit_prompt else cfg.prompt

    # ── CLIP ────────────────────────────────────────────────────────────────
    if reward_model == "clip":
        tokenizer = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=HF_CACHE
        )
        reward_obj = CLIPReward(
            device=reward_device, dtype=torch.float32,
            tokenizer=tokenizer, cache_dir=HF_CACHE,
        )
        with torch.no_grad():
            toks = tokenizer(
                prompt, return_tensors="pt", padding=True,
                max_length=77, truncation=True,
            ).to(reward_device)
            txt_feat = reward_obj.model.get_text_features(**toks)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        def r_image(z: torch.Tensor) -> torch.Tensor:
            img = sampler.decode(z)
            img_01 = ((img + 1.0) / 2.0).float().to(reward_device)
            img_feat = reward_obj.model.get_image_features(preprocess(img_01))
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            return (img_feat @ txt_feat.T).squeeze(-1).to(z.device)

    # ── HPSv2 ────────────────────────────────────────────────────────────────
    elif reward_model == "hpsv2":
        reward_obj = HPSReward(
            device=reward_device, dtype=torch.float32, cache_dir=HF_CACHE,
        )

        def r_image(z: torch.Tensor) -> torch.Tensor:
            img = sampler.decode(z)
            img_01 = ((img + 1.0) / 2.0).float().to(reward_device)
            # HPSReward.forward returns a scalar loss = 1 - sim; negate for reward
            loss = reward_obj(preprocess(img_01), prompt)
            return (1.0 - loss).unsqueeze(0).to(z.device)

    # ── PickScore ────────────────────────────────────────────────────────────
    elif reward_model == "pickscore":
        tokenizer = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=HF_CACHE
        )
        reward_obj = PickScoreReward(
            device=reward_device, dtype=torch.float32,
            tokenizer=tokenizer, cache_dir=HF_CACHE,
        )
        with torch.no_grad():
            toks = tokenizer(
                prompt, return_tensors="pt", padding=True,
                max_length=77, truncation=True,
            ).to(reward_device)
            txt_feat = reward_obj.model.get_text_features(**toks)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        def r_image(z: torch.Tensor) -> torch.Tensor:
            img = sampler.decode(z)
            img_01 = ((img + 1.0) / 2.0).float().to(reward_device)
            img_feat = reward_obj.model.get_image_features(preprocess(img_01))
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sim = (reward_obj.model.logit_scale.exp() * (img_feat @ txt_feat.T))
            return sim.squeeze(-1).to(z.device)

    # ── ImageReward ───────────────────────────────────────────────────────────
    elif reward_model == "imagereward":
        # ImageReward uses transformers 4.x APIs; apply compat shim first.
        try:
            import imagereward_compat  # noqa: F401
        except ImportError:
            pass  # only needed with transformers 5.x; silently skip if absent
        reward_obj = ImageRewardReward(
            device=reward_device, dtype=torch.float32, cache_dir=HF_CACHE,
        )

        def r_image(z: torch.Tensor) -> torch.Tensor:
            img = sampler.decode(z)
            img_01 = ((img + 1.0) / 2.0).float().to(reward_device)
            # ImageRewardReward.forward returns loss = 2 - score; negate → reward = score - 2
            loss = reward_obj(preprocess(img_01), prompt)
            return (2.0 - loss).unsqueeze(0).to(z.device)

    else:
        raise ValueError(
            f"Unknown reward_model '{reward_model}'. "
            f"Choose from: {REWARD_CHOICES}"
        )

    return r_image, reward_obj


def _run_one(
    sampler,
    cfg: RunConfig,
    device,
    reward_device,
    dtype,
    quiet: bool,
    *,
    fast: bool = False,
    low_memory: bool = True,
    reward_model: str = "clip",
) -> dict:
    """Execute a single config and return a metrics dict."""
    log = (lambda *a, **k: None) if quiet else print

    # Apply speed settings per-run so transformer checkpointing is toggled
    # correctly when drift_detach varies across configs in the same sweep.
    tune_sampler_speed(
        sampler, fast=fast, verbose=not quiet,
        control_type=cfg.control_type,
        drift_detach=cfg.drift_detach,
    )

    # (Re-)encode prompt for the sampler's cached embeddings
    if cfg.prompt != sampler.prompt_text:
        log(f"  re-encoding prompt: '{cfg.prompt[:50]}'")
        sampler.cached_prompt_embeds, sampler.cached_pooled_prompt_embeds = \
            sampler._encode_prompt_internal(cfg.prompt)
        sampler.prompt_text = cfg.prompt

    # Initial noise
    torch.manual_seed(cfg.seed)
    in_ch  = sampler.pipeline.transformer.config.in_channels // 4
    vscale = sampler.vae_scale_factor
    z0 = torch.randn(1, in_ch, cfg.height // vscale, cfg.width // vscale,
                     device=device, dtype=dtype)

    # Reward function
    r_image, _reward_obj = _build_reward_fn(
        sampler, cfg, reward_device, dtype, reward_model=reward_model,
    )

    def eval_r(z):
        with torch.no_grad():
            return r_image(z.to(device=device, dtype=dtype))

    # Uncontrolled baseline (always run, used as reference regardless of control_type)
    log("  uncontrolled rollout...")
    t0 = time.perf_counter()
    z_unc, traj_unc = _run_uncontrolled(sampler, z0, cfg, device, dtype)
    time_unc = time.perf_counter() - t0
    img_unc = _decode_pil(sampler, z_unc)
    r_unc_final = eval_r(z_unc).mean().item()
    _free(z_unc)

    # Controlled run
    z_ctrl = None
    traj_ctrl = None
    v_theta = None
    time_ctrl = 0.0
    r_ctrl_final = float("nan")

    if cfg.control_type != "none":
        v_theta, _, endpoint_map = make_flux_wrappers(
            sampler, cfg.prompt,
            guidance_scale=cfg.guidance_scale,
            n_mpc_steps=cfg.N,
            sample_mode=cfg.sample_mode,
        )

        log(f"  {cfg.control_type} run...")
        t0 = time.perf_counter()

        if cfg.control_type == "mpc":
            z_ctrl, traj_ctrl, _, _, _ = receding_horizon_flow_control(
                v_theta=v_theta,
                r=r_image,
                x0=z0,
                N=cfg.N,
                H=cfg.H,
                lam=cfg.lam,
                optimizer="adam",
                n_iters=cfg.n_inner,
                lr=cfg.lr_inner,
                warmstart=cfg.warmstart,
                drift_detach=cfg.drift_detach,
                constraint_on=cfg.constraint_on,
                endpoint_map=endpoint_map,
                verbose=not quiet,
                low_memory=low_memory,
                device=device,
            )
        elif cfg.control_type == "fmrg":
            z_ctrl, traj_ctrl, _ = fmrg_j_image(
                sampler, r_image, z0,
                N=cfg.N,
                n_opt=cfg.n_inner,
                lam=cfg.lam,
                guidance_scale=cfg.guidance_scale,
                sample_mode=cfg.sample_mode,
                return_traj=True,
                return_controls=True,
                verbose=not quiet,
                low_memory=low_memory,
            )

        time_ctrl = time.perf_counter() - t0
        r_ctrl_final = eval_r(z_ctrl).mean().item()
        img_ctrl = _decode_pil(sampler, z_ctrl)
        _free(z_ctrl)
    else:
        img_ctrl = img_unc      # placeholder for side-by-side

    _free(z0)

    # Reward trajectory curves
    r_unc_curve  = trajectory_rewards(traj_unc, r_image, device=device, dtype=dtype).float().numpy()
    r_ctrl_curve = (
        trajectory_rewards(traj_ctrl, r_image, device=device, dtype=dtype).float().numpy()
        if traj_ctrl is not None else r_unc_curve
    )
    t_grid = torch.linspace(0.0, 1.0, len(r_unc_curve)).numpy()
    _free(traj_unc)

    # Per-step control norms: u_t = x_{t+1} - x_t - dt·b_t(x_t)
    # Computed from the trajectory for both MPC and FMRG so scales match.
    ctrl_norms  = None
    ctrl_energy = float("nan")
    if traj_ctrl is not None and v_theta is not None:
        log("  computing control norms from trajectory...")
        ctrl_norms, ctrl_energy = _ctrl_norms_from_traj(
            traj_ctrl, v_theta, cfg.N, device, dtype,
        )
    if traj_ctrl is not None:
        _free(traj_ctrl)

    # Lagrangian objective J = control_energy / (2*lam) - r_ctrl, matching the
    # sign/scaling of the inner-loop loss in _horizon_objective (image_mpc.py).
    # Lower J is better; this is what makes MPC vs. FMRG comparable at a fixed lam.
    objective = float("nan")
    if cfg.control_type != "none":
        objective = ctrl_energy / (2.0 * cfg.lam) - r_ctrl_final

    # Free reward model weights after use
    if hasattr(_reward_obj, "cpu"):
        _reward_obj.cpu()
    _free(_reward_obj)

    return dict(
        prompt        = cfg.prompt,
        edit_prompt   = cfg.edit_prompt if cfg.edit_prompt else cfg.prompt,
        N             = cfg.N,
        H             = cfg.H,
        control_type  = cfg.control_type,
        lam           = cfg.lam,
        n_inner       = cfg.n_inner,
        lr_inner      = cfg.lr_inner,
        sample_mode   = cfg.sample_mode,
        warmstart     = cfg.warmstart,
        drift_detach  = cfg.drift_detach,
        constraint_on = cfg.constraint_on,
        r_unc         = r_unc_final,
        r_ctrl        = r_ctrl_final,
        delta_r       = r_ctrl_final - r_unc_final,
        ctrl_energy   = ctrl_energy,
        objective     = objective,
        time_unc_s    = time_unc,
        time_ctrl_s   = time_ctrl,
        # image objects (not CSV-serialisable, stripped before writing)
        _img_unc   = img_unc,
        _img_ctrl  = img_ctrl,
        _t_grid    = t_grid,
        _r_unc_c   = r_unc_curve,
        _r_ctrl_c  = r_ctrl_curve,
        _ctrl_norms = ctrl_norms,
    )


def _save_run(result: dict, out_dir: Path, run_idx: int) -> str:
    """Write side-by-side image + reward curve for one run. Returns cfg_tag
    (the filename prefix) so callers can record it for later image lookup."""
    cfg_tag = (
        f"{run_idx:03d}"
        f"__{result['control_type']}"
        f"__N{result['N']}"
        f"__H{result['H']}"
        f"__lam{result['lam']}"
        f"__{result['sample_mode']}"
        f"__{result['prompt'][:30].replace(' ', '_')}"
    )

    # Side-by-side images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(result["_img_unc"])
    axes[0].axis("off")
    axes[0].set_title(f"Uncontrolled  r={result['r_unc']:.4f}")
    axes[1].imshow(result["_img_ctrl"])
    axes[1].axis("off")
    ctrl_lbl = (
        f"{result['control_type']}  H={result['H']}  λ={result['lam']}"
        f"  r={result['r_ctrl']:.4f}"
    ) if result["control_type"] != "none" else "Uncontrolled (reference)"
    axes[1].set_title(ctrl_lbl)
    title = f"gen: {result['prompt'][:60]}"
    if result["edit_prompt"] != result["prompt"]:
        title += f"\nedit: {result['edit_prompt'][:60]}"
    plt.suptitle(title + f"\nN={result['N']}  {result['sample_mode']}", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / f"{cfg_tag}__images.png", dpi=100)
    plt.close(fig)

    # Reward curve + control norm (two panels when controls are available)
    has_ctrl = result["control_type"] != "none" and result["_ctrl_norms"] is not None
    n_panels = 2 if has_ctrl else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    ax_r = axes[0]
    ax_r.plot(result["_t_grid"], result["_r_unc_c"], "o-", label="Uncontrolled",
              linewidth=2, markersize=6)
    if result["control_type"] != "none":
        ax_r.plot(result["_t_grid"], result["_r_ctrl_c"], "s-",
                  label=result["control_type"], linewidth=2, markersize=6)
    ax_r.set_xlabel("Progress time $s$  (0 = noise, 1 = data)")
    ax_r.set_ylabel("Reward")
    ax_r.set_title(f"Reward — N={result['N']} H={result['H']} "
                   f"λ={result['lam']} {result['sample_mode']}")
    ax_r.legend(); ax_r.grid(True, alpha=0.3)

    if has_ctrl:
        ax_u = axes[1]
        norms = result["_ctrl_norms"]          # (N,) per outer step
        steps = range(1, len(norms) + 1)
        ax_u.bar(steps, norms, color="steelblue", alpha=0.8)
        ax_u.set_xlabel("Outer step $t$")
        ax_u.set_ylabel(r"$\|u_t\|$ (mean over batch)")
        energy_str = f"{result['ctrl_energy']:.4f}" if result['ctrl_energy'] == result['ctrl_energy'] else "n/a"
        ax_u.set_title(
            f"Control input — {result['control_type']}\n"
            r"$\sum_t\|u_t\|^2\,\Delta t$"
            f" = {energy_str}"
        )
        ax_u.grid(True, alpha=0.3, axis="y")

    title = f"gen: {result['prompt'][:60]}"
    if result["edit_prompt"] != result["prompt"]:
        title += f"   |   edit: {result['edit_prompt'][:60]}"
    plt.suptitle(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / f"{cfg_tag}__reward.png", dpi=100)
    plt.close(fig)

    result["_img_unc"].save(out_dir / f"{cfg_tag}__img_unc.png")
    if result["control_type"] != "none":
        result["_img_ctrl"].save(out_dir / f"{cfg_tag}__img_ctrl.png")

    return cfg_tag


CSV_KEYS = [
    "tag", "prompt", "edit_prompt", "N", "H", "control_type", "lam", "n_inner", "lr_inner",
    "sample_mode", "warmstart", "drift_detach", "constraint_on", "r_unc", "r_ctrl", "delta_r",
    "ctrl_energy", "objective", "time_unc_s", "time_ctrl_s",
]


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def _build_configs_from_args(args) -> List[RunConfig]:
    """Build a list of RunConfigs from CLI sweep arguments.

    Each sweep argument accepts one or more values.  The cartesian product of
    all multi-valued arguments is expanded into individual RunConfig objects,
    so ``--N 4 6 --H 2 3`` produces four configs (4×2, 4×3, 6×2, 6×3).

    If no sweep arguments are provided, returns an empty list and the caller
    falls back to SWEEP_CONFIGS (optionally filtered by --configs).
    """
    # Collect all sweep axes that were explicitly set
    axes = {}
    defaults = RunConfig()
    sweep_fields = [
        "prompt", "edit_prompt", "N", "H", "control_type", "lam", "n_inner",
        "lr_inner", "sample_mode", "guidance_scale",
        "height", "width", "warmstart", "drift_detach", "constraint_on",
    ]
    for f in sweep_fields:
        val = getattr(args, f"sweep_{f}", None)
        if val is not None:
            axes[f] = val

    if not axes:
        return []

    # Cartesian product of all axes
    keys = list(axes.keys())
    configs = []
    for combo in itertools.product(*[axes[k] for k in keys]):
        kw = {k: v for k, v in zip(keys, combo)}
        cfg = RunConfig(**{
            f: kw.get(f, getattr(defaults, f)) for f in sweep_fields
        })
        configs.append(cfg)
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Controlled vs uncontrolled FLUX sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sweep via CLI (cartesian product of all provided lists):
  python run_sweep.py \\
      --sweep_prompt "a sunset" "a cat" \\
      --sweep_N 4 6 \\
      --sweep_H 2 3 \\
      --sweep_control_type mpc fmrg \\
      --sweep_lam 50.0 200.0

  # separate generation prompt vs. editing/reward prompt:
  python run_sweep.py \\
      --sweep_prompt "a portrait of a man" \\
      --sweep_edit_prompt "a smiling face" \\
      --sweep_control_type mpc fmrg \\
      --sweep_lam 5.0 50.0 200.0

Use SWEEP_CONFIGS in the file when no --sweep_* flags are given.
Filter SWEEP_CONFIGS by index with --configs 0,2,4.
        """,
    )
    # ── infrastructure ───────────────────────────────────────────────────────
    # Absolute default (anchored to this script's own directory) so a relative
    # "results" never lands inside Shervin's original repo if this is invoked
    # from a different cwd (e.g. cwd = flow_matching to pick up its imports).
    # Timestamp-prefixed subfolder: keeps runs chronologically sortable and
    # avoids different sweeps colliding on the same run_idx-based image tags.
    default_out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results",
        time.strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--out_dir",   default=default_out_dir)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--hf_token",  default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--model_id",  default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--quantize",  default="int8",
                        choices=["none", "int8", "int4", "nf4"],
                        help="Transformer quantization (default: int8 ~6 GB; "
                             "'none' needs >24 GB VRAM)")
    parser.add_argument("--fast",      type=lambda x: x.lower() != "false",
                        default=False, metavar="BOOL",
                        help="Disable gradient checkpointing + VAE tiling "
                             "(true/false, default: false)")
    parser.add_argument("--reward",      default="clip",
                        choices=REWARD_CHOICES,
                        help="Reward model for control guidance and evaluation "
                             "(default: clip). Options: clip, hpsv2, pickscore, imagereward")
    parser.add_argument("--flux_device", default="cuda:0",
                        help="Device for FLUX transformer + VAE (default: cuda:0)")
    parser.add_argument("--reward_device", default="auto",
                        help="Device for reward model. "
                             "'auto' uses a second GPU if available, else CPU.")
    parser.add_argument("--quiet",     action="store_true")
    parser.add_argument("--configs",   default=None,
                        help="Comma-separated indices into SWEEP_CONFIGS "
                             "(ignored when any --sweep_* flag is given)")

    # ── sweep axes (each accepts 1+ values; cartesian product is taken) ──────
    d = RunConfig()   # defaults
    parser.add_argument("--sweep_prompt",        nargs="+", metavar="PROMPT")
    parser.add_argument("--sweep_edit_prompt",   nargs="+", metavar="EDIT_PROMPT",
                        help="Reward-model target prompt(s); combined via cartesian "
                             "product with --sweep_prompt. Omit to reward against "
                             "the generation prompt itself.")
    parser.add_argument("--sweep_N",             nargs="+", type=int,   metavar="N")
    parser.add_argument("--sweep_H",             nargs="+", type=int,   metavar="H")
    parser.add_argument("--sweep_control_type",  nargs="+",
                        choices=["mpc", "fmrg", "none"],   metavar="TYPE")
    parser.add_argument("--sweep_lam",           nargs="+", type=float, metavar="LAM")
    parser.add_argument("--sweep_n_inner",       nargs="+", type=int,   metavar="N_INNER")
    parser.add_argument("--sweep_lr_inner", "--lr_inner",
                        nargs="+", type=float, metavar="LR")
    parser.add_argument("--sweep_sample_mode",   nargs="+",
                        choices=["flow_map1", "flow_map2", "flow_matching"], metavar="MODE")
    parser.add_argument("--sweep_guidance_scale",nargs="+", type=float, metavar="CFG")
    parser.add_argument("--sweep_height",        nargs="+", type=int,   metavar="H_PX")
    parser.add_argument("--sweep_width",         nargs="+", type=int,   metavar="W_PX")
    parser.add_argument("--sweep_warmstart",     nargs="+", type=lambda x: x.lower() != "false",
                        metavar="BOOL")
    parser.add_argument("--sweep_drift_detach",  nargs="+", type=lambda x: x.lower() != "false",
                        metavar="BOOL")
    parser.add_argument("--sweep_constraint_on", nargs="+",
                        choices=["horizon", "endpoint"], metavar="MODE",
                        help="MPC only. No-op until a constraint fn h(x) is wired "
                             "up (currently always None) -- recorded for provenance.")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # HF auth
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)

    # Build config list: CLI sweep overrides SWEEP_CONFIGS
    configs = _build_configs_from_args(args)
    if configs:
        print(f"CLI sweep: {len(configs)} config(s) from cartesian product of --sweep_* args")
    else:
        configs = list(SWEEP_CONFIGS)
        if args.configs:
            idxs = [int(i) for i in args.configs.split(",")]
            configs = [SWEEP_CONFIGS[i] for i in idxs]
            print(f"Using SWEEP_CONFIGS indices {idxs}: {len(configs)} config(s)")
        else:
            print(f"Using all {len(configs)} SWEEP_CONFIGS entries")

    # Apply seed and out_dir to all configs
    for cfg in configs:
        cfg.seed = args.seed
        cfg.out_dir = args.out_dir

    flux_device, reward_device = _resolve_devices(args.flux_device, args.reward_device)
    if not args.quiet:
        print(f"Devices: FLUX={flux_device}  reward({args.reward})={reward_device}")

    # Load sampler once
    lora_path = args.lora_path or os.path.join(
        FMRG_DIR, "checkpoints", "flux-flowmap-lora"
    )
    print(f"Loading FluxFlowMapSampler (quantize_mode={args.quantize})...")
    # Use the first prompt as the initial cached embedding; it will be
    # re-encoded in _run_one when a different prompt is encountered.
    first_prompt = configs[0].prompt if configs else "a photo"
    sampler = FluxFlowMapSampler(
        model_id=args.model_id,
        lora_path=lora_path,
        device=str(flux_device),
        dtype=torch.bfloat16,
        prompt=first_prompt,
        quantize_mode=args.quantize,
    )

    device = next(sampler.pipeline.transformer.parameters()).device
    dtype  = next(sampler.pipeline.transformer.parameters()).dtype
    low_memory = not args.fast

    # Run sweep — CSV is written incrementally (one row per run, flushed to
    # disk immediately) so a crash/OOM/kill partway through a long sweep loses
    # at most the in-flight run, not every result computed so far.
    csv_path = out_dir / "summary.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_KEYS)
    writer.writeheader()
    csv_file.flush()

    n_rows = 0
    try:
        for run_idx, cfg in enumerate(configs):
            print(f"\n{'='*60}")
            print(f"Run {run_idx+1}/{len(configs)}: "
                  f"control={cfg.control_type} N={cfg.N} H={cfg.H} "
                  f"lam={cfg.lam} mode={cfg.sample_mode}")
            print(f"  prompt: {cfg.prompt[:60]}")
            if cfg.edit_prompt:
                print(f"  edit_prompt: {cfg.edit_prompt[:60]}")

            try:
                result = _run_one(
                    sampler, cfg, device, reward_device, dtype, quiet=args.quiet,
                    fast=args.fast, low_memory=low_memory, reward_model=args.reward,
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()
                continue

            result["tag"] = _save_run(result, out_dir, run_idx)

            row = {k: result[k] for k in CSV_KEYS}
            writer.writerow(row)
            csv_file.flush()
            os.fsync(csv_file.fileno())
            n_rows += 1

            if not args.quiet:
                energy = result["ctrl_energy"]
                energy_str = f"{energy:.4f}" if energy == energy else "n/a"
                obj = result["objective"]
                obj_str = f"{obj:.4f}" if obj == obj else "n/a"
                print(f"  r_unc={result['r_unc']:.4f}  r_ctrl={result['r_ctrl']:.4f}"
                      f"  Δr={result['delta_r']:+.4f}"
                      f"  Σ‖u‖²Δt={energy_str}"
                      f"  J={obj_str}"
                      f"  t_ctrl={result['time_ctrl_s']:.1f}s")
    finally:
        csv_file.close()

    print(f"\n{n_rows}/{len(configs)} runs completed. Results saved to {out_dir}/")
    print(f"CSV summary: {csv_path}")


if __name__ == "__main__":
    main()
