#!/usr/bin/env python3
"""
flux_inversion.py -- invert real images into FLUX's noise space, and test it on PIE-Bench++.

Everything else in fmocc_core starts from a random z0 and a generation prompt.
Editing a REAL image needs the reverse map first: image -> latent -> the noise
z1 that the sampler would turn back into that latent. This file provides it.

CONVENTION (same as FluxFlowMapSampler and run_sweep): t = 1 is noise, t = 0 is
data, and one sampler step is

    z_{t_next} = z_{t_cur} - (t_cur - t_next) * u(z_{t_cur}, t_cur, t_next)

where u is the flow-map LoRA's average velocity over [t_next, t_cur]. On the
diagonal (t_next = t_cur) it is the instantaneous flow-matching velocity.

TWO INVERSION METHODS, on the same grid ts = linspace(1, 0, N+1) the forward
sampler uses:

  euler        z_{t+dt} = z_t + dt * u(z_t, t, t).  The textbook ODE inversion
               with the instantaneous velocity. Its exact counterpart is the
               Euler sampler (run_sweep's sample_mode "flow_matching"), so that
               is what `reconstruct` uses. The mismatch -- velocity taken at the
               lower-t end going up, the higher-t end coming down -- is the usual
               Euler inversion error and shrinks with N.

  fixed_point  The exact inverse of the flow_map2 step every MPC / HardFlow run
               uses. Going up one step means solving
                   z_c = z_n + dt * u(z_c, t_c, t_n)
               for z_c, which is done by fixed-point iteration from the Euler
               guess. One flow_map2 step down from the result reproduces z_n up
               to the fixed-point residual, which is reported. The flow map is
               only trained in the denoising direction (t_next < t_cur), and this
               never asks it to go the other way -- it only ever evaluates the
               forward map, at an unknown start point.

The VAE encoder uses the posterior MEAN, not a sample (FluxFlowMapSampler.encode
samples), so inversion is deterministic. Decode(encode(x)) is the ceiling no
inversion can beat; the test reports it so VAE loss and inversion loss stay
separate.

Usage (from fmocc_core/; pin one GPU, cuda:0 in-process):

  CUDA_VISIBLE_DEVICES=1 python flux_inversion.py \
      --pie_root /home/shervink/datasets/PIE_Bench_pp \
      --examples 1:0 3:0 8:0 9:0 --N 28 --methods fixed_point --n_fp 10 \
      --guidance 3.5 --out_dir results/inversion_test
"""
import argparse
import csv
import glob
import io
import itertools
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

FLOW_MATCHING_DIR = os.environ.get("FLOW_MATCHING_DIR", "/home/shervink/flow_matching")
FMRG_DIR = os.environ.get("FMRG_DIR", os.path.join(FLOW_MATCHING_DIR, "fmrg"))
# APPENDED, not prepended: the repo root has its own (older) run_sweep.py, and
# prepending it shadowed fmocc_core/run_sweep.py for any script importing both.
for _p in (FMRG_DIR, FLOW_MATCHING_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

RECON_MODE = {"euler": "flow_matching", "fixed_point": "flow_map2"}


# ── image <-> latent ────────────────────────────────────────────────────────

def pil_to_tensor(img, size: int, device, dtype) -> torch.Tensor:
    """PIL -> (1, 3, size, size) in [-1, 1]."""
    from PIL import Image
    img = img.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    return (x * 2.0 - 1.0).to(device=device, dtype=dtype)


@torch.no_grad()
def vae_encode(sampler, x: torch.Tensor) -> torch.Tensor:
    """Posterior mean, shifted and scaled exactly as FluxFlowMapSampler.encode."""
    lat = sampler.vae.encode(x).latent_dist.mode()
    return (lat - sampler.vae.config.shift_factor) * sampler.vae.config.scaling_factor


@torch.no_grad()
def vae_decode(sampler, z: torch.Tensor) -> torch.Tensor:
    return sampler.decode(z.to(sampler.dtype)).float()


# ── the ODE in both directions ──────────────────────────────────────────────

def _u(sampler, z, t_cur, t_next, emb, guidance):
    """Flow-map velocity in fp32. z is kept in fp32 between steps so bf16
    rounding does not accumulate over the trajectory; only the model call is bf16."""
    pe, pp = emb
    return sampler.predict_vector(z.to(sampler.dtype), t_cur, prompt_embeds=pe,
                                  pooled_prompt_embeds=pp, guidance_scale=guidance,
                                  t_next=t_next).float()


@torch.no_grad()
def invert(sampler, z_data: torch.Tensor, emb, N: int, guidance: float,
           method: str = "fixed_point", n_fp: int = 3) -> Tuple[torch.Tensor, Dict]:
    """Data latent (t=0) -> noise (t=1). Returns (z_noise, info)."""
    ts = torch.linspace(1.0, 0.0, N + 1).tolist()
    z = z_data.float().clone()
    residuals = []                      # fixed_point: relative change at the last iteration, per step
    for j in range(N - 1, -1, -1):      # walk the grid upward: ts[N]=0 -> ts[0]=1
        t_n, t_c = ts[j + 1], ts[j]
        dt = t_c - t_n
        z_c = z + dt * _u(sampler, z, t_n, t_n, emb, guidance)      # Euler guess (diagonal)
        if method == "fixed_point":
            for _ in range(n_fp):
                z_new = z + dt * _u(sampler, z_c, t_c, t_n, emb, guidance)
                res = (z_new - z_c).norm().item() / max(z_new.norm().item(), 1e-12)
                z_c = z_new
            residuals.append(res)
        elif method != "euler":
            raise ValueError(f"unknown method {method!r}")
        z = z_c
    info = {"noise_mean": z.mean().item(), "noise_std": z.std().item()}
    if residuals:
        info["fp_residual_max"] = max(residuals)
        info["fp_residual_last_step"] = residuals[-1]
    return z, info


@torch.no_grad()
def generate(sampler, z_noise: torch.Tensor, emb, N: int, guidance: float,
             mode: str = "flow_map2") -> torch.Tensor:
    """Noise (t=1) -> data latent (t=0), the same loop as run_sweep._run_uncontrolled."""
    ts = torch.linspace(1.0, 0.0, N + 1).tolist()
    z = z_noise.float().clone()
    for j in range(N):
        t_c, t_n = ts[j], ts[j + 1]
        t_target = t_n if mode == "flow_map2" else t_c       # "flow_matching": diagonal
        z = z - (t_c - t_n) * _u(sampler, z, t_c, t_target, emb, guidance)
    return z


# ── PIE-Bench++ ─────────────────────────────────────────────────────────────

def load_pie(root: str, spec: str) -> Dict:
    """spec "<split digit>:<row>", e.g. "1:0" = first row of 1_change_object_80."""
    import pyarrow.parquet as pq
    from PIL import Image
    split_digit, row = spec.split(":")
    files = glob.glob(os.path.join(root, f"{split_digit}_*", "*.parquet"))
    if len(files) != 1:
        raise SystemExit(f"--examples {spec}: expected one parquet for split {split_digit}, found {files}")
    r = pq.read_table(files[0]).slice(int(row), 1).to_pylist()[0]
    return {
        "spec": spec, "split": os.path.basename(os.path.dirname(files[0])), "id": r["id"],
        "image": Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"),
        "source_prompt": r["source_prompt"],
        "target_prompt": r["target_prompt"].replace("[", "").replace("]", ""),
        "target_prompt_marked": r["target_prompt"],
        "mask_rle": r["mask"],          # PIE-Bench RLE over the 512x512 image, row-major
    }


# ── metrics ─────────────────────────────────────────────────────────────────

class Metrics:
    def __init__(self, device):
        import lpips
        self.lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, ref: torch.Tensor, img: torch.Tensor) -> Dict[str, float]:
        """ref, img: (1, 3, H, W) in [-1, 1]."""
        from skimage.metrics import structural_similarity
        a = ((ref.float() + 1) / 2).clamp(0, 1)
        b = ((img.float() + 1) / 2).clamp(0, 1)
        mse = (a - b).pow(2).mean().item()
        an = a[0].permute(1, 2, 0).cpu().numpy(); bn = b[0].permute(1, 2, 0).cpu().numpy()
        return {"psnr": 10 * np.log10(1.0 / max(mse, 1e-12)),
                "ssim": float(structural_similarity(an, bn, channel_axis=2, data_range=1.0)),
                "lpips": self.lp(ref.float(), img.float()).item()}


def to_pil(x: torch.Tensor):
    from PIL import Image
    a = ((x[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(a)


# ── figure ──────────────────────────────────────────────────────────────────

def draw_example(ex: Dict, vae_img, vae_m, rows: List[Dict], out_png: str) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(11.2, 2.95 * n + 0.9), squeeze=False)
    for i, r in enumerate(rows):
        cells = [(ex["image"], "original" if i == 0 else ""),
                 (vae_img, (f"VAE only (ceiling)\nPSNR {vae_m['psnr']:.1f}  LPIPS {vae_m['lpips']:.3f}")
                  if i == 0 else ""),
                 (r["recon_img"], f"reconstruction\nPSNR {r['psnr']:.1f}  LPIPS {r['lpips']:.3f}"),
                 (r["edit_img"], "edit: regenerate with target prompt")]
        for j, (im, title) in enumerate(cells):
            ax = axes[i, j]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            if title: ax.set_title(title, fontsize=8.5)
            if i > 0 and j < 2: ax.set_visible(False)
        k = f"  K={r['n_fp']}" if r["n_fp"] else ""
        axes[i, 2].set_ylabel(f"{r['method']}\nN={r['N']}  g={r['guidance']:g}{k}", fontsize=9)
    fig.suptitle(f"{ex['split']}  ·  id {ex['id']}\nsource: {ex['source_prompt']}\n"
                 f"target: {ex['target_prompt_marked']}", fontsize=9.5, y=1.0)
    fig.tight_layout(); fig.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    p.add_argument("--pie_root", default="/home/shervink/datasets/PIE_Bench_pp")
    p.add_argument("--examples", nargs="+", default=["1:0", "3:0", "8:0", "9:0"],
                   help='"<split digit>:<row>" per example, e.g. 1:0 = first row of 1_change_object_80')
    p.add_argument("--N", nargs="+", type=int, default=[28])
    p.add_argument("--methods", nargs="+", default=["fixed_point"], choices=list(RECON_MODE))
    p.add_argument("--guidance", nargs="+", type=float, default=[3.5])
    p.add_argument("--n_fp", nargs="+", type=int, default=[10],
                   help="fixed-point iterations per step (fixed_point only; several = one run each)")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--no_edit", action="store_true", help="skip the regenerate-with-target-prompt column")
    p.add_argument("--quantize", default="int8")
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--out_dir", default="results/inversion_test")
    a = p.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    exs = [load_pie(a.pie_root, s) for s in a.examples]
    prompts = list(dict.fromkeys(q for e in exs for q in (e["source_prompt"], e["target_prompt"])))

    from fluxfm_sampler_reward import FluxFlowMapSampler
    dev = "cuda:0"
    # Text encoders are freed after init, so every prompt must be embedded now.
    sampler = FluxFlowMapSampler(model_id=a.model_id,
                                 lora_path=os.path.join(FMRG_DIR, "checkpoints", "flux-flowmap-lora"),
                                 device=dev, dtype=torch.bfloat16, prompt=prompts[0],
                                 additional_prompts=prompts, quantize_mode=a.quantize)
    emb = {q: tuple(t.to(dev) for t in sampler.prompt_cache[q]) for q in prompts}
    metrics = Metrics(dev)

    # n_fp only matters for fixed_point; euler gets a single entry with K=0.
    configs = [(m, N, g, K) for m, N, g in itertools.product(a.methods, a.N, a.guidance)
               for K in (a.n_fp if m == "fixed_point" else [0])]
    rows_all = []
    for ex in exs:
        print(f"\n=== {ex['split']} id {ex['id']} ===\n  source: {ex['source_prompt']}\n"
              f"  target: {ex['target_prompt']}", flush=True)
        x = pil_to_tensor(ex["image"], a.size, dev, sampler.dtype)
        z0 = vae_encode(sampler, x).float()
        vae_x = vae_decode(sampler, z0)
        vae_m = metrics(x, vae_x)
        print(f"  VAE only      PSNR {vae_m['psnr']:6.2f}  SSIM {vae_m['ssim']:.3f}  LPIPS {vae_m['lpips']:.4f}")
        rows = []
        for method, N, g, K in configs:
            t0 = time.time()
            zT, info = invert(sampler, z0, emb[ex["source_prompt"]], N, g, method, K)
            t_inv = time.time() - t0
            mode = RECON_MODE[method]
            z_rec = generate(sampler, zT, emb[ex["source_prompt"]], N, g, mode)
            x_rec = vae_decode(sampler, z_rec)
            m = metrics(x, x_rec)
            lat_err = ((z_rec - z0).norm() / z0.norm()).item()
            row = {"split": ex["split"], "id": ex["id"], "method": method, "N": N, "guidance": g, "n_fp": K,
                   "recon_mode": mode, **m, "latent_rel_err": lat_err,
                   "vae_psnr": vae_m["psnr"], "vae_ssim": vae_m["ssim"], "vae_lpips": vae_m["lpips"],
                   "t_invert_s": t_inv, **info}
            row["recon_img"] = to_pil(x_rec)
            row["edit_img"] = (to_pil(vae_decode(sampler, generate(sampler, zT, emb[ex["target_prompt"]], N, g, mode)))
                               if not a.no_edit else row["recon_img"])
            fp = f"  fp_res {info['fp_residual_max']:.1e}" if "fp_residual_max" in info else ""
            print(f"  {method:<11} N={N:<3} g={g:<4g} K={K if K else '-':<3} PSNR {m['psnr']:6.2f}  SSIM {m['ssim']:.3f}  "
                  f"LPIPS {m['lpips']:.4f}  z-err {lat_err:.3f}  noise std {info['noise_std']:.3f}"
                  f"{fp}  inv {t_inv:.0f}s", flush=True)
            rows.append(row)
        draw_example(ex, to_pil(vae_x), vae_m, rows, os.path.join(a.out_dir, f"{ex['split']}__{ex['id']}.png"))
        rows_all += rows

    keys = [k for k in rows_all[0] if not k.endswith("_img")]
    with open(os.path.join(a.out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows_all:
            w.writerow({k: r.get(k, "") for k in keys})

    print(f"\n=== mean over {len(exs)} images ===")
    print(f"{'method':<12}{'N':>4}{'g':>6}{'K':>4}{'PSNR':>8}{'SSIM':>7}{'LPIPS':>8}{'z-err':>8}{'noise std':>10}")
    for method, N, g, K in configs:
        sel = [r for r in rows_all if (r["method"], r["N"], r["guidance"], r["n_fp"]) == (method, N, g, K)]
        mean = lambda k: float(np.mean([r[k] for r in sel]))
        print(f"{method:<12}{N:>4}{g:>6g}{(K if K else '-'):>4}{mean('psnr'):8.2f}{mean('ssim'):7.3f}{mean('lpips'):8.4f}"
              f"{mean('latent_rel_err'):8.3f}{mean('noise_std'):10.3f}")
    vm = lambda k: float(np.mean([r[k] for r in rows_all]))
    print(f"{'VAE ceiling':<26}{vm('vae_psnr'):8.2f}{vm('vae_ssim'):7.3f}{vm('vae_lpips'):8.4f}")
    print(f"\nfigures + summary.csv in {os.path.abspath(a.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
