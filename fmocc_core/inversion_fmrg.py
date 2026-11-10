#!/usr/bin/env python3
"""
inversion_fmrg.py -- reconstruct real images with FMRG pulling toward the original.

Unconstrained test of "reference-guided reconstruction": start the sampler from
(a) the fixed-point inversion of the image and (b) a fresh Gaussian, and steer it
with FMRG toward the original's latent using

    r(z) = -1/2 * || z_hat - z_ref ||^2        (sum over latent elements)

where z_hat is FMRG's one-jump endpoint prediction and z_ref is the VAE posterior
mean of the original. With the sum (not mean) and an endpoint Jacobian near the
identity, FMRG's pull shrinks the gap roughly like exp(-lam), so lam is readable.

The random-noise start is the control: with enough lam ANY noise reconstructs, so
reconstruction quality alone says nothing about inversion. What inversion should
buy is reaching the original for less control energy.

lam = 0 is a reference column, not an FMRG run: the plain flow_map2 sampler from
the same start (fp32 master copy, as in flux_inversion.generate).

Usage (from fmocc_core/, one GPU pinned, cuda:0 in-process):
  CUDA_VISIBLE_DEVICES=1 python inversion_fmrg.py --out_dir results/inversion_fmrg
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flux_inversion as fi            # also puts FMRG_DIR / FLOW_MATCHING_DIR on sys.path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pie_root", default="/home/shervink/datasets/PIE_Bench_pp")
    p.add_argument("--examples", nargs="+", default=["1:0", "8:0", "1:43", "4:1"])
    p.add_argument("--lams", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0])
    p.add_argument("--starts", nargs="+", default=["inverted", "random"], choices=["inverted", "random"])
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--n_fp", type=int, default=5, help="fixed-point iterations for the inversion")
    p.add_argument("--n_opt", type=int, default=3, help="FMRG guidance steps per denoising step")
    p.add_argument("--controller", default="fmrg", choices=["fmrg", "mpc"],
                   help="fmrg: fmrg_j_image (lam = gain). mpc: receding_horizon_flow_control, i.e.\n"
                        "ConicFlow (lam prices energy: sum ||u||^2/(2 lam dt) - r per horizon)")
    p.add_argument("--size", type=int, default=512, help="images are resized to size x size")
    p.add_argument("--H", type=int, default=2, help="mpc horizon")
    p.add_argument("--n_inner", type=int, default=24, help="mpc Adam steps per outer step")
    p.add_argument("--lr_inner", type=float, default=1e-2, help="mpc Adam lr")
    p.add_argument("--compare_dir", default=None,
                   help="earlier run dir (results.csv + img/) drawn as a row under each image")
    p.add_argument("--compare_label", default="comparison")
    p.add_argument("--compare_energy_scale", type=float, default=1.0,
                   help="multiply the comparison run's energies by this (1/N for runs written\n"
                        "before the switch to run_sweep's sum ||u||^2 * dt convention)")
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--noise_seed", type=int, default=0)
    p.add_argument("--out_dir", default="results/inversion_fmrg")
    a = p.parse_args(argv)
    os.makedirs(os.path.join(a.out_dir, "img"), exist_ok=True)

    from fluxfm_sampler_reward import FluxFlowMapSampler
    from image_mpc import fmrg_j_image, make_flux_wrappers
    from image_mpc_edited_sung import receding_horizon_flow_control

    exs = [fi.load_pie(a.pie_root, s) for s in a.examples]
    prompts = list(dict.fromkeys(e["source_prompt"] for e in exs))
    dev = "cuda:0"
    sampler = FluxFlowMapSampler(model_id="black-forest-labs/FLUX.1-dev",
                                 lora_path=os.path.join(fi.FMRG_DIR, "checkpoints", "flux-flowmap-lora"),
                                 device=dev, dtype=torch.bfloat16, prompt=prompts[0],
                                 additional_prompts=prompts, quantize_mode="int8")
    metrics = fi.Metrics(dev)

    csv_path = os.path.join(a.out_dir, "results.csv")
    fields = ["spec", "id", "start", "lam", "psnr", "ssim", "lpips", "latent_rel_err",
              "energy", "diverged", "time_s", "img"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    def record(row):
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"  {row['start']:<9} lam={row['lam']:<6g} PSNR {row['psnr']:6.2f}  LPIPS {row['lpips']:.4f}  "
              f"z-err {row['latent_rel_err']:.3f}  E {row['energy']:10.2f}  "
              f"{'DIVERGED ' if row['diverged'] else ''}{row['time_s']:.0f}s", flush=True)

    for ex in exs:
        P = ex["source_prompt"]
        print(f"\n=== {ex['spec']}  {ex['split']} id {ex['id']}\n    prompt: {P}", flush=True)
        sampler.set_prompt_embeddings(P)                      # fmrg_j_image reads the cached embeds
        emb = (sampler.cached_prompt_embeds, sampler.cached_pooled_prompt_embeds)
        x = fi.pil_to_tensor(ex["image"], a.size, dev, sampler.dtype)
        z_ref = fi.vae_encode(sampler, x).float()
        tag = f"{ex['spec'].replace(':', '-')}_{ex['id']}"
        ex["image"].resize((a.size, a.size)).save(os.path.join(a.out_dir, "img", f"{tag}__original.png"))
        v_theta, _, endpoint_map = make_flux_wrappers(
            sampler, P, guidance_scale=a.guidance, n_mpc_steps=a.N,
            sample_mode="flow_map2", endpoint_surrogate="jump")

        t0 = time.time()
        z_inv, info = fi.invert(sampler, z_ref, emb, a.N, a.guidance, "fixed_point", a.n_fp)
        print(f"  inversion: {time.time()-t0:.0f}s  noise std {info['noise_std']:.3f}  "
              f"fp_res max {info['fp_residual_max']:.2e}", flush=True)
        g = torch.Generator(device=dev).manual_seed(a.noise_seed)
        starts = {"inverted": z_inv,
                  "random": torch.randn(z_ref.shape, generator=g, device=dev, dtype=torch.float32)}

        def r(z):
            return -0.5 * (z.float() - z_ref).pow(2).flatten(1).sum(1)

        for start in a.starts:
            z0 = starts[start]
            for lam in [0.0] + list(a.lams):
                t0 = time.time()
                if lam == 0.0:
                    zN = fi.generate(sampler, z0, emb, a.N, a.guidance, "flow_map2")
                    energy = 0.0
                elif a.controller == "fmrg":
                    zN, _traj, ctrls = fmrg_j_image(
                        sampler, r, z0.to(sampler.dtype), N=a.N, n_opt=a.n_opt, lam=lam,
                        guidance_scale=a.guidance, sample_mode="flow_map2",
                        return_traj=True, return_controls=True,
                        verbose=False, show_progress=False, low_memory=True)
                    # FMRG's returned control IS the residual x_{k+1} - x_k - dt*v(x_k).
                    energy = float(ctrls.float().pow(2).sum()) / a.N
                    del _traj, ctrls
                else:
                    out = receding_horizon_flow_control(
                        v_theta=v_theta, r=r, x0=z0.to(sampler.dtype), N=a.N, H=a.H, lam=lam,
                        optimizer="adam", n_iters=a.n_inner, lr=a.lr_inner, warmstart=True,
                        drift_detach=True, h=None, constraint_on="endpoint",
                        endpoint_map=endpoint_map, verbose=False, show_progress=False,
                        low_memory=True, device=torch.device(dev), return_dual_h=True)
                    zN, traj = out[0], out[1]
                    energy = traj_energy(traj, v_theta, a.N, dev, sampler.dtype)
                    del out, traj
                zN = zN.float()
                diverged = not bool(torch.isfinite(zN).all())
                if diverged:
                    zN = torch.nan_to_num(zN, nan=0.0, posinf=0.0, neginf=0.0)
                xN = fi.vae_decode(sampler, zN)
                m = metrics(x, xN)
                img_name = f"{tag}__{start}__lam{lam:g}.png"
                fi.to_pil(xN).save(os.path.join(a.out_dir, "img", img_name))
                record({"spec": ex["spec"], "id": ex["id"], "start": start, "lam": lam, **m,
                        "latent_rel_err": ((zN - z_ref).norm() / z_ref.norm()).item(),
                        "energy": energy, "diverged": diverged, "time_s": time.time() - t0,
                        "img": img_name})
                torch.cuda.empty_cache()

        out = draw_grid(a, [ex], csv_path, f"grid_{tag}.png")
        print(f"### IMAGE DONE {ex['spec']} {out}", flush=True)

    out = draw_grid(a, exs, csv_path, "grid.png")
    print(f"### ALL DONE {out}", flush=True)
    return 0


@torch.no_grad()
def traj_energy(traj, v_theta, N, dev, dtype) -> float:
    """run_sweep._ctrl_norms_from_traj's convention: u_k = x_{k+1} - x_k - dt*b(x_k),
    energy = sum_k ||u_k||^2 * dt, with b the uncontrolled MPC-convention drift."""
    dt = 1.0 / N
    s = torch.linspace(0.0, 1.0, N + 1)
    e = 0.0
    for k in range(N):
        x_k = traj[k].to(device=dev, dtype=dtype)
        u = traj[k + 1].to(device=dev, dtype=dtype).float() - x_k.float() - dt * v_theta(s[k].to(dev), x_k).float()
        e += float(u.flatten(1).norm(dim=1).pow(2).mean()) * dt
    return e


def draw_grid(a, exs, csv_path, out_name="grid.png"):
    """One row per (image, start) from this run; with --compare_dir, a row from
    that earlier run directly underneath each, same lambda columns."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    main_rows = list(csv.DictReader(open(csv_path)))
    cmp_rows = []
    if a.compare_dir:
        cp = os.path.join(a.compare_dir, "results.csv")
        cmp_rows = list(csv.DictReader(open(cp))) if os.path.isfile(cp) else []
    lams = [0.0] + list(a.lams)
    R = []
    for ex in exs:
        for s in a.starts:
            R.append((ex, s, "main"))
            if a.compare_dir:
                R.append((ex, s, "cmp"))
    fig, axes = plt.subplots(len(R), len(lams) + 1, figsize=(2.35 * (len(lams) + 1), 2.6 * len(R)),
                             squeeze=False)
    for i, (ex, start, src) in enumerate(R):
        tag = f"{ex['spec'].replace(':', '-')}_{ex['id']}"
        rows = main_rows if src == "main" else cmp_rows
        img_dir = os.path.join(a.out_dir if src == "main" else a.compare_dir, "img")
        e_scale = 1.0 if src == "main" else a.compare_energy_scale
        ax = axes[i, 0]; ax.set_xticks([]); ax.set_yticks([])
        if src == "main":
            ax.imshow(Image.open(os.path.join(a.out_dir, "img", f"{tag}__original.png")))
            ax.set_ylabel(f"{ex['spec']}  ·  {a.controller.upper()}", fontsize=9, fontweight="bold")
            if i == 0: ax.set_title("original", fontsize=9.5)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{ex['spec']}  ·  {a.compare_label}", ha="center", va="center",
                    fontsize=9, fontweight="bold", wrap=True, transform=ax.transAxes)
        for j, lam in enumerate(lams, start=1):
            ax = axes[i, j]; ax.set_xticks([]); ax.set_yticks([])
            hit = [r for r in rows if r["spec"] == ex["spec"] and r["start"] == start
                   and abs(float(r["lam"]) - lam) < 1e-9]
            if i == 0: ax.set_title("λ = 0  (no control)" if lam == 0 else f"λ = {lam:g}", fontsize=9.5)
            if not hit:
                ax.text(.5, .5, "n/a", ha="center", va="center", color="0.5"); continue
            r = hit[0]
            ax.imshow(Image.open(os.path.join(img_dir, r["img"])))
            # The earlier FMRG run saved its finite-but-exploded lam=100 rows as not
            # diverged; flag anything whose latent error is far past a reconstruction.
            div = r["diverged"] == "True" or float(r["latent_rel_err"]) > 10
            ax.set_xlabel(f"PSNR {float(r['psnr']):.1f}  LPIPS {float(r['lpips']):.2f}\n"
                          f"E = {float(r['energy']) * e_scale:.3g}{' DIVERGED' if div else ''}",
                          fontsize=7.8, color="#c62828" if div else "black")
    knobs = (f"n_opt={a.n_opt}" if a.controller == "fmrg"
             else f"H={a.H}, n_inner={a.n_inner}, lr={a.lr_inner:g}")
    cmp_txt = f"\ncomparison rows: {a.compare_label}" if a.compare_dir else ""
    fig.suptitle(f"{a.controller.upper()} toward the original:  r = -½‖ẑ − z_ref‖²  ·  "
                 f"{a.size}px, N={a.N}, {knobs}, g={a.guidance:g}  ·  "
                 f"inverted start = fixed-point inversion (K={a.n_fp})\n"
                 f"E = control energy Σ‖u‖²·dt{cmp_txt}", fontsize=10.5, y=1.0)
    fig.tight_layout()
    out = os.path.abspath(os.path.join(a.out_dir, out_name))
    fig.savefig(out, dpi=95, bbox_inches="tight")
    plt.close(fig)
    return out

if __name__ == "__main__":
    sys.exit(main())
