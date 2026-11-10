#!/usr/bin/env python3
"""
pie_edit.py -- edit real PIE-Bench images with ConicFlow (constrained MPC).

Formulation, per image:
  1. z_ref = VAE posterior mean of the original; z1 = fixed-point inversion of
     z_ref with the SOURCE prompt, so the uncontrolled source-prompt flow from z1
     reconstructs the original.
  2. From z1, run MPC on the source-prompt dynamics:
         minimize   sum ||u||^2 / (2 lam dt)  -  r(z_hat)
         r(z_hat) = -1/2 || z_hat - z_ref ||^2              (stay close to the original)
         subject to pickscore(decode(z_hat), target prompt) >= tau
     The edit enters ONLY through the constraint; the reward pulls toward the original.
  3. tau = s0 + (m - 1) * |s0|, with s0 = pickscore(decode(z_ref), target prompt).

Reference column: the same MPC with the reward only (no constraint) at the same lam,
reused from --ref_dir (inversion_fmrg.py --controller mpc output) when it has the
image, computed here otherwise.

Usage (from fmocc_core/, one GPU pinned, cuda:0 in-process):
  CUDA_VISIBLE_DEVICES=1 python pie_edit.py --out_dir results/pie_edit
"""
import argparse
import csv
import gc
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flux_inversion as fi            # also puts FMRG_DIR / FLOW_MATCHING_DIR on sys.path


def outside_mask(rle: str, size: int):
    """Boolean (size, size) map of pixels OUTSIDE the PIE edit mask, or None when the
    mask covers the whole image. Drops the 1-pixel frame every PIE mask carries."""
    v = list(map(int, rle.split()))
    m = np.zeros(512 * 512, np.uint8)
    for a, b in zip(v[0::2], v[1::2]):
        m[a:a + b] = 1
    m = m.reshape(512, 512)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0
    if m[1:-1, 1:-1].mean() >= 0.999:
        return None
    if size != 512:
        from PIL import Image
        m = np.asarray(Image.fromarray(m * 255).resize((size, size), Image.NEAREST)) > 127
    return ~m.astype(bool)



def psnr_outside(ref: torch.Tensor, img: torch.Tensor, keep: np.ndarray) -> float:
    a = ((ref.float() + 1) / 2).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    b = ((img.float() + 1) / 2).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    mse = float(((a - b) ** 2)[keep].mean())
    return 10 * np.log10(1.0 / max(mse, 1e-12))


@torch.no_grad()
def traj_energy(traj, v_theta, N, dev, dtype) -> float:
    """run_sweep convention: sum_k ||x_{k+1} - x_k - dt*b(x_k)||^2 * dt."""
    dt = 1.0 / N
    s = torch.linspace(0.0, 1.0, N + 1)
    e = 0.0
    for k in range(N):
        x_k = traj[k].to(device=dev, dtype=dtype)
        u = (traj[k + 1].to(device=dev, dtype=dtype).float() - x_k.float()
             - dt * v_theta(s[k].to(dev), x_k).float())
        e += float(u.flatten(1).norm(dim=1).pow(2).mean()) * dt
    return e


def ref_row(ref_dir, spec, lam):
    if not ref_dir or not os.path.isfile(os.path.join(ref_dir, "results.csv")):
        return None
    for r in csv.DictReader(open(os.path.join(ref_dir, "results.csv"))):
        if r["spec"] == spec and r["start"] == "inverted" and abs(float(r["lam"]) - lam) < 1e-9:
            return r
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pie_root", default="/home/shervink/datasets/PIE_Bench_pp")
    p.add_argument("--examples", nargs="+", default=["1:43", "8:0", "1:0", "6:0"])
    p.add_argument("--lam", type=float, default=100.0)
    p.add_argument("--mult", nargs="+", type=float, default=[1.3, 1.5],
                   help="one edit per value: tau = s0 + (mult - 1) * |s0|")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--H", type=int, default=2)
    p.add_argument("--n_inner", type=int, default=12)
    p.add_argument("--lr_inner", type=float, default=1e-2)
    p.add_argument("--n_fp", type=int, default=5)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--mu_init", type=float, default=0.5)
    p.add_argument("--lr_dual", type=float, default=0.5)
    p.add_argument("--ref_dir", default="results/inversion_mpc_256_N16")
    p.add_argument("--out_dir", default="results/pie_edit")
    a = p.parse_args(argv)
    os.makedirs(os.path.join(a.out_dir, "img"), exist_ok=True)

    import run_sweep as rs
    assert os.path.dirname(os.path.abspath(rs.__file__)) == os.path.dirname(os.path.abspath(__file__)), (
        f"imported the wrong run_sweep: {rs.__file__}")
    from fluxfm_sampler_reward import FluxFlowMapSampler
    from image_mpc import make_flux_wrappers
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
    fields = ["spec", "id", "kind", "lam", "mult", "s0", "tau", "pickscore", "h", "feasible",
              "psnr", "ssim", "lpips", "psnr_outside_mask", "latent_rel_err", "energy",
              "mu_final", "time_s", "img"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    def record(row):
        row = {k: row.get(k, "") for k in fields}
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        return row

    for ex in exs:
        P, T = ex["source_prompt"], ex["target_prompt"]
        tag = f"{ex['spec'].replace(':', '-')}_{ex['id']}"
        print(f"\n=== {ex['spec']}  {ex['split']} id {ex['id']}\n    source: {P}\n    target: {T}", flush=True)
        sampler.set_prompt_embeddings(P)
        emb = (sampler.cached_prompt_embeds, sampler.cached_pooled_prompt_embeds)
        x = fi.pil_to_tensor(ex["image"], a.size, dev, sampler.dtype)
        z_ref = fi.vae_encode(sampler, x).float()
        keep = outside_mask(ex["mask_rle"], a.size)
        ex["image"].resize((a.size, a.size)).save(os.path.join(a.out_dir, "img", f"{tag}__original.png"))

        # PickScore against the TARGET prompt -- run_sweep's own scorer, so h is
        # exactly the constraint every earlier ConicFlow run enforced.
        cfg = rs.RunConfig(prompt=P, edit_prompt=T, N=a.N)
        ps, ps_obj = rs._build_reward_fn(sampler, cfg, torch.device(dev), torch.bfloat16,
                                         reward_model="pickscore", drop_text_tower=True)
        with torch.no_grad():
            s0 = float(ps(z_ref.to(sampler.dtype)).float().mean())
        taus = {m: s0 + (m - 1.0) * abs(s0) for m in a.mult}
        print(f"  s0 = pickscore(original, target) = {s0:.3f}   tau: "
              + ", ".join(f"m={m:g} -> {t:.3f}" for m, t in taus.items()), flush=True)

        t0 = time.time()
        z_inv, info = fi.invert(sampler, z_ref, emb, a.N, a.guidance, "fixed_point", a.n_fp)
        print(f"  inversion: {time.time()-t0:.0f}s  fp_res max {info['fp_residual_max']:.2e}", flush=True)

        v_theta, _, endpoint_map = make_flux_wrappers(
            sampler, P, guidance_scale=a.guidance, n_mpc_steps=a.N,
            sample_mode="flow_map2", endpoint_surrogate="jump")

        def r(z):
            return -0.5 * (z.float() - z_ref).pow(2).flatten(1).sum(1)

        def make_h(tau):
            def h(z):
                return (tau - ps(z).float()).reshape(-1, 1)      # (B, 1); <= 0 is satisfied
            return h

        def run(kind, mult=None):
            # recon is scored against the tightest tau so its h reads as "how far short"
            tau = taus[mult] if mult is not None else taus[max(taus)]
            t0 = time.time()
            out = receding_horizon_flow_control(
                v_theta=v_theta, r=r, x0=z_inv.to(sampler.dtype), N=a.N, H=a.H, lam=a.lam,
                optimizer="adam", n_iters=a.n_inner, lr=a.lr_inner, warmstart=True,
                drift_detach=True, h=(make_h(tau) if kind == "edit" else None),
                mu_init=a.mu_init, lr_dual=(a.lr_dual if kind == "edit" else 0.0),
                constraint_on="endpoint", endpoint_map=endpoint_map, verbose=False,
                show_progress=False, low_memory=True, device=torch.device(dev), return_dual_h=True)
            zN, traj, mu_final = out[0].float(), out[1], out[3]
            energy = traj_energy(traj, v_theta, a.N, dev, sampler.dtype)
            xN = fi.vae_decode(sampler, zN)
            with torch.no_grad():
                score = float(ps(zN.to(sampler.dtype)).float().mean())
            m = metrics(x, xN)
            img = f"{tag}__{kind}__lam{a.lam:g}" + (f"__m{mult:g}" if kind == "edit" else "") + ".png"
            fi.to_pil(xN).save(os.path.join(a.out_dir, "img", img))
            mu = float(mu_final.float().mean()) if (kind == "edit" and mu_final is not None) else ""
            row = record({"spec": ex["spec"], "id": ex["id"], "kind": kind, "lam": a.lam,
                          "mult": mult if kind == "edit" else "", "s0": s0, "tau": tau,
                          "pickscore": score, "h": tau - score, "feasible": score >= tau, **m,
                          "psnr_outside_mask": psnr_outside(x, xN, keep) if keep is not None else "",
                          "latent_rel_err": ((zN - z_ref).norm() / z_ref.norm()).item(),
                          "energy": energy, "mu_final": mu, "time_s": time.time() - t0, "img": img})
            print(f"  {kind:<5}{'' if mult is None else f' m={mult:g}'} pickscore {score:6.2f} (tau {tau:.2f}, h {tau-score:+.2f}, "
                  f"{'feasible' if score >= tau else 'INFEASIBLE'})  PSNR {m['psnr']:5.1f}  "
                  f"LPIPS {m['lpips']:.3f}  E {energy:7.2f}  mu {mu if mu == '' else f'{mu:.2f}'}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            del out, traj
            torch.cuda.empty_cache()

        # reconstruction-only reference: reuse when the earlier run has this image
        rr = ref_row(a.ref_dir, ex["spec"], a.lam)
        if rr is not None:
            import shutil
            img = f"{tag}__recon__lam{a.lam:g}.png"
            shutil.copy(os.path.join(a.ref_dir, "img", rr["img"]), os.path.join(a.out_dir, "img", img))
            from PIL import Image
            xr = fi.pil_to_tensor(Image.open(os.path.join(a.out_dir, "img", img)), a.size, dev, sampler.dtype)
            with torch.no_grad():
                score = float(ps(fi.vae_encode(sampler, xr)).float().mean())
            tau = taus[max(taus)]
            record({"spec": ex["spec"], "id": ex["id"], "kind": "recon", "lam": a.lam, "s0": s0,
                    "tau": tau, "pickscore": score, "h": tau - score, "feasible": score >= tau,
                    "psnr": rr["psnr"], "ssim": rr["ssim"], "lpips": rr["lpips"],
                    "psnr_outside_mask": psnr_outside(x, xr, keep) if keep is not None else "",
                    "latent_rel_err": rr["latent_rel_err"], "energy": rr["energy"],
                    "time_s": 0, "img": img})
            print(f"  recon reused from {a.ref_dir}: PSNR {float(rr['psnr']):.1f}  "
                  f"LPIPS {float(rr['lpips']):.3f}  pickscore {score:.2f}", flush=True)
        else:
            run("recon")
        for m in a.mult:
            run("edit", m)

        del ps, ps_obj
        gc.collect(); torch.cuda.empty_cache()
        out = draw(a, [ex], csv_path, f"grid_{tag}.png")
        print(f"### IMAGE DONE {ex['spec']} {out}", flush=True)

    out = draw(a, exs, csv_path, "grid.png")
    print(f"### ALL DONE {out}", flush=True)
    return 0


def draw(a, exs, csv_path, out_name):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    rows = list(csv.DictReader(open(csv_path)))
    ncol = 2 + len(a.mult)
    fig, axes = plt.subplots(len(exs), ncol, figsize=(2.85 * ncol, 3.35 * len(exs)), squeeze=False)
    for i, ex in enumerate(exs):
        tag = f"{ex['spec'].replace(':', '-')}_{ex['id']}"
        cells = [r for r in rows if r["spec"] == ex["spec"]]
        rec = next((r for r in cells if r["kind"] == "recon"), None)
        edits = {float(r["mult"]): r for r in cells if r["kind"] == "edit"}
        any_r = rec or next(iter(edits.values()), None)
        s0 = float(any_r["s0"]) if any_r else float("nan")
        ax = axes[i, 0]; ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(Image.open(os.path.join(a.out_dir, "img", f"{tag}__original.png")))
        ax.set_xlabel(f"pickscore(target) = {s0:.2f}", fontsize=8.5)
        ax.set_ylabel(f"{ex['spec']}\n{ex['target_prompt_marked'][:40]}", fontsize=8)
        if i == 0: ax.set_title("original", fontsize=10)
        cols = [(rec, f"reconstruction only (λ={a.lam:g})", None)] + \
               [(edits.get(m), f"edit, m={m:g}  (λ={a.lam:g})", m) for m in a.mult]
        for j, (r, title, m) in enumerate(cols, start=1):
            ax = axes[i, j]; ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(title, fontsize=9.5)
            if r is None:
                ax.text(.5, .5, "n/a", ha="center", va="center", color="0.5"); continue
            ax.imshow(Image.open(os.path.join(a.out_dir, "img", r["img"])))
            if m is None:
                ax.set_xlabel(f"pickscore {float(r['pickscore']):.2f}\n"
                              f"LPIPS to orig {float(r['lpips']):.3f}   E {float(r['energy']):.1f}", fontsize=8)
                continue
            feas = r["feasible"] == "True"
            col = "#2e7d32" if feas else "#c62828"
            for sp in ax.spines.values(): sp.set_color(col); sp.set_linewidth(2.2)
            mu = f"   μ {float(r['mu_final']):.1f}" if r["mu_final"] not in ("", None) else ""
            ax.set_xlabel(f"pickscore {float(r['pickscore']):.2f}  ≥ {float(r['tau']):.1f}?  h {float(r['h']):+.2f}\n"
                          f"LPIPS to orig {float(r['lpips']):.3f}   E {float(r['energy']):.1f}{mu}",
                          fontsize=8, color=col)
    fig.suptitle(f"ConicFlow edit of real images:  min Σ‖u‖²/(2λdt) + ½‖ẑ − z_ref‖²   s.t.  "
                 f"pickscore(target) ≥ τ = s0 + (m−1)|s0|\n{a.size}px, N={a.N}, H={a.H}, n_inner={a.n_inner}, "
                 f"lr={a.lr_inner:g}, μ₀={a.mu_init:g}, lr_dual={a.lr_dual:g}  ·  source-prompt drift  ·  "
                 f"start = fixed-point inversion (K={a.n_fp})", fontsize=9.5, y=1.0)
    fig.tight_layout()
    out = os.path.abspath(os.path.join(a.out_dir, out_name))
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out

if __name__ == "__main__":
    sys.exit(main())
