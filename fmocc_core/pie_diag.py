#!/usr/bin/env python3
"""
pie_diag.py -- why are ConicFlow's real-image edits so conservative, and what fixes it?

pie_edit.py's result: PickScore(target) rises 12-14 points and clears the threshold
while the image barely changes (LPIPS 0.04-0.11) -- the edit is a fine texture
overlay, >50% of its energy above half-Nyquist. This script runs the same edit under
one-factor-at-a-time changes, each aimed at one suspected cause:

  base          pie_edit.py's setting, rerun with per-step logging
  aug           constraint = PickScore averaged over K random views (crop, flip,
                low-pass, noise) -> the scorer can no longer be satisfied by a
                pixel-aligned high-frequency pattern            [gameable scorer]
  mu20          mu_init = 20 instead of 0.5 -> constraint pressure from step 0,
                while coarse structure is still open            [back-loaded control]
  tgt_drift     MPC on the TARGET-prompt dynamics (start is still the source-prompt
                inversion) -> the prior helps instead of fighting   [source drift]
  no_recon      drop the -1/2||z - z_ref||^2 reward -> nothing pulls the endpoint
                back to the original                             [projection objective]
  aug_mu20, aug_tgt_mu20   combinations

Every run logs, per MPC step: control norm, mu, and the realized constraint h at
the predicted endpoint. A final pass scores every output with scorers the optimizer
never saw (ImageReward, CLIP) and with PickScore on fixed augmented views, so
"did the edit happen" is not judged by the number that was optimized.

  CUDA_VISIBLE_DEVICES=0 python pie_diag.py --out_dir results/pie_diag
  python pie_diag.py --dry_run                       # print the plan, no GPU
  CUDA_VISIBLE_DEVICES=0 python pie_diag.py --eval_only --eval_dirs results/pie_edit
"""
import argparse
import csv
import gc
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flux_inversion as fi
import pie_edit as pe
from scoring_views import augment

ARMS = {
    "base":         {},
    "aug":          {"aug_k": 6},
    "mu20":         {"mu_init": 20.0},
    "tgt_drift":    {"drift": "target"},
    "no_recon":     {"beta": 0.0},
    "aug_mu20":     {"aug_k": 6, "mu_init": 20.0},
    "aug_tgt_mu20": {"aug_k": 6, "mu_init": 20.0, "drift": "target"},
}
ARM_NOTE = {
    "base": "pie_edit setting", "aug": "K-view PickScore", "mu20": "μ₀ = 20",
    "tgt_drift": "target-prompt drift", "no_recon": "no ‖z − z_ref‖² reward",
    "aug_mu20": "K-view + μ₀ = 20", "aug_tgt_mu20": "K-view + μ₀ = 20 + target drift",
}


# ── per-step traces ─────────────────────────────────────────────────────────

@torch.no_grad()
def step_trace(traj, v_theta, endpoint_map, score_plain, tau, N, dev, dtype):
    """Per MPC step k: ||u_k|| with u_k = x_{k+1} - x_k - dt*b(x_k), and the realized
    constraint h = tau - pickscore at the predicted endpoint of x_k."""
    dt = 1.0 / N
    s = torch.linspace(0.0, 1.0, N + 1)
    u_norm, h_ep = [], []
    for k in range(N + 1):
        x_k = traj[k].to(device=dev, dtype=dtype)
        if k < N:
            u = (traj[k + 1].to(device=dev, dtype=dtype).float() - x_k.float()
                 - dt * v_theta(s[k].to(dev), x_k).float())
            u_norm.append(float(u.flatten(1).norm(dim=1).mean()))
        z_ep = endpoint_map(s[k].to(dev), x_k) if k < N else x_k
        h_ep.append(tau - float(score_plain(z_ep.to(dtype))))
    energy = float(sum(n * n for n in u_norm) * dt)
    return u_norm, h_ep, energy


def _floats(seq):
    out = []
    for m in (seq if seq is not None else []):
        try:
            out.append(float(torch.as_tensor(m).float().mean()))
        except Exception:
            out.append(float("nan"))
    return out


# ── held-out evaluation of saved images ─────────────────────────────────────

def evaluate(items, dev="cuda:0", k_views=16):
    """items: dicts with 'png', 'target' (prompt), 'source'. Adds pickscore, robust and
    low-passed PickScore, ImageReward and CLIP against the target prompt. Scorers are
    run_sweep's own, fed the PNG through the decode_fn hook."""
    import run_sweep as rs
    from PIL import Image
    slot = {"img": None}
    dummy = torch.zeros(1, device=dev)
    for target in dict.fromkeys(it["target"] for it in items):
        sel = [it for it in items if it["target"] == target]
        cfg = rs.RunConfig(prompt=sel[0]["source"], edit_prompt=target, N=8)
        fns, objs = {}, []
        for typ in ("pickscore", "imagereward", "clip"):
            f, o = rs._build_reward_fn(None, cfg, torch.device(dev), torch.float32, reward_model=typ,
                                       decode_fn=lambda z: slot["img"])
            fns[typ] = f; objs.append(o)
        for it in sel:
            x = fi.pil_to_tensor(Image.open(it["png"]), Image.open(it["png"]).size[0], dev, torch.float32)
            S = x.shape[-1]
            with torch.no_grad():
                slot["img"] = x
                for typ, f in fns.items():
                    it[f"ev_{typ}"] = float(f(dummy).float().mean())
                g = torch.Generator(device=dev).manual_seed(0)
                slot["img"] = augment(x, k_views, gen=g, identity=False)
                it["ev_pickscore_robust"] = float(fns["pickscore"](dummy).float().mean())
                lo = F.interpolate(F.interpolate(x, size=(S // 2, S // 2), mode="bilinear", antialias=True),
                                   size=(S, S), mode="bilinear")
                slot["img"] = lo
                it["ev_pickscore_lowpass"] = float(fns["pickscore"](dummy).float().mean())
        del fns, objs
        slot["img"] = None
        gc.collect(); torch.cuda.empty_cache()
    return items


EV_KEYS = ["ev_pickscore", "ev_pickscore_robust", "ev_pickscore_lowpass", "ev_imagereward", "ev_clip"]


def eval_dirs(dirs, pie_root):
    """Score an existing pie_edit-style output dir (originals, recon, edits)."""
    for d in dirs:
        rows = list(csv.DictReader(open(os.path.join(d, "results.csv"))))
        items, seen = [], set()
        for r in rows:
            ex = fi.load_pie(pie_root, r["spec"])
            tag = f"{r['spec'].replace(':', '-')}_{r['id']}"
            if tag not in seen:
                seen.add(tag)
                items.append({"spec": r["spec"], "name": "original", "source": ex["source_prompt"],
                              "target": ex["target_prompt"], "png": os.path.join(d, "img", f"{tag}__original.png")})
            items.append({"spec": r["spec"], "name": r["kind"] + (f" m={r['mult']}" if r.get("mult") else ""),
                          "source": ex["source_prompt"], "target": ex["target_prompt"],
                          "png": os.path.join(d, "img", r["img"])})
        evaluate(items)
        out = os.path.join(d, "heldout_eval.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["spec", "name"] + EV_KEYS, extrasaction="ignore")
            w.writeheader(); w.writerows(items)
        print(f"\n{d}\n{'spec':<6}{'image':<14}{'pick':>8}{'pick_rob':>10}{'pick_lo':>9}{'ImgRew':>9}{'CLIP':>8}")
        for it in items:
            print(f"{it['spec']:<6}{it['name']:<14}{it['ev_pickscore']:8.2f}{it['ev_pickscore_robust']:10.2f}"
                  f"{it['ev_pickscore_lowpass']:9.2f}{it['ev_imagereward']:9.3f}{it['ev_clip']:8.4f}")
        print(f"-> {out}")


# ── main ────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pie_root", default="/home/shervink/datasets/PIE_Bench_pp")
    p.add_argument("--examples", nargs="+", default=["1:43", "8:0"])
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--mult", type=float, default=1.3)
    p.add_argument("--lam", type=float, default=100.0)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--H", type=int, default=2)
    p.add_argument("--n_inner", type=int, default=12)
    p.add_argument("--lr_inner", type=float, default=1e-2)
    p.add_argument("--n_fp", type=int, default=5)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--mu_init", type=float, default=0.5)
    p.add_argument("--lr_dual", type=float, default=0.5)
    p.add_argument("--out_dir", default="results/pie_diag")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--eval_dirs", nargs="+", default=[])
    a = p.parse_args(argv)

    if a.dry_run:
        n = len(a.examples) * len(a.arms)
        aug = sum("aug_k" in ARMS[k] for k in a.arms) * len(a.examples)
        print(f"{len(a.examples)} images x {len(a.arms)} arms = {n} runs, image-major "
              f"(m={a.mult:g}, lam={a.lam:g}, {a.size}px, N={a.N}, H={a.H}, n_inner={a.n_inner})")
        for ex in a.examples:
            for k in a.arms:
                print(f"  {ex:<6}{k:<14}{ARMS[k] or '(baseline)'}")
        print(f"estimate: {n} x ~5.0 min + {aug} x ~1 min (K-view overhead) + ~10 min "
              f"(load, inversion, traces, held-out eval) = ~{n * 5.0 + aug + 10:.0f} min")
        return 0
    if a.eval_only:
        eval_dirs(a.eval_dirs, a.pie_root)
        return 0

    os.makedirs(os.path.join(a.out_dir, "img"), exist_ok=True)
    import run_sweep as rs
    from fluxfm_sampler_reward import FluxFlowMapSampler
    from image_mpc import make_flux_wrappers
    from image_mpc_edited_sung import receding_horizon_flow_control

    exs = [fi.load_pie(a.pie_root, s) for s in a.examples]
    prompts = list(dict.fromkeys(q for e in exs for q in (e["source_prompt"], e["target_prompt"])))
    dev = "cuda:0"
    sampler = FluxFlowMapSampler(model_id="black-forest-labs/FLUX.1-dev",
                                 lora_path=os.path.join(fi.FMRG_DIR, "checkpoints", "flux-flowmap-lora"),
                                 device=dev, dtype=torch.bfloat16, prompt=prompts[0],
                                 additional_prompts=prompts, quantize_mode="int8")
    emb = {q: tuple(t.to(dev) for t in sampler.prompt_cache[q]) for q in prompts}
    metrics = fi.Metrics(dev)

    fields = ["spec", "id", "arm", "lam", "mult", "s0", "tau", "tau_kind", "pickscore", "h", "feasible",
              "psnr", "ssim", "lpips", "psnr_outside_mask", "latent_rel_err", "energy", "energy_first_half_frac",
              "mu_init", "mu_final", "time_s", "img"] + EV_KEYS
    csv_path = os.path.join(a.out_dir, "results.csv")
    rows_all = []

    def flush():
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows_all)

    for ex in exs:
        P, T = ex["source_prompt"], ex["target_prompt"]
        tag = f"{ex['spec'].replace(':', '-')}_{ex['id']}"
        print(f"\n=== {ex['spec']}  id {ex['id']}\n    source: {P}\n    target: {T}", flush=True)
        x = fi.pil_to_tensor(ex["image"], a.size, dev, sampler.dtype)
        z_ref = fi.vae_encode(sampler, x).float()
        keep = pe.outside_mask(ex["mask_rle"], a.size)
        ex["image"].resize((a.size, a.size)).save(os.path.join(a.out_dir, "img", f"{tag}__original.png"))

        # one PickScore model; `mode["k"]` switches its decode between the plain image and K views
        mode = {"k": 0}

        def decode_fn(z):
            img = sampler.decode(z)
            return augment(img, mode["k"]) if mode["k"] else img

        cfg = rs.RunConfig(prompt=P, edit_prompt=T, N=a.N)
        ps_raw, ps_obj = rs._build_reward_fn(sampler, cfg, torch.device(dev), torch.bfloat16,
                                             reward_model="pickscore", drop_text_tower=True, decode_fn=decode_fn)

        def score(z, k=0):
            mode["k"] = k
            try:
                return ps_raw(z).float().mean()
            finally:
                mode["k"] = 0

        with torch.no_grad():
            s0 = float(score(z_ref.to(sampler.dtype)))
            s0_aug = float(np.mean([float(score(z_ref.to(sampler.dtype), 8)) for _ in range(4)]))
        tau_plain = s0 + (a.mult - 1.0) * abs(s0)
        tau_aug = s0_aug + (a.mult - 1.0) * abs(s0_aug)
        print(f"  s0 = {s0:.3f} (K-view {s0_aug:.3f})   tau = {tau_plain:.3f} (K-view arms: {tau_aug:.3f})", flush=True)

        sampler.set_prompt_embeddings(P)
        t0 = time.time()
        z_inv, info = fi.invert(sampler, z_ref, emb[P], a.N, a.guidance, "fixed_point", a.n_fp)
        print(f"  inversion: {time.time()-t0:.0f}s  fp_res max {info['fp_residual_max']:.2e}", flush=True)

        def save(name, zN, extra):
            xN = fi.vae_decode(sampler, zN)
            with torch.no_grad():
                sc = float(score(zN.to(sampler.dtype)))
            m = metrics(x, xN)
            img = f"{tag}__{name}.png"
            fi.to_pil(xN).save(os.path.join(a.out_dir, "img", img))
            row = {"spec": ex["spec"], "id": ex["id"], "arm": name, "lam": a.lam, "mult": a.mult, "s0": s0,
                   "pickscore": sc, **m,
                   "psnr_outside_mask": pe.psnr_outside(x, xN, keep) if keep is not None else "",
                   "latent_rel_err": ((zN.float() - z_ref).norm() / z_ref.norm()).item(), "img": img,
                   "_png": os.path.join(a.out_dir, "img", img), "_source": P, "_target": T, **extra}
            rows_all.append(row); flush()
            return row

        # references that cost one sampling pass each: no control, source / target prompt
        for name, q in (("ref_recon", P), ("ref_tgt_regen", T)):
            zN = fi.generate(sampler, z_inv, emb[q], a.N, a.guidance, "flow_map2")
            r_ = save(name, zN, {"tau": tau_plain, "tau_kind": "plain"})
            print(f"  {name:<14} pickscore {r_['pickscore']:6.2f}  LPIPS {r_['lpips']:.3f}", flush=True)

        for arm in a.arms:
            c = {"aug_k": 0, "mu_init": a.mu_init, "drift": "source", "beta": 1.0, **ARMS[arm]}
            q = T if c["drift"] == "target" else P
            sampler.set_prompt_embeddings(q)          # make_flux_wrappers captures the cached embeds
            v_theta, _, endpoint_map = make_flux_wrappers(
                sampler, q, guidance_scale=a.guidance, n_mpc_steps=a.N,
                sample_mode="flow_map2", endpoint_surrogate="jump")
            tau = tau_aug if c["aug_k"] else tau_plain
            beta, k = c["beta"], c["aug_k"]

            def r(z):
                return -0.5 * beta * (z.float() - z_ref).pow(2).flatten(1).sum(1)

            def h(z):
                return (tau - score(z, k)).reshape(-1, 1)

            t0 = time.time()
            try:
                out = receding_horizon_flow_control(
                    v_theta=v_theta, r=r, x0=z_inv.to(sampler.dtype), N=a.N, H=a.H, lam=a.lam,
                    optimizer="adam", n_iters=a.n_inner, lr=a.lr_inner, warmstart=True,
                    drift_detach=True, h=h, mu_init=c["mu_init"], lr_dual=a.lr_dual,
                    constraint_on="endpoint", endpoint_map=endpoint_map, verbose=False,
                    show_progress=False, low_memory=True, device=torch.device(dev), return_dual_h=True)
            except Exception as e:                    # keep going; one arm must not sink the rest
                print(f"  !! {arm}: {type(e).__name__}: {str(e)[:300]}", flush=True)
                gc.collect(); torch.cuda.empty_cache()
                continue
            zN, traj, mu_final = out[0].float(), out[1], out[3]
            mu_hist = _floats(out[4]) if len(out) > 4 else []
            u_norm, h_ep, energy = step_trace(traj, v_theta, endpoint_map,
                                              lambda z: score(z), tau_plain, a.N, dev, sampler.dtype)
            e_steps = [n * n / a.N for n in u_norm]
            row = save(arm, zN, {
                "tau": tau, "tau_kind": "K-view" if k else "plain", "energy": energy,
                "energy_first_half_frac": sum(e_steps[:a.N // 2]) / max(sum(e_steps), 1e-12),
                "mu_init": c["mu_init"],
                "mu_final": float(mu_final.float().mean()) if mu_final is not None else "",
                "time_s": time.time() - t0})
            row["h"] = tau_plain - row["pickscore"]; row["feasible"] = row["pickscore"] >= tau_plain
            flush()
            with open(os.path.join(a.out_dir, f"{tag}__{arm}__trace.json"), "w") as f:
                json.dump({"arm": arm, "config": c, "tau": tau, "tau_plain": tau_plain, "u_norm": u_norm,
                           "h_endpoint": h_ep, "mu_hist": mu_hist}, f, indent=1)
            print(f"  {arm:<14} pickscore {row['pickscore']:6.2f} (tau {tau_plain:.2f}, h {row['h']:+.2f})  "
                  f"LPIPS {row['lpips']:.3f}  E {energy:6.2f}  first-half E {row['energy_first_half_frac']*100:4.0f}%  "
                  f"mu {row['mu_final'] if row['mu_final'] == '' else format(row['mu_final'], '.1f')}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            del out, traj
            gc.collect(); torch.cuda.empty_cache()

        del ps_raw, ps_obj
        gc.collect(); torch.cuda.empty_cache()
        print(f"### IMAGE DONE {ex['spec']}", flush=True)

    # held-out scoring of everything, then figures
    print("\nheld-out scoring ...", flush=True)
    items = [{"png": os.path.join(a.out_dir, "img", f"{e['spec'].replace(':', '-')}_{e['id']}__original.png"),
              "source": e["source_prompt"], "target": e["target_prompt"], "spec": e["spec"], "arm": "original"}
             for e in exs]
    for r_ in rows_all:
        r_["png"], r_["source"], r_["target"] = r_["_png"], r_["_source"], r_["_target"]
    evaluate(items + rows_all)
    flush()
    with open(os.path.join(a.out_dir, "originals_eval.json"), "w") as f:
        json.dump([{k: v for k, v in it.items() if k in ("spec", *EV_KEYS)} for it in items], f, indent=1)
    out = draw(a, exs, rows_all, items)
    print(f"### ALL DONE {out}", flush=True)
    return 0


def draw(a, exs, rows, originals):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    names = ["ref_recon", "ref_tgt_regen"] + [k for k in a.arms]
    title = {"ref_recon": "no control\n(source prompt)", "ref_tgt_regen": "no control\n(target prompt)",
             **{k: f"{k}\n{ARM_NOTE[k]}" for k in ARMS}}
    outs = []
    for ex in exs:
        tag = f"{ex['spec'].replace(':', '-')}_{ex['id']}"
        cells = {r["arm"]: r for r in rows if r["spec"] == ex["spec"]}
        o = next(it for it in originals if it["spec"] == ex["spec"])
        fig, axes = plt.subplots(1, len(names) + 1, figsize=(2.55 * (len(names) + 1), 4.1), squeeze=False)
        ev = lambda d: (f"pick {d['ev_pickscore']:.1f}  robust {d['ev_pickscore_robust']:.1f}\n"
                        f"ImgRew {d['ev_imagereward']:+.2f}  CLIP {d['ev_clip']:.3f}")
        ax = axes[0, 0]; ax.set_xticks([]); ax.set_yticks([]); ax.set_title("original", fontsize=9)
        ax.imshow(Image.open(o["png"])); ax.set_xlabel(ev(o), fontsize=7.5)
        for j, n in enumerate(names, start=1):
            ax = axes[0, j]; ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title[n], fontsize=8.5)
            r = cells.get(n)
            if r is None:
                ax.text(.5, .5, "failed", ha="center", va="center", color="0.5"); continue
            ax.imshow(Image.open(r["_png"]))
            cap = ev(r) + f"\nLPIPS {float(r['lpips']):.3f}"
            if r.get("energy") not in ("", None):
                cap += f"  E {float(r['energy']):.1f}  early {float(r['energy_first_half_frac'])*100:.0f}%"
            ax.set_xlabel(cap, fontsize=7.5)
        fig.suptitle(f"{ex['spec']}  '{ex['source_prompt']}'  →  '{ex['target_prompt_marked']}'   "
                     f"m={a.mult:g}, λ={a.lam:g}, N={a.N}", fontsize=9.5, y=1.0)
        fig.tight_layout()
        out = os.path.abspath(os.path.join(a.out_dir, f"sheet_{tag}.png"))
        fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig); outs.append(out)

        # per-step traces: control norm and realized h
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.0))
        for n in a.arms:
            f = os.path.join(a.out_dir, f"{tag}__{n}__trace.json")
            if not os.path.isfile(f): continue
            t = json.load(open(f))
            a1.plot(range(len(t["u_norm"])), t["u_norm"], "-o", ms=3, label=n)
            a2.plot(range(len(t["h_endpoint"])), t["h_endpoint"], "-o", ms=3, label=n)
        a1.set_xlabel("MPC step"); a1.set_ylabel("‖u_k‖"); a1.set_title("when control is applied"); a1.grid(alpha=.25)
        a2.axhline(0, color="0.3", ls="--", lw=1); a2.set_xlabel("MPC step")
        a2.set_ylabel("h = τ − pickscore(predicted endpoint)"); a2.set_title("when the constraint is met")
        a2.grid(alpha=.25); a2.legend(fontsize=7.5)
        fig.suptitle(f"{ex['spec']}  per-step traces", fontsize=10); fig.tight_layout()
        out = os.path.abspath(os.path.join(a.out_dir, f"trace_{tag}.png"))
        fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig); outs.append(out)
    return outs


if __name__ == "__main__":
    sys.exit(main())
