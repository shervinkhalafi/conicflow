#!/usr/bin/env python3
"""
edit_diag.py -- why are ConicFlow's edits conservative in the GENERATION setting, and
which change fixes it?  One prompt per process, one factor per arm.

Baseline = the suite/lam_sweep configuration (N=8, H=2, n_inner=24, Adam lr 1e-2, jump
surrogate, mu_init=0.5, lr_dual=0.5, no reward, PickScore >= m * s0 on each edit text),
from the same seed, so the uncontrolled image is the one already on disk.

  base         baseline, rerun with per-step logging
  lr3x         Adam lr 3e-2                          [is the step size set by the optimizer?]
  mu20         mu_init = 20                          [back-loaded control]
  lr3x_mu20    both solver changes
  views        PickScore averaged over 6 random views [gameable scorer]
  calib        PickScore only, threshold = what real samples of the edit text score
                                                     [thresholds outside the natural range]
  ens3         PickScore, CLIP and ImageReward as SEPARATE constraints, each at its own
               natural target level
  sum3         the same three scorers as ONE constraint on their average progress
  combo        ens3 + views + mu20 + lr3x
  combo_a1.5   combo at 1.5x the natural gap

CALIBRATION.  For each edit text, s_tgt = mean score of 4 uncontrolled samples generated
FROM that text, scored against it; s0 = score of this run's uncontrolled image. Progress
p = (s - s0) / (s_tgt - s0): 0 = the uncontrolled image, 1 = scores like a real image of
the edit text. Calibrated arms constrain p >= alpha. Every calibrated constraint is
multiplied by PickScore's natural gap, so h is in PickScore units in every arm and the
dual step means the same thing throughout.

Each arm also saves an EARLY-WINDOW image: control up to step --ew_step, then the
uncontrolled flow to the end. A texture written early should not survive the remaining
denoising; a semantic change should.

A final pass scores every image with all three scorers, PickScore on 16 fixed views and
on a low-passed copy, so success is not judged only by the optimized number.

  CUDA_VISIBLE_DEVICES=0 python edit_diag.py --pair redhair --out_dir results/edit_diag
  python edit_diag.py --dry_run
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
import flux_inversion as fi            # path setup + Metrics / to_pil / pil_to_tensor
from scoring_views import augment

ALL = ["pickscore", "clip", "imagereward"]
MIN_GAP = {"pickscore": 0.5, "clip": 0.01, "imagereward": 0.2}
PAIRS = {
    "redhair": dict(prompt="a portrait of a woman", texts=["a woman with bright red hair"],
                    suite="p05a_redhair"),
    "cabin":   dict(prompt="a house by a lake", texts=["a wooden log cabin", "a house in heavy fog"],
                    suite="p11_cabin_fog"),
}
ARMS = {
    "base":       {},
    "lr3x":       {"lr": 0.03},
    "mu20":       {"mu_init": 20.0},
    "lr3x_mu20":  {"lr": 0.03, "mu_init": 20.0},
    "views":      {"views": 6},
    "calib":      {"mode": "each", "types": ["pickscore"], "alpha": 1.0},
    "ens3":       {"mode": "each", "types": ALL, "alpha": 1.0},
    "sum3":       {"mode": "sum", "types": ALL, "alpha": 1.0},
    "combo":      {"mode": "each", "types": ALL, "alpha": 1.0, "views": 6, "mu_init": 20.0, "lr": 0.03},
    "combo_a1.5": {"mode": "each", "types": ALL, "alpha": 1.5, "views": 6, "mu_init": 20.0, "lr": 0.03},
}
NOTE = {"base": "suite setting", "lr3x": "Adam lr ×3", "mu20": "μ₀ = 20", "lr3x_mu20": "lr ×3, μ₀ = 20",
        "views": "6-view PickScore", "calib": "PickScore ≥ natural level",
        "ens3": "3 scorers, separate", "sum3": "3 scorers, averaged",
        "combo": "ens3 + views + μ₀ + lr", "combo_a1.5": "combo, α = 1.5"}


# ── scorers: run_sweep's formulas, one model per type shared across texts ───

class Scorers:
    def __init__(self, rs, dev, texts):
        from fluxfm_sampler_reward import create_clip_preprocess
        from transformers import AutoProcessor
        self.rs, self.dev, self.texts = rs, dev, texts
        self.pre = create_clip_preprocess(224)
        self.tok = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=rs.HF_CACHE)
        self.models, self.txt = {}, {}

    def load(self, types):
        from fluxfm_sampler_reward import CLIPReward, ImageRewardReward, PickScoreReward
        for typ in types:
            if typ in self.models:
                continue
            if typ == "imagereward":
                try:
                    import imagereward_compat  # noqa: F401
                except ImportError:
                    pass
                obj = ImageRewardReward(device=self.dev, dtype=torch.float32, cache_dir=self.rs.HF_CACHE)
            else:
                cls = PickScoreReward if typ == "pickscore" else CLIPReward
                obj = cls(device=self.dev, dtype=torch.float32, tokenizer=self.tok, cache_dir=self.rs.HF_CACHE)
                with torch.no_grad():
                    for t in self.texts:
                        toks = self.tok(t, return_tensors="pt", padding=True, max_length=77,
                                        truncation=True).to(self.dev)
                        f = obj.model.get_text_features(**toks)
                        self.txt[typ, t] = f / f.norm(dim=-1, keepdim=True)
                self.rs._drop_text_tower(obj, typ)
            self.models[typ] = obj

    def unload(self, types):
        for typ in types:
            self.models.pop(typ, None)
        gc.collect(); torch.cuda.empty_cache()

    def score_all(self, views, types):
        """views: (K, 3, S, S) in [-1, 1]. -> {(type, text): scalar tensor, mean over views}."""
        x = self.pre(((views.float() + 1.0) / 2.0).to(self.dev))
        out = {}
        for typ in types:
            obj = self.models[typ]
            if typ == "imagereward":
                for t in self.texts:
                    out[typ, t] = 2.0 - obj(x, t)           # the checkpoint's own z-score
                continue
            f = obj.model.get_image_features(x)
            f = f / f.norm(dim=-1, keepdim=True)
            for t in self.texts:
                sim = f @ self.txt[typ, t].T
                out[typ, t] = (obj.model.logit_scale.exp() * sim).mean() if typ == "pickscore" else sim.mean()
        return out


def spectrum(res):
    g = res.mean(2); Fq = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    n = g.shape[0]; yy, xx = np.mgrid[:n, :n]; rr = np.hypot(yy - n / 2, xx - n / 2) / (n / 2)
    tot = max(Fq.sum(), 1e-20)
    return float(Fq[rr < .125].sum() / tot), float(Fq[rr >= .5].sum() / tot)


# ── main ────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pair", default="redhair", choices=list(PAIRS))
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--seed", type=int, default=310)
    p.add_argument("--mult", type=float, default=1.3)
    p.add_argument("--lam", type=float, default=100.0)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--H", type=int, default=2)
    p.add_argument("--n_inner", type=int, default=24)
    p.add_argument("--lr_inner", type=float, default=1e-2)
    p.add_argument("--mu_init", type=float, default=0.5)
    p.add_argument("--lr_dual", type=float, default=0.5)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--n_cal", type=int, default=4, help="target-text samples for the natural score level")
    p.add_argument("--ew_step", type=int, default=5, help="early-window variant: control stops after this step")
    p.add_argument("--out_dir", default="results/edit_diag")
    p.add_argument("--smoke", action="store_true", help="N=4, n_inner=2, n_cal=1, arms base+combo: crash and memory check")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args(argv)
    if a.smoke:
        a.N, a.n_inner, a.n_cal, a.ew_step, a.arms = 4, 2, 1, 2, ["base", "combo"]
    pair = PAIRS[a.pair]; P, texts = pair["prompt"], pair["texts"]

    if a.dry_run:
        mins = {k: 4.3 for k in ARMS}; mins.update({"views": 5.0, "ens3": 6.0, "sum3": 6.0, "combo": 8.0, "combo_a1.5": 8.0})
        scale = len(texts) ** 0.3
        print(f"pair {a.pair}: '{P}' -> {texts}   seed {a.seed}, m={a.mult:g}, lam={a.lam:g}, N={a.N}, n_inner={a.n_inner}")
        for k in a.arms:
            print(f"  {k:<12}{NOTE[k]:<28}{ARMS[k] or ''}")
        print(f"estimate: ~{sum(mins[k] for k in a.arms) * scale + 8:.0f} min "
              f"({len(a.arms)} runs + calibration, traces, early-window images, held-out scoring)")
        return 0

    out_dir = os.path.join(a.out_dir, a.pair + ("_smoke" if a.smoke else ""))
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    import run_sweep as rs
    assert os.path.dirname(os.path.abspath(rs.__file__)) == os.path.dirname(os.path.abspath(__file__))
    from fluxfm_sampler_reward import FluxFlowMapSampler
    from image_mpc import make_flux_wrappers
    from image_mpc_edited_sung import receding_horizon_flow_control

    dev = "cuda:0"
    sampler = FluxFlowMapSampler(model_id="black-forest-labs/FLUX.1-dev",
                                 lora_path=os.path.join(fi.FMRG_DIR, "checkpoints", "flux-flowmap-lora"),
                                 device=dev, dtype=torch.bfloat16, prompt=P,
                                 additional_prompts=[P] + texts, quantize_mode="int8")
    dtype = sampler.dtype
    emb = {q: tuple(t.to(dev) for t in sampler.prompt_cache[q]) for q in [P] + texts}
    metrics = fi.Metrics(dev)
    sc = Scorers(rs, dev, texts)

    def rollout(z, q, k0=0):
        """run_sweep._run_uncontrolled's loop (flow_map2), from grid index k0, prompt q."""
        pe_, pp_ = emb[q]
        ts = torch.linspace(1.0, 0.0, a.N + 1, device=dev, dtype=dtype)
        z = z.to(device=dev, dtype=dtype).clone()
        with torch.no_grad():
            for j in range(k0, a.N):
                t_c = ts[j].item(); t_n = max(0.0, t_c - 1.0 / a.N)
                u = sampler.predict_vector(z, t_c, prompt_embeds=pe_, pooled_prompt_embeds=pp_,
                                           guidance_scale=a.guidance, t_next=t_n)
                z = z - (ts[j] - ts[j + 1]).item() * u
        return z

    def noise(seed):
        torch.manual_seed(seed)
        in_ch = sampler.pipeline.transformer.config.in_channels // 4
        v = sampler.vae_scale_factor
        return torch.randn(1, in_ch, a.size // v, a.size // v, device=dev, dtype=dtype)

    def views_of(img, k, fixed=False):
        if not k:
            return img.float()
        g = torch.Generator(device=dev).manual_seed(0) if fixed else None
        return augment(img, k, gen=g, identity=not fixed)

    @torch.no_grad()
    def score_img(img, types, k=0):
        s = sc.score_all(views_of(img, k, fixed=True), types)
        return {key: float(v) for key, v in s.items()}

    # ── uncontrolled image, calibration ─────────────────────────────────────
    z0 = noise(a.seed)
    z_unc = rollout(z0, P)
    img_unc = sampler.decode_no_grad(z_unc).float()
    rs._decode_pil(sampler, z_unc).save(os.path.join(out_dir, "img", "uncontrolled.png"))
    import glob, hashlib
    ref = glob.glob(os.path.join("results", "suite", f"{pair['suite']}_seed{a.seed}", "m1.30", "mpc", "*__img_unc.png"))
    if ref and not a.smoke:
        same = hashlib.md5(open(ref[0], "rb").read()).hexdigest() == \
            hashlib.md5(open(os.path.join(out_dir, "img", "uncontrolled.png"), "rb").read()).hexdigest()
        print(f"  uncontrolled image identical to the suite's: {same}", flush=True)

    sc.load(ALL)
    cal = {"s0": {}, "s0_views": {}, "s_tgt": {}, "s_tgt_views": {}}
    for key, v in score_img(img_unc, ALL).items():
        cal["s0"][key] = v
    for key, v in score_img(img_unc, ALL, 16).items():
        cal["s0_views"][key] = v
    for t in texts:
        acc, accv = {typ: [] for typ in ALL}, {typ: [] for typ in ALL}
        for i in range(a.n_cal):
            z_t = rollout(noise(1000 + i), t)
            im = sampler.decode_no_grad(z_t).float()
            rs._decode_pil(sampler, z_t).save(os.path.join(out_dir, "img", f"target_sample_{texts.index(t)}_{i}.png"))
            s1, s2 = score_img(im, ALL), score_img(im, ALL, 16)
            for typ in ALL:
                acc[typ].append(s1[typ, t]); accv[typ].append(s2[typ, t])
        for typ in ALL:
            cal["s_tgt"][typ, t] = float(np.mean(acc[typ])); cal["s_tgt_views"][typ, t] = float(np.mean(accv[typ]))
    print(f"\n  calibration ({a.n_cal} samples per edit text)\n  {'scorer':<12}{'text':<32}{'s0':>8}{'s_target':>10}{'gap':>8}   usable")
    usable = {}
    for t in texts:
        for typ in ALL:
            gap = cal["s_tgt"][typ, t] - cal["s0"][typ, t]
            gapv = cal["s_tgt_views"][typ, t] - cal["s0_views"][typ, t]
            usable[typ, t] = gap > MIN_GAP[typ] and gapv > MIN_GAP[typ]
            print(f"  {typ:<12}{t[:30]:<32}{cal['s0'][typ, t]:8.3f}{cal['s_tgt'][typ, t]:10.3f}{gap:8.3f}   {usable[typ, t]}")
    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump({k: {f"{typ}|{t}": v for (typ, t), v in d.items()} for k, d in cal.items()}, f, indent=1)
    sys.stdout.flush()
    sc.unload(["clip", "imagereward"])

    sampler.set_prompt_embeddings(P)
    v_theta, _, endpoint_map = make_flux_wrappers(sampler, P, guidance_scale=a.guidance, n_mpc_steps=a.N,
                                                  sample_mode="flow_map2", endpoint_surrogate="jump")
    rows = []
    fields = ["pair", "arm", "variant", "config", "constraints", "energy", "energy_first_half_frac", "mu_final",
              "time_s", "peak_gb", "lpips_to_unc", "pixel_rms", "spec_low", "spec_high", "img"]

    def flush():
        keys = fields + sorted({k for r in rows for k in r if k.startswith("ev_")})
        with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    def r_zero(z):
        return z.new_zeros(z.shape[0])

    for arm in a.arms:
        c = {"lr": a.lr_inner, "mu_init": a.mu_init, "views": 0, "mode": "raw", "types": ["pickscore"],
             "alpha": None, **ARMS[arm]}
        if a.smoke:
            c["views"] = min(c["views"], 6)
        sc.load(c["types"])
        k = c["views"]
        s0 = cal["s0_views" if k else "s0"]; s_tgt = cal["s_tgt_views" if k else "s_tgt"]
        if c["mode"] == "raw":
            cons = [("pickscore", t) for t in texts]
            tau = {t: s0["pickscore", t] + (a.mult - 1.0) * abs(s0["pickscore", t]) for t in texts}
            desc = [f"pickscore('{t}') >= {tau[t]:.2f}" for t in texts]
        else:
            cons = [(typ, t) for t in texts for typ in c["types"] if usable[typ, t]]
            G = {t: (s_tgt["pickscore", t] - s0["pickscore", t]) if usable["pickscore", t] else 2.0 for t in texts}
            if c["mode"] == "each":
                desc = [f"{typ}('{t}') >= {s0[typ, t] + c['alpha'] * (s_tgt[typ, t] - s0[typ, t]):.3f}" for typ, t in cons]
            else:
                desc = [f"mean progress over {[typ for typ, tt in cons if tt == t]} on '{t}' >= {c['alpha']:g}" for t in texts]
        if not cons:
            print(f"  !! {arm}: no usable constraint, skipped", flush=True); continue
        types_used = sorted({typ for typ, _ in cons})

        def h(z, k=k, c=c, cons=cons):
            img = sampler.decode(z)
            S = sc.score_all(augment(img, k) if k else img.float(), types_used)
            if c["mode"] == "raw":
                cols = [tau[t] - S["pickscore", t] for _, t in cons]
            else:
                prog = {(typ, t): (S[typ, t] - s0[typ, t]) / (s_tgt[typ, t] - s0[typ, t]) for typ, t in cons}
                if c["mode"] == "each":
                    cols = [G[t] * (c["alpha"] - prog[typ, t]) for typ, t in cons]
                else:
                    cols = [G[t] * (c["alpha"] - torch.stack([prog[typ, tt] for typ, tt in cons if tt == t]).mean())
                            for t in texts if any(tt == t for _, tt in cons)]
            return torch.stack([x.float().reshape(()) for x in cols]).reshape(1, -1).to(z.device)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = None
        for attempt in range(2):
            try:
                out = receding_horizon_flow_control(
                    v_theta=v_theta, r=r_zero, x0=z0, N=a.N, H=a.H, lam=a.lam, optimizer="adam",
                    n_iters=a.n_inner, lr=c["lr"], warmstart=True, drift_detach=True, h=h,
                    mu_init=c["mu_init"], lr_dual=a.lr_dual, constraint_on="endpoint",
                    endpoint_map=endpoint_map, verbose=False, show_progress=False, low_memory=True,
                    device=torch.device(dev), return_dual_h=True)
                break
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
                if k > 3 and attempt == 0:
                    print(f"  !! {arm}: OOM with {k} views, retrying with 3", flush=True)
                    k = 3
                    h.__defaults__ = (k, c, cons)
                else:
                    print(f"  !! {arm}: OOM, skipped", flush=True); break
            except Exception as e:
                print(f"  !! {arm}: {type(e).__name__}: {str(e)[:300]}", flush=True); break
        if out is None:
            continue
        zN, traj, mu_final = out[0], out[1], out[3]
        peak = torch.cuda.max_memory_allocated() / 1e9
        dt = 1.0 / a.N
        s_grid = torch.linspace(0.0, 1.0, a.N + 1)
        u_norm, h_ep = [], []
        with torch.no_grad():
            for j in range(a.N + 1):
                x_j = traj[j].to(device=dev, dtype=dtype)
                if j < a.N:
                    u = traj[j + 1].to(device=dev, dtype=dtype).float() - x_j.float() \
                        - dt * v_theta(s_grid[j].to(dev), x_j).float()
                    u_norm.append(float(u.flatten(1).norm(dim=1).mean()))
                z_ep = endpoint_map(s_grid[j].to(dev), x_j) if j < a.N else x_j
                S = sc.score_all(sampler.decode_no_grad(z_ep.to(dtype)).float(), ["pickscore"])
                h_ep.append([float(S["pickscore", t]) for t in texts])
        e_steps = [n * n * dt for n in u_norm]
        energy = float(sum(e_steps))
        mu_txt = "" if mu_final is None else "/".join(f"{float(x):.1f}" for x in mu_final.float().flatten())
        secs = time.time() - t0
        z_ew = rollout(traj[a.ew_step], P, k0=a.ew_step)
        for variant, z in (("full", zN), ("early_window", z_ew)):
            img = f"{arm}__{variant}.png"
            rs._decode_pil(sampler, z).save(os.path.join(out_dir, "img", img))
            rows.append({"pair": a.pair, "arm": arm, "variant": variant, "config": json.dumps(ARMS[arm]),
                         "constraints": " ; ".join(desc), "energy": energy if variant == "full" else
                         float(sum(e_steps[:a.ew_step])),
                         "energy_first_half_frac": sum(e_steps[:a.N // 2]) / max(energy, 1e-12),
                         "mu_final": mu_txt, "time_s": secs, "peak_gb": peak, "img": img})
        flush()
        with open(os.path.join(out_dir, f"{arm}__trace.json"), "w") as f:
            json.dump({"arm": arm, "config": c, "constraints": desc, "u_norm": u_norm,
                       "pickscore_at_predicted_endpoint": h_ep}, f, indent=1)
        print(f"  {arm:<12} E {energy:7.2f}  first-half {sum(e_steps[:a.N // 2]) / max(energy, 1e-12) * 100:3.0f}%  "
              f"pickscore {'/'.join(f'{x:.1f}' for x in h_ep[-1])}  mu {mu_txt}  peak {peak:.1f} GB  {secs:.0f}s  "
              f"| {' ; '.join(desc)}", flush=True)
        del out, traj
        gc.collect(); torch.cuda.empty_cache()

    # ── held-out scoring of every image ─────────────────────────────────────
    print("\nheld-out scoring ...", flush=True)
    sc.load(ALL)
    from PIL import Image
    unc_png = os.path.join(out_dir, "img", "uncontrolled.png")
    x_unc = fi.pil_to_tensor(Image.open(unc_png), a.size, dev, torch.float32)
    a_unc = np.asarray(Image.open(unc_png).convert("RGB"), np.float32) / 255.0
    rows.insert(0, {"pair": a.pair, "arm": "uncontrolled", "variant": "full", "img": "uncontrolled.png"})
    s0e = None
    for r_ in rows:
        png = os.path.join(out_dir, "img", r_["img"])
        x = fi.pil_to_tensor(Image.open(png), a.size, dev, torch.float32)
        plain, rob = score_img(x, ALL), score_img(x, ["pickscore"], 16)
        lo = F.interpolate(F.interpolate(x, size=(a.size // 2,) * 2, mode="bilinear", antialias=True),
                           size=(a.size,) * 2, mode="bilinear")
        low = score_img(lo, ["pickscore"])
        if r_["arm"] == "uncontrolled":
            s0e = plain
        for i, t in enumerate(texts):
            for typ in ALL:
                r_[f"ev_{typ}_t{i}"] = plain[typ, t]
                gap = cal["s_tgt"][typ, t] - s0e[typ, t]
                r_[f"ev_progress_{typ}_t{i}"] = (plain[typ, t] - s0e[typ, t]) / gap if abs(gap) > 1e-6 else float("nan")
            r_[f"ev_pickscore_robust_t{i}"] = rob["pickscore", t]
            r_[f"ev_pickscore_lowpass_t{i}"] = low["pickscore", t]
        if r_["arm"] != "uncontrolled":
            res = np.asarray(Image.open(png).convert("RGB"), np.float32) / 255.0 - a_unc
            r_["lpips_to_unc"] = metrics(x_unc, x)["lpips"]
            r_["pixel_rms"] = float(np.sqrt((res ** 2).mean()))
            r_["spec_low"], r_["spec_high"] = spectrum(res)
    flush()
    outs = draw(a, out_dir, texts, rows, P)
    print("### ALL DONE " + "  ".join(outs), flush=True)
    return 0


def draw(a, out_dir, texts, rows, P):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    arms = [k for k in a.arms if any(r["arm"] == k for r in rows)]
    unc = next(r for r in rows if r["arm"] == "uncontrolled")
    nt = len(texts)

    def cap(r):
        ps = "/".join(f"{r[f'ev_pickscore_t{i}']:.1f}" for i in range(nt))
        rb = "/".join(f"{r[f'ev_pickscore_robust_t{i}']:.1f}" for i in range(nt))
        pr = "  ".join(f"{typ[:4]} " + "/".join(f"{r[f'ev_progress_{typ}_t{i}']:+.1f}" for i in range(nt)) for typ in ALL)
        s = f"PickScore {ps}  (16-view {rb})\nprogress: {pr}"
        if r["arm"] != "uncontrolled":
            s += f"\nLPIPS {r['lpips_to_unc']:.3f}  E {float(r['energy']):.1f}  low-freq {r['spec_low']*100:.0f}%"
        return s

    fig, axes = plt.subplots(2, len(arms) + 1, figsize=(2.75 * (len(arms) + 1), 7.6), squeeze=False)
    for i, variant in enumerate(("full", "early_window")):
        ax = axes[i, 0]; ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(Image.open(os.path.join(out_dir, "img", "uncontrolled.png")))
        ax.set_ylabel("control on all steps" if i == 0 else f"control stops after step {a.ew_step}/{a.N}", fontsize=9.5)
        if i == 0: ax.set_title("uncontrolled", fontsize=9.5)
        ax.set_xlabel(cap(unc), fontsize=6.8)
        for j, k in enumerate(arms, start=1):
            ax = axes[i, j]; ax.set_xticks([]); ax.set_yticks([])
            r = next((r for r in rows if r["arm"] == k and r["variant"] == variant), None)
            if i == 0: ax.set_title(f"{k}\n{NOTE[k]}", fontsize=9)
            if r is None: continue
            ax.imshow(Image.open(os.path.join(out_dir, "img", r["img"]))); ax.set_xlabel(cap(r), fontsize=6.8)
    fig.suptitle(f"'{P}'  →  {' + '.join(repr(t) for t in texts)}   ·   seed {a.seed}, λ={a.lam:g}, N={a.N}   ·   "
                 f"progress: 0 = uncontrolled, 1 = scores like a real sample of the edit text", fontsize=10, y=1.0)
    fig.tight_layout()
    o1 = os.path.abspath(os.path.join(out_dir, "sheet.png")); fig.savefig(o1, dpi=110, bbox_inches="tight"); plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for k in arms:
        f = os.path.join(out_dir, f"{k}__trace.json")
        if not os.path.isfile(f): continue
        t = json.load(open(f))
        a1.plot(range(len(t["u_norm"])), t["u_norm"], "-o", ms=3, label=k)
        a2.plot(range(len(t["pickscore_at_predicted_endpoint"])),
                [np.mean(v) for v in t["pickscore_at_predicted_endpoint"]], "-o", ms=3, label=k)
    a1.set_xlabel("MPC step"); a1.set_ylabel("‖u_k‖"); a1.set_title("when control is applied"); a1.grid(alpha=.25)
    a2.set_xlabel("MPC step"); a2.set_ylabel("PickScore at the predicted endpoint (mean over texts)")
    a2.set_title("when the score moves"); a2.grid(alpha=.25); a2.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    o2 = os.path.abspath(os.path.join(out_dir, "traces.png")); fig.savefig(o2, dpi=110, bbox_inches="tight"); plt.close(fig)
    return [o1, o2]


if __name__ == "__main__":
    sys.exit(main())
