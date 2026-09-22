#!/usr/bin/env python3
"""
compare_constrained.py -- one constrained MPC-vs-HardFlow comparison, end to end.

Given a prompt, an edit prompt, a lambda and a constraint threshold, this runs
MPC and then HardFlow (each as its own run_sweep.py process, pinned to one GPU),
scores the resulting images, and writes a three-panel figure

    uncontrolled | MPC | HardFlow

with each panel's constraint score, feasibility, control energy and wall clock.

The constraint is  score(image, edit_prompt) >= threshold  with no reward, so
each method's objective is its control energy alone. That is the setting where
the two methods are most directly comparable: both must reach the same feasible
set, and the only question is what it costs.

Why a wrapper and not one run_sweep.py invocation: run_sweep's sweep axes form a
cartesian product with no per-method overrides, and MPC and HardFlow want very
different inner-loop settings (n_inner 24 / lr 1e-2 vs n_inner 40 / lr 2.5). Two
processes also match the repo's one-config-per-process convention (SUITE.md).

Why the images are re-scored rather than read from summary.csv: with --reward
none the CSV's r_unc/r_ctrl are 0 and only the controlled endpoints get an h_0,
so the uncontrolled panel has no score on disk. Scoring the three saved PNGs
with the same convention run_sweep uses (logit_scale.exp() * cos_sim, CLIP-224
preprocessing) gives all three panels one consistent number; it reproduces the
CSV's h_0 to within PNG quantization (~0.01-0.07) and both values are printed.

HardFlow's lambda floor: hardflow_image's inner ascent contracts only while
lr*reg*alpha^2 < 1 with reg = N/(2*lambda), so at the authors' lr=2.5 it raises
below lambda = N*lr/2. --hf_lr_alpha is passed through to run_sweep, which
substitutes lr = alpha/reg ONLY when the floor would be hit and leaves lr alone
otherwise. The default 0.1 is what worked at N=8, lambda=10 (0.5 sat exactly on
the marginal-stability boundary and blew up on the final step). The log says
which lr was used.

Usage (from fmocc_core/, inside the uv or conda env):

  python compare_constrained.py \
      --prompt "a portrait of a man" --edit_prompt "a smiling face" \
      --lam 10 --threshold 23.0 --gpu 0 --out_dir results/cc_smile

  # override per-method or shared knobs; anything else via --mpc_extra/--hf_extra
  python compare_constrained.py ... --N 10 --mpc_H 4 --hf_n_inner 60 \
      --mpc_extra "--sweep_dual_adaptive true"

  # re-draw the figure from finished runs without re-running anything
  python compare_constrained.py ... --out_dir results/cc_smile --fig_only
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
FLOW_MATCHING_DIR = os.environ.get("FLOW_MATCHING_DIR", "/home/shervink/flow_matching")
FMRG_DIR = os.environ.get("FMRG_DIR", os.path.join(FLOW_MATCHING_DIR, "fmrg"))

ARMS = ("mpc", "hardflow")
ARM_LABEL = {"mpc": "MPC", "hardflow": "HardFlow"}

# Only PickScore has a standalone scorer below. The runs themselves accept any
# run_sweep constraint type; extend _score_images before opening this up.
SCORABLE_TYPES = ("pickscore",)


# ── argument parsing ────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.split("\n\n")[1:]),
    )
    # the four inputs
    p.add_argument("--prompt", required=True, help="generation prompt")
    p.add_argument("--edit_prompt", required=True,
                   help="constraint target text (also recorded as run_sweep's edit_prompt)")
    p.add_argument("--lam", required=True, type=float,
                   help="lambda, shared by both arms. HardFlow's reg = N/(2*lam) is derived from it.")
    p.add_argument("--threshold", required=True, type=float,
                   help="constraint threshold: score(image, edit_prompt) >= threshold")

    # where / on what
    p.add_argument("--out_dir", default=None,
                   help="results root; <out_dir>/mpc, <out_dir>/hardflow, <out_dir>/compare.png "
                        "(default: results/cc_<slug>_lam<lam>_thr<thr>)")
    p.add_argument("--gpu", type=int, default=0,
                   help="physical GPU index. Pinned via CUDA_VISIBLE_DEVICES; both arms see it "
                        "as cuda:0 in-process, which is the only pinning that works on this box.")

    # shared knobs (per-method N overrides below win)
    p.add_argument("--N", type=int, default=8, help="capital N for both arms (default 8)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr_dual", type=float, default=0.2)
    p.add_argument("--mu_init", type=float, default=0.0)
    p.add_argument("--constraint_type", default="pickscore",
                   help=f"run_sweep constraint type (default pickscore; scorable: {SCORABLE_TYPES})")
    p.add_argument("--direction", default="ge", choices=["ge", "le"])

    # MPC knobs
    p.add_argument("--mpc_N", type=int, default=None, help="override --N for MPC only")
    p.add_argument("--mpc_H", type=int, default=2)
    p.add_argument("--mpc_n_inner", type=int, default=24)
    p.add_argument("--mpc_lr_inner", type=float, default=1e-2)
    p.add_argument("--mpc_endpoint_surrogate", default="jump", choices=["rollout", "jump"])
    p.add_argument("--mpc_drift_detach", default="true", choices=["true", "false"])
    p.add_argument("--mpc_extra", default="",
                   help="extra run_sweep args for the MPC arm, one quoted string "
                        "(e.g. \"--sweep_dual_adaptive true\")")

    # HardFlow knobs
    p.add_argument("--hf_N", type=int, default=None, help="override --N for HardFlow only")
    p.add_argument("--hf_n_inner", type=int, default=40)
    p.add_argument("--hf_lr_inner", type=float, default=2.5)
    p.add_argument("--hf_lr_alpha", type=float, default=0.1,
                   help="passed as --hardflow_lr_alpha; only takes effect below HardFlow's "
                        "lambda floor (lr*reg >= 1). Pass a negative value to omit it.")
    p.add_argument("--hf_extra", default="",
                   help="extra run_sweep args for the HardFlow arm, one quoted string")

    # control flow
    p.add_argument("--skip_existing", action="store_true",
                   help="skip an arm whose summary.csv already holds a run_ok=True row")
    p.add_argument("--fig_only", action="store_true",
                   help="run nothing; score and draw from what is in --out_dir")
    p.add_argument("--quiet", action="store_true",
                   help="pass --quiet to run_sweep (hides its progress bars; costs nothing either way)")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args(argv)


def _slug(s: str, n: int = 40) -> str:
    return "".join(c if c.isalnum() else "_" for c in s.strip().lower())[:n].strip("_")


# ── building and running the two arms ───────────────────────────────────────

def _common_args(a: argparse.Namespace) -> List[str]:
    return [
        "--sweep_prompt", a.prompt,
        "--sweep_edit_prompt", a.edit_prompt,
        "--sweep_lam", str(a.lam),
        "--reward", "none",
        "--constraint_types", a.constraint_type,
        "--constraint_args", a.edit_prompt,
        "--constraint_thresholds", str(a.threshold),
        "--constraint_directions", a.direction,
        "--lr_dual", str(a.lr_dual),
        "--mu_init", str(a.mu_init),
        "--flux_device", "cuda:0",
        "--reward_device", "cuda:0",
        "--seed", str(a.seed),
    ] + (["--quiet"] if a.quiet else [])


def _arm_args(a: argparse.Namespace, arm: str) -> List[str]:
    if arm == "mpc":
        args = [
            "--sweep_control_type", "mpc",
            "--sweep_N", str(a.mpc_N if a.mpc_N is not None else a.N),
            "--sweep_H", str(a.mpc_H),
            "--sweep_n_inner", str(a.mpc_n_inner),
            "--sweep_lr_inner", str(a.mpc_lr_inner),
            "--sweep_endpoint_surrogate", a.mpc_endpoint_surrogate,
            "--sweep_drift_detach", a.mpc_drift_detach,
        ] + shlex.split(a.mpc_extra)
    else:
        args = [
            "--sweep_control_type", "hardflow",
            "--sweep_N", str(a.hf_N if a.hf_N is not None else a.N),
            "--sweep_n_inner", str(a.hf_n_inner),
            "--sweep_lr_inner", str(a.hf_lr_inner),
        ]
        if a.hf_lr_alpha is not None and a.hf_lr_alpha >= 0:
            args += ["--hardflow_lr_alpha", str(a.hf_lr_alpha)]
        args += shlex.split(a.hf_extra)
    return args


def _read_rows(arm_dir: str) -> List[Dict[str, str]]:
    path = os.path.join(arm_dir, "summary.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _ok_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r for r in rows if str(r.get("run_ok", "")).strip().lower() == "true"]


def _run_arm(a: argparse.Namespace, arm: str, arm_dir: str) -> float:
    """Run one arm as a subprocess pinned to --gpu; returns wall-clock seconds."""
    os.makedirs(arm_dir, exist_ok=True)
    cmd = [sys.executable, "-u", os.path.join(HERE, "run_sweep.py")] \
        + _common_args(a) + _arm_args(a, arm) + ["--out_dir", arm_dir]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
    log_path = os.path.join(arm_dir, "run.log")

    print(f"\n{'=' * 72}\n[{ARM_LABEL[arm]}]  GPU {a.gpu}  ->  {arm_dir}\n"
          f"  CUDA_VISIBLE_DEVICES={a.gpu} python " + " ".join(shlex.quote(c) for c in cmd[1:]) + "\n"
          f"{'=' * 72}", flush=True)

    t0 = time.time()
    # Stream to the terminal AND a log file. run_sweep's tqdm bars use \r, so
    # copy raw chunks rather than lines to keep them live.
    with open(log_path, "wb") as log, subprocess.Popen(
        cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) as proc:
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read(4096), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            log.write(chunk)
        rc = proc.wait()
    dt = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"{ARM_LABEL[arm]} run_sweep exited {rc}; see {log_path}")
    return dt


def _load_result(arm: str, arm_dir: str) -> Dict[str, object]:
    rows = _read_rows(arm_dir)
    if not rows:
        raise RuntimeError(f"{ARM_LABEL[arm]}: no summary.csv in {arm_dir}")
    ok = _ok_rows(rows)
    if not ok:
        last = rows[-1]
        raise RuntimeError(
            f"{ARM_LABEL[arm]}: no run_ok=True row in {arm_dir}/summary.csv. "
            f"Last row: {last.get('error_type', '')}: {last.get('error_msg', '')[:300]}"
        )
    if len(ok) > 1:
        print(f"  NOTE: {len(ok)} run_ok rows in {arm_dir}; using the last one.")
    r = ok[-1]
    ctrl = glob.glob(os.path.join(arm_dir, "*__img_ctrl.png"))
    unc = glob.glob(os.path.join(arm_dir, "*__img_unc.png"))
    if len(ctrl) != 1 or len(unc) != 1:
        raise RuntimeError(
            f"{ARM_LABEL[arm]}: expected exactly one img_ctrl/img_unc pair in {arm_dir}, "
            f"found {len(ctrl)}/{len(unc)}. Use a fresh --out_dir per comparison."
        )
    return {
        "arm": arm, "dir": arm_dir, "row": r,
        "img_ctrl": ctrl[0], "img_unc": unc[0],
        "h_csv": float(r["h_0"]),
        "energy": float(r["ctrl_energy"]),
        "ctrl_norm_max": float(r["ctrl_norm_max"]),
        "time_s": float(r["time_ctrl_s"]),
        "N": int(r["N"]), "H": int(r["H"]),
        "n_inner": int(r["n_inner"]), "lr_inner": float(r["lr_inner"]),
        "hardflow_reg": r.get("hardflow_reg", ""),
    }


def _md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ── scoring the saved images ────────────────────────────────────────────────

def _score_images(paths: Dict[str, str], prompt: str, ctype: str, gpu: int) -> Dict[str, float]:
    """Same convention as run_sweep._build_reward_fn's pickscore branch:
    logit_scale.exp() * cos(img_feat, txt_feat) on CLIP-224 preprocessed [0,1] images."""
    if ctype not in SCORABLE_TYPES:
        raise SystemExit(
            f"--constraint_type {ctype!r} ran fine, but only {SCORABLE_TYPES} can be re-scored "
            f"for the figure. Extend _score_images or read h_0 from the CSVs."
        )
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    sys.path.insert(0, FMRG_DIR)
    from fluxfm_sampler_reward import PickScoreReward, create_clip_preprocess  # type: ignore
    from transformers import AutoProcessor

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    obj = PickScoreReward(device=dev, dtype=torch.float32, tokenizer=tok)
    pre = create_clip_preprocess(224)
    with torch.no_grad():
        t = tok(prompt, return_tensors="pt", padding=True, max_length=77, truncation=True).to(dev)
        tf = obj.model.get_text_features(**t)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        out = {}
        for k, path in paths.items():
            x = TF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(dev)
            f = obj.model.get_image_features(pre(x))
            f = f / f.norm(dim=-1, keepdim=True)
            out[k] = float((obj.model.logit_scale.exp() * (f @ tf.T)).item())
    del obj
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


# ── the figure ──────────────────────────────────────────────────────────────

def _feasible(score: float, thr: float, direction: str) -> bool:
    return score >= thr if direction == "ge" else score <= thr


def _draw(a: argparse.Namespace, res: Dict[str, Dict[str, object]],
          scores: Dict[str, float], out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    thr, d = a.threshold, a.direction
    sign = "\\geq" if d == "ge" else "\\leq"
    mpc, hf = res["mpc"], res["hardflow"]
    hf_lr_note = ""
    if hf["hardflow_reg"] not in ("", None):
        try:
            reg = float(hf["hardflow_reg"])
            if a.hf_lr_alpha is not None and a.hf_lr_alpha >= 0 and hf["lr_inner"] * reg >= 1.0:
                hf_lr_note = f", lr·reg={a.hf_lr_alpha:g}"
            else:
                hf_lr_note = f", lr={hf['lr_inner']:g}"
        except ValueError:
            pass

    panels = [
        ("Uncontrolled", mpc["img_unc"], scores["uncontrolled"], None, None),
        (f"MPC  (N={mpc['N']}, H={mpc['H']}, n_inner={mpc['n_inner']})",
         mpc["img_ctrl"], scores["mpc"], mpc["energy"], mpc["time_s"]),
        (f"HardFlow  (N={hf['N']}, n_inner={hf['n_inner']}{hf_lr_note})",
         hf["img_ctrl"], scores["hardflow"], hf["energy"], hf["time_s"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.0))
    for ax, (name, path, score, energy, secs) in zip(axes, panels):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_xticks([]); ax.set_yticks([])
        feas = _feasible(score, thr, d)
        edge = "#2e7d32" if feas else ("#b0b0b0" if energy is None else "#c62828")
        for sp in ax.spines.values():
            sp.set_linewidth(2.4); sp.set_color(edge)
        ax.set_title(name, fontsize=11.5, pad=8)
        h = thr - score if d == "ge" else score - thr
        lines = [f"{a.constraint_type} = {score:.2f}   (h = {h:+.2f})",
                 "feasible" if feas else "infeasible"]
        if energy is not None:
            lines.append(f"energy = {energy:.2f}    {secs:.0f} s")
        ax.set_xlabel("\n".join(lines), fontsize=10.5, labelpad=9,
                      color="#2e7d32" if feas else "#444444")

    fig.suptitle(
        f"{a.constraint_type} constraint  ${sign}$ {thr:g}  ·  '{a.prompt}' $\\rightarrow$ '{a.edit_prompt}'\n"
        f"no reward, $\\lambda$={a.lam:g}, seed {a.seed}  —  objective is control energy alone",
        fontsize=12.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    a = _parse_args(argv)
    if a.out_dir is None:
        a.out_dir = os.path.join(
            HERE, "results",
            f"cc_{_slug(a.prompt)}__{_slug(a.edit_prompt)}__lam{a.lam:g}_thr{a.threshold:g}")
    a.out_dir = os.path.abspath(a.out_dir)
    os.makedirs(a.out_dir, exist_ok=True)
    arm_dirs = {arm: os.path.join(a.out_dir, arm) for arm in ARMS}

    wall = {}
    if not a.fig_only:
        for arm in ARMS:
            if a.skip_existing and _ok_rows(_read_rows(arm_dirs[arm])):
                print(f"[{ARM_LABEL[arm]}] run_ok row already in {arm_dirs[arm]} -- skipping.")
                continue
            wall[arm] = _run_arm(a, arm, arm_dirs[arm])

    res = {arm: _load_result(arm, arm_dirs[arm]) for arm in ARMS}

    # Same seed and N must give the same uncontrolled image, else the two arms
    # did not control the same problem and the figure would be misleading.
    same_unc = _md5(res["mpc"]["img_unc"]) == _md5(res["hardflow"]["img_unc"])
    if not same_unc:
        print("  WARNING: the two arms' uncontrolled images differ (different N or seed?). "
              "Panels are NOT a like-for-like comparison; the uncontrolled panel is MPC's.")

    print("\nScoring the three images ...", flush=True)
    scores = _score_images(
        {"uncontrolled": res["mpc"]["img_unc"],
         "mpc": res["mpc"]["img_ctrl"],
         "hardflow": res["hardflow"]["img_ctrl"]},
        a.edit_prompt, a.constraint_type, a.gpu)

    out_png = os.path.join(a.out_dir, "compare.png")
    _draw(a, res, scores, out_png)

    # summary: terminal table + json
    thr, d = a.threshold, a.direction
    def h_of(s: float) -> float:
        return thr - s if d == "ge" else s - thr
    print(f"\n{'':14s}{'score':>9s}{'h(png)':>10s}{'h(csv)':>10s}{'feas':>7s}"
          f"{'energy':>10s}{'|u|max':>9s}{'time_s':>9s}")
    print(f"{'uncontrolled':14s}{scores['uncontrolled']:9.3f}{h_of(scores['uncontrolled']):+10.3f}"
          f"{'':>10s}{str(_feasible(scores['uncontrolled'], thr, d)):>7s}")
    for arm in ARMS:
        r = res[arm]
        print(f"{ARM_LABEL[arm]:14s}{scores[arm]:9.3f}{h_of(scores[arm]):+10.3f}{r['h_csv']:+10.3f}"
              f"{str(_feasible(scores[arm], thr, d)):>7s}{r['energy']:10.3f}"
              f"{r['ctrl_norm_max']:9.3f}{r['time_s']:9.1f}")
    summary = {
        "args": vars(a), "same_uncontrolled_image": same_unc,
        "scores_png": scores,
        "arms": {arm: {k: v for k, v in res[arm].items() if k != "row"} for arm in ARMS},
        "wall_s": wall, "figure": out_png,
    }
    with open(os.path.join(a.out_dir, "comparison.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nfigure : {out_png}\nsummary: {os.path.join(a.out_dir, 'comparison.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
