#!/usr/bin/env python3
"""
compare_constrained.py -- constrained comparisons (ConicFlow vs HardFlow, plus
any further constrained arm), end to end.

Given a prompt, an edit prompt, a lambda and one or more constraints, this runs
each arm (each as its own run_sweep.py process, pinned to one GPU), scores the
resulting images, and writes a figure

    uncontrolled | <arm 1> | <arm 2> | ...        (default arms: conicflow, hardflow)

with every constraint's score and h per panel, feasibility (all h <= 0), and
control energy (and, with --show_time, wall clock).

ARMS. --arms lists the methods to run, in panel order (default: conicflow
hardflow, the pair in the paper's Figure 3). The constrained baselines are two
more arms: `pflow` (Projected Flow, PDM-style projection; --pflow_* flags) and
`ldf` (Lagrangian Dual Flows; --ldf_* flags, initial multiplier --ldf_mu_init).
Add them to an existing suite tree with `--arms pflow ldf --skip_existing`.
--dry_run prints every run_sweep.py command it would launch and runs nothing
(with --auto_threshold the thresholds are shown symbolically, since they depend
on the probe's scores).

There is no reward, so each method's objective is its control energy alone.
That is the setting where the methods are most directly comparable: all must
reach the same feasible set, and the only question is what it costs.

CONSTRAINTS. Parallel lists, exactly as run_sweep takes them, shared by every
arm:

    --constraint_types pickscore clip --constraint_args "a smiling face" "with glasses" \
    --constraint_thresholds 24.0 0.31  --constraint_directions ge ge

--constraint_args defaults to --edit_prompt for every constraint and
--constraint_directions to "ge". The single-constraint spellings
--constraint_type / --threshold / --direction work as aliases.

AUTO THRESHOLDS. --auto_threshold replaces --constraint_thresholds with a grid
derived from the uncontrolled image of THIS seed: it runs one uncontrolled
sample, scores it with every constraint (s0_i), and lays n_grid multipliers m
from --auto_lo to --auto_hi. Point k tightens every constraint together:

    ge:  threshold_i = s0_i + (m_k - 1) * |s0_i|      (= m_k * s0_i when s0_i > 0)
    le:  threshold_i = s0_i - (m_k - 1) * |s0_i|

Tightening by |s0| rather than multiplying keeps the step a TIGHTENING when the
baseline is negative (ImageReward's z-score often is) or the direction is "le",
where a plain multiple would loosen the constraint. Each point runs in
<out_dir>/m<mult>/, so the same directory names line up across seeds even
though the absolute thresholds differ; <out_dir>/auto_threshold.json records
the actual values. This is the layout scripts/constrained/run_suite.sh
produces and plot_constrained_suite.py reads.

DUAL VARIABLES. Each arm has its own multiplier settings, because the methods
use them differently. ConicFlow takes one projected dual-ascent step per
sampling step and its multipliers persist (--lr_dual, --mu_init). HardFlow
updates every `dual_every` inner steps and resets to mu_init at every sampling
step, as in its released code (--hf_lr_dual, --hf_mu_init). The defaults are
the values used for the paper's constrained experiments.

Why a wrapper and not one run_sweep.py invocation: run_sweep's sweep axes form a
cartesian product with no per-method overrides, and the arms want very
different inner-loop settings (ConicFlow n_inner 24 / lr 1e-2 vs HardFlow
n_inner 40 / lr 0.5) and different dual settings. One process per arm also
keeps one config per process, so an OOM or a divergence in one arm cannot take
another down.

Why the images are re-scored rather than read from summary.csv: with --reward
none the CSV's r_unc/r_ctrl are 0 and only the controlled endpoints get an h_i,
so the uncontrolled panel has no score on disk. The scorer here calls
run_sweep._build_reward_fn itself, feeding it the saved PNG through the
decode_fn hook every branch already routes through, so every panel is scored by
exactly the code the control loop enforced -- for any constraint type. It
reproduces the CSV's h to within PNG quantization (~0.01-0.07) and both values
are printed.

HardFlow stability. hardflow_image's inner ascent is an explicit fixed-step
scheme whose deviation error contracts by (1 - 2*lr*reg*alpha^2) per step,
with reg = N/(2*lambda) and alpha -> 1 at the final sampling step. It raises
at lr*reg >= 1, but it is already marginal well below that: at N=8, lambda=20
the authors' lr=2.5 gives lr*reg = 0.5 (damping factor 0 on the last step) and
diverged on every threshold tried, while lr=0.5 (lr*reg = 0.1) was stable.
Hence the default --hf_lr_inner 0.5. Separately, the released code doubles the
augmented-Lagrangian penalty rho at every dual update that still sees a
violation, which over 40 inner steps can take rho from 5 to 160 inside ONE
sampling step and dominated the failures at tight thresholds; --hf_rho_mult 1.0
(the default) holds rho fixed. --hf_lr_alpha is forwarded as
--hardflow_lr_alpha and only fires when lr*reg would reach 1.

Usage (from image/src/, inside the environment):

  python compare_constrained.py \
      --prompt "a portrait of a man" --edit_prompt "a smiling face" \
      --lam 20 --threshold 24.0 --gpu 0 --out_dir results/cc_smile

  # two constraints, auto grid of 5 points from 1.1x to 1.5x the uncontrolled score
  python compare_constrained.py ... --constraint_types pickscore pickscore \
      --constraint_args "man with curly hair" "photo of very old man" \
      --auto_threshold --auto_lo 1.1 --auto_hi 1.5 --n_grid 5

  # re-draw from finished runs without re-running anything
  python compare_constrained.py ... --out_dir results/cc_smile --fig_only

  # print the commands only
  python compare_constrained.py ... --dry_run
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
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))

# One entry per constrained arm: run_sweep control type (= output directory
# name) -> figure label. A new constrained baseline is added here and in
# _arm_args / _arm_N / _arm_caption; --arms only accepts these keys.
ARM_LABEL = {"conicflow": "ConicFlow", "hardflow": "HardFlow",
             # constrained baselines (projected_flow_image / ldf_image)
             "pflow": "Projected Flow", "ldf": "LDF"}
DEFAULT_ARMS = ("conicflow", "hardflow")
PROBE_SUBDIR = "uncontrolled"        # <out_dir>/uncontrolled/ holds the auto-threshold probe
MANIFEST = "auto_threshold.json"
CONSTRAINT_TYPES = ("clip", "hpsv2", "pickscore", "imagereward", "style", "lpips")


@dataclass
class Constraint:
    ctype: str
    arg: str
    direction: str          # "ge": score >= thr satisfied; "le": score <= thr

    def h(self, score: float, thr: float) -> float:
        return thr - score if self.direction == "ge" else score - thr

    def tighten(self, s0: float, m: float) -> float:
        """Threshold at multiplier m, stepping from the uncontrolled score s0 in
        the tightening direction by (m-1)*|s0|."""
        step = (m - 1.0) * abs(s0)
        return s0 + step if self.direction == "ge" else s0 - step

    @property
    def sign(self) -> str:
        return "\\geq" if self.direction == "ge" else "\\leq"


# ── argument parsing ────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.split("\n\n")[1:]),
    )
    p.add_argument("--prompt", required=True, help="generation prompt")
    p.add_argument("--edit_prompt", required=True,
                   help="default constraint target text (also recorded as run_sweep's edit_prompt)")
    p.add_argument("--lam", required=True, type=float,
                   help="lambda, shared by every arm. HardFlow's reg = N/(2*lam) is derived from it.")
    p.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=list(ARM_LABEL),
                   metavar="ARM",
                   help=f"constrained arms to run, in panel order, from {list(ARM_LABEL)} "
                        f"(default: {' '.join(DEFAULT_ARMS)}). The first arm's N sets the probe "
                        f"and the uncontrolled panel.")

    # constraints: parallel lists, run_sweep's convention; scalar names kept as aliases
    p.add_argument("--constraint_types", "--constraint_type", nargs="+", default=["pickscore"],
                   metavar="TYPE", help=f"one per constraint, from {CONSTRAINT_TYPES} (default: pickscore)")
    p.add_argument("--constraint_args", nargs="+", default=None, metavar="ARG",
                   help="one per constraint: text target for clip/hpsv2/pickscore/imagereward, a "
                        "reference image path for style, '' for lpips. Default: --edit_prompt for each.")
    p.add_argument("--constraint_thresholds", "--threshold", nargs="+", type=float, default=None,
                   metavar="THR", help="one per constraint. Required unless --auto_threshold.")
    p.add_argument("--constraint_directions", "--direction", nargs="+", default=None,
                   choices=["ge", "le"], metavar="DIR", help="one per constraint (default: ge for each)")

    # automatic threshold grid
    p.add_argument("--auto_threshold", action="store_true",
                   help="derive a grid of thresholds from this seed's uncontrolled image and run "
                        "every point (see the docstring). Replaces --constraint_thresholds.")
    p.add_argument("--auto_lo", type=float, default=1.1, help="first multiplier (default 1.1)")
    p.add_argument("--auto_hi", type=float, default=1.5, help="last multiplier (default 1.5)")
    p.add_argument("--n_grid", type=int, default=5, help="number of grid points (default 5)")

    # where / on what
    p.add_argument("--out_dir", default=None,
                   help="results root. Single comparison: <out_dir>/{<arm>...,compare.png}. "
                        "Auto grid: <out_dir>/uncontrolled/, <out_dir>/m<mult>/..., "
                        "<out_dir>/auto_threshold.json")
    p.add_argument("--gpu", type=int, default=0,
                   help="physical GPU index. Pinned via CUDA_VISIBLE_DEVICES; everything sees it "
                        "as cuda:0 in-process (see the README on cross-device copies).")

    # shared knobs (per-method N overrides below win)
    p.add_argument("--N", type=int, default=8, help="capital N for every arm (default 8)")
    p.add_argument("--seed", type=int, default=42)

    # ConicFlow knobs
    p.add_argument("--conicflow_N", dest="conicflow_N", type=int, default=None,
                   help="override --N for ConicFlow only")
    p.add_argument("--conicflow_H", dest="conicflow_H", type=int, default=2)
    p.add_argument("--conicflow_n_inner", dest="conicflow_n_inner", type=int, default=24)
    p.add_argument("--conicflow_lr_inner", dest="conicflow_lr_inner", type=float, default=1e-2)
    p.add_argument("--conicflow_endpoint_surrogate",
                   dest="conicflow_endpoint_surrogate", default="jump", choices=["rollout", "jump"])
    p.add_argument("--conicflow_drift_detach", dest="conicflow_drift_detach",
                   default="true", choices=["true", "false"])
    p.add_argument("--lr_dual", type=float, default=0.5,
                   help="ConicFlow dual-ascent step (one projected step per sampling step)")
    p.add_argument("--mu_init", type=float, default=0.5,
                   help="ConicFlow initial multiplier for every constraint")
    p.add_argument("--conicflow_extra", dest="conicflow_extra", default="",
                   help="extra run_sweep args for the ConicFlow arm, one quoted string "
                        "(e.g. \"--sweep_dual_adaptive true\")")

    # HardFlow knobs
    p.add_argument("--hf_N", type=int, default=None, help="override --N for HardFlow only")
    p.add_argument("--hf_n_inner", type=int, default=40)
    p.add_argument("--hf_lr_inner", type=float, default=0.5,
                   help="inner ascent step. The authors' 2.5 gives lr*reg = 0.5 at N=8, lam=20 and "
                        "diverges; 0.5 (lr*reg = 0.1) is stable. See the docstring.")
    p.add_argument("--hf_lr_alpha", type=float, default=0.1,
                   help="passed as --hardflow_lr_alpha; only takes effect where lr*reg would reach 1. "
                        "Pass a negative value to omit it.")
    p.add_argument("--hf_lr_dual", type=float, default=0.2,
                   help="HardFlow dual step, applied every hardflow_dual_every inner steps")
    p.add_argument("--hf_mu_init", type=float, default=0.0,
                   help="HardFlow multiplier value at the start of EVERY sampling step (they reset)")
    p.add_argument("--hf_rho_mult", type=float, default=1.0,
                   help="growth factor of the AL penalty rho at a violating dual update. The released "
                        "code uses 2.0; 1.0 holds rho fixed, which is what keeps HardFlow stable at "
                        "tight thresholds.")
    p.add_argument("--hf_extra", default="",
                   help="extra run_sweep args for the HardFlow arm, one quoted string")

    # Projected Flow knobs (2026-10-09; run_sweep's pflow_* fields). No multipliers.
    p.add_argument("--pflow_max_iters", type=int, default=10,
                   help="Gauss-Newton updates per projection (cap)")
    p.add_argument("--pflow_final_iters", type=int, default=50,
                   help="Gauss-Newton cap for the last step's projection (decoder+scorer only)")
    p.add_argument("--pflow_activation", type=float, default=0.0,
                   help="fraction of steps before projecting starts (0 = every step, as PDM)")
    p.add_argument("--pflow_margin", type=float, default=0.0, help="project onto h + margin <= 0")
    p.add_argument("--pflow_extra", default="",
                   help="extra run_sweep args for the Projected Flow arm, one quoted string")

    # LDF knobs (2026-10-09; run_sweep's ldf_* fields)
    p.add_argument("--ldf_penalty", type=float, default=5.0, help="LDF penalty_weight (their default)")
    p.add_argument("--ldf_rescale", type=float, default=1.0,
                   help="multiplier gain; 0 = pure quadratic penalty")
    p.add_argument("--ldf_p", type=float, default=1.0, help="exponent of the 1/(1-s)^p gain")
    p.add_argument("--ldf_substeps", type=int, default=64,
                   help="substeps per sampling step (64 since 2026-10-11; see ldf_image)")
    p.add_argument("--ldf_mu_init", type=float, default=0.0,
                   help="LDF's initial multiplier (their code starts at 0)")
    p.add_argument("--ldf_extra", default="",
                   help="extra run_sweep args for the LDF arm, one quoted string")

    # control flow
    p.add_argument("--skip_existing", action="store_true",
                   help="skip an arm (or the auto-threshold probe) that already has a run_ok=True row")
    p.add_argument("--fig_only", action="store_true",
                   help="run nothing; score and draw from what is in --out_dir")
    p.add_argument("--dry_run", action="store_true",
                   help="print every run_sweep.py command that would be launched, then exit "
                        "without running, scoring or drawing anything")
    p.add_argument("--quiet", action="store_true",
                   help="pass --quiet to run_sweep (hides its progress bars; costs nothing either way)")
    p.add_argument("--show_time", action="store_true",
                   help="add each arm's wall-clock time to its figure caption (off by default; "
                        "the terminal table and comparison.json always record it)")
    p.add_argument("--dpi", type=int, default=150)
    a = p.parse_args(argv)

    if len(set(a.arms)) != len(a.arms):
        p.error(f"--arms lists an arm twice: {a.arms}")
    if a.dry_run and a.fig_only:
        p.error("--dry_run and --fig_only are mutually exclusive.")

    # ── resolve the constraint lists ────────────────────────────────────────
    n = len(a.constraint_types)
    for t in a.constraint_types:
        if t not in CONSTRAINT_TYPES:
            p.error(f"--constraint_types: {t!r} is not one of {CONSTRAINT_TYPES}")
    if a.constraint_args is None:
        a.constraint_args = [a.edit_prompt] * n
    if a.constraint_directions is None:
        a.constraint_directions = ["ge"] * n
    for name, seq in (("--constraint_args", a.constraint_args),
                      ("--constraint_directions", a.constraint_directions)):
        if len(seq) != n:
            p.error(f"{n} --constraint_types but {len(seq)} {name}: every constraint needs its "
                    f"own entry (nothing is broadcast, matching run_sweep).")
    if a.auto_threshold:
        if a.constraint_thresholds is not None:
            p.error("--auto_threshold and --constraint_thresholds are mutually exclusive.")
        if a.n_grid < 1 or a.auto_lo <= 0 or a.auto_hi < a.auto_lo:
            p.error("--auto_threshold needs n_grid >= 1 and 0 < auto_lo <= auto_hi.")
    else:
        if a.constraint_thresholds is None:
            p.error("--constraint_thresholds (or --threshold) is required unless --auto_threshold.")
        if len(a.constraint_thresholds) != n:
            p.error(f"{n} --constraint_types but {len(a.constraint_thresholds)} --constraint_thresholds.")
    a.constraints = [Constraint(t, g, d) for t, g, d
                     in zip(a.constraint_types, a.constraint_args, a.constraint_directions)]
    return a


def _slug(s: str, n: int = 40) -> str:
    return "".join(c if c.isalnum() else "_" for c in s.strip().lower())[:n].strip("_")


def _labels(cons: Sequence[Constraint]) -> List[str]:
    """Panel labels: the type alone when unique, else type plus a short arg."""
    counts = {}
    for c in cons:
        counts[c.ctype] = counts.get(c.ctype, 0) + 1
    out = []
    for c in cons:
        if counts[c.ctype] == 1:
            out.append(c.ctype)
        else:
            short = (c.arg[:14] + "…") if len(c.arg) > 15 else c.arg
            out.append(f"{c.ctype}:'{short}'")
    return out


# ── per-arm settings ────────────────────────────────────────────────────────

def _arm_N(a: argparse.Namespace, arm: str) -> int:
    override = {"conicflow": a.conicflow_N, "hardflow": a.hf_N}.get(arm)   # no per-arm N for the rest
    return override if override is not None else a.N


def _arm_args(a: argparse.Namespace, arm: str) -> List[str]:
    if arm == "conicflow":
        return [
            "--sweep_control_type", "conicflow",
            "--sweep_N", str(_arm_N(a, arm)),
            "--sweep_H", str(a.conicflow_H),
            "--sweep_n_inner", str(a.conicflow_n_inner),
            "--sweep_lr_inner", str(a.conicflow_lr_inner),
            "--sweep_endpoint_surrogate", a.conicflow_endpoint_surrogate,
            "--sweep_drift_detach", a.conicflow_drift_detach,
            "--lr_dual", str(a.lr_dual),
            "--mu_init", str(a.mu_init),
        ] + shlex.split(a.conicflow_extra)
    if arm == "hardflow":
        args = [
            "--sweep_control_type", "hardflow",
            "--sweep_N", str(_arm_N(a, arm)),
            "--sweep_n_inner", str(a.hf_n_inner),
            "--sweep_lr_inner", str(a.hf_lr_inner),
            "--lr_dual", str(a.hf_lr_dual),
            "--mu_init", str(a.hf_mu_init),
            "--sweep_hardflow_rho_mult", str(a.hf_rho_mult),
        ]
        if a.hf_lr_alpha is not None and a.hf_lr_alpha >= 0:
            args += ["--hardflow_lr_alpha", str(a.hf_lr_alpha)]
        return args + shlex.split(a.hf_extra)
    if arm == "pflow":
        # No --lr_dual / --mu_init: Projected Flow has no multipliers, and
        # run_sweep exempts it from the mu*h no-op guard.
        return [
            "--sweep_control_type", "pflow",
            "--sweep_N", str(_arm_N(a, arm)),
            "--sweep_pflow_max_iters", str(a.pflow_max_iters),
            "--sweep_pflow_final_iters", str(a.pflow_final_iters),
            "--sweep_pflow_activation", str(a.pflow_activation),
            "--sweep_pflow_margin", str(a.pflow_margin),
        ] + shlex.split(a.pflow_extra)
    if arm == "ldf":
        return [
            "--sweep_control_type", "ldf",
            "--sweep_N", str(_arm_N(a, arm)),
            "--sweep_ldf_penalty", str(a.ldf_penalty),
            "--sweep_ldf_rescale", str(a.ldf_rescale),
            "--sweep_ldf_p", str(a.ldf_p),
            "--sweep_ldf_substeps", str(a.ldf_substeps),
            "--mu_init", str(a.ldf_mu_init),
        ] + shlex.split(a.ldf_extra)
    raise ValueError(f"_arm_args: no settings for arm {arm!r}; add it next to conicflow/hardflow.")


# ── subprocess plumbing ─────────────────────────────────────────────────────

def _cmd(sweep_args: List[str], out_dir: str) -> List[str]:
    return [sys.executable, "-u", os.path.join(HERE, "run_sweep.py")] + sweep_args + ["--out_dir", out_dir]


def _print_cmd(a: argparse.Namespace, label: str, sweep_args: List[str], out_dir: str) -> None:
    print(f"[{label}]  CUDA_VISIBLE_DEVICES={a.gpu} python "
          + " ".join(shlex.quote(c) for c in _cmd(sweep_args, out_dir)[1:]), flush=True)


def _run_subprocess(a: argparse.Namespace, label: str, sweep_args: List[str], out_dir: str) -> float:
    """One run_sweep.py process pinned to --gpu, streamed to the terminal and
    <out_dir>/run.log. Returns wall-clock seconds; raises on non-zero exit."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = _cmd(sweep_args, out_dir)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
    log_path = os.path.join(out_dir, "run.log")
    print(f"\n{'=' * 72}\n[{label}]  GPU {a.gpu}  ->  {out_dir}\n"
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
        raise RuntimeError(f"{label}: run_sweep exited {rc}; see {log_path}")
    return dt


def _common_args(a: argparse.Namespace, thresholds: Sequence[object]) -> List[str]:
    cons = a.constraints
    return [
        "--sweep_prompt", a.prompt,
        "--sweep_edit_prompt", a.edit_prompt,
        "--sweep_lam", str(a.lam),
        "--reward", "none",
        "--constraint_types", *[c.ctype for c in cons],
        "--constraint_args", *[c.arg for c in cons],
        "--constraint_thresholds", *[str(t) for t in thresholds],
        "--constraint_directions", *[c.direction for c in cons],
        "--flux_device", "cuda:0",
        "--reward_device", "cuda:0",
        "--seed", str(a.seed),
    ] + (["--quiet"] if a.quiet else [])


def _probe_args(a: argparse.Namespace) -> List[str]:
    """Uncontrolled sample only: control_type=none, no reward, no constraints.
    Uses the first arm's N, because the uncontrolled panel is the first arm's."""
    return [
        "--sweep_prompt", a.prompt,
        "--sweep_edit_prompt", a.edit_prompt,
        "--sweep_control_type", "none",
        "--sweep_N", str(_arm_N(a, a.arms[0])),
        "--reward", "none",
        "--flux_device", "cuda:0",
        "--reward_device", "cuda:0",
        "--seed", str(a.seed),
        "--quiet",
    ]


def _read_rows(d: str) -> List[Dict[str, str]]:
    path = os.path.join(d, "summary.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _ok_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r for r in rows if str(r.get("run_ok", "")).strip().lower() == "true"]


def _one_png(d: str, pattern: str) -> str:
    hits = glob.glob(os.path.join(d, pattern))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {pattern} in {d}, found {len(hits)}. "
                           f"Use a fresh --out_dir per comparison.")
    return hits[0]


def _load_result(arm: str, arm_dir: str, n_con: int) -> Dict[str, object]:
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
    return {
        "arm": arm, "dir": arm_dir, "row": r,
        "img_ctrl": _one_png(arm_dir, "*__img_ctrl.png"),
        "img_unc": _one_png(arm_dir, "*__img_unc.png"),
        "h_csv": [float(r[f"h_{i}"]) for i in range(n_con)],
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

class Scorer:
    """Scores saved PNGs with run_sweep's own reward builders.

    _build_reward_fn's every branch decodes through `(decode_fn or
    sampler.decode)(z)`, so handing it a decode_fn that returns the PNG as a
    [-1, 1] tensor (and a dummy z for the device bookkeeping) reuses the exact
    scorer, prompt handling and preprocessing the control loop used -- for any
    constraint type, with no parallel implementation to drift. lpips needs the
    uncontrolled image as its reference; `score` sets ref_holder["x_ref"] from
    the uncontrolled PNG each call.

    Loads every constraint's model onto --gpu. Build it only while no
    run_sweep subprocess is running (FLUX plus the run's own scorers already
    peak near 19 GB with two constraints) and call close() before launching one.
    """

    def __init__(self, a: argparse.Namespace):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
        import dataclasses
        import torch
        sys.path.insert(0, HERE)
        import run_sweep as rs                      # sets up the FMRG import path itself
        self.torch, self.rs = torch, rs
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ref_holder: dict = {}
        self._slot: dict = {"img": None}
        slot = self._slot

        def decode_fn(z):
            return slot["img"]

        base = rs.RunConfig(prompt=a.prompt, edit_prompt=a.edit_prompt, N=_arm_N(a, a.arms[0]))
        self.fns = []
        for c in a.constraints:
            cfg = base
            if c.ctype in rs._TEXT_TARGET_REWARD_TYPES:
                cfg = dataclasses.replace(base, edit_prompt=c.arg)
            elif c.ctype == "style":
                cfg = dataclasses.replace(base, style_ref_paths=[c.arg])
            r_image, _ = rs._build_reward_fn(
                None, cfg, self.dev, torch.float32, reward_model=c.ctype,
                ref_holder=self.ref_holder, decode_fn=decode_fn,
            )
            self.fns.append(r_image)
        self._dummy = torch.zeros(1, device=self.dev)

    def _load(self, path: str):
        import numpy as np
        from PIL import Image
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        t = self.torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.dev)
        return t * 2.0 - 1.0                        # sampler.decode's [-1, 1] convention

    def score(self, png: str, unc_png: str) -> List[float]:
        with self.torch.no_grad():
            self.ref_holder["x_ref"] = self._load(unc_png)
            self._slot["img"] = self._load(png)
            out = [float(f(self._dummy).reshape(-1)[0].item()) for f in self.fns]
        self._slot["img"] = None
        return out

    def close(self) -> None:
        self.fns = []
        self.ref_holder.clear()
        self._slot["img"] = None
        import gc
        gc.collect()
        if self.dev.type == "cuda":
            self.torch.cuda.empty_cache()


# ── the figure ──────────────────────────────────────────────────────────────

def _hf_lr_note(a: argparse.Namespace, hf: Dict[str, object]) -> str:
    if hf["hardflow_reg"] in ("", None):
        return ""
    try:
        reg = float(hf["hardflow_reg"])
    except ValueError:
        return ""
    if a.hf_lr_alpha is not None and a.hf_lr_alpha >= 0 and hf["lr_inner"] * reg >= 1.0:
        return f", lr·reg={a.hf_lr_alpha:g}"
    return f", lr={hf['lr_inner']:g}"


def _arm_caption(a: argparse.Namespace, arm: str, r: Dict[str, object]) -> str:
    if arm == "conicflow":
        return f"{ARM_LABEL[arm]}  (N={r['N']}, H={r['H']}, n_inner={r['n_inner']})"
    if arm == "hardflow":
        return f"{ARM_LABEL[arm]}  (N={r['N']}, n_inner={r['n_inner']}{_hf_lr_note(a, r)})"
    row = r.get("row", {})
    if arm == "pflow":
        return (f"{ARM_LABEL[arm]}  (N={r['N']}, GN cap={row.get('pflow_max_iters', '?')}/"
                f"{row.get('pflow_final_iters', '?')})")
    if arm == "ldf":
        return (f"{ARM_LABEL[arm]}  (N={r['N']}, substeps={row.get('ldf_substeps', '?')}, "
                f"w={row.get('ldf_penalty', '?')})")
    return f"{ARM_LABEL[arm]}  (N={r['N']}, n_inner={r['n_inner']})"


def _draw(a: argparse.Namespace, thresholds: Sequence[float], res: Dict[str, Dict[str, object]],
          scores: Dict[str, List[float]], out_png: str, mult: Optional[float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    cons, labels = a.constraints, _labels(a.constraints)
    first = res[a.arms[0]]
    panels = [("Uncontrolled", first["img_unc"], scores["uncontrolled"], None, None)]
    for arm in a.arms:
        r = res[arm]
        panels.append((_arm_caption(a, arm, r), r["img_ctrl"], scores[arm], r["energy"], r["time_s"]))
    n_lines = len(cons) + 2
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6 + 0.22 * n_lines))
    for ax, (name, path, sc, energy, secs) in zip(axes, panels):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_xticks([]); ax.set_yticks([])
        hs = [c.h(s, t) for c, s, t in zip(cons, sc, thresholds)]
        feas = all(h <= 0 for h in hs)
        edge = "#2e7d32" if feas else ("#b0b0b0" if energy is None else "#c62828")
        for sp in ax.spines.values():
            sp.set_linewidth(2.4); sp.set_color(edge)
        ax.set_title(name, fontsize=11.5, pad=8)
        lines = [f"{lab} = {s:.2f}   (h = {h:+.2f})" for lab, s, h in zip(labels, sc, hs)]
        lines.append("feasible" if feas else "infeasible")
        if energy is not None:
            lines.append(f"energy = {energy:.2f}" + (f"    {secs:.0f} s" if a.show_time else ""))
        ax.set_xlabel("\n".join(lines), fontsize=10.5, labelpad=9,
                      color="#2e7d32" if feas else "#444444")

    con_txt = "  &  ".join(f"{lab} ${c.sign}$ {t:.3g}" for lab, c, t in zip(labels, cons, thresholds))
    mult_txt = f"   (m = {mult:.3g})" if mult is not None else ""
    fig.suptitle(
        f"{con_txt}{mult_txt}  ·  '{a.prompt}' $\\rightarrow$ '{a.edit_prompt}'\n"
        f"no reward, $\\lambda$={a.lam:g}, seed {a.seed}  —  objective is control energy alone",
        fontsize=12.0, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)


# ── one comparison at fixed thresholds ─────────────────────────────────────

def run_arms(a: argparse.Namespace, thresholds: Sequence[object], out_dir: str) -> Dict[str, float]:
    """Run (or skip) every arm into out_dir. Returns wall seconds per arm run."""
    wall = {}
    for arm in a.arms:
        arm_dir = os.path.join(out_dir, arm)
        sweep_args = _common_args(a, thresholds) + _arm_args(a, arm)
        if a.dry_run:
            _print_cmd(a, ARM_LABEL[arm], sweep_args, arm_dir)
            continue
        if a.skip_existing and _ok_rows(_read_rows(arm_dir)):
            print(f"[{ARM_LABEL[arm]}] run_ok row already in {arm_dir} -- skipping.")
            continue
        wall[arm] = _run_subprocess(a, ARM_LABEL[arm], sweep_args, arm_dir)
    return wall


def score_and_draw(a: argparse.Namespace, thresholds: Sequence[float], out_dir: str,
                   scorer: Scorer, wall: Dict[str, float], mult: Optional[float] = None,
                   probe_unc: Optional[str] = None) -> Dict[str, object]:
    """Load every arm's result from out_dir, score the images, draw the figure,
    write comparison.json, print the table. Returns the summary."""
    cons, labels = a.constraints, _labels(a.constraints)
    res = {arm: _load_result(arm, os.path.join(out_dir, arm), len(cons)) for arm in a.arms}
    first = a.arms[0]

    # Same seed and N must give the same uncontrolled image, else the arms did
    # not control the same problem and the figure would be misleading.
    unc_md5 = {arm: _md5(res[arm]["img_unc"]) for arm in a.arms}
    same_unc = len(set(unc_md5.values())) == 1
    if not same_unc:
        print(f"  WARNING: the arms' uncontrolled images differ (different N or seed?): "
              f"{unc_md5}. Panels are NOT a like-for-like comparison; the uncontrolled panel "
              f"is {ARM_LABEL[first]}'s.")
    if probe_unc is not None and _md5(probe_unc) != unc_md5[first]:
        print(f"  WARNING: the auto-threshold probe's uncontrolled image differs from "
              f"{ARM_LABEL[first]}'s. The grid was derived from a different baseline than the "
              f"arms controlled.")

    unc = res[first]["img_unc"]
    scores = {"uncontrolled": scorer.score(unc, unc)}
    for arm in a.arms:
        scores[arm] = scorer.score(res[arm]["img_ctrl"], unc)
    h_png = {k: [c.h(s, t) for c, s, t in zip(cons, v, thresholds)] for k, v in scores.items()}
    feas = {k: all(h <= 0 for h in v) for k, v in h_png.items()}

    out_png = os.path.join(out_dir, "compare.png")
    _draw(a, thresholds, res, scores, out_png, mult)

    # terminal table: one block per constraint, then the per-arm cost columns
    print()
    for i, lab in enumerate(labels):
        print(f"  [{lab}]  {cons[i].direction} {thresholds[i]:.4g}")
        print(f"    {'':13s}{'score':>9s}{'h(png)':>10s}{'h(csv)':>10s}")
        print(f"    {'uncontrolled':13s}{scores['uncontrolled'][i]:9.3f}{h_png['uncontrolled'][i]:+10.3f}")
        for arm in a.arms:
            print(f"    {ARM_LABEL[arm]:13s}{scores[arm][i]:9.3f}{h_png[arm][i]:+10.3f}"
                  f"{res[arm]['h_csv'][i]:+10.3f}")
    print(f"\n    {'':13s}{'feasible':>9s}{'energy':>10s}{'|u|max':>9s}{'time_s':>9s}")
    print(f"    {'uncontrolled':13s}{str(feas['uncontrolled']):>9s}")
    for arm in a.arms:
        r = res[arm]
        print(f"    {ARM_LABEL[arm]:13s}{str(feas[arm]):>9s}{r['energy']:10.3f}"
              f"{r['ctrl_norm_max']:9.3f}{r['time_s']:9.1f}")

    summary = {
        "args": vars(a) | {"constraints": None},
        "constraints": [{"type": c.ctype, "arg": c.arg, "direction": c.direction, "threshold": t}
                        for c, t in zip(cons, thresholds)],
        "multiplier": mult,
        "arms_order": list(a.arms),
        "same_uncontrolled_image": same_unc,
        "scores_png": scores, "h_png": h_png, "feasible": feas,
        "arms": {arm: {k: v for k, v in res[arm].items() if k != "row"} for arm in a.arms},
        "wall_s": wall, "figure": out_png, "figure_arms": list(a.arms),
    }
    _keep_prior_arms(summary, out_dir, a.arms)
    with open(os.path.join(out_dir, "comparison.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n    figure : {out_png}")
    return summary


def _keep_prior_arms(summary: Dict[str, object], out_dir: str, arms: Sequence[str]) -> None:
    """Carry forward, from an existing comparison.json, every arm this call did
    not run, so adding arms to a finished tree (`--arms pflow ldf
    --skip_existing`) extends the record instead of replacing it. Only when the
    thresholds match exactly: a file from a different threshold set is a
    different problem and is left out, with a note. The figure shows this
    call's arms only (`figure_arms`)."""
    p = os.path.join(out_dir, "comparison.json")
    if not os.path.isfile(p):
        return
    try:
        with open(p) as f:
            prev = json.load(f)
    except (OSError, ValueError):
        return
    old_thr = [c.get("threshold") for c in prev.get("constraints", [])]
    new_thr = [c["threshold"] for c in summary["constraints"]]
    if len(old_thr) != len(new_thr) or any(
            o is None or abs(float(o) - float(n)) > 1e-9 for o, n in zip(old_thr, new_thr)):
        print(f"  note: {p} was written for other thresholds; its arms are not carried over")
        return
    kept = [arm for arm in prev.get("arms_order", []) if arm not in arms]
    for arm in kept:
        for key in ("scores_png", "h_png", "feasible", "arms"):
            if arm in (prev.get(key) or {}):
                summary[key][arm] = prev[key][arm]
    summary["arms_order"] = kept + list(arms)


def _prior_wall(out_dir: str) -> Dict[str, float]:
    """Wall clock from an earlier comparison.json, so a --fig_only redraw or a
    --skip_existing arm keeps the time measured when it actually ran."""
    p = os.path.join(out_dir, "comparison.json")
    if os.path.isfile(p):
        try:
            with open(p) as f:
                return dict(json.load(f).get("wall_s") or {})
        except (OSError, ValueError):
            pass
    return {}


# ── main ────────────────────────────────────────────────────────────────────

def _main_single(a: argparse.Namespace) -> int:
    thresholds = list(a.constraint_thresholds)
    wall = _prior_wall(a.out_dir)
    if not a.fig_only:
        wall.update(run_arms(a, thresholds, a.out_dir))
    if a.dry_run:
        return 0
    print("\nScoring the images ...", flush=True)
    scorer = Scorer(a)
    try:
        score_and_draw(a, thresholds, a.out_dir, scorer, wall)
    finally:
        scorer.close()
    print(f"    summary: {os.path.join(a.out_dir, 'comparison.json')}")
    return 0


def _main_auto(a: argparse.Namespace) -> int:
    probe_dir = os.path.join(a.out_dir, PROBE_SUBDIR)
    manifest_path = os.path.join(a.out_dir, MANIFEST)
    mults = [a.auto_lo + (a.auto_hi - a.auto_lo) * k / max(a.n_grid - 1, 1) for k in range(a.n_grid)] \
        if a.n_grid > 1 else [a.auto_lo]
    point_dirs = [os.path.join(a.out_dir, f"m{m:.2f}") for m in mults]

    if a.dry_run:
        # The thresholds need the probe's scores, which a dry run does not
        # have; show them symbolically so every other flag can be checked.
        _print_cmd(a, "probe: uncontrolled", _probe_args(a), probe_dir)
        for m, d in zip(mults, point_dirs):
            sym = [f"<thr{i}=s0_{i}{'+' if c.direction == 'ge' else '-'}{m - 1:.2f}|s0_{i}|>"
                   for i, c in enumerate(a.constraints)]
            run_arms(a, sym, d)
        return 0

    # ── 1. probe the uncontrolled image ────────────────────────────────────
    if a.fig_only:
        if not os.path.isfile(manifest_path):
            raise SystemExit(f"--fig_only needs {manifest_path} from an earlier auto-threshold run.")
        with open(manifest_path) as f:
            man = json.load(f)
        s0 = man["uncontrolled_scores"]
        thr_grid = [pt["thresholds"] for pt in man["points"]]
        mults = [pt["multiplier"] for pt in man["points"]]
        point_dirs = [os.path.join(a.out_dir, pt["dir"]) for pt in man["points"]]
        probe_unc = man.get("probe_img_unc")
    else:
        if a.skip_existing and _ok_rows(_read_rows(probe_dir)):
            print(f"[probe] run_ok row already in {probe_dir} -- skipping.")
        else:
            _run_subprocess(a, "probe: uncontrolled", _probe_args(a), probe_dir)
        probe_unc = _one_png(probe_dir, "*__img_unc.png")

        scorer = Scorer(a)
        try:
            s0 = scorer.score(probe_unc, probe_unc)
        finally:
            scorer.close()
        thr_grid = [[c.tighten(s, m) for c, s in zip(a.constraints, s0)] for m in mults]

        labels = _labels(a.constraints)
        print(f"\n[auto-threshold] seed {a.seed}: uncontrolled scores "
              + ", ".join(f"{l}={s:.4g}" for l, s in zip(labels, s0)))
        print(f"[auto-threshold] {a.n_grid} points, multipliers {a.auto_lo:g} .. {a.auto_hi:g}:")
        for m, thr in zip(mults, thr_grid):
            print(f"    m={m:.3f}:  " + "  ".join(f"{l} {c.direction} {t:.4g}"
                                                   for l, c, t in zip(labels, a.constraints, thr)))
        # Arms recorded by an earlier run into this tree stay listed (their
        # per-point results are carried over by score_and_draw).
        prior_arms: List[str] = []
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path) as f:
                    prior_arms = list(json.load(f).get("arms") or [])
            except (OSError, ValueError):
                prior_arms = []
        man = {
            "prompt": a.prompt, "edit_prompt": a.edit_prompt, "seed": a.seed,
            "N": _arm_N(a, a.arms[0]), "lam": a.lam,
            "arms": prior_arms + [arm for arm in a.arms if arm not in prior_arms],
            "constraints": [{"type": c.ctype, "arg": c.arg, "direction": c.direction} for c in a.constraints],
            "uncontrolled_scores": s0, "probe_img_unc": probe_unc,
            "auto_lo": a.auto_lo, "auto_hi": a.auto_hi, "n_grid": a.n_grid,
            "points": [{"multiplier": m, "dir": os.path.basename(d), "thresholds": thr}
                       for m, d, thr in zip(mults, point_dirs, thr_grid)],
        }
        with open(manifest_path, "w") as f:
            json.dump(man, f, indent=2)

        # ── 2. run every point (no scorer resident on the GPU meanwhile) ──
        for pt, d, thr in zip(man["points"], point_dirs, thr_grid):
            try:
                pt["wall_s"] = _prior_wall(d) | run_arms(a, thr, d)
            except Exception as e:      # keep going; the manifest records it
                pt["error"] = f"{type(e).__name__}: {e}"
                print(f"  !! point m={pt['multiplier']:.3f} failed: {pt['error']}")
            with open(manifest_path, "w") as f:
                json.dump(man, f, indent=2)

    # ── 3. score and draw every point with one scorer load ─────────────────
    print("\nScoring every point ...", flush=True)
    scorer = Scorer(a)
    try:
        for pt, d, thr, m in zip(man["points"], point_dirs, thr_grid, mults):
            print(f"\n--- m = {m:.3f}  ({d}) ---")
            try:
                s = score_and_draw(a, thr, d, scorer, pt.get("wall_s", _prior_wall(d)), mult=m,
                                   probe_unc=probe_unc)
                pt["summary"] = {
                    "feasible": s["feasible"], "h_png": s["h_png"],
                    "energy": {arm: s["arms"][arm]["energy"] for arm in s["arms"]},
                    "ctrl_norm_max": {arm: s["arms"][arm]["ctrl_norm_max"] for arm in s["arms"]},
                    "figure": s["figure"],
                }
                pt.pop("error", None)
            except Exception as e:
                pt["error"] = f"{type(e).__name__}: {e}"
                print(f"  !! point m={m:.3f}: {pt['error']}")
    finally:
        scorer.close()
    with open(manifest_path, "w") as f:
        json.dump(man, f, indent=2)

    # ── 4. grid summary ────────────────────────────────────────────────────
    labels = _labels(a.constraints)
    print(f"\n{'m':>7}  " + "  ".join(f"{l[:12]:>12}" for l in labels)
          + "".join(f"{ARM_LABEL[arm][:9] + ' E':>12}{'feas':>6}" for arm in a.arms))
    for pt in man["points"]:
        thr_txt = "  ".join(f"{t:12.4g}" for t in pt["thresholds"])
        if "summary" in pt:
            s = pt["summary"]
            print(f"{pt['multiplier']:7.3f}  {thr_txt}"
                  + "".join(f"{s['energy'][arm]:12.2f}{str(s['feasible'][arm]):>6}" for arm in a.arms))
        else:
            print(f"{pt['multiplier']:7.3f}  {thr_txt}   {pt.get('error', 'no result')[:60]}")
    print(f"\nmanifest: {manifest_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    a = _parse_args(argv)
    if a.out_dir is None:
        tag = "auto" if a.auto_threshold else "thr" + "_".join(f"{t:g}" for t in a.constraint_thresholds)
        a.out_dir = os.path.join(HERE, "results",
                                 f"cc_{_slug(a.prompt)}__{_slug(a.edit_prompt)}__lam{a.lam:g}_{tag}")
    a.out_dir = os.path.abspath(a.out_dir)
    if not a.dry_run:
        os.makedirs(a.out_dir, exist_ok=True)
    return _main_auto(a) if a.auto_threshold else _main_single(a)


if __name__ == "__main__":
    sys.exit(main())
