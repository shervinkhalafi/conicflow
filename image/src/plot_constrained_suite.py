#!/usr/bin/env python3
r"""
plot_constrained_suite.py -- control energy and feasibility against constraint
tightness, across a constrained prompt suite.

Reads the tree scripts/constrained/run_suite.sh writes:

    <root>/<tag>_seed<seed>/auto_threshold.json
    <root>/<tag>_seed<seed>/m<mult>/{conicflow,hardflow,pflow,ldf}/summary.csv

and draws two stacked panels against the threshold multiplier m:

    top     control energy, median across (prompt, seed) with the IQR shaded,
            every run as a dot (hollow = infeasible), runs above the axis cap
            marked with a triangle rather than dropped
    bottom  feasibility rate (all h_i <= 0), with a dotted line and band that
            also count near-misses within --near of the boundary

Every arm of one (prompt, seed, m) cell shares the seed, lambda and thresholds,
so m is the only axis. Prompts, seeds and multipliers are discovered from the
directory names; --prompts / --seeds / --mults restrict them, and --arms
restricts or reorders the arms (default: every registered arm with at least
one run in the tree, in registry order).

ENERGY UNITS. summary.csv's `ctrl_energy` is the displacement residual
sum_k ||x_{k+1} - x_k - dt v||^2 * dt. The paper reports the control energy
sum_k ||u_k||^2 dt with u_k = (displacement)/dt, which is the same quantity
times N^2 (64 at N=8). That factor is applied here per row from the row's own
N; --raw_energy plots the stored column instead. Feasibility is read from the
CSV's h_i, i.e. the constraint values on the final latent as decoded in the
control loop.

    python plot_constrained_suite.py --root results/constrained_suite
    python plot_constrained_suite.py --root results/constrained_suite --paper_style \
        --out figures/con_main_plot.png          # Figure 3 (left)
    python plot_constrained_suite.py --root results/constrained_suite \
        --prompts p01_smile p09_curly_old --out figs/suite_subset.png
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# arm (run_sweep control type = directory name) -> (legend label, colour,
# marker, median-line style). ConicFlow and HardFlow keep the paper's colours
# and solid lines. A new constrained baseline needs an entry here before it can
# be plotted; arms present in the tree without one are reported, not silently
# dropped.
#
# PROVISIONAL: pflow / ldf take categorical slots 4 (yellow) and 5 (magenta) of
# the reference palette -- slot 3 (aqua) stays FMRG's, as in the unconstrained
# figures, so no colour means two methods across figures. Their pairing with
# teal/orange has not been run through a palette validator yet, so they also
# carry secondary encoding: their own marker and a dashed / dash-dot median line.
ARM_STYLE = {
    "conicflow": ("ConicFlow",     "#0d9488", "o", "-"),    # teal
    "hardflow": ("HardFlow",       "#c2410c", "s", "-"),    # deep orange
    "pflow":    ("Projected Flow", "#eda100", "^", "--"),   # palette slot 4, provisional
    "ldf":      ("LDF",            "#e87ba4", "D", "-."),   # palette slot 5, provisional
}
# Directory names each arm may have been written under: older result trees
# name the ConicFlow arm `mpc`.
ARM_DIR_ALIASES = {"conicflow": ("conicflow", "mpc")}

# Text sizes. The default is sized for reading the figure on its own;
# --paper_style uses the larger axis labels, ticks and legend of the paper's
# Figure 3 (left), which is printed at a third of the text width.
FONTS = dict(label=11, title=10.5, legend=11, tick=None, top=0.92)
PAPER_FONTS = dict(label=20, title=10.65, legend=18, tick=12, top=0.875)


def _arm_dir(cell: Path, arm: str) -> Path:
    for name in ARM_DIR_ALIASES.get(arm, (arm,)):
        if (cell / name / "summary.csv").is_file():
            return cell / name
    return cell / arm


def discover(root: Path):
    """(prompt tags, seeds, multipliers, arm directory names) present under root."""
    tags, seeds, mults, arm_dirs = set(), set(), set(), set()
    for man in root.glob("*_seed*/auto_threshold.json"):
        m = re.match(r"(.+)_seed(\d+)$", man.parent.name)
        if not m:
            continue
        tags.add(m.group(1)); seeds.add(int(m.group(2)))
        for pt in json.loads(man.read_text()).get("points", []):
            mults.add(round(float(pt["multiplier"]), 2))
        for f in man.parent.glob("m*/*/summary.csv"):
            arm_dirs.add(f.parent.name)
    return sorted(tags), sorted(seeds), sorted(mults), arm_dirs


def _arms_present(arm_dirs: set) -> list:
    """Registered arms with at least one run, in registry order."""
    return [a for a in ARM_STYLE if any(d in arm_dirs for d in ARM_DIR_ALIASES.get(a, (a,)))]


def load_suite(root: Path, prompts, seeds, mults, arms, near: float, n2: bool):
    """Energy / feasible / on-boundary arrays of shape (n_mult, n_prompt*n_seed), per arm."""
    units = [(p, s) for p in prompts for s in seeds]
    shape = (len(mults), len(units))
    E    = {a: np.full(shape, np.nan) for a in arms}
    feas = {a: np.zeros(shape, bool)  for a in arms}
    onb  = {a: np.zeros(shape, bool)  for a in arms}
    missing = []
    for u, (prompt, seed) in enumerate(units):
        base = root / f"{prompt}_seed{seed}"
        man = base / "auto_threshold.json"
        if not man.is_file():
            missing.append(str(base)); continue
        n_con = len(json.loads(man.read_text())["constraints"])
        for i, m in enumerate(mults):
            cell = base / f"m{m:.2f}"
            for arm in arms:
                f = _arm_dir(cell, arm) / "summary.csv"
                if not f.is_file():
                    continue
                rows = pd.read_csv(f)
                if rows.empty:
                    continue
                r = rows.iloc[-1]
                if not np.isfinite(float(r["ctrl_energy"])):
                    continue
                worst = max(float(r[f"h_{k}"]) for k in range(n_con))
                scale = float(r["N"]) ** 2 if n2 else 1.0
                E[arm][i, u]    = float(r["ctrl_energy"]) * scale
                feas[arm][i, u] = worst <= 0
                onb[arm][i, u]  = worst <= near
    if missing:
        print(f"  note: {len(missing)} (prompt, seed) cells absent: " + ", ".join(missing[:4])
              + (" ..." if len(missing) > 4 else ""))
    return E, feas, onb


def plot(E, feas, onb, mults, arms, near, ymin, ymax, out: Path, dpi: int, fonts=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs = fonts or FONTS
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.8, 9.4), sharex=True,
                                 gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.16})
    x = np.array(mults)

    # ---- top: energy -------------------------------------------------------
    for arm in arms:
        label, col, mk, ls = ARM_STYLE[arm]
        a1.plot(x, np.nanmedian(E[arm], axis=1), ls, color=col, marker=mk, lw=2.4, ms=8,
                label=label, zorder=4)
        a1.fill_between(x, np.nanpercentile(E[arm], 25, axis=1),
                        np.minimum(np.nanpercentile(E[arm], 75, axis=1), ymax),
                        color=col, alpha=0.16, lw=0, zorder=2)
        for u in range(E[arm].shape[1]):
            ok = feas[arm][:, u]
            y  = np.clip(E[arm][:, u], None, ymax)
            a1.scatter(x[ok],  y[ok],  s=15, color=col, alpha=0.35, zorder=3)
            a1.scatter(x[~ok], y[~ok], s=20, facecolors="none", edgecolors=col,
                       linewidths=0.9, alpha=0.5, zorder=3)
        over = E[arm] > ymax
        for i in range(len(mults)):
            if over[i].any():
                a1.scatter([x[i]] * over[i].sum(), [ymax * 0.94] * over[i].sum(),
                           marker="^", s=26, color=col, alpha=0.75, zorder=6)
    a1.set_yscale("log"); a1.set_ylim(ymin, ymax)
    a1.set_ylabel(r"control energy  $\sum_j \|u_j\|^2 \Delta t$", fontsize=fs["label"])
    a1.grid(alpha=0.25, which="both"); a1.legend(fontsize=fs["legend"], loc="upper left")
    a1.set_title("Control energy — median across (prompt, seed), IQR shaded\n"
                 r"dots are individual runs (hollow = infeasible);  $\blacktriangle$ = above the axis cap",
                 fontsize=fs["title"], pad=9)

    # ---- bottom: feasibility -----------------------------------------------
    for arm in arms:
        label, col, mk, ls = ARM_STYLE[arm]
        n_run = np.maximum((~np.isnan(E[arm])).sum(axis=1), 1)
        pct   = 100 * feas[arm].sum(axis=1) / n_run
        pctb  = 100 * onb[arm].sum(axis=1)  / n_run
        a2.plot(x, pct, ls, color=col, marker=mk, lw=2.4, ms=8, label=label, zorder=4)
        a2.plot(x, pctb, ":", color=col, lw=1.4, alpha=0.8, zorder=3)
        a2.fill_between(x, pct, pctb, color=col, alpha=0.15, lw=0, zorder=2)
    a2.set_xlabel("threshold multiplier  m", fontsize=fs["label"])
    a2.set_ylabel("feasible runs  (%)", fontsize=fs["label"])
    a2.set_xticks(x); a2.set_ylim(-4, 104); a2.set_yticks([0, 25, 50, 75, 100])
    a2.grid(alpha=0.25); a2.legend(fontsize=fs["legend"], loc="lower left")
    a2.set_title(r"Feasibility — solid: all $h_i \leq 0$;  dotted/shaded: also counting "
                 f"misses within {near} of the boundary", fontsize=fs["title"], pad=9)
    if fs["tick"]:
        a1.tick_params(labelsize=fs["tick"]); a2.tick_params(labelsize=fs["tick"])
    fig.subplots_adjust(top=fs["top"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def table(E, feas, onb, mults, arms, ymax) -> pd.DataFrame:
    """The numbers behind the figure, so caption claims can be checked. Energy
    ratios are each later arm against the first (HardFlow / ConicFlow with the
    default arms)."""
    rows = []
    ref = arms[0]
    for i, m in enumerate(mults):
        row = {"m": m}
        for arm in arms:
            label = ARM_STYLE[arm][0]
            n_run = int((~np.isnan(E[arm][i])).sum())
            row[f"{label} feasible"] = f"{int(feas[arm][i].sum())}/{n_run}"
            row[f"{label} near-boundary"] = int(onb[arm][i].sum())
        for arm in arms:
            row[f"{ARM_STYLE[arm][0]} median E"] = round(float(np.nanmedian(E[arm][i])), 3)
        for arm in arms[1:]:
            row[f"E ratio {ARM_STYLE[arm][0]}/{ARM_STYLE[ref][0]}"] = round(
                float(np.nanmedian(E[arm][i]) / np.nanmedian(E[ref][i])), 2)
        row["above cap"] = int(sum((E[arm][i] > ymax).sum() for arm in arms))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.split("\n\n")[1:]))
    p.add_argument("--root", required=True, help="the suite tree (see the docstring)")
    p.add_argument("--out", default=None,
                   help="figure path (default <root>/suite_energy_feasibility_stacked.png)")
    p.add_argument("--prompts", nargs="+", default=None, help="restrict to these <tag>s")
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--mults", nargs="+", type=float, default=None)
    p.add_argument("--arms", nargs="+", default=None, choices=list(ARM_STYLE), metavar="ARM",
                   help=f"arms to plot, in order, from {list(ARM_STYLE)} (default: every one "
                        f"with at least one run in the tree)")
    p.add_argument("--near", type=float, default=0.2,
                   help="a miss within this of the boundary counts as near-boundary (default 0.2)")
    p.add_argument("--ymin", type=float, default=44.8, help="energy-axis floor (default 0.7 x 64)")
    p.add_argument("--ymax", type=float, default=6.4e4,
                   help="energy-axis cap; runs above it are marked, not dropped (default 1e3 x 64)")
    p.add_argument("--raw_energy", action="store_true",
                   help="plot summary.csv's ctrl_energy as stored, without the N^2 factor "
                        "(then also lower --ymin/--ymax by 64)")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--paper_style", action="store_true",
                   help="the paper's text sizes (axis labels 20, legend 18, ticks 12)")
    a = p.parse_args(argv)

    root = Path(a.root)
    tags, seeds, mults, arm_dirs = discover(root)
    if not tags:
        print(f"no <tag>_seed<seed>/auto_threshold.json under {root}", file=sys.stderr); return 1
    known = {d for arm in ARM_STYLE for d in ARM_DIR_ALIASES.get(arm, (arm,))}
    unstyled = sorted(arm_dirs - known)
    if unstyled:
        print(f"  note: arm directories with no ARM_STYLE entry, not plotted: {unstyled}")
    arms = a.arms or _arms_present(arm_dirs)
    if not arms:
        print(f"no registered arm ({list(ARM_STYLE)}) has a run under {root}", file=sys.stderr); return 1
    prompts = a.prompts or tags
    seeds   = a.seeds or seeds
    mults   = [round(m, 2) for m in (a.mults or mults)]
    unknown = [t for t in prompts if t not in tags]
    if unknown:
        print(f"--prompts not found under {root}: {unknown}\n  available: {tags}", file=sys.stderr); return 1
    print(f"{len(prompts)} prompts x {len(seeds)} seeds, m = {mults}, arms = {arms}")

    E, feas, onb = load_suite(root, prompts, seeds, mults, arms, a.near, not a.raw_energy)
    out = Path(a.out) if a.out else root / "suite_energy_feasibility_stacked.png"
    plot(E, feas, onb, mults, arms, a.near, a.ymin, a.ymax, out, a.dpi,
         PAPER_FONTS if a.paper_style else FONTS)
    t = table(E, feas, onb, mults, arms, a.ymax)
    t.to_csv(out.with_suffix(".csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(t.to_string(index=False))
    print(f"\nfigure : {out}\ntable  : {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
