#!/usr/bin/env python3
"""
compare_methods.py -- one figure per comparison: uncontrolled | MPC | HardFlow | FMRG,
with each panel's control energy and reward written underneath.

THE MATCHING PROBLEM. The three methods do not reach the same operating point at
the same lambda, so putting "the lam=10 run" of each side by side compares three
different amounts of control. Two fair readings, both supported:

  --match reward   pick the triple whose achieved rewards (r_ctrl) are closest.
                   "Given they all got roughly HERE, what did it cost?" -- the
                   images then differ by control effort, which is the thing to
                   look at.
  --match energy   pick the triple whose control energies are closest.
                   "Given they all spent roughly THIS MUCH, what did it buy?"

Either way the selection is exact, not greedy: every candidate is sorted by the
chosen metric and the narrowest window containing all three methods is taken
(the classic smallest-range-covering-k-lists sweep), so the reported spread is
the best achievable from the runs supplied.

--num_rows N draws N such comparisons stacked in ONE figure, one row each. Rows
are chosen by repeating the selection with the previous row's three runs REMOVED
from the pool, so no run appears twice and each row is the tightest match still
available. It stops early -- with however many rows it got -- as soon as the
remaining pool can no longer cover all three methods, which is the honest
outcome rather than padding the figure with a loose match. Because selection is
by tightness but a figure reads better as a progression, the rows are SORTED by
the matched value before drawing, so the first row is the least-controlled
operating point and the last is the most.

WHERE THE DATA COMES FROM. Sweep directories -- anything with a summary.csv
written by run_sweep.py -- because those also contain the per-run images
(`<tag>__img_ctrl.png`, `<tag>__img_unc.png`) this figure is made of.
`lambda_autorange.py` output does NOT contain images: it calls _run_one
directly and never calls _save_run, so an autorange.json can only be used with
--dry-run, which prints the triple it would have drawn.

Runs are grouped by (reward, prompt, edit_prompt, N, seed) and one figure is
emitted per group, because comparing across any of those is comparing different
problems. r_unc is asserted equal within a group: it is the same uncontrolled
image for all three methods, and if it is not, the group is not a fair
comparison and the script says so instead of drawing it.

Usage:
  python compare_methods.py --runs results/auto/imagereward/*/  --match reward
  python compare_methods.py --runs results/compare_demo2/*/ --match both --num_rows 3
  python compare_methods.py --autorange results/autorange/imagereward/autorange.json --dry-run
"""
import argparse
import csv
import glob
import json
import math
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["mpc", "hardflow", "fmrg"]
PRETTY = {"mpc": "MPC", "hardflow": "HardFlow", "fmrg": "FMRG"}


# ═════════════════════════════════════════════════════════════════════════════
# Loading
# ═════════════════════════════════════════════════════════════════════════════
def _f(row, key, default=float("nan")):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def load_sweeps(patterns: List[str]) -> List[dict]:
    """Every run_ok row from every summary.csv under the given dirs/globs."""
    runs, seen = [], set()
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            csv_path = os.path.join(d, "summary.csv") if os.path.isdir(d) else d
            if not os.path.isfile(csv_path):
                continue
            base = os.path.dirname(csv_path)
            if csv_path in seen:
                continue
            seen.add(csv_path)
            for r in csv.DictReader(open(csv_path)):
                if r.get("run_ok") != "True" or r.get("control_type") not in METHODS:
                    continue
                tag = r.get("tag", "")
                runs.append(dict(
                    method=r["control_type"], lam=_f(r, "lam"), n_inner=r.get("n_inner", ""),
                    r_ctrl=_f(r, "r_ctrl"), r_unc=_f(r, "r_unc"),
                    dr=_f(r, "delta_r"), E=_f(r, "ctrl_energy"), J=_f(r, "objective"),
                    reward=r.get("reward", ""), prompt=r.get("prompt", ""),
                    edit_prompt=r.get("edit_prompt", ""), N=r.get("N", ""), seed=r.get("seed", ""),
                    img_ctrl=os.path.join(base, f"{tag}__img_ctrl.png") if tag else "",
                    img_unc=os.path.join(base, f"{tag}__img_unc.png") if tag else "",
                    src=base, tag=tag))
    return runs


def load_autorange(paths: List[str]) -> List[dict]:
    """Metrics only -- these runs have no images (see the module docstring)."""
    runs = []
    for p in paths:
        for f in sorted(glob.glob(p)):
            d = json.load(open(f))
            for res in d["results"]:
                for pt in res["points"]:
                    if not pt.get("ok") or pt.get("suspect"):
                        continue
                    runs.append(dict(
                        method=res["method"], lam=pt["lam"], n_inner="",
                        r_ctrl=pt["r_ctrl"], r_unc=pt["r_unc"],
                        dr=pt["dr"], E=pt["E"], J=pt["J"], reward=d["reward"], prompt=d["prompt"],
                        edit_prompt=d.get("edit_prompt") or d["prompt"], N=str(d["probe_N"]),
                        seed=str(d["args"]["seed"]), img_ctrl="", img_unc="",
                        src=os.path.dirname(f), tag=""))
    return runs


# ═════════════════════════════════════════════════════════════════════════════
# Selection
# ═════════════════════════════════════════════════════════════════════════════
def closest_triple(runs: List[dict], key: str) -> Optional[dict]:
    """Narrowest window over `key` that contains all three methods.

    Exact: sort by key, advance the right edge, pull the left edge in while the
    window still covers every method, and keep the best. O(n log n), and it
    cannot miss a better triple the way a greedy nearest-neighbour search can.
    """
    pool = sorted((r for r in runs if math.isfinite(r[key])), key=lambda r: r[key])
    if len({r["method"] for r in pool}) < len(METHODS):
        return None
    count: Dict[str, int] = {m: 0 for m in METHODS}
    have, best, lo = 0, None, 0
    for hi in range(len(pool)):
        m = pool[hi]["method"]
        count[m] += 1
        if count[m] == 1:
            have += 1
        while have == len(METHODS):
            spread = pool[hi][key] - pool[lo][key]
            if best is None or spread < best["spread"]:
                window = pool[lo:hi + 1]
                # the member of each method closest to the window's midpoint
                mid = (pool[hi][key] + pool[lo][key]) / 2.0
                pick = {mm: min((w for w in window if w["method"] == mm),
                                key=lambda w: abs(w[key] - mid)) for mm in METHODS}
                best = dict(spread=spread, lo=pool[lo][key], hi=pool[hi][key], picks=pick)
            lm = pool[lo]["method"]
            count[lm] -= 1
            if count[lm] == 0:
                have -= 1
            lo += 1
    return best


def select_rows(runs: List[dict], key: str, num_rows: int) -> List[dict]:
    """Up to `num_rows` disjoint triples, tightest first.

    Each round takes the narrowest window covering all three methods, then drops
    those three runs before the next round -- so a run is never reused and every
    later row is a genuinely different operating point. Identity is the dict
    object itself (`id`), not the metric value, because two methods can coincide
    on a value and only the actual run should be removed.

    Returns fewer than `num_rows` rows when the pool stops covering all three
    methods, which is the point at which any further row would be a match the
    data does not support.
    """
    pool = list(runs)
    rows: List[dict] = []
    for _ in range(num_rows):
        sel = closest_triple(pool, key)
        if sel is None:
            break
        rows.append(sel)
        used = {id(p) for p in sel["picks"].values()}
        pool = [r for r in pool if id(r) not in used]
    return rows


def group_key(r: dict) -> tuple:
    return (r["reward"], r["prompt"], r["edit_prompt"], r["N"], r["seed"])


# ═════════════════════════════════════════════════════════════════════════════
# Figure
# ═════════════════════════════════════════════════════════════════════════════
def draw(sels: List[dict], gkey: tuple, match: str, out_path: str, dpi: int = 130) -> str:
    """One figure, one row per selected triple: uncontrolled | MPC | HardFlow | FMRG."""
    reward, prompt, edit_prompt, N, seed = gkey
    what = "achieved reward" if match == "reward" else "control energy"
    # Selection order is by tightness; display order is by the matched value, so
    # the figure reads top-to-bottom as increasing control.
    sels = sorted(sels, key=lambda s_: (s_["lo"] + s_["hi"]) / 2.0)
    nrows = len(sels)
    fig, axes = plt.subplots(nrows, 4, figsize=(14.4, 4.55 * nrows), squeeze=False)
    for row, sel in enumerate(sels):
        picks = sel["picks"]
        unc = next((picks[m] for m in METHODS if os.path.isfile(picks[m]["img_unc"])), None)
        panels = [("Uncontrolled", unc["img_unc"] if unc else "", 0.0,
                   unc["r_unc"] if unc else picks["mpc"]["r_unc"], None, "")]
        for m in METHODS:
            pk = picks[m]
            panels.append((PRETTY[m], pk["img_ctrl"], pk["E"], pk["r_ctrl"], pk["lam"],
                           pk.get("n_inner", "")))
        for col, (title, img, E, r, lam, ni) in enumerate(panels):
            ax = axes[row][col]
            if img and os.path.isfile(img):
                ax.imshow(plt.imread(img))
            else:
                ax.text(.5, .5, "image not saved\n(autorange run)", ha="center", va="center",
                        fontsize=9, color="#888", transform=ax.transAxes)
                ax.set_facecolor("#f2f2f2")
            ax.set_xticks([]); ax.set_yticks([])
            sub = "" if lam is None else f"   \u03bb = {lam:g}" + (f",  n_inner = {ni}" if ni else "")
            ax.set_title(title + sub, fontsize=11.5, pad=7)
            ax.set_xlabel(f"energy  {E:.3g}\nreward  {r:.4f}", fontsize=10.5, linespacing=1.5)
        # Per-row match info on the left margin -- each row is its own match, so
        # the spread belongs to the row, not to the figure.
        axes[row][0].set_ylabel(f"{what}\n{sel['lo']:.4g} \u2013 {sel['hi']:.4g}\n"
                                f"(spread {sel['spread']:.3g})", fontsize=10, labelpad=10)
    head = (f'matched on {what}'
            + (f'  \u00b7  {nrows} rows, disjoint runs, ordered by {what}' if nrows > 1 else
               f':  spread {sels[0]["spread"]:.4g}  over [{sels[0]["lo"]:.4g}, {sels[0]["hi"]:.4g}]')
            + f'\n{reward}   \u00b7   "{prompt}" \u2192 "{edit_prompt}"   \u00b7   N={N}, seed={seed}')
    fig.suptitle(head, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.012 / max(nrows, 1) * 3))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def slug(gkey: tuple, match: str, nrows: int) -> str:
    reward, prompt, edit, N, seed = gkey
    p = "".join(c if c.isalnum() else "_" for c in f"{prompt}__{edit}")[:48]
    rows = f"__x{nrows}" if nrows > 1 else ""
    return f"compare_{match}__{reward}__{p}__N{N}_seed{seed}{rows}.png"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], metavar="DIR",
                    help="sweep dirs (or globs) containing summary.csv AND the per-run images")
    ap.add_argument("--autorange", nargs="*", default=[], metavar="JSON",
                    help="lambda_autorange autorange.json -- metrics only, no images; use with --dry-run")
    ap.add_argument("--match", default="reward", choices=["reward", "energy", "both"],
                    help="reward: closest achieved r_ctrl. energy: closest ctrl_energy.")
    ap.add_argument("--num_rows", type=int, default=1, metavar="N",
                    help="how many comparison rows to stack in the figure. Each row reuses none "
                         "of the previous rows' runs: the chosen triple is removed from the pool "
                         "and the match is repeated. Stops early if the remaining runs no longer "
                         "cover all three methods.")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--dry_run", "--dry-run", action="store_true", dest="dry_run",
                    help="print the chosen triples, draw nothing")
    ap.add_argument("--dpi", type=int, default=130)
    a = ap.parse_args()
    if not a.runs and not a.autorange:
        ap.error("give --runs and/or --autorange")

    runs = load_sweeps(a.runs) + load_autorange(a.autorange)
    if not runs:
        raise SystemExit("no usable runs found (need run_ok=True rows of mpc/hardflow/fmrg)")
    groups: Dict[tuple, List[dict]] = {}
    for r in runs:
        groups.setdefault(group_key(r), []).append(r)
    print(f"{len(runs)} runs in {len(groups)} group(s)")

    modes = ["reward", "energy"] if a.match == "both" else [a.match]
    os.makedirs(a.out_dir, exist_ok=True)
    made = 0
    for gkey, grp in sorted(groups.items()):
        have = {r["method"] for r in grp}
        if len(have) < len(METHODS):
            print(f"  SKIP {gkey}: only {sorted(have)} present")
            continue
        # r_unc must agree -- otherwise these are not the same uncontrolled image
        runcs = {round(r["r_unc"], 4) for r in grp if math.isfinite(r["r_unc"])}
        if len(runcs) > 1:
            print(f"  SKIP {gkey}: r_unc disagrees across runs {sorted(runcs)} -- not one problem")
            continue
        for mode in modes:
            key = "r_ctrl" if mode == "reward" else "E"
            sels = select_rows(grp, key, a.num_rows)
            if not sels:
                print(f"  SKIP {gkey} [{mode}]: no window covers all three methods")
                continue
            if len(sels) < a.num_rows:
                print(f"\n  NOTE {gkey[0]} [{mode}]: asked for {a.num_rows} rows, the pool "
                      f"supported {len(sels)} -- the rest could not cover all three methods.")
            for i, sel in enumerate(sorted(sels, key=lambda s_: (s_["lo"] + s_["hi"]) / 2.0)):
                print(f"\n  {gkey[0]} | N={gkey[3]} seed={gkey[4]} | match={mode}"
                      f"{f' | row {i + 1}/{len(sels)}' if len(sels) > 1 else ''}: "
                      f"spread {sel['spread']:.4g} over [{sel['lo']:.4g}, {sel['hi']:.4g}]")
                print(f"    {'':10} {'lam':>9} {'reward':>10} {'energy':>11}   source")
                print(f"    {'uncontrolled':10} {'-':>9} {grp[0]['r_unc']:10.4f} {0:11.4g}")
                for m in METHODS:
                    p = sel["picks"][m]
                    img = "" if not p["img_ctrl"] else ("  [img ok]" if os.path.isfile(p["img_ctrl"]) else "  [IMG MISSING]")
                    print(f"    {PRETTY[m]:10} {p['lam']:9.4g} {p['r_ctrl']:10.4f} {p['E']:11.4g}   "
                          f"{os.path.basename(p['src'])}{img}")
            if a.dry_run:
                continue
            out = os.path.join(a.out_dir, slug(gkey, mode, len(sels)))
            draw(sels, gkey, mode, out, a.dpi)
            made += 1
            print(f"    -> {out}")
    if not a.dry_run:
        print(f"\nwrote {made} figure(s) to {a.out_dir}")


if __name__ == "__main__":
    main()
