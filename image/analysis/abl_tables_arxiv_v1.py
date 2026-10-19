#!/usr/bin/env python3
"""abl_tables_arxiv_v1.py -- Tables 7-9 of arXiv v1 (Appendix D, Ablations).

    python abl_tables_arxiv_v1.py [--roots DIR [DIR ...]]

Reads the runs of `run_ablation.sh --preset arxiv_v1` (default root:
image/src/results/ablation/pickscore) and prints the three tables plus the
ratios quoted in the text of D.1-D.3, each next to the value printed in the
paper. A rerun reproduces the paper statistically, not digit for digit (see
image/scripts/ablation/README.md).

SELECTION is by CSV columns, never by directory name: editing pairs ed1-ed4,
N=8, reward pickscore, run_ok=True, the arm's (control_type, H, n_inner,
constraint_on) and fixed settings (abl_data.FIXED), the table's seeds, and
lambda in the `shared` line of image/data/grids/ablation/ed<k>.txt. The last
condition matters when the extended design has also been run into the same
tree: its HardFlow top-up lambdas on ed3/ed4 are not part of arXiv v1. A
results tree in which one operating point appears twice raises.

CONVENTIONS (those of the paper's Tables 7-9)
  delta_r  r_ctrl - r_unc, in PickScore units
  E        N^2 * ctrl_energy = sum_k ||u_k||^2 dt
  time     time_ctrl_s, wall clock of the controlled sample (shared cards)
  every cell is the plain median over the table row's operating points.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st

from abl_data import ARM_OF, DEFAULT_ROOTS, FIXED, PROMPTS, _f

HERE = os.path.dirname(os.path.abspath(__file__))
GRID_DIR = os.path.normpath(os.path.join(HERE, "..", "data", "grids", "ablation"))
TAGS = ("ed1", "ed2", "ed3", "ed4")
N_STEPS = 8

# (table, label, arm, seeds, paper values (delta_r, E, time_s))
ROWS = [
    ("7", "H=1", "conicflow_H1_ni24", ("317",), (3.98, 78.5, 207)),
    ("7", "H=2", "conicflow_H2_ni24", ("317",), (4.45, 68.7, 249)),
    ("7", "H=4", "conicflow_H4_ni24", ("317",), (4.39, 70.6, 288)),
    ("7", "H=6", "conicflow_H6_ni24", ("317",), (4.67, 90.3, 321)),
    ("7", "H=8", "conicflow_H8_ni24", ("317",), (4.95, 101.4, 264)),
    ("8", "ConicFlow  n_inner=12", "conicflow_H2_ni12", ("317", "1289"), (4.16, 84.6, 124)),
    ("8", "HardFlow   n_inner=167", "hardflow_ni167", ("317", "1289"), (2.39, 555.2, 123)),
    ("8", "FMRG-J     n_inner=13", "fmrg_nopt13", ("317", "1289"), (1.84, 32.5, 128)),
    ("9", "ConicFlow  endpoint", "conicflow_H2_ni24", ("317", "1289"), (4.79, 68.7, 238)),
    ("9", "ConicFlow  horizon", "conicflow_H2_ni24_horizon", ("317", "1289"), (3.25, 86.9, 90)),
    ("9", "HardFlow   -", "hardflow_ni40", ("317", "1289"), (2.00, 553.9, 36)),
    ("9", "FMRG-J     -", "fmrg_nopt3", ("317", "1289"), (2.39, 87.0, 33)),
]
TITLES = {"7": "Table 7: planning horizon (seed 317, 24 operating points per H)",
          "8": "Table 8: matched wall clock (seeds 317, 1289, 48 operating points)",
          "9": "Table 9: objective evaluation point (seeds 317, 1289, 48 operating points)"}


def shared_lams(tag):
    for line in open(os.path.join(GRID_DIR, f"{tag}.txt")):
        parts = line.split("#")[0].split()
        if len(parts) >= 3 and parts[0] == "shared" and parts[1] == "--sweep_lam":
            return {round(float(x), 6) for x in parts[2:]}
    raise SystemExit(f"{GRID_DIR}/{tag}.txt has no 'shared --sweep_lam' line")


def load(roots):
    """-> points[(arm, seed)] = list of (delta_r, E, time_s), one per (tag, lam)."""
    shared = {t: shared_lams(t) for t in TAGS}
    seen, points = set(), {}
    for root in roots:
        files = sorted(glob.glob(os.path.join(root, "*", "*", "summary.csv")))
        if not files:
            raise SystemExit(f"no */*/summary.csv under {root}")
        for f in files:
            for r in csv.DictReader(open(f)):
                tag = PROMPTS.get((r.get("prompt"), r.get("edit_prompt")))
                if tag not in TAGS or r.get("run_ok") != "True":
                    continue
                if r.get("N") != str(N_STEPS) or r.get("reward") != "pickscore":
                    continue
                ct = "conicflow" if r["control_type"] == "mpc" else r["control_type"]
                arm = ARM_OF.get((ct, r["H"], r["n_inner"], r["constraint_on"]))
                lam = round(_f(r, "lam"), 6)
                if arm is None or lam not in shared[tag]:
                    continue
                for k, v in FIXED[ct].items():
                    if r.get(k) != v and not (k == "lr_inner" and _f(r, k) == float(v)):
                        raise SystemExit(f"{f}: {arm} row has {k}={r.get(k)!r}, expected {v!r}")
                key = (arm, r["seed"], tag, lam)
                if key in seen:
                    raise SystemExit(f"operating point {key} appears twice (last in {f})")
                seen.add(key)
                n = _f(r, "N")
                points.setdefault((arm, r["seed"]), []).append(
                    (_f(r, "delta_r"), n * n * _f(r, "ctrl_energy"), _f(r, "time_ctrl_s")))
    return points


def medians(points, arm, seeds):
    pts = [p for s in seeds for p in points.get((arm, s), [])]
    if not pts:
        return 0, None
    return len(pts), tuple(st.median(p[i] for p in pts) for i in range(3))


def mark(ours, paper):
    return "  " if ours == paper else " *"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--roots", nargs="+", default=None,
                    help="directories holding <slug>/<arm>_seed<S>/summary.csv "
                         f"(default: {DEFAULT_ROOTS[0]})")
    a = ap.parse_args()
    points = load(a.roots or DEFAULT_ROOTS)

    med, n_diff = {}, 0
    print("Values as computed; [arXiv v1] in brackets; '*' marks a printed digit that differs.")
    for table in ("7", "8", "9"):
        print(f"\n{TITLES[table]}")
        print(f"  {'':24} {'n':>3}   {'delta_r':>14}   {'E':>16}   {'time':>13}")
        for t, label, arm, seeds, paper in ROWS:
            if t != table:
                continue
            n, m = medians(points, arm, seeds)
            want = 24 if seeds == ("317",) else 48
            if m is None:
                print(f"  {label:24} {0:>3}   (no runs)")
                continue
            med[(table, arm)] = m
            got = (f"{m[0]:.2f}", f"{m[1]:.1f}", f"{m[2]:.0f}")
            ref = (f"{paper[0]:.2f}", f"{paper[1]:.1f}", f"{paper[2]:.0f}")
            n_diff += sum(g != r for g, r in zip(got, ref))
            cells = [f"{g:>6} [{r:>6}]{mark(g, r)}" for g, r in zip(got, ref)]
            note = "" if n == want else f"   (expected {want} operating points)"
            print(f"  {label:24} {n:>3}   {cells[0]}   {cells[1]}   {cells[2]} s{note}")

    def ratio(table_a, arm_a, table_b, arm_b, i):
        a_, b_ = med.get((table_a, arm_a)), med.get((table_b, arm_b))
        return None if a_ is None or b_ is None else a_[i] / b_[i]

    def pct(x):
        return "n/a" if x is None else f"{100 * (x - 1):+.1f}%"

    def nfe(h, n_inner=24, n=N_STEPS):
        # inner-loop NFE of one ConicFlow run: n_inner * (N(N-1)/2 + H)
        return n_inner * (n * (n - 1) // 2 + h)

    H = {h: f"conicflow_H{h}_ni24" for h in (1, 2, 8)}
    print("\nText of D.1 (planning horizon)                      computed  [arXiv v1]")
    print(f"  H=1 -> H=2: delta_r                                 {pct(ratio('7', H[2], '7', H[1], 0)):>8}  [+12%]")
    print(f"  H=1 -> H=2: E                                       {pct(ratio('7', H[2], '7', H[1], 1)):>8}  [-12%]")
    print(f"  H=8 vs H=1: delta_r                                 {pct(ratio('7', H[8], '7', H[1], 0)):>8}  [+24%]")
    print(f"  H=8 vs H=1: NFE = n_inner (N(N-1)/2 + H)            {pct(nfe(8) / nfe(1)):>8}  [+24%]")
    print(f"  H=2 -> H=8: E                                       {pct(ratio('7', H[8], '7', H[2], 1)):>8}  [+48%]")
    print(f"  H=2 -> H=8: delta_r                                 {pct(ratio('7', H[8], '7', H[2], 0)):>8}  [+11%]")

    cf = "conicflow_H2_ni12"
    hf = ratio("8", "hardflow_ni167", "8", cf, 0)
    he = ratio("8", "hardflow_ni167", "8", cf, 1)
    fm = ratio("8", "fmrg_nopt13", "8", cf, 0)
    print("\nText of D.2 (matched wall clock)")
    print(f"  HardFlow delta_r as a share of ConicFlow's          {'n/a' if hf is None else f'{100 * hf:.0f}%':>8}  [57%]")
    print(f"  HardFlow E / ConicFlow E                            {'n/a' if he is None else f'{he:.1f}x':>8}  [6.6x]")
    print(f"  FMRG-J delta_r as a share of ConicFlow's            {'n/a' if fm is None else f'{100 * fm:.0f}%':>8}  [under half]")

    ep, hz = med.get(("9", "conicflow_H2_ni24")), med.get(("9", "conicflow_H2_ni24_horizon"))
    print("\nText of D.3 (objective evaluation point)")
    if ep and hz:
        print(f"  RH-H speed-up over RH-T (time ratio)                {ep[2] / hz[2]:>7.1f}x  [2.6x]")
        print(f"  median delta_r, RH-T minus RH-H                     {ep[0] - hz[0]:>8.2f}  [1.54]")
        print(f"  median E, RH-H over RH-T                            {pct(hz[1] / ep[1]):>8}  [about a quarter]")
    else:
        print("  n/a (Table 9 rows missing)")

    print(f"\n{n_diff} printed table digit(s) differ from arXiv v1.")


if __name__ == "__main__":
    main()
