#!/usr/bin/env python3
"""abl_data.py -- data layer and metrics for the appendix ablations.

    python abl_data.py [--roots DIR [DIR ...]]    # coverage + the prose numbers

`abl_figs.py` imports this module and draws the figures and tables.

SOURCES. Every `*/*/summary.csv` under each root (default: the tree written by
image/scripts/ablation/run_ablation.sh, image/src/results/ablation/pickscore).
Rows are keyed by their CSV COLUMNS (control_type, H, n_inner, constraint_on),
never by directory name, and every row is checked against its arm's fixed
config: a row that claims to be an arm but differs in lr / surrogate / alpha
raises. Rows written before the method was renamed carry control_type `mpc`;
they are read as ConicFlow.

CONVENTIONS
  E      = N^2 * ctrl_energy, with N from the row. The stored ctrl_energy is the
           displacement residual sum_k ||dx_k||^2 dt; the paper's energy is
           sum_k ||u_k||^2 dt with u_k = dx_k / dt, i.e. N^2 times larger.
  J      = E / (2 lam) - r_ctrl,   J_null = -r_unc
  G      = the paper's mean reward gain over a shared energy budget (eq. auc):
           per-lambda best run, non-dominated frontier, mean height of delta_r
           over log10 E between the two compared arms' shared budget [E1, E2].
  score  = G(arm) / G(reference) per (prompt, seed) cell; pooled as the median
           over cells with a 95% percentile bootstrap over cells (runs within a
           cell are not independent, cells are).
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import random
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOTS = [os.path.normpath(os.path.join(HERE, "..", "src", "results", "ablation", "pickscore"))]

PROMPTS = {
    ("a portrait of a woman with straight hair", "a portrait of a woman with curly hair"): "ed1",
    ("a portrait of a woman with black hair", "a portrait of a woman with white hair"): "ed2",
    ("a portrait of a young man", "a portrait of an old man"): "ed3",
    ("a portrait of a smiling man", "a portrait of an angry man"): "ed4",
    ("a portrait of a serious woman", "a portrait of a smiling woman"): "ed5",
    ("a portrait of a clean-shaven man", "a portrait of a man with a mustache"): "ed6",
    ("a portrait of a thin man", "a portrait of a chubby man"): "ed7",
    ("a portrait of a happy woman", "a portrait of a sad woman"): "ed8",
}
SEEDS = {"317", "1289", "2718", "3141", "1618", "1414"}
N_STEPS = "8"

# arm name <- (control_type, H, n_inner, constraint_on); FMRG and HardFlow rows
# carry H=2 and constraint_on=endpoint, which mean nothing for them.
ARM_OF = {}
for _h in ("1", "2", "4", "6", "8"):
    ARM_OF[("conicflow", _h, "24", "endpoint")] = f"conicflow_H{_h}_ni24"
    ARM_OF[("conicflow", _h, "24", "horizon")] = f"conicflow_H{_h}_ni24_horizon"
ARM_OF[("conicflow", "2", "12", "endpoint")] = "conicflow_H2_ni12"
ARM_OF[("hardflow", "2", "40", "endpoint")] = "hardflow_ni40"
ARM_OF[("hardflow", "2", "167", "endpoint")] = "hardflow_ni167"
ARM_OF[("fmrg", "2", "3", "endpoint")] = "fmrg_nopt3"
ARM_OF[("fmrg", "2", "13", "endpoint")] = "fmrg_nopt13"
REF = "conicflow_H2_ni24"

# fixed config every row of a method must carry
FIXED = {
    "conicflow": {"lr_inner": "0.01", "endpoint_surrogate": "jump", "drift_detach": "True",
                  "sample_mode": "flow_map2"},
    "hardflow": {"lr_inner": "2.5", "hardflow_lr_alpha": "0.5", "endpoint_surrogate": "jump",
                 "sample_mode": "flow_map2"},
    "fmrg": {"endpoint_surrogate": "jump", "sample_mode": "flow_map2"},
}


def _f(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")


class Point:
    __slots__ = ("lam", "E", "dr", "r_unc", "r_ctrl", "J", "t", "src")

    def __init__(self, r, src):
        n = _f(r, "N")
        self.lam = _f(r, "lam")
        self.E = n * n * _f(r, "ctrl_energy")
        self.dr = _f(r, "delta_r")
        self.r_unc = _f(r, "r_unc")
        self.r_ctrl = _f(r, "r_ctrl")
        self.J = self.E / (2 * self.lam) - self.r_ctrl
        self.t = _f(r, "time_ctrl_s")
        self.src = src

    @property
    def beats_null(self):
        return self.J < -self.r_unc


def load(roots=None, dedup="best"):
    """-> (data, log), data[(tag, seed)][arm] = list of Points sorted by lam.

    dedup: several runs at one lam (a re-run) -> "best" keeps the highest-reward
    run, the per-lambda selection of metric G and the paper's tables; "mean"
    averages them."""
    if dedup not in ("mean", "best"):
        raise ValueError(f"dedup must be 'mean' or 'best', got {dedup!r}")
    roots = roots or DEFAULT_ROOTS
    data = defaultdict(lambda: defaultdict(list))
    log = defaultdict(int)
    for root in roots:
        files = glob.glob(os.path.join(root, "*", "*", "summary.csv"))
        if not files:
            raise SystemExit(f"no */*/summary.csv under {root}")
        for f in files:
            for r in csv.DictReader(open(f)):
                tag = PROMPTS.get((r.get("prompt"), r.get("edit_prompt")))
                if tag is None or r.get("seed") not in SEEDS:
                    continue
                if r.get("run_ok") != "True":
                    log["failed_rows_skipped"] += 1
                    continue
                if r.get("N") != N_STEPS or r.get("reward") != "pickscore":
                    continue
                ct = "conicflow" if r["control_type"] == "mpc" else r["control_type"]
                arm = ARM_OF.get((ct, r["H"], r["n_inner"], r["constraint_on"]))
                if arm is None:
                    log[f"unmapped {(ct, r['H'], r['n_inner'], r['constraint_on'])}"] += 1
                    continue
                for k, v in FIXED[ct].items():
                    if r.get(k) != v and not (k == "lr_inner" and _f(r, k) == float(v)):
                        raise SystemExit(f"{f}: {arm} row has {k}={r.get(k)!r}, expected {v!r}")
                data[(tag, r["seed"])][arm].append(Point(r, f))
    for cell in data.values():
        for arm, pts in cell.items():
            by = defaultdict(list)
            for p in pts:
                by[round(p.lam, 6)].append(p)
            out = []
            for lam, ps in sorted(by.items()):
                if len(ps) > 1 and dedup == "best":
                    log["duplicate_lams_best_kept"] += 1
                    ps = [max(ps, key=lambda p: p.r_ctrl)]
                elif len(ps) > 1:
                    log["duplicate_lams_averaged"] += 1
                    for a in ("E", "dr", "r_ctrl", "J", "t"):
                        setattr(ps[0], a, st.mean(getattr(p, a) for p in ps))
                out.append(ps[0])
            cell[arm] = out
    # r_unc must agree within a cell, or delta_r is not comparable across arms
    for (tag, seed), cell in data.items():
        ru = {round(p.r_unc, 5) for pts in cell.values() for p in pts}
        if len(ru) > 1:
            raise SystemExit(f"r_unc disagrees within cell {tag}/{seed}: {sorted(ru)}")
    # cells in a fixed order, so the bootstrap does not depend on file-system order
    ordered = {k: data[k] for k in sorted(data, key=lambda k: (k[0], int(k[1])))}
    return ordered, log


# ── the paper's metric G (eq. auc) ──────────────────────────────────────────
def frontier(pts):
    """non-dominated (E, delta_r) runs, sorted by E: drop any run another run
    beats on both axes (no more energy, no less reward, one strictly)."""
    xy = [(p.E, p.dr) for p in pts]
    keep = [(x, y) for x, y in xy
            if not any(x2 <= x and y2 >= y and (x2 < x or y2 > y) for x2, y2 in xy)]
    return sorted(keep)


def G_on(front, a, b, n=2001):
    """mean height of delta_r over [a, b] in log10 E: trapezoid on n points of
    the piecewise-linear interpolant in log10 E."""
    lx = [math.log10(x) for x, _ in front]
    ys = [y for _, y in front]
    g = [a + (b - a) * i / (n - 1) for i in range(n)]

    def f(u):  # as np.interp: clamps outside [lx0, lx-1]; [a, b] lies inside by construction
        if u <= lx[0]:
            return ys[0]
        for (x0, y0), (x1, y1) in zip(zip(lx, ys), zip(lx[1:], ys[1:])):
            if u <= x1:
                return y0 + (u - x0) / (x1 - x0) * (y1 - y0) if x1 > x0 else y1
        return ys[-1]

    v = [f(u) for u in g]
    h = (b - a) / (n - 1)
    return h * (sum(v) - 0.5 * (v[0] + v[-1])) / (b - a)


def score_G(cell, arm, ref=REF):
    """G(arm) / G(ref), both on the PAIR's shared budget [E1, E2]: eq. auc with
    the interval taken over these two arms rather than three methods."""
    if len(cell.get(arm, [])) < 1 or len(cell.get(ref, [])) < 1:
        return None
    fa, fr = frontier(cell[arm]), frontier(cell[ref])
    E1, E2 = max(fa[0][0], fr[0][0]), min(fa[-1][0], fr[-1][0])
    if not E2 > E1:
        return None
    a, b = math.log10(E1), math.log10(E2)
    gr = G_on(fr, a, b)
    return G_on(fa, a, b) / gr if gr > 0 else None


def boot_median(vals, n=4000, seed=0):
    rng = random.Random(seed)
    meds = sorted(st.median(rng.choices(vals, k=len(vals))) for _ in range(n))
    return st.median(vals), meds[int(0.025 * n)], meds[int(0.975 * n) - 1]


def clean_time(cell_pts):
    """time_ctrl_s with a contention filter: drop runs slower than 1.3x the
    cell median (another job shared the GPU)."""
    ts = [p.t for p in cell_pts]
    if not ts:
        return []
    m = st.median(ts)
    return [t for t in ts if t <= 1.3 * m]


def paired_equal_lam(data, a, b):
    """(dr_a - dr_b, E_a / E_b) over operating points where both ran at one lam."""
    out = []
    for cell in data.values():
        A = {round(p.lam, 6): p for p in cell.get(a, [])}
        for p in cell.get(b, []):
            q = A.get(round(p.lam, 6))
            if q is not None:
                out.append((q.dr - p.dr, q.E / p.E if p.E > 0 else float("nan")))
    return out


def horizon_prose(data):
    """The equal-lambda comparison of H=4, 6, 8 against H=2 quoted in the
    planning-horizon text: median reward gain and median energy ratio."""
    lines = []
    for h in (4, 6, 8):
        pr = paired_equal_lam(data, f"conicflow_H{h}_ni24", REF)
        if not pr:
            continue
        d = st.median(x for x, _ in pr)
        e = st.median(r for _, r in pr if r == r)
        lines.append(f"H={h} vs H=2 at equal lambda ({len(pr)} points): "
                     f"median delta_r {d:+.3f}, median energy ratio {e:.2f} ({100 * (e - 1):+.0f}%)")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--roots", nargs="+", default=None,
                    help="directories holding <slug>/<arm>_seed<S>/summary.csv "
                         f"(default: {DEFAULT_ROOTS[0]})")
    a = ap.parse_args()
    data, log = load(a.roots)
    print("== coverage: cells (prompt x seed) per arm, and operating points ==")
    for arm in sorted({x for c in data.values() for x in c}):
        cells = [k for k, c in data.items() if c.get(arm)]
        npts = sum(len(data[k][arm]) for k in cells)
        seeds = sorted({s for _, s in cells}, key=int)
        print(f"  {arm:28} {len(cells):3} cells  {npts:4} points  seeds {','.join(seeds)}")
    for k, v in sorted(log.items()):
        print(f"  [log] {k}: {v}")
    print("\n== planning horizon at equal lambda ==")
    for line in horizon_prose(data):
        print("  " + line)


if __name__ == "__main__":
    main()
