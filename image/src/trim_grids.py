#!/usr/bin/env python3
"""Trim an autorange lambda grid to where the MEASURED reward curve stops moving.

    python trim_grids.py <autorange_dir> [--margin 10] [--apply]

Reads ``<autorange_dir>/evaluations.csv`` and ``grids.txt``, and for each method
truncates the emitted grid at the plateau of that method's own probe curve.
Writes the original to ``grids_raw.txt`` and the trimmed grid to ``grids.txt``
(``--apply``; without it, prints what it would do and changes nothing).

WHY THIS EXISTS (2026-09-23)
---------------------------
``--high_stop break`` walks lambda up until the IMAGE breaks or the reward
collapses.  ConicFlow does neither: it is the energy-regularised arm, so its curve
goes *flat* rather than destructive, and the walk runs to the lambda box.  On
the first real GPU run of break mode (pickscore / "A Japanese garden at dawn",
seed 811, probe N=4) ConicFlow measured

    lam   7.98  16    79.8  252   6300  1.3e5  3.5e5  2.5e6
    dr    8.99  9.05  9.57  9.52  9.25  9.01   10.58  9.25

-- flat from lam~8 to 2.5e6 -- and the logistic fit, having no upper anchor,
reported the active region as [0.8, 2.52e6].  The emitted grid was

    conicflow --sweep_lam 0.8 16 320 6300 130000 350000 940000 2.5e+06

i.e. five of eight points re-measuring one plateau point, at 239 s per ConicFlow run.
That is ~20 of the ~41 ConicFlow minutes per cell, on every cell where ConicFlow does not
break.  ``docs/lambda_autorange.md`` SS11.4 named this as the one thing a live
run would tell us that the dry runs could not; this is the answer.

WHAT IT DOES NOT DO
-------------------
It does not restore the ``near_ceiling`` stop.  That stop capped on a *constant*
-- ``REWARD_OBSERVED_RANGE``'s pickscore max of 27.0, which ConicFlow exceeds -- and
capping on it is what hid the 5.6x of ConicFlow headroom that
``results/lam_headroom_20260923`` found on the greenhouse cell (grid top lam=1,
true peak lam=320, delta_r 3.77 -> 21.12).  This trims on the measured curve
only, so a cell whose curve is still climbing at the top of its grid is left
exactly as the range finder emitted it.  On the garden cell that is HardFlow
(grid top 2.5, still rising) and FMRG (top 5.05, still rising): both untouched.

THE RULE
--------
Per method, over its ok-and-unbroken probe points sorted by lambda:

1. Smooth ``dr`` with a running median of 3.  One noisy high point is otherwise
   enough to defeat any "has it stopped rising" test -- on the garden cell the
   lone 10.58 at lam=3.5e5 is 1.3 above its neighbours and is pure noise.
2. ``lam_knee`` = the smallest lambda such that every later smoothed point is
   within ``tol = 0.10 * (dr_max - dr_min)`` of it.  That is "the curve makes no
   further meaningful progress above here".
3. ``lam_top = lam_knee * margin`` (default 10, one decade).  The margin is not
   cosmetic: the probe runs at N=4 and the sweep at N=8, and while the
   transition moves DOWN in lambda with more controlled steps (so the margin is
   already conservative), SS11.4 flags the break point specifically as less
   N-invariant than the transition.  A decade of slack costs one grid point and
   buys the case where the target-N curve is still live above the probe's knee.
4. If the emitted grid already tops out at or below ``lam_top``, leave it alone.
   Otherwise re-place the SAME NUMBER of points geometrically from the grid's
   existing low end to ``lam_top``, so the cell costs exactly what it did before
   and every point lands where the curve is moving.

The ``shared`` line is trimmed at the largest per-method ``lam_top``, since it
has to span every method's active region.  (It is only written on a FRESH
probe; a cached grids.txt was already trimmed when it was written, and the
ablations re-use Experiment 1's cached grids rather than re-range-finding.)
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics as st
import sys

METHODS = ("conicflow", "hardflow", "fmrg")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_points(eval_csv: str, method: str):
    """(lam, dr) for this method's usable probe points, sorted by lam."""
    pts = []
    with open(eval_csv) as fh:
        for r in csv.DictReader(fh):
            if r.get("method") != method:
                continue
            if str(r.get("ok")).lower() not in ("true", "1"):
                continue
            # A broken image is excluded from the fit by the range finder; it is
            # excluded here too, for the same reason.
            if str(r.get("broken")).lower() in ("true", "1"):
                continue
            lam, dr = _f(r.get("lam")), _f(r.get("dr"))
            if lam is None or dr is None:
                continue
            pts.append((lam, dr))
    pts.sort(key=lambda p: p[0])
    return pts


def running_median3(xs):
    if len(xs) < 3:
        return list(xs)
    out = [xs[0]]
    out += [st.median(xs[i - 1:i + 2]) for i in range(1, len(xs) - 1)]
    out.append(xs[-1])
    return out


def knee(pts, tol_frac=0.10):
    """Smallest lam above which the smoothed curve makes no further progress."""
    if len(pts) < 4:
        return None
    lams = [p[0] for p in pts]
    dr = running_median3([p[1] for p in pts])
    span = max(dr) - min(dr)
    if span <= 0:
        return None
    tol = tol_frac * span
    for i, (lam, _) in enumerate(zip(lams, dr)):
        if all(later <= dr[i] + tol for later in dr[i + 1:]):
            return lam
    return lams[-1]


def geomspace(lo, hi, n):
    if n <= 1 or lo <= 0 or hi <= lo:
        return [lo]
    ratio = (hi / lo) ** (1.0 / (n - 1))
    return [lo * ratio ** i for i in range(n)]


def fmt(x):
    """Match the range finder's own 2-significant-figure grid formatting."""
    return f"{float(f'{x:.2g}'):g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("autorange_dir")
    ap.add_argument("--margin", type=float, default=10.0,
                    help="keep this factor above the measured knee (default 10 = one decade)")
    ap.add_argument("--tol_frac", type=float, default=0.10,
                    help="'no further progress' tolerance, as a fraction of the dr span")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite grids.txt (original kept as grids_raw.txt); default is a dry run")
    a = ap.parse_args()

    d = a.autorange_dir
    grids_path = os.path.join(d, "grids.txt")
    eval_path = os.path.join(d, "evaluations.csv")
    for p in (grids_path, eval_path):
        if not os.path.exists(p):
            print(f"[trim] {p} missing -- nothing to do")
            return 1

    lines = open(grids_path).read().splitlines()
    tops, new_lines, changed = {}, [], False

    for line in lines:
        m = re.match(r"^(\w+)\s+--sweep_lam\s+(.*)$", line)
        if not m or m.group(1) not in METHODS:
            new_lines.append(line)
            continue
        method, rest = m.group(1), m.group(2)
        note = ""
        if "#" in rest:
            rest, note = rest.split("#", 1)
            note = "  #" + note
        grid = [float(x) for x in rest.split()]
        pts = load_points(eval_path, method)
        k = knee(pts, a.tol_frac)
        if k is None:
            print(f"[trim] {method:9s} too few probe points to judge -- unchanged")
            new_lines.append(line)
            continue
        lam_top = k * a.margin
        tops[method] = lam_top
        if grid[-1] <= lam_top:
            print(f"[trim] {method:9s} knee {k:.3g} -> keep <= {lam_top:.3g}; "
                  f"grid already tops at {grid[-1]:.3g} -- UNCHANGED (still rising)")
            new_lines.append(line)
            continue
        new_grid = geomspace(grid[0], lam_top, len(grid))
        print(f"[trim] {method:9s} knee {k:.3g} -> cap {lam_top:.3g};  "
              f"top {grid[-1]:.3g} -> {new_grid[-1]:.3g}  ({len(grid)} points, unchanged count)")
        print(f"         was: {' '.join(fmt(x) for x in grid)}")
        print(f"         now: {' '.join(fmt(x) for x in new_grid)}")
        new_lines.append(f"{method:8s} --sweep_lam " + " ".join(fmt(x) for x in new_grid) + note)
        changed = True

    # shared has to span every method's region, so it is cut at the largest cap.
    if tops:
        cap = max(tops.values())
        for i, line in enumerate(new_lines):
            m = re.match(r"^shared\s+--sweep_lam\s+(.*)$", line)
            if not m:
                continue
            rest, note = m.group(1), ""
            if "#" in rest:
                rest, note = rest.split("#", 1)
                note = "  #" + note
            grid = [float(x) for x in rest.split()]
            if grid[-1] > cap:
                ng = geomspace(grid[0], cap, len(grid))
                print(f"[trim] shared    cap {cap:.3g} (max per-method);  "
                      f"top {grid[-1]:.3g} -> {ng[-1]:.3g}")
                new_lines[i] = "shared   --sweep_lam " + " ".join(fmt(x) for x in ng) + note
                changed = True

    if not changed:
        print("[trim] nothing to trim on this cell")
        return 0
    if not a.apply:
        print("[trim] dry run; pass --apply to write")
        return 0

    raw = os.path.join(d, "grids_raw.txt")
    if not os.path.exists(raw):          # never overwrite the first original
        os.replace(grids_path, raw)
    else:
        print(f"[trim] {raw} already exists; leaving it as the original")
    with open(grids_path, "w") as fh:
        fh.write("\n".join(new_lines) + "\n")
        fh.write("# TRIMMED by trim_grids.py (margin=%g, tol_frac=%g); "
                 "range finder's own output is in grids_raw.txt\n" % (a.margin, a.tol_frac))
    print(f"[trim] wrote {grids_path}  (original -> {raw})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
