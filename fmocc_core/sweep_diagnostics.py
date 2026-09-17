#!/usr/bin/env python3
"""
sweep_diagnostics.py -- post-sweep comparison of MPC / HardFlow / FMRG.

Two diagnostics, either or both:

  win rates   For every (lambda, seed) combo in which all three methods ran,
              which one reached the lowest objective? Reports the share of
              combos each method wins, plus MPC-vs-FMRG, MPC-vs-HardFlow and
              MPC-vs-both.

  pareto      Reward against control energy, scattered per method, with each
              method's own non-dominated frontier drawn and the globally
              non-dominated points ringed.

WHAT MAKES THE WIN-RATE COMPARISON VALID, and when it is not. The objective
  J = ctrl_energy / (2*lam) - r_ctrl        (LOWER is better)
is only a common yardstick when the three runs share a lambda, because lam is
what weights energy against reward in it. At a shared lam they really are
scoring the same first-order problem: MPC's per-step linearised optimum is
resid* = lam*dt*grad r, FMRG's update is exactly that per step, and HardFlow's
reg = N/(2*lam) matches it (see lambda_autorange.shared_grid). So combos are
keyed on (lam, seed) and a method is only compared against runs at ITS OWN lam.

A sweep that gave each method its own lambda grid therefore yields ZERO combos,
and this script says so rather than inventing cross-lambda comparisons. Use
auto_sweep.sh's SHARED_GRID=1 if win rates are what you want.

The Pareto view has no such requirement: energy and reward are directly
comparable whatever lambda produced them, which is why it is the safer default
of the two.

Runs are grouped by (reward, prompt, edit_prompt, N) first -- comparing across
those is comparing different problems.

Usage:
  python sweep_diagnostics.py --runs results/my_sweep/            # both
  python sweep_diagnostics.py --runs 'results/auto/*/*/' --what winrates
  python sweep_diagnostics.py --runs results/x/ --what pareto --out_dir figs/
"""
import argparse
import csv
import glob
import itertools
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_methods import METHODS, PRETTY, load_sweeps, load_autorange  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# Win rates
# ═════════════════════════════════════════════════════════════════════════════
def combos(runs: List[dict]) -> Dict[tuple, Dict[str, dict]]:
    """(reward, prompt, edit, N, lam, seed) -> {method: best run at that key}.

    If a sweep repeated a (method, lam, seed) -- a re-run, or two out_dirs
    globbed together -- the BEST (lowest J) is kept, so a duplicate cannot
    inflate a method's win rate by giving it two chances at the same combo.
    """
    out: Dict[tuple, Dict[str, dict]] = defaultdict(dict)
    for r in runs:
        if not math.isfinite(r["J"]):
            continue
        key = (r["reward"], r["prompt"], r["edit_prompt"], r["N"], r["lam"], r["seed"])
        prev = out[key].get(r["method"])
        if prev is None or r["J"] < prev["J"]:
            out[key][r["method"]] = r
    return out


def win_rates(runs: List[dict]) -> dict:
    full, partial = {}, 0
    for key, bym in combos(runs).items():
        if len(bym) == len(METHODS):
            full[key] = bym
        else:
            partial += 1
    n = len(full)
    res = dict(n_combos=n, n_partial=partial,
               lowest={m: 0 for m in METHODS},
               mpc_beats={"fmrg": 0, "hardflow": 0, "both": 0}, rows=[])
    for key, bym in sorted(full.items()):
        best = min(METHODS, key=lambda m: bym[m]["J"])
        res["lowest"][best] += 1
        bf = bym["mpc"]["J"] < bym["fmrg"]["J"]
        bh = bym["mpc"]["J"] < bym["hardflow"]["J"]
        res["mpc_beats"]["fmrg"] += bf
        res["mpc_beats"]["hardflow"] += bh
        res["mpc_beats"]["both"] += (bf and bh)
        res["rows"].append(dict(
            reward=key[0], prompt=key[1], edit_prompt=key[2], N=key[3], lam=key[4], seed=key[5],
            winner=best, mpc_beats_fmrg=bf, mpc_beats_hardflow=bh, mpc_beats_both=bf and bh,
            **{f"J_{m}": bym[m]["J"] for m in METHODS},
            **{f"r_ctrl_{m}": bym[m]["r_ctrl"] for m in METHODS},
            **{f"energy_{m}": bym[m]["E"] for m in METHODS}))
    return res


def print_win_rates(res: dict, log=print) -> None:
    n = res["n_combos"]
    if not n:
        log("\n  win rates: NO (lambda, seed) combo has all three methods.")
        if res["n_partial"]:
            log(f"    {res['n_partial']} combo(s) had only some methods. The usual cause is a "
                f"per-method lambda grid -- the objective is only comparable at a SHARED lambda, "
                f"so run the sweep with auto_sweep.sh SHARED_GRID=1 to get this diagnostic.")
        return
    pct = lambda k: f"{100.0 * k / n:5.1f}%  ({k}/{n})"
    log(f"\n  WIN RATES over {n} (lambda, seed) combo(s) with all three methods"
        + (f"   [{res['n_partial']} incomplete combo(s) ignored]" if res["n_partial"] else ""))
    log("    lowest objective J = energy/(2*lambda) - r_ctrl   (lower is better)")
    for m in sorted(METHODS, key=lambda m: -res["lowest"][m]):
        log(f"      {PRETTY[m]:9} {pct(res['lowest'][m])}")
    log("    head to head")
    log(f"      MPC beats FMRG      {pct(res['mpc_beats']['fmrg'])}")
    log(f"      MPC beats HardFlow  {pct(res['mpc_beats']['hardflow'])}")
    log(f"      MPC beats both      {pct(res['mpc_beats']['both'])}")


def write_win_rates_csv(res: dict, path: str) -> Optional[str]:
    if not res["rows"]:
        return None
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res["rows"][0].keys()))
        w.writeheader()
        w.writerows(res["rows"])
    return path


# ═════════════════════════════════════════════════════════════════════════════
# Pareto
# ═════════════════════════════════════════════════════════════════════════════
def pareto_front(points: List[dict]) -> List[dict]:
    """Non-dominated points: nothing else has >= reward AND <= energy, strictly
    better in one. Up and to the LEFT is better."""
    front = []
    for p in points:
        if not any(q is not p and q["r_ctrl"] >= p["r_ctrl"] and q["E"] <= p["E"]
                   and (q["r_ctrl"] > p["r_ctrl"] or q["E"] < p["E"]) for q in points):
            front.append(p)
    return sorted(front, key=lambda p: p["E"])


def plot_pareto(runs: List[dict], out_dir: str, dpi: int = 130, log=print) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"mpc": "#2a78d6", "hardflow": "#eb6834", "fmrg": "#1baf7a"}
    MK = {"mpc": "o", "hardflow": "s", "fmrg": "^"}

    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in runs:
        if math.isfinite(r["E"]) and math.isfinite(r["r_ctrl"]) and r["E"] > 0:
            groups[(r["reward"], r["prompt"], r["edit_prompt"], r["N"])].append(r)
    made = []
    for (reward, prompt, edit, N), grp in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        glob_front = {id(p) for p in pareto_front(grp)}
        for m in METHODS:
            pts = [p for p in grp if p["method"] == m]
            if not pts:
                continue
            ax.scatter([p["E"] for p in pts], [p["r_ctrl"] for p in pts], s=46, marker=MK[m],
                       facecolor=COL[m], edgecolor="white", linewidth=.8, zorder=3,
                       label=f"{PRETTY[m]} ({len(pts)})")
            f = pareto_front(pts)
            if len(f) > 1:
                ax.plot([p["E"] for p in f], [p["r_ctrl"] for p in f], "-",
                        color=COL[m], lw=1.6, alpha=.55, zorder=2)
        gf = [p for p in grp if id(p) in glob_front]
        if gf:
            ax.scatter([p["E"] for p in gf], [p["r_ctrl"] for p in gf], s=190,
                       facecolor="none", edgecolor="#151d24", linewidth=1.3, zorder=4,
                       label="globally non-dominated")
        r_unc = next((p["r_unc"] for p in grp if math.isfinite(p["r_unc"])), None)
        if r_unc is not None:
            ax.axhline(r_unc, color="#8a95a0", ls="--", lw=1,
                       label=f"uncontrolled r = {r_unc:.3f}")
        ax.set_xscale("log")
        ax.set_xlabel(r"control energy  $\Sigma\|u\|^2\Delta t$   (lower is better)")
        ax.set_ylabel("achieved reward  $r_{ctrl}$   (higher is better)")
        ax.set_title(f'{reward}   ·   "{prompt}" → "{edit}"   ·   N={N}\n'
                     "up and to the left is better", fontsize=10.5)
        ax.grid(alpha=.25, lw=.6)
        ax.legend(fontsize=8.5, loc="lower right")
        fig.tight_layout()
        p = "".join(c if c.isalnum() else "_" for c in f"{prompt}__{edit}")[:48]
        path = os.path.join(out_dir, f"pareto__{reward}__{p}__N{N}.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        made.append(path)
        log(f"    -> {path}")
    return made


# ═════════════════════════════════════════════════════════════════════════════
def run_diagnostics(run_dirs: List[str], what: str = "both", out_dir: Optional[str] = None,
                    dpi: int = 130, log=print) -> dict:
    """Entry point used both by __main__ and by run_sweep.py's --post_diagnostics."""
    runs = load_sweeps(run_dirs)
    if not runs:
        log("  diagnostics: no run_ok rows of mpc/hardflow/fmrg found.")
        return {}
    out_dir = out_dir or (run_dirs[0] if os.path.isdir(run_dirs[0]) else ".")
    os.makedirs(out_dir, exist_ok=True)
    methods_seen = sorted({r["method"] for r in runs})
    log(f"\n  diagnostics: {len(runs)} run(s), methods present: {methods_seen}")
    res: dict = {"n_runs": len(runs), "methods": methods_seen}
    if what in ("winrates", "both"):
        wr = win_rates(runs)
        print_win_rates(wr, log=log)
        p = write_win_rates_csv(wr, os.path.join(out_dir, "winrates.csv"))
        if p:
            log(f"    -> {p}")
        res["winrates"] = wr
    if what in ("pareto", "both"):
        log("\n  PARETO reward vs control energy")
        res["pareto_figs"] = plot_pareto(runs, out_dir, dpi=dpi, log=log)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, metavar="DIR",
                    help="sweep dirs or globs containing summary.csv")
    ap.add_argument("--what", default="both", choices=["winrates", "pareto", "both"])
    ap.add_argument("--out_dir", default=None,
                    help="where winrates.csv and the pareto PNGs go (default: the first --runs dir)")
    ap.add_argument("--dpi", type=int, default=130)
    a = ap.parse_args()
    run_diagnostics(a.runs, what=a.what, out_dir=a.out_dir, dpi=a.dpi)


if __name__ == "__main__":
    main()
