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

Each diagnostic has its OWN flag (--diag_winrates, --diag_pareto, ...) rather
than one enum, so adding another is a single row in the DIAGS table below: the
standalone CLI flag, run_sweep.py's flag and auto_sweep.sh's env var are all
derived from it.

Usage:
  python sweep_diagnostics.py --runs results/my_sweep/ --all
  python sweep_diagnostics.py --runs 'results/auto/*/*/' --diag_winrates
  python sweep_diagnostics.py --runs results/x/ --diag_pareto --out_dir figs/
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
from compare_methods import METHODS, PRETTY, load_sweeps, load_autorange, cfg_name  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# Win rates
# ═════════════════════════════════════════════════════════════════════════════
def combos(runs: List[dict]) -> Dict[tuple, Dict[str, dict]]:
    """(reward, prompt, edit, N, lam, seed) -> {method: that method's BEST run there}.

    A method may have several configs at one combo (MPC at n_inner=12 and 24,
    say). Each method is represented by its lowest-J config at that combo --
    "best config per method per combo" -- and the chosen config is recorded on
    the run (`n_cands`, and the row's cfg_<method> column) rather than hidden.
    That is a max over a method's configs, so a method given more configs gets
    more draws; print_win_rates says so whenever the candidate counts differ.
    """
    out: Dict[tuple, Dict[str, dict]] = defaultdict(dict)
    cands: Dict[tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in runs:
        if not math.isfinite(r["J"]):
            continue
        key = (r["reward"], r["prompt"], r["edit_prompt"], r["N"], r["lam"], r["seed"])
        cands[key][r["method"]] += 1
        prev = out[key].get(r["method"])
        if prev is None or r["J"] < prev["J"]:
            out[key][r["method"]] = r
    for key, bym in out.items():
        for m, r in bym.items():
            r["n_cands"] = cands[key][m]
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
               mpc_beats={"fmrg": 0, "hardflow": 0, "both": 0}, rows=[],
               # per method: which config represented it, and how often
               cfg_wins={m: defaultdict(int) for m in METHODS},
               cands={m: set() for m in METHODS})
    for key, bym in sorted(full.items()):
        for m in METHODS:
            res["cfg_wins"][m][cfg_name(bym[m])] += 1
            res["cands"][m].add(bym[m].get("n_cands", 1))
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
            **{f"cfg_{m}": bym[m].get("cfg_label", "") for m in METHODS},
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
    # Gate on how many configs a method HAD, not on how many of them ever won:
    # one config sweeping every combo is precisely when the unequal-draws note
    # below is worth reading.
    nc = {m: max(res["cands"][m]) for m in METHODS}
    multi = {m: w for m, w in res["cfg_wins"].items() if nc[m] > 1}
    if multi:
        log("    best config per method per combo (a method with several configs is its lowest-J one at each combo)")
        for m, w in multi.items():
            log(f"      {PRETTY[m]:9} " + ", ".join(f"{k.split(' ', 1)[1] if ' ' in k else k}: {v}/{n}"
                                                 for k, v in sorted(w.items(), key=lambda kv: -kv[1])))
        if len(set(nc.values())) > 1:
            log(f"      NOTE: candidate configs per combo differ -- {', '.join(f'{PRETTY[m]} {c}' for m, c in nc.items())}. "
                f"Best-of is a max over a method's configs, so a method with more configs gets more draws.")


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


def _shades(base_hex: str, k: int) -> List[str]:
    """k shades of one hue, base colour first then progressively lighter (toward
    white), so several configs of one method stay visibly one method while the
    three methods keep their distinct hues."""
    from matplotlib.colors import to_rgb, to_hex
    r, g, b = to_rgb(base_hex)
    if k <= 1:
        return [base_hex]
    out = []
    for i in range(k):
        t = 0.5 * i / (k - 1)                       # 0 = base, up to 50% toward white
        out.append(to_hex((r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)))
    return out


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
            mruns = [p for p in grp if p["method"] == m]
            if not mruns:
                continue
            cfgs = sorted({p["cfg"] for p in mruns})
            shades = _shades(COL[m], len(cfgs))
            for cfg, col in zip(cfgs, shades):
                pts = [p for p in mruns if p["cfg"] == cfg]
                ax.scatter([p["E"] for p in pts], [p["r_ctrl"] for p in pts], s=46, marker=MK[m],
                           facecolor=col, edgecolor="white", linewidth=.8, zorder=3,
                           label=f"{cfg_name(pts[0])} ({len(pts)})")
                f = pareto_front(pts)
                if len(f) > 1:
                    ax.plot([p["E"] for p in f], [p["r_ctrl"] for p in f], "-",
                            color=col, lw=1.6, alpha=.6, zorder=2)
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
        ax.set_title(f'{reward}   \u00b7   "{prompt}" \u2192 "{edit}"   \u00b7   N={N}\n'
                     "up and to the left is better", fontsize=10.5)
        ax.grid(alpha=.25, lw=.6)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        p = "".join(c if c.isalnum() else "_" for c in f"{prompt}__{edit}")[:48]
        path = os.path.join(out_dir, f"pareto__{reward}__{p}__N{N}.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        made.append(path)
        log(f"    -> {path}")
    return made


# ═════════════════════════════════════════════════════════════════════════════
def _run_winrates(runs: List[dict], out_dir: str, dpi: int, log) -> dict:
    wr = win_rates(runs)
    print_win_rates(wr, log=log)
    p = write_win_rates_csv(wr, os.path.join(out_dir, "winrates.csv"))
    if p:
        log(f"    -> {p}")
    return wr


def _run_pareto(runs: List[dict], out_dir: str, dpi: int, log) -> dict:
    log("\n  PARETO reward vs control energy")
    return {"figures": plot_pareto(runs, out_dir, dpi=dpi, log=log)}


# One entry per diagnostic. Adding another means adding a row HERE and nothing
# else: the standalone CLI flag, run_sweep.py's flag and auto_sweep.sh's env var
# are all derived from this table, so they cannot drift apart or be forgotten.
DIAGS: Dict[str, dict] = {
    "winrates": dict(
        fn=_run_winrates,
        help="For every (lambda, seed) combo in which all three methods ran, which reached the "
             "lowest objective J = energy/(2*lambda) - r_ctrl, plus MPC vs FMRG / HardFlow / "
             "both. Writes winrates.csv. REQUIRES a shared lambda grid across the methods -- J "
             "is only comparable at equal lambda -- otherwise it reports zero combos and why."),
    "pareto": dict(
        fn=_run_pareto,
        help="Reward vs control energy scatter, per-method non-dominated frontiers, globally "
             "non-dominated points ringed. No shared-lambda requirement."),
}


def add_diag_args(parser, group_title: str = "Diagnostics") -> None:
    """Register one --diag_<name> flag per entry in DIAGS, on any parser."""
    g = parser.add_argument_group(group_title)
    for name, spec in DIAGS.items():
        g.add_argument(f"--diag_{name}", action="store_true", help=spec["help"])


def selected_diags(args) -> List[str]:
    """Which --diag_* flags are set, in DIAGS order."""
    return [n for n in DIAGS if getattr(args, f"diag_{n}", False)]


def run_diagnostics(run_dirs: List[str], which: List[str], out_dir: Optional[str] = None,
                    dpi: int = 130, log=print) -> dict:
    """Entry point for the CLI, run_sweep.py and auto_sweep.sh alike."""
    unknown = [w for w in which if w not in DIAGS]
    if unknown:
        raise ValueError(f"unknown diagnostic(s) {unknown}; known: {sorted(DIAGS)}")
    if not which:
        return {}
    runs = load_sweeps(run_dirs)
    if not runs:
        log("  diagnostics: no run_ok rows of mpc/hardflow/fmrg found.")
        return {}
    out_dir = out_dir or (run_dirs[0] if os.path.isdir(run_dirs[0]) else ".")
    os.makedirs(out_dir, exist_ok=True)
    methods_seen = sorted({r["method"] for r in runs})
    log(f"\n  diagnostics {which}: {len(runs)} run(s), methods present: {methods_seen}")
    res: dict = {"n_runs": len(runs), "methods": methods_seen}
    for name in which:
        res[name] = DIAGS[name]["fn"](runs, out_dir, dpi, log)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, metavar="DIR",
                    help="sweep dirs or globs containing summary.csv")
    add_diag_args(ap)
    ap.add_argument("--all", action="store_true", help="run every diagnostic in DIAGS")
    ap.add_argument("--out_dir", default=None,
                    help="where outputs go (default: the first --runs dir)")
    ap.add_argument("--dpi", type=int, default=130)
    a = ap.parse_args()
    which = list(DIAGS) if a.all else selected_diags(a)
    if not which:
        ap.error("pick at least one diagnostic: "
                 + ", ".join(f"--diag_{n}" for n in DIAGS) + ", or --all")
    run_diagnostics(a.runs, which, out_dir=a.out_dir, dpi=a.dpi)


if __name__ == "__main__":
    main()
