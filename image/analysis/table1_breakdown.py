"""Table 1 and its per-problem breakdown: mean reward gain over a shared energy budget.

For each of the 12 problems (reward, prompt, seed) in
image/scripts/unconstrained/table1_jobs.tsv this computes, per method, the
metric G of the paper's eq. (auc):

  * keep each method's best run per lambda (highest terminal reward),
  * drop runs another run of the SAME method beats on both energy and reward,
  * interpolate the remaining frontier piecewise-linearly in log10(E),
  * average (r_hat(E) - r_unc) over [E1, E2], the energy range attained by all
    methods in the table (trapezoid on 2001 points in log10 E).

--baselines adds gradient guidance (gg, 3 guidance steps) and FlowChef
(flowchef, 5) as two more rows, from run_table1.sh's METHODS="gg flowchef"
runs. Adding methods can only narrow [E1, E2]; the script reports, per problem,
whether it moved from the paper's three-method budget (on the paper's
problems it does not: both baselines' frontiers extend past it at both ends,
so the paper's rows are unchanged).

E is the paper's control energy sum_k dt*||u_k||^2. summary.csv stores
`ctrl_energy` as the displacement residual sum_k ||dx_k||^2 dt with
dx_k = dt*u_k, which is E/N^2, so every row is rescaled by its own N^2
(64 at N=8). ConicFlow is taken at n_inner=24, as in the paper.

Writes to --out_dir:
  table1_breakdown.csv   one row per problem: G per method, r_unc, [E1, E2]
  table1_breakdown.tex   the appendix table (per-problem rows, per-reward
                         mean +/- s.e., ConicFlow win counts)
  table1.tex             Table 1 itself (per-reward mean +/- s.e.)

  python table1_breakdown.py                       # reads ../src/results/unconstrained
  python table1_breakdown.py --results <root> --check_paper
  python table1_breakdown.py --baselines           # + gradient guidance, FlowChef
  python table1_breakdown.py --results <root> <baselines_root> --baselines

--check_paper asserts the aggregates equal the numbers printed in the paper
(to 3 decimals). It holds on the paper's own runs; a fresh re-run will be
close but not digit-identical (HardFlow's energy in particular varies
between otherwise identical runs), so it is off by default and the
comparison is printed instead.
"""
import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
IMAGE = HERE.parent
METHODS = ("ConicFlow", "HardFlow", "FMRG")            # the paper's Table 1
BASELINES = ("GradientGuidance", "FlowChef")          # --baselines
LABEL = {"ConicFlow": "ConicFlow", "HardFlow": "HardFlow", "FMRG": "FMRG",
         "GradientGuidance": "Gradient guidance", "FlowChef": "FlowChef"}
REWARDS = ("clip", "imagereward", "pickscore")
CONICFLOW_TYPES = ("conicflow", "mpc")   # the method's CSV label; older runs say "mpc"

# Table 1 as printed in the paper (mean, s.e.), for --check_paper.
PAPER = {"clip":        {"ConicFlow": (0.130, 0.018), "HardFlow": (0.118, 0.021), "FMRG": (0.065, 0.008)},
         "imagereward": {"ConicFlow": (2.048, 0.455), "HardFlow": (1.279, 0.374), "FMRG": (1.278, 0.504)},
         "pickscore":   {"ConicFlow": (6.158, 0.728), "HardFlow": (1.736, 0.472), "FMRG": (2.670, 0.073)}}
REWARD_TEX = {"clip": "CLIP", "imagereward": "ImageReward", "pickscore": "PickScore"}
INSTANCE_TEX = {("a portrait of a man", 42): r"man $\to$ smiling, 42",
                ("a portrait of a man", 7): r"man $\to$ smiling, 7",
                ("a photo of a sleeping cat", 42): r"sleeping cat $\to$ open eyes, 42",
                ("a field of green grass", 42): r"green grass $\to$ red poppies, 42"}


def slug_of(prompt, edit):
    """Same rule as auto_sweep.sh / run_table1.sh: non-alphanumerics -> '_', 60 chars."""
    s = f"{prompt}__{edit}"
    return "".join(c if c.isascii() and c.isalnum() else "_" for c in s)[:60]


def read_jobs(path):
    with open(path) as fh:
        rows = [r for r in csv.reader(fh, delimiter="\t") if r and not r[0].startswith("#")]
    return [dict(reward=r[0], seed=int(r[1]), prompt=r[2], edit_prompt=r[3]) for r in rows]


def load_problem(roots, job):
    """Every run_ok row of one problem, across all method directories of every root."""
    roots = [roots] if isinstance(roots, (str, Path)) else roots
    dirs = [Path(r) / job["reward"] / slug_of(job["prompt"], job["edit_prompt"]) for r in roots]
    files = sorted(f for d in dirs for f in glob.glob(str(d / f"*_seed{job['seed']}" / "summary.csv")))
    if not files:
        raise FileNotFoundError(f"no runs for {job} under {[str(d) for d in dirs]}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[(df.run_ok.astype(str).str.lower() == "true") & (df.seed == job["seed"])]
    df = df[(df.prompt == job["prompt"]) & (df.edit_prompt == job["edit_prompt"])].copy()
    df["E"] = df.ctrl_energy * df.N.astype(float) ** 2
    return df


def frontier(d):
    d = d.sort_values("E")
    x = d.E.to_numpy(float)
    y = d.r_ctrl.to_numpy(float)
    keep = np.array([not np.any((x <= xi) & (y >= yi) & ((x < xi) | (y > yi))) for xi, yi in zip(x, y)])
    x, y = x[keep], y[keep]
    o = np.argsort(x)
    return x[o], y[o]


SELECT = {
    "ConicFlow": lambda df: df[df.control_type.isin(CONICFLOW_TYPES) & (df.n_inner == 24)],
    "HardFlow": lambda df: df[df.control_type == "hardflow"],
    "FMRG": lambda df: df[df.control_type == "fmrg"],
    "GradientGuidance": lambda df: df[(df.control_type == "gg") & (df.n_inner == 3)],
    "FlowChef": lambda df: df[(df.control_type == "flowchef") & (df.n_inner == 5)],
}


def g_metric(df, methods=METHODS):
    r_unc = df.r_unc.dropna().unique()
    if not np.allclose(r_unc, r_unc[0]):
        raise RuntimeError(f"r_unc disagrees across methods: {r_unc}")
    runc = float(r_unc[0])
    F, n_lam = {}, {}
    for name in methods:
        sub = SELECT[name](df)
        if sub.empty:
            raise RuntimeError(f"no {name} runs")
        n_lam[name] = sub.lam.nunique()
        sub = sub.loc[sub.groupby("lam").r_ctrl.idxmax()]
        F[name] = frontier(sub)
    E1 = max(f[0].min() for f in F.values())
    E2 = min(f[0].max() for f in F.values())
    if E2 <= E1:
        raise RuntimeError("empty shared energy budget")
    a, b = np.log10(E1), np.log10(E2)
    g = np.linspace(a, b, 2001)
    # The paper's three-method budget, to report whether added methods moved it.
    P1 = max(F[m][0].min() for m in METHODS if m in F)
    P2 = min(F[m][0].max() for m in METHODS if m in F)
    out = {"r_unc": runc, "E1": E1, "E2": E2, "decades": b - a,
           "budget_moved": not (np.isclose(E1, P1) and np.isclose(E2, P2))}
    for k, (x, y) in F.items():
        out[f"G_{k}"] = float(np.trapezoid(np.interp(g, np.log10(x), y) - runc, g) / (b - a))
        out[f"n_lambda_{k}"] = n_lam[k]
        out[f"n_frontier_{k}"] = len(x)
    return out


def fmt(v, best):
    s = f"{v:.3f}"
    return rf"$\mathbf{{{s}}}$" if best else f"${s}$"


def e_tex(v):
    return f"{v:.0f}" if v >= 99.5 else f"{v:.1f}"


def breakdown_tex(t, G, methods=METHODS):
    # Wrapper and spacing as typeset in the paper's appendix.
    others = [m for m in methods if m != "ConicFlow"]
    ncol = 3 + len(methods)
    lines = [r"\begin{table}[h]",
             r"\caption{Per-problem breakdown of Table~\ref{tab:objective-improvement}.}",
             r"  \centering", r"  \footnotesize", r"  \setlength{\tabcolsep}{2.5pt}",
             r"  \begin{tabular}{@{}lrr" + "c" * len(methods) + r"@{}}", r"    \toprule",
             r"    Instance (prompt $\to$ edit, seed) & $r_{\mathrm{unc}}$ & Budget $[E_1,E_2]$ & "
             + " & ".join(LABEL[m] for m in methods) + r" \\",
             r"    \midrule"]
    for rw in REWARDS:
        sub = t[t.reward == rw]
        lines.append(rf"    \multicolumn{{{ncol}}}{{l}}{{\textbf{{{REWARD_TEX[rw]}}}}} \\")
        for _, r in sub.iterrows():
            best = max(r[G])
            cells = " & ".join(fmt(r[c], r[c] == best) for c in G)
            lines.append(rf"    \quad {INSTANCE_TEX[(r.prompt, r.seed)]} & ${r.r_unc:.3f}$ & "
                         rf"$[{e_tex(r.E1)},\,{e_tex(r.E2)}]$ & {cells} \\")
        lines.append(rf"    \cmidrule(l){{1-{ncol}}}")
        n = len(sub)
        mean = {m: sub[f"G_{m}"].mean() for m in methods}
        se = {m: sub[f"G_{m}"].std(ddof=1) / np.sqrt(n) for m in methods}
        top = max(mean.values())
        lines.append(r"    \quad \textit{mean} $\pm$ s.e. & & & " +
                     " & ".join((rf"$\mathbf{{{mean[m]:.3f}}} \pm {se[m]:.3f}$" if mean[m] == top
                                 else rf"${mean[m]:.3f} \pm {se[m]:.3f}$") for m in methods) + r" \\")
        wins = {m: int((sub.G_ConicFlow > sub[f"G_{m}"]).sum()) for m in others}
        lines.append(r"    \quad \textit{ConicFlow wins} & & & -- & "
                     + " & ".join(f"${wins[m]}/{n}$" for m in others) + r" \\")
        lines.append(r"    \midrule")
    tot = {m: int((t.G_ConicFlow > t[f"G_{m}"]).sum()) for m in others}
    lines.append(rf"    \textit{{ConicFlow wins, all {len(t)} problems}} & & & -- & "
                 + " & ".join(f"${tot[m]}/{len(t)}$" for m in others) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}",
              r"  \label{tab:objective-improvement-full}", r"\end{table}"]
    return "\n".join(lines) + "\n", tot


def table1_tex(agg, methods=METHODS):
    lines = [r"  \begin{tabular}{lccc}", r"    \toprule",
             r"    Method & CLIP $(\uparrow)$ & ImageReward $(\uparrow)$ & PickScore $(\uparrow)$ \\",
             r"    \midrule"]
    best = {rw: max(methods, key=lambda m: agg[rw][m][0]) for rw in REWARDS}
    for m in methods:
        cells = []
        for rw in REWARDS:
            mu, se = agg[rw][m]
            cells.append(rf"$\mathbf{{{mu:.3f}}} \pm {se:.3f}$" if best[rw] == m else rf"${mu:.3f} \pm {se:.3f}$")
        lines.append(f"    {LABEL[m]:9s} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", default=[str(IMAGE / "src" / "results" / "unconstrained")],
                    help="root(s) holding <reward>/<slug>/<method>_seed<seed>/summary.csv "
                         "(run_table1.sh's RESULTS_ROOT, relative to image/src); runs from "
                         "every root are pooled")
    ap.add_argument("--baselines", action="store_true",
                    help="add gradient guidance and FlowChef as rows (run_table1.sh METHODS='gg flowchef')")
    ap.add_argument("--jobs", default=str(IMAGE / "scripts" / "unconstrained" / "table1_jobs.tsv"))
    ap.add_argument("--out_dir", default=None, help="default: <results>/figures")
    ap.add_argument("--check_paper", action="store_true",
                    help="raise unless the aggregates equal the paper's Table 1 to 3 decimals")
    a = ap.parse_args()
    out = Path(a.out_dir or Path(a.results[0]) / "figures")
    methods = METHODS + (BASELINES if a.baselines else ())
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for job in read_jobs(a.jobs):
        df = load_problem(a.results, job)
        try:
            r = g_metric(df, methods)
        except RuntimeError as e:
            raise RuntimeError(f"{job}: {e}") from None
        rows.append({"reward": job["reward"], "prompt": job["prompt"], "edit_prompt": job["edit_prompt"],
                     "seed": job["seed"], **r})
    t = pd.DataFrame(rows)
    G = [f"G_{m}" for m in methods]
    t["winner"] = t[G].idxmax(axis=1).str[2:]
    t.to_csv(out / "table1_breakdown.csv", index=False)

    agg = {}
    for rw in REWARDS:
        sub = t[t.reward == rw]
        agg[rw] = {m: (sub[f"G_{m}"].mean(), sub[f"G_{m}"].std(ddof=1) / np.sqrt(len(sub))) for m in methods}

    print("Table 1 -- mean reward gain over a shared energy budget (mean +/- s.e.)")
    print(f"{'':12s}" + "".join(f"{REWARD_TEX[rw]:>26s}" for rw in REWARDS))
    for m in methods:
        print(f"{LABEL[m][:17]:17s}" + "".join(f"{agg[rw][m][0]:7.3f} +/- {agg[rw][m][1]:.3f}       " for rw in REWARDS))
    print(f"{'(paper)':12s}")
    for m in METHODS:
        print(f"{m:12s}" + "".join(f"{PAPER[rw][m][0]:12.3f} +/- {PAPER[rw][m][1]:.3f}  " for rw in REWARDS))

    if a.check_paper:
        for rw, per in PAPER.items():
            for m, (mu, se) in per.items():
                got_mu, got_se = agg[rw][m]
                if round(got_mu, 3) != mu or round(got_se, 3) != se:
                    raise RuntimeError(f"{rw}/{m}: recomputed {got_mu:.3f}+-{got_se:.3f}, paper {mu}+-{se}")
        print("--check_paper: aggregates match Table 1 exactly.")

    pd.set_option("display.width", 250)
    print()
    print(t[["reward", "prompt", "seed", "r_unc", "E1", "E2", "decades", *G, "winner"]].round(3).to_string(index=False))
    if a.baselines:
        moved = t[t.budget_moved]
        print(f"\nshared budget moved from the paper's three-method one on {len(moved)} of {len(t)} problems"
              + ("" if moved.empty else ": " + "; ".join(f"{r.reward} {r.prompt} s{r.seed}" for r in moved.itertuples())))

    tex, tot = breakdown_tex(t, G, methods)
    (out / "table1_breakdown.tex").write_text(tex)
    (out / "table1.tex").write_text(table1_tex(agg, methods))
    print(f"\nConicFlow wins over all {len(t)} problems: {tot}")
    print(f"wrote {out}/table1_breakdown.csv, table1_breakdown.tex, table1.tex")


if __name__ == "__main__":
    main()
