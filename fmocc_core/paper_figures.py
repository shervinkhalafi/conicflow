#!/usr/bin/env python
"""Regenerate every unconstrained-section figure and number from results/auto/.

Backs `paper_figures.ipynb`. The logic lives here rather than in notebook cells
so it can be run headless (`python paper_figures.py`) and so the notebook stays
short enough to read.

THREE RULES THIS ENCODES, all of them decisions from calls rather than taste:

1. SEEDS ARE NEVER POOLED. A (reward, prompt, seed) triple is one optimization
   problem -- a different seed is a different x0, a different uncontrolled image,
   sometimes a different subject. Averaging or co-plotting them was rejected on
   the 2026-09-17 call. Every plot and every win rate here is per-seed.

2. PLOTS COME FROM sweep_diagnostics.py / compare_methods.py, not from a reimpl.
   Those are the tools Shervin already reviews; a second renderer would drift.
   This module shells out to them and only does the aggregation they do not.

3. WIN RATES ARE COMPUTED FROM summary.csv, NOT BY SUMMING winrates.csv.
   There are two generations of winrates.csv on disk (pooled-seed under
   diagnostics/, per-seed under diag_seed*/), and adding them double-counts.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
AUTO = HERE / "results" / "auto"
FIGS = HERE / "results" / "paper_figures"

METHODS = ("mpc", "hardflow", "fmrg")


# ── loading ───────────────────────────────────────────────────────────────────
def load_rows(auto: Path = AUTO) -> pd.DataFrame:
    """Every successful row under results/auto/<reward>/<slug>/<method>_seed<N>/."""
    frames = []
    for csv_path in sorted(auto.glob("*/*/*_seed*/summary.csv")):
        df = pd.read_csv(csv_path)
        df["combo_dir"] = str(csv_path.parent.parent)
        frames.append(df)
    if not frames:
        raise SystemExit(f"no summary.csv under {auto}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["run_ok"].astype(str).str.lower() == "true"].copy()
    for c in ("lam", "r_ctrl", "r_unc", "delta_r", "ctrl_energy", "objective",
              "time_ctrl_s", "N", "n_inner"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["seed"] = df["seed"].astype(str)
    # MPC runs several n_inner configs; label them so they stay distinguishable
    df["config"] = df.apply(
        lambda r: f"mpc ni={int(r['n_inner'])}" if r["control_type"] == "mpc"
        else r["control_type"], axis=1)
    return df


# ── win rates ─────────────────────────────────────────────────────────────────
def win_rates(df: pd.DataFrame, energy_cap: float | None = None):
    """Per-(reward, prompt, seed, lam) cell: does MPC's best config beat each baseline?

    `energy_cap` drops cells where ANY method's control energy exceeded it. Some
    of MPC's margin comes from HardFlow/FMRG diverging at high lambda rather than
    from MPC being better; capping shows how much. Report both.
    """
    key = ["reward", "prompt", "seed", "lam"]
    best_mpc = (df[df.control_type == "mpc"]
                .sort_values("objective").groupby(key, as_index=False).first())
    out = best_mpc[key + ["objective", "ctrl_energy", "config"]].rename(
        columns={"objective": "J_mpc", "ctrl_energy": "E_mpc", "config": "cfg_mpc"})
    for m in ("hardflow", "fmrg"):
        sub = (df[df.control_type == m]
               .sort_values("objective").groupby(key, as_index=False).first())
        out = out.merge(sub[key + ["objective", "ctrl_energy"]].rename(
            columns={"objective": f"J_{m}", "ctrl_energy": f"E_{m}"}), on=key, how="inner")
    if energy_cap is not None:
        emax = out[["E_mpc", "E_hardflow", "E_fmrg"]].max(axis=1)
        out = out[emax <= energy_cap].copy()
    # objective is MINIMISED (energy/(2 lam) - r_ctrl): lower is better
    out["beats_hardflow"] = out.J_mpc < out.J_hardflow
    out["beats_fmrg"] = out.J_mpc < out.J_fmrg
    out["beats_both"] = out.beats_hardflow & out.beats_fmrg
    return out


def win_rate_summary(df: pd.DataFrame, caps=(None, 100.0, 10.0)) -> pd.DataFrame:
    rows = []
    for cap in caps:
        w = win_rates(df, cap)
        n = len(w)
        rows.append(dict(
            filter="all cells" if cap is None else f"excl. any-method E>{cap:g}",
            n=n,
            vs_hardflow=f"{w.beats_hardflow.sum()}/{n} ({100*w.beats_hardflow.mean():.0f}%)",
            vs_fmrg=f"{w.beats_fmrg.sum()}/{n} ({100*w.beats_fmrg.mean():.0f}%)",
            vs_both=f"{w.beats_both.sum()}/{n} ({100*w.beats_both.mean():.0f}%)"))
    return pd.DataFrame(rows)


SHORT_PROMPT = {
    "a portrait of a man": "portrait of a man",
    "a photo of a sleeping cat": "sleeping cat",
    "a field of green grass": "field of green grass",
}


def win_rate_by_reward(df: pd.DataFrame) -> pd.DataFrame:
    """One row per reward, aggregated over prompts, seeds and lambdas."""
    w = win_rates(df)
    g = w.groupby("reward")
    out = g.agg(n=("beats_hardflow", "size"),
                hf=("beats_hardflow", "sum"),
                fm=("beats_fmrg", "sum")).reset_index()
    out["vs_hardflow"] = out.apply(lambda r: f"{r.hf}/{r.n} ({100*r.hf/r.n:.0f}%)", axis=1)
    out["vs_fmrg"] = out.apply(lambda r: f"{r.fm}/{r.n} ({100*r.fm/r.n:.0f}%)", axis=1)
    return out[["reward", "n", "vs_hardflow", "vs_fmrg"]]


def win_rate_latex_compact(df: pd.DataFrame, label: str = "tab:winrate") -> str:
    """Reward x baseline only. Prompt/seed/lambda are aggregated away, so the
    per-combo breakdown (win_rate_by_combo) belongs in the appendix -- without it
    a reader cannot see that the totals are weighted toward the densified
    PickScore-portrait combos."""
    w = win_rates(df)
    by = win_rate_by_reward(df)
    n = len(w)
    pretty = {"clip": "CLIP", "imagereward": "ImageReward", "pickscore": "PickScore"}
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Fraction of problem instances on which our method attains a "
        r"lower objective than each baseline. An instance is one "
        r"$(\text{prompt}, \text{seed}, \lambda)$ tuple at $N=8$, $H=2$; seeds are "
        r"treated as distinct problems rather than repeats.}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lcc}", r"\toprule",
        r"Reward & vs.\ HardFlow & vs.\ FMRG \\", r"\midrule",
    ]
    esc = lambda s: s.replace("%", "\\%")      # f-strings cannot hold a backslash
    for _, r in by.iterrows():
        lines.append(f"{pretty.get(r['reward'], r['reward'])} & "
                     f"{esc(r['vs_hardflow'])} & {esc(r['vs_fmrg'])} \\\\")
    lines += [
        r"\midrule",
        rf"\textbf{{All}} & "
        rf"\textbf{{{w.beats_hardflow.sum()}/{n} ({100*w.beats_hardflow.mean():.0f}\%)}} & "
        rf"\textbf{{{w.beats_fmrg.sum()}/{n} ({100*w.beats_fmrg.mean():.0f}\%)}} \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    return "\n".join(lines)


def win_rate_latex(df: pd.DataFrame, label: str = "tab:winrate") -> str:
    """booktabs table of per-combo win rates. Regenerate rather than hand-edit:
    these numbers move whenever a baseline config changes (e.g. giving FMRG its
    own t_stop=0.25 early stopping, which it is currently missing)."""
    w = win_rates(df)
    bc = win_rate_by_combo(df)
    n = len(w)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Per-cell win rate of our method against each baseline on the "
        r"image editing task. A cell is one $(\text{reward}, \text{prompt}, "
        r"\text{seed}, \lambda)$ tuple; our method wins a cell when it attains a "
        r"lower objective $J = \tfrac{1}{2\lambda}\sum_k\|u_k\|^2\Delta t - r(x_1)$. "
        r"Seeds are reported separately rather than averaged: a seed fixes $x_0$ "
        r"and therefore defines a distinct optimization problem. All runs use "
        r"$N=8$, $H=2$, and a shared $\lambda$ grid within each combination.}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Reward & Prompt & Seed & Cells & vs.\ HardFlow & vs.\ FMRG \\",
        r"\midrule",
    ]
    prev = None
    for _, r in bc.iterrows():
        if prev is not None and r["reward"] != prev:
            lines.append(r"\addlinespace")
        prev = r["reward"]
        rw = {"clip": "CLIP", "imagereward": "ImageReward",
              "pickscore": "PickScore"}.get(r["reward"], r["reward"])
        pr = SHORT_PROMPT.get(r["prompt"], r["prompt"])
        lines.append(f"{rw} & {pr} & {r['seed']} & {r['n']} & "
                     f"{r['vs_hardflow']}/{r['n']} & {r['vs_fmrg']}/{r['n']} \\\\")
    lines += [
        r"\midrule",
        rf"\textbf{{Total}} & & & \textbf{{{n}}} & "
        rf"\textbf{{{w.beats_hardflow.sum()}/{n} ({100*w.beats_hardflow.mean():.0f}\%)}} & "
        rf"\textbf{{{w.beats_fmrg.sum()}/{n} ({100*w.beats_fmrg.mean():.0f}\%)}} \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    return "\n".join(lines)


def win_rate_by_combo(df: pd.DataFrame) -> pd.DataFrame:
    w = win_rates(df)
    g = w.groupby(["reward", "prompt", "seed"])
    return (g.agg(n=("beats_hardflow", "size"),
                  vs_hardflow=("beats_hardflow", "sum"),
                  vs_fmrg=("beats_fmrg", "sum"))
            .reset_index().sort_values(["reward", "prompt", "seed"]))


# ── Pareto frontier ownership ─────────────────────────────────────────────────
def _nondominated(pts):
    """pts: list of (reward, energy). Non-dominated = nothing is both >=r and <=E."""
    keep = []
    for i, (r, e) in enumerate(pts):
        if not any((r2 >= r and e2 <= e and (r2 > r or e2 < e))
                   for j, (r2, e2) in enumerate(pts) if j != i):
            keep.append(i)
    return keep


def frontier_share(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of each seed's non-dominated set owned by MPC."""
    rows = []
    for (rw, pr, sd), g in df.groupby(["reward", "prompt", "seed"]):
        if g.control_type.nunique() < len(METHODS):
            continue                      # incomplete: a missing baseline is free frontier
        pts = list(zip(g.r_ctrl, g.ctrl_energy))
        nd = _nondominated(pts)
        methods = g.control_type.tolist()
        cfgs = g.config.tolist()
        mpc = sum(1 for i in nd if methods[i] == "mpc")
        rows.append(dict(reward=rw, prompt=pr[:40], seed=sd, n_frontier=len(nd),
                         mpc_share=mpc / len(nd) if nd else float("nan"),
                         mpc_cfgs=", ".join(sorted({cfgs[i] for i in nd
                                                    if methods[i] == "mpc"}))))
    return pd.DataFrame(rows).sort_values("mpc_share", ascending=False)


def energy_at_reward(df: pd.DataFrame, reward: str, prompt: str, seed: str, targets):
    """Cheapest energy each method needs to REACH a reward level. The headline table."""
    g = df[(df.reward == reward) & (df.prompt == prompt) & (df.seed == str(seed))]
    rows = []
    for t in targets:
        row = {"r>=": t}
        for cfg in sorted(g.config.unique()):
            sub = g[(g.config == cfg) & (g.r_ctrl >= t)]
            row[cfg] = round(sub.ctrl_energy.min(), 3) if len(sub) else None
        rows.append(row)
    return pd.DataFrame(rows)


# ── figure generation (delegates to the reviewed tools) ───────────────────────
def _run(cmd):
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    return r.returncode == 0


def make_per_seed_figures(auto: Path = AUTO, compare_rows: int = 3) -> list[str]:
    """Per-seed Pareto (diag_seed<N>/) and matched-operating-point grids
    (compare_seed<N>/) for every combo. sweep_diagnostics.py pools seeds when
    handed a whole combo, so it is invoked once per seed instead."""
    py = sys.executable
    made = []
    for combo in sorted({p.parent for p in auto.glob("*/*/*_seed*")}):
        seeds = sorted({d.name.split("_seed")[-1] for d in combo.glob("*_seed*")})
        for sd in seeds:
            runs = sorted(str(d) for d in combo.glob(f"*_seed{sd}"))
            if len(runs) < len(METHODS):
                print(f"  skip (incomplete) {combo.name} seed{sd}: {len(runs)} arms")
                continue
            ok1 = _run([py, "sweep_diagnostics.py", "--runs", *runs,
                        "--diag_pareto", "--diag_winrates",
                        "--out_dir", str(combo / f"diag_seed{sd}")])
            ok2 = _run([py, "compare_methods.py", "--runs", *runs,
                        "--match", "both", "--num_rows", str(compare_rows),
                        "--out_dir", str(combo / f"compare_seed{sd}")])
            made.append(f"{combo.parent.name}/{combo.name} seed{sd} "
                        f"pareto={'ok' if ok1 else 'FAIL'} compare={'ok' if ok2 else 'FAIL'}")
    return made


def collect_paper_figures(auto: Path = AUTO, out: Path = FIGS) -> list[Path]:
    """Copy every generated figure into one flat dir with self-describing names."""
    import shutil
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(auto.glob("*/*/diag_seed*/pareto__*.png")):
        seed = src.parent.name.replace("diag_seed", "")
        dst = out / f"pareto__{src.parts[-4]}__{src.parts[-3][:38]}__seed{seed}.png"
        shutil.copy2(src, dst); copied.append(dst)
    for src in sorted(auto.glob("*/*/compare_seed*/compare_*.png")):
        seed = src.parent.name.replace("compare_seed", "")
        kind = "reward" if "compare_reward" in src.name else "energy"
        dst = out / f"images_{kind}__{src.parts[-4]}__{src.parts[-3][:38]}__seed{seed}.png"
        shutil.copy2(src, dst); copied.append(dst)
    return copied


# ── the draft manifest ────────────────────────────────────────────────────────
# Every figure currently placed in the Overleaf draft, in reading order. The
# Overleaf filename is NOT always the generated one -- the body Pareto was
# uploaded with an `unconstrained_` prefix -- so the mapping is recorded
# explicitly rather than derived. export_draft_figures() is what keeps the two
# from drifting: it copies under the Overleaf name, so re-running the pipeline
# and re-uploading that one directory cannot silently change which figure an
# \includegraphics resolves to.
#
# The 38-char prompt truncation in collect_paper_figures leaves some names with
# a run of underscores ("a_photo_of___seed42"). That is not a typo and has to be
# reproduced exactly.

BODY_COMBO = ("pickscore", "a portrait of a man", "42")

DRAFT_FIGURES = [
    dict(role="body: Pareto",
         generated="pareto__pickscore__a_portrait_of_a_man__a_smiling_face__seed42.png",
         overleaf="unconstrained_pareto__pickscore__a_portrait_of_a_man__a_smiling_face__seed42.png",
         combo=("pickscore", "a portrait of a man", "42")),
    dict(role="body: image rows, seed 42",
         generated="images_reward__pickscore__a_portrait_of_a_man__a_smiling_face__seed42.png",
         overleaf="images_reward__pickscore__a_portrait_of_a_man__a_smiling_face__seed42.png",
         combo=("pickscore", "a portrait of a man", "42")),
    dict(role="body: image rows, seed 7",
         generated="images_reward__pickscore__a_portrait_of_a_man__a_smiling_face__seed7.png",
         overleaf="images_reward__pickscore__a_portrait_of_a_man__a_smiling_face__seed7.png",
         combo=("pickscore", "a portrait of a man", "7")),
    dict(role="appendix: Pareto",
         generated="pareto__imagereward__a_photo_of_a_sleeping_cat__a_photo_of___seed42.png",
         overleaf="pareto__imagereward__a_photo_of_a_sleeping_cat__a_photo_of___seed42.png",
         combo=("imagereward", "a photo of a sleeping cat", "42")),
    dict(role="appendix: Pareto",
         generated="pareto__clip__a_portrait_of_a_man__a_smiling_face__seed7.png",
         overleaf="pareto__clip__a_portrait_of_a_man__a_smiling_face__seed7.png",
         combo=("clip", "a portrait of a man", "7")),
]

DRAFT = HERE / "results" / "draft_figures"


def draft_manifest(figs: Path = FIGS) -> pd.DataFrame:
    """One row per draft figure: does the generated file exist, and how big."""
    rows = []
    for f in DRAFT_FIGURES:
        src = figs / f["generated"]
        rows.append(dict(role=f["role"], overleaf=f["overleaf"],
                         exists=src.is_file(),
                         MB=round(src.stat().st_size / 2**20, 2) if src.is_file() else None))
    return pd.DataFrame(rows)


def export_draft_figures(figs: Path = FIGS, out: Path = DRAFT) -> list[Path]:
    """Copy just the draft figures into one directory under their Overleaf names.

    Raises on a missing source rather than exporting a partial set: a draft
    quietly missing one figure still compiles, and shows the wrong thing."""
    import shutil
    missing = [f["generated"] for f in DRAFT_FIGURES if not (figs / f["generated"]).is_file()]
    if missing:
        raise SystemExit("missing generated figure(s) -- run make_per_seed_figures() "
                         "and collect_paper_figures() first:\n  " + "\n  ".join(missing))
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for f in DRAFT_FIGURES:
        dst = out / f["overleaf"]
        shutil.copy2(figs / f["generated"], dst)
        written.append(dst)
    return written


# ── per-combo numbers that captions quote ─────────────────────────────────────
def combo_stats(df: pd.DataFrame, reward: str, prompt: str, seed) -> dict:
    """Everything a caption for one (reward, prompt, seed) instance asserts.

    Recomputed from summary.csv on every call so a caption number cannot drift
    away from the figure beside it -- same reason win rates are not summed from
    winrates.csv. Frontier counts use the global non-dominated set over all
    methods, matching what sweep_diagnostics.py circles."""
    seed = str(seed)
    w = win_rates(df)
    cell = w[(w.reward == reward) & (w.prompt == prompt) & (w.seed == seed)]
    g = df[(df.reward == reward) & (df.prompt == prompt) & (df.seed == seed)]
    if cell.empty or g.empty:
        raise SystemExit(f"no rows for ({reward}, {prompt!r}, seed {seed})")
    pts = list(zip(g.r_ctrl, g.ctrl_energy))
    nd = _nondominated(pts)
    methods = g.control_type.tolist()
    cfgs = g.config.tolist()
    lams = sorted(g.lam.dropna().unique())
    return dict(
        reward=reward, prompt=prompt, seed=seed,
        edit_prompt=str(g.edit_prompt.iloc[0]),
        N=int(g.N.iloc[0]), H=int(g.H.iloc[0]),
        cells=len(cell),
        wins_hardflow=int(cell.beats_hardflow.sum()),
        wins_fmrg=int(cell.beats_fmrg.sum()),
        frontier_n=len(nd),
        frontier_mpc=sum(1 for i in nd if methods[i] == "mpc"),
        mpc_cfgs=sorted({cfgs[i] for i in nd if methods[i] == "mpc"}),
        n_lam=len(lams), lam_min=lams[0], lam_max=lams[-1],
    )


PRETTY_REWARD = {"clip": "CLIP", "imagereward": "ImageReward", "pickscore": "PickScore"}


def _ni_phrase(cfgs: list[str]) -> str:
    """', at both $n_inner = 12$ and 24' / ', at $n_inner = 24$' / '' for captions."""
    nis = sorted({c.split("ni=")[-1] for c in cfgs if "ni=" in c}, key=int)
    if not nis:
        return ""
    if len(nis) == 1:
        return rf", at $n_{{\text{{inner}}}} = {nis[0]}$"
    # every value stays inside math mode -- "$12$ and 24" renders the second
    # number in text digits next to a math one, which is visibly inconsistent
    both = "both " if len(nis) == 2 else ""
    return (rf", at {both}$n_{{\text{{inner}}}} = {nis[0]}$ and "
            + " and ".join(rf"${n}$" for n in nis[1:]))


# ── LaTeX blocks for the draft ────────────────────────────────────────────────
# These emit the figure environments actually pasted into the draft, with every
# quoted number interpolated from combo_stats rather than typed. Regenerate and
# re-paste after any sweep that adds cells. Do NOT hand-edit the captions in the
# .tex: the two copies would disagree and only one of them is checkable.

def latex_body_pareto(df: pd.DataFrame, prefix: str = "figures/",
                      label: str = "fig:pareto-edit-pickscore") -> str:
    s = combo_stats(df, *BODY_COMBO)
    name = DRAFT_FIGURES[0]["overleaf"]
    return "\n".join([
        r"\begin{figure}[t]",
        r"  \centering",
        rf"  \includegraphics[width=\linewidth]{{{prefix}{name}}}",
        r"  \caption{Reward--effort trade-off on the editing task, unconstrained.",
        rf"  {PRETTY_REWARD[s['reward']]} reward, base prompt ``{s['prompt']}'' edited toward",
        rf"  ``{s['edit_prompt']}'', seed~{s['seed']}, $N={s['N']}$, $H={s['H']}$. Each marker is",
        r"  one value of $\lambda$; the horizontal axis is the control energy",
        r"  $\sum_k \lVert u_k \rVert^2 \Delta t$ and the vertical axis the achieved",
        r"  reward $r(x_1)$, so up and to the left is better. Circled markers are",
        rf"  globally non-dominated: our method owns {s['frontier_mpc']} of the"
        rf" {s['frontier_n']} frontier",
        r"  points. Seeds are never pooled --- a seed fixes $x_0$ and therefore a",
        r"  distinct optimization problem --- so the remaining reward/prompt/seed",
        r"  combinations appear separately in Appendix~\ref{app:pareto}.}",
        rf"  \label{{{label}}}",
        r"\end{figure}",
    ])


def latex_body_image_rows(df: pd.DataFrame, prefix: str = "figures/",
                          label: str = "fig:edit-rows") -> str:
    a = combo_stats(df, "pickscore", "a portrait of a man", "42")
    n42, n7 = DRAFT_FIGURES[1]["overleaf"], DRAFT_FIGURES[2]["overleaf"]
    return "\n".join([
        r"% requires \usepackage{subcaption}",
        r"\begin{figure*}[t]",
        r"  \centering",
        r"  \begin{subfigure}[t]{0.49\textwidth}",
        r"    \centering",
        rf"    \includegraphics[width=\linewidth]{{{prefix}{n42}}}",
        r"    \caption{Seed 42.}",
        r"    \label{fig:edit-rows-seed42}",
        r"  \end{subfigure}\hfill",
        r"  \begin{subfigure}[t]{0.49\textwidth}",
        r"    \centering",
        rf"    \includegraphics[width=\linewidth]{{{prefix}{n7}}}",
        r"    \caption{Seed 7.}",
        r"    \label{fig:edit-rows-seed7}",
        r"  \end{subfigure}",
        r"  \caption{Editing task at matched operating points,",
        rf"  {PRETTY_REWARD[a['reward']]} reward, base prompt ``{a['prompt']}'' edited toward",
        rf"  ``{a['edit_prompt']}'', $N={a['N']}$, $H={a['H']}$. Columns are the uncontrolled",
        r"  sample and the three methods; each row is a triple of runs selected to",
        r"  have nearly equal \emph{achieved reward}, so the comparison within a row",
        r"  is what each method paid in control energy to reach that reward. Energy",
        r"  and reward are printed under every panel and the matched reward window",
        r"  under each row; rows increase in reward top to bottom. The two seeds are",
        r"  separate optimization problems rather than repeats --- a seed fixes $x_0$,",
        r"  hence a different uncontrolled image --- so (b) is a replication of (a),",
        r"  not an average over it.}",
        rf"  \label{{{label}}}",
        r"\end{figure*}",
    ])


def latex_appendix_pareto(df: pd.DataFrame, prefix: str = "figures/",
                          label: str = "app:pareto") -> str:
    body = combo_stats(df, *BODY_COMBO)
    panels = [combo_stats(df, *f["combo"]) for f in DRAFT_FIGURES[3:]]
    names = [f["overleaf"] for f in DRAFT_FIGURES[3:]]
    sublabels = ["fig:pareto-app-ir-cat", "fig:pareto-app-clip-portrait"]

    out = [
        r"% requires \usepackage{subcaption}",
        r"\section{Additional Pareto frontiers}",
        rf"\label{{{label}}}",
        r"",
        r"Figure~\ref{fig:pareto-edit-pickscore} in the main text reports one",
        r"$(\text{reward}, \text{prompt}, \text{seed})$ instance. Because seeds are not",
        r"pooled --- a seed fixes $x_0$ and therefore a distinct optimization problem ---",
        r"that figure is a single problem instance rather than an average, so we report",
        r"two further instances here that vary all three factors at once: a different",
        r"reward, a different prompt pair, and a different seed. The $\lambda$ grid is",
        r"calibrated per reward rather than shared, since the control-effort term",
        r"$\tfrac{1}{2\lambda}\sum_k \lVert u_k \rVert^2 \Delta t$ is only commensurate",
        r"with a reward at that reward's own scale; the grids therefore differ between",
        r"the two panels below.",
        r"",
        r"\begin{figure*}[t]",
        r"  \centering",
    ]
    for i, (s, name, sub) in enumerate(zip(panels, names, sublabels)):
        out += [
            r"  \begin{subfigure}[t]{0.49\textwidth}",
            r"    \centering",
            rf"    \includegraphics[width=\linewidth]{{{prefix}{name}}}",
            rf"    \caption{{{PRETTY_REWARD[s['reward']]}, ``{s['prompt']}'' $\rightarrow$",
            rf"    ``{s['edit_prompt']}'', seed~{s['seed']}. Our method attains a lower",
            rf"    objective than both baselines on all ${s['wins_hardflow']}$ of"
            rf" ${s['cells']}$ cells and owns",
            rf"    ${s['frontier_mpc']}$ of the ${s['frontier_n']}$ non-dominated points"
            + _ni_phrase(s["mpc_cfgs"]) + r".}",
            rf"    \label{{{sub}}}",
            r"  \end{subfigure}" + (r"\hfill" if i == 0 else ""),
        ]
    out += [
        r"  \caption{Reward--effort trade-off on two further editing instances,",
        rf"  $N={panels[0]['N']}$, $H={panels[0]['H']}$. Axes and markers follow",
        r"  Figure~\ref{fig:pareto-edit-pickscore}: the horizontal axis is control",
        r"  energy $\sum_k \lVert u_k \rVert^2 \Delta t$ and the vertical axis the",
        r"  achieved reward $r(x_1)$, so up and to the left is better, and circled",
        r"  markers are globally non-dominated. Each marker is one value of $\lambda$,",
        r"  on a grid calibrated separately for each reward. Together with the",
        rf"  main-text instance --- {PRETTY_REWARD[body['reward']]}, ``{body['prompt']}'',",
        rf"  seed~{body['seed']}, where our method wins ${body['wins_hardflow']}$ of",
        rf"  ${body['cells']}$ cells against HardFlow and ${body['wins_fmrg']}$ of",
        rf"  ${body['cells']}$ against FMRG and owns ${body['frontier_mpc']}$ of",
        rf"  ${body['frontier_n']}$ frontier points --- these cover three distinct",
        r"  rewards, two prompt pairs and two seeds.}",
        r"  \label{fig:pareto-appendix}",
        r"\end{figure*}",
    ]
    return "\n".join(out)


def latex_all(df: pd.DataFrame) -> str:
    """Every block in the draft, in reading order, separated by comments."""
    return "\n\n".join([
        "% ===== body: Pareto =====",
        latex_body_pareto(df),
        "% ===== body: image rows =====",
        latex_body_image_rows(df),
        "% ===== appendix =====",
        latex_appendix_pareto(df),
    ])


if __name__ == "__main__":
    df = load_rows()
    print(f"loaded {len(df)} successful rows, "
          f"{df.groupby(['reward','prompt','seed']).ngroups} combos\n")
    print("== win rates ==")
    print(win_rate_summary(df).to_string(index=False))
    print("\n== per combo ==")
    print(win_rate_by_combo(df).to_string(index=False))
    print("\n== MPC share of per-seed Pareto frontier ==")
    print(frontier_share(df).to_string(index=False))
    print("\n== figures in the draft ==")
    print(draft_manifest().to_string(index=False))
