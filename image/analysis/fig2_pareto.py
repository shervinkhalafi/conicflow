"""Reward vs. control energy frontiers for the unconstrained image experiment.

Draws, from run_table1.sh's results tree:

  uncon_main_pareto.png      Figure 2 (left): PickScore, "a portrait of a man"
                             -> "a smiling face", seed 42
  pareto__imagereward__a_photo_of_a_sleeping_cat__a_photo_of___seed42.png
  pareto__clip__a_portrait_of_a_man__a_smiling_face__seed7.png
                             the two appendix frontiers

One point per lambda per method, each annotated with its lambda; the grey
dashed line is the uncontrolled sample's reward. ConicFlow is shown at
n_inner=24 (H=2), HardFlow at n_inner=40, FMRG-J at 3 guidance steps.

Energy is the paper's E = sum_k dt*||u_k||^2. summary.csv stores it divided by
N^2, so every row is rescaled by its own N^2 on load (64 at N=8).

  python fig2_pareto.py                                 # reads ../src/results/unconstrained
  python fig2_pareto.py --results <root> --out_dir <dir>
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
IMAGE = HERE.parent
CONICFLOW_TYPES = ("conicflow", "mpc")   # older runs label the method "mpc"

# ---- sizes (the paper's figure settings) -------------------------------------
FIG = dict(scale=1.00, dpi=150, pareto=(10.5, 8.5))
FONTS = dict(scale=1.0, title=15.0, label=20.0, tick=12.0, legend=20.0, annot=6.0)
FIG2_SCALE = 1.5      # axis labels / tick numbers / legend on these panels
FIG2_ANNOT_FS = 12.0  # the lambda printed next to each point


def fs(key):
    return FONTS[key] * FONTS["scale"]


def fs2(key):
    return fs(key) * FIG2_SCALE


def figsize(key="pareto"):
    w, h = FIG[key]
    return (w * FIG["scale"], h * FIG["scale"])


STYLE = dict(
    figsize=figsize("pareto"), dpi=FIG["dpi"],
    title_fontsize=fs("title"), label_fontsize=fs("label"), tick_fontsize=fs("tick"),
    legend_fontsize=fs("legend"), annot_fontsize=fs("annot"),
    marker_size=75, marker_edge="black", marker_edge_lw=0.5, marker_alpha=0.95,
    line_lw=1.2, line_alpha=0.5, draw_lines=True, annotate_lam=True, annot_offset=(5, 4),
    legend_loc="lower right", legend=True, grid=False, grid_alpha=0.25,
    baseline_color="gray", baseline_lw=1.0, baseline_ls="--",
    xlabel=r"Total control energy  $\sum_j \|u_j\|^2 \Delta t$",
    ylabel=r"Terminal reward  $\bar r(x_N)$",
)
IMG_STYLE = dict(
    annot_fontsize=FIG2_ANNOT_FS, annot_offset=(6, 5),
    label_fontsize=fs2("label"), tick_fontsize=fs2("tick"), legend_fontsize=fs2("legend") * 0.8,
    legend_loc="upper left", grid=True, grid_alpha=0.25,
    xlabel=r"Total control energy  $\sum_j \|u_j\|^2 \Delta t$",
    ylabel=r"Terminal reward  r$(x_N)$",
)
METHOD_COLORS = dict(conicflow="#0d9488", hardflow="#c2410c", fmrg="#6b21a8")

# (output file, reward, prompt, edit_prompt, seed, ylim, save dpi). The main
# panel was exported at 100 dpi, the two appendix panels at 150.
PANELS = [
    ("uncon_main_pareto.png", "pickscore", "a portrait of a man", "a smiling face", 42, None, 100),
    ("pareto__imagereward__a_photo_of_a_sleeping_cat__a_photo_of___seed42.png", "imagereward",
     "a photo of a sleeping cat", "a photo of a cat with wide open eyes", 42, (-1.5, 3.5), 150),
    ("pareto__clip__a_portrait_of_a_man__a_smiling_face__seed7.png", "clip",
     "a portrait of a man", "a smiling face", 7, (0.20, 0.66), 150),
]


def slug_of(prompt, edit):
    s = f"{prompt}__{edit}"
    return "".join(c if c.isascii() and c.isalnum() else "_" for c in s)[:60]


def load_problem(root, reward, prompt, edit, seed):
    """Every run_ok row of one problem, renamed to the plotting schema."""
    d = Path(root) / reward / slug_of(prompt, edit)
    files = sorted(glob.glob(str(d / f"*_seed{seed}" / "summary.csv")))
    if not files:
        raise FileNotFoundError(f"no runs for {reward}/{prompt} -> {edit} seed {seed} under {d}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[(df.run_ok.astype(str).str.lower() == "true") & (df.seed == seed)
            & (df.prompt == prompt) & (df.edit_prompt == edit)].copy()
    df["E"] = df.ctrl_energy * df.N.astype(float) ** 2
    df["objective_J"] = df.E / (2 * df.lam) - df.r_ctrl
    return df.reset_index(drop=True)


def best_per_lam(sub, by="objective_J"):
    """One row per lambda: the config minimising `by` (energy/(2 lam) - reward)."""
    sub = sub.dropna(subset=[by])
    if not len(sub):
        return sub
    return sub.loc[sub.groupby("lam")[by].idxmin()].sort_values("E")


def plot_pareto(series, baseline=None, xscale="linear", yscale="linear", xlim=None, ylim=None, style=None):
    st = {**STYLE, **(style or {})}
    fig, ax = plt.subplots(figsize=st["figsize"])
    for label, (sub, override) in series.items():
        b = best_per_lam(sub)
        if not len(b):
            continue
        color, marker = override.get("color", "C0"), override.get("marker", "o")
        size = override.get("s", st["marker_size"])
        if st["draw_lines"]:
            ax.plot(b.E, b.r_ctrl, color=color, lw=st["line_lw"], alpha=st["line_alpha"], zorder=2)
        ax.scatter(b.E, b.r_ctrl, color=color, marker=marker, s=size, label=label,
                   edgecolor=st["marker_edge"], linewidth=st["marker_edge_lw"],
                   alpha=st["marker_alpha"], zorder=3)
        if st["annotate_lam"]:
            for _, r in b.iterrows():
                ax.annotate(f"{r.lam:g}", (r.E, r.r_ctrl), textcoords="offset points",
                            xytext=st["annot_offset"], fontsize=st["annot_fontsize"], color=color, alpha=0.9)
    if baseline is not None:
        ax.axhline(baseline, color=st["baseline_color"], lw=st["baseline_lw"], ls=st["baseline_ls"],
                   label="Uncontrolled")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(st["xlabel"] + ("  (log scale)" if xscale == "log" else ""), fontsize=st["label_fontsize"])
    ax.set_ylabel(st["ylabel"] + ("  (log scale)" if yscale == "log" else ""), fontsize=st["label_fontsize"])
    ax.tick_params(labelsize=st["tick_fontsize"])
    if st["grid"]:
        ax.grid(alpha=st["grid_alpha"], zorder=0)
    if st.get("legend", True):
        ax.legend(fontsize=st["legend_fontsize"], loc=st["legend_loc"])
    fig.tight_layout()
    return fig, ax


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(IMAGE / "src" / "results" / "unconstrained"))
    ap.add_argument("--out_dir", default=None, help="default: <results>/figures")
    a = ap.parse_args()
    out = Path(a.out_dir or Path(a.results) / "figures")
    out.mkdir(parents=True, exist_ok=True)

    for fname, reward, prompt, edit, seed, ylim, dpi in PANELS:
        df = load_problem(a.results, reward, prompt, edit, seed)
        baseline = float(df.r_unc.dropna().iloc[0])
        series = {
            "ConicFlow (RH-T)": (df[df.control_type.isin(CONICFLOW_TYPES) & (df.H == 2) & (df.n_inner == 24)],
                                 dict(color=METHOD_COLORS["conicflow"], marker="o")),
            "HardFlow": (df[df.control_type == "hardflow"], dict(color=METHOD_COLORS["hardflow"], marker="s")),
            "FMRG": (df[df.control_type == "fmrg"], dict(color=METHOD_COLORS["fmrg"], marker="^")),
        }
        fig, _ = plot_pareto(series, baseline=baseline, xscale="log", ylim=ylim, style=IMG_STYLE)
        fig.savefig(out / fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"{reward:12s} {prompt} -> {edit}, seed {seed}: {len(df)} runs, r_unc={baseline:.4f} -> {out / fname}")


if __name__ == "__main__":
    main()
