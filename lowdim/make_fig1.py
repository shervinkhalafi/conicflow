"""Regenerate paper Figure 1 (two-moons, unconstrained): reward vs. control energy.

    python make_fig1.py                         # from the sweep CSVs shipped in paper_data/
    python make_fig1.py --data results_fig1     # from a fresh run of reproduce_fig1.sh

Writes, into --out (default figures/), the three panels under their paper names:

    rewards_lowdim.png      the two reward functions over the two-moons data (right)
    lowdim_lowlambda.png    upper-moon reward, reward vs. energy (left)
    lowdim_highlambda.png   off-manifold reward, reward vs. energy (middle)

and prints the uncontrolled baselines and a cost-at-reward table.

--data must contain the sub-directories reproduce_fig1.sh writes (and paper_data/
ships), each holding one run_sweep.py `sweep_results.csv`:

    upper_moon/{base,fmrg,conicflow_short_h,lam5_10_long_h}/
    off_manifold/{base,fmrg,fmrg_hi_lam,extended,saturation}/

Each plotted point is the best configuration at that lambda over the inner
hyperparameter grid, selected by the control objective J = E/(2*lam) - mean r.

Method naming: ConicFlow is called `mpc` in run_sweep.py and in the CSV `method`
column. "RH-H" is ConicFlow with the reward evaluated at the end of the planning
window (`constraint_on=horizon`); "RH-T" evaluates it at the flow-map prediction
of the terminal state (`constraint_on=endpoint`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from scipy.spatial import cKDTree

from sampling_utils import sample_two_moons

HERE = Path(__file__).resolve().parent

UPPER_MOON_PARTS = ["upper_moon/base", "upper_moon/fmrg",
                    "upper_moon/conicflow_short_h", "upper_moon/lam5_10_long_h"]
OFFTOP_PARTS = ["off_manifold/base", "off_manifold/fmrg", "off_manifold/fmrg_hi_lam",
                "off_manifold/extended", "off_manifold/saturation"]

# The paper's PNGs are the notebook's inline renders: 100 dpi, tight bounding box.
DPI = 100


# ── loading ──────────────────────────────────────────────────────────────────

def load_sweep(root: Path, parts, restrict_grid: bool = False) -> pd.DataFrame:
    """Concatenate the sweep CSVs of one reward problem.

    The CSV schema grew between runs, so parts are concatenated here (columns
    aligned by name, missing ones NaN) rather than appended on disk.

    restrict_grid=True keeps one inner-solver grid so that points run at
    different times are selected from the same set of configs (ConicFlow
    mpc_n_iters == 50; HardFlow hf_n_outer == hf_n_inner == 10; FMRG all).
    The upper-moon panel uses it: its lam <= 3 and lam in {5, 10} points were
    swept with different grids.
    """
    files = [root / p / "sweep_results.csv" for p in parts]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError("missing sweep CSVs:\n  " + "\n  ".join(missing))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # ENERGY SCALE -- applied here, on load, and NOT in run_sweep.py.
    # run_sweep.py writes ctrl_cost_raw = sum_j ||dx_j||^2 with dx_j = dt*u_j a
    # displacement, which is the controller's cost sum_j ||u_j||^2 dt short by
    # a factor 1/dt = N. Multiplying by N gives the energy E on the x-axis, and
    # J is rebuilt from it because best_per_lam selects each point by J. If
    # run_sweep.py is ever changed to write E directly, delete this block.
    n = df["N"].astype(float)
    if n.isna().any() or (n <= 0).any():
        raise ValueError("load_sweep: non-positive or missing N; refusing to rescale")
    df["ctrl_cost_raw"] = df["ctrl_cost_raw"] * n            # -> sum_j ||u_j||^2 dt
    df["ctrl_cost_obj"] = df["ctrl_cost_raw"] / (2 * df["lam"])
    df["objective_J"] = df["ctrl_cost_obj"] - df["avg_r_ctrl"]

    if restrict_grid:
        keep = (((df.method == "mpc") & (df.mpc_n_iters == 50)) |
                ((df.method == "hardflow") & (df.hf_n_outer == 10) & (df.hf_n_inner == 10)) |
                (df.method == "fmrg"))
        df = df[keep]
    return df.reset_index(drop=True)


def conicflow(df, H, constraint_on):
    return df[(df.method == "mpc") & (df.H == H) & (df.constraint_on == constraint_on)]


def hardflow(df):
    return df[df.method == "hardflow"]


def fmrg(df, variant):
    return df[(df.method == "fmrg") & (df.fmrg_variant == variant)]


def best_per_lam(sub, by="objective_J"):
    """One row per lambda: the config minimising `by`."""
    sub = sub.dropna(subset=[by])
    if not len(sub):
        return sub
    return sub.loc[sub.groupby("lam")[by].idxmin()].sort_values("ctrl_cost_raw")


# ── style (values as used for the paper) ─────────────────────────────────────

FIGSIZE = (10.5, 8.5)
FONTS = dict(label=20.0, tick=12.0, legend=20.0, annot=6.0, clabel=20.0)
FIG1_SCALE = 1.5          # axis labels / ticks / legends on the three panels
FIG1_LEGEND_FS = 26.0     # the shared legend, on the upper-moon panel
FIG1_ANNOT_FS = 12.0      # per-point lambda labels (upper-moon panel)
OFFTOP_ANNOT_MULT = 1.0   # off-manifold panel: same lambda-label size (12 pt) as the paper

STYLE = dict(
    marker_size=75, marker_edge="black", marker_edge_lw=0.5, marker_alpha=0.95,
    line_lw=1.2, line_alpha=0.5, annot_offset=(5, 4), legend_loc="lower right",
    baseline_color="gray", baseline_lw=1.0, baseline_ls="--",
    xlabel=r"Total control energy  $\sum_j \|u_j\|^2 \Delta t$",
    ylabel=r"Terminal reward  $\bar r(x_N)$",
)

TEAL = ["#0d9488", "#115e59", "#2dd4bf"]
COLORS = dict(hardflow="#c2410c", fmrg="#6b21a8", fmrg_alt="#a855f7")


def fs1(key):
    return FONTS[key] * FIG1_SCALE


def series(df):
    return {
        "ConicFlow RH-H (H=4)": (conicflow(df, 4, "horizon"), dict(color=TEAL[0], marker="o")),
        "ConicFlow RH-T (H=4)": (conicflow(df, 4, "endpoint"), dict(color=TEAL[1], marker="o")),
        "ConicFlow RH-H (H=8)": (conicflow(df, 8, "horizon"), dict(color=TEAL[2], marker="P")),
        "HardFlow": (hardflow(df), dict(color=COLORS["hardflow"], marker="s")),
        "FMRG (J)": (fmrg(df, "J"), dict(color=COLORS["fmrg"], marker="^")),
        "FMRG (E)": (fmrg(df, "E"), dict(color=COLORS["fmrg_alt"], marker="^")),
    }


def plot_pareto(series_, baseline, xlim, ylim=None, annot_fs=6.0, legend=True,
                legend_fs=20.0, lam_filter=None):
    st = STYLE
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, (sub, ov) in series_.items():
        b = best_per_lam(sub)
        if lam_filter is not None and len(b):
            b = b[b.lam.map(lam_filter)]
        if not len(b):
            continue
        c, mk, s = ov["color"], ov["marker"], ov.get("s", st["marker_size"])
        ax.plot(b.ctrl_cost_raw, b.avg_r_ctrl, color=c, lw=st["line_lw"],
                alpha=st["line_alpha"], zorder=2)
        ax.scatter(b.ctrl_cost_raw, b.avg_r_ctrl, color=c, marker=mk, s=s, label=label,
                   edgecolor=st["marker_edge"], linewidth=st["marker_edge_lw"],
                   alpha=st["marker_alpha"], zorder=3)
        for _, r in b.iterrows():
            ax.annotate(f"{r.lam:g}", (r.ctrl_cost_raw, r.avg_r_ctrl), textcoords="offset points",
                        xytext=st["annot_offset"], fontsize=annot_fs, color=c, alpha=0.9)
    ax.axhline(baseline, color=st["baseline_color"], lw=st["baseline_lw"],
               ls=st["baseline_ls"], label="Uncontrolled")
    ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(st["xlabel"], fontsize=fs1("label"))
    ax.set_ylabel(st["ylabel"], fontsize=fs1("label"))
    ax.tick_params(labelsize=fs1("tick"))
    if legend:
        ax.legend(fontsize=legend_fs, loc=st["legend_loc"])
    fig.tight_layout()
    return fig


# ── rewards panel ────────────────────────────────────────────────────────────
# These mirror run_sweep._build_upper_moon_reward_fn / _build_off_top_ball_reward_fn
# for drawing; they are re-implemented so this script does not import torch.

REWARD_SIGMA, N_REF = 0.35, 400
OFFTOP_CENTER, OFFTOP_SIGMA = np.array([-0.1, 1.75]), 0.5


def upper_moon_reward(P, seed=42):
    rng = np.random.default_rng(seed)
    pts, labels = sample_two_moons(50_000, return_labels=True)
    upper = pts[labels == 0]
    ref = upper[rng.choice(len(upper), N_REF, replace=False)]
    d, _ = cKDTree(ref).query(P)
    return np.exp(-d ** 2 / (2 * REWARD_SIGMA ** 2))


def offtop_ball_reward(P):
    d2 = ((P - OFFTOP_CENTER) ** 2).sum(-1)
    return np.exp(-d2 / (2 * OFFTOP_SIGMA ** 2))


def plot_rewards(seed: int):
    # sample_two_moons draws from numpy's global RNG; seeding it here makes this
    # panel deterministic (the paper's render was not seeded, so its scatter
    # points and upper-moon reference points differ from this one in detail).
    np.random.seed(seed)
    colors = {"upper-moon": (34 / 255, 148 / 255, 127 / 255, 0.5),
              "off-manifold": (214 / 255, 109 / 255, 24 / 255, 1.0)}
    E, G = 2.7, 400
    xs = np.linspace(-E, E, G)
    XX, YY = np.meshgrid(xs, xs, indexing="xy")
    P = np.stack([XX.ravel(), YY.ravel()], axis=1)
    R = {"upper-moon": upper_moon_reward(P).reshape(G, G),
         "off-manifold": offtop_ball_reward(P).reshape(G, G)}
    pts, _ = sample_two_moons(4000, return_labels=True)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name, Z in R.items():
        color = colors[name]
        rgb, a = color[:3], color[3]
        rgba = to_rgba(color)
        cmap = LinearSegmentedColormap.from_list("", [(rgba[0], rgba[1], rgba[2], 0.0), rgba])
        ax.contourf(XX, YY, Z, levels=np.linspace(0, 1.0, 10), cmap=cmap, vmin=0, vmax=1.0, zorder=2)
        cs = ax.contour(XX, YY, Z, levels=[0.3, 0.6, 0.9], colors=[rgb], linewidths=1.3,
                        alpha=a * 0.9, zorder=3)
        ax.clabel(cs, inline=True, fontsize=FONTS["clabel"], fmt="%.1f")
        ax.plot([], [], color=rgb, lw=6, alpha=a, label=f"$r$: {name}")
    ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.30, c="0.35", zorder=6)
    ax.set_xlim(-2.4, 1.7)
    ax.set_ylim(-2.2, 2.7)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$", fontsize=fs1("label"))
    ax.set_ylabel("$x_2$", fontsize=fs1("label"))
    ax.tick_params(labelsize=fs1("tick"))
    ax.legend(fontsize=fs1("legend"), loc="lower right", framealpha=0.95)
    fig.tight_layout()
    return fig


def cost_at_reward(series_, targets):
    rows = []
    for label, (sub, _) in series_.items():
        if not len(sub):
            continue
        row = {"series": label}
        for t in targets:
            ok = sub[sub.avg_r_ctrl >= t]
            row[f"r>={t}"] = round(ok.ctrl_cost_raw.min(), 3) if len(ok) else np.nan
        row["max r"] = round(sub.avg_r_ctrl.max(), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(HERE / "paper_data"),
                    help="Root holding upper_moon/ and off_manifold/ sweep dirs "
                         "(default: the shipped paper_data/).")
    ap.add_argument("--out", default=str(HERE / "figures"))
    ap.add_argument("--seed", type=int, default=0, help="Seed for the rewards panel's samples.")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fig = plot_rewards(a.seed)
    fig.savefig(out / "rewards_lowdim.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    df_um = load_sweep(data, UPPER_MOON_PARTS, restrict_grid=True)
    base_um = float(df_um.avg_r_unc.dropna().iloc[0])
    s_um = series(df_um)
    fig = plot_pareto(s_um, base_um, xlim=(-0.16, 1.6), ylim=(0.55, 1.05),
                      annot_fs=FIG1_ANNOT_FS, legend_fs=FIG1_LEGEND_FS)
    fig.savefig(out / "lowdim_lowlambda.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    df_ot = load_sweep(data, OFFTOP_PARTS)
    base_ot = float(df_ot.avg_r_unc.dropna().iloc[0])
    s_ot = series(df_ot)
    fig = plot_pareto(s_ot, base_ot, xlim=(-0.8, 28.8), annot_fs=FIG1_ANNOT_FS * OFFTOP_ANNOT_MULT,
                      legend=False, lam_filter=lambda l: l != 70)   # lambda=70 is not plotted
    fig.savefig(out / "lowdim_highlambda.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"upper moon:   {len(df_um)} rows, uncontrolled r = {base_um:.4f}")
    print(f"off-manifold: {len(df_ot)} rows, uncontrolled r = {base_ot:.4f}")
    print("\nenergy needed to reach a given mean reward (min over all plotted configs)")
    print("UPPER MOON\n", cost_at_reward(s_um, (0.7, 0.85, 0.95, 0.99)).to_string(index=False))
    print("\nOFF-MANIFOLD\n", cost_at_reward(s_ot, (0.7, 0.85, 0.95, 0.98)).to_string(index=False))
    print(f"\nwrote {out}/rewards_lowdim.png, lowdim_lowlambda.png, lowdim_highlambda.png")


if __name__ == "__main__":
    main()
