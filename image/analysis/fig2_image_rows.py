"""Matched-reward image rows: Figure 2 (right) and appendix rows u01-u10.

Each row is Uncontrolled | ConicFlow | HardFlow | FMRG for one (reward, prompt,
seed), with each method pinned at the lambda whose ACHIEVED reward matches the
others' -- so the row is read across on control energy. The lambdas are pinned
in ROWS / UNC_ROWS below rather than searched for, so adding runs to a results
tree can never silently change a figure.

Runs come from two results trees:

  --unconstrained_root   run_table1.sh's tree (endpoint_surrogate=rollout runs):
                         Figure 2 row A and appendix rows u05-u10
  --editing_root         the PickScore editing runs (endpoint_surrogate=jump,
                         n_inner=24) on ed1-ed4 at seeds 317/1289 -- the core
                         arms of the ablation experiment: Figure 2 rows B/C
                         and appendix rows u01-u04

Every summary.csv under a root is read and runs are selected by CSV columns
(reward, prompt, edit_prompt, seed, control_type, lam, n_inner, H,
endpoint_surrogate, constraint_on), never by directory name, so any layout
works. A pin that matches zero or several runs raises.

Writes to --out_dir:
  uncon_main_images.png                 Figure 2 (right), three stacked rows
  unconstrained_u01_..._u10_*.png       the appendix rows, one PNG per row
  rows_table.csv                        lambda / reward / energy behind every panel

Energy is the paper's E = sum_k dt*||u_k||^2 = N^2 * summary.csv's ctrl_energy.
"""
import argparse
import glob
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
IMAGE = HERE.parent
CONICFLOW_TYPES = ("conicflow", "mpc")   # older runs label the method "mpc"
PANELS = ("unc", "conicflow", "hardflow", "fmrg")
TITLES = {"unc": "Uncontrolled", "conicflow": "ConicFlow", "hardflow": "HardFlow", "fmrg": "FMRG"}

MAN, SMILE = "a portrait of a man", "a smiling face"
CAT, EYES = "a photo of a sleeping cat", "a photo of a cat with wide open eyes"
GRASS, POPPIES = "a field of green grass", "a field of red poppies"
YOUNG, OLD = "a portrait of a young man", "a portrait of an old man"
STRAIGHT, CURLY = "a portrait of a woman with straight hair", "a portrait of a woman with curly hair"
SMILING, ANGRY = "a portrait of a smiling man", "a portrait of an angry man"

# Figure 2 (right). lams = (ConicFlow, HardFlow, FMRG); ConicFlow at n_inner=24.
ROWS = [
    dict(name="rowA", src="unconstrained", reward="pickscore", prompt=MAN, edit=SMILE, seed=7,
         lams=(15, 23, 23), right_label="smile"),
    dict(name="rowB", src="editing", reward="pickscore", prompt=YOUNG, edit=OLD, seed=317,
         lams=(14, 25, 55), right_label="old"),
    dict(name="rowC", src="editing", reward="pickscore", prompt=STRAIGHT, edit=CURLY, seed=1289,
         lams=(12, 39, 39), right_label="curly hair"),
]
# Appendix "additional unconstrained instances": one row per (reward, prompt,
# seed), at the highest reward level all three methods reach within 20% of the
# instance's headroom.
UNC_ROWS = [
    dict(file="unconstrained_u01_pickscore_young_old_seed1289.png", src="editing", reward="pickscore",
         prompt=YOUNG, edit=OLD, seed=1289, lams=(14, 25, 25)),
    dict(file="unconstrained_u02_pickscore_straight_curly_seed317.png", src="editing", reward="pickscore",
         prompt=STRAIGHT, edit=CURLY, seed=317, lams=(5.9, 5.9, 12)),
    dict(file="unconstrained_u03_pickscore_smiling_angry_seed317.png", src="editing", reward="pickscore",
         prompt=SMILING, edit=ANGRY, seed=317, lams=(7, 22, 22)),
    dict(file="unconstrained_u04_pickscore_smiling_angry_seed1289.png", src="editing", reward="pickscore",
         prompt=SMILING, edit=ANGRY, seed=1289, lams=(7, 22, 22)),
    dict(file="unconstrained_u05_clip_man_smile_seed42.png", src="unconstrained", reward="clip",
         prompt=MAN, edit=SMILE, seed=42, lams=(320, 320, 1400)),
    dict(file="unconstrained_u06_imagereward_man_smile_seed7.png", src="unconstrained", reward="imagereward",
         prompt=MAN, edit=SMILE, seed=7, lams=(0.75, 2.7, 1.6)),
    dict(file="unconstrained_u07_clip_grass_poppies_seed42.png", src="unconstrained", reward="clip",
         prompt=GRASS, edit=POPPIES, seed=42, lams=(86, 86, 250)),
    dict(file="unconstrained_u08_imagereward_grass_poppies_seed42.png", src="unconstrained", reward="imagereward",
         prompt=GRASS, edit=POPPIES, seed=42, lams=(63, 63, 120000)),
    dict(file="unconstrained_u09_clip_cat_eyes_seed42.png", src="unconstrained", reward="clip",
         prompt=CAT, edit=EYES, seed=42, lams=(120, 380, 380)),
    dict(file="unconstrained_u10_imagereward_cat_eyes_seed42.png", src="unconstrained", reward="imagereward",
         prompt=CAT, edit=EYES, seed=42, lams=(19, 38, 38)),
]
SURROGATE = {"unconstrained": "rollout", "editing": "jump"}

# ---- style -------------------------------------------------------------------
STYLE = dict(panel_in=3.3, panel_h=3.3, dpi=200, wspace=0.04, interpolation="lanczos",
             title_fs=13, title_pad=6, title_weight="normal", title_color="black",
             caption_fs=12, caption_color="0.15", caption_pad=4,
             frame=True, frame_lw=0.8, frame_color="0.75")
STACK_DPI = 100                 # Figure 2 (right) was exported at 100 dpi
STACK_HSPACE = 0.25
STACK_UNCONTROLLED_GAP = 0.025
STACK_LABEL_FS = 12
STACK_LABEL_FMT = r"$Matched\ Reward\ \mathbf{{r\approx {r:.2f}}}$"
STACK_E_MIN_COLOR = "#1baf7a"   # lowest energy among the three methods
STACK_E_OTHER_COLOR = "#d62728"
STACK_TITLE_COLORS = dict(unc="0.35", conicflow="#0d9488", hardflow="#c2410c", fmrg="#6b21a8")
STACK_RIGHT_FMT = "edit prompt = \"{prompt}\""
STACK_ACCENT = "#A51C30"
STACK_BOX_LW = 1.3
STACK_BOX_PAD = 0.005
STACK_BOX_PAD_B = 0.035
UNC = dict(row_w=6.2, cell_w=1.0, inner_gap=0.30, side_in=0.025, top_in=0.30, bottom_in=0.52,
           wspace=0.05, title_fs=11, caption_fs=10, label_fs=8, box_pad=0.004, box_pad_b_in=0.42)


# ---- selection ---------------------------------------------------------------
_TREES = {}


def load_tree(root):
    """Every summary.csv under `root`, with each row's directory kept for its images."""
    root = str(root)
    if root not in _TREES:
        files = sorted(glob.glob(f"{root}/**/summary.csv", recursive=True))
        if not files:
            raise FileNotFoundError(f"no summary.csv under {root}")
        frames = []
        for f in files:
            d = pd.read_csv(f)
            d["_dir"] = str(Path(f).parent)
            frames.append(d)
        with warnings.catch_warnings():
            # some CSVs carry all-empty columns (e.g. unused constraint slots)
            warnings.simplefilter("ignore", FutureWarning)
            df = pd.concat(frames, ignore_index=True)
        _TREES[root] = df[df.run_ok.astype(str).str.lower() == "true"].reset_index(drop=True)
    return _TREES[root]


def pin(spec, roots):
    """The four panels of one row: {panel: dict(img, lam, r, E)}."""
    df = load_tree(roots[spec["src"]])
    df = df[(df.reward == spec["reward"]) & (df.prompt == spec["prompt"])
            & (df.edit_prompt == spec["edit"]) & (df.seed == spec["seed"])]
    lam_c, lam_h, lam_f = spec["lams"]
    sel = {
        "conicflow": (df.control_type.isin(CONICFLOW_TYPES) & (df.n_inner == 24) & (df.H == 2)
                      & (df.endpoint_surrogate == SURROGATE[spec["src"]])
                      & (df.constraint_on == "endpoint") & np.isclose(df.lam, lam_c, rtol=1e-6)),
        "hardflow": (df.control_type == "hardflow") & (df.n_inner == 40) & np.isclose(df.lam, lam_h, rtol=1e-6),
        "fmrg": (df.control_type == "fmrg") & (df.n_inner == 3) & np.isclose(df.lam, lam_f, rtol=1e-6),
    }
    picks = {}
    for m, mask in sel.items():
        hit = df[mask]
        if len(hit) != 1:
            raise ValueError(f"{spec['reward']} {spec['prompt']!r} -> {spec['edit']!r} seed {spec['seed']}: "
                             f"{m} lam={spec['lams'][list(sel).index(m)]} matched {len(hit)} runs under "
                             f"{roots[spec['src']]}")
        picks[m] = hit.iloc[0]
    c = picks["conicflow"]
    out = {"unc": dict(img=f"{c._dir}/{c.tag}__img_unc.png", lam=np.nan, r=float(c.r_unc), E=0.0)}
    for m, r in picks.items():
        out[m] = dict(img=f"{r._dir}/{r.tag}__img_ctrl.png", lam=float(r.lam), r=float(r.r_ctrl),
                      E=float(r.ctrl_energy) * float(r.N) ** 2)
    return out


# ---- drawing -----------------------------------------------------------------
def _energy_caption(ax, reward, energy, e_color, e_weight, st, stack=False):
    """Centered caption under a panel: reward in the default colour, energy tinted."""
    Packer = VPacker if stack else HPacker
    packed = Packer(children=[
        TextArea(f"$r = {reward:.2f}$" + ("" if stack else "    "),
                 textprops=dict(size=st["caption_fs"], color=st["caption_color"])),
        TextArea(f"$E = {energy}$", textprops=dict(size=st["caption_fs"], color=e_color, weight=e_weight)),
    ], align="center" if stack else "baseline", pad=0, sep=2 if stack else 0)
    ax.add_artist(AnnotationBbox(packed, (0.5, 0.0), xycoords=ax.transAxes,
                                 xybox=(0, -st["caption_pad"]), boxcoords="offset points",
                                 box_alignment=(0.5, 1.0), frameon=False, pad=0, annotation_clip=False))
    # Invisible xlabel so the layout still leaves room under the panel.
    ax.set_xlabel(f"$r = {reward:.2f}$" + ("\n" if stack else "    ") + f"$E = {energy}$",
                  fontsize=st["caption_fs"], color="none", labelpad=st["caption_pad"])


def _box_methods(fig, method_axes, r_avg, pad=STACK_BOX_PAD, pad_b=STACK_BOX_PAD_B, fontsize=STACK_LABEL_FS):
    """Rectangle around the three method panels; matched-reward label on its left."""
    pos = [ax.get_position() for ax in method_axes]
    x0, y0 = min(p.x0 for p in pos) - pad, min(p.y0 for p in pos) - pad_b
    x1, y1 = max(p.x1 for p in pos) + pad, max(p.y1 for p in pos) + pad
    fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure, fill=False,
                             lw=STACK_BOX_LW, edgecolor=STACK_ACCENT, zorder=10, clip_on=False))
    fig.text(x0 - 0.006, 0.5 * (y0 + y1), STACK_LABEL_FMT.format(r=r_avg), transform=fig.transFigure,
             rotation=90, va="center", ha="right", fontsize=fontsize, fontweight="bold", color=STACK_ACCENT)


def _energy_style(p, key, e_min, st):
    if key == "unc":
        return "0", st["caption_color"], "normal"
    if p["E"] == e_min:
        return f"{p['E']:,.0f}", STACK_E_MIN_COLOR, "bold"
    return f"{p['E']:,.0f}", STACK_E_OTHER_COLOR, "normal"


def _frame(ax, st):
    for sp in ax.spines.values():
        sp.set_visible(st["frame"])
        sp.set_linewidth(st["frame_lw"])
        sp.set_color(st["frame_color"])


def draw_stack(rows):
    """Figure 2 (right): rows stacked, column titles on the top row only."""
    st = STYLE
    fig, axes = plt.subplots(len(rows), 4, squeeze=False,
                             figsize=(4 * st["panel_in"], len(rows) * st["panel_h"]))
    for i, (spec, panels) in enumerate(rows):
        e_min = min(panels[m]["E"] for m in PANELS[1:])
        for ax, key in zip(axes[i], PANELS):
            p = panels[key]
            ax.imshow(plt.imread(p["img"]), interpolation=st["interpolation"])
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(TITLES[key], fontsize=st["title_fs"], pad=st["title_pad"],
                             fontweight=st["title_weight"], color=STACK_TITLE_COLORS.get(key, st["title_color"]))
            _energy_caption(ax, p["r"], *_energy_style(p, key, e_min, st), st)
            _frame(ax, st)
        rax = axes[i][-1]
        rax.yaxis.set_label_position("right")
        rax.set_ylabel(STACK_RIGHT_FMT.format(prompt=spec["right_label"]), fontsize=STACK_LABEL_FS,
                       color=STACK_ACCENT, rotation=90, labelpad=8)
    fig.subplots_adjust(wspace=st["wspace"], hspace=STACK_HSPACE)
    for row in axes:
        for ax in row[1:]:
            p = ax.get_position()
            ax.set_position([p.x0 + STACK_UNCONTROLLED_GAP, p.y0, p.width, p.height])
    for i, (_, panels) in enumerate(rows):
        _box_methods(fig, axes[i][1:], np.mean([panels[m]["r"] for m in PANELS[1:]]))
    return fig


def draw_unc_row(panels):
    """One appendix row on a fixed canvas (no tight bbox), so rows set two-up line up."""
    st = {**STYLE, "title_fs": UNC["title_fs"], "caption_fs": UNC["caption_fs"]}
    e_min = min(panels[m]["E"] for m in PANELS[1:])
    w = [UNC["cell_w"], UNC["inner_gap"]] + [UNC["cell_w"]] * 3
    cell_in = (UNC["row_w"] - 2 * UNC["side_in"]) / sum(w)
    fig_h = UNC["top_in"] + cell_in + UNC["bottom_in"]
    fig = plt.figure(figsize=(UNC["row_w"], fig_h))
    gs = fig.add_gridspec(1, len(w), width_ratios=w, wspace=UNC["wspace"],
                          left=UNC["side_in"] / UNC["row_w"], right=1 - UNC["side_in"] / UNC["row_w"],
                          top=1 - UNC["top_in"] / fig_h, bottom=UNC["bottom_in"] / fig_h)
    axs = []
    for key in PANELS:
        p = panels[key]
        ax = fig.add_subplot(gs[0, 0 if key == "unc" else 1 + PANELS.index(key)])
        ax.imshow(plt.imread(p["img"]), interpolation=st["interpolation"])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(TITLES[key], fontsize=st["title_fs"], pad=st["title_pad"],
                     color=STACK_TITLE_COLORS.get(key, st["title_color"]))
        _energy_caption(ax, p["r"], *_energy_style(p, key, e_min, st), st, stack=True)
        _frame(ax, st)
        axs.append(ax)
    _box_methods(fig, axs[1:], float(np.mean([panels[m]["r"] for m in PANELS[1:]])),
                 pad=UNC["box_pad"], pad_b=UNC["box_pad_b_in"] / fig_h, fontsize=UNC["label_fs"])
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unconstrained_root", default=str(IMAGE / "src" / "results" / "unconstrained"))
    ap.add_argument("--editing_root", default=str(IMAGE / "src" / "results" / "ablation"))
    ap.add_argument("--out_dir", default=None, help="default: <unconstrained_root>/figures")
    a = ap.parse_args()
    roots = {"unconstrained": a.unconstrained_root, "editing": a.editing_root}
    out = Path(a.out_dir or Path(a.unconstrained_root) / "figures")
    out.mkdir(parents=True, exist_ok=True)

    table = []

    def record(name, spec, panels):
        rec = dict(figure=name, reward=spec["reward"], prompt=spec["prompt"], edit_prompt=spec["edit"],
                   seed=spec["seed"], r_unc=panels["unc"]["r"])
        for m in PANELS[1:]:
            rec.update({f"{m}_lam": panels[m]["lam"], f"{m}_r": panels[m]["r"], f"{m}_E": panels[m]["E"]})
        rs = [panels[m]["r"] for m in PANELS[1:]]
        rec["reward_spread"] = max(rs) - min(rs)
        rec["cheapest"] = min(PANELS[1:], key=lambda m: panels[m]["E"])
        table.append(rec)

    stack = []
    for spec in ROWS:
        panels = pin(spec, roots)
        stack.append((spec, panels))
        record("uncon_main_images.png:" + spec["name"], spec, panels)
    fig = draw_stack(stack)
    fig.savefig(out / "uncon_main_images.png", dpi=STACK_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out / 'uncon_main_images.png'}")

    for spec in UNC_ROWS:
        panels = pin(spec, roots)
        record(spec["file"], spec, panels)
        fig = draw_unc_row(panels)
        fig.savefig(out / spec["file"], dpi=STYLE["dpi"])   # fixed canvas: no bbox_inches
        plt.close(fig)
        print(f"-> {out / spec['file']}")

    t = pd.DataFrame(table)
    t.to_csv(out / "rows_table.csv", index=False)
    pd.set_option("display.width", 250)
    print(t[["figure", "r_unc", "conicflow_r", "hardflow_r", "fmrg_r", "conicflow_E", "hardflow_E",
             "fmrg_E", "cheapest"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
