#!/usr/bin/env python3
r"""
constrained_image_rows.py -- the image panels of the constrained comparison
(Figure 3, middle and right): Uncontrolled | one column per method, one row per
(instance, seed, m) cell, rows stacked with the threshold multiplier m
increasing downwards.

The methods are a list (--arms): the default is the paper's pair, ConicFlow
and HardFlow; add the constrained baselines with
`--arms conicflow hardflow pflow ldf` once scripts/constrained/run_suite.sh has
run them (ARMS="pflow ldf") into the same trees.

Reads the tree that scripts/constrained/run_suite.sh writes
(compare_constrained.py --auto_threshold):

    <root>/<tag>_seed<seed>/auto_threshold.json
    <root>/<tag>_seed<seed>/m<m>/<arm>/summary.csv, *__img_{ctrl,unc}.png

Older result trees name the ConicFlow arm `mpc`; both spellings are read.

WHAT EACH PANEL SHOWS. Under every image: h_i for each constraint (h_i <= 0 is
satisfied, coloured green/red) and the control energy E. The frame is green
when every h_i <= 0, red otherwise. h is the in-run value: PickScore of the
final decoded image, before it is saved, i.e. the paper's definition of
feasibility (App C.1.3) and the value Figure 3 (left) counts. Until
2026-10-21 these panels used the saved 8-bit PNGs re-scored
(comparison.json), so they and Figure 3 (left) disagreed on 5 HardFlow runs,
3 of them drawn in the appendix grids. Saving to PNG lowers PickScore by a
median 0.01-0.04 for every method, which flips any run ending within a few
hundredths of the boundary -- and Projected Flow ends exactly there by
construction (80 of its 85 in-run feasible runs read infeasible after
re-scoring).

The Uncontrolled panel's h is tau_i - s0_i, with s0_i the probe score that
defines the thresholds (tau_i = m * s0_i, auto_threshold.json).

ENERGY UNITS. summary.csv's ctrl_energy is the displacement residual
sum_k ||dx_k||^2 dt; the paper reports sum_k ||u_k||^2 dt with u_k = dx_k/dt,
i.e. the stored value times N^2. The factor is applied per panel from that
run's own N.

    python constrained_image_rows.py                     # both paper figures
    python constrained_image_rows.py --arms conicflow hardflow pflow ldf
    python constrained_image_rows.py --out_dir /tmp/fig3
    python constrained_image_rows.py --suite_root ../src/results/constrained_suite \
        --triple_root ../src/results/constrained_triple

Writes <out_dir>/con_main_images_single.png and con_main_images_multi.png, plus
<out_dir>/con_main_images.csv with every number on the panels.
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
DEFAULT_SUITE = SRC / "results" / "constrained_suite"
DEFAULT_TRIPLE = SRC / "results" / "constrained_triple"
FIGURES = HERE.parent / "figures"

ARMS = ("conicflow", "hardflow")                    # the paper's Figure 3
ALL_ARMS = ("conicflow", "hardflow", "pflow", "ldf")
# ConicFlow's arm was called `mpc` in older result trees.
ARM_ALIASES = {"conicflow": ("conicflow", "mpc"), "hardflow": ("hardflow",),
               "pflow": ("pflow",), "ldf": ("ldf",)}
TITLES = {"unc": "Uncontrolled", "conicflow": "ConicFlow", "hardflow": "HardFlow",
          "pflow": "Projected Flow", "ldf": "LDF"}

# The paper's selection for Figure 3: (tag, seed, m, row label, short constraint names).
# `root` is "suite" (one/two constraints) or "triple" (the three-constraint instance).
SELECTED = {
    "single": [
        dict(root="suite", tag="p05a_redhair", seed=698, m=1.3, label="dark → red hair",
             prompts=["red hair"]),
        dict(root="suite", tag="p05b_sad", seed=310, m=1.4, label="neutral → sad",
             prompts=["sad"]),
        dict(root="suite", tag="p01_smile", seed=310, m=1.5, label="man → smiling",
             prompts=["smile"]),
    ],
    "multi": [
        dict(root="suite", tag="p09_curly_old", seed=310, m=1.3,
             label="straight → curly + old", prompts=["curly", "old"]),
        dict(root="suite", tag="p11_cabin_fog", seed=698, m=1.4,
             label="house → log cabin + fog", prompts=["log cabin", "fog"]),
        dict(root="triple", tag="t4_red_sun_path", seed=698, m=1.5,
             label="garden → red flowers + sun + path",
             prompts=["flowers", "sunny", "stone path"]),
    ],
}

# ---- style ----------------------------------------------------------------
STYLE = dict(
    dpi=200,                    # sources are 256x256, so this upsamples ~2x
    interpolation="lanczos",
    title_fs=13, title_pad=6, title_weight="normal", title_color="black",
    caption_fs=9, caption_color="0.15",
    caption_fs_by_ncon={3: 7.0},  # three h's overflow the panel at the default size
    h_sep="  ",
    energy_fmt="$E = {E:.3g}$", energy_unc="$E = 0$",
    h_color_ok="#1a7f37", h_color_bad="#b3261e",
    frame=True, frame_lw=2.0, frame_by_feasibility=True, frame_color="0.75",
    frame_feasible="#2e7d32", frame_infeas="#c62828", frame_unc="#b0b0b0",
    label_color="black", label_pad=10,
)
STACK = dict(
    hspace=0.3, wspace=0.01, panel_in=3.0, label_fs=12,
    edit_fs_by_ncon={3: 9},       # side-label size by constraint count
    title_only_top=True,          # column titles on the first row only
    box=True,                     # frame the method columns; m left, edit label right
    accent="#A51C30",
    uncontrolled_gap=0.04,        # extra space after the Uncontrolled column
    m_fmt=r"$same\ threshold,\mathbf{{\ m = {m:g}}}$",
    box_lw=1.3, box_pad=0.005, box_pad_b=0.04,
    # pflow / ldf: plot_constrained_suite.py's provisional palette slots 4 and 5.
    title_colors=dict(unc="0.35", conicflow="#0d9488", hardflow="#c2410c",
                      pflow="#eda100", ldf="#e87ba4"),
    arrow=True, arrow_label=r"harder constraints ($\mathbf{{m \uparrow}}$) ",
    arrow_gap=0.025, arrow_label_gap=0.018,
    arrow_tail_w=1.2, arrow_head_w=2.4, arrow_head_l=2.0,
    edit_label_rows="all",        # "all" or "middle" (one label per stack)
)


# ---- loading --------------------------------------------------------------
def point_dir(root, folder, m):
    return Path(root) / folder / f"m{m:.2f}"


def _arm_dir(d: Path, arm: str) -> Path:
    for name in ARM_ALIASES[arm]:
        if (d / name).is_dir():
            return d / name
    raise FileNotFoundError(f"{d}: no {' or '.join(ARM_ALIASES[arm])}/ directory")


def _one_png(d: Path, pattern: str) -> str:
    hits = glob.glob(str(d / pattern))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {pattern} in {d}, found {len(hits)}")
    return hits[0]


def _last_row(arm_dir: Path) -> dict:
    rows = list(csv.DictReader(open(arm_dir / "summary.csv")))
    if not rows:
        raise RuntimeError(f"{arm_dir}/summary.csv is empty")
    return rows[-1]


def load_point(root, folder, m, arms=ARMS):
    """Every number for one (folder, m) cell, all of it in-run: h and energy from
    each arm's summary.csv, thresholds and the uncontrolled baselines from
    auto_threshold.json. comparison.json (the re-scored PNGs) is not read; see
    the module docstring. Images are located by globbing the point's own arm
    directories, so a moved results tree still loads."""
    d = point_dir(root, folder, m)
    if not d.is_dir():
        avail = sorted(x.name for x in (Path(root) / folder).glob("m*"))
        raise FileNotFoundError(f"{d} not found; points present for {folder}: {avail or 'none'}")
    dirs = {arm: _arm_dir(d, arm) for arm in arms}
    img = {arm: _one_png(dirs[arm], "*__img_ctrl.png") for arm in arms}
    img["unc"] = _one_png(dirs[arms[0]], "*__img_unc.png")
    rows = {arm: _last_row(dirs[arm]) for arm in arms}
    N = {arm: int(float(rows[arm]["N"])) for arm in arms}

    man = json.loads((Path(root) / folder / "auto_threshold.json").read_text())
    pt = next(p for p in man["points"] if abs(float(p["multiplier"]) - m) < 1e-6)
    cons = [dict(c, threshold=t) for c, t in zip(man["constraints"], pt["thresholds"])]
    # Each run records the thresholds it was actually given (thr_k). They must be
    # the ones this panel prints, or the h shown would be against a different ask.
    for arm in arms:
        for k, c in enumerate(cons):
            thr = float(rows[arm][f"thr_{k}"])
            if abs(thr - c["threshold"]) > 1e-3:
                raise RuntimeError(f"{dirs[arm]}: thr_{k}={thr} but auto_threshold.json "
                                   f"has {c['threshold']} for m={m:g}")
    h_unc = [c["threshold"] - s if c["direction"] == "ge" else s - c["threshold"]
             for c, s in zip(cons, man["uncontrolled_scores"])]
    h = {"uncontrolled": h_unc}
    for arm in arms:
        h[arm] = [float(rows[arm][f"h_{k}"]) for k in range(len(cons))]
    return dict(
        constraints=cons, multiplier=round(m, 2), h=h,
        feasible={k: max(v) <= 0 for k, v in h.items()},
        energy={arm: float(rows[arm]["ctrl_energy"]) for arm in arms},
        N=N, img=img, source="summary.csv")


def harvest(specs, arms=ARMS):
    """One record per panel. `E` is the energy as the paper reports it
    (stored ctrl_energy x N^2 of that run); `E_stored` is the CSV value."""
    recs = []
    for spec in specs:
        pt = load_point(spec["root"], spec["folder"], spec["m"], arms=arms)
        names = [c["arg"] for c in pt["constraints"]]
        for panel in ("unc",) + tuple(arms):
            key = "uncontrolled" if panel == "unc" else panel
            e_stored = 0.0 if panel == "unc" else pt["energy"][panel]
            n = pt["N"][arms[0]] if panel == "unc" else pt["N"][panel]
            recs.append(dict(
                name=spec["name"], label=spec["label"], group=spec.get("group", ""),
                panel=panel, img=pt["img"][panel], m=pt["multiplier"],
                N=n, E_stored=e_stored, E=e_stored * n ** 2,
                feasible=bool(pt["feasible"][key]),
                hs=json.dumps([round(x, 4) for x in pt["h"][key]]),
                names=json.dumps(names),
                prompts=json.dumps(spec.get("prompts") or names),
                thrs=json.dumps([round(c["threshold"], 4) for c in pt["constraints"]]),
                n_con=len(names), folder=spec["folder"], source=pt["source"],
            ))
    return recs


# ---- drawing ---------------------------------------------------------------
def _panel_caption(ax, hs, energy, st, e_color=None, e_weight="normal"):
    """Under an image: h1 = ...  h2 = ... on the first line, E on the second."""
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
    bits = []
    for i, h in enumerate(hs, start=1):
        if bits:
            bits.append(TextArea(st["h_sep"], textprops=dict(size=st["caption_fs"])))
        col = st["h_color_ok"] if h <= 0 else st["h_color_bad"]
        bits.append(TextArea(rf"$h_{i} = {h:+.2f}$",
                             textprops=dict(size=st["caption_fs"], color=col)))
    packed = VPacker(children=[
        HPacker(children=bits, align="baseline", pad=0, sep=0),
        TextArea(energy, textprops=dict(size=st["caption_fs"],
                                        color=e_color or st["caption_color"], weight=e_weight)),
    ], align="center", pad=0, sep=1)
    ax.add_artist(AnnotationBbox(
        packed, (0.5, 0.0), xycoords=ax.transAxes, xybox=(0, -2), boxcoords="offset points",
        box_alignment=(0.5, 1.0), frameon=False, pad=0, annotation_clip=False))


def _edit_h_label(names):
    return ", ".join(f"h{i}: '{n}'" for i, n in enumerate(names, 1))


def _box_methods(fig, method_axes, m, edit_label, sk, edit_fs=None):
    """Rectangle around the method columns; m on the left, edit label on the right."""
    from matplotlib.patches import Rectangle
    pos = [ax.get_position() for ax in method_axes]
    x0 = min(p.x0 for p in pos) - sk["box_pad"]
    y0 = min(p.y0 for p in pos) - sk["box_pad_b"]
    x1 = max(p.x1 for p in pos) + sk["box_pad"]
    y1 = max(p.y1 for p in pos) + sk["box_pad"]
    fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
                             fill=False, lw=sk["box_lw"], edgecolor=sk["accent"],
                             zorder=10, clip_on=False))
    fig.text(x0 - 0.006, 0.5 * (y0 + y1), sk["m_fmt"].format(m=m), transform=fig.transFigure,
             rotation=90, va="center", ha="right", fontsize=sk["label_fs"],
             fontweight="bold", color=sk["accent"])
    return fig.text(x1 + 0.006, 0.5 * (y0 + y1), edit_label, transform=fig.transFigure,
                    rotation=90, va="center", ha="left",
                    fontsize=edit_fs or sk["label_fs"], color=sk["accent"])


def _difficulty_arrow(fig, axes, right_artists, sk):
    """Downward arrow right of the stack: constraints tighten down the rows."""
    from matplotlib.patches import FancyArrowPatch
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    x_right = max(inv.transform(a.get_window_extent(r))[1, 0] for a in right_artists)
    top, bot = axes[0, -1].get_position().y1, axes[-1, -1].get_position().y0
    x = x_right + sk["arrow_gap"]
    fig.add_artist(FancyArrowPatch(
        (x, top), (x, bot), transform=fig.transFigure,
        arrowstyle=f"simple,head_width={sk['arrow_head_w']},"
                   f"head_length={sk['arrow_head_l']},tail_width={sk['arrow_tail_w']}",
        mutation_scale=10, color=sk["accent"], clip_on=False, zorder=10))
    fig.text(x + sk["arrow_label_gap"], 0.5 * (top + bot), sk["arrow_label"],
             transform=fig.transFigure, rotation=270, va="center", ha="left",
             fontsize=sk["label_fs"], fontweight="bold", color=sk["accent"])


def draw_stack(recs, style=None, stack=None):
    """Stacked rows from harvest() records, one row per distinct `name`."""
    st = {**STYLE, **(style or {})}
    sk = {**STACK, **(stack or {})}
    df = pd.DataFrame(recs)
    names = list(dict.fromkeys(df.name))
    m_uniform = len({round(float(x), 2) for x in df.m}) == 1
    # Columns in harvest()'s order: Uncontrolled, then the arms as given.
    ncol = len(dict.fromkeys(df.panel))

    fig, axes = plt.subplots(len(names), ncol, squeeze=False,
                             figsize=(ncol * sk["panel_in"], sk["panel_in"] * len(names)))
    for i, nm in enumerate(names):
        grp = df[df.name == nm]
        panels = grp.to_dict("records")
        e_min = min(p["E"] for p in panels if p["panel"] != "unc")
        n_con = int(grp.n_con.iloc[0])
        st_row = {**st, "caption_fs": st["caption_fs_by_ncon"].get(n_con, st["caption_fs"])}
        for j, p in enumerate(panels):
            ax, key = axes[i, j], p["panel"]
            ax.imshow(plt.imread(p["img"]), interpolation=st["interpolation"])
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0 or not sk["title_only_top"]:
                title = TITLES[key]
                if key != "unc" and not sk["box"] and (m_uniform or not sk["title_only_top"]):
                    title += "   " + rf"$m={p['m']:g}$"
                ax.set_title(title, fontsize=st["title_fs"], pad=st["title_pad"],
                             fontweight=st["title_weight"],
                             color=sk["title_colors"].get(key, st["title_color"]))
            if st["frame"]:
                if not st["frame_by_feasibility"]:
                    col = st["frame_color"]
                elif key == "unc":
                    col = st["frame_unc"]
                else:
                    col = st["frame_feasible"] if p["feasible"] else st["frame_infeas"]
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_linewidth(st["frame_lw"]); sp.set_color(col)
            # The cheaper arm's energy is green, the other red.
            if key == "unc":
                etxt, e_color = st["energy_unc"], st["caption_color"]
            else:
                etxt = st["energy_fmt"].format(E=p["E"])
                e_color = st["h_color_ok"] if p["E"] == e_min else st["h_color_bad"]
            etxt = r"$\mathbf{" + etxt.strip("$") + "}$"   # mathtext ignores fontweight
            _panel_caption(ax, json.loads(p["hs"]), etxt, st_row, e_color=e_color,
                           e_weight="bold")
        if not sk["box"]:
            row_label = grp.label.iloc[0]
            if not m_uniform:
                row_label += f"\n(m = {round(float(grp.m.iloc[0]), 2):g})"
            axes[i, 0].set_ylabel(row_label, fontsize=sk["label_fs"],
                                  color=st["label_color"], labelpad=st["label_pad"])

    fig.subplots_adjust(wspace=sk["wspace"], hspace=sk["hspace"])
    if sk["uncontrolled_gap"]:
        for row in axes:
            for ax in row[1:]:
                p = ax.get_position()
                ax.set_position([p.x0 + sk["uncontrolled_gap"], p.y0, p.width, p.height])
    right = []
    if sk["box"]:
        for i, nm in enumerate(names):
            grp = df[df.name == nm]
            show = (sk["edit_label_rows"] == "all") or (i == len(names) // 2)
            right.append(_box_methods(
                fig, axes[i, 1:], m=round(float(grp.m.iloc[0]), 2),
                edit_label=_edit_h_label(json.loads(grp.prompts.iloc[0])) if show else "",
                sk=sk, edit_fs=sk["edit_fs_by_ncon"].get(int(grp.n_con.iloc[0]))))
    if sk["arrow"]:
        _difficulty_arrow(fig, axes, right or list(axes[:, -1]), sk)
    return fig


def specs_for(group, rows, roots):
    return [dict(r, group=group, root=roots[r["root"]], folder=f"{r['tag']}_seed{r['seed']}",
                 name=f"{r['tag']}__seed{r['seed']}_m{r['m']:g}") for r in rows]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.split("\n\n")[1:]))
    p.add_argument("--suite_root", default=str(DEFAULT_SUITE),
                   help="tree with the one/two-constraint instances (default: %(default)s)")
    p.add_argument("--triple_root", default=str(DEFAULT_TRIPLE),
                   help="tree with the three-constraint instance (default: %(default)s)")
    p.add_argument("--out_dir", default=str(FIGURES / "constrained"),
                   help="output directory (default: %(default)s)")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ALL_ARMS),
                   help="method columns, in order (default: %(default)s)")
    # The paper's Figure 3 panels are saved at 100 dpi (the appendix grids at
    # STYLE's 200); at 100 the default arms reproduce them exactly.
    p.add_argument("--dpi", type=int, default=100)
    a = p.parse_args(argv)

    roots = {"suite": Path(a.suite_root), "triple": Path(a.triple_root)}
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    allrecs = []
    for group, rows in SELECTED.items():
        recs = harvest(specs_for(group, rows, roots), arms=tuple(a.arms))
        allrecs += recs
        fig = draw_stack(recs)
        f = out / f"con_main_images_{group}.png"
        fig.savefig(f, dpi=a.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {f}")
    t = pd.DataFrame(allrecs).drop(columns=["img"])
    t.to_csv(out / "con_main_images.csv", index=False)
    cols = ["group", "folder", "m", "panel", "n_con", "E", "feasible", "hs", "source"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t[cols].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
