#!/usr/bin/env python3
"""abl_figs.py -- the appendix ablation figures and tables.

    python abl_figs.py [--roots DIR [DIR ...]] [--out figs/ablation]

Reads the runs of image/scripts/ablation/run_ablation.sh (see abl_data.py for
the data layer and the metric) and writes

  fig_abl_horizon.pdf/.png   (a) G / G_default vs planning horizon H, with the
                             baselines as reference lines and 95% CI bands
                             (b) control wall clock vs H
  fig_abl_compute.pdf/.png   G / G_default vs wall clock, one line per method
                             through its two inner-iteration settings
  fig_abl_endpoint.pdf/.png  G / G_default vs H for the terminal (RH-T) and
                             window-end (RH-H) objective, on the 16 cells
                             (seeds 317, 1289) where RH-H ran at every H
  tables.md                  every plotted number, plus the equal-lambda
                             numbers quoted in the planning-horizon text

default = ConicFlow RH-T, H=2, n_inner=24. Wall clock is the median control
time per run with the contention filter of abl_data.clean_time; it was measured
on shared A5000s and is indicative only.
"""
from __future__ import annotations

import argparse
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from abl_data import (REF, boot_median, clean_time, horizon_prose, load,  # noqa: E402
                      paired_equal_lam, score_G)

# ---- style: the paper's other result figures ---------------------------------
# matplotlib defaults (DejaVu Sans, full axes box), grid at alpha 0.25, framed
# legend; ConicFlow teal circles / HardFlow orange squares / FMRG purple
# triangles with thin black marker edges; bands at alpha 0.16. The second
# ConicFlow variant (RH-H) takes a darker teal plus a dashed line and hollow
# marker, so it never relies on the shade alone. Sized for a 5.5 in text width.
plt.rcdefaults()
plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "figure.dpi": 200, "savefig.dpi": 300,
})
METHOD_COLORS = dict(conicflow=["#0d9488", "#115e59"],   # teal 600 (primary), teal 800
                     hardflow="#c2410c", fmrg="#6b21a8")
COLOR = {"ConicFlow": METHOD_COLORS["conicflow"][0], "HardFlow": METHOD_COLORS["hardflow"],
         "FMRG-J": METHOD_COLORS["fmrg"]}
MARK = {"ConicFlow": "o", "HardFlow": "s", "FMRG-J": "^"}
LW, MS = 1.7, 5.5
EDGE = dict(markeredgecolor="black", markeredgewidth=0.5)
BAND, GRID = 0.16, 0.25
CF = "ConicFlow (RH-T)"
YLAB = r"$G\,/\,G_{\mathrm{default}}$"
BASE = (("HardFlow", "hardflow_ni40"), ("FMRG-J", "fmrg_nopt3"))


def H_arm(h, suffix=""):
    return f"conicflow_H{h}_ni24{suffix}"


def scores(data, arm, ref=REF):
    return [s for c in data.values() if arm in c and ref in c
            and (s := score_G(c, arm, ref)) is not None]


def arm_time(data, arm):
    t = [x for c in data.values() for x in clean_time(c.get(arm, []))]
    q = st.quantiles(t, n=4)
    return st.median(t), q[0], q[2]


def style_ax(ax):
    ax.grid(alpha=GRID)
    ax.set_axisbelow(True)


def finish(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def ladder(data):
    """H values to plot: those on at least as many cells as H=1 (the 32-cell
    ladder), so a partially-run H is left out rather than drawn on a different
    cell set."""
    n1 = len(scores(data, H_arm(1)))
    return [h for h in (1, 2, 4, 6, 8) if h == 2 or len(scores(data, H_arm(h))) >= n1]


# ── planning horizon ────────────────────────────────────────────────────────
def fig_horizon(data, out, tab):
    Hs = ladder(data)
    xlo, xhi = 0.6, 8.4
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25), gridspec_kw=dict(wspace=0.32))

    ax = axes[0]
    for name, arm in BASE:
        m, lo, hi = boot_median(scores(data, arm))
        ax.axhspan(lo, hi, color=COLOR[name], alpha=BAND, lw=0)
        ax.plot([xlo, xhi], [m, m], color=COLOR[name], lw=LW, ls="--", label=f"{name} (no $H$)")
    ms, los, his = zip(*[boot_median(scores(data, H_arm(h))) for h in Hs])
    ax.fill_between(Hs, los, his, color=COLOR["ConicFlow"], alpha=BAND, lw=0)
    ax.plot(Hs, ms, "-", marker="o", color=COLOR["ConicFlow"], lw=LW, ms=MS, label=CF, **EDGE)
    ax.set_xlim(xlo, xhi)
    ax.set_xticks(Hs)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("planning horizon $H$")
    ax.set_ylabel(YLAB)
    ax.set_title("(a) reward gain at matched energy", loc="left")
    style_ax(ax)

    ax = axes[1]
    tm = {h: arm_time(data, H_arm(h)) for h in Hs}
    for name, arm in BASE:
        ax.plot([xlo, xhi], [arm_time(data, arm)[0]] * 2, color=COLOR[name], lw=LW, ls="--")
    ax.errorbar(Hs, [tm[h][0] for h in Hs],
                yerr=[[tm[h][0] - tm[h][1] for h in Hs], [tm[h][2] - tm[h][0] for h in Hs]],
                fmt="-o", color=COLOR["ConicFlow"], lw=LW, ms=MS, capsize=0, elinewidth=1.0, **EDGE)
    ax.set_xlim(xlo, xhi)
    ax.set_xticks(Hs)
    ax.set_ylim(0, 340)
    ax.set_xlabel("planning horizon $H$")
    ax.set_ylabel("control wall clock per run (s)")
    ax.set_title("(b) cost", loc="left")
    style_ax(ax)
    # one framed legend for both panels (same series); HardFlow and FMRG-J sit
    # on top of each other at ~33 s in (b), and the legend names both
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(CF)] + [i for i, x in enumerate(l) if x != CF]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center",
               bbox_to_anchor=(0.5, 0.97), ncol=3, handlelength=2.2, columnspacing=1.4)
    finish(fig, out, "fig_abl_horizon")

    tab.append("## Planning horizon (fig_abl_horizon)\n")
    tab.append("| arm | cells | G / G(default), median [95% CI] | control s, median [IQR] |")
    tab.append("|---|---|---|---|")
    rows = [(H_arm(h), f"ConicFlow H={h}") for h in (1, 2, 4, 6, 8)] + [(a, n) for n, a in BASE]
    for arm, label in rows:
        v = scores(data, arm)
        m, lo, hi = boot_median(v)
        t = arm_time(data, arm)
        note = "" if arm != H_arm(6) or 6 in Hs else " (not plotted: partial cell set)"
        tab.append(f"| {label}{note} | {len(v)} | {m:.2f} [{lo:.2f}, {hi:.2f}] | "
                   f"{t[0]:.0f} [{t[1]:.0f}, {t[2]:.0f}] |")
    tab.append("")
    tab.append("At equal lambda against H=2 (quoted in the text):\n")
    tab.extend(f"- {line}" for line in horizon_prose(data))
    tab.append("")


# ── matched compute ─────────────────────────────────────────────────────────
RUNGS = (("ConicFlow", (("conicflow_H2_ni12", "12"), (REF, "24"))),
         ("HardFlow", (("hardflow_ni40", "40"), ("hardflow_ni167", "167"))),
         ("FMRG-J", (("fmrg_nopt3", "3"), ("fmrg_nopt13", "13"))))


def fig_compute(data, out, tab):
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    tab.append("## Matched compute (fig_abl_compute)\n")
    tab.append("| method | n_inner | cells | control s (median) | G / G(default), median [95% CI] |")
    tab.append("|---|---|---|---|---|")
    ax.axvspan(105, 140, color="0.5", alpha=0.12, lw=0, zorder=0)
    ax.text(122, 1.16, "matched\ncompute", ha="center", va="top", fontsize=7, color="0.3")
    for meth, rungs in RUNGS:
        xs, ms, lo, hi = [], [], [], []
        for arm, lab in rungs:
            v = scores(data, arm)
            m, a, b = boot_median(v)
            t = arm_time(data, arm)[0]
            xs.append(t); ms.append(m); lo.append(m - a); hi.append(b - m)
            tab.append(f"| {meth} | {lab} | {len(v)} | {t:.0f} | {m:.2f} [{a:.2f}, {b:.2f}] |")
        ax.errorbar(xs, ms, yerr=[lo, hi], fmt="-", marker=MARK[meth], color=COLOR[meth], lw=LW,
                    ms=MS, capsize=0, elinewidth=1.0, label=CF if meth == "ConicFlow" else meth, **EDGE)
        for (arm, lab), x, y in zip(rungs, xs, ms):
            left = meth == "ConicFlow"   # ConicFlow's 12 would sit on FMRG-J's 13
            ax.annotate(lab, (x, y), xytext=(-6 if left else 6, 4), ha="right" if left else "left",
                        textcoords="offset points", fontsize=7, color=COLOR[meth])
    ax.set_xscale("log")
    ax.set_xticks([30, 60, 120, 240])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(24, 330)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("control wall clock per run (s, log scale)")
    ax.set_ylabel(YLAB)
    style_ax(ax)
    ax.legend(loc="lower right", handlelength=2.0)
    finish(fig, out, "fig_abl_compute")
    tab.append("")


# ── objective evaluation point ──────────────────────────────────────────────
def fig_endpoint(data, out, tab):
    cells = {k: c for k, c in data.items() if all(c.get(H_arm(h, "_horizon")) for h in (1, 2, 4, 8))}
    Hs = [h for h in (1, 2, 4, 6, 8) if all(c.get(H_arm(h, "_horizon")) for c in cells.values())]
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    rows = []
    variants = ((CF, "", COLOR["ConicFlow"], "-", "o", None),
                ("ConicFlow (RH-H)", "_horizon", METHOD_COLORS["conicflow"][1], "--", "D", "white"))
    for label, suf, col, ls, mk, face in variants:
        ms, lo, hi = [], [], []
        for h in Hs:
            v = scores(cells, H_arm(h, suf))
            m, a, b = boot_median(v)
            ms.append(m); lo.append(a); hi.append(b)
            pr = [d for d, _ in paired_equal_lam(cells, H_arm(h), H_arm(h, "_horizon"))]
            rows.append((label, h, len(v), m, a, b, sum(d > 0 for d in pr), len(pr)))
        ax.fill_between(Hs, lo, hi, color=col, alpha=BAND, lw=0)
        ax.plot(Hs, ms, ls=ls, marker=mk, color=col, lw=LW, ms=MS, label=label,
                markerfacecolor=face or col, markeredgecolor="black" if face is None else col,
                markeredgewidth=0.5 if face is None else 1.2)
    ax.set_xticks(Hs)
    ax.set_xlim(0.6, 8.4)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("planning horizon $H$  ($N=8$)")
    ax.set_ylabel(YLAB)
    style_ax(ax)
    ax.legend(loc="lower right", handlelength=2.6)
    finish(fig, out, "fig_abl_endpoint")

    pr = paired_equal_lam(data, REF, H_arm(2, "_horizon"))
    v = scores(data, H_arm(2, "_horizon"))
    m, lo, hi = boot_median(v)
    tab.append("## Objective evaluation point (fig_abl_endpoint)\n")
    tab.append(f"H=2, all cells: RH-H G ratio {m:.2f} [{lo:.2f}, {hi:.2f}] on {len(v)} cells; "
               f"RH-T better on delta_r at equal lambda at {sum(d > 0 for d, _ in pr)}/{len(pr)}; "
               f"median energy ratio RH-T/RH-H {st.median(r for _, r in pr if r == r):.2f}; "
               f"control s {arm_time(data, REF)[0]:.0f} / {arm_time(data, H_arm(2, '_horizon'))[0]:.0f}.\n")
    tab.append(f"Across H ({len(cells)} cells, seeds 317 + 1289):\n")
    tab.append("| objective at | H | cells | G / G(default), median [95% CI] | RH-T better at equal lambda |")
    tab.append("|---|---|---|---|---|")
    for label, h, n, m, a, b, w, nn in rows:
        tab.append(f"| {label} | {h} | {n} | {m:.2f} [{a:.2f}, {b:.2f}] | {w}/{nn} |")
    t = {h: (arm_time(cells, H_arm(h))[0], arm_time(cells, H_arm(h, "_horizon"))[0]) for h in Hs}
    tab.append(f"\ncontrol s on the same {len(cells)} cells, RH-T / RH-H: "
               + ", ".join(f"H={h} {a:.0f}/{b:.0f}" for h, (a, b) in t.items()))
    tab.append("")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--roots", nargs="+", default=None,
                    help="directories holding <slug>/<arm>_seed<S>/summary.csv "
                         "(default: image/src/results/ablation/pickscore)")
    ap.add_argument("--out", default=os.path.join("figs", "ablation"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data, log = load(a.roots, dedup="best")
    tab = ["# Ablation tables, paper metric G (table view of the figures)\n",
           "Generated by abl_figs.py. default = ConicFlow RH-T, H=2, n_inner=24.\n"]
    fig_horizon(data, a.out, tab)
    fig_compute(data, a.out, tab)
    fig_endpoint(data, a.out, tab)
    with open(os.path.join(a.out, "tables.md"), "w") as fh:
        fh.write("\n".join(tab) + "\n")
    print("\n".join(tab))
    print(dict(log))


if __name__ == "__main__":
    main()
