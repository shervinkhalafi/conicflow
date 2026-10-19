#!/usr/bin/env python3
"""
plot_autorange_overlay.py -- overlay ConicFlow / HardFlow / FMRG on one panel per
reward, from the data lambda_autorange.py already wrote.

Reads results/autorange/<reward>/autorange.json (skipping _-prefixed archive
dirs) and produces two figures:

  lambda_overlay.png    delta_r vs lambda, one panel per reward, three methods
                        overlaid. Within a reward every method shares the same
                        seed, N, prompt and therefore the same r_unc (asserted
                        below), so delta_r is directly comparable. lambda is
                        NOT: it is a Lagrange weight for ConicFlow, enters HardFlow
                        as reg = N/(2*lambda), and is an open-loop gradient
                        gain for FMRG. Read the x positions as "where each
                        method's own transition sits", not as a shared setting.

  frontier_overlay.png  delta_r vs control energy, same panels. This one IS
                        apples-to-apples -- it is the comparison the repo's
                        README says does not depend on what each knob means --
                        so it is the figure to use when asking which method
                        buys more reward per unit of control effort.

Usage: python plot_autorange_overlay.py [--root results/autorange] [--out .]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = [("conicflow", "#2a78d6", "o"), ("hardflow", "#eb6834", "s"), ("fmrg", "#1baf7a", "^")]
REWARD_ORDER = ["imagereward", "clip", "pickscore"]


def load(root):
    runs = {}
    for f in sorted(glob.glob(os.path.join(root, "[!_]*", "autorange.json"))):
        d = json.load(open(f))
        runs[d["reward"]] = d
    return runs


def ordered(runs):
    keys = [r for r in REWARD_ORDER if r in runs] + [r for r in runs if r not in REWARD_ORDER]
    return [(k, runs[k]) for k in keys]


def panels(n, ylabel, xlabel):
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.4), squeeze=False)
    axes = axes[0]
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=.25, lw=.6)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    return fig, axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "results", "autorange"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or a.root
    runs = load(a.root)
    if not runs:
        raise SystemExit(f"no autorange.json under {a.root}")
    items = ordered(runs)

    # ── Figure 1: delta_r vs lambda ───────────────────────────────────────
    fig, axes = panels(len(items), "Δr  (r_ctrl − r_unc)", "λ")
    for ax, (reward, d) in zip(axes, items):
        # r_unc is what makes delta_r comparable between the methods; verify it.
        runcs = {round(p["r_unc"], 6) for r in d["results"] for p in r["points"] if p["ok"]}
        assert len(runcs) == 1, f"{reward}: methods disagree on r_unc {runcs}"
        ax.axhline(d["ceiling"], color="#8a95a0", ls="--", lw=1, alpha=.8,
                   label=f"ceiling est. {d['ceiling']:.2f}")
        ax.axhline(0, color="#8a95a0", lw=.8, alpha=.5)
        for name, col, mk in METHODS:
            r = next((x for x in d["results"] if x["method"] == name), None)
            if r is None:
                continue
            pts = sorted((p for p in r["points"] if p["ok"]), key=lambda p: p["lam"])
            if pts:
                ax.plot([p["lam"] for p in pts], [p["dr"] for p in pts], "-", color=col, lw=1.8,
                        marker=mk, ms=5, mec="white", mew=.8, label=name, zorder=3)
            for p in r["points"]:                      # e.g. HardFlow's lr*reg >= 1 floor
                if not p["ok"]:
                    ax.axvline(p["lam"], color=col, ls=":", lw=1.4, alpha=.7, zorder=1)
                    ax.plot([p["lam"]], [0], "x", color=col, ms=8, mew=2, zorder=4)
        ax.set_title(f"{reward}   (r_unc = {d['probe']['r_unc']:.3f},  ‖∇r‖ = {d['probe']['grad_norm']:.3g})",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper left" if reward == "pickscore" else "best")
    fig.suptitle("λ response by method, per reward   ·   N=4 probe, "
                 '"a portrait of a man" → "a smiling face", seed 42.   '
                 "λ is not the same quantity across methods — compare the shape and location, not the value.",
                 fontsize=9.5, y=1.005)
    fig.tight_layout()
    p1 = os.path.join(out, "lambda_overlay.png")
    fig.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: the λ-independent comparison ────────────────────────────
    fig, axes = panels(len(items), "Δr  (r_ctrl − r_unc)", "control energy  Σ‖u‖²Δt")
    for ax, (reward, d) in zip(axes, items):
        ax.axhline(d["ceiling"], color="#8a95a0", ls="--", lw=1, alpha=.8, label=f"ceiling est. {d['ceiling']:.2f}")
        for name, col, mk in METHODS:
            r = next((x for x in d["results"] if x["method"] == name), None)
            if r is None:
                continue
            pts = sorted((p for p in r["points"] if p["ok"] and p["E"] > 0), key=lambda p: p["E"])
            if pts:
                ax.plot([p["E"] for p in pts], [p["dr"] for p in pts], "-", color=col, lw=1.8,
                        marker=mk, ms=5, mec="white", mew=.8, label=name, zorder=3)
        ax.set_title(f"{reward}", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Energy/reward frontier — up and to the LEFT is better.   "
                 "This comparison does not depend on what λ means in each method.", fontsize=9.5, y=1.005)
    fig.tight_layout()
    p2 = os.path.join(out, "frontier_overlay.png")
    fig.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p1}\nwrote {p2}")


if __name__ == "__main__":
    main()
