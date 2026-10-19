#!/usr/bin/env python3
r"""
constrained_appendix_grids.py -- the appendix grids of the constrained
comparison: one 3x3 figure per (instance, seed), rows m = 1.1, 1.3, 1.5,
columns Uncontrolled | ConicFlow | HardFlow.

Same drawing code, captions, frames and energy convention as
constrained_image_rows.py (Figure 3); see its docstring for what each panel
shows. Two differences: every row is the same instance, so the constraint
label is drawn once (on the middle row) instead of per row, and it is drawn at
full size.

    python constrained_appendix_grids.py
    python constrained_appendix_grids.py --out_dir /tmp/appendix \
        --suite_root ../src/results/constrained_suite --triple_root ../src/results/constrained_triple

Writes <out_dir>/constrained_<tag>_seed<seed>.png (18 files) and
<out_dir>/constrained_appendix.csv with every number on the panels.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constrained_image_rows import (DEFAULT_SUITE, DEFAULT_TRIPLE, FIGURES,  # noqa: E402
                                    STYLE, draw_stack, harvest)

# (tag, root, short constraint names shown beside the grid)
INSTANCES = [
    ("p01_smile",        "suite",  ["smile"]),
    ("p02_sunglasses",   "suite",  ["sunglasses"]),
    ("p04_night",        "suite",  ["night"]),
    ("p05a_redhair",     "suite",  ["red hair"]),
    ("p05b_sad",         "suite",  ["sad"]),
    ("p09_curly_old",    "suite",  ["curly", "old"]),
    ("p10_glass_blonde", "suite",  ["glasses", "blonde"]),
    ("p11_cabin_fog",    "suite",  ["log cabin", "fog"]),
    ("t4_red_sun_path",  "triple", ["flowers", "sunny", "stone path"]),
]
SEEDS = (310, 698)
MS = (1.1, 1.3, 1.5)
GRID_STACK = dict(edit_label_rows="middle", edit_fs_by_ncon={})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.split("\n\n")[1:]))
    p.add_argument("--suite_root", default=str(DEFAULT_SUITE),
                   help="tree with the one/two-constraint instances (default: %(default)s)")
    p.add_argument("--triple_root", default=str(DEFAULT_TRIPLE),
                   help="tree with the three-constraint instance (default: %(default)s)")
    p.add_argument("--out_dir", default=str(FIGURES / "constrained_appendix"),
                   help="output directory (default: %(default)s)")
    p.add_argument("--tags", nargs="+", default=None, help="restrict to these instances")
    p.add_argument("--dpi", type=int, default=STYLE["dpi"])
    a = p.parse_args(argv)

    roots = {"suite": Path(a.suite_root), "triple": Path(a.triple_root)}
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    allrecs = []
    for tag, root, prompts in INSTANCES:
        if a.tags and tag not in a.tags:
            continue
        for seed in SEEDS:
            folder = f"{tag}_seed{seed}"
            specs = [dict(name=f"{folder}_m{m:g}", root=roots[root], folder=folder, m=m,
                          label=folder, prompts=prompts, group=folder) for m in MS]
            recs = harvest(specs)
            allrecs += recs
            fig = draw_stack(recs, stack=GRID_STACK)
            f = out / f"constrained_{folder}.png"
            fig.savefig(f, dpi=a.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"-> {f}")
    t = pd.DataFrame(allrecs).drop(columns=["img"])
    t.to_csv(out / "constrained_appendix.csv", index=False)
    print(f"{len(t)} panels -> {out / 'constrained_appendix.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
