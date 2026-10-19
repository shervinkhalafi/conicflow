#!/usr/bin/env python3
"""make_jobs.py -- build the job lists of the appendix ablations.

    python make_jobs.py --preset arxiv_v1 > jobs_arxiv_v1.tsv   # arXiv v1, Tables 7-9
    python make_jobs.py > jobs.tsv                              # extended design

One row = one (cell, arm) = one `run_sweep.py` invocation over that arm's
lambda list, where a cell is one (editing prompt pair, seed). The output is
committed next to this script; regenerating it from the frozen grids in
image/data/grids/ablation/ reproduces it exactly.

COLUMNS (tab-separated):
    tag  seed  arm  unlock  lams  prompt  edit_prompt  args

`unlock` is the FLOWMAPS_UNLOCK value for the row ('-' for none); only the
horizon-objective arms set it. `args` are the arm's run_sweep.py flags.

PRESET arxiv_v1 (the ablations of arXiv v1, Appendix D, Tables 7-9)
  ed1-ed4 only, every arm on the six `shared` lambdas (no top-up):
  * Table 7, planning horizon: ConicFlow H in {1, 2, 4, 6, 8} at seed 317.
  * Table 8, matched wall clock: ConicFlow n_inner 12, HardFlow 167, FMRG-J 13
    at seeds 317 and 1289.
  * Table 9, evaluation point: ConicFlow H=2 endpoint and horizon, HardFlow 40,
    FMRG-J 3 at seeds 317 and 1289.
  72 rows. Every one of them is also a row of the extended design (same output
  directory, a subset of its lambdas), so the extended design can be run on
  top of the preset without re-running anything.

EXTENDED DESIGN (prepared for a later revision; not in arXiv v1)
  8 editing pairs (ed1-ed8) x seeds {317, 1289, 2718, 3141} = 32 cells, each
  running all 11 arms of the H ladder, the compute ladder and the H=2 horizon
  arm. Seeds 317 and 1289 (16 cells) also run the horizon objective at
  H in {1, 4, 6, 8}. Seeds 1618 and 1414 (16 more cells) run only the three
  default arms, which is where run-to-run noise in the baseline comparison
  matters most.

GRIDS
  Every lambda comes from the frozen grids, never from a fresh range-find.
  * ed1-ed4 at seeds 317 and 1289 use the `shared` line for every arm. On ed3
    and ed4 the shared grid put only 3/6 and 2/6 points in HardFlow's active
    region, so both HardFlow arms there also run the points of HardFlow's own
    line that are not within 1.25x of a shared point (the "top-up").
  * Every other cell uses each method's own line; all ConicFlow arms of a cell
    share the ConicFlow line, so ConicFlow-vs-ConicFlow pairs sit at equal
    lambda. Across methods the comparison is at matched control energy, not at
    equal lambda, so there is no reason to place points outside a method's
    active region.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GRID_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "data", "grids", "ablation"))

PROMPTS = [
    ("ed1", "a portrait of a woman with straight hair", "a portrait of a woman with curly hair"),
    ("ed2", "a portrait of a woman with black hair", "a portrait of a woman with white hair"),
    ("ed3", "a portrait of a young man", "a portrait of an old man"),
    ("ed4", "a portrait of a smiling man", "a portrait of an angry man"),
    ("ed5", "a portrait of a serious woman", "a portrait of a smiling woman"),
    ("ed6", "a portrait of a clean-shaven man", "a portrait of a man with a mustache"),
    ("ed7", "a portrait of a thin man", "a portrait of a chubby man"),
    ("ed8", "a portrait of a happy woman", "a portrait of a sad woman"),
]
LADDER_SEEDS = (317, 1289, 2718, 3141)
HORIZON_SEEDS = (317, 1289)
CORE_SEEDS = (1618, 1414)
SHARED_TAGS, SHARED_SEEDS = {"ed1", "ed2", "ed3", "ed4"}, {317, 1289}
TOPUP_TAGS, TOPUP_RATIO = {"ed3", "ed4"}, 1.25

COMMON = "--reward pickscore --sweep_N 8 --sweep_endpoint_surrogate jump"
CF = "--sweep_control_type conicflow --sweep_lr_inner 1e-2"
HF = "--sweep_control_type hardflow --sweep_H 2 --sweep_lr_inner 2.5 --hardflow_lr_alpha 0.5"
FM = "--sweep_control_type fmrg"

# arm -> (grid line, unlock, args)
CORE = {
    "conicflow_H2_ni24": ("conicflow", "-", f"{CF} --sweep_H 2 --sweep_n_inner 24"),
    "hardflow_ni40":     ("hardflow", "-", f"{HF} --sweep_n_inner 40"),
    "fmrg_nopt3":        ("fmrg", "-", f"{FM} --sweep_n_inner 3"),
}
COMPUTE = {
    "conicflow_H2_ni12": ("conicflow", "-", f"{CF} --sweep_H 2 --sweep_n_inner 12"),
    "hardflow_ni167":    ("hardflow", "-", f"{HF} --sweep_n_inner 167"),
    "fmrg_nopt13":       ("fmrg", "-", f"{FM} --sweep_n_inner 13"),
}


def _horizon(h):
    return ("conicflow", "constraint_on",
            f"{CF} --sweep_H {h} --sweep_n_inner 24 --sweep_constraint_on horizon")


H_LADDER = {f"conicflow_H{h}_ni24": ("conicflow", "-", f"{CF} --sweep_H {h} --sweep_n_inner 24")
            for h in (1, 4, 6, 8)}
LADDER_ARMS = {**CORE, **COMPUTE, "conicflow_H2_ni24_horizon": _horizon(2), **H_LADDER}
HORIZON_X_H = {f"conicflow_H{h}_ni24_horizon": _horizon(h) for h in (1, 4, 6, 8)}


def read_grids(path):
    g = {}
    for line in open(path):
        parts = line.split("#")[0].split()
        if len(parts) >= 3 and parts[1] == "--sweep_lam":
            g[parts[0]] = parts[2:]
    for need in ("conicflow", "hardflow", "fmrg", "shared"):
        if need not in g:
            raise SystemExit(f"{path}: no '{need} --sweep_lam' line")
    return g


def lams_for(tag, seed, arm, line, g):
    if tag in SHARED_TAGS and seed in SHARED_SEEDS:
        lams = list(g["shared"])
        if tag in TOPUP_TAGS and arm.startswith("hardflow"):
            shared = [float(x) for x in g["shared"]]
            lams += [x for x in g["hardflow"]
                     if all(max(float(x), s) / min(float(x), s) > TOPUP_RATIO for s in shared)]
        return lams
    return list(g[line])


def arxiv_v1_rows():
    """(tag, seed, arm) of arXiv v1's Tables 7-9, in table order."""
    keep = []
    for tag in ("ed1", "ed2", "ed3", "ed4"):
        keep += [(tag, 317, f"conicflow_H{h}_ni24") for h in (1, 2, 4, 6, 8)]
        for seed in (317, 1289):
            keep += [(tag, seed, a) for a in COMPUTE]
            keep += [(tag, seed, a) for a in ("conicflow_H2_ni24", "conicflow_H2_ni24_horizon",
                                              "hardflow_ni40", "fmrg_nopt3")]
    return list(dict.fromkeys(keep))   # Table 7's H=2 at seed 317 is also Table 9's


def main():
    preset = None
    if sys.argv[1:] == ["--preset", "arxiv_v1"]:
        preset = "arxiv_v1"
    elif sys.argv[1:]:
        raise SystemExit("usage: make_jobs.py [--preset arxiv_v1]")
    rows = []
    for tag, p, e in PROMPTS:
        g = read_grids(os.path.join(GRID_DIR, f"{tag}.txt"))
        cells = [(s, LADDER_ARMS) for s in LADDER_SEEDS]
        cells += [(s, HORIZON_X_H) for s in HORIZON_SEEDS]
        cells += [(s, CORE) for s in CORE_SEEDS]
        for seed, arms in cells:
            for arm, (line, unlock, args) in arms.items():
                lams = lams_for(tag, seed, arm, line, g)
                rows.append("\t".join([tag, str(seed), arm, unlock, " ".join(lams),
                                       p, e, f"{COMMON} {args}"]))
    if preset == "arxiv_v1":
        # same rows, but on the six shared lambdas only: the HardFlow top-up
        # points were added after arXiv v1 and are not in its tables
        by_key = {tuple(r.split("\t")[:3]): r.split("\t") for r in rows}
        grids = {tag: read_grids(os.path.join(GRID_DIR, f"{tag}.txt")) for tag, _, _ in PROMPTS}
        rows = []
        for tag, seed, arm in arxiv_v1_rows():
            f = by_key[(tag, str(seed), arm)]
            f[4] = " ".join(grids[tag]["shared"])
            rows.append("\t".join(f))
    print("#tag\tseed\tarm\tunlock\tlams\tprompt\tedit_prompt\targs")
    print("\n".join(rows))
    print(f"make_jobs: {len(rows)} jobs", file=sys.stderr)


if __name__ == "__main__":
    main()
