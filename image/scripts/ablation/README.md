# Appendix ablations

Three questions, all on the unconstrained PickScore editing task at `N=8` with
`endpoint_surrogate=jump`:

1. **Planning horizon.** Does ConicFlow's advantage depend on `H`?
2. **Matched wall clock.** Does it survive when the baselines get the same compute?
3. **Objective evaluation point.** Does it matter whether the horizon objective
   is scored at the predicted terminal latent (RH-T, the default) or at the
   window end (RH-H)?

## Design

A *cell* is one (editing prompt pair, seed). There are eight portrait edits,
ed1–ed8, listed in `make_jobs.py` and in the headers of
`image/data/grids/ablation/ed*.txt`.

| arm | what it varies | cells | seeds | GPU-h |
|---|---|---|---|---|
| `conicflow_H2_ni24` | default (RH-T, `H=2`, 24 inner steps) | 48 | 317 1289 2718 3141 1618 1414 | 19 |
| `hardflow_ni40` | HardFlow, authors' 40 inner steps | 48 | same | 3.5 |
| `fmrg_nopt3` | FMRG-J, authors' 3 guidance steps | 48 | same | 4 |
| `conicflow_H{1,4,6,8}_ni24` | planning horizon | 32 each | 317 1289 2718 3141 | 11 / 15 / 16 / 14.5 |
| `conicflow_H2_ni12` | compute: ConicFlow at ~120 s/run | 32 | 317 1289 2718 3141 | 7 |
| `hardflow_ni167` | compute: HardFlow at ~120 s/run | 32 | same | 6 |
| `fmrg_nopt13` | compute: FMRG-J at ~120 s/run | 32 | same | 7.5 |
| `conicflow_H2_ni24_horizon` | RH-H objective | 32 | same | 5 |
| `conicflow_H{1,4,6,8}_ni24_horizon` | RH-H across `H` | 16 each | 317 1289 | 1 / 5 / 7 / 7 |
| **total** | 464 runs of `run_sweep.py`, 5–9 λ each | | | **~130** |

GPU-hours are measured on one RTX A5000 (24 GB) per process, model loads
included.

The horizon-objective arms lift the locked default `constraint_on=endpoint` with
`FLOWMAPS_UNLOCK=constraint_on`. The launcher sets this per row, and
`run_sweep.py` records it in the `unlocked` CSV column.

**λ grids are frozen.** Every λ comes from `image/data/grids/ablation/`, which
holds the grids the range finder (`image/src/lambda_autorange.py`) wrote when
the paper's runs were made. Re-range-finding could land on different λ.

- **ed1–ed4 at seeds 317 and 1289:** every arm uses the `shared` line. On ed3
  and ed4, both HardFlow arms also run the points of HardFlow's own line that
  are not within 1.25× of a shared point (shared points cover too little of
  HardFlow's active region there).
- **All other cells:** each method uses its own line, and all ConicFlow arms
  share the ConicFlow line.

`jobs.tsv` is the result: one row per (cell, arm) with its λ list and flags.
Regenerate it with `python make_jobs.py > jobs.tsv`. Its λ sets match the
paper's runs exactly on all 464 rows.

## Running

```bash
cd image/scripts/ablation
bash run_ablation.sh 0                                  # everything, on GPU 0
bash run_ablation.sh 0 --arms core                      # the three default arms only (~27 GPU-h)
bash run_ablation.sh 1 --arms 'conicflow_H4_ni24'       # one horizon
bash run_ablation.sh 0 --arms '*_horizon'               # the objective-point arms
bash run_ablation.sh 0 --seeds '317 1289' &             # two cards: split by seed
bash run_ablation.sh 1 --seeds '2718 3141 1618 1414' &
bash run_ablation.sh 0 --dry_run                        # print the commands
```

`--arms`, `--tags` and `--seeds` take space-separated names or shell globs.
The launcher is resumable: a row is skipped once every λ in it has a
`run_ok=True` row in its `summary.csv`. Otherwise `run_sweep.py --resume` runs
only the missing or failed λ.

Runs land in `image/src/results/ablation/pickscore/<prompt slug>/<arm>_seed<S>/`,
with per-row logs in `image/src/results/ablation/logs/`.

## Figures and tables

```bash
cd image/analysis
python abl_figs.py --out figs/ablation     # fig_abl_{horizon,compute,endpoint}.{pdf,png} + tables.md
python abl_data.py                         # coverage per arm + the equal-λ numbers in the text
```

Both default to the results tree above. `--roots` points them elsewhere.

The metric is the paper's `G` (mean reward gain over a shared energy budget),
taken over the two arms being compared. It is reported as `G / G_default` per
cell and pooled as the median with a 95% bootstrap over cells.

## What reproduces exactly and what does not

- **Exact:** the uncontrolled reward `r_unc` is bit-identical for a given
  prompt and seed.
- **Statistical:** the controlled runs reproduce statistically, not digit for
  digit. HardFlow's `ctrl_energy` in particular moves by up to ~1.8× between two
  runs of one config.
- **Bootstrap intervals:** with the paper's data, `abl_figs.py` reproduces every
  median, cell count and timing in the appendix tables. Four bootstrap-interval
  endpoints differ by 0.01, because the paper's bootstrap visited cells in
  file-system order and this script visits them in a fixed (edit, seed) order.
- **Wall clock:** times were measured on shared GPUs and depend on the card.
