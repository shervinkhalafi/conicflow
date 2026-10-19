# Appendix ablations

Three questions, all on the unconstrained PickScore editing task at `N=8` with
`endpoint_surrogate=jump`:

1. **Planning horizon.** Does ConicFlow's advantage depend on `H`?
2. **Matched wall clock.** Does it survive when the baselines get the same compute?
3. **Objective evaluation point.** Does it matter whether the horizon objective
   is scored at the predicted terminal latent (RH-T, the default) or at the
   window end (RH-H)?

A *cell* is one (editing prompt pair, seed). The prompt pairs ed1–ed8 are
listed in `make_jobs.py` and in the headers of `image/data/grids/ablation/ed*.txt`.

## arXiv v1, Appendix D (Tables 7–9)

| table | arms | cells | operating points per row |
|---|---|---|---|
| 7, planning horizon | ConicFlow `H` ∈ {1, 2, 4, 6, 8}, `n_inner=24` | ed1–ed4 × seed 317 | 24 |
| 8, matched wall clock | ConicFlow `n_inner=12`, HardFlow 167, FMRG-J 13 | ed1–ed4 × seeds 317, 1289 | 48 |
| 9, evaluation point | ConicFlow `H=2` RH-T and RH-H; HardFlow 40, FMRG-J 3 | ed1–ed4 × seeds 317, 1289 | 48 |

Every arm runs on the six λ of its prompt pair's `shared` grid line.

```bash
cd image/scripts/ablation
bash run_ablation.sh 0 --preset arxiv_v1           # 72 runs of run_sweep.py, ~20 GPU-h
cd ../../analysis
python abl_tables_arxiv_v1.py                      # Tables 7-9 + the ratios quoted in D.1-D.3
```

`abl_tables_arxiv_v1.py` prints each value next to the one in the paper. In the
paper's tables, `E = N² · ctrl_energy` and every cell is a plain median over the
row's operating points.

**Check against the paper's runs:** run on the original `summary.csv` files,
the script reproduces the paper on almost everything:

- All 36 cells of Tables 7–9 except one: Table 8's ConicFlow time comes out
  at 123 s, against 124 s in the paper. The median is 123.46 s, and no other
  aggregation reproduces all twelve times at once; the 124 was typed into the
  table by hand.
- Every ratio quoted in D.1–D.3, to the paper's rounding. The paper took them
  from the rounded table values, so H=1→H=2 energy shows as −12.6% here
  against −12% there.

**NFE formula:** the "24% more NFE" in D.1 uses
`NFE = n_inner · (N(N−1)/2 + H)`.

## Extended design (prepared for a later revision; not in arXiv v1)

This version scores every arm with the paper's metric `G` instead of raw medians.
It also widens the ablations to eight prompt pairs and four to six seeds.
Every arXiv v1 row is also a row here, with the same output directory and a
subset of its λ, so running the extended design after the preset only adds
runs.

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

**Grid rule:**

- **ed1–ed4 at seeds 317 and 1289:** the `shared` line, as in arXiv v1. On
  ed3 and ed4, both HardFlow arms also run the points of HardFlow's own line
  that are not within 1.25× of a shared point. Shared points cover too little
  of HardFlow's active region there.
- **All other cells:** each method uses its own line, and all ConicFlow arms
  share the ConicFlow line.

`jobs.tsv` holds one row per (cell, arm), with its λ list and run flags.

```bash
cd image/scripts/ablation
bash run_ablation.sh 0                                  # everything, on GPU 0
bash run_ablation.sh 0 --arms core                      # the three default arms only (~27 GPU-h)
bash run_ablation.sh 1 --arms 'conicflow_H4_ni24'       # one horizon
bash run_ablation.sh 0 --seeds '317 1289' &             # two cards: split by seed
bash run_ablation.sh 1 --seeds '2718 3141 1618 1414' &
cd ../../analysis
python abl_figs.py --out figs/ablation     # fig_abl_{horizon,compute,endpoint}.{pdf,png} + tables.md
python abl_data.py                         # coverage per arm + equal-λ horizon numbers
```

The metric is `G`, the mean reward gain over a shared energy budget. Here the
budget is taken over the two arms being compared. Results are reported as
`G / G_default` per cell, pooled as the median with a 95% bootstrap over cells.

**Check against the extended runs:** on those runs, `abl_figs.py` reproduces
every median, cell count and timing of its tables. Four bootstrap-interval
endpoints differ by 0.01: the original bootstrap visited cells in file-system
order, and this script uses a fixed (edit, seed) order.

## Running, common to both

- **One card per process.** `run_ablation.sh <gpu>` pins the card with
  `CUDA_VISIBLE_DEVICES` and runs FLUX and the reward model on `cuda:0`.
- **Filters.** `--arms`, `--tags` and `--seeds` take space-separated names or
  shell globs, `--arms core` selects the three default arms, and `--dry_run`
  prints the commands. To use two cards, run two instances on disjoint seeds.
- **Resumable.** A row is skipped once every λ in it has a `run_ok=True` row in
  its `summary.csv`. Otherwise `run_sweep.py --resume` runs only the missing
  or failed λ.
- **Output.** Runs land in
  `image/src/results/ablation/pickscore/<prompt slug>/<arm>_seed<S>/`, with
  per-row logs in `image/src/results/ablation/logs/`. The analysis scripts read
  this tree by default; `--roots` points them elsewhere.
- **The RH-H arms** lift the locked default `constraint_on=endpoint` with
  `FLOWMAPS_UNLOCK=constraint_on`. The launcher sets this per row, and
  `run_sweep.py` records it in the `unlocked` CSV column.
- **λ grids are frozen.** They are the grids the range finder
  (`image/src/lambda_autorange.py`) wrote when the paper's runs were made; a
  fresh range-find could land on different λ. `python make_jobs.py
  [--preset arxiv_v1]` regenerates both job lists from them. Their λ sets match
  the original runs on every row (72 and 464).
- **GPU-hours** are measured on one RTX A5000 (24 GB) per process, model loads
  included.

## What reproduces exactly and what does not

- **Exact:** the uncontrolled reward `r_unc` is bit-identical for a given
  prompt and seed.
- **Statistical:** the controlled runs reproduce statistically, not digit for
  digit. HardFlow's `ctrl_energy` in particular moves by up to ~1.8× between two
  runs of one config, so a rerun's medians land near the paper's, not on them.
- **Wall clock:** times were measured on shared GPUs and depend on the card.
