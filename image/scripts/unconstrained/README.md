# Unconstrained image experiment (Table 1, Figure 2)

Reward-guided image editing with FLUX.1-dev: ConicFlow vs HardFlow vs FMRG on
12 problems (CLIP, ImageReward, PickScore × 4 prompt/seed instances), each
method swept over the same frozen λ grid. Two baselines added for the
rebuttal, gradient guidance and FlowChef, run on the same problems and λ
(see "Baselines" below).

| Paper artifact | Produced by |
|---|---|
| Table 1, appendix per-problem breakdown | `run_table1.sh` → `analysis/table1_breakdown.py` |
| Figure 2 (left), appendix frontiers (ImageReward cat s42, CLIP man s7) | `run_table1.sh` → `analysis/fig2_pareto.py` |
| Figure 2 (right) row A, appendix rows u05–u10 | `run_table1.sh` → `analysis/fig2_image_rows.py` |
| Figure 2 (right) rows B/C, appendix rows u01–u04 | the ablation experiment's core arms (ed1–ed4, seeds 317/1289, see `../ablation/`) → `analysis/fig2_image_rows.py` |

## Run

From the repository root, inside the environment (FLUX.1-dev access and the
FMRG LoRA weights are required, see the top-level README):

```bash
bash image/scripts/unconstrained/run_table1.sh 0              # GPU 0; all 12 problems
bash image/scripts/unconstrained/run_table1.sh 0 --dry_run    # print the 36 run_sweep.py commands
```

Each (problem, method) is one `run_sweep.py` process writing
`image/src/results/unconstrained/<reward>/<slug>/<method>_seed<seed>/summary.csv`
plus per-run images. The script is resumable: re-running it skips every
config already recorded with `run_ok=True`.

**Cost.** About 26 A5000 GPU-hours. ConicFlow at `n_inner=24` is about 60% of
that (~10 min/run, against ~40 s for HardFlow and FMRG). The paper also ran
ConicFlow at `n_inner=12`, but no table or figure uses it.
`CONICFLOW_NI=24` skips it and brings the total to about 18 GPU-hours.
`ONLY_REWARD=pickscore` runs one reward's four problems.

To split across two cards, run one reward subset per card, e.g.
`ONLY_REWARD=pickscore bash run_table1.sh 0` and `ONLY_REWARD=clip bash run_table1.sh 1`.

## Baselines

Gradient guidance (`gg`, HardFlow's gradient-guidance baseline, 3 guidance
steps per sampling step) and FlowChef (`flowchef`, the FMRG authors' FLUX port,
5 steps) take λ as a gain, as FMRG does, and run on each problem's frozen λ
list, so every method in the table runs at exactly the same λ:

```bash
METHODS="gg flowchef" bash image/scripts/unconstrained/run_table1.sh 0     # the 12 Table 1 problems
# the image-row cells of Fig. 2 (right) rows B/C and Fig. 5 (a)-(d):
METHODS="gg flowchef" JOBS=image_rows_jobs.tsv RESULTS_ROOT=results/ablation/baselines \
    bash image/scripts/unconstrained/run_table1.sh 1
cd image/analysis && python table1_breakdown.py --baselines             # Table 1 with both rows
```

About 2.6 GPU-hours for the Table 1 problems and 1.1 for the image-row cells
(~50 s per λ for gradient guidance and ~30 s for FlowChef, model load
included). `table1_breakdown.py --baselines` reports, per problem, whether the
added methods moved the shared energy budget; on the paper's problems they do
not, so the paper's rows are unchanged. `docs/HYPERPARAMETERS.md` §9 has the
sources and settings.

## Analyse

```bash
cd image/analysis
python table1_breakdown.py      # -> results/unconstrained/figures/table1.tex, table1_breakdown.{tex,csv}
python fig2_pareto.py           # -> uncon_main_pareto.png + the two appendix frontiers
python fig2_image_rows.py       # -> uncon_main_images.png + unconstrained_u01..u10_*.png
```

Defaults read `image/src/results/unconstrained` (and, for `fig2_image_rows.py`,
`image/src/results/ablation` via `--editing_root`); every path is a flag.
On the paper's own runs all three reproduce Table 1, the breakdown table and
every figure above exactly (`table1_breakdown.py --check_paper` asserts the
Table 1 numbers). A fresh re-run will land close but not digit-identical:
HardFlow's control energy in particular varies between otherwise identical
runs, so compare a re-run against the printed paper numbers rather than
asserting them.

## Settings

All as in Appendix "Unconstrained image experiment": N=8, int8 transformer,
fp32 reward, guidance 3.5, 256×256 (all but N are `run_sweep.py` defaults).

- **ConicFlow:** H=2, n_inner 24 (and 12), Adam lr 1e-2, terminal estimate by
  rollout, drift detached for ℓ>0. The last two are `run_sweep.py`'s defaults
  (`endpoint_surrogate=rollout`, `drift_detach=True`), which is why the script
  does not pass them.
- **HardFlow:** n_inner 40, lr 2.5 with `--hardflow_lr_alpha 0.5` (lr → 0.5/reg
  wherever lr·reg ≥ 1). `reg = N/(2λ)` is derived per run.
- **FMRG-J:** 3 guidance steps.
- **Gradient guidance / FlowChef:** 3 / 5 guidance steps; λ is a gain.

**λ grids** are frozen in `image/data/grids/unconstrained/<reward>__<slug>.txt`.
`run_table1.sh` uses each file's `shared` line, a 6-point grid common to all
three methods. Three problems get 5 extra log-midpoints each, so the plotted
frontiers have 11 points: PickScore man→smiling at seeds 42 and 7, and
ImageReward grass→poppies at seed 42. Those extra λ are listed in
`table1_jobs.tsv`.

`summary.csv` stores `ctrl_energy` as E/N². The analysis scripts rescale every
row by its own N² to get the paper's E = Σ Δt‖u‖².

## Re-deriving the λ grids instead

The grids came from `lambda_autorange.py`, which probes at a cheap N=4 proxy.
To range-find afresh rather than use the frozen grids:

```bash
cd image/scripts
export TARGET_N=8 CONICFLOW_H=2 CONICFLOW_NI="12 24" HF_NI=40 HF_LR=2.5 FMRG_NOPT=3 \
       SHARED_GRID=1 HF_ALPHA=0.5 AUTORANGE_EXTRA="--n_grid_shared 6" \
       AUTO_ROOT=results/unconstrained_reprobe
SEEDS="42 7" ./auto_sweep.sh 0 pickscore "a portrait of a man" "a smiling face"
SEEDS=42     ./auto_sweep.sh 0 pickscore "a photo of a sleeping cat" "a photo of a cat with wide open eyes"
SEEDS=42     ./auto_sweep.sh 0 pickscore "a field of green grass" "a field of red poppies"
# ... and the same three lines for clip and imagereward
```

This adds about 15 min of range finding per (reward, prompt pair). The result
can land on a different grid from the paper's: the probe is a measurement, and
on a fresh probe `auto_sweep.sh` also trims the grid with `trim_grids.py`,
which the paper's grids predate. The densify points are not added. The output
tree has the same layout, so
`python table1_breakdown.py --results ../src/results/unconstrained_reprobe`
reads it.
