# Constrained image editing (Figure 3 and the appendix grids)

ConicFlow against HardFlow with **no reward** and one to three PickScore
**constraints**, each `h_i(x) = tau_i - PickScore(x, c_i) <= 0`, enforced at
the final sample. Each method minimises its control energy alone, so both must
reach the same feasible set; the comparison is what that costs and whether
they get there. FMRG has no constraint mechanism and is not part of it.
The constrained baselines added for the rebuttal, Projected Flow (PDM) and
LDF, run on the same instances, thresholds and seeds (see "Baselines" below).

Thresholds are set per instance from its own uncontrolled sample:
`tau_i(m) = m * PickScore(x_unc, c_i)`, for a tightness multiplier `m`.

## Instances

All seeds are `310` and `698`. Job files are in `jobs/`.

| tag | base prompt | constraint prompts `c_i` | `m` grid |
|---|---|---|---|
| `p01_smile` | a portrait of a man | a smiling face | 1.1 1.2 1.3 1.4 1.5 |
| `p02_sunglasses` | a photo of a golden retriever | a dog wearing sunglasses | 1.1 … 1.5 |
| `p04_night` | a city street at noon | a city street at night | 1.1 … 1.5 |
| `p05a_redhair` | a portrait of a woman | a woman with bright red hair | 1.1 … 1.5 |
| `p05b_sad` | a portrait of a woman | sad frowning woman | 1.1 … 1.5 |
| `p09_curly_old` | portrait of a young man with straight hair | man with curly hair; photo of very old man | 1.1 … 1.5 |
| `p10_glass_blonde` | a portrait of a woman | a woman wearing glasses; a woman with short blonde hair | 1.1 … 1.5 |
| `p11_cabin_fog` | a house by a lake | a wooden log cabin; a house in heavy fog | 1.1 … 1.5 |
| `t4_red_sun_path` (`jobs/triple.tsv`) | a green garden on an overcast day | a garden with red flowers; a garden in bright sunshine; a garden with a stone path | 1.1 1.3 1.5 |

The first eight (`jobs/suite.tsv`) give Figure 3 (left): 16 (instance, seed)
problems per `m`. The three-constraint instance appears in the image panels
and the appendix grids only.

Settings: `lambda = 20`, `N = 8`. ConicFlow uses `H = 2`, 24 inner Adam steps
at lr `1e-2`, the jump terminal estimate, `mu_init = 0.5` and dual step `0.5`.
HardFlow uses 40 inner steps at lr `0.5`, `reg = N/(2*lambda) = 0.2`,
`mu_init = 0` reset at every sampling step, dual step `0.2` every 8 inner
steps, and `rho = 5` held fixed. The header of `run_suite.sh` and
`docs/HYPERPARAMETERS.md` §8 explain the HardFlow choices.

## Baselines

Two more arms, run into the same trees so every method shares the probe and
thresholds of each (instance, seed):

| arm | method | settings |
|---|---|---|
| `pflow` | Projected Flow: PDM-style Gauss-Newton projection of the state onto `{h <= 0}` after every sampling step, through the flow-map endpoint | at most 10 updates per step, 50 at the last, margin 0 |
| `ldf` | Lagrangian Dual Flows: a multiplier ODE beside the sample, split-step with the shared nominal step | penalty 5, gain 1, p = 1, 64 substeps per step, initial multiplier 0 |

Neither uses `lambda` or a horizon. `docs/HYPERPARAMETERS.md` §9 gives the
sources and explains each setting; `src/control_loops.py` lists every
deviation from the authors' code.

## Commands

From anywhere, inside the environment set up as in the top-level README:

```bash
# 1. the runs (resumable; one card per process)
bash image/scripts/constrained/run_suite.sh 0               # 8 instances x 2 seeds x 5 m
bash image/scripts/constrained/run_suite.sh 1 triple        # the 3-constraint instance
bash image/scripts/constrained/run_suite.sh 0 --dry_run     # print the commands only
ARMS="pflow ldf" bash image/scripts/constrained/run_suite.sh 0          # the baselines, same trees
ARMS="pflow ldf" bash image/scripts/constrained/run_suite.sh 1 triple

# 2. the figures (CPU, seconds)
cd image/src
python plot_constrained_suite.py --root results/constrained_suite --paper_style \
    --out ../figures/constrained/con_main_plot.png          # Figure 3, left
python ../analysis/constrained_image_rows.py                # Figure 3, middle and right
python ../analysis/constrained_appendix_grids.py            # appendix, 18 grids
# with the baselines (the plot picks up every arm present in the tree)
python plot_constrained_suite.py --root results/constrained_suite --paper_style \
    --out ../figures/constrained/con_main_plot_4arms.png
python ../analysis/constrained_image_rows.py --arms conicflow hardflow pflow ldf \
    --out_dir ../figures/constrained_4arms
python ../analysis/constrained_appendix_grids.py --arms conicflow hardflow pflow ldf \
    --out_dir ../figures/constrained_appendix_4arms
```

`plot_constrained_suite.py` prints the numbers behind the plot and saves them
as a `.csv` next to the figure. The two image scripts write a `.csv` with
every number on their panels. Run directories are written to
`image/src/results/constrained_suite/` and `.../constrained_triple/`; the
figures go to `image/figures/constrained/` and `.../constrained_appendix/`.

## Cost

On one RTX A5000 (24 GB), the paper's runs took about 11.5 GPU-hours for the
8-instance suite and 0.7 GPU-hours for the three-constraint instance, about
40 min per (instance, seed) on the five-point grid. Peak memory grows with the
number of constraint scorers: 16.2 / 19.2 / 22.2 GB for one / two / three
constraints, so the three-constraint instance needs a card to itself.
The two baselines together add about 6 GPU-hours for the suite and 0.7 for the
three-constraint instance (Projected Flow 80-140 s per run, ~190 s with three
constraints, peaking at 22.8 GB; LDF 45-70 s).

## Notes for reproduction

- **Two HardFlow runs in the paper's data ended with `run_ok=False`:**
  `p04_night` seed 698 at `m = 1.1` and `t4_red_sun_path` seed 310 at
  `m = 1.1`. This was a false positive of a guard in `run_sweep.py`, fixed
  since (`docs/HYPERPARAMETERS.md` §8); the rows' numbers are complete.
  `compare_constrained.py` wrote no `comparison.json` for those points.
  `plot_constrained_suite.py` and the image scripts read each arm's last
  `summary.csv` row whatever its `run_ok`, so both runs are counted and drawn.
- **Feasibility is the in-run value, everywhere.** Every figure judges
  feasibility from the `h_i` in each arm's `summary.csv`: the constraint
  values of the final decoded image, computed inside the control loop
  (Appendix C.1.3's definition). `comparison.json` also records the saved PNGs
  re-scored with the same PickScore code; saving to 8 bits lowers PickScore by
  a median 0.01-0.04, enough to flip a run that ends on the boundary. Until
  2026-10-21 the image panels used the re-scored values, which disagreed with
  Figure 3 (left) on five HardFlow runs, three of them drawn in the appendix
  grids; Projected Flow, which ends just inside the boundary by construction,
  would read infeasible on 80 of its 85 feasible runs.
- **Energy units.** `summary.csv` stores `ctrl_energy` as `sum_k ||dx_k||^2 dt`.
  The paper reports `sum_k ||u_k||^2 dt` with `u_k = dx_k / dt`, which is `N^2`
  (64) times larger. All three scripts apply that factor per run from its own
  `N`.
- Older result trees name the ConicFlow arm `mpc/`; all three scripts read
  either name.
