# Low-dimensional experiment: two moons (paper Figure 1)

ConicFlow, HardFlow and FMRG steering a pretrained two-moons flow map toward
a terminal reward, traced as reward against control energy by sweeping the
control-cost weight λ. Everything runs on CPU.

| file | what it is |
|---|---|
| `run_sweep.py` | the sweep driver: one CSV row per (method, λ, inner hyperparameters) |
| `controllers.py` | ConicFlow (receding horizon), HardFlow, and FMRG on the 2D flow map |
| `sampling_utils.py`, `visualization_utils.py` | data sampler and per-run diagnostic plots |
| `model/two_moons_lsd_15k.npz` | the pretrained flow map (EMA weights at 15k steps), 0.8 MB |
| `paper_data/` | the sweep CSVs behind the paper's figure |
| `make_fig1.py` | draws the three Figure 1 panels from a set of sweep CSVs |
| `reproduce_fig1.sh` | re-runs every sweep from scratch, then calls `make_fig1.py` |
| `fig1_pareto.ipynb` | runs `make_fig1.py` and displays the panels |
| `convert_checkpoint.py` | how the `.npz` was made from the flow-maps (JAX/Flax) checkpoint |

ConicFlow is called `mpc` inside `run_sweep.py` (the `--methods mpc` choice,
the `mpc_*` flags and the CSV `method` column).

## Regenerate Figure 1 from the shipped data (seconds)

```bash
cd lowdim
python make_fig1.py            # -> figures/lowdim_lowlambda.png, lowdim_highlambda.png, rewards_lowdim.png
```

The two reward-vs-energy panels come out pixel-identical to the paper's. The
reward-map panel differs only in where its randomly drawn data points and
reference points fall; the paper's render was not seeded, this one is.

## Re-run the sweeps from scratch (about 3 h on CPU)

```bash
cd lowdim
DRY_RUN=1 bash reproduce_fig1.sh   # optional: list all 1,210 configurations
bash reproduce_fig1.sh             # writes results_fig1/, then figures/
```

`reproduce_fig1.sh` regenerates exactly the configurations in `paper_data/`,
part by part. All runs share one batch of 512 initial particles drawn with
seed 42 on the run device. The paper's runs were on a GPU, so a CPU run starts
from different particles and matches the paper statistically, not digit for
digit. In a spot check of 14 configurations, mean terminal reward agreed to
within about 2% on the upper-moon problem and about 6% on the off-manifold
problem, whose rewards are much smaller. Use `DEVICE=cuda` to run on a GPU.

## What the figure shows

- **Energy axis.** E = Σ_j ‖u_j‖² Δt, the quantity the controllers penalise.
  `run_sweep.py` stores the displacement energy Σ_j ‖Δx_j‖² = E/N in the
  `ctrl_cost_raw` column, and `make_fig1.py` multiplies it by N when loading.
- **One point per λ.** Each plotted point is the run at that λ minimising
  J = E/(2λ) − r̄ over the inner hyperparameter grid, with E as above.
- **Upper-moon panel.** λ ∈ {0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5, 10}. Selection
  is over one inner grid at every λ: ConicFlow at 50 inner iterations,
  HardFlow at 10 outer × 10 inner. This is because λ = 5 and 10 were swept with
  a smaller grid than λ ≤ 3.
- **Off-manifold panel.** Selection is over every configuration run. λ = 70
  was swept but is not plotted.
- **Network.** The checkpoint's MLP takes (s, t, x/σ_d) ∈ ℝ⁴. It has four
  GELU-activated layers of width 256 (`Dense_0`–`Dense_3`) and a linear output
  layer to ℝ².

## Dependencies

numpy, scipy, pandas, matplotlib, scikit-learn and torch; all are in the
repository's environment. jax/flax are only needed to convert a newly trained
flow-maps checkpoint with `convert_checkpoint.py`. The shipped `.npz` loads
without them and gives bit-identical weights. To retrain the flow map, use
`flow-maps/py` with `configs/two_moons.py` (slot 0) for 15k steps. Data
sampling there is unseeded, so a retrained model will not match the shipped
one exactly.

σ_d, the data scale the network divides its input by, is recomputed at every
model load from 500k unseeded samples. It therefore varies at about the 1e-4
level from run to run.
