# fmocc

Reward-steered FLUX sampling: run and compare **MPC** (receding-horizon flow
control), **HardFlow**, and **FMRG** on image latents, with any of several
reward models, unconstrained or under dual-ascent constraints.

This is the core extracted from a larger research scratch directory. It is the
code needed to *run the comparisons* — the plotting scripts, the accumulated
results and the working notes were deliberately left out.

---

## 1. It does not run standalone — read this first

`run_sweep.py` imports, read-only, from a separate checkout:

| symbol | module | where |
|---|---|---|
| `make_flux_wrappers`, `fmrg_j_image`, `trajectory_rewards`, `tune_sampler_speed` | `image_mpc.py` | `$FLOW_MATCHING_DIR` |
| `FluxFlowMapSampler`, `RewardEnsemble`, `CLIPReward`, `HPSReward`, `ImageRewardReward`, `PickScoreReward` | `fluxfm_sampler_reward.py` | `$FMRG_DIR` |

```bash
export FLOW_MATCHING_DIR=/path/to/flow_matching     # default: /home/shervink/flow_matching
export FMRG_DIR=$FLOW_MATCHING_DIR/fmrg             # default
```

Two things to know about that boundary:

- **The control loops are *not* imported from there.** `receding_horizon_flow_control`
  and `hardflow_image` come from `image_mpc_edited_sung.py` **in this repo** — a
  derived copy of the MPC loop plus the HardFlow port, which does not exist
  upstream. Anything touching the hot loop belongs here, not in `image_mpc.py`.
- `fluxfm_sampler_reward.py` in the upstream checkout carries one change made
  from this side on 2026-09-05 (by agreement): `_load_model` streams the
  int8/int4 GPU move block-by-block, removing a 24.21 GB load-time transient.
  Verified bit-identical across all 4268 quantized tensors. If you are working
  from a fresh clone of the upstream, you need that change or FLUX will spike
  above 24 GB while loading.

Environment: conda env `flowmaps`. FLUX (`black-forest-labs/FLUX.1-dev`) and the
reward models are pulled from HuggingFace on first run and cached at `HF_HOME`.

---

## 2. Quickstart

One unconstrained run, MPC, ImageReward:

```bash
python run_sweep.py \
    --sweep_prompt "a portrait of a man" --sweep_edit_prompt "a smiling face" \
    --sweep_control_type mpc --sweep_N 10 --sweep_H 4 \
    --sweep_n_inner 12 --sweep_lr_inner 1e-2 --sweep_lam 10000.0 \
    --reward imagereward \
    --flux_device cuda:0 --reward_device cuda:0 \
    --seed 42 --quiet --out_dir results/my_run
```

Constrained (MPC or HardFlow only):

```bash
python run_sweep.py \
    --sweep_prompt "a portrait of a man" --sweep_edit_prompt "a smiling face" \
    --sweep_control_type mpc --sweep_N 10 --sweep_H 4 --sweep_lam 10000.0 \
    --reward none \
    --constraint_types clip imagereward \
    --constraint_args "a smiling face" "a smiling face" \
    --constraint_thresholds 0.43 1.5 \
    --constraint_directions ge ge \
    --lr_dual 0.2 --mu_init 0.0 \
    --flux_device cuda:0 --reward_device cuda:0 \
    --seed 42 --quiet --out_dir results/my_constrained_run
```

`python run_sweep.py --help` has the full epilog with more worked examples.
For sweeps, see **[SUITE.md](SUITE.md)**. For what the numbers have actually
been set to lately, see **[HYPERPARAMETERS.md](HYPERPARAMETERS.md)**.

---

## 3. The three methods and their knobs

| | control type | knobs | notes |
|---|---|---|---|
| **MPC** | `mpc` | `N`, `H`, `n_inner`, `lr_inner`, `lam` | inner loop backprops through the endpoint rollout. Cost `NFE = n_inner·(N(N−1)/2 + H)` — **quadratic in N**, and `H` is nearly free. |
| **HardFlow** | `hardflow` | `N`, `n_inner`, `lr_inner` (=lr), `hardflow_reg` | decision variable is the predicted terminal latent; the transformer is never differentiated. **Linear in N**, ~4.3 s/step unconstrained. |
| **FMRG** | `fmrg` | `N`, `n_inner` (=`n_opt`), `lam` | `z ← z + (dt/n_opt)·lam·∇r(endpoint)`. |

`boffi_fmrg` runs the upstream Boffi implementation instead of the port.

**`lam` does not mean the same thing in all three.** For MPC it is a Lagrange
weight the optimiser solves against (`J = energy/(2·lam) − r`). For HardFlow the
equivalent enters as `hardflow_reg` (below). For **FMRG it is an open-loop gain
on the reward gradient** — nothing in FMRG minimises energy. Bigger `lam` means
more control in all three, which is why they get plotted on one axis, but a
shared-`lam` comparison across methods is not apples-to-apples. The
energy-vs-reward frontier is the comparison that does not depend on what each
knob means.

---

## 4. Things that will bite you

**`hardflow_reg` ↔ `lam`.** Same physical quantity (control effort), but `lam`
divides and `hardflow_reg` multiplies, and the conversion carries an
N-dependence:

```
hardflow_reg = 1 / (2·lam·dt) = N / (2·lam)       # NOT 1/lam
```

A `reg` matched to a `lam` at one step count is *not* matched at another. Set it
explicitly per arm; `run_preflight` prints the implied equivalent `lam` on every
HardFlow run so a mismatch shows up in the log.

**HardFlow has a `lam` floor.** `hardflow_image` raises when `lr·reg ≥ 1`, i.e.
below `lam = N·lr/2` (12.5 at N=10, lr=2.5). Its inner loop is explicit
fixed-step ascent, so the deviation error evolves as `e ← e·(1 − 2·lr·reg·α²)`
and contracts only while `lr·reg·α² < 1`, with `α → 1` at the final step. This is
a *solver* limit, not a limit of the formulation — the penalised minimiser is
well defined at any `reg`. Note also that the authors' nominal
`regularization=0.0005` is `.mean()`-scaled to ~1e-9 and **inert**, so as
published HardFlow applies no effective effort penalty at all; the floor only
appears once you add one to match `lam`.

**`lam` is relative to the reward's scale.** Switching LPIPS → ImageReward
de-regularises by ~200× at a fixed `lam`. A `lam` grid tuned for one reward is
not tuned for another.

**ImageReward saturates at `r_ctrl ≈ 1.98–2.00`** for every prompt pair tried so
far. Near the ceiling the objective stops discriminating between methods even
when the images differ enormously — including visibly destroyed images still
scoring ~1.97. Sweep `lam` low enough to get off the ceiling.

**The uncontrolled sample depends on N.** Same seed and `z0`, different N, is a
different image — sometimes a different *subject* (measured: `r_unc` −1.581 at
N=8 vs +1.234 at N=100 for one seed). So `delta_r` is only comparable across
methods when they share capital N, and a figure with one shared "uncontrolled"
column is only honest under equal N.

**Never `--quantize nf4` for gradient-based control.** Correct forward pass,
~zero input gradient — the control silently does nothing. Default `int8` is
12.26 GB resident; `none` is 23.8 GB and does not fit on a 24 GB card.

**Pin one card per process** and always pass `--reward_device` equal to
`--flux_device`. On the dev box cross-GPU copies silently return ~zeros, so
`--reward_device auto` corrupts the reward model with no error. `gpu_lib.sh`'s
`run_on_gpu` enforces both.

**Locked defaults.** `drift_detach=False`, `constraint_on="endpoint"`,
`warm_start=True` are asserted in `run_preflight` on every run path. To measure
what a lock costs, set `FLOWMAPS_UNLOCK=drift_detach` for that one invocation —
the run prints a `*** LOCK LIFTED ***` banner and is stamped in the `unlocked`
CSV column. It is an env var and not a sweep flag on purpose, so a lock can
never come off as a side effect of a grid definition.

---

## 5. Key flags

| flag | notes |
|---|---|
| `--reward` | `clip`, `hpsv2`, `pickscore`, `imagereward`, `style`, `lpips`, `none`. `none` = steering comes entirely from constraints. |
| `--sweep_*` | Cartesian product over all sweep axes, in a fixed order. |
| `--constraint_types` / `--constraint_args` / `--constraint_thresholds` / `--constraint_directions` | **`mpc` and `hardflow` only** — `fmrg`/`boffi_fmrg` raise. So does `lr_dual=0` with `mu_init=0` (a silent no-op). |
| `--quantize` | `int8` (default) / `int4` / `nf4` / `none`. See the nf4 warning above. |
| `--reward_dtype` | `fp32` default, matching every result on disk. `fp16` halves reward-model memory but changes their values. |
| `--resume` | Skips configs already in `<out_dir>/summary.csv` with `run_ok=True`, retries failures, appends rather than truncating. |
| `--continue_on_oom` | Off by default: an OOM aborts rather than marching the rest of a sweep through a contended card. |

## 6. Output

`<out_dir>/summary.csv` — one self-describing row per run: config, `r_unc`,
`r_ctrl`, `delta_r`, `ctrl_energy`, `objective`, `time_unc_s`, `time_ctrl_s`,
`run_ok`, `error_*`, peak memory, and per-constraint `thr_i`/`h_i`/`mu_i`/`viol_i`.
Plus per-run `__img_ctrl.png`, `__img_unc.png`, `__images.png`, `__reward.png`,
and `__mu_trace.csv` when constraints are active.

`objective` is `ctrl_energy/(2·lam) − r_ctrl`, **minimised** — more negative is
better.

Failed runs are never deleted or skipped: they get `run_ok=False`, a
`FAILED_`-prefixed artifact dir, and the exception text in `error_msg`.

## 7. Conventions

- **Loud fail only** — never silently clamp, fall back, or substitute a default.
- **Fresh process per config** — no sweeping several configs in one process
  (text encoders are freed after load, so a new prompt cannot be re-encoded).
- **Comment every change to a core file** at the change site, saying why. Agreed
  2026-09-15 so the history of this loop stays legible.
- Result directories are immutable records — new runs get new timestamped dirs.
