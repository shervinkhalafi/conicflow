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

Environment: conda env `flowmaps`, or the uv project at the repo root — `uv
sync && ./scripts/uv_postinstall.sh`, then `uv run --project .. python
run_sweep.py ...` from here. uv resolves the same stack through a lockfile, so
two machines can be made identical; `environment.yml` is annotated with what
each conda pin is for. The uv env deliberately omits the jax/flax/tensorflow
flow-map *training* stack, which cannot share a resolved environment with torch
2.5.1's cuDNN pin (that combination is broken in the conda env too — see the
comment block in `pyproject.toml`); it lives in
`requirements-flowmaps-training.txt` and gets its own venv.

FLUX (`black-forest-labs/FLUX.1-dev`) and the reward models are pulled from
HuggingFace on first run and cached at `HF_HOME`.

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
| **MPC** | `mpc` | `N`, `H`, `n_inner`, `lr_inner`, `lam`, `endpoint_surrogate` | inner loop backprops through the endpoint rollout. Cost `NFE = n_inner·N(N−1)/2` — **quadratic in N**, and `H` is nearly free. The `+ H` this formula used to carry was a wasted `dt=0` rollout, removed 2026-09-16. `endpoint_surrogate=jump` makes the whole term **linear**. |
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

**As of 2026-09-16 this is derived, not supplied.** Omit `--sweep_hardflow_reg`
and each config computes its own `reg = N/(2·lam)`, so a HardFlow arm is
regularized at the `lam` on its own CSV row. Supplying the flag overrides the
derivation and is never recomputed — that is how `reg=0.0` (the pre-2026-09-09
no-penalty runs) stays reproducible.

Before that date `reg` defaulted to a fixed `5e-4`, which is `N/(2·lam)` at
`lam=10000` and `N=10` **and nowhere else**. Because `reg` is N-dependent, every
other `(N, lam)` point silently mismatched the MPC arm it was plotted against:
an `N=20` HardFlow arm ran half as regularized as its MPC counterpart. Runs
before 2026-09-16 that did not pass `--sweep_hardflow_reg` explicitly are only
matched to their `lam` if they happened to use `N=10, lam=10000`.

`run_preflight` still prints the implied equivalent `lam` on every HardFlow run.
It now also warns when an explicitly-supplied `reg` disagrees with the row's
`lam`, and predicts the `lr·reg ≥ 1` raise at config time instead of after FLUX
has loaded.

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

**Locked defaults.** `constraint_on="endpoint"` and `warm_start=True` are
asserted in `run_preflight` on every run path. To measure what a lock costs, set
`FLOWMAPS_UNLOCK=constraint_on` for that one invocation — the run prints a
`*** LOCK LIFTED ***` banner and is stamped in the `unlocked` CSV column. It is
an env var and not a sweep flag on purpose, so a lock can never come off as a
side effect of a grid definition.

**`drift_detach` is no longer locked, and now defaults to `True`** (2026-09-16).
It was locked at `False` on the grounds that `True`'s only benefit was speed and
that the speed gain was small. Measurement reversed both halves: at
`N=8, H=4, n_inner=6` with ImageReward it ran **33% faster** *and* produced a
**better** objective at every λ tested (J −1.096 vs −0.756 at λ=1, about 6
run-to-run standard deviations; also better at λ=10 and λ=100). A closed
decision stays closed on its reasoning, not its conclusion, so the axis is open
again rather than pinned to the opposite value. `FLOWMAPS_UNLOCK=drift_detach`
is still accepted and is now a no-op. **Rows from before 2026-09-16 used
`False`** — check the `drift_detach` column before comparing energy or objective
across that date.

**The reward in the MPC progress bar is off by default** (2026-09-16). It cost a
no-grad endpoint rollout plus a decode and a reward-model pass every outer step,
and another decode + reward per inner iteration — about 4% of a run, for a
display field. Pass `--pbar_rewards` to get it back. Note `show_progress=False`
alone never suppressed this: `tqdm(disable=True)` still returns a live object,
so the old `if pbar is not None` guards always fired.

---

## 5. Comparing methods after a sweep

Two scripts read `summary.csv` files and compare runs *across* methods. Both
take `--runs <dir> ...` (globs are fine), read only what is on disk, and never
touch a GPU.

| script | what it produces |
|---|---|
| `sweep_diagnostics.py` | **win rates** — per (λ, seed) combo, which method reached the lowest objective, plus MPC-beats-FMRG/HardFlow/both — and a **Pareto** reward-vs-energy scatter with per-method frontiers and the globally non-dominated points ringed. One flag per diagnostic (`--diag_winrates`, `--diag_pareto`, `--all`), derived from a registry so a new diagnostic needs no change in its callers. |
| `compare_methods.py` | figures of four images in a row — uncontrolled, MPC, HardFlow, FMRG — at a **matched operating point**: `--match reward` holds achieved reward as equal as the grid allows and lets energy differ, `--match energy` does the reverse, `both` writes both. Each panel carries its λ and config; energy and reward go underneath. `--num_rows N` removes the runs already used and repeats the match, so one figure can show several operating points. |

`compare_constrained.py` is the opposite shape: it *runs* the comparison rather
than reading one, so it needs a GPU. Given a prompt, edit prompt, λ and
constraint threshold it runs MPC and HardFlow in turn and writes an
uncontrolled | MPC | HardFlow figure with each panel's score, feasibility,
energy and wall clock. See **[COMPARE_CONSTRAINED.md](COMPARE_CONSTRAINED.md)**.

The same `--diag_*` flags are accepted by `run_sweep.py` (run after the sweep)
and by `auto_sweep.sh` via `DIAG=` (run once all three methods are done, since
each writes its own `out_dir` and no single `run_sweep.py` invocation ever sees
all three).

**Win rates need a shared λ grid** (`SHARED_GRID=1`): the objective is only
comparable between methods at equal λ. With per-method grids no combo contains
all three and the diagnostic honestly reports zero combos. The Pareto plot is
unaffected — energy-vs-reward does not depend on what λ means.

### Several configs per method

A sweep may run one method at more than one setting: `MPC_NI="12 24"` in
`auto_sweep.sh` (every per-method knob there takes a space-separated list), or
`--sweep_n_inner 12 24` directly. Each distinct tuple of a method's
hyperparameter columns is a **config**, labelled by only the fields that vary
within that method (`ni=24`), and both scripts keep configs apart rather than
pooling them:

- **Pareto** draws each config in its own shade of the method's hue, with its
  own frontier line, so configs stay separable without blurring the three
  methods into each other.
- **Win rates** represent a method at each combo by its lowest-J config;
  `winrates.csv` records which one in `cfg_<method>` and the printout names it.
  **Best-of is a max, so a method handed more configs gets more draws** — an
  advantage that belongs to the sweep design, not to the method. The printout
  warns whenever the candidate counts differ; a clean comparison gives every
  method the same number.
- **`compare_methods.py --configs best`** (the default) first picks, per method,
  the config that reached the lowest J most often across that method's own
  combos — ties on mean J; a method whose configs never share a combo falls back
  to lowest mean J and says so — then matches operating points using only those
  three. Each method is therefore shown at its strongest, not at whichever
  config happened to land near the others. `--configs all` pools every config
  into the candidate set instead.

Measured on the seed-50 sweeps by adding an MPC `n_inner=24` arm next to
`n_inner=12` (2× the compute, 607 s vs 309 s per λ): `ni=24` had the lower
objective at 5/5 λ under both rewards. It moved **CLIP** win rates from MPC 40%
to 80% and shifted MPC's whole frontier left — at λ=6.1, more reward (0.161 vs
0.152) for 42% less energy (0.144 vs 0.251); nearer the top the gap narrows to
about 1.5×. **ImageReward** stayed at 40%: that reward is already near its
`r≈1.98` ceiling, so the doubled inner budget buys energy, not wins.

---

## 6. Key flags

| flag | notes |
|---|---|
| `--reward` | `clip`, `hpsv2`, `pickscore`, `imagereward`, `style`, `lpips`, `none`. `none` = steering comes entirely from constraints. |
| `--sweep_*` | Cartesian product over all sweep axes, in a fixed order. |
| `--constraint_types` / `--constraint_args` / `--constraint_thresholds` / `--constraint_directions` | **`mpc` and `hardflow` only** — `fmrg`/`boffi_fmrg` raise. So does `lr_dual=0` with `mu_init=0` (a silent no-op). |
| `--sweep_endpoint_surrogate` | `rollout` (default) or `jump`. **mpc only.** `jump` replaces the inner objective's (N−j−H)-step endpoint rollout with ONE flow-map call at `t_next=0`, making MPC's cost linear in N instead of quadratic. Only the surrogate the optimiser differentiates changes — the applied trajectory, and everything scored from it, are unaffected. Measured at N=8/H=4: −33% wall clock, objective better at λ=1 and inside the noise floor at λ=10/100. Opt-in because it has not been re-checked at the suite's N=10. |
| `--pbar_rewards` | Off by default. Computes the reward shown in the MPC progress bar; costs ~4% of a run. |
| `--hardflow_lr_alpha` | Off by default. Lifts HardFlow's `lr·reg ≥ 1` admissible floor: any run whose `lr_inner·reg` would reach 1 uses `lr = α/reg` instead (0 < α < 1; 0.5 is dead-beat at the final step), so the inner loop's contraction factor is uniform in λ. Runs above the floor are unchanged; the lr actually used is written to the `hardflow_lr_eff` column. Needed to run HardFlow on a shared λ grid with MPC/FMRG in the regularised regime. |
| `lambda_autorange.py` / `auto_sweep.sh` | Automatic λ range finding per (reward, prompt, method), and a wrapper that chains it into the sweep. `SHARED_GRID=1` runs all methods on one grid (at a shared λ all three solve the same first-order problem — see `lambda_autorange.shared_grid`); `HF_ALPHA=0.5` passes the floor-lifting alpha through; `DIAG="--diag_winrates --diag_pareto"` runs the cross-method diagnostics at the end. Per-method knobs (`MPC_NI`, `MPC_H`, `HF_NI`, `HF_LR`, `FMRG_NOPT`) take lists. See `docs/lambda_autorange_report.html`. |
| `--diag_winrates` / `--diag_pareto` / `--all` | Post-sweep cross-method diagnostics, on `run_sweep.py` and standalone on `sweep_diagnostics.py`. See §5. |
| `--quantize` | `int8` (default) / `int4` / `nf4` / `none`. See the nf4 warning above. **`int4` cannot be used for MPC or FMRG at all**: `aten::_weight_int4pack_mm` has no backward kernel, so any gradient-based control raises. |
| `--reward_dtype` | `fp32` default, matching every result on disk. `fp16` halves reward-model memory but changes their values. |
| `--resume` | Skips configs already in `<out_dir>/summary.csv` with `run_ok=True`, retries failures, appends rather than truncating. |
| `--continue_on_oom` | Off by default: an OOM aborts rather than marching the rest of a sweep through a contended card. |

## 7. Output

`<out_dir>/summary.csv` — one self-describing row per run: config, `r_unc`,
`r_ctrl`, `delta_r`, `ctrl_energy`, `objective`, `time_unc_s`, `time_ctrl_s`,
`run_ok`, `error_*`, peak memory, and per-constraint `thr_i`/`h_i`/`mu_i`/`viol_i`.
Plus per-run `__img_ctrl.png`, `__img_unc.png`, `__images.png`, `__reward.png`,
and `__mu_trace.csv` when constraints are active.

`objective` is `ctrl_energy/(2·lam) − r_ctrl`, **minimised** — more negative is
better.

Failed runs are never deleted or skipped: they get `run_ok=False`, a
`FAILED_`-prefixed artifact dir, and the exception text in `error_msg`.

The §5 scripts write next to the run dirs rather than into them:
`<out>/diagnostics/winrates.csv` (one row per (λ, seed) combo: winner, the
head-to-heads, the config that represented each method, and every method's
`J_`/`r_ctrl_`/`energy_`), `<out>/diagnostics/pareto__*.png`, and
`<out>/figs/compare_{reward,energy}__*.png`.

## 8. Conventions

- **Loud fail only** — never silently clamp, fall back, or substitute a default.
- **Fresh process per config** — no sweeping several configs in one process
  (text encoders are freed after load, so a new prompt cannot be re-encoded).
- **Comment every change to a core file** at the change site, saying why. Agreed
  2026-09-15 so the history of this loop stays legible.
- Result directories are immutable records — new runs get new timestamped dirs.
