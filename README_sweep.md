# run_sweep.py

Controlled vs uncontrolled FLUX image generation sweep. For each configuration it runs an uncontrolled FLUX denoising baseline and (optionally) a controlled run using MPC or FMRG-J, then saves images, reward trajectory plots, control input plots, and a CSV summary.

---

## Quick start

```bash
# Single run, FMRG, default CLIP reward
python run_sweep.py \
  --sweep_prompt "a sunset over the ocean" \
  --sweep_control_type fmrg \
  --sweep_N 4 --sweep_H 2 --sweep_lam 50.0

# Lambda sweep, both methods, HPSv2 reward
python run_sweep.py \
  --sweep_prompt "a sunset" \
  --sweep_control_type mpc fmrg \
  --sweep_N 10 --sweep_H 4 \
  --sweep_lam 100.0 500.0 1000.0 5000.0 \
  --reward hpsv2

# Use the hardcoded SWEEP_CONFIGS list in the file
python run_sweep.py

# Run only entries 0 and 2 from SWEEP_CONFIGS
python run_sweep.py --configs 0,2

# MPC with CLIP constraints, sweep thresholds
python run_sweep.py \
  --sweep_prompt "a portrait of a man" \
  --sweep_control_type mpc \
  --sweep_N 10 --sweep_H 3 \
  --constraint_prompts "a smiling face" "with glasses" \
  --sweep_constraint_thresholds 0.3 0.4 0.5 \
  --lr_dual 0.5 --mu_init 0.0 \
  --reward clip
```

---

## Outputs

All outputs land in `--out_dir` (default: `results/`). Each run produces files tagged with its config:

```
{idx}__{control_type}__N{N}__H{H}__lam{lam}__{sample_mode}__{prompt}__images.png
{idx}__{control_type}__N{N}__H{H}__lam{lam}__{sample_mode}__{prompt}__reward.png
{idx}__{control_type}__N{N}__H{H}__lam{lam}__{sample_mode}__{prompt}__constraints.png  (MPC + constraints only)
{idx}__{control_type}__N{N}__H{H}__lam{lam}__{sample_mode}__{prompt}__thr{...}__...     (when thresholds are swept)
{idx}__{control_type}__N{N}__H{H}__lam{lam}__{sample_mode}__{prompt}__img_unc.png
{idx}__{control_type}__N{N}__H{H}__lam{lam}__{sample_mode}__{prompt}__img_ctrl.png
results/summary.csv
```

| File | Contents |
|------|----------|
| `__images.png` | Side-by-side uncontrolled vs controlled image |
| `__reward.png` | Reward trajectory over denoising steps + control input bar chart |
| `__constraints.png` | Constraint values `h_i(z_t)` and Lagrange multipliers `μ_i` over time (MPC + constraints only) |
| `__img_unc.png` | Uncontrolled image (standalone) |
| `__img_ctrl.png` | Controlled image (standalone) |
| `summary.csv` | One row per run: all hyperparams, `r_unc`, `r_ctrl`, `delta_r`, `ctrl_energy`, wall times |

### CSV columns

| Column | Description |
|--------|-------------|
| `prompt` | Text prompt |
| `N` | Total denoising steps |
| `H` | MPC planning horizon |
| `control_type` | `mpc`, `fmrg`, or `none` |
| `lam` | Control / guidance strength λ |
| `n_inner` | Inner iterations per outer step |
| `sample_mode` | Flow map variant |
| `warmstart` | MPC warm-start enabled |
| `drift_detach` | MPC drift-detach approximation enabled |
| `constraint_on` | Where MPC evaluates reward/constraints: `horizon` or `endpoint` |
| `n_constraints` | Number of CLIP inequality constraints |
| `constraint_thresholds` | Comma-separated CLIP thresholds (one per constraint prompt) |
| `lr_dual` | Dual-ascent step size for Lagrange multipliers |
| `mu_init` | Initial Lagrange multiplier value |
| `r_unc` | Final reward of uncontrolled image |
| `r_ctrl` | Final reward of controlled image |
| `delta_r` | `r_ctrl − r_unc` |
| `ctrl_energy` | Σ_t ‖u_t‖² Δt (integrated control effort, dynamics-consistent) |
| `time_unc_s` | Wall time for uncontrolled rollout (seconds) |
| `time_ctrl_s` | Wall time for controlled run (seconds) |

---

## All CLI arguments

### Infrastructure

| Argument | Default | Description |
|----------|---------|-------------|
| `--out_dir` | `results` | Directory to write all outputs |
| `--seed` | `42` | Random seed for initial noise |
| `--hf_token` | `None` | HuggingFace token (or set `HF_TOKEN` env var) |
| `--model_id` | `black-forest-labs/FLUX.1-dev` | HuggingFace model ID for FLUX |
| `--lora_path` | `~/flow_matching/fmrg/checkpoints/flux-flowmap-lora` | Path to FlowMap LoRA weights |
| `--quiet` | `False` | Suppress per-step output |
| `--configs` | `None` | Comma-separated indices into `SWEEP_CONFIGS` (ignored if any `--sweep_*` flag is given) |

### Memory / speed

| Argument | Default | Description |
|----------|---------|-------------|
| `--quantize` | `int8` | Transformer weight quantization. `none` = bfloat16 (~12 GB, fastest); `int8` (~6 GB, GPU-side quantize after load); `int4` (~3 GB, GPU-side); `nf4` (~3 GB, CPU layer swap then GPU pack). Fixed at load time. |
| `--fast` | `false` | `true` disables VAE tiling/slicing (always safe) and transformer gradient checkpointing (only when safe — kept on for FMRG and MPC with `drift_detach=false`). Also skips per-step `gc/empty_cache`. |

> **Note:** With `drift_detach=true` (the MPC default), the transformer is never backpropped through in the inner solve, so `--fast true` only speeds up VAE decode. The largest `--fast` benefit is for FMRG or MPC with `drift_detach=false`.

### Reward model

| Argument | Default | Description |
|----------|---------|-------------|
| `--reward` | `clip` | Reward model used for both guidance and evaluation. Choices: `clip`, `hpsv2`, `pickscore`, `imagereward`, `none`. Use `none` when steering is done entirely via constraints. |

| Model | Basis | Extra package |
|-------|-------|---------------|
| `clip` | CLIP ViT-H-14 cosine similarity | — (transformers) |
| `hpsv2` | HPSv2 aesthetic/human-preference | `hpsv2` |
| `pickscore` | PickScore human-preference | — (transformers) |
| `imagereward` | ImageReward (BLIP-based) | `ImageReward` |

### Multi-GPU device placement

| Argument | Default | Description |
|----------|---------|-------------|
| `--flux_device` | `cuda:0` | GPU for FLUX transformer + VAE |
| `--reward_device` | `auto` | Device for the reward model. `auto` picks a second GPU if available, otherwise CPU. Explicit values: `cpu`, `cuda:0`, `cuda:1`, etc. |

With two GPUs, FLUX runs on `cuda:0` and the reward model on `cuda:1`, eliminating reward-model VRAM pressure on the generation GPU.

### CLIP constraints (MPC only)

Inequality constraints of the form `h_i(z) = threshold_i − clip_score(z, prompt_i) ≤ 0`, i.e. each constraint enforces **clip_score ≥ threshold** (a lower bound on similarity). Constraints are enforced softly via a Lagrangian penalty; dual ascent adapts the multipliers when `lr_dual > 0`.

| Argument | Default | Description |
|----------|---------|-------------|
| `--constraint_prompts` | `[]` | One or more CLIP constraint prompts (global across all runs) |
| `--constraint_thresholds` | `[]` | Per-prompt thresholds paired with `--constraint_prompts`. Missing values default to `0.0`. Used when `--sweep_constraint_thresholds` is not set. |
| `--lr_dual` | `0.0` | Dual-ascent step size for Lagrange multipliers. `0` = frozen at `mu_init` (fixed penalty weight). |
| `--mu_init` | `0.0` | Initial value for all Lagrange multipliers |

Constraints require a CLIP model (loaded automatically when constraints are set, even if `--reward none`).

---

## Sweep hyperparameters

Each `--sweep_*` argument accepts **one or more values**. The script takes the Cartesian product of all provided lists. Any axis not specified keeps its default from `RunConfig`.

| Argument | Alias | Default | Description |
|----------|-------|---------|-------------|
| `--sweep_prompt` | — | `"a cat wearing sunglasses on a beach"` | Text prompt(s) |
| `--sweep_N` | — | `4` | Total number of denoising steps (FLUX time 1 → 0) |
| `--sweep_H` | — | `2` | MPC planning horizon in steps (must be ≤ N; ignored for FMRG) |
| `--sweep_control_type` | — | `mpc` | Control algorithm: `mpc`, `fmrg`, or `none` (uncontrolled baseline only) |
| `--sweep_lam` | — | `10.0` | Control strength λ. Larger = more aggressive guidance, higher control energy. |
| `--sweep_n_inner` | — | `5` | Inner iterations per outer step. For MPC: Adam steps per receding-horizon solve. For FMRG: Jacobian guidance steps per denoising step. |
| `--sweep_lr_inner` | `--lr_inner` | `0.01` | Inner Adam learning rate (MPC only) |
| `--sweep_sample_mode` | — | `flow_map2` | Flow map variant (see below) |
| `--sweep_guidance_scale` | — | `3.5` | CFG guidance scale passed to the FLUX transformer |
| `--sweep_height` | — | `256` | Output image height in pixels |
| `--sweep_width` | — | `256` | Output image width in pixels |
| `--sweep_warmstart` | — | `true` | MPC only. Reuse the previous window's optimized trajectory as the warm-start for the next step. Usually converges faster. |
| `--sweep_drift_detach` | — | `true` | MPC only. Treat the flow drift as locally constant inside the inner solve (no transformer backward). Much faster; gradient cosine similarity ~0.999 in practice. Set `false` for exact gradients (requires more VRAM, keep `--fast false`). |
| `--sweep_constraint_on` | — | `horizon` | MPC only. Where to evaluate reward and constraints: `horizon` (at `x_H`) or `endpoint` (flow-map prediction to final image). |
| `--sweep_constraint_thresholds` | — | — | Sweep CLIP constraint thresholds. Each spec is one float broadcast to all `--constraint_prompts` (e.g. `0.3`) or comma-separated floats, one per prompt (e.g. `0.3,0.4`). Requires `--constraint_prompts`. |

Boolean sweep flags accept `true`/`false` strings, e.g. `--sweep_drift_detach true false`.

### Sweeping constraint thresholds

`--constraint_prompts` is global (same prompts for every run). Thresholds can be swept with `--sweep_constraint_thresholds`, which participates in the usual Cartesian product with other `--sweep_*` axes.

**Broadcast one threshold to all constraints** — a single float is applied to every prompt:

```bash
python run_sweep.py \
  --sweep_prompt "a portrait of a man" \
  --sweep_control_type mpc \
  --constraint_prompts "a smiling face" "with glasses" \
  --sweep_constraint_thresholds 0.3 0.4 0.5 \
  --lr_dual 0.5 --mu_init 0.0
```

This produces 3 runs with thresholds `[0.3, 0.3]`, `[0.4, 0.4]`, and `[0.5, 0.5]`.

**Different threshold per constraint** — comma-separated values, count must match `--constraint_prompts`:

```bash
python run_sweep.py \
  --sweep_prompt "a portrait of a man" \
  --constraint_prompts "a smiling face" "with glasses" \
  --sweep_constraint_thresholds 0.28,0.25 0.32,0.30 0.36,0.35 \
  --sweep_lam 5000.0 10000.0
```

This produces 3 × 2 = 6 runs.

**Combined with other axes** — e.g. `--sweep_N 8 10 --sweep_constraint_thresholds 0.3 0.4` → 4 runs.

When thresholds are swept, output filenames include a `__thr{...}` suffix so runs do not overwrite each other.

### Sample modes

| Mode | `t_next` passed to transformer | Description |
|------|-------------------------------|-------------|
| `flow_map2` | `t_cur − dt` (next step) | Canonical FlowMap, one-step ahead conditioning |
| `flow_map1` | `0` | Single-step direct prediction to data |
| `flow_matching` | `t_cur` | Standard probability-flow ODE |

---

## Control algorithms

### MPC — Receding-Horizon Path-Space Control

At each outer step `j`, solves a finite-horizon optimization over `H` future trajectory states to minimize control energy subject to maximizing terminal reward. Only the first step of the solution is applied, then the window shifts forward.

Key parameters: `N`, `H`, `lam`, `n_inner`, `lr_inner`, `warmstart`, `drift_detach`, `constraint_on`, `constraint_prompts`, `constraint_thresholds`, `lr_dual`, `mu_init`.

**Cost per inner tick** (with `drift_detach=true`, `H=2`): 1 transformer fwd (no_grad) + 1 VAE fwd + 1 reward backward. No transformer backward.

### FMRG-J — Flow-Map Reward Guidance (Jacobian)

At each denoising step, advances the latent by one Euler step of the FLUX velocity, then applies `n_inner` Jacobian guidance steps: predicts the endpoint via a one-step flow map, computes `u = J_X^T ∇r` via autograd, and updates the latent.

Key parameters: `N`, `lam`, `n_inner`.

**Cost per inner tick**: 1 transformer fwd + 1 transformer backward (through endpoint prediction) + 1 VAE fwd + 1 reward backward.

### Control energy definition

Both methods report control energy using the dynamics-consistent definition:

```
u_t = x_{t+1} − x_t − dt · b_t(x_t)
ctrl_energy = Σ_t ‖u_t‖² · dt
```

where `b_t` is the uncontrolled FLUX drift. This is computed post-hoc from the stored trajectory (N no-grad transformer calls) so the scale is directly comparable between MPC and FMRG.

---

## Hardcoded sweep configs

When no `--sweep_*` flags are given, the script runs `SWEEP_CONFIGS` defined at the top of `run_sweep.py`. Edit that list to add custom configurations. Use `--configs 0,2,4` to run a subset by index.

---

## Environment

```bash
# Required env vars
export FMRG_DIR=~/flow_matching/fmrg   # default, can omit
export HF_TOKEN=hf_...                  # or pass --hf_token

# Recommended: set to avoid xet download issues
export HF_HUB_DISABLE_XET=1
```

The script expects the `flowmaps` conda environment with `torch`, `diffusers`, `transformers`, `torchao`, and the FMRG repo on the path.
