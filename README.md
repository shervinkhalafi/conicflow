# opt_ctrl_constrained_fm

Optimization-based control for FLUX flow-map image generation: receding-horizon MPC and FMRG-J guidance in latent space, with reward-driven steering (CLIP, HPSv2, PickScore, ImageReward).

## Repository layout

| Path | Description |
|------|-------------|
| `image_mpc.py` | Core MPC + FMRG engine, FLUX wrappers, trajectory utilities |
| `run_sweep.py` | Batch sweep script (controlled vs uncontrolled, CSV + plots) |
| `README_sweep.md` | Full CLI reference for `run_sweep.py` |
| `image_tests.ipynb` | Interactive MPC / FMRG demos on FLUX latents |
| `evalution.ipynb` | Analyze `results/summary.csv` |
| `flow_map_tests.ipynb` | 2D flow-map MPC / FMRG experiments |
| `mpc_cell.py` | Standalone MPC cell utilities |
| `fmrg/` | FLUX FlowMap sampler + reward models ([upstream](https://github.com/jrrhuang/fmrg), locally patched) |
| `flow-maps/` | Flow-map training code ([upstream](https://github.com/nmboffi/flow-maps), locally patched) |

## Setup

```bash
conda activate flowmaps   # or your PyTorch + diffusers env

export HF_TOKEN=hf_...    # required for FLUX.1-dev
export HF_HUB_DISABLE_XET=1   # recommended

# FMRG LoRA checkpoint (~345 MB) — not in repo
mkdir -p fmrg/checkpoints
# download flux-flowmap-lora into fmrg/checkpoints/flux-flowmap-lora
# (see fmrg/README or HuggingFace model card)
```

Install Python dependencies (minimum):

```bash
pip install torch diffusers transformers torchao matplotlib tqdm huggingface_hub
# optional reward models:
pip install hpsv2          # --reward hpsv2
pip install ImageReward    # --reward imagereward
```

## Quick start

```bash
python run_sweep.py \
  --sweep_prompt "a sunset over the ocean" \
  --sweep_control_type mpc fmrg \
  --sweep_N 4 --sweep_H 2 \
  --sweep_lam 50.0 500.0 \
  --reward clip
```

See [`README_sweep.md`](README_sweep.md) for all CLI options.

## Control methods

- **MPC** — receding-horizon path-space control with inner Adam optimization per step
- **FMRG-J** — Jacobian reward guidance via one-step flow-map endpoint prediction

Both steer a FLUX.1-dev + FlowMap LoRA sampler using a differentiable reward on decoded images.

## Credits

- FLUX FlowMap sampler and reward ensemble: [jrrhuang/fmrg](https://github.com/jrrhuang/fmrg)
- Flow-map training: [nmboffi/flow-maps](https://github.com/nmboffi/flow-maps)
- FLUX.1-dev: Black Forest Labs (gated — requires HF login)
