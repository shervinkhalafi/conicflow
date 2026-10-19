# Local modifications to FMRG

This directory is a vendored copy of [FMRG](https://github.com/jrrhuang/fmrg)
(*How to Guide Your Flow: Few-Step Alignment via Flow Map Reward Guidance*),
licensed under the Apache License 2.0 (`LICENSE` in this directory). It
supplies the FLUX.1-dev FlowMap sampler and the reward-model wrappers that all
image experiments in this repository run on, and the FMRG baseline.

It was taken from upstream commit `be485ba` (2026-06-10). Two files differ from
upstream; everything else is unchanged.

**`fluxfm_sampler_reward.py`** — memory-related changes so that FLUX.1-dev plus
a reward model fit on one 24 GB GPU, and a sampling entry point for the
control loops:

- The pipeline is loaded on CPU first; only the text encoders are moved to the
  GPU to precompute prompt embeddings, and are freed before the transformer is
  moved.
- Weight-only quantization of the transformer (`int8` default, `int4`, `nf4`).
  For `int8`/`int4` the transformer is moved to the GPU one block at a time
  and quantized as each block lands (`_stream_move_and_quantize`), which cuts
  the load-time peak from 24.2 GB to about 13 GB. The result is bit-identical
  to moving the whole model and then quantizing.
- `sample_simple` and packed-latent decode helpers used by the control loops.

**`install.sh`** — installs FMRG's pip dependencies into the existing
`flowmaps` conda environment (reusing its PyTorch/CUDA stack) instead of
creating a separate `fmrg_env`, and repairs the `peft`/`protobuf` pins that
the ImageReward/HPSv2 install downgrades.
