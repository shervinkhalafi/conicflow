#!/usr/bin/env bash
# Post-`uv sync` fixups that a lockfile cannot express.
#
#   uv sync && ./scripts/uv_postinstall.sh
#
# This is the uv counterpart of scripts/setup_reward_models.sh, and it is much
# smaller: uv's resolver + the override-dependencies in pyproject.toml already
# handle everything that script did with `--no-build-isolation` and its
# "repair pins pulled in by image-reward / hpsv2" step. What is left is one
# genuine packaging bug that no resolver can fix.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-uv run python}"

# The hpsv2 1.2.0 wheel ships its vendored open_clip WITHOUT the BPE vocab file
# it loads at tokenizer-construction time, so `--reward hpsv2` dies on a missing
# file. openai/CLIP ships the identical vocab; copy it across.
HPSV2_OC="$($PY -c "import hpsv2, os; print(os.path.join(os.path.dirname(hpsv2.__file__), 'src', 'open_clip'))" 2>/dev/null)"
CLIP_BPE="$($PY -c "import clip, os; print(os.path.join(os.path.dirname(clip.__file__), 'bpe_simple_vocab_16e6.txt.gz'))" 2>/dev/null)"
if [[ -f "$CLIP_BPE" && -d "$HPSV2_OC" && ! -f "$HPSV2_OC/bpe_simple_vocab_16e6.txt.gz" ]]; then
    cp "$CLIP_BPE" "$HPSV2_OC/"
    echo "Copied OpenCLIP BPE vocab into hpsv2."
else
    echo "hpsv2 BPE vocab already present (or hpsv2/clip not installed) -- nothing to do."
fi

$PY - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
for pkg in ("diffusers", "transformers", "peft", "torchao", "timm", "datasets"):
    m = __import__(pkg)
    print(f"  {pkg:13} {getattr(m, '__version__', '?')}")
for pkg in ("clip", "ImageReward", "hpsv2", "lpips"):
    try:
        __import__(pkg)
        print(f"  {pkg:13} OK")
    except Exception as e:
        print(f"  {pkg:13} FAILED: {type(e).__name__}: {e}")
PY
echo "Done."
