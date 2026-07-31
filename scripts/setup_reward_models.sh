#!/usr/bin/env bash
# Install optional reward backends for run_sweep.py --reward hpsv2|imagereward.
# Run after: conda env create -f environment.yml && conda activate flowmaps

set -euo pipefail

echo "Installing reward model packages (requires --no-build-isolation)..."

pip install 'setuptools<70'
pip install --no-build-isolation git+https://github.com/openai/CLIP.git
pip install --no-build-isolation image-reward hpsv2

# Repair pins pulled in by image-reward / hpsv2
pip install 'peft>=0.15.0,<0.18' 'protobuf>=4.21.6,<5'

# hpsv2 wheel may omit the OpenCLIP BPE vocab — copy from CLIP install
HPSV2_OC="$(python -c "import hpsv2, os; print(os.path.join(os.path.dirname(hpsv2.__file__), 'src', 'open_clip'))")"
CLIP_BPE="$(python -c "import clip, os; print(os.path.join(os.path.dirname(clip.__file__), 'bpe_simple_vocab_16e6.txt.gz'))")"
if [[ -f "$CLIP_BPE" && -d "$HPSV2_OC" && ! -f "$HPSV2_OC/bpe_simple_vocab_16e6.txt.gz" ]]; then
  cp "$CLIP_BPE" "$HPSV2_OC/"
  echo "Copied OpenCLIP BPE vocab into hpsv2."
fi

python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
for pkg in ("diffusers", "transformers", "peft"):
    m = __import__(pkg)
    print(f"  {pkg} {getattr(m, '__version__', '?')}")
try:
    import hpsv2
    print("  hpsv2 OK")
except ImportError:
    print("  hpsv2 not installed")
try:
    import ImageReward
    print("  ImageReward OK")
except ImportError:
    print("  ImageReward not installed")
PY

echo "Done."
