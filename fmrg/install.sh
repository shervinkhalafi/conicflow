#!/bin/bash
set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )"
ENV_FILE="$SCRIPT_DIR/environment.yml"
ENV_NAME="${CONDA_ENV_NAME:-flowmaps}"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "environment.yml not found at $ENV_FILE"
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "Conda env '$ENV_NAME' not found."
    echo "Create/activate your flowmaps env first, then re-run this script."
    exit 1
fi

echo "Installing FMRG dependencies into existing conda env: $ENV_NAME"
conda activate "$ENV_NAME"

PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Detected Python $PY_VER in $ENV_NAME"

# Build a filtered requirements file from environment.yml:
# - keep fmrg pip deps
# - skip torch/torchvision/triton (reuse the flowmaps CUDA stack)
# - skip numpy/scipy/matplotlib/pandas (keep flowmaps/JAX stack intact)
# - relax pins that require Python >= 3.10 when running on 3.9
REQ_FILE="$(mktemp)"
python - "$ENV_FILE" << 'PY' > "$REQ_FILE"
import sys

skip_prefixes = ("torch", "torchvision", "triton", "numpy", "scipy", "matplotlib", "pandas")
skip_exact = {"--extra-index-url https://download.pytorch.org/whl/cu121"}

# Exact pins from environment.yml -> py39-compatible substitutes
py39_overrides = {
    "accelerate==1.12.0": "accelerate==1.10.1",
    "peft>=0.11.1": "peft>=0.15.0,<0.18",
    "transformers>=4.41.2": "transformers>=4.41.2,<4.54",
    "datasets==4.4.1": "datasets>=2.19,<4",
    "huggingface-hub==0.34.4": "huggingface-hub>=0.23,<0.34",
    "torchmetrics==1.3.2": "torchmetrics>=1.3,<1.4",
}

def adjust_pkg(pkg: str) -> str:
    if sys.version_info >= (3, 10):
        return pkg
    if pkg in py39_overrides:
        print(f"# py39 override: {pkg} -> {py39_overrides[pkg]}", file=sys.stderr)
        return py39_overrides[pkg]
    return pkg

with open(sys.argv[1]) as f:
    in_pip = False
    for raw in f:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "- pip:":
            in_pip = True
            continue
        if in_pip:
            if line.startswith("- ") and not line.startswith("- --"):
                pkg = line[2:].strip()
                if "#" in pkg:
                    pkg = pkg.split("#", 1)[0].strip()
                if not pkg:
                    continue
                if pkg in skip_exact:
                    continue
                if any(pkg == p or pkg.startswith(p + "==") or pkg.startswith(p + ">=")
                       for p in skip_prefixes):
                    continue
                print(adjust_pkg(pkg))
PY

echo "Installing pip requirements into $ENV_NAME (excluding torch stack)..."
pip install -r "$REQ_FILE"
rm -f "$REQ_FILE"

# CLIP, image-reward, and hpsv2 need --no-build-isolation because their
# setup.py modules import pkg_resources, which the latest setuptools dropped.
pip install 'setuptools<70'
pip install --no-build-isolation git+https://github.com/openai/CLIP.git
pip install --no-build-isolation image-reward hpsv2

# image-reward/hpsv2 pull older pins (peft, protobuf) that break diffusers / tensorflow.
echo "Repairing fmrg pins after reward-backend install..."
pip install 'peft>=0.15.0,<0.18' 'protobuf>=4.21.6,<5'

# hpsv2's PyPI wheel omits its vendored open_clip BPE vocab file. Copy it from
# the openai-clip install.
HPSV2_OC="$(python -c "import hpsv2,os;print(os.path.join(os.path.dirname(hpsv2.__file__),'src','open_clip'))")"
CLIP_BPE="$(python -c "import clip,os;print(os.path.join(os.path.dirname(clip.__file__),'bpe_simple_vocab_16e6.txt.gz'))")"
if [ -f "$CLIP_BPE" ] && [ -d "$HPSV2_OC" ] && [ ! -f "$HPSV2_OC/bpe_simple_vocab_16e6.txt.gz" ]; then
    cp "$CLIP_BPE" "$HPSV2_OC/"
fi

python -c "import torch, diffusers, transformers, peft; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} | diffusers {diffusers.__version__} | transformers {transformers.__version__} | peft {peft.__version__}')"
echo "Done. Use: conda activate $ENV_NAME"
echo "Note: image-reward pins timm==0.6.13; fmrg FLUX code does not import timm directly."
