#!/usr/bin/env bash
# One seed's floor probe: both base prompts x all four edit prompts,
# uncontrolled only, in a single process against a single model load.
set -u
R="${FMOCC_ROOT:-$HOME/fmocc}"
cd "$R/fmocc_core" || exit 3
export PATH="$R/.venv/bin:$PATH"
export FLOW_MATCHING_DIR="$R" FMRG_DIR="$R/fmrg"
export CUDA_VISIBLE_DEVICES=0

SEED="$1"
LOG=results/floor2/log_s${SEED}.txt
echo "[$(date +%H:%M)] floor seed $SEED"

# control_type=none -> uncontrolled rollout only. r_ctrl/ctrl_energy come back
# nan by design (run_sweep.py documents this); r_unc is what we are after.
# lam/N/H are inert here but still required by the config builder.
python -u run_sweep.py \
  --sweep_prompt "a portrait of a young man with straight hair" \
                 "a portrait of a serious man with dark hair" \
  --sweep_edit_prompt "a face with curly hair" "an old face" \
                      "a smiling face" "a face with white hair" \
  --reward clip --sweep_control_type none \
  --sweep_N 8 --sweep_H 2 --sweep_lam 5000 \
  --flux_device cuda:0 --reward_device cuda:0 \
  --seed "$SEED" --quiet --resume \
  --out_dir "results/floor2/s${SEED}" >> "$LOG" 2>&1
rc=$?
echo "[$(date +%H:%M)] done  seed $SEED rc=$rc"
exit $rc
