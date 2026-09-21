#!/usr/bin/env bash
# One probe job.  Argument is a single pipe-delimited field:
#     <tag>|<base-key A|B>|<edit prompt>|<arm>|<seed>
# Pipe-delimited and passed as ONE xargs field on purpose: the edit prompts
# contain spaces, so whitespace-split dispatch (xargs -n5) would tear them
# apart and silently probe the wrong prompt.
set -u
R="${FMOCC_ROOT:-$HOME/fmocc}"
cd "$R/fmocc_core" || exit 3
export PATH="$R/.venv/bin:$PATH"
export FLOW_MATCHING_DIR="$R" FMRG_DIR="$R/fmrg"
export CUDA_VISIBLE_DEVICES=0

IFS='|' read -r TAG BKEY EDIT ARM SEED <<< "$1"
case "$BKEY" in
  A) BASE="a portrait of a young man with straight hair" ;;
  B) BASE="a portrait of a serious man with dark hair" ;;
  *) echo "bad base key $BKEY"; exit 3 ;;
esac
case "$ARM" in
  mpc)      EXTRA="--sweep_n_inner 24 --sweep_lr_inner 1e-2" ;;
  hardflow) EXTRA="--sweep_n_inner 40 --sweep_lr_inner 2.5 --hardflow_lr_alpha 0.5" ;;
  *) echo "bad arm $ARM"; exit 3 ;;
esac

OUT=results/probe2/${TAG}__${ARM}__s${SEED}
LOG=results/probe2/log_${TAG}_${ARM}_s${SEED}.txt
echo "[$(date +%H:%M)] probe $TAG $ARM s$SEED  '$BASE' -> '$EDIT'"

# lam is deliberately LARGE: we want the ceiling the arm reaches when control
# effort is cheap, not a Pareto point. This is a ceiling probe.
python -u run_sweep.py \
  --sweep_prompt "$BASE" --sweep_edit_prompt "$EDIT" \
  --reward clip --sweep_control_type "$ARM" \
  --sweep_N 8 --sweep_H 2 --sweep_lam 100000 \
  --flux_device cuda:0 --reward_device cuda:0 \
  --seed "$SEED" --quiet --resume \
  --out_dir "$OUT" $EXTRA >> "$LOG" 2>&1
rc=$?
echo "[$(date +%H:%M)] done  $TAG $ARM s$SEED rc=$rc"
exit $rc
