#!/usr/bin/env bash
# THE 2026-09-21 constrained run. One invocation = one (pair, arm-group, seed).
#
#   bash crun2.sh <A|B> <mpc|hardflow> <seed>
#
# CHANGES FROM crun.sh (all agreed with Shervin on the 09-21 call):
#   1. THREE arms, not two: vanilla MPC, MPC + adaptive dual, HardFlow.
#      Vanilla MPC is kept as the control -- the point is to measure the
#      adaptive dual's effect, so both must run on identical everything else.
#   2. lam = 5000, single value. 3000 vs 10000 made almost no difference
#      ("the control energy isn't really being penalised, it's just the
#      LPIPS"), so a lambda sweep is spent budget for no signal. 5000 is the
#      midpoint Shervin picked.
#   3. The GENERATION prompt is now the OPPOSITE of the edit, so the edit has
#      something to change. Previously "a portrait of a man" -- often already
#      middle-aged, sometimes already curly -- which is why several panels
#      looked identical to the uncontrolled image.
#   4. "gray hair" -> "white hair", per Shervin, for a bigger visible change.
#
# WHY "mpc" IS ONE JOB, NOT TWO. Shervin exposed the option as
# --sweep_dual_adaptive, a sweep AXIS, so `false true` runs BOTH MPC arms in
# one process against ONE model load. The load is ~2-3 min against a ~4.5 min
# run, so folding them saves ~2.5 min per (pair, seed) -- about 40 min of GPU
# time over the planned grid. The two arms land in the same out_dir and are
# told apart by the dual_adaptive CSV column; his --resume change
# (dual_adaptive in _RESUME_KEY_FIELDS) is what makes that safe, otherwise
# resume would see them as one config and skip the second.
#
# TAU comes from probe2.sh via probe2_read.py and is passed in as THRS_A /
# THRS_B. It is NOT defaulted: yesterday's grid was derived from the OLD base
# prompts' ceilings, and silently reusing it here is the single most likely
# way to burn the whole budget on runs that are all infeasible (or all
# trivially feasible). Fail loudly instead.
#
# ONE tau is broadcast to BOTH constraints on purpose. Giving each constraint
# its own hand-tuned tau would paper over precisely the cross-constraint scale
# mismatch that dual_adaptive exists to correct, and the three-way comparison
# would then measure nothing.
set -u
R="${FMOCC_ROOT:-$HOME/fmocc}"
cd "$R/fmocc_core" || exit 3
export PATH="$R/.venv/bin:$PATH"
export FLOW_MATCHING_DIR="$R" FMRG_DIR="$R/fmrg"
export CUDA_VISIBLE_DEVICES=0

PAIR="${1:?usage: crun2.sh <A|B> <mpc|hardflow> <seed>}"
ARM="${2:?usage: crun2.sh <A|B> <mpc|hardflow> <seed>}"
SEED="${3:?usage: crun2.sh <A|B> <mpc|hardflow> <seed>}"
LAM=5000

case "$PAIR" in
  A) BASE="a portrait of a young man with straight hair"
     E1="a face with curly hair"; E2="an old face"
     THRS="${THRS_A:-}" ;;
  B) BASE="a portrait of a serious man with dark hair"
     E1="a smiling face";         E2="a face with white hair"
     THRS="${THRS_B:-}" ;;
  *) echo "PAIR must be A or B"; exit 3 ;;
esac

if [ -z "$THRS" ]; then
  echo "THRS_$PAIR is unset. Run probe2.sh, then probe2_read.py, and export the"
  echo "THRS_A / THRS_B lines it prints. Refusing to fall back to the 09-20"
  echo "grid -- it was derived from different base prompts."
  exit 4
fi

case "$ARM" in
  mpc)      CTRL=mpc
            # false true = vanilla and adaptive, one model load, same process.
            EXTRA="--sweep_n_inner 24 --sweep_lr_inner 1e-2 --sweep_dual_adaptive false true" ;;
  hardflow) CTRL=hardflow
            EXTRA="--sweep_n_inner 40 --sweep_lr_inner 2.5 --hardflow_lr_alpha 0.5 --sweep_hardflow_margin 0.02" ;;
  *) echo "ARM must be mpc or hardflow"; exit 3 ;;
esac

OUT=results/crun2/pair${PAIR}_${ARM}_s${SEED}
LOG=results/crun2/log_${PAIR}_${ARM}_s${SEED}.txt
mkdir -p results/crun2

echo "=== $(date +%H:%M) pair $PAIR  arm $ARM  seed $SEED  lam $LAM" >> $LOG
echo "    base='$BASE'" >> $LOG
echo "    c0='$E1'  c1='$E2'  tau: $THRS" >> $LOG

# python -u: block-buffered logs cost us several false "is it stuck?" alarms
# on 2026-09-18/19. The flush is free; the uncertainty was not.
python -u run_sweep.py \
  --sweep_prompt "$BASE" --sweep_edit_prompt "$E1" \
  --reward lpips --sweep_control_type $CTRL \
  --sweep_N 8 --sweep_H 2 --sweep_lam $LAM \
  --constraint_types clip clip \
  --constraint_args "$E1" "$E2" \
  --constraint_directions ge ge \
  --sweep_constraint_thresholds $THRS \
  --lr_dual 0.2 --mu_init 0.0 \
  --flux_device cuda:0 --reward_device cuda:0 \
  --seed $SEED --quiet --resume \
  --out_dir $OUT $EXTRA >> $LOG 2>&1
rc=$?
echo "=== $(date +%H:%M) DONE $PAIR $ARM s$SEED rc=$rc" >> $LOG
exit $rc
