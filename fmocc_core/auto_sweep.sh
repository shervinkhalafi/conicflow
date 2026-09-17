#!/usr/bin/env bash
# auto_sweep.sh -- range-find lambda, then run the real sweep on that grid.
#
#   auto_sweep.sh <gpu> <reward> "<prompt>" "<edit_prompt>" [extra run_sweep args...]
#
# Step 1 runs lambda_autorange.py at a cheap proxy N (default 4) for mpc,
# hardflow and fmrg and writes grids.txt with one --sweep_lam line per method.
# Step 2 runs run_sweep.py once per method at TARGET_N on exactly that grid.
# Everything lands under results/auto/<reward>/<slug>/.
#
# Environment overrides (all optional):
#   TARGET_N=10  PROBE_N=4  N_GRID=6  SEEDS="42"  MPC_H=2  MPC_NI=12
#   HF_NI=40 HF_LR=2.5  FMRG_NOPT=3  AUTORANGE_EXTRA="..."
#
# The card is pinned with CUDA_VISIBLE_DEVICES and both --*_device are cuda:0
# in-process (never pass cuda:1 to run_sweep.py: ImageReward misbehaves there).
set -u
GPU="$1"; REWARD="$2"; PROMPT="$3"; EDIT="$4"; shift 4
TARGET_N="${TARGET_N:-10}"; PROBE_N="${PROBE_N:-4}"; N_GRID="${N_GRID:-6}"; SEEDS="${SEEDS:-42}"
MPC_H="${MPC_H:-2}"; MPC_NI="${MPC_NI:-12}"; HF_NI="${HF_NI:-40}"; HF_LR="${HF_LR:-2.5}"; FMRG_NOPT="${FMRG_NOPT:-3}"
cd "$(dirname "$0")" || exit 3
SLUG="$(printf '%s__%s' "$PROMPT" "$EDIT" | tr -c 'A-Za-z0-9' '_' | cut -c1-60)"
OUT="results/auto/$REWARD/$SLUG"; mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "[auto_sweep] step 1/2: lambda range finding ($REWARD, proxy N=$PROBE_N -> target N=$TARGET_N)"
python lambda_autorange.py --reward "$REWARD" --prompt "$PROMPT" --edit_prompt "$EDIT" \
    --methods mpc hardflow fmrg --probe_N "$PROBE_N" --target_N "$TARGET_N" --n_grid "$N_GRID" \
    --flux_device cuda:0 --reward_device cuda:0 --out_dir "$OUT/autorange" ${AUTORANGE_EXTRA:-} \
    || { echo "[auto_sweep] range finding FAILED"; exit 1; }

echo "[auto_sweep] step 2/2: sweeps at N=$TARGET_N"
while read -r method flag grid; do
    [ "$flag" = "--sweep_lam" ] || continue
    grid="${grid%%#*}"                      # strip the trailing '# ...' note
    case "$method" in
        mpc)      extra=(--sweep_H "$MPC_H" --sweep_n_inner "$MPC_NI" --sweep_lr_inner 1e-2) ;;
        hardflow) extra=(--sweep_H 2 --sweep_n_inner "$HF_NI" --sweep_lr_inner "$HF_LR") ;;
        fmrg)     extra=(--sweep_n_inner "$FMRG_NOPT") ;;
        *)        continue ;;
    esac
    for seed in $SEEDS; do
        echo "[auto_sweep]   $method  lam = $grid  seed=$seed"
        python run_sweep.py --sweep_prompt "$PROMPT" --sweep_edit_prompt "$EDIT" --reward "$REWARD" \
            --sweep_control_type "$method" --sweep_N "$TARGET_N" ${extra[@]+"${extra[@]}"} \
            --sweep_lam $grid --seed "$seed" --flux_device cuda:0 --reward_device cuda:0 \
            --out_dir "$OUT/${method}_seed${seed}" --resume "$@" \
            || echo "[auto_sweep]   $method seed=$seed exited non-zero (rows with run_ok=False are in its summary.csv)"
    done
done < "$OUT/autorange/grids.txt"
echo "[auto_sweep] done -> $OUT"
