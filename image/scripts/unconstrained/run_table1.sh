#!/usr/bin/env bash
# run_table1.sh -- the unconstrained image experiment: Table 1, Figure 2 (left),
# the two appendix reward-energy frontiers, and appendix rows u05-u10.
#
#   bash run_table1.sh <gpu> [--dry_run]
#
# Runs every problem in table1_jobs.tsv (3 rewards x 4 prompt/seed instances)
# for ConicFlow, HardFlow and FMRG on that problem's FROZEN lambda grid, by
# calling run_sweep.py directly. Nothing is range-found: the grids are the ones
# the paper used, read from image/data/grids/unconstrained/ (see README.md in
# this directory for how to re-derive them instead).
#
# Per (problem, method) this is one run_sweep.py process writing
#   image/src/$RESULTS_ROOT/<reward>/<slug>/<method>_seed<seed>/summary.csv
# which is the layout image/analysis/table1_breakdown.py and fig2_pareto.py read.
#
# Settings (all as in the paper, Appendix "Unconstrained image experiment"):
#   all        N=8, int8 transformer, fp32 reward (run_sweep.py defaults)
#   ConicFlow  H=2, n_inner 12 and 24, Adam lr 1e-2, endpoint_surrogate=rollout
#              and drift_detach=True -- both are run_sweep.py's defaults, so
#              neither is passed here
#   HardFlow   n_inner 40, lr 2.5, --hardflow_lr_alpha 0.5 (lr -> 0.5/reg where
#              lr*reg >= 1); reg = N/(2*lam) is derived per run, never passed
#   FMRG       n_opt 3 (the J variant)
#
# The GPU is pinned with CUDA_VISIBLE_DEVICES and both devices are cuda:0
# in-process. Run one instance per card (ConicFlow peaks at 15-19 GB).
# Resumable: run_sweep.py --resume skips configs already in summary.csv with
# run_ok=True and retries the rest, so re-running the script after an
# interruption continues where it stopped.
#
# Environment overrides:
#   CONICFLOW_NI="12 24"   ConicFlow inner-iteration counts. Table 1 and every
#                          figure use 24 only; CONICFLOW_NI=24 skips the n_inner=12
#                          configuration and saves roughly a quarter of the cost.
#   METHODS="conicflow hardflow fmrg"
#   JOBS=table1_jobs.tsv   RESULTS_ROOT=results/unconstrained   PYTHON=python
#   ONLY_REWARD=pickscore  run one reward's rows only
set -u
GPU="${1:?usage: run_table1.sh <gpu> [--dry_run]}"
DRY=0; [ "${2:-}" = "--dry_run" ] && DRY=1
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../../src" && pwd)"
GRIDS="$(cd "$HERE/../../data/grids/unconstrained" && pwd)"
JOBS="${JOBS:-$HERE/table1_jobs.tsv}"
RESULTS_ROOT="${RESULTS_ROOT:-results/unconstrained}"
CONICFLOW_NI="${CONICFLOW_NI:-12 24}"
METHODS="${METHODS:-conicflow hardflow fmrg}"
PYTHON="${PYTHON:-python}"
ONLY_REWARD="${ONLY_REWARD:-}"

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$SRC" || exit 3

n_fail=0
while IFS=$'\t' read -r reward seed prompt edit densify; do
    case "$reward" in ''|'#'*) continue ;; esac
    [ -n "$ONLY_REWARD" ] && [ "$reward" != "$ONLY_REWARD" ] && continue
    slug="$(printf '%s__%s' "$prompt" "$edit" | tr -c 'A-Za-z0-9' '_' | cut -c1-60)"
    grid_file="$GRIDS/${reward}__${slug}.txt"
    [ -s "$grid_file" ] || { echo "[table1] missing grid $grid_file"; exit 1; }
    lams="$(awk '$1=="shared" && $2=="--sweep_lam" {$1="";$2="";print}' "$grid_file")"
    [ -n "$lams" ] || { echo "[table1] no shared line in $grid_file"; exit 1; }
    [ "$densify" != "-" ] && lams="$lams $densify"

    for method in $METHODS; do
        # shellcheck disable=SC2206 -- CONICFLOW_NI is a list on purpose
        case "$method" in
            conicflow) extra=(--sweep_H 2 --sweep_n_inner $CONICFLOW_NI --sweep_lr_inner 1e-2) ;;
            hardflow)  extra=(--sweep_H 2 --sweep_n_inner 40 --sweep_lr_inner 2.5 --hardflow_lr_alpha 0.5) ;;
            fmrg)      extra=(--sweep_n_inner 3) ;;
            *) echo "[table1] unknown method $method"; exit 1 ;;
        esac
        out="$RESULTS_ROOT/$reward/$slug/${method}_seed${seed}"
        # shellcheck disable=SC2086 -- $lams is a list on purpose
        cmd=("$PYTHON" run_sweep.py --reward "$reward"
             --sweep_prompt "$prompt" --sweep_edit_prompt "$edit"
             --sweep_control_type "$method" --sweep_N 8 "${extra[@]}"
             --sweep_lam $lams --seed "$seed"
             --flux_device cuda:0 --reward_device cuda:0 --quiet --resume
             --out_dir "$out")
        if [ "$DRY" = 1 ]; then
            printf '%q ' "${cmd[@]}"; echo
            continue
        fi
        echo "=== $(date +%H:%M:%S) $reward | seed $seed | $method | $prompt -> $edit"
        echo "    lam = $lams"
        "${cmd[@]}" || { n_fail=$((n_fail + 1))
            echo "[table1] $method exited non-zero (failed rows carry run_ok=False in $out/summary.csv)"; }
    done
done < "$JOBS"
[ "$DRY" = 1 ] && exit 0
echo "[table1] done; $n_fail run_sweep.py invocation(s) exited non-zero"
[ "$n_fail" -eq 0 ]
