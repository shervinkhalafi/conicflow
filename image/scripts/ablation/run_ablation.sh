#!/usr/bin/env bash
# run_ablation.sh -- run (part of) the appendix ablations from jobs.tsv.
#
#   bash run_ablation.sh <gpu> [--arms "<names or globs>"] [--tags "ed1 ed2 ..."]
#                              [--seeds "317 1289 ..."] [--dry_run]
#
# Examples
#   bash run_ablation.sh 0                                 # everything (~130 GPU-h)
#   bash run_ablation.sh 0 --arms core                     # the three default arms
#   bash run_ablation.sh 1 --arms 'conicflow_H4_ni24'      # one horizon
#   bash run_ablation.sh 0 --arms '*_horizon' --seeds 317  # objective-point arms, one seed
#   bash run_ablation.sh 0 --dry_run                       # print the commands only
#
# One process, one card: the card is pinned with CUDA_VISIBLE_DEVICES and the
# run uses cuda:0 for both FLUX and the reward model. To use two cards, start
# two instances with disjoint filters (e.g. split by --seeds); two instances
# on the SAME rows would race on the same output directory.
#
# RESUMABLE. A row is skipped when its summary.csv already holds a run_ok=True
# row for every lambda in the list; otherwise run_sweep.py is called with
# --resume, which re-runs only the missing or failed lambdas. A failed row is
# logged and the queue moves on; a row that fails in under a minute (bad
# arguments, import error, no GPU) stops the queue instead.
#
# Output: $OUT_ROOT/pickscore/<prompt slug>/<arm>_seed<seed>/  (summary.csv,
# images, run.log), with OUT_ROOT relative to image/src/. Per-row logs go to
# $OUT_ROOT/logs/.
set -u
usage() { sed -n '4,5p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
GPU="${1:-}"; [ -n "$GPU" ] || usage; shift
ARMS="*"; TAGS="*"; SEEDS="*"; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --arms)    ARMS="${2:?}"; shift 2 ;;
        --tags)    TAGS="${2:?}"; shift 2 ;;
        --seeds)   SEEDS="${2:?}"; shift 2 ;;
        --dry_run) DRY=1; shift ;;
        *) usage ;;
    esac
done
[ "$ARMS" = core ] && ARMS="conicflow_H2_ni24 hardflow_ni40 fmrg_nopt3"

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../../src" && pwd)"
JOBS="${JOBS:-$HERE/jobs.tsv}"
OUT_ROOT="${OUT_ROOT:-results/ablation}"
cd "$SRC" || exit 3
[ "$DRY" -eq 1 ] || mkdir -p "$OUT_ROOT/logs"

matches() {   # matches <value> <space-separated globs>
    local v="$1" pat
    set -f; for pat in $2; do set +f; case "$v" in $pat) return 0 ;; esac; done; set +f
    return 1
}
slug_of() { printf '%s__%s' "$1" "$2" | tr -c 'A-Za-z0-9' '_' | cut -c1-60; }
complete() {  # complete <out_dir> <lams>: every lambda has a run_ok=True row
    python - "$1/summary.csv" "$2" <<'PY'
import csv, os, sys
path, lams = sys.argv[1], {round(float(x), 6) for x in sys.argv[2].split()}
if not os.path.exists(path):
    sys.exit(1)
done = {round(float(r["lam"]), 6) for r in csv.DictReader(open(path)) if r.get("run_ok") == "True"}
sys.exit(0 if lams <= done else 1)
PY
}

n_run=0; n_skip=0; n_fail=0
while IFS=$'\t' read -r tag seed arm unlock lams prompt edit args; do
    case "$tag" in ''|'#'*) continue ;; esac
    matches "$arm" "$ARMS" && matches "$tag" "$TAGS" && matches "$seed" "$SEEDS" || continue
    out="$OUT_ROOT/pickscore/$(slug_of "$prompt" "$edit")/${arm}_seed${seed}"
    env_prefix=""; [ "$unlock" != "-" ] && env_prefix="FLOWMAPS_UNLOCK=$unlock "
    cmd="CUDA_VISIBLE_DEVICES=$GPU ${env_prefix}python run_sweep.py --sweep_prompt \"$prompt\" --sweep_edit_prompt \"$edit\" --seed $seed $args --sweep_lam $lams --flux_device cuda:0 --reward_device cuda:0 --out_dir $out --resume"
    if [ "$DRY" -eq 1 ]; then echo "$cmd"; continue; fi
    if complete "$out" "$lams"; then n_skip=$((n_skip + 1)); continue; fi

    log="$OUT_ROOT/logs/${tag}_seed${seed}_${arm}.log"
    echo "=== $(date +%H:%M:%S) START $tag seed=$seed $arm  lam=[$lams]"
    t0=$(date +%s)
    # The locked default constraint_on=endpoint is lifted for the horizon arms
    # only; run_sweep.py stamps that in the `unlocked` CSV column.
    if [ "$unlock" != "-" ]; then export FLOWMAPS_UNLOCK="$unlock"; else unset FLOWMAPS_UNLOCK; fi
    # shellcheck disable=SC2086 -- args and lams are lists on purpose
    CUDA_VISIBLE_DEVICES="$GPU" python run_sweep.py \
        --sweep_prompt "$prompt" --sweep_edit_prompt "$edit" --seed "$seed" \
        $args --sweep_lam $lams \
        --flux_device cuda:0 --reward_device cuda:0 \
        --out_dir "$out" --resume >> "$log" 2>&1
    rc=$?; t1=$(date +%s)
    unset FLOWMAPS_UNLOCK
    if [ "$rc" -eq 0 ]; then
        n_run=$((n_run + 1))
        echo "=== $(date +%H:%M:%S) OK    $tag seed=$seed $arm in $(( (t1 - t0) / 60 )) min"
    else
        n_fail=$((n_fail + 1))
        echo "=== $(date +%H:%M:%S) FAIL  $tag seed=$seed $arm rc=$rc (log: $SRC/$log)"
        if [ $((t1 - t0)) -lt 60 ]; then
            echo "failed in under a minute -- something systematic; stopping"; exit 1
        fi
    fi
done < "$JOBS"
[ "$DRY" -eq 1 ] || echo "done: $n_run run, $n_skip already complete, $n_fail failed"
[ "$n_fail" -eq 0 ]
