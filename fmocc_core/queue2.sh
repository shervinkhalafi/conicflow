#!/usr/bin/env bash
# Work queue for the 2026-09-21 run. 4 workers on one H200.
#
#   export THRS_A="..." THRS_B="..."          # from probe2_read.py
#   bash queue2.sh 42 7 1 2 3                 # seeds, in priority order
#
# ORDERING IS BY SEED, NOT BY ARM. The budget is a hard ~4 h wall clock and we
# will probably not finish every seed. Cutting off mid-seed would leave, say,
# MPC but no HardFlow for that seed -- which is exactly the comparison the run
# exists to make, so that seed would be wasted. Emitting all 4 jobs of seed k
# before any job of seed k+1 means the cutoff always falls on a seed boundary
# and every completed seed is a usable data point.
#
# 4 jobs per seed, not 6: the "mpc" job runs BOTH MPC arms (vanilla and
# adaptive) in one process via --sweep_dual_adaptive, sharing one model load.
# Within a seed the long mpc jobs are emitted first so they start while
# workers are free; hardflow (~15 s/run vs ~4.5 min) backfills.
#
# No gpu_lib here: one card, deliberately oversubscribed 4x. gpu_lib's
# one-job-per-card lock is right on the shared 24 GB A5000s and would idle 90%
# of a 140 GB H200. Do NOT copy this file back to the Penn box.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
R="${FMOCC_ROOT:-$HOME/fmocc}"
NWORK="${NWORK:-4}"
SEEDS=("$@")
[ ${#SEEDS[@]} -gt 0 ] || { echo "usage: queue2.sh <seed> [seed...]"; exit 2; }
: "${THRS_A:?export THRS_A first (see probe2_read.py)}"
: "${THRS_B:?export THRS_B first (see probe2_read.py)}"
export THRS_A THRS_B FMOCC_ROOT="$R"

JOBS="$R/fmocc_core/results/crun2/jobs.txt"
mkdir -p "$R/fmocc_core/results/crun2"
: > "$JOBS"
for s in "${SEEDS[@]}"; do
  for arm in mpc hardflow; do
    for p in A B; do
      echo "$p $arm $s" >> "$JOBS"
    done
  done
done
echo "$(wc -l < "$JOBS") jobs, $NWORK workers"

# xargs -P is the whole scheduler: it keeps exactly NWORK children alive and
# hands out the next line as each finishes, which is what we want and is one
# less hand-rolled thing to get wrong. -n3 feeds one job's three fields.
# Each job appends to its own log, so parallel writers never interleave.
< "$JOBS" xargs -P "$NWORK" -n 3 bash -c \
  'echo "[$(date +%H:%M)] start $0 $1 s$2"; bash '"$HERE"'/crun2.sh "$0" "$1" "$2"; \
   echo "[$(date +%H:%M)] end   $0 $1 s$2 rc=$?"'

echo "QUEUE DRAINED $(date +%H:%M)"
