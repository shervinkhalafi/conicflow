#!/usr/bin/env bash
# THRESHOLD + SEED PROBE for the 2026-09-21 run. Run this FIRST, on the H200.
#
#   bash probe2.sh              # ~30 min on an H200 with 4 workers
#   python probe2_read.py       # then read the answer out
#
# WHY IT EXISTS. The base prompts changed today (Shervin: make the generation
# prompt the OPPOSITE of the edit, so the edit is visibly doing something).
# That moves BOTH ends of the CLIP scale: the uncontrolled image now starts
# further from the edit prompt (r_unc down, which is the point) and the
# reachable ceiling moves too. Yesterday's tau grid was derived from the OLD
# prompts' ceilings and is not transferable. Set tau above the ceiling and
# every run is infeasible; set it below r_unc and mu never leaves zero and the
# run measures nothing. Probe, do not guess.
#
# WHAT IT MEASURES, with CLIP as the REWARD and no constraints:
#   r_unc   -- CLIP(uncontrolled image, edit prompt). Shervin's "someone who is
#              not already curly / not already old" check, per seed. LOW is
#              what we want, and it is what picks the seeds.
#   r_ctrl  -- the reachable ceiling for that arm at this lam.
#
# COST. Model load (~2-3 min, int8) dominates a hardflow run (~15 s on an
# H200), so this is load-bound, not compute-bound -- which is exactly why it
# runs under xargs -P rather than serially. mpc gets ONE seed: the ceiling is
# a property of the prompt pair far more than of the seed, and mpc is ~25x
# hardflow's cost.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
R="${FMOCC_ROOT:-$HOME/fmocc}"
NWORK="${NWORK:-4}"
SEEDS="${SEEDS:-42 7 1 2 3 4}"
export FMOCC_ROOT="$R"

OUT="$R/fmocc_core/results/probe2"
mkdir -p "$OUT"
JOBS="$OUT/jobs.txt"
: > "$JOBS"

# tag  base-key  edit-prompt
COMBOS=(
  "A0|A|a face with curly hair"
  "A1|A|an old face"
  "B0|B|a smiling face"
  "B1|B|a face with white hair"
)

# mpc first: it is ~25x slower, so it must start while workers are free or it
# becomes the tail that decides the wall clock.
for c in "${COMBOS[@]}"; do echo "$c|mpc|42" >> "$JOBS"; done
for s in $SEEDS; do
  for c in "${COMBOS[@]}"; do echo "$c|hardflow|$s" >> "$JOBS"; done
done
echo "$(wc -l < "$JOBS") probe jobs, $NWORK workers"

< "$JOBS" xargs -P "$NWORK" -I{} bash "$HERE/probe2_one.sh" "{}"
echo "PROBE DONE $(date +%H:%M)  ->  python probe2_read.py"
