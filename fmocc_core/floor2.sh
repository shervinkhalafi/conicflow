#!/usr/bin/env bash
# MINIMAL floor probe -- the 8-minute version of probe2.sh, for when the clock
# matters more than the ceiling.
#
#   bash floor2.sh              # ~8 min on an H200, 4 workers
#   python floor2_read.py       # prints THRS_A / THRS_B
#
# WHAT IT SKIPS AND WHY THAT IS OK. probe2.sh measures two things: the floor
# (r_unc, CLIP of the UNCONTROLLED image against each edit prompt) and the
# ceiling (what each arm can actually reach). Today only the BASE prompts
# changed -- the edit prompts are the same four as 09-20 -- and the ceiling is
# mostly a property of the method's control authority at fixed N and lambda,
# so 09-20's measured ceilings still apply. The floor is what moved, by
# design, and it is the one that cannot be guessed.
#
# WHY THE FLOOR IS THE DANGEROUS ONE. Set tau below it and the constraints are
# already satisfied at the uncontrolled image: mu never leaves zero, all three
# arms reduce to the uncontrolled trajectory, and the whole block measures
# nothing. That failure consumes the entire run, not 30 minutes of it. Setting
# tau slightly wrong against the ceiling only costs the top row of the grid.
#
# HOW IT IS CHEAP. --sweep_control_type none does the uncontrolled rollout
# only (~8 forward passes, seconds) and still records r_unc. --sweep_prompt
# and --sweep_edit_prompt are sweep AXES, so one model load covers both base
# prompts x all four edit prompts. Cost is then ~1 model load per seed, not
# per measurement.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
R="${FMOCC_ROOT:-$HOME/fmocc}"
NWORK="${NWORK:-4}"
SEEDS="${SEEDS:-42 7 1 2 3 4}"
export FMOCC_ROOT="$R"

mkdir -p "$R/fmocc_core/results/floor2"
echo "floor probe: seeds [$SEEDS], $NWORK workers"
printf '%s\n' $SEEDS | xargs -P "$NWORK" -I{} bash "$HERE/floor2_one.sh" "{}"
echo "FLOOR DONE $(date +%H:%M)  ->  python floor2_read.py"
