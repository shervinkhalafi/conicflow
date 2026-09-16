#!/bin/bash
# ---------------------------------------------------------------------------
# lambda_suite_v2_20260915.sh -- the unconstrained lambda suite. CURRENT.
# Written 2026-09-15 16:45. Supersedes lambda_suite_20260915.sh (v1, ran 20 min)
# and fmrg_niopt_suite_20260915.sh (the n_opt companion, cancelled as overkill).
# Runs unattended. ~11.3 h of GPU, ~5.7 h wall across the two cards.
#
# ── WHAT CHANGED FROM v1 ───────────────────────────────────────────────────
# FMRG's n_inner: 20 -> 3. Sung's call: use what the authors used, and do not
# sweep it. 3 is Boffi's published `num_optim_iters` default.
#
# Note this is LOWER than the 5 every previous FMRG run on disk used, and it is
# not "pushed a bit further up" as Shervin phrased it on the call -- so it is a
# deliberate override in favour of the published default. The evidence supports
# it: the only clean n_opt ablation we have (FMRG at fixed N=10, reward=clip,
# unconstrained, 4 runs/cell) shows the objective getting WORSE as n_opt rises,
# with ni=1 best at every lam >= 1:
#
#     lam=10    ni=1     ni=3     ni=5     ni=10
#     J        -0.2002  -0.1963  -0.1942  -0.1888
#     lam=100  -0.2774  -0.2692  -0.2748  -0.2577
#
# n_opt does not change FMRG's control budget -- the n_opt steps of
# (dt/n_opt)*lam*grad r sum to dt*lam*(mean grad) -- it trades AGGRESSION for
# FIDELITY, lowering both reward and energy. So a bigger n_opt is not "more
# effort", and 3 is a defensible, published operating point rather than a number
# we invented.
#
# ── THE SETUP, AND WHERE EACH PIECE WAS DECIDED ────────────────────────────
#   equal capital N = 10 for all three   2026-09-15 call (Shervin: "keep capital
#                                        N the same... this would make the
#                                        images also compare"). Supersedes the
#                                        09-14 matched-wall-clock plan.
#   mpc   N=10 H=4 ni=12 lr=1e-2         2026-09-15 call, explicitly: "capital
#                                        N 10, n inner 12".
#   hardflow N=10 ni=40 lr=2.5           ni=40 is the authors' value; lr=2.5 was
#                                        fixed by the 09-14 decision ("HardFlow
#                                        LR stays at 2.5"), and pushing ni to
#                                        260 was already shown to destabilise
#                                        (energy 5.98e9 vs 3.79 at ni=40).
#   fmrg  N=10 ni=3                      this file; Boffi's default.
#   lam 0.1 ... 10000, 8 values          2026-09-15 call ("10,000 the maximum,
#                                        but start from like 0.1... 7 or 8").
#   reward = imagereward                 09-14 call, fixed for the test suite.
#   unconstrained                        09-14/09-15: unconstrained results
#                                        first, then constrained.
#   3 seeds x 2 prompts                  09-14 asked for 2 prompts x 2 seeds;
#                                        3 seeds because ctrl_energy is unusable
#                                        at n=1. Prompt 2 ("a portrait of a
#                                        woman") is MY choice, not discussed.
#   wall clock NOT matched               09-15 call accepted this as the cost of
#                                        holding N fixed ("we'll figure out how
#                                        we present that later"). MPC ~710 s,
#                                        FMRG ~79 s, HardFlow ~86 s per run.
#
# Locked defaults are untouched: drift_detach=False, constraint_on="endpoint",
# warm_start=True. No FLOWMAPS_UNLOCK anywhere in this file.
#
# ── WHY EQUAL N MATTERS (the measurement behind the 09-15 decision) ────────
# The uncontrolled sample depends on N. Same seed, same z0, no control, seed 43:
# r_unc = -1.581 at N=8, +0.658 at N=50, +1.234 at N=100 -- and the N=8 image is
# a DIFFERENT SUBJECT than the N=50/100 one. So under the old matched-wall-clock
# setup the methods were not controlling the same problem, and HardFlow started
# ~1.4 reward points ahead purely from having a finer grid. With N equal, every
# method sees the identical uncontrolled sample for a given seed, delta_r is
# meaningful again, and the qualitative grid can share one uncontrolled column.
#
# ── hardflow_reg, PER LAMBDA, BY HAND ──────────────────────────────────────
# reg = N/(2*lam) at this arm's own N (=10), maintained manually per Sung's
# standing instruction rather than computed in the loop -- the table is the point:
#
#     lam      reg = 5/lam     lr*reg (lr=2.5)   status
#     0.1      50.0            125.0             BLOCKED
#     1         5.0             12.5             BLOCKED
#     10        0.5              1.25            BLOCKED
#     50        0.1              0.25            ok
#     200       0.025            0.0625          ok
#     1000      0.005            0.0125          ok
#     3000      0.0016666667     0.00417         ok
#     10000     0.0005           0.00125         ok  <- the authors' nominal value
#                                                     and RunConfig's default
#
# hardflow_image raises when lr*reg >= 1, because its inner loop is EXPLICIT
# fixed-step ascent and the deviation penalty's error evolves as
# e <- e*(1 - 2*lr*reg*alpha^2), which contracts only while lr*reg*alpha^2 < 1;
# alpha -> 1 at the final step. That is a solver stability limit, NOT a limit of
# HardFlow's formulation -- the penalised minimiser is well defined at any reg.
# Per Sung's 09-14 decision the floor is REPORTED AS A RESULT and lr is not
# touched. The three blocked lambdas are launched ONCE each, single attempt, so
# the guard's message lands in a FAILED_ dir and in error_msg as the receipt.
# ---------------------------------------------------------------------------
set -uo pipefail
# Repo-relative, not an absolute home path. This file ships in a git repo that
# other people clone, and `cd /home/sungcho` (what it said until 2026-09-15)
# breaks on any box that is not mine.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1

source ./gpu_lib.sh
export MAX_ATTEMPTS=3
export BACKOFF=60

TS=$(date '+%Y%m%d_%H%M%S')
ROOT="results"
LABEL="lamsuite2"
RUNDIR="${ROOT}/${TS}_${LABEL}"
QUEUE="${RUNDIR}/QUEUE.txt"
QLOCK="${RUNDIR}/.queue.lock"
POISON="${RUNDIR}/POISONED_ARMS.txt"
DIRLIST="${RUNDIR}/dirlist.txt"
mkdir -p "$RUNDIR"
: > "$POISON"
ledger_init "$RUNDIR"

REWARD="imagereward"
NCAP=10                      # capital N, the same for every method
LAMS=(0.1 1 10 50 200 1000 3000 10000)
SEEDS=(42 43 44)
PROMPTS=("a portrait of a man|a smiling face"
         "a portrait of a woman|a smiling face")

declare -A HF_REG=( [0.1]=50.0 [1]=5.0 [10]=0.5 [50]=0.1 [200]=0.025
                    [1000]=0.005 [3000]=0.0016666667 [10000]=0.0005 )
declare -A HF_BLOCKED=( [0.1]=1 [1]=1 [10]=1 )

BUDGET_MIN=840               # ~14 h; the 10:00 meeting is the real deadline
DEADLINE=$(( $(date +%s) + BUDGET_MIN*60 ))
declare -A PRED=( [mpc]=13 [hardflow]=2 [fmrg]=2 )

slog() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] lamsuite2: $*" >&2; }

gpu_preflight
slog "=================================================================="
slog "${LABEL} ${TS} -- UNCONSTRAINED lambda suite, EQUAL N=${NCAP}"
slog "mpc N=10 H=4 ni=12 | hardflow N=10 ni=40 lr2.5 | fmrg N=10 ni=3 (Boffi)"
slog "lambdas ${LAMS[*]}"
slog "seeds ${SEEDS[*]} x ${#PROMPTS[@]} prompts; reward=${REWARD}; unconstrained"
slog "wall clock deliberately NOT matched (N held fixed instead)"
slog "deadline $(date -d "@${DEADLINE}" '+%a %H:%M')"
slog "=================================================================="

# prompt-major so prompt 1 is a COMPLETE figure first; lambda-major within it so
# a deadline cut costs whole lambda points rather than holes in every curve.
: > "$QUEUE"
for pi in "${!PROMPTS[@]}"; do
  for lam in "${LAMS[@]}"; do
    for arm in mpc fmrg hardflow; do
      if [ "$arm" = "hardflow" ] && [ -n "${HF_BLOCKED[$lam]:-}" ]; then
        echo "${arm} ${lam} 42 ${pi} blocked" >> "$QUEUE"
        continue
      fi
      for s in "${SEEDS[@]}"; do echo "${arm} ${lam} ${s} ${pi} ok" >> "$QUEUE"; done
    done
  done
done
slog "queued $(wc -l < "$QUEUE") jobs"
touch "$QLOCK"

next_job() {
  local line
  exec 9>>"$QLOCK"; flock 9
  line=$(head -n1 "$QUEUE" 2>/dev/null)
  [ -n "$line" ] && sed -i '1d' "$QUEUE"
  flock -u 9; exec 9>&-
  printf '%s' "$line"
}

arm_extra() {
  local arm="$1" lam="$2"
  case "$arm" in
    mpc)      echo "--sweep_control_type mpc --sweep_N ${NCAP} --sweep_H 4 --sweep_n_inner 12 --sweep_lr_inner 1e-2" ;;
    hardflow) echo "--sweep_control_type hardflow --sweep_N ${NCAP} --sweep_H 2 --sweep_n_inner 40 --sweep_lr_inner 2.5 --sweep_hardflow_reg ${HF_REG[$lam]}" ;;
    fmrg)     echo "--sweep_control_type fmrg --sweep_N ${NCAP} --sweep_n_inner 3" ;;
    *) return 1 ;;
  esac
}

one_run() {
  local arm="$1" lam="$2" seed="$3" pi="$4" blocked="$5" wlog="$6"
  local spec="${PROMPTS[$pi]}"
  local prompt="${spec%%|*}" edit="${spec##*|}"
  local label="${arm}_lam${lam}_p${pi}"
  local out="${RUNDIR}/${label}_seed${seed}"
  local need; need=$(need_mib_for "$arm") || return 1
  local extra; extra=$(arm_extra "$arm" "$lam") || return 1

  local maxtry="$MAX_ATTEMPTS"
  [ "$blocked" = "blocked" ] && maxtry=1

  local attempt=1
  while [ "$attempt" -le "$maxtry" ]; do
    local gpu; gpu=$(pick_gpu "$need")
    slog "START ${label} seed=${seed} gpu=${gpu} (try ${attempt}/${maxtry})$(
          [ "$blocked" = "blocked" ] && echo ' [EXPECTED TO RAISE: lr*reg>=1]')"
    local t0; t0=$(date +%s)

    # shellcheck disable=SC2086
    run_on_gpu "$gpu" "$wlog" \
      --sweep_prompt "$prompt" --sweep_edit_prompt "$edit" \
      --reward "$REWARD" --sweep_lam "$lam" \
      $extra \
      --resume --seed "$seed" --quiet \
      --out_dir "$out"
    local rc=$?
    release_gpu "$gpu"

    if [ $rc -eq 0 ]; then
      local el=$(( $(date +%s) - t0 ))
      slog "DONE  ${label} seed=${seed} in $((el/60))m$((el%60))s"
      echo "${label}_seed${seed}" >> "$DIRLIST"
      return 0
    fi
    if [ "$blocked" = "blocked" ]; then
      slog "RAISED as expected: ${label} (the lambda floor; this is the receipt)"
      ledger_add "${label} -- EXPECTED failure, lr*reg>=1 at lam=${lam} (reg=${HF_REG[$lam]})"
      echo "${label}_seed${seed} EXPECTED_FAIL" >> "$DIRLIST"
      return 0
    fi
    slog "FAILED ${label} seed=${seed} rc=${rc} (try ${attempt})"
    attempt=$((attempt+1))
    [ "$attempt" -le "$maxtry" ] && sleep "$BACKOFF"
  done

  slog "GAVE UP ${label} seed=${seed}"
  ledger_add "${label} seed=${seed} -- gave up after ${maxtry} attempts"
  echo "${label}_seed${seed} FAILED" >> "$DIRLIST"
  return 1
}

worker() {
  local wid="$1"
  local wlog="${RUNDIR}/worker${wid}.log"
  export GPU_HOLDER_PID=$BASHPID      # before any pick_gpu; see gpu_lib.sh
  install_gpu_trap
  slog "worker${wid} up (pid ${BASHPID})"
  while true; do
    local job; job=$(next_job)
    [ -z "$job" ] && break
    # shellcheck disable=SC2086
    set -- $job
    local arm="$1" lam="$2" seed="$3" pi="$4" blocked="$5"

    if grep -qx "$arm" "$POISON" 2>/dev/null; then
      slog "SKIP ${arm} lam=${lam} seed=${seed} -- arm poisoned"
      ledger_add "${arm} lam=${lam} seed=${seed} -- SKIPPED, arm poisoned"
      continue
    fi
    local left=$(( (DEADLINE - $(date +%s)) / 60 ))
    if [ "${PRED[$arm]}" -gt "$left" ]; then
      slog "DEFER ${arm} lam=${lam} seed=${seed} -- needs ~${PRED[$arm]}m, ${left}m left"
      ledger_add "${arm} lam=${lam} seed=${seed} -- DEFERRED, out of budget"
      continue
    fi
    if ! one_run "$arm" "$lam" "$seed" "$pi" "$blocked" "$wlog"; then
      if [ "$seed" = "42" ] && [ "$lam" = "${LAMS[0]}" ] && [ "$blocked" != "blocked" ]; then
        echo "$arm" >> "$POISON"
        slog "POISON ${arm}: failed at the first lambda on seed 42"
      fi
    fi
  done
  slog "worker${wid} done"
}

worker 0 & W0=$!
worker 1 & W1=$!
wait "$W0" "$W1"

slog "ALL WORKERS DONE."
ledger_report
slog "rundir: ${RUNDIR}"
slog "plot with:"
slog "  python plot_fig2_unconstrained.py ${RUNDIR} --out figures/fig2_curves.png"
slog "  python plot_fig2_qualgrid.py ${RUNDIR} --lam 1000 --out figures/fig2_qualgrid_lam1000.png"
