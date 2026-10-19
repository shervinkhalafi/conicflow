#!/usr/bin/env bash
# run_suite.sh -- the constrained experiment: ConicFlow vs HardFlow under
# PickScore CONSTRAINTS and no reward, over a grid of constraint tightness.
# This is the run behind Figure 3 and the constrained image grids in the
# appendix.
#
#   bash run_suite.sh <gpu> [suite|triple] [--dry_run]
#
#   suite   (default) the 8 one/two-constraint instances in jobs/suite.tsv,
#           m = 1.1 1.2 1.3 1.4 1.5, written to results/constrained_suite/.
#           This tree alone is Figure 3 (left).
#   triple  the three-constraint instance in jobs/triple.tsv, m = 1.1 1.3 1.5,
#           written to its own tree results/constrained_triple/, so it never
#           enters the Figure 3 statistics. Needs a 24 GB card to itself.
#   --dry_run  print the compare_constrained.py commands and exit (no GPU).
#
#   Paths are relative to image/src/, where the work happens. The GPU is pinned
#   with CUDA_VISIBLE_DEVICES; every process sees it as cuda:0. Run ONE
#   instance per card. Peak memory grows with the number of constraint heads
#   resident next to the transformer: 16.2 GB with one, 19.2 GB with two,
#   22.2 GB with three (both methods, within 0.3 GB).
#
# THE PROBLEM. Each line of the job file is a base prompt plus one to three
# attribute prompts. There is no reward; each attribute prompt c_i is a
# constraint  h_i = tau_i - PickScore(image, c_i) <= 0,  and each method's
# objective is its control energy alone. Both methods must reach the same
# feasible set; the question is what it costs, and whether they get there.
#
# PER (instance, seed), via compare_constrained.py --auto_threshold:
#   1. one uncontrolled sample at this seed, scored with every attribute
#      prompt -> baselines s0_i;
#   2. a grid of multipliers m from AUTO_LO to AUTO_HI, each setting EVERY
#      threshold of the instance at once: tau_i(m) = m * s0_i;
#   3. at each m, ConicFlow then HardFlow, as separate run_sweep.py processes
#      on the same seed, lambda and thresholds (their uncontrolled images are
#      checked to be byte-identical);
#   4. every saved image re-scored with the same PickScore code the control
#      loop enforced, and a three-panel compare.png per m.
#   Output per cell: $OUT_ROOT/<tag>_seed<seed>/
#     auto_threshold.json        baselines, every point's thresholds
#     uncontrolled/              the probe run
#     m1.10/ ... m1.50/          conicflow/ and hardflow/ (run_sweep.py output
#                                dirs), compare.png, comparison.json
#
# SETTINGS (the paper's; see docs/HYPERPARAMETERS.md section 8)
#   lambda = 20 for both arms; HardFlow's reg = N/(2*lambda) = 0.2.
#   ConicFlow  N=8 H=2 n_inner=24 lr=1e-2, jump surrogate, drift detached,
#              mu_init=0.5, dual step 0.5
#   HardFlow   N=8 n_inner=40 lr=0.5, mu_init=0 reset each sampling step,
#              dual step 0.2 every 8 inner steps, rho=5 held FIXED (rho_mult=1)
#   HardFlow's lr 0.5 (instead of its released 2.5) and fixed rho (instead of
#   doubling) are stability settings: at lambda=20 the released values diverge
#   at the tighter thresholds. Both are recorded per row.
#
# COST (A5000, recorded): ~43 min per (instance, seed) on the 5-point grid,
# ~11.5 GPU-h for the 8-instance x 2-seed suite; ~20 min per seed for the
# 3-point triple (~0.7 GPU-h).
#
# RESUMABLE. A cell is recorded in $OUT_ROOT/.done.txt only on a zero exit,
# and compare_constrained.py --skip_existing skips any probe or arm that
# already has a run_ok=True row, so an interrupted cell resumes at the arm that
# did not finish. Every setting below can be overridden from the environment.
set -u
usage="usage: run_suite.sh <gpu> [suite|triple] [--dry_run]"
GPU="${1:?$usage}"; shift
WHICH=suite
DRY_RUN="${DRY_RUN:-0}"
for arg in "$@"; do
    case "$arg" in
        suite|triple) WHICH="$arg" ;;
        --dry_run) DRY_RUN=1 ;;
        *) echo "$usage" >&2; exit 2 ;;
    esac
done
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ "$WHICH" = triple ]; then
    JOBS="${JOBS:-$HERE/jobs/triple.tsv}"
    N_GRID="${N_GRID:-3}"                                # m = 1.1 1.3 1.5
    OUT_ROOT="${OUT_ROOT:-results/constrained_triple}"
else
    JOBS="${JOBS:-$HERE/jobs/suite.tsv}"
    N_GRID="${N_GRID:-5}"                                # m = 1.1 1.2 1.3 1.4 1.5
    OUT_ROOT="${OUT_ROOT:-results/constrained_suite}"
fi
case "$JOBS" in /*) ;; *) JOBS="$PWD/$JOBS" ;; esac      # pin before the cd below

SEEDS="${SEEDS:-310 698}"
LAM="${LAM:-20}"
AUTO_LO="${AUTO_LO:-1.1}"
AUTO_HI="${AUTO_HI:-1.5}"
# ConicFlow dual settings; the inner loop is compare_constrained.py's default
# (N=8, H=2, n_inner=24, lr=1e-2, jump surrogate, drift detached).
CF_MU_INIT="${CF_MU_INIT:-0.5}"
CF_LR_DUAL="${CF_LR_DUAL:-0.5}"
# HardFlow: N=8, n_inner=40 (its released value); the rest are the stability settings.
HF_LR="${HF_LR:-0.5}"
HF_LR_DUAL="${HF_LR_DUAL:-0.2}"
HF_MU_INIT="${HF_MU_INIT:-0.0}"
HF_RHO_MULT="${HF_RHO_MULT:-1.0}"
HF_ALPHA="${HF_ALPHA:-0.1}"            # forwarded; inert unless lr*reg would reach 1

cd "$HERE/../../src" || exit 3
DONE="${DONE:-$OUT_ROOT/.done.txt}"
if [ "$DRY_RUN" != 1 ]; then mkdir -p "$(dirname "$DONE")"; touch "$DONE"; fi
echo "[constrained] $WHICH  gpu=$GPU  jobs=$JOBS  out=$OUT_ROOT  seeds=[$SEEDS]  lam=$LAM  grid=$N_GRID pts $AUTO_LO..$AUTO_HI"
while IFS=$'\t' read -r tag prompt edit rest; do
    case "$tag" in ''|'#'*) continue ;; esac
    # The remaining fields are the attribute prompts, one constraint each.
    IFS=$'\t' read -r -a attrs <<< "$rest"
    n=${#attrs[@]}
    [ "$n" -ge 1 ] || { echo "[constrained] $tag: no attribute prompts; skipping"; continue; }
    types=(); dirs=()
    for ((i = 0; i < n; i++)); do types+=(pickscore); dirs+=(ge); done

    for seed in $SEEDS; do
        key="$tag|$seed"
        if [ "$DRY_RUN" != 1 ] && grep -qxF "$key" "$DONE"; then
            echo "[constrained] skip $key (done)"; continue
        fi
        cmd=(python -u compare_constrained.py
             --prompt "$prompt" --edit_prompt "$edit"
             --constraint_types "${types[@]}"
             --constraint_args "${attrs[@]}"
             --constraint_directions "${dirs[@]}"
             --auto_threshold --auto_lo "$AUTO_LO" --auto_hi "$AUTO_HI" --n_grid "$N_GRID"
             --lam "$LAM" --seed "$seed" --gpu "$GPU"
             --mu_init "$CF_MU_INIT" --lr_dual "$CF_LR_DUAL"
             --hf_lr_inner "$HF_LR" --hf_lr_dual "$HF_LR_DUAL" --hf_mu_init "$HF_MU_INIT"
             --hf_rho_mult "$HF_RHO_MULT" --hf_lr_alpha "$HF_ALPHA"
             --skip_existing --quiet
             --out_dir "$OUT_ROOT/${tag}_seed${seed}")
        if [ "$DRY_RUN" = 1 ]; then
            printf '%q ' "${cmd[@]}"; echo; continue
        fi
        echo "########## $(date +%H:%M)  $tag  seed $seed  ($n constraint(s)) ##########"
        t0=$(date +%s)
        "${cmd[@]}"
        rc=$?; t1=$(date +%s)
        if [ "$rc" -eq 0 ]; then
            echo "$key" >> "$DONE"
            echo "########## $(date +%H:%M)  $tag  seed $seed  OK in $(( (t1-t0)/60 )) min ##########"
        else
            echo "########## $(date +%H:%M)  $tag  seed $seed  FAIL rc=$rc after $(( (t1-t0)/60 )) min ##########"
            # Do not spin on an instant failure (bad job line, missing checkpoint).
            [ $((t1-t0)) -lt 60 ] && { echo "[constrained] failed in under a minute; stopping"; exit 1; }
        fi
    done
done < "$JOBS"
[ "$DRY_RUN" = 1 ] && exit 0
echo "[constrained] done; $(wc -l < "$DONE") cells recorded in image/src/$DONE"
[ "$WHICH" = suite ] && echo "[constrained] figure: python plot_constrained_suite.py --root $OUT_ROOT --paper_style"
exit 0
