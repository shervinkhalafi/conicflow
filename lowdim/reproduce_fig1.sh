#!/usr/bin/env bash
# Re-run every sweep behind paper Figure 1 from scratch, then draw it.
#
#   bash reproduce_fig1.sh                 # writes results_fig1/, then figures/
#   DEVICE=cpu bash reproduce_fig1.sh      # force CPU (default: cuda if available)
#   RESULTS=my_dir bash reproduce_fig1.sh
#   DRY_RUN=1 bash reproduce_fig1.sh       # list every configuration, run nothing
#
# Each block below regenerates exactly the configurations in the matching
# paper_data/<problem>/<part>/sweep_results.csv (same lambda grids, same inner
# grids), so `python make_fig1.py --data results_fig1` draws the figure from
# the fresh runs the same way it draws it from the shipped ones. run_sweep.py
# appends to an existing CSV, so start from an empty RESULTS dir.
#
# Every run in a problem shares one batch of 512 initial particles drawn with
# seed 42. That draw happens on the run device, so CPU and GPU runs start from
# different particles: the paper's runs were on a GPU, and a CPU rerun matches
# them statistically (to about 0.5% in mean reward), not digit for digit.
#
# Some parts include configurations the figure does not plot (ConicFlow at
# H = 1, 2, 16); they are kept so that each part reproduces its paper_data
# counterpart one to one. Cost: 1,210 controller runs, about 100 min of
# controller time on the GPU used for the paper and about 3 h on a CPU.
set -euo pipefail
cd "$(dirname "$0")"

RESULTS=${RESULTS:-results_fig1}
CKPT=${CKPT:-model/two_moons_lsd_15k.npz}
DEV_ARGS=()
[[ -n "${DEVICE:-}" ]] && DEV_ARGS=(--device "$DEVICE")
[[ -n "${DRY_RUN:-}" ]] && DEV_ARGS+=(--dry_run)

LAM_A="0.1 0.25 0.5 0.75 1.0 2.0 3.0"
LAM_B="$LAM_A 5.0 10.0 20.0 40.0 70.0 100.0"
LAM_HI="5.0 10.0 20.0 40.0 70.0 100.0"

sweep() {   # sweep <reward_type> <part dir> <run_sweep args...>
  local reward=$1 part=$2; shift 2
  python run_sweep.py --ckpt "$CKPT" "${DEV_ARGS[@]}" --N 16 --no_constraint --no_plots \
    --reward_type "$reward" --output_dir "$RESULTS/$part" "$@"
}

HF_ADAM=(--methods hardflow --hf_optimizer adam  --hf_lr 0.01 0.05)
HF_LBFGS=(--methods hardflow --hf_optimizer lbfgs --hf_lr 0.1 0.5)
FMRG=(--methods fmrg --fmrg_n_iters 1 3 5 10 --fmrg_variant J E)
CF=(--methods mpc --mpc_optimizer adam)          # ConicFlow is `mpc` in run_sweep.py

# ── upper-moon reward (left panel) ───────────────────────────────────────────
U=upper_moon
sweep upper_moon $U/base "${HF_ADAM[@]}"  --hf_n_outer 3 5 10 --hf_n_inner 10 20 --lam $LAM_A
sweep upper_moon $U/base "${HF_LBFGS[@]}" --hf_n_outer 3 5 10 --hf_n_inner 10 20 --lam $LAM_A
sweep upper_moon $U/base "${CF[@]}" --H 4 --constraint_on endpoint horizon \
      --mpc_n_iters 25 50 --mpc_lr 0.01 0.05 --lam $LAM_A
sweep upper_moon $U/fmrg "${FMRG[@]}" --lam $LAM_A
sweep upper_moon $U/conicflow_short_h "${CF[@]}" --H 1 2 --constraint_on endpoint \
      --mpc_n_iters 25 50 --mpc_lr 0.01 0.05 --lam $LAM_A
# lambda = 5, 10 for every method, and the H = 8 (and 16) curves over the whole grid
sweep upper_moon $U/lam5_10_long_h "${FMRG[@]}" --lam 5.0 10.0
sweep upper_moon $U/lam5_10_long_h "${HF_ADAM[@]}"  --hf_n_outer 10 --hf_n_inner 10 --lam 5.0 10.0
sweep upper_moon $U/lam5_10_long_h "${HF_LBFGS[@]}" --hf_n_outer 10 --hf_n_inner 10 --lam 5.0 10.0
sweep upper_moon $U/lam5_10_long_h "${CF[@]}" --H 1 2 --constraint_on endpoint \
      --mpc_n_iters 50 --mpc_lr 0.01 0.05 --lam 5.0 10.0
sweep upper_moon $U/lam5_10_long_h "${CF[@]}" --H 4 --constraint_on endpoint horizon \
      --mpc_n_iters 50 --mpc_lr 0.01 0.05 --lam 5.0 10.0
sweep upper_moon $U/lam5_10_long_h "${CF[@]}" --H 8 16 --constraint_on horizon \
      --mpc_n_iters 50 --mpc_lr 0.01 0.05 --lam $LAM_A 5.0 10.0

# ── off-manifold reward (middle panel) ───────────────────────────────────────
O=off_manifold
sweep off_top_ball $O/base "${HF_ADAM[@]}"  --hf_n_outer 3 5 10 --hf_n_inner 10 20 --lam $LAM_B
sweep off_top_ball $O/base "${HF_LBFGS[@]}" --hf_n_outer 3 5 10 --hf_n_inner 10 20 --lam $LAM_B
sweep off_top_ball $O/base "${CF[@]}" --H 1 2 4 --constraint_on endpoint \
      --mpc_n_iters 25 50 --mpc_lr 0.01 0.05 --lam $LAM_B
sweep off_top_ball $O/base "${CF[@]}" --H 4 --constraint_on horizon \
      --mpc_n_iters 25 50 --mpc_lr 0.01 0.05 --lam $LAM_B
sweep off_top_ball $O/base "${CF[@]}" --H 4 --constraint_on horizon \
      --mpc_n_iters 100 200 --mpc_lr 0.001 0.005 0.01 --lam $LAM_HI
sweep off_top_ball $O/fmrg "${FMRG[@]}" --lam $LAM_A
sweep off_top_ball $O/fmrg_hi_lam "${FMRG[@]}" --lam $LAM_HI
# ConicFlow keeps improving past lambda = 100: extend it until the reward saturates
sweep off_top_ball $O/extended "${CF[@]}" --H 4 --constraint_on endpoint horizon \
      --mpc_n_iters 50 200 --mpc_lr 0.05 --lam 200.0 500.0 1000.0 2000.0
sweep off_top_ball $O/extended "${CF[@]}" --H 8 16 --constraint_on horizon \
      --mpc_n_iters 100 --mpc_lr 0.05 --lam 10.0 40.0 100.0 500.0 2000.0
sweep off_top_ball $O/saturation "${CF[@]}" --H 4 --constraint_on endpoint horizon \
      --mpc_n_iters 50 200 --mpc_lr 0.05 0.2 --lam 500.0 2000.0 5000.0 10000.0 20000.0 50000.0 100000.0
sweep off_top_ball $O/saturation "${CF[@]}" --H 8 16 --constraint_on horizon \
      --mpc_n_iters 50 200 --mpc_lr 0.05 0.2 --lam 500.0 2000.0 5000.0 10000.0 20000.0 50000.0 100000.0

[[ -n "${DRY_RUN:-}" ]] && exit 0
python make_fig1.py --data "$RESULTS" --out figures
