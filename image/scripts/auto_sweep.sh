#!/usr/bin/env bash
# auto_sweep.sh -- range-find lambda, then run the real sweep on that grid.
#
#   auto_sweep.sh <gpu> <reward> "<prompt>" "<edit_prompt>" [extra run_sweep args...]
#
# Step 1 runs lambda_autorange.py at a cheap proxy N (default 4) for conicflow,
# hardflow and fmrg and writes grids.txt with one --sweep_lam line per method.
# Step 2 runs run_sweep.py once per method at TARGET_N on exactly that grid.
# Everything lands under results/auto/<reward>/<slug>/.
#
# Environment overrides (all optional):
#   TARGET_N=10  PROBE_N=4  N_GRID=6  SEEDS="42"  CONICFLOW_H=2  CONICFLOW_NI=12
#   AUTO_ROOT=results/auto        root of the output tree (see below)
#   REDO_AUTORANGE=1              re-probe even if grids.txt already exists
#   HF_NI=40 HF_LR=2.5  FMRG_NOPT=3  AUTORANGE_EXTRA="..."
#   Per-method knobs take LISTS: CONICFLOW_NI="12 24" sweeps ConicFlow at both inner counts
#   (cartesian with the lambda grid, in the same out_dir). The diagnostics and
#   compare_methods.py treat each setting as its own config of that method.
#   SHARED_GRID=1   run every method on ONE grid (the `shared` line of grids.txt)
#                   so objectives can be compared between methods at the same lam.
#                   At a shared lam all three solve the same first-order problem
#                   (see lambda_autorange.shared_grid); per-method grids resolve
#                   each method's own transition better but are not comparable.
#   HIGH_STOP=break  range-find up to where the IMAGE breaks instead of stopping
#                   at the reward plateau, and extend the grid above the knee
#                   (lambda_autorange.py --high_stop break). Use this when the
#                   sweep wants the aggressive end of the frontier -- the default
#                   stops where reward stops paying, which is 1-2 decades below
#                   where the edit stops being usable. NOTE it makes the grid
#                   LONGER (n_grid + n_high, so 8-9 points instead of 6), i.e.
#                   ~50% more runs per method-seed, and it writes one image per
#                   evaluated lambda to <out>/autorange/images/ -- look at those
#                   before trusting the top of the grid, since the break
#                   thresholds are calibrated on a single cell.
#   HF_ALPHA=0.5    lift HardFlow's lr*reg >= 1 floor (lr = HF_ALPHA/reg below it,
#                   passed to both the range finder and the hardflow sweep).
#                   Without it HardFlow cannot reach the shared grid's low end.
#   DIAG="--diag_winrates --diag_pareto"
#                   after ALL methods have run, compare them across every out_dir
#                   at once. Pass sweep_diagnostics.py's flags verbatim (one per
#                   diagnostic, so new ones need no change here); DIAG="--all"
#                   runs every diagnostic there is. Run at the end rather than
#                   per-sweep because each method writes its own out_dir, so no
#                   single run_sweep.py invocation ever sees all three -- the
#                   cross-method comparison only exists once they are all done.
#                   --diag_winrates additionally needs SHARED_GRID=1, since the
#                   objective is only comparable at equal lambda.
#
# The card is pinned with CUDA_VISIBLE_DEVICES and both --*_device are cuda:0
# in-process (never pass cuda:1 to run_sweep.py: ImageReward misbehaves there).
set -u
GPU="$1"; REWARD="$2"; PROMPT="$3"; EDIT="$4"; shift 4
TARGET_N="${TARGET_N:-10}"; PROBE_N="${PROBE_N:-4}"; N_GRID="${N_GRID:-6}"; SEEDS="${SEEDS:-42}"
CONICFLOW_H="${CONICFLOW_H:-2}"; CONICFLOW_NI="${CONICFLOW_NI:-12}"; HF_NI="${HF_NI:-40}"; HF_LR="${HF_LR:-2.5}"; FMRG_NOPT="${FMRG_NOPT:-3}"
SHARED_GRID="${SHARED_GRID:-0}"; HF_ALPHA="${HF_ALPHA:-}"
DIAG="${DIAG:-}"
HF_ALPHA_ARGS=(); [ -n "$HF_ALPHA" ] && HF_ALPHA_ARGS=(--hardflow_lr_alpha "$HF_ALPHA")
# (2026-09-23) Only the range finder takes it; the sweep just runs whatever grid
# grids.txt ends up holding, so there is nothing to pass on to run_sweep.py.
HIGH_STOP_ARGS=(); [ -n "${HIGH_STOP:-}" ] && HIGH_STOP_ARGS=(--high_stop "$HIGH_STOP")
# Everything below runs from src/, where the Python lives. Output paths are
# therefore relative to src/ unless AUTO_ROOT is absolute.
cd "$(dirname "$0")/../src" || exit 3
SLUG="$(printf '%s__%s' "$PROMPT" "$EDIT" | tr -c 'A-Za-z0-9' '_' | cut -c1-60)"
# AUTO_ROOT (2026-09-22) so a sweep can be kept in its own tree instead of
# landing in results/auto/ with everything else. paper_figures.py already takes
# the root as a parameter (collect_paper_figures(auto=...)), so a sibling tree
# with the same <reward>/<slug>/<method>_seed<N>/ layout is read by every
# downstream tool without further change. Default is unchanged.
OUT="${AUTO_ROOT:-results/auto}/$REWARD/$SLUG"; mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU"

# Reuse an existing grids.txt (2026-09-22). Range finding costs ~14 min and
# depends only on (reward, prompt, edit_prompt) -- never on the seed. The loop
# below is method-outer/seed-inner, so a run killed partway leaves some seeds
# with ConicFlow but no HardFlow, which is exactly the comparison the cell exists to
# make. Calling this script once per seed instead makes every cutoff land on a
# seed boundary, but that re-ran range finding per seed. Caching it makes the
# per-seed call cheap, so the safe ordering is also the affordable one.
# REDO_AUTORANGE=1 forces a fresh probe.
if [ -s "$OUT/autorange/grids.txt" ] && [ "${REDO_AUTORANGE:-0}" != "1" ]; then
    echo "[auto_sweep] step 1/2: reusing cached grid ($OUT/autorange/grids.txt)"
else
echo "[auto_sweep] step 1/2: lambda range finding ($REWARD, proxy N=$PROBE_N -> target N=$TARGET_N)"
python lambda_autorange.py --reward "$REWARD" --prompt "$PROMPT" --edit_prompt "$EDIT" \
    --methods conicflow hardflow fmrg --probe_N "$PROBE_N" --target_N "$TARGET_N" --n_grid "$N_GRID" \
    --flux_device cuda:0 --reward_device cuda:0 --out_dir "$OUT/autorange" ${AUTORANGE_EXTRA:-} \
    ${HF_ALPHA_ARGS[@]+"${HF_ALPHA_ARGS[@]}"} ${HIGH_STOP_ARGS[@]+"${HIGH_STOP_ARGS[@]}"} \
    || { echo "[auto_sweep] range finding FAILED"; exit 1; }

# (2026-09-23) Trim the emitted grid to where the measured curve stops moving.
# --high_stop break has no upward stop for ConicFlow: ConicFlow's plateau is flat, not
# destructive, so neither the image-break test nor the reward-collapse test ever
# fires and the walk runs to the lambda box.  On the first live break-mode cell
# (garden, seed 811) that put five of ConicFlow's eight points above lam=6300, all
# re-measuring one plateau point at 239 s each -- half the ConicFlow budget per cell.
# trim_grids.py cuts on the probe curve only, so a method still climbing at its
# grid top is left alone (HardFlow and FMRG were, on that cell).  It is NOT the
# near_ceiling stop coming back; see its module docstring.  Runs only on a fresh
# probe -- a cached grids.txt was already trimmed when it was written.
python trim_grids.py "$OUT/autorange" --apply \
    || echo "[auto_sweep] grid trim failed; sweeping the range finder's raw grid"
fi

echo "[auto_sweep] step 2/2: sweeps at N=$TARGET_N (SHARED_GRID=$SHARED_GRID)"
if [ "$SHARED_GRID" = "1" ]; then
    SHARED="$(awk '$1=="shared" && $2=="--sweep_lam" {$1="";$2="";print}' "$OUT/autorange/grids.txt")"
    [ -n "$SHARED" ] || { echo "[auto_sweep] no shared grid in grids.txt"; exit 1; }
    echo "[auto_sweep]   shared grid: $SHARED"
    grep '^ *# ' "$OUT/autorange/grids.txt" | tail -3 | sed 's/^/[auto_sweep]   /'
    # one line per method, all on the shared grid
    METHOD_LINES="$(printf 'conicflow --sweep_lam %s\nhardflow --sweep_lam %s\nfmrg --sweep_lam %s\n' "$SHARED" "$SHARED" "$SHARED")"
else
    METHOD_LINES="$(grep -E '^(conicflow|hardflow|fmrg) +--sweep_lam' "$OUT/autorange/grids.txt")"
fi
while read -r method flag grid; do
    [ "$flag" = "--sweep_lam" ] || continue
    grid="${grid%%#*}"                      # strip the trailing '# ...' note
    # Unquoted on purpose: each knob may be a space-separated list, and
    # run_sweep.py's --sweep_* flags take several values.
    # shellcheck disable=SC2206
    case "$method" in
        conicflow)      extra=(--sweep_H $CONICFLOW_H --sweep_n_inner $CONICFLOW_NI --sweep_lr_inner 1e-2) ;;
        hardflow) extra=(--sweep_H 2 --sweep_n_inner $HF_NI --sweep_lr_inner $HF_LR ${HF_ALPHA_ARGS[@]+"${HF_ALPHA_ARGS[@]}"}) ;;
        fmrg)     extra=(--sweep_n_inner $FMRG_NOPT) ;;
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
done <<< "$METHOD_LINES"
if [ -n "$DIAG" ]; then
    echo "[auto_sweep] diagnostics ($DIAG) across every method's out_dir"
    case "$DIAG" in
        *winrates*|*--all*)
            if [ "$SHARED_GRID" != "1" ]; then
                echo "[auto_sweep]   NOTE: SHARED_GRID=1 is not set, so the methods ran on per-method"
                echo "[auto_sweep]   lambda grids and no (lambda, seed) combo will contain all three."
                echo "[auto_sweep]   Win rates will report zero combos; other diagnostics are unaffected."
            fi ;;
    esac
    # shellcheck disable=SC2086 -- DIAG is a list of flags on purpose
    python sweep_diagnostics.py --runs "$OUT"/*_seed*/ $DIAG --out_dir "$OUT/diagnostics" \
        || echo "[auto_sweep]   diagnostics failed; the sweeps themselves are complete"
fi
echo "[auto_sweep] done -> $OUT"
