# shellcheck shell=bash
# ---------------------------------------------------------------------------
# gpu_lib.sh -- the ONE place GPU acquisition and memory gates are defined.
#
# Source this from every new sweep script:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/gpu_lib.sh"
#
# WHY THIS FILE EXISTS. Every launcher in this directory grew its own copy of
# "wait for a GPU, then launch", and the copies disagreed. The 2026-09-04
# post-mortem (the GPU-safety post-mortem) traced ~1.25 h of wall clock and
# zero completed runs to two independent bugs in two of those copies, and the
# 2026-09-05 memory work then invalidated the memory constant that most of them
# had hardcoded. Three generations currently coexist:
#
#   Gen 1  fill_grid_sweep.sh, hp_grid_sweep_v2.sh, push_ladder.sh
#          -- no GPU gate at all.
#   Gen 2  overnight_sweep.sh's wait_for_gpu0
#          -- waits 15 min, then prints "proceeding anyway" and launches into a
#             full card. Do not copy it.
#   Gen 3  hardflow_*.sh, lpips_reward_2c_sweep.sh
#          -- the mature pattern, which is what this file is extracted from.
#
# Those scripts are records of experiments that already ran and are NOT being
# retrofitted. This file is what new work uses.
#
# EVERY RULE BELOW WAS PAID FOR. Do not "simplify" one without reading its note.
# ---------------------------------------------------------------------------

# ── Memory gates ────────────────────────────────────────────────────────────
# MEASURED, never guessed. Re-derive with a real run and update HERE, not in a
# copy.
#
# UNITS -- READ THIS BEFORE EDITING. summary.csv's peak_reserved_flux_gb is
# `torch.cuda.max_memory_reserved() / 1e9`, i.e. DECIMAL GB. free_mib() compares
# against nvidia-smi MiB (2^20 bytes). So:
#
#     MiB = GB * 1e9 / 1048576 = GB * 953.674          NOT  GB * 1024
#
# The pre-2026-09-05 constants used GB*1024, over-stating every gate by 7.4%.
# That was conservative, so it never caused an OOM -- but it is why the numbers
# below look smaller than the ones they replace by more than the real saving.
#
# WHAT CHANGED. The old NEED_MIB_CONICFLOW=24100 was never the RUN requirement.
# FluxFlowMapSampler used to move the full bf16 transformer (23.8 GB) onto the
# card and only then quantize it to 12.26 GB -- a 24.21 GB LOAD spike, ~3 GB
# above the peak of the run that followed. The gate was therefore sized by the
# load, and ConicFlow appeared to need an effectively empty card. It does not, and has
# not since 2026-09-05 (the memory-budget notes). Load now peaks at
# 12.33 GB, so the run genuinely binds for the first time.
#
# MEASURED 2026-09-05, both arms on the SAME worst-case config
# (--reward lpips, constraints clip + imagereward -- the heaviest combination on
# disk, 3 resident scorers), peak_reserved_flux_gb:
#
#     arm  config                        was       now       now MiB   gate
#     conicflow  lpips + clip + imagereward    23.23 GB  21.29 GB  20304     21000
#     hf   lpips + clip + imagereward    20.63 GB  18.83 GB  17958     18500
#
# For reference, ConicFlow's COMMON config (--reward none, clip + lpips) is lighter
# still at 18.52 GB = 17662 MiB. The gates below deliberately use the WORST
# config per arm, because a gate that clears a card the heaviest config cannot
# fit is worse than useless.
#
# Note ConicFlow > HardFlow now. That ordering is real: ConicFlow holds H-step horizon
# graphs plus the endpoint rollout, HardFlow does not. The old "HardFlow is the
# lighter arm" gap was mostly the shared load spike, not the algorithms.
#
# Margin is +3% over measured, rounded up to the next 100 MiB.
#
# Do NOT lower these to "get runs through". A launch that cannot fit does not
# fail fast; it wastes a ~90 s model load and one attempt.
: "${NEED_MIB_CONICFLOW:=21000}"
: "${NEED_MIB_HF:=18500}"

# FMRG added 2026-09-14 for the unconstrained wall-clock probe. need_mib_for
# used to gpu_die on this arm, which is correct behaviour for an unmeasured one
# -- so it is measured here rather than borrowed from ConicFlow.
#
# MEASURED from the only fmrg rows on disk (results/*/summary.csv,
# control_type in {fmrg, boffi_fmrg}, 4 rows, all --reward clip, N=10,
# n_inner=5, unconstrained): peak_reserved_flux_gb = 20.61 / 22.02 / 22.11 /
# 22.12 GB. Worst = 22.12 GB = 21095 MiB; +3% and rounded up:
: "${NEED_MIB_FMRG:=21800}"
#
# TWO CAVEATS, because this gate is weaker evidence than the other two:
#  1. Those rows used CLIP as the scorer. ImageReward is a heavier model, so a
#     --reward imagereward fmrg run may peak above 22.12 GB. 21800 MiB of 24564
#     already means "an effectively empty card", so there is no headroom left to
#     add on a 24 GB card -- if an fmrg+imagereward run OOMs, that is the finding,
#     not a gate to lower.
#  2. n_inner was 5. The probe runs n_inner=46. FMRG's inner loop should not
#     accumulate graph across inner steps, so this is expected to be flat in
#     n_inner -- but it is an expectation, not a measurement.
# Re-derive from the probe's own peak_reserved_flux_gb and update HERE.

# Unconstrained baselines gg / flowchef, added 2026-10-09 (see
# run_sweep.UNCONSTRAINED_BASELINE_TYPES). MEASURED in
# results/baselines_smoke_20261009 (N=8, 256^2, int8, one card, per-run logs'
# "peak GPU mem" line), each arm on all three paper rewards:
#
#                 pickscore   clip      imagereward
#     gg          16.29 GB    18.77 GB  15.38 GB
#     flowchef    16.23 GB    18.77 GB  15.10 GB
#
# Worst config is CLIP for both, and it is the same 18.77 GB every other arm
# shows with CLIP -- scorer-dominated, not method-dominated (each method holds
# at most one transformer graph at a time). 18.77 GB = 17901 MiB; +3% and
# rounded up: 18500 MiB for each. Peak is flat in n_inner (no graph is carried
# across guidance steps), so the n_inner=1 CLIP rows hold.
: "${NEED_MIB_GG:=18500}"
: "${NEED_MIB_FLOWCHEF:=18500}"

# Constrained baselines pflow / ldf, added 2026-10-11. MEASURED in
# results/cb_smoke_20261011 (N=8, 256^2, int8), PickScore constraints, peak by
# number of constraints (the constraint heads dominate, as for every arm):
#
#                 1          2          3
#     pflow       16.37 GB   19.82 GB   22.78 GB
#     ldf         16.23 GB   18.96 GB   21.92 GB
#
# Worst = three constraints: pflow 22.78 GB = 21725 MiB -> +3%, rounded up:
# 22400; ldf 21.92 GB = 20905 MiB -> 21600. pflow sits ~0.6 GB above the
# others because its projection holds one transformer graph while it takes a
# backward per active constraint (retain_graph). Flat in max_iters / substeps.
: "${NEED_MIB_PFLOW:=22400}"
: "${NEED_MIB_LDF:=21600}"

# ── Polling / retry ─────────────────────────────────────────────────────────
# STABLE_POLLS: a single free-memory sample happily grabs a transient dip while
# some other process is between allocations. FLUX spends ~90 s loading, so the
# card has to look free for a while, not for an instant.
: "${STABLE_POLLS:=3}"
: "${POLL_SEC:=30}"
# Attempts are consumed only by LOST RACES -- the waiting itself happens inside
# pick_gpu, which blocks. A low cap therefore turns transient contention into a
# dead sweep: the 2026-09-04 22:12 launch gave up on its first run after 8
# consecutive OOMs in 28 minutes.
: "${MAX_ATTEMPTS:=40}"
: "${BACKOFF:=240}"

# FIXED and machine-wide. NEVER timestamp this. Two scripts on 2026-09-04 each
# made their own timestamped lockdir, so the mutex excluded runs *within* a
# script but not *across* scripts: both polled the same GPU, both saw it free in
# the same second, both launched, both OOMed, 13-14 times in a row.
: "${LOCKDIR:=/tmp/flowmaps_gpu_locks}"
mkdir -p "$LOCKDIR"

# run_sweep.py sets PYTORCH_CUDA_ALLOC_CONF itself as of 2026-09-05 (before
# `import torch`, which is the only point where it takes effect). Exported here
# too so anything else launched from a sourcing script gets it.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"


# ── Logging ─────────────────────────────────────────────────────────────────
# Writes to stderr, not stdout. pick_gpu's stdout is load-bearing (see below),
# and a log() built on `tee` to stdout is exactly what corrupted
# CUDA_VISIBLE_DEVICES once already.
gpu_log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }

gpu_die() { gpu_log "FATAL: $*"; exit 1; }


# ── GPU acquisition ─────────────────────────────────────────────────────────
free_mib() {
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" 2>/dev/null \
    | awk -F, '{gsub(/ /,"");print $2-$1}'
}

# Reclaim a lock whose holder is gone. Without this, one killed run deadlocks
# every other script on the box forever.
reap_stale() {
  local idx="$1" pid
  [ -d "${LOCKDIR}/${idx}" ] || return 0
  pid=$(cat "${LOCKDIR}/${idx}/pid" 2>/dev/null)
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    gpu_log "  reclaiming stale gpu${idx} lock (holder pid=${pid:-none} is gone)"
    rm -rf "${LOCKDIR}/${idx}"
  fi
}

# pick_gpu <need_mib>  -- blocks until a card has `need_mib` free for
# STABLE_POLLS consecutive polls, takes the lock, and echoes the index.
#
# STDOUT CARRIES ONLY THE INDEX. Callers do `gpu=$(pick_gpu ...)` and then
# `CUDA_VISIBLE_DEVICES=$gpu`. One stray echo here and that variable becomes a
# log line. All diagnostics go through gpu_log, which writes to stderr.
pick_gpu() {
  local need="$1"
  [ -n "$need" ] || { gpu_log "pick_gpu: need_mib is required"; return 1; }
  local waited=0
  declare -A streak
  while true; do
    local idx
    for idx in $(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' '); do
      reap_stale "$idx"
      local f; f=$(free_mib "$idx")
      if [ "${f:-0}" -ge "$need" ]; then
        streak[$idx]=$(( ${streak[$idx]:-0} + 1 ))
      else
        streak[$idx]=0
      fi
      if [ "${streak[$idx]:-0}" -ge "$STABLE_POLLS" ] && mkdir "${LOCKDIR}/${idx}" 2>/dev/null; then
        # THE HOLDER PID MUST OUTLIVE THE RUN, or reap_stale (above) deletes
        # this lock on the next poll while the card is still in use, and the
        # mutex silently degrades to "memory gate only" -- the 2026-09-04
        # double-launch failure mode this file exists to prevent.
        #
        # Two pids that look right and are NOT:
        #   $$       in a script that backgrounds several worker shells, $$ is
        #            the PARENT's pid in every worker, so all workers stamp the
        #            same holder and the first to exit has its install_gpu_trap
        #            delete the others' locks.
        #   BASHPID  pick_gpu is always called as `gpu=$(pick_gpu "$need")`, and
        #            command substitution forks: BASHPID here is that subshell,
        #            which dies the instant pick_gpu returns. Every lock is then
        #            born with a dead holder. (Tried 2026-09-14; caught live by
        #            lock_pid_watchdog.sh, which exists only to paper over a
        #            run that had already sourced this mistake.)
        #
        # So the caller supplies it, BEFORE the command substitution forks:
        #   worker() { export GPU_HOLDER_PID=$BASHPID; ...; gpu=$(pick_gpu ...) }
        # Single-process scripts need no change: $$ is the script itself, which
        # is exactly the right lifetime.
        echo "${GPU_HOLDER_PID:-$$}" > "${LOCKDIR}/${idx}/pid"
        echo "$idx"
        return 0
      fi
    done
    if [ $((waited % 600)) -eq 0 ]; then
      gpu_log "  (waiting for ${need} MiB: gpu0=$(free_mib 0) gpu1=$(free_mib 1); ${waited}s)"
    fi
    sleep "$POLL_SEC"
    waited=$((waited+POLL_SEC))
  done
}

release_gpu() { rm -rf "${LOCKDIR}/${1}" 2>/dev/null; }

# Call once, near the top of a sourcing script, so a Ctrl-C or a kill does not
# leave our locks behind for reap_stale to clean up later. In a multi-worker
# script, call it INSIDE each worker shell (after exporting GPU_HOLDER_PID, see
# pick_gpu) so a worker only ever reclaims the locks it took itself.
install_gpu_trap() {
  trap 'for _i in $(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d " "); do
          [ "$(cat "${LOCKDIR}/${_i}/pid" 2>/dev/null)" = "${GPU_HOLDER_PID:-$$}" ] && rm -rf "${LOCKDIR}/${_i}";
        done' EXIT
}


# ── Launching ───────────────────────────────────────────────────────────────
# run_on_gpu <gpu_idx> <logfile> [args...]  -- run run_sweep.py pinned to ONE card.
#
# Two non-negotiables are enforced here rather than left to each caller:
#
#  1. CUDA_VISIBLE_DEVICES pins the process to a single card, which is then
#     cuda:0 from inside. Anything using DataParallel claims every VISIBLE GPU.
#  2. --flux_device and --reward_device are BOTH cuda:0, i.e. the same card.
#     Cross-GPU copies on the development machine silently return approximately zeros with no
#     error raised (the GPU-safety post-mortem B1). --reward_device defaults to
#     "auto", which resolves to the OTHER card whenever two are visible, and
#     that is how ImageReward came back as exactly -0.16733 for 294 images.
#     Never pass --reward_device auto here.
#
# Pass exactly ONE config per call (no multi-valued --sweep_* flags): a new
# prompt cannot be re-encoded in-process, because the text encoders are freed
# after load.
run_on_gpu() {
  local gpu="$1" logfile="$2"; shift 2
  CUDA_VISIBLE_DEVICES="$gpu" python run_sweep.py \
    --flux_device cuda:0 --reward_device cuda:0 \
    "$@" >> "$logfile" 2>&1
}

# need_mib_for <control_type>  -- the gate for an arm, from the constants above.
need_mib_for() {
  case "$1" in
    conicflow)       echo "$NEED_MIB_CONICFLOW" ;;
    hardflow)  echo "$NEED_MIB_HF" ;;
    fmrg)      echo "$NEED_MIB_FMRG" ;;
    gg)        echo "$NEED_MIB_GG" ;;        # 2026-10-09, measured above
    flowchef)  echo "$NEED_MIB_FLOWCHEF" ;;  # 2026-10-09, measured above
    pflow)     echo "$NEED_MIB_PFLOW" ;;     # 2026-10-11, measured above
    ldf)       echo "$NEED_MIB_LDF" ;;       # 2026-10-11, measured above
    *) gpu_die "need_mib_for: no measured gate for control_type '$1'. Measure it
       (run once on an empty card, read peak_reserved_flux_gb from summary.csv)
       and add it to gpu_lib.sh -- do not substitute another arm's number." ;;
  esac
}

# ── Failure ledger ──────────────────────────────────────────────────────────
# Durable and inside the results dir. fill_grid_sweep.sh wrote its markers into
# a /tmp scratchpad that does not survive; only run_phase2_controllability.sh
# kept a real ledger, and only it re-printed one at exit.
ledger_init() { FAILED_LEDGER="$1/FAILED_RUNS.txt"; : > "$FAILED_LEDGER"; }
ledger_add()  { echo "$*" >> "${FAILED_LEDGER:?ledger_init not called}"; }
ledger_report() {
  local n; n=$(wc -l < "${FAILED_LEDGER:-/dev/null}" 2>/dev/null || echo 0)
  if [ "${n:-0}" -gt 0 ]; then
    gpu_log "!! ${n} failed run(s):"; cat "$FAILED_LEDGER" >&2; return 1
  fi
  gpu_log "no failures recorded."; return 0
}

# ── Preflight ───────────────────────────────────────────────────────────────
gpu_preflight() {
  command -v nvidia-smi >/dev/null || gpu_die "nvidia-smi not found."
  python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null \
    || gpu_die "torch cannot see a GPU. Did you 'conda activate flowmaps'?"
  python -m py_compile run_sweep.py || gpu_die "run_sweep.py does not compile."
  gpu_log "preflight OK. GPU state:"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
             --format=csv,noheader >&2
}
