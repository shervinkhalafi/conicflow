# Running a suite

A "suite" is a bash launcher that runs many single-config `run_sweep.py`
invocations across the available GPUs, unattended. This repo ships one worked
example, `lambda_suite_v2_20260915.sh`, plus the library it is built on,
`gpu_lib.sh`.

```bash
conda activate flowmaps
cd /path/to/fmocc
setsid nohup bash lambda_suite_v2_20260915.sh > results/suite.log 2>&1 &
```

`setsid` matters: it detaches the launcher from your terminal session, so
closing a laptop or dropping SSH does not kill a 6-hour run.

---

## 1. Why a launcher at all, instead of one `run_sweep.py` with many configs

**One config per process, always.** The text encoders are freed after load, so a
prompt that was not encoded at startup cannot be re-encoded in-process. Beyond
that, a fresh process means one bad config cannot poison the rest of a sweep.

So the launcher's whole job is: pick a free GPU, run exactly one config pinned
to it, record the outcome, repeat.

---

## 2. `gpu_lib.sh` — the part you should not re-implement

Every launcher sources it. It is the single source of truth for four things that
each broke at least once when they were copy-pasted between scripts:

**`pick_gpu <need_mib>`** blocks until a card has `need_mib` free for
`STABLE_POLLS` (3) consecutive polls at `POLL_SEC` (30 s), takes a lock, and
echoes the index. The stability requirement exists because a single sample
happily catches a transient dip while another process is between allocations.
Its stdout carries *only* the index — diagnostics go to stderr via `gpu_log`,
because `gpu=$(pick_gpu ...)` is load-bearing.

**The lock** is a directory at a fixed, machine-wide `/tmp/flowmaps_gpu_locks`
containing the holder's pid. Fixed and machine-wide on purpose: two scripts once
made their own timestamped lockdirs, so the mutex excluded runs *within* a script
but not *across* scripts, and they OOMed each other 13 times in a row.
`reap_stale` reclaims a lock whose holder pid is gone.

> **Multi-worker launchers must export `GPU_HOLDER_PID=$BASHPID` in each worker
> before calling `pick_gpu`.** `pick_gpu` runs inside a command substitution,
> which forks — so it cannot observe the calling shell's pid itself. `$$` would
> give the parent (shared by all workers, so one worker's exit trap deletes
> another's lock) and `BASHPID` inside `pick_gpu` would give a subshell that dies
> immediately (so every lock is born stale and gets reaped while in use).

**`run_on_gpu <idx> <logfile> [args...]`** runs `run_sweep.py` pinned to one
card, and enforces two non-negotiables: `CUDA_VISIBLE_DEVICES` pins a single GPU
(anything using DataParallel claims every visible one), and `--flux_device` and
`--reward_device` are both `cuda:0`, i.e. the same card. Cross-GPU copies on the
dev box silently return ~zeros; that is how ImageReward once returned exactly
−0.16733 for 294 consecutive images.

**`need_mib_for <control_type>`** returns the measured memory gate. It `die`s on
an unknown arm rather than substituting another one's number — if you add a
method, measure its gate and add it to `gpu_lib.sh`.

Plus `ledger_init` / `ledger_add` / `ledger_report`, a durable failure ledger
written inside the results dir.

---

## 3. The suite pattern

`lambda_suite_v2_20260915.sh` is the current shape. Its structure:

**A queue file and N workers.** Jobs are lines in `<rundir>/QUEUE.txt`; each
worker pops the head under `flock` and runs it. Work-stealing, so a long job on
one card does not idle the other — which matters here because MPC runs are ~8×
HardFlow's.

**Ordering is a design decision.** The λ suite is prompt-major, then λ-major,
then method, then seed. Prompt-major so prompt 1 is a *complete* figure before
prompt 2 starts; λ-major within it so running out of time costs whole λ points
at one end rather than punching holes in every curve. Choose the order by asking
what a half-finished run should still be good for.

**A deadline.** A job is not *started* if its predicted duration does not fit
before `DEADLINE`; it is logged `DEFERRED` in the ledger, which is distinct from
`FAILED`. Set `BUDGET_MIN` from when you actually need the results.

**A circuit breaker.** If an arm fails on its first seed at the first λ, its
remaining jobs are skipped. Without it, an arm that cannot fit on the card burns
a model load per job for the rest of the run.

**Expected failures are first-class.** The three λ values where HardFlow's guard
raises are launched *once each*, with a single attempt, and recorded as
`EXPECTED_FAIL`. A deterministic `ValueError` does not become true on retry, and
the resulting `FAILED_` dir plus `error_msg` is the evidence for the claim that
the floor exists. They do not trip the circuit breaker.

**`--resume` on every invocation.** Skips configs already recorded `run_ok=True`,
retries failures, appends rather than truncating. Makes a re-launch after an
abort finish the sweep instead of recomputing it.

---

## 4. Writing a new suite

Copy `lambda_suite_v2_20260915.sh` and replace the configuration block and
`arm_extra()`. Keep the scaffolding. The header should say **why** the
experiment is being run and why each hyperparameter is what it is — a grid with
no rationale is not reproducible six weeks later, and these headers are where
the decision history actually lives.

Checklist before launching something long:

- [ ] `bash -n <script>` passes
- [ ] The CLI is valid for every arm. Fastest check without burning a GPU:
      `CUDA_VISIBLE_DEVICES="" python run_sweep.py <args> --out_dir /tmp/x`
      should get all the way to `No CUDA GPUs are available` at model load —
      that means argparse, config building and preflight all passed.
- [ ] `hardflow_reg` recomputed for **this** arm's N and λ (`N/(2λ)`), and
      `lr·reg < 1` for every λ you expect to run
- [ ] `need_mib_for` knows every `control_type` you use
- [ ] `export GPU_HOLDER_PID=$BASHPID` at the top of each worker
- [ ] `BUDGET_MIN` reflects a real deadline
- [ ] Launched with `setsid nohup ... &` so it survives a disconnect

---

## 5. Watching a run

```bash
grep -E "START|DONE|FAILED|DEFER|RAISED" results/suite.log | tail
wc -l < results/<rundir>/QUEUE.txt          # jobs remaining
cat results/<rundir>/FAILED_RUNS.txt        # ledger: failures, defers, skips
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for d in /tmp/flowmaps_gpu_locks/*/; do echo "$d $(cat $d/pid)"; done
```

Two gotchas from experience: `pkill -f <pattern>` will happily match the shell
running the `pkill` itself if the pattern appears in its command line — kill by
explicit pid instead. And a bash script sitting in `sleep 60` will not run its
`trap` until the sleep returns, so a `kill` can appear to do nothing for up to a
minute; use `sleep 60 & wait $!` in any script that needs to die promptly.
