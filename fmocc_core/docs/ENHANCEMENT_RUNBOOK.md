# Runbook: unconstrained aesthetic-enhancement sweep (1× H200 studio)

**Audience: Claude Code running in the Lightning studio.** Follow this top to
bottom. It is a runbook, not a suggestion: the hyperparameters are frozen to
match an earlier run, and changing one silently destroys the comparison this
sweep exists to make.

Expected wall clock: **4–5 hours** at 4 workers on one H200.

---

## 0. What this run is, and why it is shaped this way

The paper compares three methods — MPC (ours), HardFlow, FMRG — on two image
tasks:

- **editing**: the reward is scored against a *different* prompt than the one
  that generated the image. Already done, lives in `results/auto/`.
- **aesthetic enhancement**: the reward is scored against the *same* prompt.
  There is no target attribute to install; control can only make the image a
  better realization of the prompt it already depicts. **This is the run.**

Enhancement therefore means passing the same string as both `<prompt>` and
`<edit_prompt>`. That is the only structural difference from the editing run.

A prompt screen already ran (`results/screen2/`, FMRG-only at N=4) and was used
to pick the six prompts in §2. **Do not reuse the screen's lambda values** — it
was N=4 and FMRG-only, and the grid is re-derived per method at N=8 in step 1.

### The one rule that matters most

Every hyperparameter in §3 is copied from `main0.sh`, the launcher that
produced the editing results. If enhancement runs at a different `N`, horizon,
`n_inner`, or grid policy, the two tasks cannot be compared and the run is
wasted. **Do not tune anything to make a number look better.** If something
seems wrong, stop and report rather than adjusting — that is the standing rule
in `CLAUDE.md` ("loud fail only") and it applies here.

---

## 1. Preconditions — verify all of these before launching

```bash
R="${FMOCC_ROOT:-$HOME/fmocc}"
cd "$R/fmocc_core" || exit 3

# a. the tooling exists
ls auto_sweep.sh lambda_autorange.py run_sweep.py sweep_diagnostics.py

# b. GPU is free and is the H200
nvidia-smi

# c. python env
ls "$R/.venv/bin/python"
```

**d. Branch check.** The editing baseline was produced from `main`. If this
checkout is on another branch (e.g. `adaptive-dual-20260921`), confirm nothing
on the control path differs:

```bash
git log --oneline main..HEAD -- fmocc_core/run_sweep.py fmocc_core/image_mpc_edited_sung.py
```

Empty output → safe to proceed on this branch. Any commits → `git checkout main`
for this run and report what those commits were. (`dual_adaptive` alone cannot
affect an unconstrained run — there are no multipliers — but any other change to
the MPC/HardFlow/FMRG path breaks comparability.)

**e. Nothing else is using the card.** In particular the constrained batch
(`crun2.sh` / `queue2.sh`) must not be running. Four enhancement workers plus
four constrained workers make every timing number meaningless. If `crun2` is
mid-queue, **stop and report** — it is the higher priority.

---

## 2. The six prompts — final, do not edit

Three are FMRG Fig. 12 verbatim (`f*`), three are original (`o*`). The split is
deliberate: if only the literature prompts respond, the result is a replicated
cherry-pick rather than a property of the method, and that is only visible if
both families are present in equal number.

```
Abstract human form, impressionist strokes, pastel colors
Close-up of an eye with the Earth inside the pupil
A Japanese garden at dawn, ink illustration, sumi-e
Fractured geometric landscape, cubist, muted ochre and teal
A Victorian greenhouse at midnight: brass lanterns, overgrown ferns, and a sleeping fox, watercolor
Portrait of an elderly fisherman, charcoal sketch, harsh side lighting
```

Copy them byte-for-byte. Note the third contains `sumi-e`, the fifth contains a
colon and several commas. None contains a quote or a backslash, which is what
makes the plain `xargs -I{}` in §4 safe.

**Reward: `pickscore`.** All six move under PickScore in the screen (Δr 4.2–8.0).
ImageReward only moves four of them (f1/f2/f3 and the cubist one; the greenhouse
is 0.52 and the fisherman 1.00), so an ImageReward pass is a separate decision
and is *not* part of this run.

---

## 3. Frozen hyperparameters

Copied from `main0.sh`. Do not change any of them.

| var | value | why |
|---|---|---|
| `TARGET_N` | 8 | matches the editing runs |
| `MPC_H` | 2 | matches |
| `MPC_NI` | `"12 24"` | both inner counts; in 5 of 12 editing combos `ni=24` owns the frontier alone, so dropping it understates MPC |
| `HF_NI` / `HF_LR` | 40 / 2.5 | matches |
| `FMRG_NOPT` | 3 | matches |
| `SHARED_GRID` | 1 | **required** — win rates are only defined when all three methods share a λ |
| `HF_ALPHA` | 0.5 | lifts HardFlow's `lr*reg ≥ 1` floor so it can reach the shared grid's low end |
| `AUTORANGE_EXTRA` | `--n_grid_shared 6` | 5 left MPC only 2 of 5 points inside its own active region |
| `DIAG` | `--all` | diagnostics after all three methods finish |
| `SEEDS` | `"42 7"` | two seeds; seeds are never pooled, so one seed is an anecdote |

`auto_sweep.sh` pins `CUDA_VISIBLE_DEVICES` itself and passes
`--flux_device cuda:0 --reward_device cuda:0`. Never pass `cuda:1` or
`--reward_device auto`. Never use `--quantize nf4` (correct forward pass, zero
input gradient). Do **not** source `gpu_lib.sh` — its one-job-per-card lock is
right for the 24 GB A5000s and would idle most of a 140 GB H200.

---

## 4. Step 1 — smoke-test the range finder, and gate on it

This decides every λ in the run, so validate it on one prompt (~15 min) before
committing four workers to five hours.

```bash
cd "$R/fmocc_core"
./.venv/bin/python lambda_autorange.py --reward pickscore \
  --prompt "Abstract human form, impressionist strokes, pastel colors" \
  --edit_prompt "Abstract human form, impressionist strokes, pastel colors" \
  --methods mpc hardflow fmrg --probe_N 4 --target_N 8 --n_grid 6 \
  --n_grid_shared 6 --hardflow_lr_alpha 0.5 \
  --flux_device cuda:0 --reward_device cuda:0 --out_dir /tmp/ar_smoke
cat /tmp/ar_smoke/grids.txt
```

**Gate:** each method's line reports a logistic `R2`. The editing runs got
0.94–1.00. Enhancement has a higher `r_unc` floor and a smaller relative Δr, so
the fit may be noisier.

- All three ≥ 0.85 → proceed to §5.
- Any below 0.85, or a `shared` line with fewer than 4 of 6 points inside a
  method's active region → **stop and report the grids.txt contents.** Do not
  widen the range, do not rerun with different knobs. A bad grid puts the whole
  sweep on useless λ and that is worth a message, not a workaround.

---

## 5. Step 2 — create the launcher

Two scripts, mirroring the `floor2.sh` / `floor2_one.sh` shape already used in
this project.

```bash
cat > ~/enh_one.sh <<'EOF'
#!/usr/bin/env bash
# One aesthetic-enhancement combo. Enhancement == the generation prompt IS the
# reward prompt, so the prompt is passed to auto_sweep.sh twice. Every
# hyperparameter below is copied from main0.sh (the editing run) so the two
# tasks stay comparable -- do not "improve" one of them here.
set -u
R="${FMOCC_ROOT:-$HOME/fmocc}"
cd "$R/fmocc_core" || exit 3
export PATH="$R/.venv/bin:$PATH"
export FLOW_MATCHING_DIR="$R" FMRG_DIR="$R/fmrg"
export TARGET_N=8 MPC_H=2 MPC_NI="12 24" HF_NI=40 HF_LR=2.5 FMRG_NOPT=3
export SHARED_GRID=1 HF_ALPHA=0.5
export AUTORANGE_EXTRA="--n_grid_shared 6"
export DIAG="--all"
P="${1:?usage: enh_one.sh <prompt>}"
REWARD="${REWARD:-pickscore}"
mkdir -p results/enh_logs
SLUG="$(printf '%s' "$P" | tr -c 'A-Za-z0-9' '_' | cut -c1-40)"
LOG="results/enh_logs/${REWARD}__${SLUG}.log"
echo "[$(date +%H:%M)] START $REWARD | $P" | tee -a "$LOG"
SEEDS="${SEEDS:-42 7}" ./auto_sweep.sh 0 "$REWARD" "$P" "$P" >> "$LOG" 2>&1
echo "[$(date +%H:%M)] END rc=$? | $P" | tee -a "$LOG"
EOF

cat > ~/enh.sh <<'EOF'
#!/usr/bin/env bash
# 6 prompts, 4 workers, one card. No gpu_lib on purpose (see runbook section 3).
# None of the prompts contains a quote or backslash, so plain xargs -I{} is safe.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
NWORK="${NWORK:-4}"
PROMPTS=(
"Abstract human form, impressionist strokes, pastel colors"
"Close-up of an eye with the Earth inside the pupil"
"A Japanese garden at dawn, ink illustration, sumi-e"
"Fractured geometric landscape, cubist, muted ochre and teal"
"A Victorian greenhouse at midnight: brass lanterns, overgrown ferns, and a sleeping fox, watercolor"
"Portrait of an elderly fisherman, charcoal sketch, harsh side lighting"
)
printf '%s\n' "${PROMPTS[@]}" | xargs -P "$NWORK" -I{} bash "$HERE/enh_one.sh" "{}"
echo "ALL DONE $(date +%H:%M)"
EOF
chmod +x ~/enh.sh ~/enh_one.sh
```

---

## 6. Step 3 — launch

```bash
tmux new -s enh
NWORK=4 bash ~/enh.sh
```

Each of the six invocations runs, in order: λ range-finding, then MPC (2 inner
counts) × 2 seeds, HardFlow × 2 seeds, FMRG × 2 seeds, all on the shared grid,
then diagnostics. `auto_sweep.sh` always passes `--resume`, so a re-launch after
an interruption skips completed rows rather than redoing them.

Output lands in `results/auto/pickscore/<slug>/` with one subdirectory per
`<method>_seed<N>`, plus `autorange/` and `diagnostics/`.

---

## 7. Monitoring

```bash
tail -f ~/fmocc/fmocc_core/results/enh_logs/*.log
nvidia-smi          # expect ~4 python processes, roughly 17 GB each
```

Healthy signs, from the editing run's medians on a smaller card (the H200 should
beat these): MPC `n_inner=12` ≈ 315 s/run, `n_inner=24` ≈ 620 s, HardFlow ≈ 35 s,
FMRG ≈ 32 s. Four workers contending will inflate these somewhat; a factor of
more than ~2× over the serial numbers means the card is oversubscribed and the
timings should not be quoted.

Each prompt is roughly 15 min of range-finding plus 2 seeds × ~100 min of
sweeping, so ~3.5 GPU-hours per prompt, ~21 GPU-hours total, ≈5 h wall at 4×
parallelism.

---

## 8. Failure modes

| symptom | what to do |
|---|---|
| CUDA OOM | Do **not** pass `--continue_on_oom`. Reduce `NWORK` to 3 and relaunch; `--resume` picks up where it stopped. |
| A prompt exits non-zero | `auto_sweep.sh` logs it and continues to the next method. Rows with `run_ok=False` are retained in that method's `summary.csv` by convention — never delete them. Report which prompt and which method. |
| `no shared grid in grids.txt` | Range finding failed for that prompt. Stop, report, do not hand-write a grid. |
| Diagnostics fail at the end | The sweeps themselves are complete and the data is fine. Report it; `sweep_diagnostics.py` can be rerun from the Penn box. |
| Win rates report zero combos | `SHARED_GRID=1` was not set. The λ grids are per-method and no (λ, seed) cell contains all three. The sweep must be rerun. |
| Wall clock blows past ~6 h | Check for a competing job on the card before assuming the model is slow. |

Partial completion is acceptable and is why the queue is ordered one prompt at a
time: a finished prompt is a usable data point. Report exactly which of the six
completed.

---

## 9. On completion

**a. Verify completeness.**

```bash
cd ~/fmocc/fmocc_core
for d in results/auto/pickscore/*/; do
  echo "$d $(ls -d "$d"*_seed* 2>/dev/null | wc -l) arm-dirs"
done
```

Each enhancement slug should show **6** arm directories (mpc/hardflow/fmrg ×
seeds 42 and 7).

**b. Report, do not analyze here.** `paper_figures.py` is untracked and is not in
this checkout. Analysis happens on the Penn box.

**c. The sync target matters — this is the easiest way to corrupt existing
results.** `paper_figures.load_rows()` globs `results/auto/*/*/*_seed*/summary.csv`.
If the enhancement directories are copied into the Penn box's `results/auto/`,
they silently merge with the editing rows and every published number (83% vs
HardFlow, 90% vs FMRG) becomes a blend of two different tasks with nothing in
the output to reveal it. Sync into a **sibling** directory instead:

```bash
# run FROM the Penn box
rsync -av <studio>:~/fmocc/fmocc_core/results/auto/pickscore/ \
  /home/sungcho/fmocc-sherv/fmocc_core/results/auto_enh/pickscore/
```

Then load it explicitly, which needs no code change —
`load_rows` already takes the root as an argument:

```python
df_enh = pf.load_rows(pf.HERE / 'results' / 'auto_enh')
```

**d. Shut the studio down.** Charges accrue whenever a GPU is allocated, running
or not — a previous overnight run billed ~12 h against 4–5 h of actual compute.

---

## 10. Do not

- Change any hyperparameter in §3, or edit the prompts in §2.
- Reuse the λ values from `results/screen2/` (N=4, FMRG-only).
- Run this concurrently with `crun2.sh` / `queue2.sh`.
- Source `gpu_lib.sh`, or set `FMOCC_FAT_GPU` expecting it to matter here.
- Delete or edit anything under `results/` — result directories are immutable
  experimental records, and failed runs are kept deliberately.
- Silently work around a failure. Stop and report; that is more useful than a
  sweep that completed with something quietly adjusted.
