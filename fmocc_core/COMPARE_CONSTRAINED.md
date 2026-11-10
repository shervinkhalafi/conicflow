# `compare_constrained.py` — one constrained MPC-vs-HardFlow comparison

Give it a prompt, an edit prompt, a λ and one or more constraints. It runs
both methods, scores the results and writes a three-panel figure:

```
uncontrolled | MPC | HardFlow
```

Each constraint is `score_i(image, arg_i) >= threshold_i` (or `<=`) with
**no reward**, so each method's objective is its control energy alone. That is
the setting where the two are most directly comparable: both must reach the
same feasible set, and the only question is what it costs.

Unlike the Section 5 tools in the main README, this one **launches runs and
needs a GPU**. It is the whole experiment, not a reader of finished ones.

```bash
python compare_constrained.py \
    --prompt "a portrait of a man" --edit_prompt "a smiling face" \
    --lam 20 --threshold 24.0 --gpu 0 --out_dir results/cc_smile
```

Several constraints are parallel lists, exactly as `run_sweep.py` takes them,
and are shared by both arms:

```bash
python compare_constrained.py ... \
    --constraint_types pickscore clip \
    --constraint_args "a smiling face" "with glasses" \
    --constraint_thresholds 24.0 0.31 \
    --constraint_directions ge ge
```

`--constraint_args` defaults to `--edit_prompt` for every constraint and
`--constraint_directions` to `ge`. The single-constraint spellings
`--constraint_type` / `--threshold` / `--direction` are aliases, so every
older invocation still works. Feasible means every `h_i ≤ 0`; the figure lists
each constraint's score and `h` per panel.

## Automatic thresholds

`--auto_threshold` derives the grid from **this seed's** uncontrolled image
instead of taking `--constraint_thresholds`:

```bash
python compare_constrained.py ... --constraint_types pickscore clip \
    --auto_threshold --auto_lo 1.1 --auto_hi 2.0 --n_grid 5
```

It runs one uncontrolled sample (`control_type=none`, no reward, ~1 min),
scores it with every constraint to get `s0_i`, lays `n_grid` multipliers `m`
evenly from `--auto_lo` to `--auto_hi`, and runs a full comparison at each.
Point `k` tightens **every constraint together**:

```
ge:  threshold_i = s0_i + (m_k − 1)·|s0_i|      (= m_k·s0_i when s0_i > 0)
le:  threshold_i = s0_i − (m_k − 1)·|s0_i|
```

Stepping by `|s0|` rather than multiplying matters in two cases a plain
multiple gets backwards: a negative baseline (ImageReward's z-score often is,
and `1.1×` of a negative number *loosens* a `ge` constraint) and `le`
constraints (where multiples above 1 loosen too). For the common case —
positive score, `ge` — it is exactly `m·s0`.

Each point lands in `<out_dir>/m<mult>/` (`m1.10/`, `m1.33/`, …), so the
directory names line up across seeds even though the absolute thresholds
differ per seed. `<out_dir>/auto_threshold.json` records the uncontrolled
scores, every point's thresholds, and each point's energies and feasibility
once scored; `<out_dir>/uncontrolled/` holds the probe run. A point that fails
is recorded in the manifest and the grid continues. `--fig_only` re-scores and
redraws every point from the manifest; `--skip_existing` skips the probe and
any arm that already has a `run_ok` row.

---

## What it writes

```
<out_dir>/mpc/          full run_sweep output dir (summary.csv, images, mu_trace, run.log)
<out_dir>/hardflow/     same
<out_dir>/compare.png   the three-panel figure
<out_dir>/comparison.json   every number in the figure, plus the resolved args

# with --auto_threshold, the above repeats under <out_dir>/m<mult>/, plus
<out_dir>/uncontrolled/     the probe run (its __img_unc.png is what was scored)
<out_dir>/auto_threshold.json   uncontrolled scores, per-point thresholds and results
```

and prints, per panel, the score, `h` from both the re-scored PNG and the CSV,
feasibility, control energy, `|u|max` and wall clock.

---

## Defaults

The four inputs are required; everything else carries the values that were
actually used for the PickScore comparison on 2026-09-22.

| | shared | MPC | HardFlow |
|---|---|---|---|
| | `N=8`, `seed=42`, `lr_dual=0.2`, `mu_init=0.0` | `H=2`, `n_inner=24`, `lr_inner=1e-2`, `endpoint_surrogate=jump`, `drift_detach=true` | `n_inner=40`, `lr_inner=2.5`, `lr_alpha=0.1` |

`--mpc_N` / `--hf_N` override `--N` for one arm. `--mpc_extra "..."` and
`--hf_extra "..."` pass anything else straight through to `run_sweep.py`
(e.g. `--mpc_extra "--sweep_dual_adaptive true"`), so no wrapper flag is needed
for a knob that already exists there.

`--fig_only` re-scores and redraws from finished runs without touching the
control loop. `--skip_existing` skips an arm that already has a `run_ok=True`
row, so an interrupted comparison resumes at the arm that did not finish.
Neither overwrites the wall clock recorded in `comparison.json` by the run
that actually produced an arm.

Captions show score, `h`, feasibility and control energy. Wall clock is left
off by default — it depends on the card, contention and model-load caching
more than on the method, and the two arms' inner loops are not matched for
compute — so a caption would invite a comparison the figure cannot support.
`--show_time` puts it back; the terminal table always prints it.

---

## Four things it does on purpose

**Two processes, not one sweep.** `run_sweep.py`'s sweep axes form a cartesian
product with no per-method overrides, and the two methods want very different
inner loops (`n_inner` 24 / `lr` 1e-2 versus 40 / 2.5). One invocation would
force both onto the same values or run four configs, half of them nonsense.
Separate processes also match the one-config-per-process rule in
[SUITE.md](SUITE.md).

**It re-scores the saved PNGs with run_sweep's own scorers.** With `--reward
none` the CSV's `r_unc`/`r_ctrl` are 0 and only the controlled endpoints get an
`h_i`, so the uncontrolled panel has no score anywhere on disk. Rather than
carry a parallel scorer per reward type, the script imports `run_sweep` and
calls `_build_reward_fn` for each constraint, feeding it the saved image through
the `decode_fn` hook every branch already routes through. Every panel is
therefore scored by exactly the code the control loop enforced — for any
constraint type, including style and LPIPS — and it reproduces the CSV's `h_i`
to within PNG quantization (~0.01–0.07); both columns are printed so the
agreement is visible rather than asserted. The scorer is built only between
subprocesses and freed before the next one launches: FLUX plus a run's own
scorers already peak near 19 GB on this card.

**It checks the two uncontrolled images are byte-identical.** Same seed and same
N must give the same starting image; if they differ, the arms were not
controlling the same problem and the figure would be quietly misleading. It
warns rather than failing, and says the uncontrolled panel is MPC's.

**It forwards `--hf_lr_alpha`, and says whether it fired.** `hardflow_image`'s
inner ascent contracts only while `lr·reg·α² < 1` with `reg = N/(2λ)`, so at the
authors' `lr=2.5` it raises below `λ = N·lr/2`. `run_sweep` substitutes
`lr = alpha/reg` **only** when that floor would be hit, so the flag is inert
above it. The HardFlow panel title reads `lr=2.5` when the floor was clear and
`lr·reg=0.1` when it was not. Pass a negative value to omit the flag entirely
and let the floor raise.

The default `0.1` is not arbitrary: at `N=8, λ=10` — where the floor sits at
exactly λ=10 — `0.5` leaves the damping factor `1 − 2·lr·reg·α²` at **zero** when
`α → 1` on the final step, and the run blew up there (`|u|` 219, energy 6184,
constraint thrown from nearly-satisfied back to its starting violation). `0.1`
gives 0.8 and is stable. Anything at or near the floor is a detuned solver, so
prefer a λ comfortably above `N·lr/2` when the comparison has to be clean.

---

## Reading the output

Single runs are **not** bit-reproducible in the control loop. The sampler is —
the uncontrolled image is md5-identical across reruns — but int8 kernels and
nondeterministic CUDA reductions move the controlled result. Two runs of the
identical command gave:

| | run A | run B |
|---|---|---|
| MPC energy / `h` | 1.75 / −1.52 | 1.87 / −1.58 |
| HardFlow energy / `h` | 4.69 / −0.11 | 3.97 / −0.07 |
| energy ratio | 2.7× | 2.1× |

So expect ~10–15% run-to-run noise on energy. From `n=1` the honest claim is
"MPC is 2–3× cheaper here", not a ratio to two digits. Sweep `--seed` and
aggregate before putting a number in a figure caption.
