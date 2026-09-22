# `compare_constrained.py` — one constrained MPC-vs-HardFlow comparison

Give it a prompt, an edit prompt, a λ and a threshold. It runs both methods,
scores the results and writes a three-panel figure:

```
uncontrolled | MPC | HardFlow
```

The constraint is `score(image, edit_prompt) >= threshold` with **no reward**,
so each method's objective is its control energy alone. That is the setting
where the two are most directly comparable: both must reach the same feasible
set, and the only question is what it costs.

Unlike the Section 5 tools in the main README, this one **launches runs and
needs a GPU**. It is the whole experiment, not a reader of finished ones.

```bash
python compare_constrained.py \
    --prompt "a portrait of a man" --edit_prompt "a smiling face" \
    --lam 10 --threshold 23.0 --gpu 0 --out_dir results/cc_smile
```

---

## What it writes

```
<out_dir>/mpc/          full run_sweep output dir (summary.csv, images, mu_trace, run.log)
<out_dir>/hardflow/     same
<out_dir>/compare.png   the three-panel figure
<out_dir>/comparison.json   every number in the figure, plus the resolved args
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

---

## Four things it does on purpose

**Two processes, not one sweep.** `run_sweep.py`'s sweep axes form a cartesian
product with no per-method overrides, and the two methods want very different
inner loops (`n_inner` 24 / `lr` 1e-2 versus 40 / 2.5). One invocation would
force both onto the same values or run four configs, half of them nonsense.
Separate processes also match the one-config-per-process rule in
[SUITE.md](SUITE.md).

**It re-scores the saved PNGs.** With `--reward none` the CSV's `r_unc`/`r_ctrl`
are 0 and only the controlled endpoints get an `h_0`, so the uncontrolled panel
has no score anywhere on disk. The script scores all three images with the same
convention `run_sweep._build_reward_fn` uses for PickScore —
`logit_scale.exp() * cos_sim` on CLIP-224 preprocessed `[0,1]` images — which
puts every panel in one unit. It reproduces the CSV's `h_0` to within PNG
quantization (~0.01–0.07) and prints both columns so the agreement is visible
rather than asserted. Only `pickscore` has a scorer here; another
`--constraint_type` will run fine and then stop at the figure, with a message
saying to extend `_score_images`.

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
