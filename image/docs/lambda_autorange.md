# λ auto-ranging: how it works

**`src/lambda_autorange.py` — ConicFlow / HardFlow / FMRG on FLUX.1-dev**
Describes the range finder as used for every sweep in the paper, including
the `--high_stop break` mode (§11). An earlier HTML report that this document
superseded is not part of the release; references to its sections (§05 etc.)
are historical.

---

## 1. What the tool does, in one paragraph

Given a `(reward, prompt, edit_prompt)` cell, it measures each method's
response curve `Δr(λ) = r_ctrl − r_unc` at a cheap proxy `N`, finds where that
curve is actually moving, and writes a λ grid placed across the transition —
per method, plus one shared grid on which all three methods can be compared at
equal λ. Output is a ready-to-paste `--sweep_lam` line per method. It exists so
that a large unattended sweep over several rewards and prompts does not need λ
hand-tuned per cell, which is what the prior hand-tuned lambda decision was doing by
hand.

Cost on one A5000: median 296 s per method-search, 178–449 s for 48 of the 51
searches on disk, with three contended outliers (865, 983, 1209 s) — so ~15 min
for a three-method cell.

---

## 2. Why a fixed λ grid cannot work

**λ is three different objects.** In ConicFlow it is the Lagrange weight the inner
optimiser solves against (`J = E/(2λ) − r`). In HardFlow it enters as the
deviation penalty `reg = N/(2λ)` — the same physical quantity per step, per the
derivation in `RunConfig.hardflow_reg` — but is additionally bounded below by a
*solver*-stability floor `λ ≥ N·lr/2`. In FMRG it is an open-loop gain on the
reward gradient, with nothing minimising energy at all.

**And its useful range moves with the reward's scale.** The repo's standing
note is that swapping LPIPS → ImageReward de-regularises by ~200× at fixed λ.
The measured gradient norm at the uncontrolled endpoint, which is what sets the
scale, varies by more than two decades across the cells on disk:

| reward | ‖∇_z r‖ at the uncontrolled endpoint (3 prompt cells) |
|---|---|
| clip | 0.19, 0.22, 0.61 |
| pickscore | 1.37 … 6.14 (11 cells) |
| imagereward | 0.072, 8.66, 12.29 |

So the prior λ₀ this implies ranges from 0.27 to 57.8 *for the same three
prompts*, purely from the choice of reward. A grid tuned for one cell lands on
the plateau or on the dead floor of another. The old suite grid
`{0.1, 1, 10, 50, 200, 1000, 3000, 10000}` was built for ConicFlow on ImageReward, and
even there six of its eight points measure the same plateau.

**What *is* invariant is the shape.** For every method and reward, `Δr(λ)` is
≈0 while the effort penalty dominates, rises through a transition, and plateaus
at whatever the reward model can be pushed to. A grid point is only worth its
GPU time where that curve is moving. So the automation problem is: locate the
transition for this `(method, reward, prompt)` cheaply, then put the points
there.

---

## 3. Stage 0 — probes (seconds, once per cell)

`probe_uncontrolled()` does one uncontrolled rollout and one backward pass:

- `r_unc` — the uncontrolled reward, so Δr is measured against the right
  baseline.
- `Δr_max ≈ r_max − r_unc`, with `r_max` from
  `reward_ranges.REWARD_OBSERVED_RANGE`. This is the **ceiling estimate**, used
  as a sanity bound and as the scale for the flatness tolerance
  (`tol = max(--tol · ceiling, --tol_abs)`, default `--tol 0.05`).
- `‖∇_z r‖` at the uncontrolled endpoint — one `torch.autograd.grad` through
  the decoder and the reward model.

The prior is then

```
λ₀ = c · Δr_max / ‖∇_z r‖ ,    c = 1.0 (conicflow, hardflow), 2.0 (fmrg)
```

The rationale (`scale_prior`'s docstring): the reward has to be moved by
`Δr_max`, and the gradient norm at the endpoint sets how much latent
displacement that costs; every method's λ multiplies a displacement that is
linear in λ to first order. A `1/‖∇r‖²` law was tried first and came out three
decades low on ImageReward; the `1/‖∇r‖` form with `c ≈ 1` put the measured
knees (ConicFlow ≈0.3 on ImageReward, ≈30 on CLIP) inside a decade. FMRG's knee was
~2× ConicFlow's on both, hence `c = 2`.

**The prior only has to be right to within a decade or two** — the bracket
corrects it, and a wrong prior costs extra evaluations, not correctness. It is
calibrated on n=2 rewards and should be treated as a heuristic.

---

## 4. Stage 1 — bracket

From λ₀, step **×10 upward** until the top of the curve is flat, and **×10
downward** until the bottom is inactive. Budget `--max_evals 10` per method;
the search never leaves `LAM_ABS_MIN=1e-4 … LAM_ABS_MAX=1e7`.

`_inactive` is simply: the lowest point has `Δr ≤ tol`.

`_saturated` is four tests on the top two points (`a`, `b`), with `rise` and
`gain` as defined in the code:

| test | fires when | why it exists |
|---|---|---|
| `flat` | `gain < tol` | the ordinary plateau |
| `near_ceiling` | `b.dr ≥ ceiling_stop_frac · ceiling` (default 0.9) | the reward is nearly maxed |
| `energy_waste` | `b.E > 3·a.E` and `gain < 2·tol` | paying energy for nothing |
| `falling` | `b.dr < max(dr) − tol` | **overshoot** — the image is being destroyed and the scorer marks the wreckage down |

Three of the four are gated on `risen = (rise ≥ 2·tol)`, which guards against
calling two ≈0 points "saturated". `2·tol` rather than a larger multiple
because HardFlow's admissible floor can sit partway up the curve, so its whole
visible rise may be small. **§10.1 is how that gate silently disabled `falling`
in exactly the case it was written for, and what the fix changed.** The function
is `saturation_reasons` since 2026-09-23 and returns *which* tests fired, so the
plateau can be recorded without being stopped at (§11).

Runs that raise are recorded, not retried, and treated as the admissible
boundary — HardFlow below its `lr·reg ≥ 1` floor is the designed case, and OOM
lands in the same branch.

The **active region** is read off the bracketed points: `lo` = the highest
point that is still inactive (else the lowest point), `hi` = the first point at
which `_saturated` becomes true (else the top point).

---

## 5. Stage 1b — energy-spike rejection

Added 2026-09-17 (`mark_suspects`). HardFlow's explicit fixed-step inner ascent
can diverge at a *single* λ and still return a finite number — the code's own
example is `ctrl_energy = 1.15e18` at λ=27 between neighbours at 2.9 and 11.8,
with Δr falling to −0.83. `_run_one`'s guards only catch non-finite values, so
that point survives into the curve, where it trips `falling`/`energy_waste` and
truncates the bracket, and drags the logistic fit and the grid with it.

The test is deliberately a **spike, not a threshold**: a point is suspect only
if its `log10 E` stands `--spike_decades` (default 3) above **both** its
λ-neighbours, i.e. the curve comes back down afterwards. Energy legitimately
grows by many decades across a real sweep (FMRG's top is ~1e3 above its bottom
and that is genuine), and the genuine overshoot regime has energy going up and
*staying* up — that must not be filtered away.

Endpoints are never flagged (with no point beyond it there is no evidence the
energy came back down), so a spike that is still the last point stays visible to
the stopping tests until a later point appears above it. `mark_suspects` is
re-run after every evaluation and once more at the end.

Suspect points stay in `points`, in `evaluations.csv`, and in the plots — they
are marked, not deleted, so a run that hit an instability is never silently
prettier than it was.

One live example on disk: `pickscore / Victorian greenhouse`, HardFlow, λ=104 —
`E = 1.35e6` between neighbours at 511 and 733, `Δr = −3.93` between +8.8 and
+12.3. Flagged, excluded from the fit, still in the CSV. It is the only spike
across 529 evaluations.

---

## 6. Stage 2 — fit, grid, refine

`fit_logistic` fits `Δr(x) = A / (1 + exp(−k(x − m)))` in `x = log10 λ` by a
121×58 grid search over `(m, k)` with `A` solved in closed form and clamped to
`1.5·max(ceiling, y.max())`. numpy only, no scipy. It reports `r2`.

`quantile_grid` then places the `--n_grid` points where the *fitted* curve
passes through equally spaced fractions of its rise, **8% … 92%**, clipped to
the active region. That is the whole point of the fit: every grid point buys a
distinct point on the energy/reward frontier instead of another sample of the
plateau.

Three fallbacks to a log-uniform grid over the active region:

1. no fit (fewer than 3 valid points, or no spread in `y`);
2. `r2 ≤ 0.5`;
3. the quantiles collapsed — fewer than `max(3, n_grid//2)` distinct points
   survive 2-significant-digit rounding, or two neighbours are closer than a
   factor 1.25.

**Refinement** (`--evaluate_grid`, default on) then evaluates every proposed
grid point that has not already been run, re-flags spikes, and re-fits. So the
output is the whole measured curve, not just a proposal — which is why
`n_evals` on disk runs to 7–15 rather than the bracketing budget of 10.

---

## 7. Transfer to the target N, and HardFlow's floor

Range-finding runs at `--probe_N` (default 4); the grid is for `--target_N`
(default 10, but every sweep on disk used 8).

ConicFlow's transition is approximately N-invariant because λ is per-step normalised
through `dt`, so the grid transfers; the HTML report's §05 measured this
directly at N=4 → N=10 and it held (Δr crossed 0.5 at λ≈0.19 at both N, even
though `r_unc` differed, −1.44 vs −1.60).

HardFlow's admissible floor is **not** N-invariant:

```
hardflow_image raises when lr·reg ≥ 1,  reg = N/(2λ)   ⇒   λ ≥ N·lr/2
```

which is 5 at N=4 and 12.5 at N=10 with the authors' `lr = 2.5`. So the floor
is computed at both N: the probe-N floor bounds the search, and the target-N
floor re-cuts the grid. Points below it are reported in
`clipped_by_target_floor` and named in `grids.txt` — never silently shifted.

If the lowest *admissible* point is already on the plateau, the transition lies
below the floor and no grid above it can measure it. The tool says so —
`status = "saturated_at_admissible_floor"` plus a note naming the fix — rather
than handing back six plateau points.

**Floor lifting (`--hardflow_lr_alpha`, 2026-09-17).** Below the floor, the run
uses `lr = α/reg`, pinning `lr·reg = α < 1`. This is not a hack around the
guard: the inner loop's contraction factor becomes `1 − 2α·α_t²`, uniform in λ,
so convergence no longer depends on where the run sits on the grid (α=0.5 is
dead-beat at the final step). The effective lr is recorded per row in
`hardflow_lr_eff`. It exists because on ImageReward the floor sits **above**
HardFlow's entire transition, so without it HardFlow can never be compared to
ConicFlow at a shared λ in the regularised regime. Every sweep on disk used
`HF_ALPHA=0.5`, and with it there is no admissible boundary left to report.

---

## 8. The shared grid

Per-method grids resolve each method's own transition best, but runs at
different λ are not comparable. `shared_grid` builds one grid for all three.

**Why that is a legitimate comparison** (from the function's docstring): ConicFlow's
per-step linearised problem `min ‖resid‖²/(2λ·dt) − ⟨∇r, resid⟩` has the closed
form `resid* = λ·dt·∇r`; FMRG's update — `n_opt` steps of `(dt/n_opt)·λ·∇r` — is
exactly that per step; and HardFlow's `reg = N/(2λ)` is matched per step by the
`RunConfig` derivation. At a shared λ all three are solving the same first-order
problem, and the CSV's `objective` column (`E/(2λ) − r_ctrl`) scores each of
them against it. Where they differ is feedback: ConicFlow re-solves, FMRG is open-loop
and overshoots at high λ.

**Construction.** Take the union of every method's quantile grid (each already
spans 8–92% of that method's own rise) → merge neighbours closer than
`--shared_min_ratio` (default 1.4) by replacing the pair with their log-midpoint
→ while more than `--n_grid_shared` points remain, drop, one at a time, the
point whose removal least widens the largest log-gap. This keeps every method's
transition resolved rather than spreading points log-uniformly over a union
region that one method may not use.

It then reports **coverage**: how many shared points fall inside each method's
own active region, with an explicit `# UNDER-RESOLVED: raise --n_grid_shared`
when a method gets fewer than 3. That flag is load-bearing — see §10.2.

`--diag_winrates` requires the shared grid, since the objective is only
comparable at equal λ.

---

## 9. Implementation and outputs

**One process, one FLUX load.** The tool calls `run_sweep._run_one` directly,
so every guard in that path still applies: locked defaults, reward
plausibility ranges, non-finite checks, the `FAILED_` bookkeeping. It does not
reimplement the control loop.

**Reward-model memoisation.** `_run_one` rebuilds the reward model on every call
and parks it on CPU at the end, which dominates the per-λ cost for cheap proxy
configs. `lambda_autorange` monkeypatches `rs._build_reward_fn` with a cache
keyed on `(reward, edit_prompt, device, dtype, drop_text_tower, style_refs)` and
moves the cached object back to the reward device on reuse (a few hundred ms).
`lpips` and `none` are never cached — lpips needs a per-run `ref_holder`
populated after the uncontrolled rollout.

**Proxy configs** (all flags): ConicFlow `H=2, n_inner=6, lr_inner=1e-2`,
`--endpoint_surrogate jump` (~3× cheaper than `rollout`, and the transition
matched within noise at N=8); HardFlow `n_inner=40, lr=2.5` (the authors'
values); FMRG `n_opt=3`. 256×256, `flow_map2`, one seed (default 42).

**Outputs** in `--out_dir`:

| file | contents |
|---|---|
| `autorange.json` | everything: args, probe, per-method status/fit/active region/points, shared grid |
| `evaluations.csv` | one row per λ evaluated: `method, lam, ok, r_unc, r_ctrl, dr, E, J, t_ctrl, err, suspect, wall_s` |
| `curves.png` | Δr vs λ per method: points, logistic fit, grid ticks, ceiling, HardFlow floor, red ticks where a run raised |
| `grids.txt` | ready-to-paste `--sweep_lam` lines, plus the shared line and coverage notes |

A real `grids.txt` (`results/auto/imagereward/a_portrait_of_a_man__a_smiling_face`):

```
conicflow      --sweep_lam 0.072 0.13 0.19 0.25 0.36 0.65
         # active region lam in [0.0272, 2.72], 9 evals, 287s, logistic R2=0.96
hardflow --sweep_lam 0.069 1.1 2.7
         # active region lam in [0.0272, 2.72], 7 evals, 178s, logistic R2=0.68
fmrg     --sweep_lam 0.36 0.63 0.87 1.2 1.6 2.8
         # active region lam in [0.0544, 5.44], 9 evals, 214s, logistic R2=0.93

shared   --sweep_lam 0.07 0.13 0.36 0.75 1.6 2.7
         # conicflow: 6 of 6 shared points inside its active region [0.0272, 2.72]
         # hardflow: 6 of 6 shared points inside its active region [0.0272, 2.72]
         # fmrg: 6 of 6 shared points inside its active region [0.0544, 5.44]
```

### Usage

```bash
CUDA_VISIBLE_DEVICES=0 python lambda_autorange.py --reward imagereward \
    --prompt "a portrait of a man" --edit_prompt "a smiling face" \
    --methods conicflow hardflow fmrg --probe_N 4 --target_N 8 --n_grid 6 \
    --hardflow_lr_alpha 0.5 \
    --flux_device cuda:0 --reward_device cuda:0 --out_dir results/autorange/imagereward
```

`auto_sweep.sh` chains it into the real sweep — one `run_sweep.py` invocation
per method at `TARGET_N` on exactly the grid found, with `--resume`:

```bash
TARGET_N=8 CONICFLOW_NI=12 SEEDS="42 43 44" SHARED_GRID=1 HF_ALPHA=0.5 \
DIAG="--diag_winrates --diag_pareto" \
  ./auto_sweep.sh 0 pickscore "a portrait of a man" "a smiling face"
```

Two things worth knowing about the wrapper: range finding is **cached** — it
depends only on `(reward, prompt, edit_prompt)`, never on the seed, so an
existing `grids.txt` is reused unless `REDO_AUTORANGE=1`, which is what makes
the safe method-outer/seed-inner ordering affordable. And `AUTO_ROOT` puts the
output tree somewhere other than `results/auto/` while keeping the
`<reward>/<slug>/<method>_seed<N>/` layout every downstream tool expects.

---

## 10. Measured behaviour, and what goes wrong

Across the 17 cells on disk (`results/auto/`, 3 rewards × 3 prompts, and
the Experiment 1 sweep, pickscore × 8 prompts): **51 method-searches, 529
evaluations, all 51 returning `status = ok`.** Logistic quality by method:

| method | R² ≥ 0.9 | 0.5 ≤ R² < 0.9 | R² < 0.5 (→ log-uniform grid) |
|---|---|---|---|
| conicflow | 16 / 17 | 1 / 17 (0.887) | — |
| hardflow | 9 / 17 | 8 / 17 (0.55 … 0.83) | — |
| fmrg | 9 / 17 | 2 / 17 | 6 / 17 (−0.00 … 0.35) |

ConicFlow's response is a textbook logistic in log λ (12 of 17 fits at R² ≥ 0.99).
HardFlow's is noisier — single-evaluation nondeterminism — but always fits well
enough to use the quantile grid. FMRG's is the problem case: 6 of 17 searches
fall back to a log-uniform grid, five of them on PickScore.

### 10.1 Defect: `risen` disables the `falling` stop on a monotone collapse

`_saturated` computes `rise = b.dr − min(dr)`, i.e. *the last point minus the
curve's minimum*. On a collapsing curve the last point **is** the minimum, so
`rise = 0`, `risen = False`, and `flat`, `energy_waste` and `falling` are all
gated off. The upward bracket then cannot stop until it hits `LAM_ABS_MAX = 1e7`.

Verified by replaying the actual predicates (function source extracted from the
module) on the recorded points of `pickscore / A Japanese garden at dawn`, FMRG:

| λ | Δr | min Δr so far | `rise` | `risen` | `_saturated` |
|---|---|---|---|---|---|
| 1.6 | 2.893 | 2.893 | 0.000 | False | False |
| 16 | 3.978 | 2.893 | 1.085 | True | False |
| 160 | 2.095 | 2.095 | 0.000 | **False** | False |
| 1600 | 0.089 | 0.089 | 0.000 | False | False |
| 16 000 | −2.003 | −2.003 | 0.000 | False | False |
| 160 000 | −5.514 | −5.514 | 0.000 | False | False |
| 1 600 000 | −6.078 | −6.078 | 0.000 | False | False |

`tol = 0.245`, `ceiling = 4.90`. The curve peaks at Δr = 4.19 (λ=25) and
collapses to −6.08 with `ctrl_energy = 3.1e10` — the image is destroyed — and
the stop written for precisely this case never fires. With `rise` defined as
`max(dr) − min(dr)` the same replay stops at λ=160, on the first point past the
peak (3 fewer evaluations here).

Cost across all 51 searches: **60 of 529 evaluations (11%) were spent above the
peak on a falling curve**, and 6 searches (all FMRG; 5 PickScore, 1
ImageReward) ran their active region out to ≥1e5.

**Fixed 2026-09-23** (`saturation_reasons`, formerly `_saturated`): `rise` is now
`max(dr) − min(dr)`. This changes the default path — it is the one behavioural
change in that commit that is not opt-in — and it was a prerequisite for §11,
where the reward-collapse test has to be able to fire at all.

### 10.2 Consequence: a collapsed FMRG curve drags the shared grid

Because `active_hi` is taken from the bracket, the garden cell above ended with
`active_hi = 1.6e6`, `r2 = −0.00` → log-uniform fallback → the FMRG grid
`0.016, 0.64, 25, 1000, 40 000, 1.6e6`, spanning 8 decades with Δr = −6.08 at
the top. The shared grid is the union of the per-method grids, so it inherited
that span:

```
shared   --sweep_lam 0.016 1.2 80 1000 40000 1.6e+06
         # conicflow: 2 of 6 shared points inside its active region [0.00798, 7.98]   # UNDER-RESOLVED
         # hardflow: 0 of 6 shared points inside its active region [7.98, 79.8]   # UNDER-RESOLVED
         # fmrg: 6 of 6 shared points inside its active region [0.016, 1.6e+06]
```

The coverage report did its job and said so loudly. But **this cell is one of
the three rows in Figure 3(b)** (`A Japanese garden at dawn`), so any
shared-λ reading of that cell rests on 2/6 and 0/6 coverage. Figure 3's rows are
selected on achieved reward/energy rather than at equal λ, so the figure itself
is unaffected; a win-rate table over this cell would not be.

### 10.3 Other caveats

- **FMRG's response is a cliff, and it is not reproducible near it.** On
  ImageReward it goes from Δr 0.14 at λ=0.36 to 0.62 at 0.544 to 1.59 at 0.63 to
  2.96 at 0.87; the earlier study measured 2.94 at λ=0.83 on one pass and 0.09 at
  λ=0.77 on the next. An open-loop gain has no energy term pulling it back, so a
  small change in λ (or kernel nondeterminism) flips which basin the sample lands
  in. FMRG grids should be denser across the knee than the logistic quantiles
  give, or averaged over seeds.
- **The ceiling is the reward model's, not the prompt's.** CLIP's observed max
  0.80 gave a ceiling of 0.65 on the portrait cell while the attainable plateau
  was 0.24. An earlier ceiling-only stopping rule walked λ to 3.5e6 looking for
  it; that is why flatness is judged on the observed rise and the ceiling is only
  a sanity bound plus the tolerance scale.
- **Above its floor HardFlow is not regularised in any useful sense**, and on
  PickScore it reward-hacks: on the portrait cell its peak is Δr 11.7 at energy
  342 against ConicFlow's 4.5 at 14.4, and on the greenhouse cell 15.8 at 1.9e3
  against 9.8 at 7.4 — 1.4 and 2.4 decades more energy for the higher score.
  Read those as scorer exploitation, not as HardFlow winning.
- **HardFlow's run-to-run nondeterminism** (three identical commands spread Δr by
  ~70% of the smallest value on CLIP) makes its curve the least reliable of the
  three at one evaluation per λ. Its grids are the right decade, not the right
  point.
- **One seed, one prompt pair per search.** The tool re-derives the grid per
  cell, which is the point; no grid here is a constant to copy.
- **Proxy N changes the uncontrolled image** (`r_unc` differs between N), so
  absolute Δr differs between proxy and target N. What transfers is the λ
  location of the transition.

---

## 11. The permissive high end (`--high_stop break`, 2026-09-23)

Added so that lambda keeps increasing until the images 'break', in order to
reach even less conservative edits.

### 11.1 What was actually capping the top

Two separate conservatisms, both measured on the 17 cells:

**The stop.** `near_ceiling` ends the upward walk in **24 of 51** searches (14/17
HardFlow, 9/17 ConicFlow) — more than every other stop combined. It fires at
`0.9 · (r_max − r_unc)` with `r_max` from `REWARD_OBSERVED_RANGE`, whose
PickScore entry `(18.0, 27.0)` is labelled *"MEASURED but WEAK: n=4 only"* in
`reward_ranges.py`. It is too low: ConicFlow on `pickscore / a portrait of a man`
reaches `r_ctrl ≈ 32.8`, five points past that "maximum". Breakdown of the
remaining stops: `flat` 8, `flat+energy_waste+falling` 7, `energy_waste` 3,
`falling` 2, budget/λ-box 5.

**The placement, which is the bigger effect.** The quantile grid spans 8–92% of
the *fitted* rise, so it never places a point on the plateau even when the
bracket went there. Emitted grid top vs measured active region, ConicFlow:

| cell (pickscore) | grid top | active region top | Δr at grid top | Δr at peak |
|---|---|---|---|---|
| a portrait of a man → a smiling face | 21 | 279 | 3.76 | 4.50 |
| a field of green grass → red poppies | 21 | 3180 | 6.30 | 7.15 |
| a portrait of a smiling man → angry | 150 | 5170 | 4.57 | 5.35 |
| a portrait of a young man → old man | 43 | 552 | 5.95 | 6.29 |

![Top of the range on pickscore / "a portrait of a man" → "a smiling face", seed 42, N=8. ConicFlow at λ=280 is the strongest clean edit in the row; only FMRG breaks.](lambda_autorange_breakpoints.png)

**And the images up there are fine.** The `pickscore / a portrait of a man` sweep
at N=8 ran λ up to 280. At λ=280 ConicFlow gives Δr = 13.04 — the strongest clean
smile in the row — with relative pixel L2 0.26, TV ratio 1.12, no clipping. Its
own grid stopped at λ=21 (Δr ≈ 7.2 at the nearest swept point). Only FMRG breaks
at 280: a cartoon-smiley overlay with 20.8% of channel values clipped.

### 11.2 Why the break test has to be pixel-side

Reward cannot be the break signal on PickScore, because PickScore *rewards the
wreckage*: HardFlow scores Δr 11.7 and 15.8 on visibly destroyed images, past the
documented maximum, at 1.4–2.4 decades more energy than ConicFlow. So
`image_break_stats` compares the controlled image to the uncontrolled one at the
same seed — free, since `_run_one` already returns both decoded PIL images — on
four statistics: relative pixel L2, total-variation ratio, pixel-std ratio, and
clipped-pixel excess.

Calibrated on the one cell where "clean" and "destroyed" are both available:

| | conicflow λ=280 | hf λ=280 | fmrg λ=130 | fmrg λ=280 | threshold |
|---|---|---|---|---|---|
| clip excess | 0.000 | 0.000 | 0.038 | **0.206** | 0.10 |
| TV ratio | 1.12 | 1.11 | 1.35 | **1.52** | 1.45 |
| std ratio | 1.03 | 1.00 | 1.50 | **1.95** | 1.80 |
| pixel L2 | 0.26 | 0.26 | 0.74 | **1.31** | 1.00 |

Only FMRG λ=280 trips them, and it trips all four. **These are a catastrophe
stop, not a quality judgment** — n=1 calibration, and they do not catch
HardFlow's crunchy-texture degradation at λ=280, which is obvious to the eye and
moves none of them. That is why break mode also writes the decoded image for
every λ it evaluates to `<out_dir>/images/`: the operator decides where "too far"
is by looking.

### 11.3 What the mode does

One switch; the other knobs are resolved from it in one place so it cannot be
half-applied. `plateau` (default) keeps every pre-2026-09-23 value.

| | plateau | break |
|---|---|---|
| `ceiling_stop_frac` | 0.9 | disabled |
| upward stop | `flat`/`near_ceiling`/`energy_waste`/`falling` | image broke, or Δr collapsed by ≥ `break_dr_drop`·rise |
| `up_step` | ×10 | ×√10 (lands closer to the break point) |
| `n_high` | 0 | 2 extra points from the top quantile up to the edge |
| `max_evals` | 10 | 14 |
| `save_images` | off | on |

The plateau tests still run in break mode, but only to record `lam_knee` — where
the default would have stopped — so the two modes stay comparable on a cell. A λ
whose image broke is kept in `points`, `evaluations.csv` and the plot, but is
excluded from the logistic fit and dropped from the grid: the same treatment
energy spikes get.

**`q_hi` is deliberately *not* part of the preset.** Raising it to 0.995 was
tried and is worse: the offset from the fitted midpoint is `ln(q/(1−q))/k`
decades, so the top quantile shoots up the tail — on `clip / cat` (k=2.2) it
jumped from 660 to 13 000 and left a 2-decade hole below it. `n_high` fills the
top uniformly in log λ instead.

Effect on the pickscore portrait cell, re-placed on the recorded curve:
`0.95 2.2 3.6 5.6 9.2 21` → `0.73 2.2 4.1 7.2 14 41 78 150 280`.

### 11.4 Cost, and what is not yet verified

Grids get longer: `n_grid + n_high`, so 8–9 points instead of 6 — **~50% more
runs per method-seed in the sweep itself**, which is where the real cost is.
Range finding goes from ~11 to ~18 evaluations per method on a synthetic
end-to-end dry run (+3 min per method, ~+8 min per three-method cell).

Verified: compiles; break detection reproduces the table in §11.2 on the real
PNGs; the reward-collapse stop replayed on the garden FMRG curve stops at λ=160
instead of running to 1.6e6; grid placement recomputed for all 17 cells; both
modes dry-run end to end against a synthetic evaluator (break mode walks to the
synthetic break at λ=3200, excludes it, and puts the grid edge at 1010).

**Not verified: a real GPU run.** Both A5000s were at 100% with ~9.7 GB free
when this landed, and an int8 FLUX needs 12.3 GB. The one thing a live run will
tell us that none of the above can: **where the walk actually stops for
ConicFlow/HardFlow**, since no run on disk has ever gone above λ=280 on these cells.
Also unverified is whether the break λ transfers from the probe N to the target
N the way the transition does (§7) — the honest expectation is that it is
*less* N-invariant, because the number of controlled steps changes how much total
displacement a given λ buys. Worth one target-N run at the top grid point before
committing a long sweep.

## 12. Where things live

| | |
|---|---|
| implementation | `src/lambda_autorange.py` |
| grid trimming | `src/trim_grids.py` |
| sweep wrapper | `scripts/auto_sweep.sh` |
| overlay plot | `src/plot_autorange_overlay.py` |
| output, per cell | `<AUTO_ROOT>/<reward>/<slug>/autorange/` — `autorange.json`, `evaluations.csv`, `grids.txt`, `curves.png` |

For the released experiments `AUTO_ROOT` is `results/main_comparison`, written
by Experiment 1 and then read (never regenerated) by Experiments 2–4.
