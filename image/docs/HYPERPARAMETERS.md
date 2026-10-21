# Hyperparameters, and why they are what they are

> **How the numbering below maps onto the paper and this repository.** This
> file numbers the image experiments 1–5, from an earlier layout of the code.
> - **Experiment 1** (PickScore editing, `jump`, edits ed1–ed4): its seed-317
>   and seed-1289 runs are part of the ablation's core arms
>   (`scripts/ablation/run_ablation.sh <gpu> --arms core --tags "ed1 ed2 ed3 ed4" --seeds "317 1289"`),
>   and they supply Fig. 2 (right) rows 2–3 and the appendix image rows u01–u04.
>   Its seed-4703 runs are not used in the paper.
> - **Experiments 2–4** are the appendix ablations: matched wall clock, planning
>   horizon `H`, and objective evaluation point. They were later extended to
>   8 edits × 4–6 seeds; `scripts/ablation/README.md` has the final design.
> - **Experiment 5** is the constrained experiment (Fig. 3):
>   `scripts/constrained/run_suite.sh`.
> - **Not covered here:** the Table 1 / Fig. 2 (left) runs use
>   `endpoint_surrogate = rollout` and are documented in
>   `scripts/unconstrained/README.md`.
> - **Section 9** covers the four baselines added for the rebuttal: gradient
>   guidance and FlowChef (Table 1), Projected Flow and LDF (Fig. 3).

Every number the released unconstrained experiments use, in one place. The
scripts in `scripts/` are the authority; this file explains them.

All four experiments share: **reward `pickscore`**, **image editing**,
**`N = 8`**, **`sample_mode = flow_map2`**, **`endpoint_surrogate = jump`**,
**`drift_detach = True`**, **`warmstart = True`**, **`quantize = int8`**,
**`reward_dtype = fp32`**; no constraints in Experiments 1-4 (Experiment 5 is the constrained run, section 8).

Of those, only `endpoint_surrogate` is passed explicitly by the scripts — the
`RunConfig` default is `rollout`, and `jump` is what makes ConicFlow's cost linear
in `N` rather than quadratic. The rest are already the defaults, and every one
of them is recorded as a column in `summary.csv`, so no row is ambiguous about
what produced it.

---

## 1. Experiment 1 — the headline comparison

All three methods at **equal `N`**, each with its own inner-loop setting.

| | ConicFlow (ours) | HardFlow | FMRG |
|---|---|---|---|
| `control_type` | `conicflow` | `hardflow` | `fmrg` |
| **N** | 8 | 8 | 8 |
| `H` | 2 | 2 | — |
| `n_inner` | 24 | 40 | 3 (`n_opt`) |
| `lr_inner` | 1e-2 | 2.5 | — |
| `hardflow_lr_alpha` | — | 0.5 | — |
| `hardflow_reg` | — | `N/(2λ)`, derived per run | — |
| measured wall clock / run | ~238 s | ~36 s | ~33 s |

- **prompts** 4 editing pairs, **seeds** 317, 1289, 4703
- **λ** a 6-point grid per prompt pair, found by `lambda_autorange.py` and
  **shared** by all three methods (`SHARED_GRID=1`)

**Why equal `N` and not equal wall clock.** Matching wall clock forces a
different `N` per method, and the *uncontrolled* sample depends on `N` strongly
enough to change the subject of the image — in one seed, `r_unc` −1.581 at
`N=8` against +1.234 at `N=100`, a young bearded man in a coat versus an older
man in a suit. The methods would then not be controlling the same problem, and
the finer-grid method starts ~1.4 reward points ahead for free. Equal `N`
removes that, at the cost of a ~6.7× runtime spread. Experiment 2 answers the
objection that raises.

**Why these `n_inner`.**
- **ConicFlow 24.** Halving it to 12 costs about 0.68 `delta_r` (Experiment 2), so
  the inner loop is not where ConicFlow's advantage comes from — Experiment 4 shows
  that is *where* the objective is evaluated, not how long it is optimised.
- **HardFlow 40** is the authors' value. Raising it does not saturate so much
  as destabilise: at 167 (Experiment 2) the achieved reward gets slightly
  *worse*, and its energy frontier is non-monotone at most evaluated levels.
- **FMRG 3** is the published `num_optim_iters` default. `n_opt` does **not**
  change FMRG's control budget — the `n_opt` steps of `(dt/n_opt)·λ·∇r` sum to
  `dt·λ·mean(∇r)` — it trades aggression for fidelity, lowering both reward and
  energy. At fixed `N` the objective gets *worse* as `n_opt` rises, so a large
  value is not "more effort":

  | λ | n_opt=1 | n_opt=3 | n_opt=5 | n_opt=10 |
  |---|---|---|---|---|
  | 10 | **−0.2002** | −0.1963 | −0.1942 | −0.1888 |
  | 100 | **−0.2774** | −0.2692 | −0.2748 | −0.2577 |

**`hardflow_lr_alpha = 0.5`.** HardFlow's step size has an `lr·reg ≥ 1` floor.
Without lifting it, HardFlow cannot reach the low-λ end of the shared grid at
all, so it would be compared only where it happens to be admissible. `alpha`
sets `lr = alpha/reg` below the floor, for the range finder and the sweep
alike. The lr actually used is recorded per row as `hardflow_lr_eff`.

---

## 2. `hardflow_reg` ↔ `λ`

Same physical quantity — control effort — but `λ` **divides** and
`hardflow_reg` **multiplies**, and the conversion carries an N-dependence:

```
hardflow_reg = 1 / (2·λ·dt) = N / (2·λ)        # NOT 1/λ
```

A `reg` matched to a `λ` at one step count is **not** matched at another.
It is computed automatically per config: omit `--sweep_hardflow_reg` and each
run derives its own. `run_preflight` prints the implied equivalent `λ` on every
HardFlow run, so an N-mismatch appears in the log rather than silently.

Two cautions. The correspondence is exact *per step* only: ConicFlow sums its penalty
over an `H`-step window that is re-solved and mostly discarded, so this matches
the marginal cost of control effort, not the total. And `reg > 0` is known to
destabilise HardFlow at very large `n_inner`.

`hardflow_image`'s own function default stays `0.0` deliberately — it documents
the authors' *effective* behaviour, since their nominal `0.0005` is
`.mean()`-scaled to ~1e-9 and numerically inert. `run_sweep.py` always passes
the value explicitly, so the two never conflict.

---

## 3. Experiment 2 — matched wall clock

Identical to Experiment 1 except the inner loops, which were **measured**
against a ~122 s/run target at this `N`, not extrapolated:

| | ConicFlow | HardFlow | FMRG |
|---|---|---|---|
| `n_inner` | 24 → **12** | 40 → **167** | 3 → **13** |
| wall clock / run | 238 → ~122 s | 36 → ~122 s | 33 → ~122 s |

`N`, `H`, `lr_inner`, the λ grid, the seeds and the prompts are unchanged, so
every row pairs 1:1 with a full-compute row from Experiment 1. Cells are the
4 editing prompt pairs at seeds 317 and 1289.

---

## 4. Experiment 3 — planning horizon `H`

ConicFlow only, `H ∈ {1, 4, 6, 8}` against Experiment 1's `H = 2`, everything else
fixed (`n_inner = 24`, `lr_inner = 1e-2`), on Experiment 1's cached λ grids at
seeds 317 and 1289.

`H` enters the per-inner-iteration cost through the horizon rollout, but the
horizon is clipped near the end of sampling (`H_j = min(H, N−j)`), so the
average effective horizon at `N = 8` is not `H`:

```
H=1 -> 1.00   H=2 -> 1.875   H=4 -> 3.25   H=6 -> 4.125   H=8 -> 4.50
```

Read runtime differences against those, not against `H`.

---

## 5. Experiment 4 — `constraint_on`: endpoint vs horizon

ConicFlow only, at exactly Experiment 1's config, varying only where the inner
objective is evaluated. The `endpoint` arm **is** Experiment 1; only the
`horizon` arm is run, so each row is paired at the same prompt, seed and λ.

`constraint_on` is a locked default, so the horizon arm needs
`FLOWMAPS_UNLOCK=constraint_on`; every such row carries `unlocked=constraint_on`
in `summary.csv` and printed a `*** LOCK LIFTED ***` banner.

Measured: the horizon arm costs ~90 s/run against the endpoint arm's ~238 s.
Under `drift_detach`, the differentiated jump is the only transformer graph in
the objective, so `horizon` removes the last transformer backward — which is
both why it is fast and why it optimises a different problem.

---

## 6. Cost models

For planning a sweep without running it. All unconstrained, `N` as stated.

- **ConicFlow**: `NFE = n_inner·N(N−1)/2`, ≈1.09–1.16 s/NFE — reproduces every
  measured config within 1% (`N=10, n_inner=20`: predicted 1133 s, measured
  1137 s). With **`endpoint_surrogate=jump`** the term becomes **linear**:
  `n_inner·N` worth of rollout, plus the `H−1` drift forwards per step. That is
  the whole reason `jump` is used everywhere here — measured at 240 s/run
  against 824 s for `rollout` at the identical `N=8, H=2, n_inner=24`.
- **HardFlow**: **linear** in `N`, ~4.3 s/step unconstrained. Not quadratic.
- **FMRG**: ~1.2 s per `N·n_opt` unit.
- **Per-run overhead**: `≈ 30 + 1.33·N` s on top of `time_ctrl_s` — ~30 s fixed
  (model load, decode, save) plus two `trajectory_rewards` passes that scale
  with `N`.
- **Range finding**: ~14 min per prompt pair, cached across seeds.

---

## 7. Memory gates (measured, in `scripts/gpu_lib.sh`)

On 24 GB cards, from `peak_reserved_flux_gb` on the heaviest config per arm:

| arm | gate |
|---|---|
| `conicflow` | 21000 MiB |
| `hardflow` | 18500 MiB |
| `fmrg` | 21800 MiB |
| `gg`, `flowchef` | 18500 MiB (CLIP is the heaviest reward, 18.8 GB) |
| `pflow` | 22400 MiB (three constraints, 22.8 GB) |
| `ldf` | 21600 MiB (three constraints, 21.9 GB) |

ConicFlow at `N=8, H=2, n_inner=24` peaks at 16.3 GB resident, so **two ConicFlow jobs do
not fit on one card**. Run one job per GPU.

Note `MiB = GB × 953.674`, not `GB × 1024`.

## 8. Experiment 5 — the constrained comparison

No reward (`--reward none`); every attribute prompt is one PickScore
constraint `h_i = tau_i - s_i <= 0` at the endpoint, and each method's
objective is its control energy alone. All values below are what
`scripts/constrained/run_suite.sh` passes; `compare_constrained.py`'s
defaults are the same numbers.

| | value | why |
|---|---|---|
| thresholds | `tau_i = m * s0_i`, `s0_i` = the seed's own uncontrolled score | absolute PickScore values are not comparable across prompts or seeds (baselines run 19.2-22.5 here); a multiplier of the instance's own baseline is. The code's general form is `s0 + (m-1)*abs(s0)`, which stays a tightening for negative baselines and `le` constraints. |
| `m` grid | `1.1 .. 1.5`, 5 points (3 for the three-constraint instance) | `m = 1` is the uncontrolled image; past 1.5 PickScore can be satisfied by near-copies of the prompt for some pairs, so the edit stops being the thing measured. |
| `lambda` | `20`, both arms | not a trade-off knob here: with no reward it only weights energy against `sum mu_i h_i`, i.e. selects which feasible point is reached. 20 keeps HardFlow's `reg = N/(2*lambda) = 0.2` in the range where its inner loop is well conditioned (below). |
| ConicFlow `mu_init` / dual step | `0.5` / `0.5`, multipliers persist across sampling steps | one projected step `mu <- max(0, mu + 0.5 h)` per sampling step, on `h` at the optimiser's horizon endpoint. `mu_init = 0` wastes the first steps (zero gradient until a dual update has happened); the response to `mu_init` saturates by 0.5. A larger dual step helped only at the tightest thresholds. |
| ConicFlow inner loop | `N=8, H=2, n_inner=24, lr=1e-2`, `jump` surrogate, drift detached | as Experiment 1. |
| HardFlow `n_inner` / dual | `40` / `mu_init=0` reset every sampling step, dual step `0.2` every 8 inner steps | the released code's scheme, retained. |
| HardFlow inner `lr` | **`0.5`** (`lr*reg = 0.1`), not the authors' `2.5` | the inner ascent is an explicit fixed-step scheme; its deviation error contracts by `1 - 2*lr*reg*alpha^2` with `alpha -> 1` on the last sampling step. `run_sweep` raises at `lr*reg >= 1`, but at `2.5 * 0.2 = 0.5` the factor is already 0 on that step and the runs diverged (`|u|` in the hundreds, constraint thrown back to its starting violation) on every threshold tried. `0.5` is recorded per row in `hardflow_lr_eff`. |
| HardFlow `rho` | `5`, **held fixed** (`hardflow_rho_mult = 1.0`) | the released code doubles `rho` at each dual update that still sees a violation; with 40 inner steps and a dual update every 8 that is up to `5 * 2^5 = 160` inside a single sampling step, and the failure energies at tight `m` were a monotone function of that peak `rho`. Holding it fixed removed the blow-ups it caused; what remains is the method. |
| `hardflow_lr_alpha` | `0.1`, forwarded | inert here (`lr*reg = 0.1 < 1`); present so the same command stays valid at a `lambda` where the floor would bind. |
| margin / activation | `0` / `0` | as Experiments 1-4. |
| seeds | `310, 698` | two per instance; seeds are not pooled, since a seed fixes the initial latent and hence the problem. |

**Reading `summary.csv` for this experiment.** `r_unc`/`r_ctrl` are 0 (no
reward). The constraint values are `h_0 .. h_{C-1}` on the final latent, with
`sat_i`/`viol_i`; feasibility is `max_i h_i <= 0`. `ctrl_energy` is the
displacement residual `sum_k ||dx_k||^2 dt`; the paper's `sum_k ||u_k||^2 dt`
is that times `N^2 = 64` (`plot_constrained_suite.py` applies it).

**A guard that used to fire on HardFlow.** `run_sweep` flags a row
`run_ok=False` with `stale_decode_cache_signature` when a step's constraint
value equals the previous step's endpoint value, which would mean a reused
decode. At the final step that equality is legitimate: the nominal step to
`t = 0` is the same flow-map jump as the endpoint prediction, so a method that
applies no correction there ends exactly on the previous endpoint. The flag was
a false positive on 2 of 86 HardFlow rows in the paper's suite (their numbers
are complete and correct; `compare_constrained.py` drew no figure for those
points, `plot_constrained_suite.py` read the rows regardless). Since
2026-10-11 the guard exempts a step whose LATENT is bit-identical to the
previous endpoint, so new runs are not flagged; equal `h` from different
latents is still caught.

**Feasibility in the figures** is always the in-run value above, for every
method: Fig. 3 (left) and, since 2026-10-21, the image panels too. Re-scoring
the saved 8-bit PNG (`comparison.json`) lowers PickScore by a median
0.01-0.04, which would flip runs that end within a few hundredths of the
boundary.

## 9. The baselines (rebuttal): gradient guidance, FlowChef, Projected Flow, LDF

Each is a `--sweep_control_type` in `run_sweep.py`, implemented next to
`hardflow_image` in `src/control_loops.py`; each function's docstring names its
source code and lists every deviation from it. All four advance with the same
nominal flow-map step as the other methods, so `ctrl_energy` measures the same
thing for every arm.

### Unconstrained (Table 1): `gg`, `flowchef`

Run with `METHODS="gg flowchef" scripts/unconstrained/run_table1.sh`, on the
same problems, seeds and frozen λ grids as the paper's three methods.

| | gradient guidance (`gg`) | FlowChef (`flowchef`) |
|---|---|---|
| source | HardFlow's "Gradient Guidance" baseline (`azizanlab/HardFlow@016adb2`, `utils/methods.py:210-241`) | the FMRG authors' FLUX port of FlowChef (Patel et al., ICCV 2025), `fmrg/fluxfm_sampler_ip.py:736-828`, reward in place of the inverse-problem loss |
| per sampling step | `n_inner` ascent steps on the reward of the one-step clean estimate, differentiated through the transformer, then the nominal step | `n_inner` ascent steps on the clean estimate itself (gradient through the decoder and reward only), the displacement added to the nominal step |
| what `lam` is | a gain: step `lam * dt / n_inner`, FMRG's convention | the same |
| `n_inner` | **3** (FMRG-J's guidance steps) | **5** (the FMRG authors' setting) |
| transformer calls at N=8 | `N (n_inner + 1)` = 32, backward on 24 | `2N` = 16, no backward |
| control time per run (A5000, N=8) | ~30-38 s | ~11 s |

On the paper's 12 problems both frontiers extend past the shared budget
`[E1, E2]` at both ends, so adding them leaves the other methods' `G` unchanged
(`analysis/table1_breakdown.py --baselines` reports this per problem).

### Constrained (Fig. 3): `pflow`, `ldf`

Run with `ARMS="pflow ldf" scripts/constrained/run_suite.sh` into the same
trees as ConicFlow/HardFlow, so all four share the probe, thresholds and seeds.
Both are constraint-only (`--reward none`); neither uses `lambda` or a horizon.

| | Projected Flow (`pflow`) | LDF (`ldf`) |
|---|---|---|
| source | Projected Diffusion Models (Christopher, Baek & Fioretto, NeurIPS 2024); Zampini et al., NeurIPS 2025 for latent models. The projector is a PyTorch port of LDF's Gauss-Newton `project_batch` | Lagrangian Dual Flows (Kurtz & Davydov 2026), the inequality closed-form-slack flow of `cfm/methods/ldf.py` |
| per sampling step | the nominal step, then a Gauss-Newton projection of the STATE onto `{h(X(z, t)) <= 0}`, with the Jacobian through the flow-map endpoint `X` | the nominal step, then the correction and multiplier ODE integrated over the same interval in midpoint substeps |
| knobs | `pflow_max_iters` **10**, `pflow_final_iters` **50**, `pflow_activation` 0 (project from the first step, as PDM), `pflow_margin` **0** | `ldf_penalty` 5, `ldf_rescale` 1, `ldf_p` 1 (the authors' values), `ldf_substeps` **64**, initial multiplier 0 |
| time per run (A5000) | 80-140 s with one or two constraints, ~190 s with three | 45-70 s |

Why the settings in bold:
- **Projection in bf16.** The latent is bf16, and the min-norm correction for a
  small violation (~5e-5 per element) is below a bf16 spacing, so a plain
  Gauss-Newton step leaves the evaluated point unchanged. The projector keeps an
  fp32 master copy and sizes each step so that the step *as rounded* is
  predicted to remove at least 95% of every active violation; in fp32 it is the
  plain Gauss-Newton step.
- **`pflow_final_iters` 50.** The last step's projection acts on the output
  image and is cheap (decoder and scorer only, ~0.2 s per update); one hard ask
  needed 19 updates, past the shared cap of 10.
- **No margin.** The projection stops only when the true `h <= 0` (not its
  linearization), so it ends just inside the boundary (median final
  `max_i h_i` -0.005 over the suite). That is why feasibility is judged in-run:
  re-scored from the 8-bit PNG, most of these runs read a few hundredths
  infeasible. HardFlow's released code tightens every constraint by 0.02 to
  make its augmented-Lagrangian solve land inside; the paper runs HardFlow
  without it (section 8), and Projected Flow, which needs no margin to be
  feasible, gets none either.
- **64 LDF substeps.** The authors integrate ~100 Euler steps of the whole ODE;
  at N=8 the late-time gain `1/(1-s)^p` makes few substeps overshoot. At 16 the
  hardest asks (m = 1.5) failed or cost 3x the energy, and raising the gain
  instead made the explicit substeps diverge; 64 at the authors' gain was
  feasible on both at 2-3x lower energy. Chosen on two pilot cells (p01_smile
  and p09_curly_old, seed 310), as ConicFlow's dual settings were chosen on a
  pilot instance.

OC-Flow (HardFlow's other baseline) was also ported and is not included: on
FLUX at N=8 no update rule tried beat the uncontrolled sample.
