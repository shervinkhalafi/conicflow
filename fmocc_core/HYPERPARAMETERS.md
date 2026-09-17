# Hyperparameters actually used in recent runs

A snapshot of what the numbers have been set to, and why. Current as of
**2026-09-15**. Where a value was decided on a call, that is noted — several of
these were changed deliberately and the reasoning matters more than the number.

---

## 1. Current unconstrained suite (the λ sweep)

This is what `lambda_suite_v2_20260915.sh` runs. **All three methods share
capital N = 10**, and wall clock is deliberately *not* matched.

| | MPC | HardFlow | FMRG |
|---|---|---|---|
| `control_type` | `mpc` | `hardflow` | `fmrg` |
| **N** | 10 | 10 | 10 |
| `H` | 4 | 2 | — |
| `n_inner` | 12 | 40 | 3 |
| `lr_inner` | 1e-2 | 2.5 | — |
| `hardflow_reg` | — | `5/λ`, see §2 | — |
| measured wall clock / run | ~703 s | ~86 s | ~87 s |

- **reward** `imagereward`, **unconstrained**, `sample_mode=flow_map2`
- **λ** `0.1, 1, 10, 50, 200, 1000, 3000, 10000`
- **seeds** 42, 43, 44
- **prompts** `a portrait of a man` → `a smiling face`; `a dog` → `a dog wearing sunglasses`

**Why equal N rather than equal wall clock.** Matching wall clock forced a
different N per method (MPC 8, HardFlow 100, FMRG 40), and the uncontrolled
sample turned out to depend on N strongly enough to change the *subject* of the
image — one seed gave `r_unc` −1.581 at N=8 versus +1.234 at N=100, a young
bearded man in a coat versus an older man in a suit. The methods were therefore
not controlling the same problem, and HardFlow started ~1.4 reward points ahead
purely from a finer grid. Equal N removes that entirely, at the cost of a ~8×
spread in runtime.

**Why these `n_inner`.**
- MPC 12 — screened against N=10/`n_inner`=20 and gave an indistinguishable
  objective for 2.4× less compute.
- HardFlow 40 — the authors' value. Raising it to 260 does not saturate, it
  *destabilises*: control energy went from 3.79 to 5.98e9 with two constraints,
  and runtime 5–7×, with more failures.
- FMRG 3 — Boffi's published `num_optim_iters` default. `n_opt` does **not**
  change FMRG's control budget (the `n_opt` steps of `(dt/n_opt)·λ·∇r` sum to
  `dt·λ·mean(∇r)`); it trades aggression for fidelity, lowering both reward and
  energy. The only clean ablation at fixed N shows the objective getting *worse*
  as `n_opt` rises, so a big value is not "more effort":

  | λ | ni=1 | ni=3 | ni=5 | ni=10 |
  |---|---|---|---|---|
  | 10 | **−0.2002** | −0.1963 | −0.1942 | −0.1888 |
  | 100 | **−0.2774** | −0.2692 | −0.2748 | −0.2577 |

---

## 2. `hardflow_reg` per λ (N = 10, lr = 2.5)

`reg = N/(2λ)`. **Computed automatically as of 2026-09-16** — omit
`--sweep_hardflow_reg` and each config derives its own. The table below is kept
as the expected values, and is reproduced exactly by the derivation (verified
2026-09-16 against every row), so it remains the auditable reference even though
nothing is now maintained by hand.

| λ | reg | `lr·reg` | status |
|---|---|---|---|
| 0.1 | 50.0 | 125.0 | **raises** |
| 1 | 5.0 | 12.5 | **raises** |
| 10 | 0.5 | 1.25 | **raises** |
| 50 | 0.1 | 0.25 | ok |
| 200 | 0.025 | 0.0625 | ok |
| 1000 | 0.005 | 0.0125 | ok |
| 3000 | 0.0016666667 | 0.00417 | ok |
| 10000 | 0.0005 | 0.00125 | ok — also the authors' nominal value and `RunConfig`'s default |

`hardflow_image` raises when `lr·reg ≥ 1`, so HardFlow covers 5 of the 8 λ
points. The three blocked points are still launched once each, single attempt,
so the guard's message is recorded rather than the gap being unexplained.

**Runs before 2026-09-09 used `hardflow_reg=0.0`** — no effort penalty at all —
so their energy metrics are not comparable with later ones. Pass
`--sweep_hardflow_reg 0.0` to reproduce them; an explicit value still overrides
the derivation and is never recomputed.

**Runs between 2026-09-09 and 2026-09-16** used the fixed `5e-4` default unless
the flag was passed. That is `N/(2λ)` only at `N=10, λ=10000`, so any such run
at another `(N, λ)` was not matched to its own `λ` — check the `hardflow_reg`
and `N` columns before comparing one against an MPC arm.

---

## 3. Constrained runs (most recent settings)

The constrained work is not what the current suite runs, but these are the
values it last used:

| | |
|---|---|
| control | `mpc` (also supported: `hardflow`) |
| `N`/`H` | 10 / 4, also 8/4 and 6/4 in budget probes |
| `n_inner` | 20 (10 and 12 in budget probes) |
| `lam` | 10000 |
| reward | `lpips` (steering came from the constraints) |
| constraints | `clip` + `imagereward`, both `ge` |
| thresholds | CLIP 0.43 / 0.46 / 0.49, ImageReward 1.5 |
| `lr_dual` | 0.2 (0.1 in older runs) |
| `mu_init` | 0.0 |
| HardFlow `margin` | 0.02 — the authors' value, needed because HardFlow re-inits μ every step and otherwise only approaches the threshold asymptotically. MPC's μ accumulates and does not need it. |

λ=10000 was found best for ImageReward-constrained runs: lower λ was too
conservative to meet the threshold, higher λ overshot and raised energy.

---

## 4. Measured cost models

Useful for planning a sweep without running it. All unconstrained, N as stated.

- **MPC**: `NFE = n_inner·N(N−1)/2`, ≈1.09–1.16 s/NFE. Reproduces every
  measured config within 1% (N=10/ni=20: predicted 1133 s vs measured 1137 s).
  **Updated 2026-09-16**: the formula previously read `n_inner·(N(N−1)/2 + H)`.
  That `+ H` was not the horizon's cost — it was `endpoint_map` running one
  `dt=0` transformer forward *and its backward* at every outer step with
  `j ≥ N−H`, producing `z` bit-for-bit. It has been removed, so subtract
  `n_inner·H` from any pre-2026-09-16 prediction (8% of the N=10/H=4 suite
  config, 25% of an N=4/H=2 smoke run).
  With `endpoint_surrogate=jump` the term becomes **linear**: `NFE = n_inner·N`
  worth of rollout plus the `H−1` drift forwards per step.
- **HardFlow**: **linear** in N, ~4.3 s/step unconstrained (~8.3 s/step with two
  constraints). It is *not* quadratic.
- **FMRG**: ~1.2 s per `N·n_opt` unit.
- **Per-run overhead**: `≈ 30 + 1.33·N` seconds on top of `time_ctrl_s` — ~30 s
  fixed (model load, decode, save) plus two `trajectory_rewards` passes that
  scale with N. 43 s at N=10.
- **Inter-job gap**: ~60 s per job when driven by `gpu_lib`'s `pick_gpu`
  (3 stability polls at `POLL_SEC=30`). Override `POLL_SEC` if a run is tight.

Constrained runs are roughly 2× the unconstrained control time at the same
config, because each inner step also evaluates the constraint gradients.

---

## 5. Memory gates (measured, in `gpu_lib.sh`)

On 24 GB cards, from `peak_reserved_flux_gb` on the heaviest config per arm:

| arm | gate |
|---|---|
| `mpc` | 21000 MiB |
| `hardflow` | 18500 MiB |
| `fmrg` | 21800 MiB |

The FMRG gate is the weakest evidence of the three — derived from CLIP-reward,
N=10, `n_inner`=5 rows. Re-derive it from a real ImageReward run's
`peak_reserved_flux_gb` and update `gpu_lib.sh`, not a copy.

Note `MiB = GB × 953.674`, not `GB × 1024`; the pre-2026-09-05 constants used
the wrong factor and over-stated every gate by 7.4%.
