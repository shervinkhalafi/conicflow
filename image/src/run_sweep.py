"""
run_sweep.py — Controlled vs uncontrolled FLUX image generation sweep
======================================================================

Runs every combination in SWEEP_CONFIGS, saves images + a CSV summary.

Usage
-----
    # inside the flowmaps conda env, with the sampler already loadable:
    python run_sweep.py [--out_dir results] [--seed 42] [--quiet]

Requires
--------
  - sampler loaded via FluxFlowMapSampler (configured below, or pass --hf_token)
  - FMRG repo on PYTHONPATH (set FMRG_DIR below or in the environment)
  - CLIP model cached at HF_HOME (auto-downloaded on first run)

Sweep dimensions
----------------
  prompt, N (total steps), H (horizon), control_type, lam, sample_mode
  Add/remove entries in SWEEP_CONFIGS to taste.
"""

import argparse
import csv
import gc
import itertools
import json
import math
import os
import re
import sys
import time
import weakref
from dataclasses import dataclass, asdict, field, replace as _dc_replace
from pathlib import Path
from typing import Callable, List, Optional

# Set BEFORE torch is imported -- the allocator reads this once at init, so
# setting it later is a silent no-op. Recommended by the OOM message itself;
# ConicFlow reserves ~0.3 GB more than it allocates and this reclaims part of that
# fragmentation gap. Two of the ~25 launcher scripts exported it, so whether a
# sweep got it depended on which script you happened to use; setting it here
# makes it unconditional. An explicit env var still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt

# --reward_dtype -> torch dtype. fp32 is the default and matches every result
# currently on disk; see the flag's help text for the measured savings.
_REWARD_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# ── path setup ───────────────────────────────────────────────────────────────
# `image_conicflow.py` is vendored next to this file, so FLOW_MATCHING_DIR defaults to
# this directory and nothing needs to be set for it.
#
# FMRG (github.com/jrrhuang/fmrg, Apache-2.0) is vendored at the repository
# root as `fmrg/`, with the local changes listed in fmrg/MODIFICATIONS.md, so
# FMRG_DIR defaults to it. Its FLUX FlowMap LoRA weights are not in git: fetch
# them with `bash fmrg/checkpoints_download.sh`. The only symbols taken from it
# are the sampler and the reward-model wrappers in `fluxfm_sampler_reward.py`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
FLOW_MATCHING_DIR = os.environ.get("FLOW_MATCHING_DIR", _HERE)
FMRG_DIR = os.environ.get("FMRG_DIR", os.path.join(_REPO_ROOT, "fmrg"))
sys.path.insert(0, FMRG_DIR)
sys.path.insert(0, FLOW_MATCHING_DIR)           # image_conicflow.py lives here
sys.path.insert(0, _HERE)

from image_conicflow import (
    make_flux_wrappers,
    fmrg_j_image,
    trajectory_rewards,
    tune_sampler_speed,
)
# receding_horizon_flow_control and hardflow_image come from control_loops.py,
# NOT from image_conicflow.py: the ConicFlow loop there has no `n_alt` (primal/dual
# alternation per fixed window) and no HardFlow at all. See control_loops.py's
# module docstring for the full list of divergences. Every other import above
# still comes from image_conicflow.py unmodified.
from control_loops import receding_horizon_flow_control, hardflow_image


def stack_constraints(fns):
    """Combine scalar constraint functions into one (B, C) callable.

    Inlined from conicflow_cell.py to avoid importing that module's demo code.
    Each fn must return (B,) or (B,1); the result is (B, C) column-stacked.
    """
    def combined(x):
        cols = []
        for fn in fns:
            v = fn(x)
            if v.dim() == 1:
                v = v.unsqueeze(-1)
            cols.append(v)
        return torch.cat(cols, dim=-1)
    return combined


from fluxfm_sampler_reward import (
    FluxFlowMapSampler,
    RewardEnsemble,
    CLIPReward,
    HPSReward,
    ImageRewardReward,
    PickScoreReward,
    create_clip_preprocess,
    HF_CACHE,
)

try:
    from transformers import AutoProcessor
except ImportError:
    raise SystemExit("transformers not installed. Run: pip install transformers")

# PickScore's `AutoModel.from_pretrained("yuvalkirstain/PickScore_v1", ...)`
# is a plain CLIP checkpoint and never touches timm -- but loading it makes
# transformers 4.53's model registry import Gemma3n's config module, which
# does `from timm.data import ImageNetInfo` when timm is present. ImageReward's
# BLIP backbone needs the legacy timm==0.6.13 API, which predates ImageNetInfo,
# so that probe crashes with an unrelated ImportError. Tell transformers timm
# isn't available (harmless -- PickScore doesn't use it) instead of pinning a
# newer timm, which would break ImageReward's BLIP import in this shared env.
import transformers.utils.import_utils as _tf_import_utils
_tf_import_utils._timm_available = False

REWARD_CHOICES = ["clip", "hpsv2", "pickscore", "imagereward", "style", "lpips", "none"]

# Hard cap on the number of simultaneous constraints. The CSV records each
# constraint in its own numeric column (h_0..h_3 / mu_0..mu_3 / thr_0..thr_3)
# rather than a comma-joined string, so the column set is fixed at import time.
# Exceeding this RAISES -- it is never truncated and never warned-and-continued:
# the requirement is "arbitrary number of constraints", so hitting an arbitrary
# implementation cap is an error condition, not a degraded mode. Raising the cap
# is a one-line change here (CSV_KEYS is generated from it).
MAX_CONSTRAINTS = 4

# Control types that accept inequality constraints. "conicflow" is ours; "hardflow"
# is the HardFlow baseline (Li, Alim & Azizan, TPAMI 2026), added 2026-08-31 as
# the constrained comparison -- see the HardFlow port audit. "fmrg" and
# "boffi_fmrg" are unconstrained methods and are deliberately NOT in this set:
# requesting constraints with them raises in _validate_constraint_cfg rather
# than silently running unconstrained.
CONSTRAINED_CONTROL_TYPES = frozenset({"conicflow", "hardflow"})

# Settings the design plan ("Locked decisions -- do not re-litigate") fixes,
# not merely defaults.
#
# drift_detach WAS one of them (locked at False as the known-correct setting,
# with speed as True's only expected benefit). Lifted 2026-09-16: measurement
# showed True is both faster and BETTER on the objective, so the rationale no
# longer holds and the axis is open again. See _RETIRED_LOCKS below.
#
#   constraint_on="endpoint" -- resolved a disagreement between two
#                           development copies. A WRONG default here once made an entire
#                           ablation (horizon-mode) structurally
#                           uninterpretable after the fact -- the exact failure
#                           _assert_locked_defaults exists to catch before it
#                           happens again, not after.
#   warmstart=True      -- Step 7's dropped third candidate (a non-dynamics-
#                           exact warm start) was ruled out specifically
#                           because it conflicts with this.
_LOCKED_DEFAULTS = {"constraint_on": "endpoint", "warmstart": True}

# Locks that USED to exist and have since been lifted deliberately. Naming one
# in FLOWMAPS_UNLOCK is accepted and does nothing, instead of raising the way an
# unrecognised name does -- a launcher written against the old lock set should
# not fail, it should just stop having anything to lift.
#
# drift_detach, lifted 2026-09-16. The lock's recorded rationale (2026-08-25)
# was "True's only benefit is speed, and the speed gain is small". Both legs
# were measured and both were wrong: at N=8/H=4/n_inner=6 with ImageReward,
# drift_detach=True ran 33% faster (157 s vs 236 s) AND produced a BETTER
# objective at every lambda tested -- J -1.096 vs -0.756 at lam=1, which is
# ~6 run-to-run standard deviations, plus -0.04 at lam=10 and -0.01 at lam=100.
# A closed decision stays closed on its reasoning, not its conclusion; the
# reasoning no longer held, so the axis is open again rather than pinned to the
# opposite value. See results/effstudy/ for the run data.
_RETIRED_LOCKS = frozenset({"drift_detach"})


def _assert_locked_defaults(cfg, log=print) -> None:
    """Print the resolved config for the three locked settings and RAISE if
    any diverges (the design plan's acceptance criterion verbatim: "a trivial run
    prints all three correctly, and the assertions fire if any is changed").

    RAISES rather than silently running under a divergent config -- the
    standing rule (loud fail only) applied to config, not just data. The CLI
    still exposes --sweep_constraint_on / --sweep_warmstart, so nothing here
    removes the ability to SET these fields -- setting them away from the
    locked values now trips this assert instead of quietly reproducing a run
    under a config nobody meant to reintroduce.

    --sweep_drift_detach is no longer one of them: that lock was lifted on
    2026-09-16 after measurement reversed its rationale, so the axis sweeps
    freely again. See _RETIRED_LOCKS.

    THE ESCAPE HATCH (added 2026-09-08). Unlocking now requires naming the
    field explicitly, per run, via ``FLOWMAPS_UNLOCK=drift_detach`` in the
    environment (comma-separated for several). It is deliberately an env var
    and not a ``--sweep_*`` flag: a sweep axis would let a lock come off as a
    side effect of a grid definition, which is the failure this guard exists to
    prevent, whereas an env var has to be set by the person launching and is
    recorded in the launcher.

    Every unlock prints a banner and is written to the CSV row's
    ``unlocked`` column, so a run made under a lifted lock is never
    indistinguishable afterwards from one made under the lock -- which was the
    whole point of the original assert.

    WHY drift_detach is being unlocked today: as of 2026-09-08 ConicFlow wall
    clock is "priority one". The lock's recorded rationale (2026-08-25) was "True's
    only benefit is speed, the speed gain is small" -- a judgement made when
    speed did not matter, and made one day BEFORE the 2026-08-26 endpoint-gradient
    fix that turned ``endpoint_map(..., grad=True)`` into the sole gradient path
    through the transformer. Both legs of that rationale have moved, so the
    number is worth re-measuring. This does NOT reopen the decision: the default
    is unchanged, and any run that lifts the lock is labelled as such.
    """
    unlocked = _unlocked_fields()
    resolved = {k: getattr(cfg, k) for k in _LOCKED_DEFAULTS}
    log(f"  [locked-defaults] resolved config: {resolved}")
    bad = {k: v for k, v in resolved.items()
           if v != _LOCKED_DEFAULTS[k] and k not in unlocked}
    lifted = {k: resolved[k] for k in unlocked
              if resolved[k] != _LOCKED_DEFAULTS[k]}
    if lifted:
        log(f"  [locked-defaults] *** LOCK LIFTED via FLOWMAPS_UNLOCK for "
            f"{lifted} (locked value(s): "
            f"{({k: _LOCKED_DEFAULTS[k] for k in lifted})}). This run is NOT "
            f"comparable to the locked-default corpus on any quantity these "
            f"fields affect. ***")
    if bad:
        raise ValueError(
            f"config diverges from the plan's locked defaults (see "
            f"the design plan, 'Locked decisions -- do not re-litigate'): "
            f"got {bad}, expected {({k: _LOCKED_DEFAULTS[k] for k in bad})}. "
            f"These are closed decisions, not sweepable axes. To lift a lock "
            f"deliberately, set FLOWMAPS_UNLOCK to a comma-separated list of "
            f"field names (e.g. FLOWMAPS_UNLOCK=drift_detach); the run is then "
            f"labelled in the CSV's `unlocked` column."
        )


def _unlocked_fields() -> frozenset:
    """Locked-default fields the operator has explicitly lifted, from
    ``FLOWMAPS_UNLOCK``. An unknown name RAISES rather than being ignored: a
    typo'd unlock would otherwise read as "I unlocked it" while the assert
    still fired, or worse, silently unlock nothing.
    """
    raw = os.environ.get("FLOWMAPS_UNLOCK", "").strip()
    if not raw:
        return frozenset()
    names = {n.strip() for n in raw.split(",") if n.strip()}
    # A retired lock is accepted and ignored (see _RETIRED_LOCKS): the field is
    # simply no longer locked, so there is nothing to lift and nothing to warn
    # the reader about afterwards -- but an old launcher naming it should not
    # die. Only genuinely unrecognised names still raise.
    retired = names & _RETIRED_LOCKS
    if retired:
        print(f"  [locked-defaults] FLOWMAPS_UNLOCK names {sorted(retired)}, "
              f"which is no longer a locked default -- ignoring. It is now an "
              f"ordinary sweepable field.")
    names -= _RETIRED_LOCKS
    unknown = names - set(_LOCKED_DEFAULTS)
    if unknown:
        raise ValueError(
            f"FLOWMAPS_UNLOCK names {sorted(unknown)}, which are not locked "
            f"defaults. Locked fields are {sorted(_LOCKED_DEFAULTS)}; "
            f"retired (accepted, no-op) are {sorted(_RETIRED_LOCKS)}."
        )
    return frozenset(names)


def _assert_boffi_rng_reachability_closed(cfg, latent: Optional[torch.Tensor]) -> None:
    """Raise unless every stochastic torch.randn(...) site inside upstream's
    sample_reward_guided (fmrg/fluxfm_sampler_reward.py) is structurally
    unreachable for this call.

    WHY THIS EXISTS (2026-08-25, following on from the numerical-inertness
    check in the development log's Step 1 write-up, section 1.4d). That check
    found CLIPReward construction ADVANCES the global RNG, and concluded the
    constraint-CLIP-loading narrowing is inert only because nothing
    DOWNSTREAM of the load consumes global RNG on the paths run_sweep.py
    currently drives. That conclusion was a SNAPSHOT verified against today's
    configuration, not a structural guarantee -- two of its three legs are
    settings, not facts:
      - warmup_steps=0 is a RunConfig default with no CLI override today, but
        Step 4 adds CLI surface generally and nothing stops a future flag
        reaching this field.
      - noise_mode="deterministic" is a default with no override today, for
        the same reason.
    If either becomes reachable, sample_reward_guided draws from the SAME
    global RNG stream that a constraint-CLIP load may have already advanced,
    and a downstream sampling pass (most concretely Step 3's calibration
    pass, which exists specifically to characterize an UNCONTROLLED reward
    distribution from ~100 seed-varied samples) could silently draw from a
    shifted stream relative to the runs it is meant to calibrate. A
    normalization computed against the wrong distribution is invisible in
    every downstream number -- there is no symptom to notice it by.

    So: not a comment recording that this was checked once. An assert that
    fails loudly the moment any of the three enabling conditions stops
    holding, so a stochastic path becoming reachable is a decision that trips
    this (and requires deliberately editing it), not a change that alters
    results quietly. Step 3's calibration script must carry the same guard
    when it is built -- it is a NEW code path driving the sampler and is not
    covered by this call site.

    The three conditions mirror the exact gating in sample_reward_guided:
      - L1065 torch.randn(latent_shape, ...)   gated on `latent is None`
      - L1127 torch.randn(latent_shape, ...)   gated on
              `warmup_steps > 0 and warmup_particles > 1 and step_size > 0`
      - L1363 torch.randn_like(z1_guided)      gated on
              `noise_mode == "stochastic"`
    (line numbers as of fmrg/fluxfm_sampler_reward.py, checked 2026-08-25).
    """
    if latent is None:
        raise ValueError(
            "boffi_fmrg RNG-reachability guard: latent is None, which would "
            "make sample_reward_guided draw its own torch.randn(...) instead "
            "of using the caller-provided z0 -- reachable, unverified RNG "
            "consumption. run_sweep.py must always pass an explicit latent."
        )
    if cfg.boffi_noise_mode != "deterministic":
        raise ValueError(
            f"boffi_fmrg RNG-reachability guard: boffi_noise_mode="
            f"{cfg.boffi_noise_mode!r}, but only 'deterministic' is verified "
            f"RNG-inert (see _assert_boffi_rng_reachability_closed's "
            f"docstring). 'stochastic' reaches a torch.randn_like(...) draw "
            f"that this guard exists to keep unreachable until it is a "
            f"deliberate decision, not a quiet config change."
        )
    if cfg.boffi_warmup_steps > 0 and cfg.boffi_warmup_particles > 1 and cfg.lam > 0:
        raise ValueError(
            f"boffi_fmrg RNG-reachability guard: boffi_warmup_steps="
            f"{cfg.boffi_warmup_steps}, boffi_warmup_particles="
            f"{cfg.boffi_warmup_particles}, lam={cfg.lam} together enable "
            f"upstream's best-of-N particle reinitialization, which draws a "
            f"fresh torch.randn(...) per particle. (Note: this combination "
            f"already raises NotImplementedError elsewhere in this file for a "
            f"DIFFERENT reason -- the trajectory-reconstruction hook assumes "
            f"exactly one reward_consistency_xt call per outer step -- but "
            f"that is coincidental, not a guarantee. This check exists so "
            f"the RNG invariant does not silently depend on an unrelated "
            f"assumption remaining true.)"
        )

# ── plausibility ranges: EXTRACTED to reward_ranges.py (2026-08-25) as a
# single source of truth shared with reward_normalization.py's low-spread
# check, which needs the same per-reward native-scale bounds to judge
# whether a calibration std/rms is too small relative to them. Kept under
# the old `_REWARD_PLAUSIBLE_RANGE` name here (an import alias) so no other
# line in this file needed to change. See reward_ranges.py for the
# per-reward provenance comments (moved there verbatim, not summarized).
from reward_ranges import REWARD_PLAUSIBLE_RANGE as _REWARD_PLAUSIBLE_RANGE
import reward_normalization as _rn

# Step 4.2: explicit per-constraint direction flag. Never inferred from a
# reward's name or sign -- CLIP/hpsv2/pickscore/imagereward/style all return
# higher-is-better scores from _build_reward_fn (style already negates its
# raw distance internally), so "ge" (score >= threshold satisfied) is the
# common case; "le" exists for the rare inverted case and must be requested
# explicitly, never guessed.
VALID_CONSTRAINT_DIRECTIONS = ("ge", "le")

# Default StatsStore JSON backing ConstraintSpec normalizers. Single source
# of truth shared by ConstraintSpec's own field default and the general-path
# CLI's --constraint_stats_json default, so the two cannot drift apart.
DEFAULT_STATS_JSON = "calibration_stats.json"

# Reward types whose ConstraintSpec.arg is a TEXT target (scored against the
# image), as opposed to "style" (arg = a reference image path) or "lpips"
# (no free arg -- its reference is always the same-seed uncontrolled
# rollout, via ref_holder). Used by ConstraintSpec.__post_init__'s
# empty-arg guard and by the general-path CLI's own validation, so the
# two cannot disagree about which types need a text argument.
_TEXT_TARGET_REWARD_TYPES = ("clip", "hpsv2", "pickscore", "imagereward")


@dataclass(frozen=True)
class ConstraintSpec:
    """One generalized constraint: h(z) <= 0 <=> satisfied.

    Step 4.1's "per-constraint spec": reward type, its argument, threshold,
    explicit inequality direction, and an optional normalizer binding loaded
    from the Step 3 stats JSON. Replaces the CLIP-only
    ``constraint_prompts``/``constraint_thresholds`` pair -- those two fields
    stay on RunConfig for backward compatibility (see
    ``_resolve_constraint_specs``) and are resolved into a list of
    ``ConstraintSpec`` with ``reward_type="clip"`` before this dataclass is
    ever touched, so no existing invocation (CLI or SWEEP_CONFIGS) changes
    behavior.

    Fields:
        reward_type: one of REWARD_CHOICES, excluding "none" (a constraint
            with no scorer is meaningless).
        arg: the reward-type-specific argument -- a text prompt for
            clip/hpsv2/pickscore/imagereward, a style reference image path
            for "style". "lpips" has no free argument (its reference is
            always the same-seed uncontrolled rollout, wired through
            ref_holder exactly as the lpips REWARD path already does) --
            `arg` is ignored for lpips and should be left "".
        threshold: raw-scale (or normalized-scale, if `normalize=True)
            threshold value.
        direction: "ge" (h = threshold - score; satisfied when score >=
            threshold) or "le" (h = score - threshold; satisfied when score
            <= threshold). No default inference -- must be set explicitly
            per 4.2.
        normalize: if True, the score is normalized via a Step 3
            ``reward_normalization.Normalizer`` (built from
            ``calibration_stats.json``) BEFORE the threshold comparison, so
            `threshold` is then in normalized units. Defaults to False:
            every constraint usage before Step 4 (CLI `--constraint_prompts`
            / `--constraint_thresholds`, and every planned Step 7/8/10 config
            drafted so far) passes RAW-scale thresholds (e.g. CLIP cosine
            0.28), and flipping the default to True would silently
            reinterpret those same numbers in normalized units -- exactly
            the kind of silent reinterpretation the standing rules forbid.
            Turning this on is an explicit per-constraint opt-in once a
            normalized threshold is actually intended.
        bind_mode: "global" or "prompt" -- only consulted when
            normalize=True; passed straight through to
            ``reward_normalization.build_normalizer``. Per Step 3.4's
            provisional recommendation, "global" is the default.
        stats_json: path to the StatsStore JSON backing the normalizer. Only
            consulted when normalize=True.
    """
    reward_type: str
    arg: str
    threshold: float
    direction: str
    normalize: bool = False
    bind_mode: str = "global"
    stats_json: str = DEFAULT_STATS_JSON

    def __post_init__(self):
        if self.reward_type not in REWARD_CHOICES or self.reward_type == "none":
            raise ValueError(
                f"ConstraintSpec.reward_type={self.reward_type!r} invalid -- "
                f"must be one of {[c for c in REWARD_CHOICES if c != 'none']}."
            )
        if self.direction not in VALID_CONSTRAINT_DIRECTIONS:
            raise ValueError(
                f"ConstraintSpec.direction={self.direction!r} invalid -- must "
                f"be one of {VALID_CONSTRAINT_DIRECTIONS}. Not inferred from "
                f"the reward's sign (plan 4.2) -- set explicitly."
            )
        if self.reward_type == "style" and not self.arg:
            raise ValueError(
                "ConstraintSpec(reward_type='style') requires `arg` = a style "
                "reference image path."
            )
        # Text-target reward types: an empty `arg` is NOT a legal "unset" --
        # _build_constraint_fn passes it through as cfg.edit_prompt, and
        # _build_reward_fn resolves `cfg.edit_prompt or cfg.prompt`, so an
        # empty arg would SILENTLY score the constraint against the
        # generation prompt instead of the intended target. That is a
        # different constraint than the one asked for, with nothing in the
        # logs saying so -- the exact silent-substitution class the standing
        # rules forbid. (Added 2026-08-26 alongside the general-path CLI,
        # which made an empty arg reachable for the first time; the legacy
        # constraint_prompts path always supplied a non-empty prompt.)
        if self.reward_type in _TEXT_TARGET_REWARD_TYPES and not self.arg:
            raise ValueError(
                f"ConstraintSpec(reward_type={self.reward_type!r}) requires a "
                f"non-empty `arg` (the text target to score against). An "
                f"empty arg would silently fall back to the generation "
                f"prompt via `cfg.edit_prompt or cfg.prompt`, producing a "
                f"different constraint than intended with no error."
            )

# ── NaN/corruption detection (added 2026-08-20 after an unexplained r_unc
# drift: a run produced NaN reward/energy, and r_unc for the *same*
# (prompt, seed) silently changed on the next run in the same process,
# with nothing flagging it -- see conversation history for the writeup).
# Detection/logging only; does not change any numerics or attempt recovery.
_R_UNC_BASELINE: dict = {}   # (prompt, seed) -> first-seen r_unc, within this process
R_UNC_DRIFT_RTOL = 1e-3
R_UNC_DRIFT_ATOL = 1e-4


# ═════════════════════════════════════════════════════════════════════════════
# Sweep configuration — edit this block
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RunConfig:
    prompt:       str   = "a cat wearing sunglasses on a beach"     # image-generation prompt
    edit_prompt:  Optional[str] = None  # reward-model target prompt; falls back to `prompt` if None
    N:            int   = 4       # total denoising steps
    H:            int   = 2       # ConicFlow planning horizon (ignored for fmrg/boffi_fmrg)
    control_type: str   = "conicflow"   # "conicflow" | "fmrg" | "boffi_fmrg" | "hardflow" | "none"
    lam:          float = 10.0    # control / guidance strength (== upstream step_size for boffi_fmrg)
    n_inner:      int   = 5       # inner iters (ConicFlow), n_opt (fmrg), or num_optim_iters (boffi_fmrg)
    lr_inner:     float = 1e-2    # inner Adam lr (ConicFlow only)
    sample_mode:  str   = "flow_map2"
    guidance_scale: float = 3.5
    height:       int   = 256
    width:        int   = 256
    warmstart:    bool  = True    # ConicFlow only
    n_alt:        int   = 1       # ConicFlow only. Primal<->dual alternations per fixed
                                      # window before it shifts (the design plan).
                                      # 1 = original behavior (single solve, single
                                      # dual update). See
                                      # control_loops.receding_horizon_flow_control's
                                      # n_alt docstring -- requires constraints
                                      # active (h + lr_dual>0) when > 1.
    skip_first_primal_step: bool = False  # ConicFlow only. design plan option (b):
                                      # one dual-ascent update on the
                                      # UNCONTROLLED rollout's h, before the
                                      # loop starts, instead of wasting j=0's
                                      # primal solve on a zero-gradient
                                      # objective (mu_init=0 + dynamics-exact
                                      # warm start => energy=0 => nothing to
                                      # optimize at step 0). Composable with
                                      # a nonzero mu_init (option (a)) -- see
                                      # control_loops.py's docstring.
                                      # Requires constraints active
                                      # (h + lr_dual>0) when True.
    # ConicFlow only. Default flipped False -> True and the lock LIFTED on 2026-09-16
    # (see _RETIRED_LOCKS for the full rationale and numbers). True evaluates
    # the drift for ell>0 under no_grad, dropping the J_v^T term from the
    # gradient. That is an approximation of the gradient, but the measured
    # result was both faster (-33%) and BETTER on the objective at every lambda
    # -- the exact drift term is apparently not what the inner solve needs.
    #
    # Two consequences worth knowing:
    #   - It does NOT remove the transformer backward. Since the 2026-08-26
    #     endpoint-gradient fix, endpoint_map(..., grad=True) is the sole
    #     gradient path and runs regardless, which is why tune_sampler_speed
    #     now also needs constraint_on to decide whether checkpointing can be
    #     dropped (it was corrected in the same change).
    #   - Runs before 2026-09-16 used False. Any energy/objective comparison
    #     across that date must check the drift_detach column, which every row
    #     carries.
    drift_detach: bool  = True
    # How endpoint_map predicts the terminal latent inside the inner objective.
    # "rollout" (default) integrates the remaining N-j-H steps, which is what
    # makes ConicFlow quadratic in N. "jump" takes ONE flow-map call with t_next=0 --
    # the surrogate only; the applied trajectory is unchanged. Measured
    # 2026-09-16 at N=8/H=4: -33% wall clock, J better at lam=1 (2.2 sd) and
    # within noise at lam=10/100. Default stays "rollout" because it has not
    # been re-checked at the suite's N=10 and because switching it reproduces
    # none of the results currently on disk. See image_conicflow.make_flux_wrappers.
    endpoint_surrogate: str = "rollout"
    constraint_on: str  = "endpoint"  # ConicFlow only; "horizon" evaluates reward and
                                      # constraint at raw x_H, "endpoint" at the
                                      # flow-map-predicted terminal latent.
                                      # Default is "endpoint" per the locked
                                      # decision (an earlier copy defaulted to
                                      # "horizon"; deliberately not ported).
    # Inequality constraints: h_i(z) = threshold_i - clip(z, prompt_i) <= 0.
    # Each prompt pairs with the same-index threshold. Empty -> no constraint.
    # ConicFlow-only by design: requesting constraints with control_type != "conicflow"
    # RAISES (see _run_one) rather than silently running unconstrained.
    # Step 4 of the plan replaces this pair with a general per-constraint spec
    # (arbitrary reward type + direction + normalizer); until then every
    # constraint is a CLIP text constraint.
    constraint_prompts:    List[str]   = field(default_factory=list)
    constraint_thresholds: List[float] = field(default_factory=list)
    # Step 4.1's generalized constraint spec list -- arbitrary reward types,
    # thresholds, and directions, each with its own optional normalizer.
    # Independent of constraint_prompts/constraint_thresholds above: if BOTH
    # are set, _resolve_constraint_specs raises rather than silently picking
    # one (see that function) -- specifying a run's constraints two different
    # ways at once is a configuration error, not a case to guess at.
    constraint_specs: List["ConstraintSpec"] = field(default_factory=list)
    lr_dual:  float = 0.0   # dual-ascent step size (0 = frozen multipliers)
    mu_init:  float = 0.0   # initial value for all Lagrange multipliers
    # ConicFlow only. Normalize the dual step (and the Lagrangian's constraint term)
    # per constraint by a running RMS scale, so constraints whose typical
    # violation is small are not starved of dual pressure under a shared
    # lr_dual. See receding_horizon_flow_control's dual_adaptive docstring.
    dual_adaptive: bool = False
    # boffi_fmrg only -- passed through to FluxFlowMapSampler.sample_reward_guided
    # (upstream's vendored FMRG-J implementation). Left at upstream's "off" defaults
    # so the paired comparison is a minimal single-particle run, not upstream's full
    # warmup/early-stop pipeline. Reward-scoring for boffi_fmrg currently supports
    # --reward clip only (see _clip_score_pixel).
    # boffi_use_adam=False + boffi_grad_norm_mode="normalize" is upstream's *other*
    # branch, which implements FMRG paper App. E.2 literally: u_tilde = (u/||u||)*||v||,
    # lambda_t = eta*dt (undecayed, "step_size" == the paper's eta). Default True matches
    # upstream's own recommended config (generate_aesthetic.py, use_adam=True).
    boffi_use_adam:         bool = True
    boffi_grad_norm_mode:   str = "none"          # "none" | "normalize"
    boffi_warmup_steps:     int = 0                # 0 = no best-of-N particle reinit
    boffi_warmup_particles: int = 1
    boffi_early_stop:       int = 0                # 0 = no uncontrolled tail
    boffi_unguided_steps:   int = 0
    boffi_noise_mode:       str = "deterministic"
    # hardflow only -- passed through to control_loops.hardflow_image.
    # Defaults reproduce the authors' released implementation (utils/methods.py
    # in azizanlab/HardFlow@016adb2), per the 2026-08-31 decision to "keep their
    # implementation the same and just change the model". Two exceptions, both
    # deliberate and both recorded in hardflow_image's docstring:
    #   - hardflow_activation defaults to 0.0 (control from step 0) rather than
    #     their hard-coded 0.5, for parity with our ConicFlow setting. It is
    #     exposed as a hyperparameter rather than hard-coded; set 0.5 to
    #     reproduce their schedule.
    #   - the regularizer SIGN is the paper's, not their code's (theirs rewards
    #     deviation from the nominal endpoint -- see the docstring).
    #   - hardflow_reg defaults to 5e-4, NOT their effectively-inert 0.0 --
    #     see the conversion note directly below.
    #
    # CONVERSION BETWEEN `lam` AND `hardflow_reg` (corrected 2026-09-09; the
    # previous note here said "lam <-> 1/hardflow_reg", which is WRONG by a
    # factor of 2*dt and hides an N-dependence). Derivation, both in terms of
    # the same u_t that summary.csv's ctrl_energy is built from:
    #
    #   ConicFlow        resid_l = dt*u_l, so the energy term is
    #              sum_l ||dt*u_l||^2 / (2*lam*dt) = ctrl_energy_window/(2*lam)
    #   HardFlow   the applied displacement is D = alpha*(x_hat - z_pred_ref)
    #              = z - z_next_ref, and the stored control is u_k = D/dt, so
    #              alpha^2 * dev = ||D||^2 = dt^2 * ||u_k||^2, and the penalty
    #              is hardflow_reg * dt^2 * ||u_k||^2
    #
    # Matching per step:  reg*dt^2 = dt/(2*lam)  =>
    #
    #       hardflow_reg = 1 / (2 * lam * dt) = N / (2 * lam)
    #
    # At the standard operating point (lam=10000, N=10) that is 5e-4 -- which
    # is also, not coincidentally, the authors' nominal `regularization`.
    # IT DEPENDS ON N: change N and this default is no longer matched to lam.
    # run_preflight prints the implied equivalent lam on every hardflow run so
    # a mismatch is visible in the log rather than silent.
    #
    # The correspondence is exact PER STEP only. ConicFlow's penalty is summed over an
    # H-step planning window that is re-solved each outer step and mostly
    # discarded, so no single scalar makes the two objectives identical -- this
    # matches the marginal cost of control effort, not the total.
    #
    # NOTE hardflow_reg MULTIPLIES (bigger = stronger regularization), the
    # opposite of `lam` above, which divides.
    # n_inner and lr_inner above are REUSED for hardflow rather than shadowed:
    # they are the same axis (inner iterations per outer step, and the inner
    # step size), and the compute-budget comparison needs one
    # column to mean one thing across control types. The authors' values are
    # n_inner=40, lr_inner=2.5 -- set them explicitly on hardflow configs; they
    # are NOT the RunConfig defaults, which are ConicFlow's.
    hardflow_rho:        float = 5.0    # AL quadratic penalty weight
    hardflow_rho_mult:   float = 2.0    # rho growth factor on violation
    hardflow_dual_every: int   = 8      # inner steps between dual updates
    # 0.0 as of 2026-09-10 (was 0.02, the authors' hard-coded value). Decision of
    # 2026-09-10: the margin tightens every constraint by 0.02 while the run
    # is scored against the UNtightened threshold, and it is undocumented in
    # their paper. Our reading is that it is specific to the setting they
    # constrained in; ours does not want it. It also removed a real asymmetry --
    # the ConicFlow arm gets no margin at all, so every HardFlow-vs-ConicFlow row before
    # today gave HardFlow a 0.02 head start on feasibility (generous to the
    # baseline, not stingy, so earlier "HardFlow still loses" conclusions stand).
    #
    # RUNS BEFORE 2026-09-10 USED 0.02 and are not margin-comparable with runs
    # after it: expect slightly LOWER feasibility and slightly LOWER energy at
    # 0.0, since HardFlow now aims at the threshold instead of past it. Pass
    # --sweep_hardflow_margin 0.02 to reproduce them. The `hardflow_margin` CSV
    # column records the value on every row, so no row is ambiguous.
    hardflow_margin:     float = 0.0    # was 0.02 = their constraint tightening
    hardflow_activation: float = 0.0    # fraction of steps to skip (theirs: 0.5)
    # None = DERIVE as reg = N/(2*lam) (derivation above), the value that makes
    # HardFlow's control-effort penalty match ConicFlow's per step.
    #
    # Was a fixed 5e-4 until 2026-09-16. WHY THAT CHANGED: 5e-4 is N/(2*lam) at
    # lam=10000 and N=10 AND NOWHERE ELSE -- reg is N-dependent, so every other
    # (N, lam) point silently mismatched the ConicFlow arm it was plotted against (an
    # N=20 hardflow arm ran half as regularized as its ConicFlow counterpart). A
    # constant cannot track two axes. Deriving also makes `lam` mean the same
    # thing on both paths; it used to be read-but-inert on this one.
    #
    # An explicit --sweep_hardflow_reg is passed through UNTOUCHED and never
    # recomputed -- that is how reg=0.0 (the pre-2026-09-09 no-effort-penalty
    # runs) and any deliberate lam/reg mismatch stay reproducible.
    #
    # Resolved to a float by _resolve_hardflow_reg before any run executes;
    # run_preflight raises if an unresolved None reaches a hardflow run.
    hardflow_reg:        Optional[float] = None  # deviation penalty; None = derive
    # "flow_map1" | "flow_matching" | "conicflow_rollout".
    # The first two are single-NFE terminal estimates, which is HardFlow's
    # efficiency claim. "conicflow_rollout" is the ESTIMATOR ABLATION (2026-09-08):
    # it hands hardflow_image ConicFlow's endpoint_map instead, so the head-to-head
    # stops confounding "which control algorithm" with "which endpoint
    # estimate" -- the two disagree by ~2 ImageReward units at step 0
    # (the two-constraint report SS3). Affordable because HardFlow
    # evaluates the endpoint once per OUTER step, not per inner step; it takes
    # NFE from 2N to N + N(N+1)/2, so ~115 s/run rather than ~90 s.
    hardflow_endpoint_mode: str = "flow_map1"
    # (2026-09-17) Lift HardFlow's admissible floor. hardflow_image raises when
    # lr*reg >= 1 because its explicit fixed-step inner ascent stops contracting
    # (error evolves as e <- e*(1 - 2*lr*reg*alpha_t^2)). With reg = N/(2*lam)
    # that is a floor lam >= N*lr/2 -- 12.5 at N=10 with the authors' lr=2.5 --
    # which for ImageReward sits ABOVE HardFlow's entire transition, so every
    # admissible lam is a plateau point and the method can never be compared to
    # ConicFlow at a shared lam in the regularised regime (results/autorange/).
    # When set, any run with lr_inner*reg >= 1 uses lr = hardflow_lr_alpha/reg
    # instead, i.e. the product lr*reg is pinned at hardflow_lr_alpha < 1. That
    # is not a hack around the guard: the inner loop's contraction factor
    # becomes 1 - 2*hardflow_lr_alpha*alpha_t^2, uniform in lam, so convergence
    # no longer depends on where the run sits on the grid (0.5 is dead-beat at
    # the final step). None = off: the guard raises exactly as before. The
    # EFFECTIVE lr is recorded per row in `hardflow_lr_eff`.
    hardflow_lr_alpha: Optional[float] = None
    # style reward only (reward_model=="style") -- one or more reference image
    # paths; Gram matrices are averaged across references when >1 (style
    # mixing). Neither the FMRG paper nor MPGD specify multi-reference
    # combination -- averaging is our choice, not theirs.
    style_ref_paths: Optional[List[str]] = None
    # lpips reward only (reward_model=="lpips"). Backbone is OUR choice, not
    # HardFlow's -- their paper delegates the LPIPS backbone to their ref [26]
    # and never states which network they used. 'alex' chosen over 'vgg'
    # (best perceptual correlation in the original LPIPS paper, but by far
    # the most expensive: no early aggressive downsampling keeps activations
    # large through many layers) given only ~4.8GB headroom measured at the
    # ConicFlow H=4/drift_detach=False worst case (see style reward's S4 memory
    # measurement, same mechanism applies here). 'squeeze' is a fallback if
    # 'alex' doesn't fit; not yet measured on GPU.
    lpips_net: str = "alex"   # "alex" | "vgg" | "squeeze"
    # populated at runtime:
    seed:         int   = 42
    out_dir:      str   = "results"


SWEEP_CONFIGS: List[RunConfig] = [
    # ── prompt sweep ────────────────────────────────────────────────────────
    RunConfig(prompt="a sunset over the ocean",         N=4, H=2, control_type="conicflow",  lam=50.0),
    RunConfig(prompt="a sunset over the ocean",         N=4, H=2, control_type="boffi_fmrg", lam=5.0),
    RunConfig(prompt="a sunset over the ocean",         N=4, H=2, control_type="none"),
    # ── horizon sweep (conicflow) ─────────────────────────────────────────────────
    RunConfig(prompt="a cat wearing sunglasses",        N=4, H=1, control_type="conicflow",  lam=50.0),
    RunConfig(prompt="a cat wearing sunglasses",        N=4, H=2, control_type="conicflow",  lam=50.0),
    RunConfig(prompt="a cat wearing sunglasses",        N=4, H=3, control_type="conicflow",  lam=50.0),
    # ── N sweep ─────────────────────────────────────────────────────────────
    RunConfig(prompt="a cat wearing sunglasses",        N=3, H=2, control_type="conicflow",  lam=50.0),
    RunConfig(prompt="a cat wearing sunglasses",        N=6, H=2, control_type="conicflow",  lam=50.0),
    # ── lambda sweep ────────────────────────────────────────────────────────
    RunConfig(prompt="a cat wearing sunglasses",        N=4, H=2, control_type="conicflow",  lam=5.0),
    RunConfig(prompt="a cat wearing sunglasses",        N=4, H=2, control_type="conicflow",  lam=200.0),
    # ── sample mode comparison ───────────────────────────────────────────────
    RunConfig(prompt="a cat wearing sunglasses",        N=4, H=2, control_type="conicflow",  lam=50.0, sample_mode="flow_map1"),
    # ── edit_prompt example: generate with one prompt, reward against another ──
    RunConfig(prompt="a portrait of a man",             edit_prompt="a smiling face",
              N=4, H=2, control_type="conicflow",  lam=50.0),
    RunConfig(prompt="a portrait of a man",             edit_prompt="a smiling face",
              N=4, H=2, control_type="boffi_fmrg", lam=5.0),
]


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _free(*tensors):
    for t in tensors:
        del t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_uncontrolled(sampler, z0, cfg: RunConfig, device, dtype, log: Callable = print):
    """Roll the uncontrolled FLUX flow and return (z_final, traj).

    Calls run_preflight itself (not just relying on a caller having already
    done so) because this function is directly, independently callable --
    _run_one calls it, but so, plausibly, will the Step 3 calibration script,
    which sample-only-varies-seed from the uncontrolled model and has no
    reason to route through _run_one's full ConicFlow/reward-closure machinery at
    all. check_boffi_rng is never set: this function has nothing to do with
    boffi_fmrg regardless of cfg.control_type (it's the plain uncontrolled
    rollout for whatever N/sample_mode is configured), so that check is
    simply not applicable here -- stated explicitly by omission of the
    parameter, not a silent gap.
    """
    run_preflight(cfg, log=log)
    dt_flux = 1.0 / cfg.N
    ts = torch.linspace(1.0, 0.0, cfg.N + 1, device=device, dtype=dtype)
    z = z0.clone()
    traj = [z.clone()]
    with torch.no_grad():
        for j in range(cfg.N):
            t_c = ts[j].item()
            if cfg.sample_mode == "flow_map2":
                t_n = max(0.0, t_c - dt_flux)
            elif cfg.sample_mode == "flow_map1":
                t_n = 0.0
            else:
                t_n = t_c
            dt = (ts[j] - ts[j + 1]).item()
            u = sampler.predict_vector(
                z, t_c,
                prompt_embeds=sampler.cached_prompt_embeds,
                pooled_prompt_embeds=sampler.cached_pooled_prompt_embeds,
                guidance_scale=cfg.guidance_scale,
                t_next=t_n,
            )
            z = z - dt * u
            traj.append(z.clone())
    return z, torch.stack(traj, dim=0)


def _ctrl_norms_from_traj(
    traj: torch.Tensor,
    v_theta_conicflow,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """Compute per-step control inputs from the trajectory using the dynamics.

    Uses the definition   u_t = x_{t+1} - x_t - dt * b_t(x_t)
    (i.e. the residual between the actual step and the uncontrolled drift).

    This is independent of how the control was parameterized internally (ConicFlow
    stores u/dt, FMRG stores the guidance displacement), so norms are
    comparable across methods.

    Requires N transformer fwd passes (no_grad).

    Returns:
        ctrl_norms:  (N,) numpy array — mean ‖u_t‖ over batch per step
        ctrl_energy: scalar — Σ_t ‖u_t‖² · dt
    """
    dt = 1.0 / N
    t_grid = torch.linspace(0.0, 1.0, N + 1)
    norms = []
    with torch.no_grad():
        for t_idx in range(N):
            x_t   = traj[t_idx    ].to(device=device, dtype=dtype)
            x_next = traj[t_idx + 1].to(device=device, dtype=dtype)
            b_t   = v_theta_conicflow(t_grid[t_idx].to(device), x_t)
            u_t   = x_next - x_t - dt * b_t          # displacement residual
            norms.append(u_t.flatten(1).norm(dim=1).mean().item())
    import numpy as np
    norms = np.array(norms, dtype=np.float32)
    ctrl_energy = float((norms ** 2 * dt).sum())
    return norms, ctrl_energy


def _ctrl_norms_from_traj_nonuniform(
    traj: torch.Tensor,
    v_theta_conicflow,
    dts: list,
    t_curs: list,
    device: torch.device,
    dtype: torch.dtype,
    dump_path: Optional[str] = None,
) -> tuple:
    """Same definition as _ctrl_norms_from_traj (u_t = x_{t+1} - x_t - dt*b_t(x_t),
    ctrl_energy = sum ||u_t||^2 * dt), generalized to a non-uniform time grid.

    boffi_fmrg's trajectory lives on FLUX's mu-shifted schedule, not the
    uniform grid _ctrl_norms_from_traj assumes -- reusing that function
    unchanged here would evaluate the uncontrolled drift b_t at the wrong
    time for every step past the first, silently miscomputing ctrl_energy.
    dts/t_curs are the ACTUAL per-step values realized by that trajectory.

    t_curs are in FLUX time convention (1=noise, 0=data); converted to ConicFlow's
    s = 1 - t_flux to match what v_theta_conicflow (from make_flux_wrappers)
    expects, same convention used throughout image_conicflow.py.

    Instrumentation (added 2026-08-21, Phase A of the boffi_fmrg
    ctrl_energy=nan investigation): raises FloatingPointError at the FIRST
    non-finite intermediate value found, with a full dump of that step's
    inputs/outputs -- not silent nan propagation to the final scalar. Also,
    if dump_path is given, writes every step's x_t/x_next/b_t/u_t plus the
    outcome to a .npz, on both success and failure, so a future occurrence
    (or this one) can be re-examined offline without a live repro.
    """
    import numpy as np

    def _stats(name, t):
        t = t.detach()
        finite_mask = torch.isfinite(t)
        n_nonfinite = int((~finite_mask).sum().item())
        finite_vals = t[finite_mask]
        lo = finite_vals.min().item() if finite_vals.numel() else float("nan")
        hi = finite_vals.max().item() if finite_vals.numel() else float("nan")
        mu = finite_vals.mean().item() if finite_vals.numel() else float("nan")
        return (f"{name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
                f"nonfinite_count={n_nonfinite}/{t.numel()} min={lo:.6g} max={hi:.6g} mean={mu:.6g}")

    traj_finite_on_arrival = bool(torch.isfinite(traj).all().item())
    dump_records = []

    norms = []
    with torch.no_grad():
        for i, (dt_i, t_cur_i) in enumerate(zip(dts, t_curs)):
            x_t = traj[i].to(device=device, dtype=dtype)
            x_next = traj[i + 1].to(device=device, dtype=dtype)
            x_t_finite = bool(torch.isfinite(x_t).all().item())
            x_next_finite = bool(torch.isfinite(x_next).all().item())

            s_conicflow = torch.tensor(1.0 - t_cur_i, device=device)
            b_t = v_theta_conicflow(s_conicflow, x_t)
            b_t_finite = bool(torch.isfinite(b_t).all().item())

            u_t = x_next - x_t - dt_i * b_t
            u_t_finite = bool(torch.isfinite(u_t).all().item())
            step_norm = u_t.flatten(1).norm(dim=1).mean().item()

            dump_records.append(dict(
                step=i, dt=float(dt_i), t_cur=float(t_cur_i),
                x_t=x_t.float().cpu().numpy(), x_next=x_next.float().cpu().numpy(),
                b_t=b_t.detach().float().cpu().numpy(), u_t=u_t.detach().float().cpu().numpy(),
                step_norm=step_norm,
            ))

            if not (x_t_finite and x_next_finite and b_t_finite and u_t_finite):
                arrived_bad = not (x_t_finite and x_next_finite)
                msg = "\n".join([
                    f"_ctrl_norms_from_traj_nonuniform: non-finite value at step {i}/{len(dts)}",
                    f"  whole traj tensor non-finite on arrival (checked before loop start): {not traj_finite_on_arrival}",
                    f"  ARRIVED BAD at this step (x_t/x_next already non-finite entering it): {arrived_bad}",
                    f"  CREATED BAD inside this step (b_t or u_t non-finite despite finite x_t/x_next): {not arrived_bad}",
                    f"  dt_i={dt_i}  t_cur_i={t_cur_i}  s_conicflow={s_conicflow.item()}",
                    "  " + _stats("x_t", x_t),
                    "  " + _stats("x_next", x_next),
                    "  " + _stats("b_t (v_theta_conicflow output)", b_t),
                    "  " + _stats("u_t (= x_next - x_t - dt*b_t)", u_t),
                ])
                if dump_path:
                    np.savez(dump_path, records=np.array(dump_records, dtype=object),
                              failed_at_step=i, traj_finite_on_arrival=traj_finite_on_arrival)
                raise FloatingPointError(msg)

            norms.append(step_norm)
    norms = np.array(norms, dtype=np.float32)
    ctrl_energy = float(sum(n ** 2 * dt_i for n, dt_i in zip(norms, dts)))
    if dump_path:
        np.savez(dump_path, records=np.array(dump_records, dtype=object),
                  failed_at_step=None, traj_finite_on_arrival=traj_finite_on_arrival,
                  norms=norms, ctrl_energy=ctrl_energy)
    return norms, ctrl_energy


def _sample_boffi_fmrg_with_trajectory(sampler, boffi_reward, z0, cfg, device):
    """Call sampler.sample_reward_guided unchanged, but also reconstruct the
    latent trajectory it produces internally (not exposed by its own return
    value) via a temporary instance-level monkey-patch on
    reward_consistency_xt.

    Non-invasive: wraps the ORIGINAL bound method (called with the exact same
    arguments, its return value passed through completely unmodified), and
    restores the original afterward. This is a runtime attribute shadow on
    one Python object in one process -- nothing on disk (fluxfm_sampler_reward.py,
    image_conicflow.py) is touched by it.

    Reconstruction: for grad_mode="jac", sample_mode in {flow_map1, flow_map2},
    reward_consistency_xt is called exactly once per outer step (no warmup)
    with zt=<trajectory state entering this step> and velocity=<u_step, which
    for flow_map1 IS exactly the velocity sample_reward_guided uses to build
    guided_velocity>, returning grad_zt such that (per fluxfm_sampler_reward.py's
    own code, ~line 1358-1366) the next trajectory state is exactly
    `zt_original - dt*(velocity.detach() + grad_zt)`. dt for each step is
    recovered from consecutive captured t_cur values; the final step's t_next
    is always 0.0 when early_stop=0 (guaranteed by sample_reward_guided's own
    time_steps construction, which always ends in torch.zeros(1) unless the
    early_stop trimming branch fires).
    """
    if cfg.boffi_early_stop != 0 or cfg.boffi_warmup_steps != 0:
        raise NotImplementedError(
            "trajectory reconstruction only supports boffi_early_stop=0 and "
            "boffi_warmup_steps=0 -- it assumes exactly one reward_consistency_xt "
            "call per outer step and a trailing t_next==0.0."
        )
    # Independent of the check above (which exists for a different reason --
    # see its docstring): assert every RNG draw inside sample_reward_guided
    # stays structurally unreachable. Routed through run_preflight (not a
    # direct call to _assert_boffi_rng_reachability_closed) so this file has
    # exactly one function that calls the three guards.
    run_preflight(cfg, check_boffi_rng=True, latent=z0)

    captured = []
    orig_fn = sampler.reward_consistency_xt

    def _hooked(*args, **kwargs):
        t_cur = kwargs.get("t_cur")
        velocity = kwargs.get("velocity")
        grad_zt, loss_val, zt_original = orig_fn(*args, **kwargs)
        captured.append((
            t_cur, velocity.detach().clone(),
            grad_zt.detach().clone(), zt_original.detach().clone(),
        ))
        return grad_zt, loss_val, zt_original

    sampler.reward_consistency_xt = _hooked
    try:
        img = sampler.sample_reward_guided(
            reward_model=boffi_reward,
            num_steps=cfg.N,
            guidance_scale=cfg.guidance_scale,
            step_size=cfg.lam,
            img_shape=(cfg.height, cfg.width),
            latent=z0.clone(),
            num_optim_iters=cfg.n_inner,
            use_adam=cfg.boffi_use_adam,
            grad_mode="jac",
            sample_mode=cfg.sample_mode if cfg.sample_mode in
                ("flow_map1", "flow_map2", "flow_matching") else "flow_map1",
            early_stop=cfg.boffi_early_stop,
            warmup_steps=cfg.boffi_warmup_steps,
            warmup_particles=cfg.boffi_warmup_particles,
            unguided_steps=cfg.boffi_unguided_steps,
            noise_mode=cfg.boffi_noise_mode,
            grad_norm_mode=cfg.boffi_grad_norm_mode,
            enable_callback=False,
            grad_checkpointing=True,
        )
    finally:
        sampler.reward_consistency_xt = orig_fn

    n_calls = len(captured)
    traj = [captured[0][3]]   # zt_original of the first call == the initial latent
    dts, t_curs = [], []
    for i in range(n_calls):
        t_cur_i, velocity_i, grad_zt_i, zt_original_i = captured[i]
        t_next_i = captured[i + 1][0] if i + 1 < n_calls else 0.0
        dt_i = t_cur_i - t_next_i
        guided_velocity = velocity_i + grad_zt_i
        z_next = zt_original_i - dt_i * guided_velocity
        traj.append(z_next)
        dts.append(dt_i)
        t_curs.append(t_cur_i)

    traj = torch.stack(traj, dim=0)
    return img, traj, dts, t_curs


def _decode_pil(sampler, z):
    img = sampler.decode_no_grad(z).float().detach().cpu()
    return sampler.pipeline.image_processor.postprocess(img, output_type="pil")[0]


def _resolve_devices(flux_device: str, reward_device: str) -> tuple[torch.device, torch.device]:
    """Resolve FLUX and reward-model devices, with sensible multi-GPU defaults."""
    flux_dev = torch.device(flux_device)
    if reward_device != "auto":
        return flux_dev, torch.device(reward_device)

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return flux_dev, torch.device("cpu")

    flux_idx = flux_dev.index if flux_dev.type == "cuda" else 0
    for idx in range(torch.cuda.device_count()):
        if idx != flux_idx:
            return flux_dev, torch.device(f"cuda:{idx}")
    return flux_dev, torch.device("cpu")


class _DroppedTower(torch.nn.Module):
    """Stand-in for a submodule freed by ``_drop_text_tower``.

    Raises on any use rather than returning something plausible -- the standing
    "loud fail only" rule applied to a freed module. A bare ``None`` would give
    ``AttributeError: 'NoneType' object has no attribute ...``, which reads like
    a load bug rather than "you asked for text features after we dropped the
    text tower to save 1.4 GB".
    """

    def __init__(self, owner: str, name: str):
        super().__init__()
        self._owner, self._name = owner, name

    def forward(self, *a, **k):
        raise RuntimeError(
            f"{self._owner}.{self._name} was freed by _drop_text_tower(): this "
            f"reward's text embedding is precomputed once at build time, so the "
            f"text tower is dead weight during the run. Something is asking for "
            f"text features after the fact. If that call is legitimate (e.g. a "
            f"new boffi_fmrg-style scoring path), pass drop_text_tower=False "
            f"when building this reward."
        )


def _drop_text_tower(reward_obj, label: str, log=print) -> float:
    """Free a CLIP-family model's text tower once its text embedding is cached.

    Added 2026-09-05. The clip/pickscore closures below precompute a normalized
    ``txt_feat`` at BUILD time and thereafter call only ``get_image_features``,
    so ``text_model`` + ``text_projection`` sit resident for the whole run and
    are never read. For CLIP ViT-H-14 that is a MEASURED 1.42 GB of the model's
    4.01 GB (986.1 M params total, 632.1 M of them the vision tower).

    Exact, not an approximation: nothing about the vision tower or the cached
    text embedding changes -- weights are simply released.

    NOT safe when ``_clip_score_pixel`` may still run, which is only the
    ``boffi_fmrg`` path; callers gate on that.
    """
    model = getattr(reward_obj, "model", None)
    if model is None:
        return 0.0
    freed = 0
    before = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    for name in ("text_model", "text_projection"):
        sub = getattr(model, name, None)
        if sub is None or isinstance(sub, _DroppedTower):
            continue
        freed += sum(p.numel() for p in sub.parameters())
        setattr(model, name, _DroppedTower(label, name))
    if not freed:
        return 0.0
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    saved = before - (torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0)
    log(f"  [mem] dropped {label} text tower: {freed/1e6:.1f}M params, {saved:.2f} GB freed")
    return saved


def _build_reward_fn(sampler, cfg, reward_device, dtype, reward_model: str = "clip",
                      ref_holder: Optional[dict] = None,
                      decode_fn: Optional[Callable] = None,
                      drop_text_tower: bool = False,
                      reward_dtype: torch.dtype = torch.float32):
    """Return ``(r_image, reward_obj)`` for the given config and reward model.

    All reward models run on ``reward_device``.  VAE decode stays on the FLUX
    device; only the decoded image is moved for scoring.

    ``decode_fn``: optional replacement for ``sampler.decode`` (Step 4.3,
    "decode sharing"). Every branch below calls ``(decode_fn or
    sampler.decode)(z)`` rather than ``sampler.decode(z)`` directly, so a
    caller building several constraints against the SAME z (the generalized
    constraint builder, see ``_build_constraint_fn``) can pass a small
    single-slot memoizing wrapper and avoid one redundant VAE forward pass
    per extra constraint sharing a reward type at that z. Defaults to None
    (plain ``sampler.decode``), so every existing call site (the reward
    path, which never shares a z with anything) is unaffected.

    Supported ``reward_model`` values:
        "clip"        — CLIP ViT-H-14 cosine similarity (default, fast)
        "hpsv2"       — HPSv2 aesthetic/human-preference score
        "pickscore"   — PickScore human-preference model
        "imagereward" — ImageReward (requires ``ImageReward`` package)
        "lpips"       — perceptual distance to the uncontrolled output (upper-
                        bound-style reward, not a target to maximize)
        "none"        — zero reward everywhere; use when steering is done
                        entirely via the constraint function h. ``reward_obj``
                        is None in this branch, which _run_one's device
                        preflight and plausibility guard both special-case.

    ``ref_holder``: optional mutable dict, ignored by every branch except
    "lpips". Unlike "style", lpips's reference image (x_ref = decode of the
    uncontrolled rollout, z_unc) does not exist yet at the point _run_one
    calls this function -- z_unc is only produced later by the uncontrolled
    rollout. The caller passes an empty dict here and populates
    ref_holder["x_ref"] once z_unc is decoded, before ever calling r_image;
    the lpips closure reads the holder lazily at call time rather than
    capturing a value at build time. See _run_one for the exact insertion
    point and conversation history for why (no image-to-latent inversion
    path exists in this codebase, so x_ref can't be a real input image the
    way HardFlow uses one -- x_ref = "the unguided output from the same
    seed" is our own design choice, not theirs).
    """
    reward_model = reward_model.lower()
    preprocess = create_clip_preprocess(224)
    # Reward is scored against edit_prompt when set, else falls back to the
    # generation prompt (e.g. cfg.prompt="a portrait of a man",
    # cfg.edit_prompt="a smiling face" edits toward the CLIP-smiling direction
    # while still generating a man's portrait as the starting image).
    prompt = cfg.edit_prompt if cfg.edit_prompt else cfg.prompt

    # ── CLIP ────────────────────────────────────────────────────────────────
    if reward_model == "clip":
        tokenizer = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=HF_CACHE
        )
        reward_obj = CLIPReward(
            device=reward_device, dtype=reward_dtype,
            tokenizer=tokenizer, cache_dir=HF_CACHE,
        )
        with torch.no_grad():
            toks = tokenizer(
                prompt, return_tensors="pt", padding=True,
                max_length=77, truncation=True,
            ).to(reward_device)
            txt_feat = reward_obj.model.get_text_features(**toks)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        # txt_feat is now cached; the closure below only reads image features.
        if drop_text_tower:
            _drop_text_tower(reward_obj, "clip")

        def r_image(z: torch.Tensor) -> torch.Tensor:
            img = (decode_fn or sampler.decode)(z)
            img_01 = ((img + 1.0) / 2.0).to(device=reward_device, dtype=reward_dtype)
            img_feat = reward_obj.model.get_image_features(preprocess(img_01))
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            return (img_feat @ txt_feat.T).squeeze(-1).to(z.device)

    # ── HPSv2 ────────────────────────────────────────────────────────────────
    elif reward_model == "hpsv2":
        reward_obj = HPSReward(
            device=reward_device, dtype=reward_dtype, cache_dir=HF_CACHE,
        )

        def r_image(z: torch.Tensor) -> torch.Tensor:
            img = (decode_fn or sampler.decode)(z)
            img_01 = ((img + 1.0) / 2.0).to(device=reward_device, dtype=reward_dtype)
            # HPSReward.forward returns a scalar loss = 1 - sim; negate for reward
            loss = reward_obj(preprocess(img_01), prompt)
            return (1.0 - loss).unsqueeze(0).to(z.device)

    # ── PickScore ────────────────────────────────────────────────────────────
    elif reward_model == "pickscore":
        tokenizer = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=HF_CACHE
        )
        reward_obj = PickScoreReward(
            device=reward_device, dtype=reward_dtype,
            tokenizer=tokenizer, cache_dir=HF_CACHE,
        )
        with torch.no_grad():
            toks = tokenizer(
                prompt, return_tensors="pt", padding=True,
                max_length=77, truncation=True,
            ).to(reward_device)
            txt_feat = reward_obj.model.get_text_features(**toks)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        # logit_scale lives on the top-level model, not the text tower, so the
        # closure below still works after this.
        if drop_text_tower:
            _drop_text_tower(reward_obj, "pickscore")

        def r_image(z: torch.Tensor) -> torch.Tensor:
            img = (decode_fn or sampler.decode)(z)
            img_01 = ((img + 1.0) / 2.0).to(device=reward_device, dtype=reward_dtype)
            img_feat = reward_obj.model.get_image_features(preprocess(img_01))
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sim = (reward_obj.model.logit_scale.exp() * (img_feat @ txt_feat.T))
            return sim.squeeze(-1).to(z.device)

    # ── ImageReward ───────────────────────────────────────────────────────────
    elif reward_model == "imagereward":
        # ImageReward uses transformers 4.x APIs; apply compat shim first.
        try:
            import imagereward_compat  # noqa: F401
        except ImportError:
            pass  # only needed with transformers 5.x; silently skip if absent
        reward_obj = ImageRewardReward(
            device=reward_device, dtype=reward_dtype, cache_dir=HF_CACHE,
        )

        def r_image(z: torch.Tensor) -> torch.Tensor:
            img = (decode_fn or sampler.decode)(z)
            img_01 = ((img + 1.0) / 2.0).to(device=reward_device, dtype=reward_dtype)
            # ImageRewardReward.forward returns loss = 2 - z, where z is the
            # checkpoint's OWN z-scored score ((raw - model.mean)/model.std).
            # So 2.0 - loss == z exactly -- this closure returns the z-score
            # itself, NOT "score - 2" as this comment previously claimed
            # (corrected 2026-08-26). Consequence worth knowing: imagereward's
            # reward therefore has population std 1.0 BY CONSTRUCTION, which is
            # why row 1's n=100 calibration std came back 1.0041 -- that
            # agreement is a design property, not a lucky draw, and it is the
            # best evidence available that this path is wired correctly. It is
            # also why reward_ranges.py's analytic [-6, 6] is a genuine 6-sigma
            # bound rather than an arbitrary one.
            loss = reward_obj(preprocess(img_01), prompt)
            return (2.0 - loss).unsqueeze(0).to(z.device)

    # ── Style (Gram-matrix, MPGD-faithful) ───────────────────────────────────
    elif reward_model == "style":
        if not cfg.style_ref_paths:
            raise ValueError("reward_model='style' requires cfg.style_ref_paths (one or more image paths).")
        from transformers import CLIPVisionModel, CLIPVisionConfig
        from huggingface_hub import hf_hub_download
        from PIL import Image
        import numpy as np
        import torchvision.transforms as TV

        # CLIP ViT-B/16, per the FMRG paper (Appendix E.4, "Frobenius norm
        # between the Gram matrices of CLIP ViT-B/16 layer-2 features... following
        # MPGD [40]") and MPGD's own style-guidance code (model_name = "ViT-B/16"
        # in their base_clip.py) -- NOT the ViT-H-14 the "clip" reward above uses.
        # Different architecture (depth/width/patch size), so the two encoders
        # cannot share one nn.Module; this loads a second, much smaller one
        # (vision tower only, no text encoder, ~86M params before truncation
        # below -- see conversation history for the measured memory cost).
        style_config = CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch16", cache_dir=HF_CACHE)
        style_vision_model = CLIPVisionModel(style_config)
        # from_pretrained's own loader refuses this repo's pickle .bin under
        # torch<2.6 (no safetensors export exists for it) -- load the state
        # dict directly instead of routing through from_pretrained.
        _style_bin_path = hf_hub_download("openai/clip-vit-base-patch16", "pytorch_model.bin", cache_dir=HF_CACHE)
        _style_sd = torch.load(_style_bin_path, map_location="cpu", weights_only=True)
        style_vision_model.load_state_dict(_style_sd, strict=False)
        # "layer 2" (paper's term) == MPGD's features[2] == HF's hidden_states[3]:
        # verified numerically against MPGD's own OpenAI-CLIP-JIT forward on an
        # identical input -- max abs diff 0.0 at index 3, every neighboring index
        # differs (see conversation history). hidden_states[3] = output after
        # encoder.layers[0:3], so the remaining 9 blocks are never needed --
        # drop them before moving to device.
        style_vision_model.vision_model.encoder.layers = style_vision_model.vision_model.encoder.layers[:3]
        style_vision_model = style_vision_model.to(reward_device).float().eval()
        for p in style_vision_model.parameters():
            p.requires_grad = False

        style_preprocess = TV.Compose([
            TV.Resize(224, interpolation=TV.InterpolationMode.BICUBIC),
            TV.CenterCrop(224),
            TV.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ])

        def _style_patch_tokens(img_01: torch.Tensor) -> torch.Tensor:
            """img_01: [N,3,H,W] in [0,1]. Returns patch-token features [N,196,768].
            fp32 regardless of the vision tower's own dtype: CLIP hidden states
            carry large-magnitude outlier dimensions and the Gram sums 196 terms
            per entry with no normalization (MPGD's own construction -- see
            conversation history), so this stays fp32 the way every other reward
            model in this file already does."""
            x = style_preprocess(img_01)
            out = style_vision_model(pixel_values=x, output_hidden_states=True)
            h = out.hidden_states[-1]      # == hidden_states[3] before truncation
            return h[:, 1:, :].float()     # drop CLS -- MPGD scores patch tokens only

        def _style_gram(patch: torch.Tensor) -> torch.Tensor:
            """patch: [N,L,D] fp32 -> [N,D,D], one Gram matrix per batch element.
            MPGD's own snippet (`feat[..., 0, :]`) hardcodes batch size 1; ported
            literally, a batch>1 run would silently score only the first sample
            and ignore the rest -- same "finite but wrong" failure class as
            everything else this session has been chasing. bmm keeps every
            element distinct (verified: a 3-element batch where element 0 is
            literally the reference image gives that element ~0 and the other
            two their own distinct nonzero distances -- see conversation history)."""
            return torch.bmm(patch.transpose(1, 2), patch)

        # Reference(s): encoded once under no_grad through the exact same path
        # (preprocess, fp32 cast) as the generated image. Gram(s) averaged when
        # there's more than one reference (style mixing) -- neither the FMRG
        # paper nor MPGD specify how multiple references combine; averaging the
        # Grams is our choice, not theirs.
        with torch.no_grad():
            _ref_grams = []
            for _ref_path in cfg.style_ref_paths:
                _ref_img = Image.open(_ref_path).convert("RGB")
                _ref_arr = np.array(_ref_img).astype(np.float32) / 255.0
                _ref_01 = torch.from_numpy(_ref_arr).permute(2, 0, 1).unsqueeze(0).to(reward_device).float()
                _ref_patch = _style_patch_tokens(_ref_01)
                _ref_grams.append(_style_gram(_ref_patch))
            ref_gram = torch.stack(_ref_grams, dim=0).mean(dim=0)   # [1, D, D]

        reward_obj = style_vision_model

        def r_image(z: torch.Tensor) -> torch.Tensor:
            img = (decode_fn or sampler.decode)(z)
            # NOT reward_dtype: style's vision tower is loaded .float() above and
            # stays fp32. --reward_dtype deliberately covers only the four large
            # scorers (clip / pickscore / imagereward / hpsv2, 1.8-4.0 GB each);
            # this tower is a ViT-B/16 truncated to 3 layers and measures
            # 0.12 GB, so half precision would buy nothing and would change the
            # Gram-matrix numerics of a reward that is already MPGD-faithful.
            img_01 = ((img + 1.0) / 2.0).float().to(reward_device)
            gen_patch = _style_patch_tokens(img_01)
            gen_gram = _style_gram(gen_patch)                             # [N, D, D]
            # literal Frobenius norm (not squared) -- matches the paper's wording
            # verbatim and MPGD's own `torch.linalg.norm(residual)` (no `ord`,
            # which defaults to Frobenius for a 2D matrix -- see conversation
            # history), computed per batch element, fp32 throughout.
            dist = torch.linalg.norm(gen_gram - ref_gram, dim=(1, 2))
            return (-dist).to(z.device)   # higher-is-better, matching clip/hpsv2/pickscore/imagereward

    # ── LPIPS (perceptual distance to the uncontrolled output) ───────────────
    elif reward_model == "lpips":
        if ref_holder is None:
            raise ValueError(
                "reward_model='lpips' requires ref_holder (an empty dict the "
                "caller populates with ref_holder['x_ref'] after the "
                "uncontrolled rollout decodes -- see _run_one)."
            )
        import lpips as _lpips_pkg

        # net='alex': OUR choice (see RunConfig.lpips_net docstring above for
        # why, incl. the memory tradeoff against 'vgg'). eval() + frozen
        # params, same pattern as style's CLIP tower -- only the generated
        # image carries a gradient, never the backbone weights or x_ref.
        lpips_net = _lpips_pkg.LPIPS(net=cfg.lpips_net, verbose=False)
        lpips_net = lpips_net.to(reward_device).float().eval()
        for p in lpips_net.parameters():
            p.requires_grad = False

        reward_obj = lpips_net

        def r_image(z: torch.Tensor) -> torch.Tensor:
            x_ref = ref_holder.get("x_ref")
            if x_ref is None:
                # LOUD FAIL: no silent fallback reference, no default image.
                raise RuntimeError(
                    "lpips r_image called before ref_holder['x_ref'] was "
                    "populated. x_ref must be set from the uncontrolled "
                    "rollout's decoded output before this closure is ever "
                    "invoked -- check the insertion point in _run_one."
                )
            # sampler.decode() returns [-1,1], clamped
            # (fluxfm_sampler_reward.py:489-493) -- exactly the range lpips's
            # own default (normalize=False) expects, so no [0,1] rescale here
            # unlike the clip/hpsv2/pickscore/imagereward/style branches
            # above. Verified explicitly, not assumed.
            img = (decode_fn or sampler.decode)(z).float().to(reward_device)
            x_ref_b = x_ref.to(reward_device).expand(img.shape[0], -1, -1, -1)
            dist = lpips_net(img, x_ref_b, normalize=False)   # [N,1,1,1], >=0 always
            dist = dist.view(dist.shape[0])
            return (-dist).to(z.device)   # negated here, same higher-is-better convention as every other branch

    # ── none ─────────────────────────────────────────────────────────────────
    elif reward_model == "none":
        # Zero reward everywhere -- use when steering is done entirely via
        # the constraint function h (CLIP-as-constraint mode). With this,
        # obj = energy + sum_i mu_i h_i, which is what Step 7 of the plan
        # analyses (zero objective AND zero gradient at t=0 when mu_init=0).
        # No model is loaded, so reward_obj is None; every consumer of
        # reward_obj in _run_one guards for that explicitly.
        reward_obj = None

        def r_image(z: torch.Tensor) -> torch.Tensor:
            return z.new_zeros(z.shape[0])

    else:
        raise ValueError(
            f"Unknown reward_model '{reward_model}'. "
            f"Choose from: {REWARD_CHOICES}"
        )

    return r_image, reward_obj


def _resolve_constraint_specs(cfg) -> List[ConstraintSpec]:
    """Single source of truth for "what constraints does this cfg have".

    Step 4 generalizes constraints beyond CLIP text prompts, but
    ``constraint_prompts``/``constraint_thresholds`` stay on RunConfig
    unchanged so every pre-Step-4 CLI invocation and SWEEP_CONFIGS entry
    keeps behaving exactly as before -- this function is the ONE place that
    decides which of the two constraint mechanisms is in effect, so that
    decision never has to be re-derived (or, worse, re-guessed differently)
    at each call site.

    Both set -> raise. Specifying a run's constraints two different ways at
    once is a configuration error (which one wins?), not a case to silently
    prioritize -- same standing rule as everywhere else in this file.
    """
    legacy_set = bool(cfg.constraint_prompts) or bool(cfg.constraint_thresholds)
    if legacy_set and cfg.constraint_specs:
        raise ValueError(
            "Both cfg.constraint_specs and the legacy cfg.constraint_prompts/"
            "constraint_thresholds are set. Use exactly one mechanism -- "
            "mixing them leaves it ambiguous which constraints are actually "
            "active."
        )
    if cfg.constraint_specs:
        return list(cfg.constraint_specs)
    if not legacy_set:
        return []
    # Legacy path: N CLIP text constraints, direction="ge" (score >=
    # threshold satisfied -- the only direction the old CLIP-only builder
    # ever implemented), unnormalized (raw cosine-similarity thresholds, the
    # only units that ever existed before Step 3's normalizer existed).
    n_p, n_t = len(cfg.constraint_prompts), len(cfg.constraint_thresholds)
    if n_t != n_p:
        raise ValueError(
            f"constraint threshold count mismatch: {n_p} constraint prompt(s) "
            f"{list(cfg.constraint_prompts)!r} but {n_t} threshold(s) "
            f"{list(cfg.constraint_thresholds)!r}. Every constraint needs an "
            f"explicit threshold -- missing ones are NOT defaulted to 0.0, "
            f"because a 'clip score >= 0' constraint is vacuously satisfied "
            f"and would appear active in the logs while doing nothing."
        )
    return [
        ConstraintSpec(reward_type="clip", arg=p, threshold=t, direction="ge")
        for p, t in zip(cfg.constraint_prompts, cfg.constraint_thresholds)
    ]


def _specs_from_cli_lists(
    types: List[str],
    cargs: List[str],
    thresholds: List[float],
    directions: List[str],
    normalize: Optional[List[bool]] = None,
    bind_mode: str = "global",
    stats_json: str = DEFAULT_STATS_JSON,
) -> List[ConstraintSpec]:
    """Build ConstraintSpecs from the general-path CLI's parallel lists.

    WHY THIS EXISTS (2026-08-26). Step 4 generalized the constraint builder
    to arbitrary reward types, but left NO CLI surface for it: the only
    constraint flags were the legacy CLIP-only
    ``--constraint_prompts``/``--constraint_thresholds`` pair, so
    ``RunConfig.constraint_specs`` -- the field the whole generalized path
    reads -- was reachable only by editing SWEEP_CONFIGS or writing a driver
    script in Python. That made several planned experiments unlaunchable from
    the command line, since every one of them needs a non-CLIP constraint
    (style / ImageReward / LPIPS). Found while assembling the run commands
    for the first real GPU window.

    Parallel-list shape (``--constraint_types`` / ``--constraint_args`` /
    ``--constraint_thresholds`` / ``--constraint_directions``) deliberately
    mirrors the existing ``--constraint_prompts``/``--constraint_thresholds``
    convention rather than introducing a second style (colon-delimited
    strings, JSON) -- and every count mismatch RAISES, exactly as the legacy
    path's own threshold-count check does, rather than broadcasting or
    padding.

    `directions` is REQUIRED, never defaulted, per plan 4.2 ("Do not infer
    direction from the reward's name or sign"). NOTE the convention, which is
    easy to get backwards: ``_build_reward_fn`` returns higher-is-better for
    every type (style negates its raw Gram distance internally), so style is
    "ge" in THESE units even though the plan's 4.2 text describes
    style-*distance* as an upper bound.
    """
    n = len(types)
    for name, seq in (("--constraint_args", cargs),
                      ("--constraint_thresholds", thresholds),
                      ("--constraint_directions", directions)):
        if len(seq) != n:
            raise ValueError(
                f"constraint list length mismatch: {n} --constraint_types "
                f"{list(types)!r} but {len(seq)} {name} {list(seq)!r}. Every "
                f"constraint needs its own explicit entry in each list -- "
                f"missing ones are NOT broadcast or defaulted, since a "
                f"silently-defaulted threshold or direction produces a "
                f"constraint that looks active in the logs while enforcing "
                f"something other than what was asked."
            )
    if normalize is None:
        normalize = [False] * n
    elif len(normalize) != n:
        raise ValueError(
            f"constraint list length mismatch: {n} --constraint_types but "
            f"{len(normalize)} --constraint_normalize {list(normalize)!r}. "
            f"Omit the flag entirely to leave every constraint unnormalized."
        )
    return [
        ConstraintSpec(
            reward_type=t, arg=a, threshold=float(th), direction=d,
            normalize=bool(nz), bind_mode=bind_mode, stats_json=stats_json,
        )
        for t, a, th, d, nz in zip(types, cargs, thresholds, directions, normalize)
    ]


def _constraint_names(cfg) -> List[str]:
    """Ordered, human-readable identifier for each constraint column.

    The CSV stores h_i / mu_i / thr_i in fixed positional columns, so the
    column index alone does not say WHICH reward h_0 is -- and the Task 3
    constraint pair may change (style+ImageReward vs. the CLIP+style
    fallback), at which point h_0 would silently change meaning. This is the
    same defect that forced the reward model to be inferred from directory
    names for 222 of 232 historical run dirs; the `constraint_names` CSV
    column exists so that never has to be inferred again.

    Format is "<reward_type>:<arg>" -- "style:xingkong_style_ref.jpg",
    "imagereward:a smiling man", or "clip:a smiling face" for the legacy
    path. Commas inside an arg are replaced with ';' because the column is
    written comma-joined and must stay splittable.
    """
    return [
        f"{spec.reward_type}:{spec.arg.replace(',', ';')}"
        for spec in _resolve_constraint_specs(cfg)
    ]


def _validate_constraint_cfg(cfg) -> int:
    """Validate one config's constraint settings; return the constraint count.

    All checks RAISE. None warns-and-continues: each describes a run that
    would complete successfully while doing something other than what was
    asked, and would be indistinguishable from a real run afterwards.

    Called twice -- from main() before FLUX is loaded (so a bad sweep fails in
    seconds rather than after a multi-minute model load) and from _run_one,
    where the config is authoritative and may have come from SWEEP_CONFIGS
    rather than the CLI.
    """
    specs = _resolve_constraint_specs(cfg)   # raises on the both-set case, on
                                              # threshold-count mismatch (legacy
                                              # path), and on any invalid
                                              # ConstraintSpec (bad reward_type/
                                              # direction, via __post_init__)
    n_constraints = len(specs)
    if not n_constraints:
        return 0

    # (i) Constraints are supported ONLY by the constrained control types.
    # Originally ConicFlow-only (locked decision); "hardflow" joined it on
    # 2026-08-31 as the constrained baseline -- it is a constrained method by
    # construction, so running it with no constraint would be pointless
    # rather than merely degraded. fmrg/boffi_fmrg remain unconstrained
    # methods and still RAISE here: an earlier copy logged
    # "WARNING: ... ignoring.", ran unconstrained with h=None and exited 0,
    # while the CSV row still claimed n_constraints=2.
    if cfg.control_type not in CONSTRAINED_CONTROL_TYPES:
        raise ValueError(
            f"constraints requested ({n_constraints}: "
            f"{[s.reward_type + ':' + s.arg for s in specs]!r}) with "
            f"control_type={cfg.control_type!r}. Constraints are supported "
            f"only by {sorted(CONSTRAINED_CONTROL_TYPES)}; this is a "
            f"configuration error, not a degraded mode. Use one of those, or "
            f"drop the constraints."
        )

    # (ii) The silent no-op (see the design plan): lr_dual and mu_init BOTH default to
    # 0.0, which makes sum_i mu_i h_i identically zero for the whole run. The
    # constraint is then a no-op with no error anywhere.
    if cfg.lr_dual == 0.0 and cfg.mu_init == 0.0:
        raise ValueError(
            f"constraints requested ({n_constraints}) but "
            f"lr_dual=0.0 AND mu_init=0.0. The constraint term sum_i mu_i h_i "
            f"is then identically zero for the entire run: the constraint is a "
            f"silent no-op. Set --mu_init > 0 for a fixed penalty, "
            f"--lr_dual > 0 for dual ascent, or both."
        )

    # (iii) See MAX_CONSTRAINTS -- the CSV column set is fixed at import time.
    if n_constraints > MAX_CONSTRAINTS:
        raise ValueError(
            f"{n_constraints} constraints requested but MAX_CONSTRAINTS="
            f"{MAX_CONSTRAINTS}. Not truncated and not warned-and-continued: "
            f"raise MAX_CONSTRAINTS (CSV_KEYS is generated from it) instead."
        )

    # (iv) lpips normalization is ill-posed on uncontrolled calibration
    # samples (plan §3.7 -- x_ref is the same-seed uncontrolled output, so
    # LPIPS is exactly 0 there and Family B's sqrt(E[x^2]) divides by zero).
    # Raised here, at validation time, rather than left to surface as an
    # opaque ZeroDivisionError deep inside the normalizer once the run is
    # already underway.
    for spec in specs:
        if spec.reward_type == "lpips" and spec.normalize:
            raise ValueError(
                "ConstraintSpec(reward_type='lpips', normalize=True): LPIPS "
                "normalization is ill-posed on uncontrolled calibration "
                "samples (plan section 3.7, unresolved). "
                "Pass normalize=False to use a raw-scale LPIPS threshold, or "
                "wait for that decision."
            )

    return n_constraints


def _nvidia_proc_uuid_to_bus() -> dict:
    """Map GPU UUID -> PCI bus location, read from /proc/driver/nvidia/gpus/.

    Used instead of nvidia-smi (no subprocess) and instead of
    torch.cuda.get_device_properties(i).pci_bus_id, which does NOT exist in
    torch 2.5.1 (verified: _CudaDeviceProperties exposes `uuid` but not
    `pci_bus_id`). UUID is a stronger identifier than the index anyway --
    under CUDA_VISIBLE_DEVICES the torch index is a remap, but the UUID
    always names the physical card.
    """
    out = {}
    root = "/proc/driver/nvidia/gpus"
    try:
        entries = os.listdir(root)
    except OSError:
        return out
    for bus in entries:
        try:
            with open(os.path.join(root, bus, "information")) as f:
                txt = f.read()
        except OSError:
            continue
        uuid = None
        for line in txt.splitlines():
            if line.startswith("GPU UUID:"):
                uuid = line.split(":", 1)[1].strip()
            elif line.startswith("Bus Location:") and uuid:
                out[uuid] = line.split(":", 1)[1].strip()
    return out


def _report_cuda_topology(log=print) -> None:
    """Print the CUDA device topology this PROCESS actually sees.

    WHY (2026-08-26). We are testing the hypothesis that the reproducible
    ImageReward fault on cuda:1 (5/5) is an AMBIENT-DEVICE mismatch:
    torch.cuda.current_device() stays 0 for the whole process while every
    tensor lives on cuda:1, so any kernel that reads the ambient device
    rather than dispatching on tensor device operates on the wrong card.
    The hypothesis is ABOUT current_device(), which we had been inferring
    rather than observing -- so it is printed, not assumed.

    The UUID and PCI bus location are printed because under
    CUDA_VISIBLE_DEVICES the torch index is a REMAP: "cuda:0" may be
    physical GPU 1. Without them a failure is uninterpretable -- one cannot
    tell a refuted hypothesis from a remap that never took effect. Match the
    bus location against `nvidia-smi --query-gpu=index,pci.bus_id,uuid`.
    """
    if not torch.cuda.is_available():
        log("  [cuda topology] torch.cuda.is_available() = False -- no GPU visible")
        return
    log(f"  [cuda topology] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}")
    log(f"  [cuda topology] torch.cuda.device_count()   = {torch.cuda.device_count()}")
    log(f"  [cuda topology] torch.cuda.current_device() = {torch.cuda.current_device()}"
        f"   <-- the ambient device the hypothesis is about")
    uuid_to_bus = _nvidia_proc_uuid_to_bus()
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        uuid = str(getattr(props, "uuid", "<no uuid attr>"))
        # torch reports the raw UUID; /proc prefixes it with "GPU-"
        bus = uuid_to_bus.get(uuid) or uuid_to_bus.get(f"GPU-{uuid}") or "<unresolved>"
        log(f"  [cuda topology]   torch cuda:{i} -> {props.name!r}  "
            f"uuid={uuid}  pci_bus={bus}  total={props.total_memory/1e9:.2f} GB")
    if not uuid_to_bus:
        log("  [cuda topology]   (could not read /proc/driver/nvidia/gpus -- "
            "cross-check with nvidia-smi manually)")


def _validate_dual_options_cfg(cfg, n_constraints: int) -> None:
    """Fail-fast mirror of the two primal/dual option guards that otherwise
    only fire INSIDE control_loops.receding_horizon_flow_control.

    WHY THIS EXISTS (2026-08-26). `n_alt` (the design plan) and
    `skip_first_primal_step` (the design plan option (b)) both validate
    themselves at the top of `receding_horizon_flow_control` -- correct, but
    that function is only reached AFTER the FLUX load AND after the
    uncontrolled baseline rollout completes, so a mis-specified Step 7
    comparison sweep would burn several minutes per config before saying so.
    Every other configuration error in this file fails before the model
    load (Step 1's whole D3/D4/D5 guard set, verified at the CLI level as
    "exit 1 before FLUX loads"), so these two were the odd ones out. The
    in-function guards are deliberately KEPT -- this is a second, earlier
    check of the same conditions, not a relocation, because
    `receding_horizon_flow_control` is importable and callable directly
    (e.g. from notebooks) without going through run_preflight at all.

    Third check, which the in-function guards structurally CANNOT make:
    both options are ConicFlow-only and are simply not passed on the fmrg /
    boffi_fmrg / none paths, so setting either away from its default with a
    non-conicflow control_type is a silent no-op -- the run completes, exits 0,
    and its CSV row records `n_alt=2` for a run that never alternated. Same
    class as D5 (constraints + non-conicflow control_type) and given the same
    treatment: raise.
    """
    dual_opts_set = []
    if cfg.n_alt != 1:
        dual_opts_set.append(f"n_alt={cfg.n_alt}")
    if cfg.skip_first_primal_step:
        dual_opts_set.append("skip_first_primal_step=True")
    if cfg.dual_adaptive:
        dual_opts_set.append("dual_adaptive=True")
    if not dual_opts_set:
        return

    # (i) ConicFlow-only. Neither option is forwarded on any other control path, so
    # a non-conicflow run would silently ignore it while the CSV row still claims it.
    if cfg.control_type != "conicflow":
        raise ValueError(
            f"{' and '.join(dual_opts_set)} requested with "
            f"control_type={cfg.control_type!r}. Both are ConicFlow-only (they are "
            f"arguments to receding_horizon_flow_control and are not passed "
            f"on the fmrg/boffi_fmrg/none paths), so this would run to "
            f"completion having silently ignored them while the CSV row "
            f"still recorded them -- indistinguishable afterwards from a run "
            f"that honored them. Use control_type=conicflow, or leave both at "
            f"their defaults (n_alt=1, skip_first_primal_step=False)."
        )

    # (ii) n_alt's own domain check, hoisted from receding_horizon_flow_control.
    if cfg.n_alt < 1:
        raise ValueError(f"n_alt must be >= 1, got {cfg.n_alt}.")

    # (iii) Both options are no-ops without an ACTIVE dual update: n_alt>1
    # re-solves the same primal window to the same optimum, and
    # skip_first_primal_step's entire effect IS one dual-ascent update.
    # Mirrors the in-function guards' `h is not None and lr_dual > 0.0`,
    # expressed in cfg terms (h is built from the constraint specs, so
    # "h is not None" <=> n_constraints > 0).
    if not (n_constraints > 0 and cfg.lr_dual > 0.0):
        raise ValueError(
            f"{' and '.join(dual_opts_set)} requested but the dual update is "
            f"not active (n_constraints={n_constraints}, "
            f"lr_dual={cfg.lr_dual}). Both options only do anything when "
            f"there is at least one constraint AND lr_dual > 0: n_alt>1 "
            f"re-solves the same primal window to the same optimum every "
            f"time without a dual step in between, and "
            f"skip_first_primal_step's whole effect is one projected "
            f"dual-ascent update. Set constraints and --lr_dual > 0, or "
            f"leave these at their defaults. (Note: mu_init alone is not "
            f"enough -- a fixed penalty never updates, so there is nothing "
            f"to alternate with.)"
        )


def run_preflight(
    cfg: RunConfig,
    *,
    check_boffi_rng: bool = False,
    latent: Optional[torch.Tensor] = None,
    log: Callable = print,
) -> int:
    """Single preflight entry point for every code path that executes a
    RunConfig. Existing entry points: _run_one (ConicFlow/fmrg/boffi_fmrg),
    _run_uncontrolled (the uncontrolled baseline -- called both from inside
    _run_one and, standalone, by the Step 3 calibration script once it's
    built), main()'s pre-FLUX-load sweep check, and
    _sample_boffi_fmrg_with_trajectory.

    WHY ONE FUNCTION. Before 2026-08-25 each of those call sites called
    _assert_locked_defaults / _validate_constraint_cfg / (for the boffi path)
    _assert_boffi_rng_reachability_closed directly and separately. Three
    independent call sites are three places to drift out of sync -- exactly
    the class of failure Step 1 spent an entire session undoing (two
    diverged run_sweep.py copies). Every future entry point
    (starting with the Step 3 calibration script) should call THIS function,
    not the three guards individually.

    Always runs (no parameter to skip these -- they are cheap, pure
    functions of `cfg`, and never produce a false positive regardless of
    which caller invokes them, so there is no real case where skipping is
    correct, only cases where a caller forgets to call the underlying guard
    at all):
      - _assert_locked_defaults  (constraint_on / warmstart; drift_detach was
        lifted 2026-09-16 -- see _RETIRED_LOCKS)
      - _validate_constraint_cfg (D5 conicflow-only / D4 silent-no-op / D3
        threshold-count / MAX_CONSTRAINTS)
      - _validate_dual_options_cfg (n_alt / skip_first_primal_step: ConicFlow-only,
        and both no-ops without an active dual update -- an EARLIER copy of
        guards that also live inside receding_horizon_flow_control, so a bad
        Step 5/7 sweep fails before the FLUX load rather than minutes into
        each config; see that function's docstring for why both exist)

    Opt-in, via an explicit parameter rather than an implicit omission at the
    call site (per instruction: "if some check genuinely doesn't apply... make
    it a parameter of the preflight, not an omission at the call site"):
      - check_boffi_rng + latent: only _sample_boffi_fmrg_with_trajectory
        passes True, because it's the only call site that (a) only ever
        applies to control_type="boffi_fmrg" and (b) has the actual latent
        tensor in hand -- z0 does not exist yet at _run_one's own top-level
        preflight call, which runs before the sampler is touched at all
        (Step 1 §1.4j), so that check structurally cannot be folded into the
        same call as the other two. Every other call site passes the
        default (False, no latent) EXPLICITLY by not overriding it -- this is
        documented here, not a silent gap discovered by reading each site.

    Returns n_constraints (from _validate_constraint_cfg).
    """
    _assert_locked_defaults(cfg, log=log)
    n_constraints = _validate_constraint_cfg(cfg)
    _validate_dual_options_cfg(cfg, n_constraints)
    if check_boffi_rng:
        _assert_boffi_rng_reachability_closed(cfg, latent)
    # Deliberately NOT routed through `log`: every sweep launcher passes
    # --quiet, which sets log to a no-op, and this one line is run provenance
    # rather than progress chatter -- suppressing it is how the reg/lam
    # mismatch would go unnoticed in exactly the runs that matter.
    # main() resolves hardflow_reg for every config it builds, but this is the
    # universal funnel and a future entry point (the Step 3 calibration script,
    # say) could construct a RunConfig directly. An unresolved None would reach
    # hardflow_image as a penalty weight and TypeError deep in the inner loop;
    # raise here, at the top of the run, naming the fix.
    if cfg.control_type == "hardflow" and cfg.hardflow_reg is None:
        raise ValueError(
            "hardflow config reached run_preflight with hardflow_reg=None. "
            "Call _resolve_hardflow_reg(cfg) after building the config (main() "
            "does this for every sweep path), or set an explicit value."
        )
    _log_hardflow_reg_equivalence(cfg)
    return n_constraints


# The pre-2026-09-16 fixed default. Retained ONLY for non-hardflow control
# types, which never read reg -- see _resolve_hardflow_reg's docstring for why
# those rows deliberately keep the old number instead of acquiring a new one.
_HARDFLOW_REG_LEGACY_DEFAULT = 5e-4


def _resolve_hardflow_reg(cfg: RunConfig, *, log: Callable = print) -> None:
    """Resolve ``cfg.hardflow_reg`` in place; ``None`` means derive N/(2*lam).

    WHY DERIVED RATHER THAN SUPPLIED (2026-09-16). reg and lam are the same
    physical quantity -- the coefficient on the squared displacement away from
    the nominal flow -- but ConicFlow gets N/(2*lam) structurally out of its energy
    term (``resid^2 / (2*lam*dt)`` with ``dt = 1/N``), while HardFlow took reg
    as a raw number that nothing derived and nothing checked. Since reg carries
    an N-dependence, a value matched to a lam at one step count is wrong at
    every other one, silently. Deriving removes that entire class of error.

    NOT APPLIED to non-hardflow control types. reg is never passed to the
    conicflow/fmrg/boffi_fmrg paths, so it is inert there; those rows keep
    _HARDFLOW_REG_LEGACY_DEFAULT rather than acquiring a freshly-computed
    number in an existing CSV column, which keeps pre-2026-09-16 conicflow/fmrg rows
    comparable with post- ones on a column that never meant anything for them.

    LOUD FAIL on lam <= 0 rather than a fallback: N/(2*lam) is undefined at
    lam=0 and NEGATIVE for lam<0, and a negative reg flips the penalty into a
    reward for deviation -- precisely the sign bug this port exists to fix.
    """
    if cfg.hardflow_reg is not None:
        return                          # explicit override: never recomputed
    if cfg.control_type != "hardflow":
        cfg.hardflow_reg = _HARDFLOW_REG_LEGACY_DEFAULT
        return
    if cfg.lam <= 0.0:
        raise ValueError(
            f"hardflow with lam={cfg.lam!r}: reg = N/(2*lam) is undefined for "
            f"lam <= 0. Either use lam > 0, or pass --sweep_hardflow_reg to "
            f"set the deviation penalty directly (0.0 = no penalty at all, "
            f"the pre-2026-09-09 behaviour)."
        )
    cfg.hardflow_reg = cfg.N / (2.0 * cfg.lam)


def _log_hardflow_reg_equivalence(cfg: RunConfig, *, log: Callable = print) -> None:
    """Print the ConicFlow `lam` that this run's `hardflow_reg` is equivalent to.

    WHY THIS IS PRINTED RATHER THAN ASSERTED. hardflow_reg and lam are the same
    physical quantity under reg = N/(2*lam) (see the RunConfig comment for the
    derivation), so a HardFlow run implies an equivalent lam whether or not
    anyone intended one.

    UPDATED 2026-09-16. This docstring used to say `cfg.lam` is INERT on the
    hardflow path and that a guard comparing reg to N/(2*cfg.lam) "would fire
    on every correctly-configured run". Both stopped being true when
    _resolve_hardflow_reg started deriving reg FROM lam: the derived case now
    agrees with cfg.lam by construction, so a disagreement means someone passed
    --sweep_hardflow_reg explicitly and it no longer matches the lam on the
    same row. That is worth saying out loud.

    Still WARNED, not raised. A deliberate lam/reg mismatch is a legitimate
    ablation -- reg=0.0 reproducing the pre-2026-09-09 no-penalty runs is
    exactly one -- so this reports what the run actually is instead of refusing
    to run it.
    """
    if cfg.control_type != "hardflow":
        return
    if cfg.hardflow_reg == 0.0:
        log("  hardflow_reg=0.0 -- NO control-effort penalty (equivalent lam = inf). "
            "This is the pre-2026-09-09 behaviour; energy metrics are not "
            "comparable with runs that derive reg from lam.")
        return
    lam_equiv = cfg.N / (2.0 * cfg.hardflow_reg)
    log(f"  hardflow_reg={cfg.hardflow_reg:g} at N={cfg.N} == ConicFlow lam {lam_equiv:g} "
        f"(reg = N/(2*lam)). Compare against ConicFlow arms run at that lam.")

    # See the docstring: derived reg agrees with cfg.lam by construction, so a
    # disagreement can only come from an explicit --sweep_hardflow_reg.
    if not math.isclose(lam_equiv, cfg.lam, rel_tol=1e-9):
        log(f"  !! reg was supplied explicitly and does NOT match cfg.lam="
            f"{cfg.lam:g} -- this run is regularized as lam={lam_equiv:g}, and "
            f"the `lam` column on its CSV row is NOT what it was regularized "
            f"at. Omit --sweep_hardflow_reg to derive reg from lam.")

    # hardflow_image raises when lr*reg >= 1 (its inner ascent is explicit
    # fixed-step, so the deviation error stops contracting). Predicted here, at
    # config time, because the alternative is discovering it after FLUX has
    # spent ~90s loading. Deriving reg makes this reachable from a low lam,
    # which is why it is worth predicting now and was not before.
    if cfg.lr_inner * cfg.hardflow_reg >= 1.0:
        if cfg.hardflow_lr_alpha is not None:
            log(f"  lr_inner={cfg.lr_inner:g} * reg={cfg.hardflow_reg:g} = "
                f"{cfg.lr_inner * cfg.hardflow_reg:g} >= 1 -- below the floor "
                f"lam >= N*lr/2 = {cfg.N * cfg.lr_inner / 2.0:g}; hardflow_lr_alpha="
                f"{cfg.hardflow_lr_alpha:g} will set lr = "
                f"{cfg.hardflow_lr_alpha / cfg.hardflow_reg:.4g} for this run.")
        else:
            log(f"  !! lr_inner={cfg.lr_inner:g} * reg={cfg.hardflow_reg:g} = "
                f"{cfg.lr_inner * cfg.hardflow_reg:g} >= 1 -- hardflow_image WILL "
                f"raise. The floor is lam >= N*lr/2 = {cfg.N * cfg.lr_inner / 2.0:g}. "
                f"Pass --hardflow_lr_alpha to lift it.")


def _build_constraint_fn(
    sampler,
    cfg,
    specs: List[ConstraintSpec],
    reward_device: torch.device,
    dtype,
    ref_holder: dict,
    drop_text_tower: bool = False,
    reward_dtype: torch.dtype = torch.float32,
):
    """Build h(z) -> (B, C) from a list of ConstraintSpec (Step 4.1).

    Reuses ``_build_reward_fn`` per spec rather than writing parallel scorers
    (4.5): each spec gets a shallow-copied `cfg` (via ``dataclasses.replace``)
    with ``edit_prompt`` (text-target reward types) or ``style_ref_paths``
    (style) overridden to that spec's `arg`, so e.g. two CLIP constraints
    against two DIFFERENT text targets ("a smiling face", "with glasses")
    each get correctly-scoped text features -- exactly the override the old
    CLIP-only builder did by hand, now general to every reward type
    ``_build_reward_fn`` supports. "lpips" ignores `arg`: its reference is
    always the same-seed uncontrolled rollout via `ref_holder`, identically
    to the reward path.

    Sign convention (4.2), never inferred:
        direction == "ge":  h_i(z) = threshold_i - score_i(z)   (score >= threshold satisfied)
        direction == "le":  h_i(z) = score_i(z) - threshold_i   (score <= threshold satisfied)
    where score_i is the (optionally normalized) output of that spec's
    reward closure. Normalization is applied as plain tensor arithmetic
    (Normalizer.__call__ / Stats.normalize do `(r-mean)/std` or `r/rms` with
    python-float mean/std/rms against whatever `r` is), so it stays
    differentiable w.r.t. z exactly like the unnormalized case -- required,
    since h participates in the primal solve's objective gradient.

    Decode sharing (4.3): every spec's r_image independently calls
    sampler.decode(z) inside _build_reward_fn unless given a shared
    decode_fn. stack_constraints's combined(x) calls every h_i with the SAME
    x, so one single-slot cache built here and threaded into every
    _build_reward_fn call lets C constraints at one z share one VAE decode
    instead of paying C. The slot is keyed by tensor identity and evicted on
    every miss (capacity 1), so it never accumulates memory across the many
    DIFFERENT z's one ConicFlow run's inner solve evaluates h at -- it only ever
    helps the C constraints within a single h(z) call, never across calls.

    Returns None when specs is empty.
    """
    if not specs:
        return None

    # ── single-slot decode cache ────────────────────────────────────────────
    # BUG FIX 2026-09-07. This was `key = id(z)` into a dict. `id()` is a
    # CPython object ADDRESS, and addresses are recycled the moment an object
    # dies -- so a freshly allocated latent could land on the address of the
    # freed previous one, hit the cache, and silently receive the PREVIOUS z's
    # decoded image. Not hypothetical: in the 20260904_231018_lpr2c grid the
    # constraint-trajectory loop
    #
    #     zt = traj_ctrl[t_idx].to(...)          # new object, may reuse an address
    #     raw_vals.append(h_fn(zt))
    #     eval_vals.append(h_fn(endpoint_map(s_t, zt)))   # frees an intermediate
    #
    # produced h_raw[k] == h_endpoint[k-1] EXACTLY, in both constraint channels
    # at once, in 28 of 240 step evaluations (~12%) spread over 17 of the 24
    # runs. The endpoint/raw alignment guard only caught the 3 cases that
    # happened to land on the final row; every other hit was silent.
    #
    # The fix is a WEAK reference to the key tensor. If the cached z is still
    # alive, `ref() is z` identifies it exactly; if it died, the weakref reads
    # None, so an address-recycled newcomer is a miss rather than a false hit.
    # A strong reference would also be correct but would pin the cached z's
    # autograd graph alive between inner-loop iterations, which the
    # memory budget cannot spare.
    _decode_slot: dict = {"ref": None, "val": None}

    def _shared_decode(z: torch.Tensor) -> torch.Tensor:
        ref = _decode_slot["ref"]
        if ref is not None and ref() is z:
            return _decode_slot["val"]
        # Capacity 1: drop the old decode BEFORE computing the new one.
        _decode_slot["ref"] = None
        _decode_slot["val"] = None
        out = sampler.decode(z)
        _decode_slot["ref"] = weakref.ref(z)
        _decode_slot["val"] = out
        return out

    def _make_hi(r_image_i, threshold, direction, normalizer):
        """Factory to capture (r_image_i, threshold, direction, normalizer) by value."""
        def hi(z: torch.Tensor) -> torch.Tensor:
            score = r_image_i(z)
            if normalizer is not None:
                score = normalizer(score)
            if direction == "ge":
                return threshold - score
            else:  # "le", validated by ConstraintSpec.__post_init__
                return score - threshold
        return hi

    h_fns = []
    for spec in specs:
        spec_cfg = cfg
        if spec.reward_type in _TEXT_TARGET_REWARD_TYPES:
            spec_cfg = _dc_replace(cfg, edit_prompt=spec.arg)
        elif spec.reward_type == "style":
            spec_cfg = _dc_replace(cfg, style_ref_paths=[spec.arg])
        # lpips: no cfg override -- arg is unused, ref_holder wiring is shared
        # verbatim with the reward path.

        r_image_i, _ = _build_reward_fn(
            sampler, spec_cfg, reward_device, dtype,
            reward_model=spec.reward_type, ref_holder=ref_holder,
            decode_fn=_shared_decode,
            drop_text_tower=drop_text_tower, reward_dtype=reward_dtype,
        )

        normalizer = None
        if spec.normalize:
            _store = _rn.StatsStore(spec.stats_json)
            # BUG FIX 2026-08-26: this passed `prompt=spec.arg`, which is the
            # WRONG key and made bind_mode="prompt" unusable for every
            # non-CLIP constraint. The calibration stats are keyed by the
            # GENERATION prompt (calibrate_rewards.py: `prompt_key =
            # args.prompt`), because what was calibrated is "the distribution
            # of reward R over uncontrolled images generated from prompt P".
            # `spec.arg` is something else entirely per reward type -- a
            # style REFERENCE IMAGE PATH, or an imagereward TEXT TARGET -- so
            # the lookup was for keys that can never exist
            # (store.get('style', '/path/to/style_ref.jpg') -> None,
            # raising "No calibration stats"). It coincidentally looked right
            # for a CLIP constraint whose text target happened to equal the
            # generation prompt, which is why it survived Step 4's tests.
            # cfg.prompt is the correct key for every reward type.
            normalizer = _rn.build_normalizer(
                _store, spec.reward_type, spec.bind_mode,
                prompt=(cfg.prompt if spec.bind_mode == "prompt" else None),
            )

        h_fns.append(_make_hi(r_image_i, spec.threshold, spec.direction, normalizer))

    return stack_constraints(h_fns)


def _build_boffi_reward_ensemble(reward_device, reward_model: str) -> RewardEnsemble:
    """RewardEnsemble with only `reward_model`'s weight turned on.

    Lets sample_reward_guided's guidance be driven by a single reward model,
    matching what our fmrg_j_image/ConicFlow paths score against -- the other three
    models are never loaded (RewardEnsemble skips loading a model whose weight
    is 0).
    """
    weight_kw = {"hps_weight": 0.0, "imagereward_weight": 0.0,
                 "pickscore_weight": 0.0, "clip_weight": 0.0}
    key = {
        "clip": "clip_weight", "hpsv2": "hps_weight",
        "pickscore": "pickscore_weight", "imagereward": "imagereward_weight",
    }.get(reward_model.lower())
    if key is None:
        raise ValueError(
            f"boffi_fmrg needs a single scoring reward model, got --reward {reward_model!r}. "
            f"Choose one of: clip, hpsv2, pickscore, imagereward."
        )
    weight_kw[key] = 1.0
    return RewardEnsemble(device=reward_device, dtype=torch.float32, cache_dir=HF_CACHE, **weight_kw)


def _clip_score_pixel(clip_reward_obj, preprocess, img_01: torch.Tensor, prompt: str, device) -> torch.Tensor:
    """Raw CLIP cosine similarity for an already-decoded [0,1] pixel image.

    Same convention as _build_reward_fn's "clip" branch (raw cosine, no
    logit_scale, no loss offset) so it's directly comparable to r_unc/r_ctrl
    from the conicflow/fmrg paths when --reward clip is used. Reuses whatever CLIP
    weights are already loaded via _build_reward_fn -- no extra GPU memory.
    """
    with torch.no_grad():
        toks = clip_reward_obj.tokenizer(
            prompt, return_tensors="pt", padding=True, max_length=77, truncation=True,
        ).to(device)
        txt_feat = clip_reward_obj.model.get_text_features(**toks)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        img_feat = clip_reward_obj.model.get_image_features(preprocess(img_01.to(device)))
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        return (img_feat @ txt_feat.T).squeeze(-1)


def _independent_clip_rescore(sampler, z: torch.Tensor, prompt: str, device) -> float:
    """Re-score a latent's raw CLIP cosine similarity via a code path that
    shares NO object with _build_reward_fn's clip closure or _clip_score_pixel
    (which explicitly reuses the closure's weights). Added 2026-08-21, Phase B:
    ConicFlow shares one r_image closure between the optimization gradient and the
    logged r_ctrl -- a clean-looking log is not by itself evidence the
    optimized objective was clean. This builds a fresh CLIPModel + tokenizer,
    computes a fresh text embedding, and re-decodes the latent fresh, so the
    only thing in common with the logged value is the underlying math (raw
    cosine similarity, same convention as r_unc/r_ctrl) and the CLIP checkpoint
    weights themselves (loaded fresh from disk, not the resident instance).
    Deliberately not reused for speed -- reuse would defeat the point.
    """
    from transformers import CLIPModel, AutoProcessor
    with torch.no_grad():
        img = sampler.decode(z)
        img_01 = ((img + 1.0) / 2.0).float().to(device)

        fresh_tokenizer = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=HF_CACHE
        )
        fresh_model = CLIPModel.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=HF_CACHE
        ).to(device, dtype=torch.float32).eval()
        fresh_preprocess = create_clip_preprocess(224)

        toks = fresh_tokenizer(
            prompt, return_tensors="pt", padding=True, max_length=77, truncation=True,
        ).to(device)
        txt_feat = fresh_model.get_text_features(**toks)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        img_feat = fresh_model.get_image_features(fresh_preprocess(img_01))
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        score = (img_feat @ txt_feat.T).squeeze(-1).mean().item()

        del fresh_model, fresh_tokenizer
    return score


def _run_one(
    sampler,
    cfg: RunConfig,
    device,
    reward_device,
    dtype,
    quiet: bool,
    *,
    fast: bool = False,
    low_memory: bool = True,
    reward_model: str = "clip",
    reward_dtype: torch.dtype = torch.float32,
    probe_h_repeat: int = 0,
    probe_hardflow_duals: bool = False,
    pbar_rewards: bool = False,
) -> dict:
    """Execute a single config and return a metrics dict.

    ``probe_h_repeat`` / ``probe_hardflow_duals`` are DIAGNOSTICS, default-off
    and inert when off. Both exist to answer questions raised by the
    ``20260904_231018_lpr2c`` grid (see the two-constraint report):

      probe_h_repeat K       -- 3 of 24 runs tripped the endpoint/raw alignment
                                guard, i.e. h(endpoint_map(1, z_N)) != h(z_N)
                                although endpoint_map at s=1 reduces to
                                ``z - 0.0*u`` and is bitwise the identity. The
                                other 21 agreed to 0.000e+00 exactly, so the
                                gap is sporadic, not a convention error. This
                                evaluates h K extra times on both tensors and
                                records the spread, which separates "h_fn is
                                nondeterministic" from "endpoint_map is not the
                                identity" -- the two remaining explanations.
      probe_hardflow_duals   -- HardFlow's mu_1 reads exactly 0.000 at every
                                step in 10 of 12 runs. ``mu_hist`` only records
                                the multiplier after each step's LAST dual
                                update, so "never rose" and "rose then was
                                driven back to zero" are indistinguishable in
                                it. This records every dual update.
    """
    log = (lambda *a, **k: None) if quiet else print

    # 2026-09-05 memory work. The clip/pickscore closures cache a normalized
    # text embedding at build time and thereafter read only image features, so
    # their text towers (1.42 GB of CLIP ViT-H-14's measured 4.01 GB) are dead
    # weight for the whole run. The one code path that still needs text features
    # afterwards is _clip_score_pixel, reachable ONLY from the boffi_fmrg branch
    # below -- so gate on exactly that rather than on a flag someone has to
    # remember to set. Freed towers are replaced by _DroppedTower, which raises
    # if anything reads them, so a future scoring path that needs text features
    # fails loudly instead of silently getting the wrong number.
    _drop_towers = cfg.control_type != "boffi_fmrg"

    # Single preflight call (see run_preflight): locked-defaults + constraint
    # validation, both before tune_sampler_speed / the prompt re-encode /
    # z0's manual_seed touch anything -- printed + asserted at the true start
    # of every run, not only in main()'s precheck, and covers SWEEP_CONFIGS
    # entries that never go through argparse. check_boffi_rng is NOT set here
    # (z0 doesn't exist yet) -- that check runs separately, later, inside
    # _sample_boffi_fmrg_with_trajectory, once the latent exists.
    n_constraints = run_preflight(cfg, log=log)

    # ── Peak-memory instrumentation (added 2026-08-26 for the design plan) ──────
    # Step 6's stated deliverable is "a peak-memory number... for the actual
    # Task 3 pair under the actual loop" -- but nothing in this file measured
    # memory at all: the only peak-memory numbers on record came from
    # calibrate_rewards.py (no ConicFlow, no gradients) and from an early
    # single fwd+bwd probe. So Step 6 was not satisfiable by running this
    # script, independently of GPU availability. Reset here (per run, after
    # preflight so a rejected config doesn't clear a prior run's stats) and
    # read back into the result dict / CSV below, so EVERY run from now on
    # records its own peak rather than this being a one-off measurement that
    # has to be redone whenever the question comes up again.
    _mem_devices = {}
    for _label, _dev in (("flux", device), ("reward", reward_device)):
        _d = torch.device(_dev) if not isinstance(_dev, torch.device) else _dev
        if _d.type == "cuda" and torch.cuda.is_available():
            _mem_devices[_label] = _d
    for _d in set(_mem_devices.values()):
        torch.cuda.reset_peak_memory_stats(_d)
    if n_constraints and cfg.lr_dual == 0.0:
        log(f"  NOTE: lr_dual=0 -> Lagrange multipliers are frozen at "
            f"mu_init={cfg.mu_init} (fixed penalty, not adaptive dual ascent).")

    # Apply speed settings per-run so transformer checkpointing is toggled
    # correctly when drift_detach varies across configs in the same sweep.
    tune_sampler_speed(
        sampler, fast=fast, verbose=not quiet,
        control_type=cfg.control_type,
        drift_detach=cfg.drift_detach,
        # 2026-09-16: drift_detach alone no longer implies "no transformer
        # backward" -- endpoint_map(..., grad=True) is the sole gradient path
        # under constraint_on="endpoint" and runs either way. Passing
        # constraint_on lets tune_sampler_speed keep gradient checkpointing on
        # in exactly the case that needs it; without this, --fast true plus the
        # new drift_detach=True default would drop checkpointing while the
        # backward is live, which OOMs FLUX at H >= 2.
        constraint_on=cfg.constraint_on,
    )

    # (Re-)encode prompt for the sampler's cached embeddings
    if cfg.prompt != sampler.prompt_text:
        log(f"  re-encoding prompt: '{cfg.prompt[:50]}'")
        sampler.cached_prompt_embeds, sampler.cached_pooled_prompt_embeds = \
            sampler._encode_prompt_internal(cfg.prompt)
        sampler.prompt_text = cfg.prompt

    # Initial noise
    torch.manual_seed(cfg.seed)
    in_ch  = sampler.pipeline.transformer.config.in_channels // 4
    vscale = sampler.vae_scale_factor
    z0 = torch.randn(1, in_ch, cfg.height // vscale, cfg.width // vscale,
                     device=device, dtype=dtype)

    # ── Resolve constraint specs (Step 4.1) ──────────────────────────────────
    # _build_constraint_fn (below, after ref_holder exists) builds each
    # constraint's scorer via _build_reward_fn directly -- no dedicated
    # CLIP-only preload block needed anymore (that block existed only to feed
    # the old CLIP-specific _build_clip_constraint_fn).
    _constraint_specs = _resolve_constraint_specs(cfg)

    # Double-load notice, generalized beyond CLIP (was CLIP-only pre-Step-4).
    # _build_reward_fn takes no model arguments and every branch constructs
    # its own scorer (see its signature/docstring), so if the SAME reward
    # type is used as both --reward and a constraint, that checkpoint resides
    # TWICE simultaneously (once per _build_reward_fn call). Registry-based
    # sharing across reward-path and constraint-path scorers is still
    # deferred (the design plan's TODO, carried forward unresolved) -- this is
    # only the notice, not the fix. NOT quiet-gated (plain print, not log()):
    # memory-relevant, not routine progress. Costs zero on every currently
    # planned config, all of which run --reward none.
    _dup_types = sorted({s.reward_type for s in _constraint_specs if s.reward_type == reward_model.lower()})
    if _dup_types:
        print(
            f"  NOTICE: --reward {reward_model} together with a "
            f"{_dup_types[0]!r} constraint resides TWO independent copies of "
            f"that scorer (one for the reward closure, one for the "
            f"constraint closure -- see _build_reward_fn's docstring / plan "
            f"Step 4.3). Not fixed here: costs zero on every currently "
            f"planned run, all of which use --reward none."
        )

    # Reward function
    # ref_holder: empty for every reward_model except "lpips", where
    # _build_reward_fn's closure needs a reference image that doesn't exist
    # yet at this point (z_unc is produced by the uncontrolled rollout
    # below). Populated in place after that rollout decodes, before r_image
    # is ever called -- see the insertion point a few lines down.
    ref_holder: dict = {}
    r_image, _reward_obj = _build_reward_fn(
        sampler, cfg, reward_device, dtype, reward_model=reward_model,
        ref_holder=ref_holder,
        drop_text_tower=_drop_towers, reward_dtype=reward_dtype,
    )

    # Device preflight (added 2026-08-21): print and assert the ACTUAL tensor
    # devices, not the CLI args -- a prior audit found the device-print line
    # missing from exactly the runs that got corrupted. LOUD FAIL: raises
    # rather than silently continuing on a device that wasn't what was asked.
    _cur_dev = torch.cuda.current_device() if torch.cuda.is_available() else None
    log(f"  [device preflight] torch.cuda.current_device() = {_cur_dev}"
        f"   (ambient; NOT necessarily where the tensors are)")
    log(f"  [device preflight] FLUX latent (z0) device = {z0.device}")
    if _cur_dev is not None and z0.device.type == "cuda" and z0.device.index != _cur_dev:
        # NOT an assert: this mismatch is exactly the suspected state,
        # and we want the run to PROCEED so we can observe whether it
        # fails. Loud warning, not a hard stop.
        log(f"  [device preflight] !! AMBIENT-DEVICE MISMATCH: tensors on "
            f"cuda:{z0.device.index} but current_device()={_cur_dev}. Any kernel "
            f"reading the ambient device operates on the WRONG card. This is the "
            f"suspected cause of the reproducible ImageReward fault on cuda:1.")
    assert z0.device == device, (
        f"z0 landed on {z0.device}, expected {device} (--flux_device)"
    )
    # reward_model="none" loads no model at all (reward_obj is None) and its
    # r_image returns z.new_zeros(...), which is on z's device by construction,
    # so there is no parameter device to assert. Still logged explicitly rather
    # than skipped silently -- the whole point of this preflight is that the
    # device line was missing from exactly the runs that got corrupted.
    if _reward_obj is None:
        log(f"  [device preflight] reward model = 'none' (no model loaded; "
            f"r_image returns zeros on the latent's own device)")
        assert reward_model.lower() == "none", (
            f"reward_obj is None but reward_model={reward_model!r}; only "
            f"'none' is allowed to produce a null reward object"
        )
    else:
        _reward_param_device = next(_reward_obj.parameters()).device
        log(f"  [device preflight] reward model parameter device = {_reward_param_device}"
            f"   (current_device()={_cur_dev})")
        assert _reward_param_device == torch.device(reward_device), (
            f"reward model parameter landed on {_reward_param_device}, expected {reward_device} (--reward_device)"
        )

    # ── Constraint function h(z) -> (B, C) ───────────────────────────────────
    h_fn = None
    if n_constraints:
        h_fn = _build_constraint_fn(
            sampler, cfg, _constraint_specs, reward_device, dtype, ref_holder,
            drop_text_tower=_drop_towers, reward_dtype=reward_dtype,
        )
        log(f"  constraint: {n_constraints} constraint(s)  "
            f"{[(s.reward_type, s.arg, s.direction, s.threshold, s.normalize) for s in _constraint_specs]}  "
            f"lr_dual={cfg.lr_dual}  mu_init={cfg.mu_init}  "
            f"constraint_on={cfg.constraint_on}")

    def eval_r(z):
        with torch.no_grad():
            return r_image(z.to(device=device, dtype=dtype))

    # Uncontrolled baseline (always run, used as reference regardless of control_type)
    log("  uncontrolled rollout...")
    t0 = time.perf_counter()
    z_unc, traj_unc = _run_uncontrolled(sampler, z0, cfg, device, dtype, log=log)
    time_unc = time.perf_counter() - t0
    img_unc = _decode_pil(sampler, z_unc)
    # ref_holder["x_ref"] must be populated whenever LPIPS scores ANYTHING in
    # this run -- either the reward (reward_model == "lpips") or ANY
    # constraint (a ConstraintSpec with reward_type == "lpips", built above
    # via _build_constraint_fn using this SAME ref_holder dict). Checking
    # reward_model alone (as this block did before Step 4 generalized
    # constraints beyond CLIP) misses the constraint case entirely: an
    # LPIPS constraint's r_image closure would then hit the "called before
    # ref_holder['x_ref'] was populated" RuntimeError on every run using it
    # -- e.g. the planned Step 8 usage (--reward none, LPIPS as a
    # constraint). The design plan explicitly asks for "an explicit assert that
    # ref_holder is populated before the first h evaluation"; populating it
    # correctly here is the fix that assert is checking for.
    _lpips_needs_ref = (reward_model == "lpips") or any(
        s.reward_type == "lpips" for s in _constraint_specs
    )
    if _lpips_needs_ref:
        # x_ref = the raw fp32 decode of z_unc, NOT img_unc -- img_unc has
        # already round-tripped through _decode_pil's uint8 PIL conversion.
        # decode_no_grad() just wraps decode() in a no_grad context (same
        # underlying VAE forward, fluxfm_sampler_reward.py:489-497), so this
        # and the decode eval_r(z_unc) performs immediately below are the
        # same deterministic computation on the same z_unc -- expect
        # r_unc_final == 0.0 exactly, by construction, not as a measurement,
        # WHEN reward_model == "lpips" specifically (an lpips CONSTRAINT
        # with a different reward_model has no such expectation).
        with torch.no_grad():
            ref_holder["x_ref"] = sampler.decode_no_grad(z_unc).float().to(reward_device).detach()
    assert not (h_fn is not None
                and any(s.reward_type == "lpips" for s in _constraint_specs)
                and "x_ref" not in ref_holder), (
        "an LPIPS constraint is active but ref_holder['x_ref'] was not "
        "populated before h_fn's first evaluation -- see the block above "
        "this assert; this should be structurally unreachable, so hitting "
        "it means the _lpips_needs_ref condition itself has a gap."
    )
    r_unc_final = eval_r(z_unc).mean().item()
    _free(z_unc)

    # Controlled run
    z_ctrl = None
    traj_ctrl = None
    v_theta = None
    endpoint_map = None
    time_ctrl = 0.0
    r_ctrl_final = float("nan")
    r_ctrl_independent = float("nan")   # ConicFlow only -- see the independent re-scoring block below
    mu_final = None      # (B, C) tensor of final multipliers, ConicFlow + constraints only
    mu_hist = None       # (N, B, C) tensor of per-outer-step multipliers
    h_dual_hist = None   # (N, B, C) the h each dual update actually consumed (conicflow and
                         # hardflow as of 2026-09-08; fmrg has no duals. Formerly ConicFlow-only;
                         # None elsewhere). Recorded because the reconstruction below
                         # evaluates h at the REALIZED state while ConicFlow's dual reads it at
                         # the optimizer's horizon endpoint, H steps ahead -- so the two
                         # disagree and mu appears to fall while h is still positive.
    boffi_img_tensor = None   # only set for control_type == "boffi_fmrg" (decoded pixel image, no latent)

    # Diagnostic only; None keeps hardflow_image's hook inert (see _run_one's
    # docstring). A list is passed ONLY when the flag is on, so the default
    # path allocates nothing and behaves exactly as before.
    _dual_probe = [] if (probe_hardflow_duals and cfg.control_type == "hardflow") else None
    if probe_hardflow_duals and cfg.control_type != "hardflow":
        raise ValueError(
            f"--probe_hardflow_duals requires --sweep_control_type hardflow, "
            f"got {cfg.control_type!r}. Refusing to run a probe that would "
            f"silently record nothing."
        )

    # Bound before the guard, not inside it: control_type="none" skips the whole
    # control block but _run_one still writes hardflow_lr_eff into its result
    # dict, so leaving it to the hardflow path made every "none" run die with
    # UnboundLocalError. The constrained experiment's threshold probe is a
    # control_type="none" run, so this is load-bearing there.
    hardflow_lr_eff = ""          # filled on the hardflow path; blank elsewhere

    if cfg.control_type != "none":
        v_theta, _, endpoint_map = make_flux_wrappers(
            sampler, cfg.prompt,
            guidance_scale=cfg.guidance_scale,
            n_conicflow_steps=cfg.N,
            sample_mode=cfg.sample_mode,
            endpoint_surrogate=cfg.endpoint_surrogate,
        )

        log(f"  {cfg.control_type} run...")
        t0 = time.perf_counter()

        if cfg.control_type == "conicflow":
            z_ctrl, traj_ctrl, _, mu_final, mu_hist, h_dual_hist = receding_horizon_flow_control(
                v_theta=v_theta,
                r=r_image,
                x0=z0,
                N=cfg.N,
                H=cfg.H,
                lam=cfg.lam,
                optimizer="adam",
                n_iters=cfg.n_inner,
                lr=cfg.lr_inner,
                warmstart=cfg.warmstart,
                n_alt=cfg.n_alt,
                skip_first_primal_step=cfg.skip_first_primal_step,
                dual_adaptive=cfg.dual_adaptive,
                drift_detach=cfg.drift_detach,
                h=h_fn,
                mu_init=cfg.mu_init,
                lr_dual=cfg.lr_dual,
                constraint_on=cfg.constraint_on,
                endpoint_map=endpoint_map,
                verbose=not quiet,
                # The bar itself is nearly free; the reward it displays is not
                # (see pbar_rewards' docstring). Off unless --pbar_rewards.
                show_progress=not quiet,
                pbar_rewards=pbar_rewards,
                low_memory=low_memory,
                device=device,
                return_dual_h=True,
            )
        elif cfg.control_type == "hardflow":
            # Effective inner lr: see RunConfig.hardflow_lr_alpha. Below the
            # lr*reg >= 1 floor the configured lr is used unchanged, so rows
            # above the floor are bit-for-bit what they were before this flag.
            hardflow_lr_eff = cfg.lr_inner
            if (cfg.hardflow_lr_alpha is not None and cfg.hardflow_reg
                    and cfg.lr_inner * cfg.hardflow_reg >= 1.0):
                hardflow_lr_eff = cfg.hardflow_lr_alpha / cfg.hardflow_reg
                log(f"  hardflow: lr_inner={cfg.lr_inner:g} * reg={cfg.hardflow_reg:g} "
                    f">= 1 -> using lr={hardflow_lr_eff:.4g} (= hardflow_lr_alpha "
                    f"{cfg.hardflow_lr_alpha:g} / reg)")
            # HardFlow (Li, Alim & Azizan, TPAMI 2026), ported in
            # control_loops.hardflow_image. Same return contract as
            # receding_horizon_flow_control, so mu_final/mu_hist feed the
            # existing constraint bookkeeping (mu_trace CSV, constraint plots)
            # with no changes. Note it takes `sampler` rather than v_theta:
            # HardFlow needs the two-timestep flow-map call directly, since its
            # terminal estimate must be a single NFE (that is the method's whole
            # efficiency claim) rather than endpoint_map's rollout.
            z_ctrl, traj_ctrl, _, mu_final, mu_hist, h_dual_hist = hardflow_image(
                sampler, r_image, z0,
                N=cfg.N,
                n_inner=cfg.n_inner,
                lr=hardflow_lr_eff,
                h=h_fn,
                mu_init=cfg.mu_init,
                lr_dual=cfg.lr_dual if cfg.lr_dual > 0.0 else None,
                rho=cfg.hardflow_rho,
                rho_mult=cfg.hardflow_rho_mult,
                dual_every=cfg.hardflow_dual_every,
                constraint_margin=cfg.hardflow_margin if h_fn is not None else 0.0,
                activation_frac=cfg.hardflow_activation,
                hardflow_reg=cfg.hardflow_reg,
                guidance_scale=cfg.guidance_scale,
                sample_mode=cfg.sample_mode,
                endpoint_mode=cfg.hardflow_endpoint_mode,
                # hardflow_image ignores endpoint_mode when endpoint_map is
                # given, so the mode string is the switch and only one of the
                # two is ever in effect.
                endpoint_map=(endpoint_map
                              if cfg.hardflow_endpoint_mode == "conicflow_rollout"
                              else None),
                # Records h at HardFlow's own decision variable x_hat, so the
                # constraint panel has a comparable series for this arm
                # (2026-09-08). Costs nothing -- the value is already computed
                # by the inner loop. Margin is removed before recording; see
                # hardflow_image.
                return_dual_h=True,
                verbose=not quiet,
                low_memory=low_memory,
                dual_probe=_dual_probe,
            )
        elif cfg.control_type == "fmrg":
            z_ctrl, traj_ctrl, _ = fmrg_j_image(
                sampler, r_image, z0,
                N=cfg.N,
                n_opt=cfg.n_inner,
                lam=cfg.lam,
                guidance_scale=cfg.guidance_scale,
                sample_mode=cfg.sample_mode,
                return_traj=True,
                return_controls=True,
                verbose=not quiet,
                low_memory=low_memory,
            )
        elif cfg.control_type == "boffi_fmrg":
            # Upstream (Boffi et al.) FMRG-J, vendored in fmrg/fluxfm_sampler_reward.py.
            # sample_reward_guided() itself only returns a decoded pixel image (no
            # latent/trajectory) -- _sample_boffi_fmrg_with_trajectory reconstructs the
            # latent trajectory via a temporary, non-invasive monkey-patch so control
            # energy can be computed in the same convention as the conicflow/fmrg paths.
            boffi_reward = _build_boffi_reward_ensemble(reward_device, reward_model)
            boffi_img_tensor, boffi_traj, boffi_dts, boffi_t_curs = _sample_boffi_fmrg_with_trajectory(
                sampler, boffi_reward, z0, cfg, device,
            )
            del boffi_reward

        time_ctrl = time.perf_counter() - t0

        if cfg.control_type == "boffi_fmrg":
            if reward_model.lower() != "clip":
                raise NotImplementedError(
                    "boffi_fmrg final-reward scoring currently only supports --reward clip "
                    "(see _clip_score_pixel)."
                )
            score_prompt = cfg.edit_prompt if cfg.edit_prompt else cfg.prompt
            with torch.no_grad():
                img_01 = ((boffi_img_tensor + 1.0) / 2.0).clamp(0, 1).float()
            r_ctrl_final = _clip_score_pixel(
                _reward_obj, create_clip_preprocess(224), img_01, score_prompt, reward_device,
            ).mean().item()
            img_ctrl = sampler.pipeline.image_processor.postprocess(
                boffi_img_tensor.float().detach().cpu(), output_type="pil")[0]
        else:
            r_ctrl_final = eval_r(z_ctrl).mean().item()
            # Independent re-scoring (added 2026-08-21, Phase B): ConicFlow shares
            # one r_image closure between the optimization gradient and this
            # logging call -- a clean-looking log is not by itself evidence
            # the optimized objective was clean (Category A of the
            # contamination audit corrupted exactly this shared closure).
            # This builds a SEPARATE CLIPModel + tokenizer + text embedding
            # from scratch, sharing no object with _build_reward_fn's clip
            # branch, and re-decodes z_ctrl fresh -- a genuinely independent
            # code path, not a second call into the same closure.
            r_ctrl_independent = float("nan")
            if reward_model.lower() == "clip":
                r_ctrl_independent = _independent_clip_rescore(
                    sampler, z_ctrl,
                    cfg.edit_prompt if cfg.edit_prompt else cfg.prompt,
                    reward_device,
                )
            img_ctrl = _decode_pil(sampler, z_ctrl)
            _free(z_ctrl)
    else:
        img_ctrl = img_unc      # placeholder for side-by-side

    _free(z0)

    # Reward trajectory curves
    r_unc_curve  = trajectory_rewards(traj_unc, r_image, device=device, dtype=dtype).float().numpy()
    r_ctrl_curve = (
        trajectory_rewards(traj_ctrl, r_image, device=device, dtype=dtype).float().numpy()
        if traj_ctrl is not None else r_unc_curve
    )
    t_grid = torch.linspace(0.0, 1.0, len(r_unc_curve)).numpy()
    _free(traj_unc)

    # Per-step control norms: u_t = x_{t+1} - x_t - dt·b_t(x_t)
    # Computed from the trajectory for both ConicFlow and FMRG so scales match.
    ctrl_norms  = None
    ctrl_energy = float("nan")
    ctrl_dts    = None   # per-step dt, needed to split ctrl_energy across steps
    if traj_ctrl is not None and v_theta is not None:
        log("  computing control norms from trajectory...")
        ctrl_norms, ctrl_energy = _ctrl_norms_from_traj(
            traj_ctrl, v_theta, cfg.N, device, dtype,
        )
        ctrl_dts = [1.0 / cfg.N] * cfg.N
    elif cfg.control_type == "boffi_fmrg" and v_theta is not None:
        log("  computing control norms from reconstructed boffi_fmrg trajectory...")
        # Persistent per-step dump for every run (Phase A, 2026-08-21): the
        # previous ctrl_energy=nan corruption couldn't be diagnosed offline
        # because nothing like this existed. Written on both success and
        # failure (see _ctrl_norms_from_traj_nonuniform).
        _dump_tag = (f"{cfg.control_type}__N{cfg.N}__H{cfg.H}__lam{cfg.lam}"
                     f"__{cfg.sample_mode}__dd{cfg.drift_detach}"
                     f"__{cfg.prompt[:30].replace(' ', '_')}")
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        _dump_path = str(Path(cfg.out_dir) / f"{_dump_tag}__traj_dump.npz")
        ctrl_norms, ctrl_energy = _ctrl_norms_from_traj_nonuniform(
            boffi_traj, v_theta, boffi_dts, boffi_t_curs, device, dtype,
            dump_path=_dump_path,
        )
        # boffi_fmrg lives on FLUX's mu-shifted (non-uniform) schedule, so the
        # trace must use the ACTUAL per-step dts, not 1/N.
        ctrl_dts = list(boffi_dts)
        _free(boffi_traj)
    # Constraint values along the controlled trajectory: h_i(z_t) for each step t.
    # Shape: (N+1, C) -- mean over batch (B=1 here so trivial).
    #
    # The LAST row is h on the final image, which is what Step 4.4's feasibility
    # check needs: at t=N the trajectory state IS the terminal latent, so
    # evaluating h_fn on it directly is correct even under
    # constraint_on="endpoint" (where h during optimization is evaluated at
    # endpoint_map(t_H, x_H) rather than at the raw horizon state).
    # TWO conventions, and they are not interchangeable:
    #
    #   constraint_traj_raw  -- h(z_t) at the realized latent itself.
    #   constraint_traj      -- h in the SAME convention as cfg.constraint_on.
    #                           Under "endpoint" (the locked default) that is
    #                           h(endpoint_map(s_t, z_t)), i.e. h at the
    #                           flow-map-predicted TERMINAL latent, which is the
    #                           quantity the dual update actually responds to
    #                           (image_conicflow.py's own progress bar uses exactly
    #                           this, ~L501-503). Under "horizon" the two coincide.
    #
    # This matters for Step 10.4: mu_hist is driven by the endpoint-convention
    # violation, so pairing mu against the RAW series would make "mu rises but
    # the violation does not shrink" (mode 1) and "mu stable, still violated"
    # (mode 2) ambiguous for a reason that is an artifact of mismatched
    # conventions rather than anything about the run.
    #
    # Caveat that must not be glossed: the dual step at outer step j evaluates h
    # at the OPTIMIZER's horizon endpoint xH_star, which receding_horizon_flow_control
    # does not return. What is reconstructed here is h at the endpoint predicted
    # from the REALIZED state after step j. Same convention, same units, and it is
    # the honest post-hoc proxy -- but it is not literally the number the dual
    # step consumed. Recovering that exactly would require changing image_conicflow.py,
    # which is out of scope (locked: "image_conicflow.py -- no change needed").
    #
    # Cost: the endpoint series adds ~N(N-1)/2 transformer forwards over the run
    # (endpoint_map rolls N-j-1 steps at outer step j). Deliberately NOT behind a
    # flag: a diagnostic that can be switched off is switched off for exactly the
    # run whose failure needed diagnosing, and Step 10 cannot be re-run cheaply.
    constraint_traj = None       # numpy (N+1, C), constraint_on convention
    constraint_traj_raw = None   # numpy (N+1, C), raw realized latent
    h_final = None               # numpy (C,) -- h on the final image
    _h_probe_result = None       # diagnostic only; see probe_h_repeat
    run_ok_stale = False         # set by the stale-decode guard below
    if traj_ctrl is not None and h_fn is not None:
        import numpy as np
        log("  computing constraint trajectory (raw + %s convention)..." % cfg.constraint_on)
        n_pts = traj_ctrl.shape[0]
        s_grid = torch.linspace(0.0, 1.0, n_pts)
        raw_vals, eval_vals = [], []
        use_endpoint = (cfg.constraint_on == "endpoint" and endpoint_map is not None)
        with torch.no_grad():
            for t_idx in range(n_pts):
                zt = traj_ctrl[t_idx].to(device=device, dtype=dtype)
                raw_vals.append(h_fn(zt).mean(0).cpu())          # (C,)
                if use_endpoint:
                    s_t = s_grid[t_idx].to(device).expand(zt.shape[0])
                    eval_vals.append(h_fn(endpoint_map(s_t, zt)).mean(0).cpu())
        constraint_traj_raw = torch.stack(raw_vals, dim=0).float().numpy()
        constraint_traj = (
            torch.stack(eval_vals, dim=0).float().numpy() if use_endpoint
            else constraint_traj_raw
        )
        # h on the final image. endpoint_map at s=1.0 is exactly the identity
        # (t_start = 0 -> dt_r = 0 -> z_run = z), so the two conventions MUST
        # agree in the last row. Cross-checked rather than assumed: this single
        # comparison validates the whole step<->row alignment the trace depends
        # on, and a mismatch means the alignment is wrong, not that the run is.
        h_final = constraint_traj[-1]

        # ── h-determinism probe (diagnostic, default-off) ───────────────────
        # Answers: is the sporadic endpoint/raw gap below caused by h_fn being
        # nondeterministic, or by endpoint_map not actually being the identity
        # at s=1? Those are the only two possibilities, and they have opposite
        # consequences -- a nondeterministic h_fn taints every reported h and
        # every dual update, while a non-identity endpoint_map taints only the
        # final-row cross-check. Recorded, not inferred.
        if probe_h_repeat > 0 and use_endpoint:
            zt = traj_ctrl[-1].to(device=device, dtype=dtype) if traj_ctrl is not None else None
            _probe = {"repeats": int(probe_h_repeat), "constraint_names": _constraint_names(cfg)}
            with torch.no_grad():
                s_end = s_grid[-1].to(device).expand(zt.shape[0])
                z_ep = endpoint_map(s_end, zt)
                # Is endpoint_map(1, z) the identity? Bitwise, then in norm.
                _probe["endpoint_is_bitwise_identity"] = bool(torch.equal(z_ep, zt))
                _probe["endpoint_max_abs_diff"] = float((z_ep - zt).abs().max())
                _probe["endpoint_dtype"] = str(z_ep.dtype)
                # Repeat h on the SAME tensor, and on the endpoint tensor.
                _probe["h_on_raw"] = [h_fn(zt).mean(0).cpu().tolist()
                                      for _ in range(probe_h_repeat)]
                _probe["h_on_endpoint"] = [h_fn(z_ep).mean(0).cpu().tolist()
                                           for _ in range(probe_h_repeat)]
                # And on a fresh clone -- same values, different allocation,
                # which is what endpoint_map returns (z.detach().clone()).
                _probe["h_on_clone"] = [h_fn(zt.clone()).mean(0).cpu().tolist()
                                        for _ in range(probe_h_repeat)]
            _probe["h_raw_recorded"] = constraint_traj_raw[-1].tolist()
            _probe["h_endpoint_recorded"] = constraint_traj[-1].tolist()
            _h_probe_result = _probe
            print(f"  [h-probe] endpoint_map(1,z) bitwise-identical to z: "
                  f"{_probe['endpoint_is_bitwise_identity']}  "
                  f"max|diff|={_probe['endpoint_max_abs_diff']:.3e}")
            for _k in ("h_on_raw", "h_on_endpoint", "h_on_clone"):
                _arr = np.asarray(_probe[_k], dtype=float)
                print(f"  [h-probe] {_k:14s} spread over {probe_h_repeat} calls "
                      f"(max-min per constraint): {(_arr.max(0) - _arr.min(0)).tolist()}")

        # ── stale-decode guard (2026-09-07) ─────────────────────────────────
        # Catches the signature of the id()-keyed decode-cache bug fixed in
        # _build_constraint_fn: a latent that reused a freed address got the
        # PREVIOUS z's decoded image, which shows up as raw[k] being bit-equal
        # to endpoint[k-1] in EVERY constraint channel at once. Those two are
        # different conventions evaluated at different states, so exact
        # equality across all channels cannot happen by chance -- one hit is
        # proof, not a threshold to tune. The fix is structural, but this stays
        # because the failure was silent for a whole grid: 28 of 240 step
        # evaluations in 20260904_231018_lpr2c were corrupted and only the 3
        # that landed on the final row were ever flagged.
        if use_endpoint and len(constraint_traj) > 1:
            _stale = [k for k in range(1, len(constraint_traj))
                      if np.array_equal(constraint_traj_raw[k], constraint_traj[k - 1])]
            if _stale:
                run_ok_stale = True
                print(
                    f"  WARNING: stale-decode signature at step(s) {_stale}: "
                    f"h_raw[k] is bit-equal to h_endpoint[k-1] in every "
                    f"constraint channel. That is the id()-keyed decode-cache "
                    f"collision (fixed 2026-09-07 in _build_constraint_fn); if "
                    f"it appears again the cache has regressed. "
                    f"prompt={cfg.prompt!r} control={cfg.control_type}"
                )
            else:
                run_ok_stale = False
        else:
            run_ok_stale = False

        if use_endpoint:
            _end_gap = float(np.abs(constraint_traj[-1] - constraint_traj_raw[-1]).max())
            if not (_end_gap <= 1e-4):
                print(
                    f"  WARNING: endpoint/raw h disagree on the FINAL image by "
                    f"{_end_gap:.3e} (expected 0 -- endpoint_map is the identity "
                    f"at s=1). The per-step trace's step<->row alignment is "
                    f"therefore suspect for prompt={cfg.prompt!r} "
                    f"control={cfg.control_type} N={cfg.N} H={cfg.H}"
                )
                _endpoint_align_ok = False
            else:
                _endpoint_align_ok = True
        else:
            _endpoint_align_ok = True
    else:
        _endpoint_align_ok = True

    if traj_ctrl is not None:
        _free(traj_ctrl)

    # Applied control magnitude, mean/max over the horizon's per-step ‖u_t‖
    # (ctrl_norms itself, from _ctrl_norms_from_traj[_nonuniform] above -- the
    # trajectory-derived u_t = x_{t+1}-x_t-dt*b_t(x_t), never the progress-bar's
    # "|u|" field: fmrg_j_image's is the pre-lambda reward gradient, not a
    # control at all, and ConicFlow's inner/outer pbar values are a live, per-inner-
    # iteration preview of the same quantity computed here, not a substitute
    # for it. See EVIDENCE_REPORT.md's drift_detach-field investigation for why
    # these are named separately instead of repurposing the pbar field.
    ctrl_norm_mean = float(ctrl_norms.mean()) if ctrl_norms is not None else float("nan")
    ctrl_norm_max  = float(ctrl_norms.max())  if ctrl_norms is not None else float("nan")

    # Lagrangian objective J = control_energy / (2*lam) - r_ctrl, matching the
    # sign/scaling of the inner-loop loss in _horizon_objective (image_conicflow.py).
    # Lower J is better; this is what makes ConicFlow vs. FMRG comparable at a fixed lam.
    objective = float("nan")
    if cfg.control_type != "none":
        objective = ctrl_energy / (2.0 * cfg.lam) - r_ctrl_final

    # Free reward model weights after use
    if hasattr(_reward_obj, "cpu"):
        _reward_obj.cpu()
    _free(_reward_obj)

    # ── NaN/inf detection ──────────────────────────────────────────────────
    # Not gated by --quiet: a non-finite run is a data-integrity issue, not
    # routine progress logging, and must not be silently buried in the CSV.
    # r_ctrl/ctrl_energy are legitimately nan-by-design when control_type=="none"
    # (no controlled rollout exists) -- only check them when a control ran.
    # 2026-09-05: four independent guards below can set run_ok=False, and until
    # now the CSV recorded only the boolean -- so 68 of the 70 failed rows on
    # disk say nothing about WHICH guard fired, and an OOM was indistinguishable
    # from a NaN. Each guard now appends its name here, and the list lands in
    # the error_type / error_msg columns.
    _fail_reasons: list = []

    _check_vals = [r_unc_final]
    if cfg.control_type != "none":
        _check_vals += [r_ctrl_final, ctrl_energy]
    run_ok = all(math.isfinite(v) for v in _check_vals)
    if not run_ok:
        _fail_reasons.append(
            f"non_finite(r_unc={r_unc_final}, r_ctrl={r_ctrl_final}, "
            f"ctrl_energy={ctrl_energy})"
        )
        print(
            f"  WARNING: non-finite result for prompt={cfg.prompt!r} "
            f"control={cfg.control_type} H={cfg.H} lam={cfg.lam} "
            f"drift_detach={cfg.drift_detach}: "
            f"r_unc={r_unc_final} r_ctrl={r_ctrl_final} ctrl_energy={ctrl_energy}"
        )

    # ── plausibility guard ──────────────────────────────────────────────────
    # isfinite() alone doesn't catch a wrong-but-finite value (e.g. a reward
    # computed under the wrong convention/offset) -- Category A of the
    # 2026-08-21 contamination audit. Fails loudly (run_ok=False), does not
    # clamp/substitute/retry -- same contract as the isfinite check above.
    _lo, _hi = _REWARD_PLAUSIBLE_RANGE[reward_model.lower()]
    _plaus_vals = [("r_unc", r_unc_final)]
    if cfg.control_type != "none":
        _plaus_vals.append(("r_ctrl", r_ctrl_final))
    for _name, _val in _plaus_vals:
        if math.isfinite(_val) and not (_lo - 1e-6 <= _val <= _hi + 1e-6):
            run_ok = False
            _fail_reasons.append(
                f"implausible_{_name}({_val:.6g} outside [{_lo:.4g}, {_hi:.4g}] "
                f"for reward={reward_model})"
            )
            print(
                f"  WARNING: implausible {_name}={_val} for reward_model={reward_model!r} "
                f"(expected range [{_lo:.4g}, {_hi:.4g}]) -- prompt={cfg.prompt!r} "
                f"control={cfg.control_type} H={cfg.H} lam={cfg.lam} "
                f"drift_detach={cfg.drift_detach}"
            )

    # ── constraint-trace alignment guard ───────────────────────────────────
    # Set by the endpoint-vs-raw cross-check above. A mismatch means the
    # per-step trace's step<->row mapping cannot be trusted, which is exactly
    # the thing Step 10.4 relies on -- so the run is marked not-ok rather than
    # shipping a trace that silently mis-attributes mu to the wrong step.
    if not _endpoint_align_ok:
        run_ok = False
        _fail_reasons.append("constraint_trace_misaligned")

    # Stale-decode signature (see the guard in the constraint-trajectory block).
    if run_ok_stale:
        run_ok = False
        _fail_reasons.append("stale_decode_cache_signature")

    # ── baseline drift guard ───────────────────────────────────────────────
    # r_unc should be identical for every run sharing (prompt, seed) within
    # this process (same seed -> same z0 -> same uncontrolled rollout). If it
    # drifts, something shared (sampler/transformer/vae/cached embeddings) got
    # corrupted by an earlier run -- this is the exact silent-corruption
    # pattern observed once and never explained; flag it loudly if it recurs.
    baseline_key = (cfg.prompt, cfg.seed)
    if math.isfinite(r_unc_final):
        if baseline_key not in _R_UNC_BASELINE:
            _R_UNC_BASELINE[baseline_key] = r_unc_final
        else:
            baseline = _R_UNC_BASELINE[baseline_key]
            if not math.isclose(r_unc_final, baseline, rel_tol=R_UNC_DRIFT_RTOL, abs_tol=R_UNC_DRIFT_ATOL):
                print(
                    f"  WARNING: r_unc drifted for prompt={cfg.prompt!r} seed={cfg.seed}: "
                    f"baseline={baseline!r} this_run={r_unc_final!r} "
                    f"(control={cfg.control_type} H={cfg.H} lam={cfg.lam} "
                    f"drift_detach={cfg.drift_detach}) -- shared sampler state may be corrupted"
                )
                run_ok = False
                _fail_reasons.append(
                    f"r_unc_baseline_drift(baseline={baseline!r} this_run={r_unc_final!r})"
                )

    # ── per-constraint CSV cells ────────────────────────────────────────────
    # Fixed positional columns (thr_i / h_i / mu_i for i in 0..MAX_CONSTRAINTS-1)
    # rather than comma-joined strings: these are read back as floats on every
    # analysis pass, and a string round-trip both costs a split() each time and
    # loses precision. Unused slots are left as "" -- NEVER 0 or NaN, which
    # would be indistinguishable from a measured value.
    #
    # `constraint_names` carries the ordering. It is mandatory: the column index
    # alone does not say which reward h_0 is, and the Task 3 pair may change
    # (style+ImageReward vs. the CLIP+style fallback), at which point h_0 would
    # silently change meaning -- the same defect that forced the reward model to
    # be inferred from directory names for 222 of 232 historical run dirs.
    _mu_final_np = (
        mu_final.float().mean(0).cpu().numpy() if mu_final is not None else None
    )
    # Step 4.4's remaining piece: per-constraint satisfied yes/no + violation
    # magnitude on the FINAL image, derived from h_final (already in the
    # constraint_on convention -- see _write_constraint_trace/§1.4b). h_i <= 0
    # means satisfied by this file's convention throughout
    # (_build_constraint_fn / stack_constraints), so sat/viol are a direct
    # read of h_final, not a new computation -- nothing here can disagree
    # with the h_i column for the same row.
    constraint_cells = {}
    for i in range(MAX_CONSTRAINTS):
        _h_i = float(h_final[i]) if h_final is not None and i < len(h_final) else None
        constraint_cells[f"thr_{i}"] = (
            _constraint_specs[i].threshold if i < len(_constraint_specs) else ""
        )
        constraint_cells[f"h_{i}"] = _h_i if _h_i is not None else ""
        constraint_cells[f"mu_{i}"] = (
            float(_mu_final_np[i]) if _mu_final_np is not None and i < len(_mu_final_np) else ""
        )
        constraint_cells[f"sat_{i}"] = (_h_i <= 0.0) if _h_i is not None else ""
        constraint_cells[f"viol_{i}"] = max(_h_i, 0.0) if _h_i is not None else ""

    # Peak memory for this run (the design plan). Read at the very end so it
    # covers the whole run -- uncontrolled baseline, the ConicFlow loop with its
    # horizon graphs and Adam state, every constraint decode/score, and the
    # independent re-scoring pass. Empty (not 0.0) on CPU-only runs, so a
    # missing measurement is never mistaken for "used no memory".
    _mem_cells = {"peak_alloc_flux_gb": "", "peak_reserved_flux_gb": "",
                  "peak_alloc_reward_gb": ""}
    if "flux" in _mem_devices:
        _fd = _mem_devices["flux"]
        _mem_cells["peak_alloc_flux_gb"] = torch.cuda.max_memory_allocated(_fd) / 1e9
        _mem_cells["peak_reserved_flux_gb"] = torch.cuda.max_memory_reserved(_fd) / 1e9
    if "reward" in _mem_devices and _mem_devices["reward"] != _mem_devices.get("flux"):
        _mem_cells["peak_alloc_reward_gb"] = (
            torch.cuda.max_memory_allocated(_mem_devices["reward"]) / 1e9
        )
    if _mem_cells["peak_alloc_flux_gb"] != "":
        _extra = ""
        if _mem_cells["peak_alloc_reward_gb"] != "":
            _extra = (f"  reward({_mem_devices['reward']}) "
                      f"alloc={_mem_cells['peak_alloc_reward_gb']:.2f} GB")
        log(f"  peak GPU mem: alloc={_mem_cells['peak_alloc_flux_gb']:.2f} GB  "
            f"reserved={_mem_cells['peak_reserved_flux_gb']:.2f} GB "
            f"on {_mem_devices['flux']}{_extra}")

    return dict(
        run_ok        = run_ok,
        # Empty on a clean run, so a populated error_type always means a guard
        # fired. Multiple guards can fire at once; all of them are recorded.
        error_type    = ";".join(r.split("(")[0] for r in _fail_reasons),
        error_msg     = "; ".join(_fail_reasons)[:500],
        **_mem_cells,
        # `reward` and `seed` were missing entirely before the 2026-08-25 merge.
        # Their absence is why 222 of 232 historical run dirs needed the reward
        # model inferred from directory names and value ranges; every step after
        # this one depends on the CSV being self-describing.
        reward        = reward_model,
        seed          = cfg.seed,
        prompt        = cfg.prompt,
        edit_prompt   = cfg.edit_prompt if cfg.edit_prompt else cfg.prompt,
        N             = cfg.N,
        H             = cfg.H,
        control_type  = cfg.control_type,
        lam           = cfg.lam,
        n_inner       = cfg.n_inner,
        lr_inner      = cfg.lr_inner,
        sample_mode   = cfg.sample_mode,
        warmstart     = cfg.warmstart,
        n_alt         = cfg.n_alt,
        skip_first_primal_step = cfg.skip_first_primal_step,
        dual_adaptive = cfg.dual_adaptive,
        drift_detach  = cfg.drift_detach,
        constraint_on = cfg.constraint_on,
        endpoint_surrogate = cfg.endpoint_surrogate,
        # Only the locks that were BOTH lifted and actually diverge from the
        # locked value -- an unlock that changed nothing is not worth labelling
        # a row with, and would make the column useless for filtering.
        unlocked      = ",".join(sorted(
            k for k in _unlocked_fields()
            if getattr(cfg, k) != _LOCKED_DEFAULTS[k])),
        n_constraints = n_constraints,
        constraint_names = ",".join(_constraint_names(cfg)),
        lr_dual       = cfg.lr_dual,
        mu_init       = cfg.mu_init,
        **constraint_cells,
        boffi_use_adam = cfg.boffi_use_adam,
        hardflow_rho          = cfg.hardflow_rho,
        hardflow_rho_mult     = cfg.hardflow_rho_mult,
        hardflow_dual_every   = cfg.hardflow_dual_every,
        hardflow_margin       = cfg.hardflow_margin,
        hardflow_activation   = cfg.hardflow_activation,
        hardflow_reg          = cfg.hardflow_reg,
        hardflow_endpoint_mode = cfg.hardflow_endpoint_mode,
        hardflow_lr_alpha     = "" if cfg.hardflow_lr_alpha is None else cfg.hardflow_lr_alpha,
        hardflow_lr_eff       = hardflow_lr_eff,
        lpips_net     = cfg.lpips_net,
        r_unc         = r_unc_final,
        r_ctrl        = r_ctrl_final,
        r_ctrl_independent = r_ctrl_independent,
        delta_r       = r_ctrl_final - r_unc_final,
        ctrl_energy   = ctrl_energy,
        ctrl_norm_mean = ctrl_norm_mean,
        ctrl_norm_max  = ctrl_norm_max,
        objective     = objective,
        time_unc_s    = time_unc,
        time_ctrl_s   = time_ctrl,
        # image objects (not CSV-serialisable, stripped before writing)
        _img_unc   = img_unc,
        _img_ctrl  = img_ctrl,
        _t_grid    = t_grid,
        _r_unc_c   = r_unc_curve,
        _r_ctrl_c  = r_ctrl_curve,
        _ctrl_norms = ctrl_norms,
        # constraint diagnostics (None when no constraints were active)
        _ctrl_dts  = ctrl_dts,
        _constraint_traj = constraint_traj,   # numpy (N+1, C), constraint_on convention
        _constraint_traj_raw = constraint_traj_raw,   # numpy (N+1, C), raw latent
        _h_probe = _h_probe_result,           # diagnostic only, None unless --probe_h_repeat
        _dual_probe = _dual_probe,            # diagnostic only, None unless --probe_hardflow_duals
        _mu_hist         = mu_hist,           # tensor (N, B, C) or None
        _h_dual_hist     = h_dual_hist,       # tensor (N, B, C) or None -- conicflow + hardflow
        _constraint_specs      = list(_constraint_specs),
    )


def _save_run(result: dict, out_dir: Path, run_idx: int) -> str:
    """Write side-by-side image + reward curve for one run. Returns cfg_tag
    (the filename prefix) so callers can record it for later image lookup."""
    cfg_tag = (
        f"{run_idx:03d}"
        f"__{result['control_type']}"
        f"__N{result['N']}"
        f"__H{result['H']}"
        f"__lam{result['lam']}"
        f"__{result['sample_mode']}"
        f"__dd{result['drift_detach']}"
        f"__{result['prompt'][:30].replace(' ', '_')}"
    )
    # Mark non-finite runs in the filename itself so their artifacts are kept
    # (never deleted) but are unmistakable from valid runs when browsing the
    # output dir -- see run_ok / the baseline-drift guard in _run_one.
    if result.get("n_constraints", 0) > 0 and result.get("_constraint_specs"):
        _thr_str = _format_thresholds([s.threshold for s in result["_constraint_specs"]])
        cfg_tag += f"__thr{_thr_str.replace(',', '_')}"
    if not result.get("run_ok", True):
        cfg_tag = f"FAILED_{cfg_tag}"

    # Side-by-side images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(result["_img_unc"])
    axes[0].axis("off")
    axes[0].set_title(f"Uncontrolled  r={result['r_unc']:.4f}")
    axes[1].imshow(result["_img_ctrl"])
    axes[1].axis("off")
    ctrl_lbl = (
        f"{result['control_type']}  H={result['H']}  λ={result['lam']}"
        f"  r={result['r_ctrl']:.4f}"
    ) if result["control_type"] != "none" else "Uncontrolled (reference)"
    axes[1].set_title(ctrl_lbl)
    title = f"gen: {result['prompt'][:60]}"
    if result["edit_prompt"] != result["prompt"]:
        title += f"\nedit: {result['edit_prompt'][:60]}"
    plt.suptitle(title + f"\nN={result['N']}  {result['sample_mode']}", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / f"{cfg_tag}__images.png", dpi=100)
    plt.close(fig)

    # Reward curve + control norm (two panels when controls are available)
    has_ctrl = result["control_type"] != "none" and result["_ctrl_norms"] is not None
    n_panels = 2 if has_ctrl else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    ax_r = axes[0]
    ax_r.plot(result["_t_grid"], result["_r_unc_c"], "o-", label="Uncontrolled",
              linewidth=2, markersize=6)
    if result["control_type"] != "none":
        ax_r.plot(result["_t_grid"], result["_r_ctrl_c"], "s-",
                  label=result["control_type"], linewidth=2, markersize=6)
    ax_r.set_xlabel("Progress time $s$  (0 = noise, 1 = data)")
    ax_r.set_ylabel("Reward")
    ax_r.set_title(f"Reward — N={result['N']} H={result['H']} "
                   f"λ={result['lam']} {result['sample_mode']}")
    ax_r.legend(); ax_r.grid(True, alpha=0.3)

    if has_ctrl:
        ax_u = axes[1]
        norms = result["_ctrl_norms"]          # (N,) per outer step
        steps = range(1, len(norms) + 1)
        ax_u.bar(steps, norms, color="steelblue", alpha=0.8)
        ax_u.set_xlabel("Outer step $t$")
        ax_u.set_ylabel(r"$\|u_t\|$ (mean over batch)")
        energy_str = f"{result['ctrl_energy']:.4f}" if result['ctrl_energy'] == result['ctrl_energy'] else "n/a"
        ax_u.set_title(
            f"Control input — {result['control_type']}\n"
            r"$\sum_t\|u_t\|^2\,\Delta t$"
            f" = {energy_str}"
        )
        ax_u.grid(True, alpha=0.3, axis="y")

    title = f"gen: {result['prompt'][:60]}"
    if result["edit_prompt"] != result["prompt"]:
        title += f"   |   edit: {result['edit_prompt'][:60]}"
    plt.suptitle(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / f"{cfg_tag}__reward.png", dpi=100)
    plt.close(fig)

    result["_img_unc"].save(out_dir / f"{cfg_tag}__img_unc.png")
    if result["control_type"] != "none":
        result["_img_ctrl"].save(out_dir / f"{cfg_tag}__img_ctrl.png")

    # ── DATA BEFORE COSMETICS (reordered 2026-08-26) ─────────────────────
    # The per-outer-step trace is written FIRST, before the constraint plot.
    # It used to be the other way round, and a NameError in
    # _save_constraint_plot (a Step 4 refactor leftover) therefore destroyed
    # the trace of a completed 4:22 run -- the one artifact that answers what
    # mu and h actually did. A cosmetic figure must never be able to cost us
    # data: the trace is the measurement, the plot is a rendering of it.
    #
    # Per-outer-step trace. Written for every controlled run, constrained or
    # not -- summary.csv holds only endpoint values, and Step 10.4's failure
    # modes are distinguished by trajectory shape (see _write_constraint_trace).
    _write_constraint_trace(result, out_dir, cfg_tag)

    # Diagnostic probes (see _run_one's docstring). Written BEFORE the plot,
    # for the same reason the trace is: a probe that costs a full run must not
    # be lost to a rendering failure.
    for _key, _suffix in (("_h_probe", "h_determinism"),
                          ("_dual_probe", "dual_probe")):
        _payload = result.get(_key)
        if _payload is not None:
            _p = out_dir / f"{cfg_tag}__{_suffix}.json"
            with open(_p, "w") as _f:
                json.dump(_payload, _f, indent=2)
            print(f"  wrote {_p}")

    # Constraint diagnostic plot (only when constraints were active). LAST,
    # because it is the only thing here that is purely presentational.
    if result.get("_constraint_traj") is not None:
        _save_constraint_plot(result, out_dir, cfg_tag)

    return cfg_tag


def _save_constraint_plot(result: dict, out_dir: Path, cfg_tag: str) -> None:
    """Save the three-panel per-run diagnostic figure:
        (a) ||u_j||  — control displacement magnitude per outer step.
        (b) mu_i     — Lagrange multiplier trajectory.
        (c) h_i      — constraint value, at three different endpoint estimates.

    PANEL ORDER AND CONTENT match the low-dim
    ``lowdim/visualization_utils.py:visualize_control_results`` figures 3/4/5,
    and the post-hoc ``plot_lowdim_panels.py``, so the three figures can be read
    against each other (2026-09-08: the three should be as overlapping as
    possible). Panel (a) is new as of that date; it was
    previously only a bar chart in ``__reward.png`` on an integer outer-step
    axis, which does not line up with the other two panels' progress-time axis.
    Existing result dirs keep the old 1x2 layout -- they are immutable records.
    """
    import numpy as np

    c_traj     = result["_constraint_traj"]      # (N+1, C), constraint_on convention
    c_traj_raw = result.get("_constraint_traj_raw")
    mu_hist    = result["_mu_hist"]              # tensor (N, B, C) or None
    h_dual_hist = result.get("_h_dual_hist")     # tensor (N, B, C) or None
    ctrl_norms = result.get("_ctrl_norms")       # (N,) array or None
    specs    = result["_constraint_specs"]
    N_steps  = c_traj.shape[0] - 1
    C        = c_traj.shape[1]

    # RAISE on a count mismatch rather than falling back to a positional
    # label (2026-08-26). The previous code did
    # `specs[i]... if i < len(specs) else f"h_{i}"`, i.e. it silently
    # mislabeled when the lists disagreed -- and a plot whose legend names
    # the wrong constraint is worse than no plot at all, because it gets
    # read and believed. There is no legitimate way for these to differ:
    # both come from the same run, C is the width of the h trajectory and
    # specs is what produced it.
    if len(specs) != C:
        raise ValueError(
            f"_save_constraint_plot: {C} constraint column(s) in the h "
            f"trajectory but {len(specs)} ConstraintSpec(s) "
            f"({[s.reward_type for s in specs]!r}). These must agree -- a "
            f"mismatch means the legend would name the wrong constraint, "
            f"so this raises instead of guessing."
        )

    def _spec_label(i: int) -> str:
        """Legend label for constraint i, from its spec (Step 4 generalized
        constraints beyond CLIP text prompts, so there is no prompt list to
        label from any more). `arg` is a text target for
        clip/hpsv2/pickscore/imagereward and a reference image path for
        style, so the basename is used for the latter to keep the legend
        readable. Falls back to the bare index only for a malformed spec."""
        s = specs[i]
        rtype = getattr(s, "reward_type", None)
        if not rtype:
            return f"{i}"
        arg = getattr(s, "arg", "") or ""
        if rtype == "style" and arg:
            arg = os.path.basename(arg)
        return f"{rtype}:{arg}"[:40] if arg else rtype

    # x-axis: progress time s=0..1 sampled at each ConicFlow outer step
    t_steps = np.linspace(0.0, 1.0, N_steps + 1)

    # The h the DUAL actually consumed, one row per outer step (N), so it
    # shares the mu panel's x-axis, not t_steps. Note (2026-09-03): the
    # reconstructed series below evaluates h at the REALIZED state x_j, while
    # the dual reads it at the optimizer's horizon endpoint x_{j+H} -- the two
    # disagree by ~H steps, which is why mu looked like it fell while h was
    # still positive. Plotting only the reconstruction was the whole confusion;
    # this is the "exact fix" agreed on that call. Kept as an OVERLAY rather
    # than a replacement: the reconstruction is the honest answer to "was the
    # constraint satisfied along the realized trajectory", which is a real and
    # different question, so neither series is redundant.
    hd_np = (h_dual_hist.float().mean(dim=1).cpu().numpy()
             if h_dual_hist is not None else None)   # (N, C)
    t_dual = np.linspace(1.0 / N_steps, 1.0, N_steps) if hd_np is not None else None
    if hd_np is not None and hd_np.shape != (N_steps, C):
        # Same loud-fail posture as the specs/C check above: a silently
        # misaligned overlay would be read as "the dual saw this at step j"
        # when it did not.
        raise ValueError(
            f"_save_constraint_plot: h_dual history has shape "
            f"{hd_np.shape} but ({N_steps}, {C}) is required to align with "
            f"the mu panel. A misaligned dual-h overlay would misattribute "
            f"h to the wrong outer step, so this raises instead of plotting."
        )

    fig, (ax_u, ax_mu, ax_h) = plt.subplots(1, 3, figsize=(18, 4))

    # ── (a) control displacement magnitude ────────────────────────────────
    # ctrl_norms[j] is the control APPLIED at outer step j, which produced the
    # state recorded at trace row j+1 -- so it sits on linspace(1/N, 1, N),
    # the same axis as mu, not on t_steps. (The trace CSV leaves step 0 blank
    # for exactly this reason: no control has been applied yet.)
    if ctrl_norms is not None and len(ctrl_norms) == N_steps:
        t_u = np.linspace(1.0 / N_steps, 1.0, N_steps)
        ax_u.plot(t_u, np.asarray(ctrl_norms, dtype=float), "o-",
                  linewidth=2, markersize=5, color="0.15")
        ax_u.set_xlabel("Progress time $s$")
        ax_u.set_ylabel(r"$\|u_j\|$  (control displacement magnitude)")
        ax_u.set_title(
            "Control displacement magnitude\n"
            f"ctrl_energy = {result.get('ctrl_energy', float('nan')):.4g}")
        ax_u.grid(True, alpha=0.3)
    else:
        # Single series, so no legend to lose; but say WHY it is empty rather
        # than showing blank axes. Length disagreement is reported, not
        # silently truncated -- a shifted control series would misattribute
        # every step.
        why = ("no control norms recorded" if ctrl_norms is None else
               f"{len(ctrl_norms)} control norm(s) but {N_steps} outer step(s)")
        ax_u.text(0.5, 0.5, f"no control panel\n({why})",
                  ha="center", va="center", transform=ax_u.transAxes, fontsize=11)
        ax_u.set_axis_off()

    # ── (c) constraint values + threshold lines ───────────────────────────
    show_raw = c_traj_raw is not None and c_traj_raw is not c_traj
    for i in range(C):
        label = _spec_label(i)
        thr = specs[i].threshold
        ax_h.plot(t_steps, c_traj[:, i], "o-", linewidth=2, markersize=5,
                  label=f"$h_{{{i}}}$: {label}  (thr={thr:.3f})")
        color_i = ax_h.lines[-1].get_color()
        if hd_np is not None:
            # Deliberately unlabelled per constraint (one shared legend entry
            # below) so C constraints do not triple the legend.
            ax_h.plot(t_dual, hd_np[:, i], "x--", linewidth=1.5, markersize=6,
                      alpha=0.85, color=color_i)
        if show_raw:
            # Raw-latent h, for comparison only. Deliberately unlabelled per
            # constraint (one shared legend entry below) so C constraints do
            # not double the legend.
            ax_h.plot(t_steps, c_traj_raw[:, i], ":", linewidth=1.2, alpha=0.6,
                      color=color_i)
    if hd_np is not None:
        ax_h.plot([], [], "x--", linewidth=1.5, markersize=6, alpha=0.85,
                  color="0.35",
                  label=r"(dashed) $h$ at $x_{j+H}$ — what $\mu$ consumed")
    if show_raw:
        ax_h.plot([], [], ":", linewidth=1.2, alpha=0.6, color="0.35",
                  label="(dotted) $h$ at the raw latent")
    # h=0 is EVERY constraint's threshold by construction (whatever the
    # reward type or direction), so it is ONE shared line --
    # one shared line, not C identical ones stacked on top of each other.
    ax_h.axhline(0.0, color="0.35", linestyle="--", linewidth=1.2, alpha=0.9,
                 label="$h_i = 0$ (threshold)")

    ax_h.set_xlabel("Progress time $s$  (0 = noise, 1 = data)")
    # Generalized: h_i is threshold-minus-score for direction "ge" and
    # score-minus-threshold for "le", over an arbitrary reward type -- the
    # old CLIP-specific label was wrong for every non-CLIP constraint.
    ax_h.set_ylabel(r"$h_i$  (satisfied when $h_i \leq 0$)")
    _conv = result.get("constraint_on", "endpoint")
    _sub = (r"solid = $h$ at realized $x_j$   |   dashed = $h$ at $x_{j+H}$ "
            r"(drives $\mu$)" if hd_np is not None
            else r"(satisfied when $h_i \leq 0$)")
    ax_h.set_title(
        f"Constraint trajectory — evaluated at: {_conv}\n" + _sub)
    ax_h.legend(fontsize=7, loc="upper right")
    ax_h.grid(True, alpha=0.3)

    # ── (b) mu trajectory ────────────────────────────────────────────────
    if mu_hist is not None:
        # mu_hist: (N, B, C) → mean over batch → (N, C)
        mu_np = mu_hist.float().mean(dim=1).cpu().numpy()  # (N, C)
        t_mu  = np.linspace(1.0 / N_steps, 1.0, N_steps)   # after each step
        for i in range(C):
            label = _spec_label(i)
            ax_mu.plot(t_mu, mu_np[:, i], "s-", linewidth=2, markersize=5,
                       label=f"$\\mu_{{{i}}}$: {label}")
        ax_mu.set_xlabel("Progress time $s$")
        ax_mu.set_ylabel(r"$\mu_i$  (Lagrange multiplier)")
        ax_mu.set_title(
            f"Lagrange multiplier trajectory\n"
            f"lr_dual={result['lr_dual']}  mu_init={result['mu_init']}"
        )
        ax_mu.legend(fontsize=7, loc="upper left")
        ax_mu.grid(True, alpha=0.3)
    else:
        ax_mu.text(0.5, 0.5, "no multiplier history\n(h was None)",
                   ha="center", va="center", transform=ax_mu.transAxes, fontsize=11)
        ax_mu.set_axis_off()

    title = f"gen: {result['prompt'][:60]}"
    if result["edit_prompt"] != result["prompt"]:
        title += f"  |  edit: {result['edit_prompt'][:40]}"
    plt.suptitle(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / f"{cfg_tag}__constraints.png", dpi=100)
    plt.close(fig)


# `reward` and `seed` are new as of the 2026-08-25 merge and are the reason the
# rest of this list can be trusted: without them, 222 of 232 historical run dirs
# required inferring the reward model from directory names and value ranges.
# Every step after Step 1 depends on the CSV being self-describing.
#
# The per-constraint block is fixed-width (thr_i / h_i / mu_i, i < MAX_CONSTRAINTS)
# so the values read back as floats with no string round-trip. Unused slots are
# written as "" -- never 0 or NaN. `constraint_names` carries the ordering and is
# mandatory: h_0 alone does not say which reward it is.
def _write_constraint_trace(result: dict, out_dir: Path, cfg_tag: str) -> Optional[str]:
    """Write the per-outer-step trace for one run: <cfg_tag>__mu_trace.csv.

    WHY THIS EXISTS. summary.csv holds one row per run, so it can only carry the
    ENDPOINT values mu_i / h_i. Step 10.4's first two failure modes are both
    "constraint still violated at the end" and are distinguished only by the
    SHAPE of the trajectory that got there:

        mode 1 — mu rises monotonically while the violation does not shrink
        mode 2 — mu is stable and the constraint is still violated (infeasible)

    A scalar mu_final is identical in both. Without this file a threshold sweep
    produces failures that cannot be classified after the fact and the runs have
    to be repeated. Mode 3 ("ctrl_energy blows up while remaining feasible") needs
    the per-step energy for the same reason.

    ROW ALIGNMENT (the part that must be exactly right). For outer step j,
    0-based, receding_horizon_flow_control does: solve window -> dual update ->
    append mu -> apply the step -> append the new state. So mu_history[j] is the
    multiplier AFTER step j's dual update, traj[j+1] is the state it produced,
    and controls[j] is the control applied. Row `step = j+1` therefore pairs
    mu_hist[j] with constraint_traj[j+1] and ctrl_norms[j].

    Row `step = 0` is the initial condition: h at x0, mu at its mu_init value
    before any dual update, and no control yet. It is included because "did the
    violation shrink?" is unanswerable without the value it started from.

    COLUMNS
        step               outer ConicFlow step, 0..N (0 = initial condition)
        s                  progress time, step/N (0 = noise, 1 = data)
        constraint_names   ordered names, repeated per row so the file is
                           self-describing standalone; also joins to summary.csv
        mu_i               multiplier after this step's dual update
        h_i                constraint value in the cfg.constraint_on convention
                           -- the one the dual update responds to
        h_i_raw            constraint value at the raw realized latent; equal to
                           h_i when constraint_on="horizon", different under
                           "endpoint" (see _run_one for why both are kept)
        viol_i             max(h_i, 0) -- the violation itself, precomputed so
                           "does it shrink?" is a column, not a derivation
        ctrl_norm          ||u|| applied at this step
        ctrl_energy_step   ||u||^2 * dt for this step
        ctrl_energy_cum    running sum, ending at summary.csv's ctrl_energy

    Written for every controlled run, not only constrained ones: the energy
    columns are what mode 3 is diagnosed from, and they cost nothing extra.
    Returns the filename written, or None when there was no controlled run.
    """
    import numpy as np

    ctrl_norms = result.get("_ctrl_norms")
    if ctrl_norms is None:
        return None                     # control_type="none", or norms unavailable

    dts = result.get("_ctrl_dts") or [1.0 / result["N"]] * len(ctrl_norms)
    c_traj     = result.get("_constraint_traj")        # (N+1, C) or None
    c_traj_raw = result.get("_constraint_traj_raw")    # (N+1, C) or None
    mu_hist    = result.get("_mu_hist")                # (N, B, C) tensor or None
    names      = result.get("constraint_names", "") or ""
    C          = c_traj.shape[1] if c_traj is not None else 0
    n_steps    = len(ctrl_norms)

    mu_np = mu_hist.float().mean(dim=1).cpu().numpy() if mu_hist is not None else None

    # LOUD FAIL on any length disagreement. Silently zipping mismatched arrays
    # would mis-attribute mu to the wrong step -- producing a trace that looks
    # fine and diagnoses the wrong failure mode, which is worse than no trace.
    if c_traj is not None and c_traj.shape[0] != n_steps + 1:
        raise ValueError(
            f"constraint trajectory has {c_traj.shape[0]} rows but there are "
            f"{n_steps} control steps; expected {n_steps + 1} (one per state, "
            f"including the initial condition). Step<->row alignment is unsafe."
        )
    if mu_np is not None and mu_np.shape[0] != n_steps:
        raise ValueError(
            f"mu history has {mu_np.shape[0]} entries but there are {n_steps} "
            f"control steps; expected one multiplier update per step."
        )
    if len(dts) != n_steps:
        raise ValueError(
            f"{len(dts)} per-step dt value(s) but {n_steps} control steps."
        )

    cols  = ["step", "s", "constraint_names"]
    cols += [f"mu_{i}" for i in range(C)]
    cols += [f"h_{i}" for i in range(C)]
    _hd = result.get("_h_dual_hist")
    _hd = _hd.float().mean(dim=1).cpu().numpy() if _hd is not None else None
    if _hd is not None:
        cols += [f"h_{i}_dual" for i in range(C)]
    if c_traj_raw is not None and c_traj is not c_traj_raw:
        cols += [f"h_{i}_raw" for i in range(C)]
    cols += [f"viol_{i}" for i in range(C)]
    cols += ["ctrl_norm", "ctrl_energy_step", "ctrl_energy_cum"]

    path = out_dir / f"{cfg_tag}__mu_trace.csv"
    cum = 0.0
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for step in range(n_steps + 1):
            row = {"step": step, "s": step / n_steps, "constraint_names": names}
            for i in range(C):
                # step 0 predates any dual update, so mu is still at mu_init.
                if step == 0:
                    row[f"mu_{i}"] = result.get("mu_init", "")
                elif mu_np is not None:
                    row[f"mu_{i}"] = float(mu_np[step - 1, i])
                else:
                    row[f"mu_{i}"] = ""
                hv = float(c_traj[step, i])
                row[f"h_{i}"] = hv
                row[f"viol_{i}"] = max(hv, 0.0)
                if f"h_{i}_raw" in cols:
                    row[f"h_{i}_raw"] = float(c_traj_raw[step, i])
                if f"h_{i}_dual" in cols:
                    # h_dual_hist has ONE row per outer step (N), while the trace has
                    # N+1 rows (step 0 predates any dual update) -- blank at step 0,
                    # never 0.0, which would read as "the dual saw h=0".
                    hd = _hd
                    row[f"h_{i}_dual"] = ("" if step == 0 or hd is None or step-1 >= hd.shape[0]
                                          else float(hd[step-1, i]))
            if step == 0:
                # No control has been applied yet -- blank, never 0, which would
                # read as "a control of magnitude zero was applied".
                row["ctrl_norm"] = ""
                row["ctrl_energy_step"] = ""
            else:
                nrm = float(ctrl_norms[step - 1])
                e = nrm ** 2 * float(dts[step - 1])
                cum += e
                row["ctrl_norm"] = nrm
                row["ctrl_energy_step"] = e
            row["ctrl_energy_cum"] = cum
            w.writerow(row)
    return path.name


CSV_KEYS = [
    "tag", "reward", "seed", "prompt", "edit_prompt",
    "N", "H", "control_type", "lam", "n_inner", "lr_inner",
    "sample_mode", "warmstart", "n_alt", "skip_first_primal_step",
    "drift_detach", "constraint_on", "endpoint_surrogate",
    # Which locked defaults this run lifted via FLOWMAPS_UNLOCK (2026-09-08).
    # Blank for the overwhelming majority of rows. Recorded so a run made under
    # a lifted lock is never indistinguishable afterwards from one made under
    # it -- which is the entire reason the lock assert exists.
    "unlocked",
    # Peak memory per run (the design plan). Empty on CPU-only runs -- never 0.0,
    # so an absent measurement is distinguishable from a real zero.
    "peak_alloc_flux_gb", "peak_reserved_flux_gb", "peak_alloc_reward_gb",
    "n_constraints", "constraint_names", "lr_dual", "mu_init",
    *[f"thr_{i}"  for i in range(MAX_CONSTRAINTS)],
    *[f"h_{i}"    for i in range(MAX_CONSTRAINTS)],
    *[f"mu_{i}"   for i in range(MAX_CONSTRAINTS)],
    # Step 4.4: per-constraint satisfaction flag + violation magnitude on the
    # FINAL image, derived from h_i (h_i <= 0 <=> satisfied). Without these,
    # Step 10.4's "breaks" definitions (mode 1/2 = still violated at the end)
    # are not directly readable from summary.csv -- only recoverable by
    # re-deriving them from h_i in every downstream analysis script.
    *[f"sat_{i}"  for i in range(MAX_CONSTRAINTS)],
    *[f"viol_{i}" for i in range(MAX_CONSTRAINTS)],
    "boffi_use_adam",
    # hardflow only; blank/default on other control types. Recorded even when
    # unused so a hardflow row is fully self-describing -- the schema's whole
    # point (see the CSV_KEYS note above).
    "hardflow_rho", "hardflow_rho_mult", "hardflow_dual_every",
    "hardflow_margin", "hardflow_activation", "hardflow_reg",
    "hardflow_endpoint_mode",
    # (2026-09-17) the floor-lifting alpha and the lr the run ACTUALLY used.
    "hardflow_lr_alpha", "hardflow_lr_eff",
    "lpips_net", "r_unc", "r_ctrl",
    "r_ctrl_independent", "delta_r",
    "ctrl_energy", "ctrl_norm_mean", "ctrl_norm_max", "objective", "time_unc_s", "time_ctrl_s",
    "run_ok", "error",
    # 2026-09-05: `error` alone was not enough to tell WHY a run failed. Of the
    # 70 run_ok=False rows on disk, 68 have an empty `error` (they came from the
    # non-finite guard, which never set it) and 2 carry a raw string, so an OOM
    # and a NaN blow-up were indistinguishable after the fact. `error_type` is
    # the exception class name (or "non_finite" for the guard), which makes
    # "how many of last night's failures were contention?" a one-line query.
    # `error_msg` is the truncated, newline-stripped message.
    "error_type", "error_msg",
    # dual_adaptive (2026-09-20) is APPENDED, not inserted: --resume appends
    # rows under the header already on disk, so inserting a column mid-list
    # silently shifts every column after it in the appended rows.
    "dual_adaptive",
]


# ── --resume support (2026-09-05) ────────────────────────────────────────────
# A resumed sweep must decide, for each config it is about to run, "is this
# already in summary.csv as a SUCCESS?". That identity has to be reconstructed
# from CSV columns, so it can only be built from fields the CSV actually
# carries. These are every sweep axis that round-trips (see
# _build_configs_from_args's sweep_fields), plus the identity fields that are
# not sweep axes but do distinguish runs.
#
# constraint_thresholds is deliberately absent: it is a LIST on RunConfig and
# lands in the CSV as the fixed positional thr_0..thr_3 columns, which are what
# _resume_key uses instead.
_RESUME_KEY_FIELDS = [
    "prompt", "edit_prompt", "N", "H", "control_type", "lam", "n_inner",
    "lr_inner", "sample_mode", "warmstart", "n_alt", "skip_first_primal_step",
    "drift_detach", "constraint_on", "endpoint_surrogate",
    "boffi_use_adam",
    "hardflow_rho", "hardflow_rho_mult", "hardflow_dual_every",
    "hardflow_margin", "hardflow_activation", "hardflow_reg",
    "hardflow_endpoint_mode", "hardflow_lr_alpha",
    "seed", "lr_dual", "mu_init", "lpips_net",
    "dual_adaptive",
]

# Sweep axes that do NOT round-trip through the CSV. If a resumed sweep varies
# any of these, two rows that differ only in that axis are indistinguishable,
# so resume would skip a config it has not actually run. _plan_resume RAISES in
# that case rather than guessing -- the standing "loud fail only" rule applied
# to resume. (Adding any of these to CSV_KEYS removes the restriction; the
# assertion below keeps this list honest if someone does.)
_RESUME_BLIND_AXES = ["guidance_scale", "height", "width", "boffi_grad_norm_mode"]

assert not (set(_RESUME_KEY_FIELDS) - set(CSV_KEYS)), (
    f"_RESUME_KEY_FIELDS not in CSV_KEYS: "
    f"{sorted(set(_RESUME_KEY_FIELDS) - set(CSV_KEYS))} -- a resume key cannot "
    f"be rebuilt from a column that is never written."
)
assert not (set(_RESUME_BLIND_AXES) & set(CSV_KEYS)), (
    f"_RESUME_BLIND_AXES now IS in CSV_KEYS: "
    f"{sorted(set(_RESUME_BLIND_AXES) & set(CSV_KEYS))} -- move it into "
    f"_RESUME_KEY_FIELDS instead of leaving resume refusing to run."
)


def _resume_norm(v) -> str:
    """Normalize one field to a string that compares equal across the
    RunConfig -> CSV -> str round trip.

    Numbers go through float so 10 == "10" == "10.0"; everything else is
    compared as a stripped string, which covers bools ("True") and prompts.
    """
    if v is None:
        return ""
    # bool BEFORE float: bool is a subclass of int, so float(True) is 1.0 and a
    # RunConfig's warmstart=True would normalize to "1" while the CSV carries
    # the str() form "True" -- the key would never match and --resume would
    # silently re-run everything.
    if isinstance(v, bool):
        return str(v)
    try:
        return f"{float(v):.12g}"
    except (TypeError, ValueError):
        return str(v).strip()


def _resume_key_from_cfg(cfg, reward_model: str) -> tuple:
    parts = [("reward", _resume_norm(reward_model))]
    for f in _RESUME_KEY_FIELDS:
        if f == "edit_prompt":
            # _run_one writes `cfg.edit_prompt if cfg.edit_prompt else cfg.prompt`,
            # so an empty edit_prompt reads back as the generation prompt. Apply
            # the same fallback here or every such config resumes as "not done".
            val = cfg.edit_prompt if cfg.edit_prompt else cfg.prompt
        else:
            val = getattr(cfg, f)
        parts.append((f, _resume_norm(val)))
    # Thresholds MUST come from the resolved specs, not cfg.constraint_thresholds.
    # The two are mutually exclusive (_resolve_constraint_specs raises if both
    # are set): the legacy path fills constraint_thresholds, the spec path
    # leaves it empty and carries thresholds on the specs. Reading the raw field
    # would therefore give empty thr_i for every modern config -- and since
    # constraint_names carries the reward type and arg but NOT the threshold,
    # two configs differing only in threshold would produce the same key and
    # resume would skip one it never ran. _run_one writes the CSV from the
    # resolved specs, so this reads from the same place.
    specs = _resolve_constraint_specs(cfg)
    parts.append(("constraint_names", ",".join(_constraint_names(cfg))))
    for i in range(MAX_CONSTRAINTS):
        parts.append((f"thr_{i}",
                      _resume_norm(specs[i].threshold) if i < len(specs) else ""))
    return tuple(parts)


# Fields added to _RESUME_KEY_FIELDS after rows were already written: a CSV
# predating the column has no value for it, so reading it as "" would never
# match a config carrying the field's default. Map missing -> the default.
_RESUME_FIELD_DEFAULTS = {"dual_adaptive": False}


def _resume_key_from_row(row: dict) -> tuple:
    parts = [("reward", _resume_norm(row.get("reward", "")))]
    for f in _RESUME_KEY_FIELDS:
        raw = row.get(f, None)
        if raw is None or raw == "":
            raw = _RESUME_FIELD_DEFAULTS.get(f, "")
        parts.append((f, _resume_norm(raw)))
    parts.append(("constraint_names", str(row.get("constraint_names", "")).strip()))
    for i in range(MAX_CONSTRAINTS):
        parts.append((f"thr_{i}", _resume_norm(row.get(f"thr_{i}", ""))))
    return tuple(parts)


def _plan_resume(csv_path, configs, reward_model: str, log=print):
    """Return (done_keys, n_existing_rows) for a --resume run, or (set(), 0).

    Only rows with run_ok=True count as done: a run_ok=False row is a recorded
    FAILURE, and the whole point of resuming is to retry it.
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        log(f"  [resume] {csv_path} does not exist yet -- running the full sweep.")
        return set(), 0

    # Refuse rather than guess when an axis the CSV cannot express is swept.
    for axis in _RESUME_BLIND_AXES:
        seen = {_resume_norm(getattr(c, axis)) for c in configs}
        if len(seen) > 1:
            raise ValueError(
                f"--resume cannot be used with a sweep that varies {axis!r} "
                f"(values {sorted(seen)}): {axis} is not a summary.csv column, "
                f"so two rows differing only in it are indistinguishable and "
                f"resume would skip configs it never ran. Either drop --resume "
                f"or split the sweep so {axis} is constant per out_dir."
            )

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = sorted(set(_RESUME_KEY_FIELDS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(
                f"--resume: {csv_path} predates the current CSV schema "
                f"(missing {missing}). Its rows cannot be matched against the "
                f"configs being run. Resume into a fresh --out_dir instead."
            )
        rows = list(reader)

    done = {_resume_key_from_row(r) for r in rows
            if str(r.get("run_ok", "")).strip().lower() == "true"}
    n_failed = sum(1 for r in rows
                   if str(r.get("run_ok", "")).strip().lower() != "true")
    log(f"  [resume] {csv_path}: {len(rows)} row(s), {len(done)} successful "
        f"(will be skipped), {n_failed} failed/incomplete (will be RETRIED).")
    return done, len(rows)


def _parse_threshold_sweep_spec(spec: str, n_prompts: int) -> List[float]:
    """Parse one --sweep_constraint_thresholds entry into a per-prompt list.

    A single float (e.g. ``0.3``) is broadcast to every constraint prompt.
    Comma-separated floats (e.g. ``0.3,0.4``) must match the prompt count.
    """
    parts = [float(x.strip()) for x in spec.split(",")]
    if len(parts) == 1:
        return parts * max(n_prompts, 1)
    if len(parts) != n_prompts:
        raise ValueError(
            f"threshold spec {spec!r} has {len(parts)} value(s) but "
            f"{n_prompts} constraint prompt(s); use one float to broadcast "
            f"or {n_prompts} comma-separated floats."
        )
    return parts


def _format_thresholds(thresholds: List[float]) -> str:
    """Compact string for filename tags and plot titles.

    NOT used for any CSV column -- thresholds go to CSV as the numeric
    thr_0..thr_3 columns, so no row mixes the two encodings.
    """
    return ",".join(f"{t:g}" for t in thresholds)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def _build_configs_from_args(args) -> List[RunConfig]:
    """Build a list of RunConfigs from CLI sweep arguments.

    Each sweep argument accepts one or more values.  The cartesian product of
    all multi-valued arguments is expanded into individual RunConfig objects,
    so ``--N 4 6 --H 2 3`` produces four configs (4×2, 4×3, 6×2, 6×3).

    If no sweep arguments are provided, returns an empty list and the caller
    falls back to SWEEP_CONFIGS (optionally filtered by --configs).
    """
    # Collect all sweep axes that were explicitly set
    axes = {}
    defaults = RunConfig()
    sweep_fields = [
        "prompt", "edit_prompt", "N", "H", "control_type", "lam", "n_inner",
        "lr_inner", "sample_mode", "guidance_scale",
        "height", "width", "warmstart", "n_alt", "skip_first_primal_step",
        "drift_detach", "constraint_on",
        "constraint_thresholds",
        "boffi_use_adam", "boffi_grad_norm_mode",
        # hardflow axes are APPENDED, never inserted: the product order is
        # positionally load-bearing (cf. recover_lr_inner.py, which recovers a
        # missing CSV column from a directory name by index). Appending leaves
        # every pre-2026-08-31 result directory recoverable.
        "hardflow_rho", "hardflow_rho_mult", "hardflow_dual_every",
        "hardflow_margin", "hardflow_activation", "hardflow_reg",
        "hardflow_endpoint_mode",
        # endpoint_surrogate (2026-09-16) is APPENDED for the same reason the
        # hardflow axes were: the cartesian product order is positionally
        # load-bearing for tools that recover a missing CSV column from a
        # directory name by index.
        "endpoint_surrogate",
        # dual_adaptive (2026-09-20) APPENDED for the same reason.
        "dual_adaptive",
    ]
    for f in sweep_fields:
        val = getattr(args, f"sweep_{f}", None)
        if val is not None:
            axes[f] = val

    # Parse threshold sweep specs into List[float] entries.
    if "constraint_thresholds" in axes:
        # Constraint count comes from whichever mechanism is in use -- the
        # legacy CLIP shorthand (--constraint_prompts) or the general path
        # (--constraint_types). main() has already rejected both-at-once, so
        # at most one of these is non-empty here.
        n_prompts = len(args.constraint_prompts) or len(args.constraint_types)
        if n_prompts == 0:
            raise SystemExit(
                "--sweep_constraint_thresholds requires --constraint_prompts "
                "or --constraint_types."
            )
        try:
            axes["constraint_thresholds"] = [
                _parse_threshold_sweep_spec(spec, n_prompts)
                for spec in axes["constraint_thresholds"]
            ]
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    if not axes:
        return []

    # Cartesian product of all axes
    keys = list(axes.keys())
    configs = []
    for combo in itertools.product(*[axes[k] for k in keys]):
        kw = {k: v for k, v in zip(keys, combo)}
        cfg = RunConfig(**{
            f: kw.get(f, getattr(defaults, f)) for f in sweep_fields
        })
        configs.append(cfg)
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Controlled vs uncontrolled FLUX sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sweep via CLI (cartesian product of all provided lists):
  python run_sweep.py \\
      --sweep_prompt "a sunset" "a cat" \\
      --sweep_N 4 6 \\
      --sweep_H 2 3 \\
      --sweep_control_type conicflow fmrg \\
      --sweep_lam 50.0 200.0

  # separate generation prompt vs. editing/reward prompt:
  python run_sweep.py \\
      --sweep_prompt "a portrait of a man" \\
      --sweep_edit_prompt "a smiling face" \\
      --sweep_control_type conicflow fmrg \\
      --sweep_lam 5.0 50.0 200.0

  # CLIP as constraint only (r = zero):
  python run_sweep.py \\
      --sweep_prompt "a portrait of a man" \\
      --sweep_control_type conicflow \\
      --sweep_lam 10000.0 \\
      --reward none \\
      --constraint_prompts "a smiling face" \\
      --constraint_thresholds 0.28 \\
      --lr_dual 0.1 --mu_init 1.0

  # Two CLIP constraints, reward=none (the shape every planned config uses --
  # Steps 7/8/10 all steer entirely through constraints, never a scored reward):
  python run_sweep.py \\
      --sweep_prompt "a portrait of a man" \\
      --sweep_control_type conicflow \\
      --sweep_lam 10000.0 \\
      --reward none \\
      --constraint_prompts "a smiling face" "with glasses" \\
      --sweep_constraint_thresholds 0.28 0.32 0.36 \\
      --lr_dual 0.05 --mu_init 0.5

  # Per-constraint threshold pairs (must match prompt count exactly):
  python run_sweep.py \\
      --sweep_prompt "a portrait of a man" \\
      --constraint_prompts "a smiling face" "with glasses" \\
      --sweep_constraint_thresholds 0.28,0.25 0.32,0.30 0.36,0.35 \\
      --lr_dual 0.05 --mu_init 0.5

  # MIXED reward types (Step 4's general path) -- the actual Task 3 pair,
  # style + ImageReward. --constraint_types/_args/_thresholds/_directions are
  # parallel lists; every one is required and lengths must match exactly.
  # Direction is NEVER inferred: note style is "ge" here because
  # _build_reward_fn returns higher-is-better for every type (style negates
  # its raw Gram distance internally), even though the plan describes
  # style-DISTANCE as an upper bound.
  python run_sweep.py \\
      --sweep_prompt "a portrait of a man" --sweep_edit_prompt "a smiling face" \\
      --sweep_control_type conicflow --sweep_N 10 --sweep_H 4 --sweep_lam 10000.0 \\
      --reward none \\
      --constraint_types style imagereward \\
      --constraint_args "/path/to/style_ref.jpg" "a smiling face" \\
      --constraint_thresholds -900.0 -1.0 \\
      --constraint_directions ge ge \\
      --lr_dual 0.1 --mu_init 0.01

  # Same pair, sweeping BOTH thresholds together (Step 10.4 "vary until it
  # breaks"). Comma-joined negative specs are accepted:
  python run_sweep.py \\
      --sweep_prompt "a portrait of a man" --sweep_edit_prompt "a smiling face" \\
      --sweep_control_type conicflow --reward none \\
      --constraint_types style imagereward \\
      --constraint_args "/path/to/style_ref.jpg" "a smiling face" \\
      --constraint_directions ge ge \\
      --sweep_constraint_thresholds "-900,-1.0" "-800,-0.5" "-700,0.0" \\
      --lr_dual 0.1 --mu_init 0.01

  # NORMALIZED-units thresholds (Step 3's calibration). Requires the matching
  # entry to already exist in --constraint_stats_json -- build_normalizer
  # RAISES rather than falling back, so run the Step 3.3 calibration and
  # pool_calibration_stats.py first. bind_mode=global needs a GLOBAL entry.
  python run_sweep.py \\
      --sweep_prompt "a portrait of a man" --sweep_edit_prompt "a smiling face" \\
      --sweep_control_type conicflow --reward none \\
      --constraint_types style imagereward \\
      --constraint_args "/path/to/style_ref.jpg" "a smiling face" \\
      --constraint_thresholds 1.0 1.0 \\
      --constraint_directions ge ge \\
      --constraint_normalize true true --constraint_bind_mode global \\
      --lr_dual 0.1 --mu_init 0.01

Constraints are supported by --sweep_control_type conicflow and hardflow only.
Requesting them with fmrg/boffi_fmrg raises (those are unconstrained methods),
and so does lr_dual=0 together with mu_init=0 (a silent no-op).
--constraint_prompts (legacy, CLIP-only) and --constraint_types (general) are
mutually exclusive. n_alt>1 / skip_first_primal_step also require an ACTIVE
dual update (>=1 constraint AND lr_dual>0) -- mu_init alone is not enough.

Use SWEEP_CONFIGS in the file when no --sweep_* flags are given.
Filter SWEEP_CONFIGS by index with --configs 0,2,4.
        """,
    )
    # ── infrastructure ───────────────────────────────────────────────────────
    # Absolute default (anchored to this script's own directory) so a relative
    # "results" never lands inside another checkout if this is invoked
    # from a different cwd (e.g. cwd = flow_matching to pick up its imports).
    # Timestamp-prefixed subfolder: keeps runs chronologically sortable and
    # avoids different sweeps colliding on the same run_idx-based image tags.
    default_out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results",
        time.strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--out_dir",   default=default_out_dir)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--hf_token",  default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--model_id",  default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--quantize",  default="int8",
                        choices=["none", "int8", "int4", "nf4"],
                        help="Transformer quantization (default: int8, MEASURED "
                             "12.26 GB resident / 12.40 GB load peak; 'none' is "
                             "23.8 GB and does not fit a 24 GB card. Do NOT use "
                             "nf4 for gradient-based control: correct forward "
                             "pass, ~zero input gradient.)")
    # ── diagnostics (default-off; see _run_one's docstring) ──────────────────
    parser.add_argument("--probe_h_repeat", type=int, default=0, metavar="K",
                        help="DIAGNOSTIC. After the constraint trajectory is "
                             "computed, evaluate h K extra times on the final "
                             "latent and on endpoint_map(1, z_N), and record "
                             "whether that map is bitwise the identity. Writes "
                             "<tag>__h_determinism.json. Separates a "
                             "nondeterministic h_fn from a non-identity "
                             "endpoint_map as the cause of the sporadic "
                             "endpoint/raw alignment warning (3 of 24 runs in "
                             "the 20260904_231018_lpr2c grid). 0 = off.")
    parser.add_argument("--probe_hardflow_duals", action="store_true",
                        help="DIAGNOSTIC, hardflow only. Record mu and h at "
                             "EVERY dual update, not just each sampling step's "
                             "last one, to <tag>__dual_probe.json. mu_hist "
                             "cannot distinguish a multiplier that never rose "
                             "from one that rose and was driven back to zero "
                             "inside the step -- which is the open question "
                             "about HardFlow's second constraint.")
    parser.add_argument("--reward_dtype", default="fp32",
                        choices=["fp32", "fp16", "bf16"],
                        help="dtype for reward/constraint model weights "
                             "(default: fp32, matching every result on disk). "
                             "fp16 halves them -- MEASURED for CLIP ViT-H-14: "
                             "4.01 GB fp32 -> 1.97 GB fp16, and 2.59 -> 1.27 GB "
                             "once the text tower is dropped. Changes reward "
                             "VALUES slightly, so runs are no longer bit-"
                             "comparable with existing CSVs; measure the delta "
                             "with probe_reward_dtype.py before switching. "
                             "Applies to clip/pickscore/imagereward/hpsv2 only "
                             "-- style (0.12 GB) and lpips (0.01 GB) stay fp32 "
                             "because half precision would buy nothing there.")
    parser.add_argument("--fast",      type=lambda x: x.lower() != "false",
                        default=False, metavar="BOOL",
                        help="Disable gradient checkpointing + VAE tiling "
                             "(true/false, default: false)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs already present in "
                             "<out_dir>/summary.csv with run_ok=True, and APPEND "
                             "to that file instead of overwriting it. Rows with "
                             "run_ok=False are retried, not skipped. Matching is "
                             "on the config's identity fields, not on row order, "
                             "so the sweep list may be reordered between "
                             "attempts. Refuses to run if the sweep varies an "
                             "axis summary.csv does not record.")
    parser.add_argument("--continue_on_oom", action="store_true",
                        help="Keep running the remaining configs after a CUDA "
                             "OOM instead of aborting. Off by default: an OOM "
                             "means the card is contended, so continuing yields "
                             "a partially-populated CSV whose gaps are invisible "
                             "downstream.")
    parser.add_argument("--reward",      default="clip",
                        choices=REWARD_CHOICES,
                        help="Reward model for control guidance and evaluation "
                             "(default: clip). Options: clip, hpsv2, pickscore, imagereward, "
                             "style, lpips, none. Use 'none' when steering is done entirely "
                             "via --constraint_prompts.")
    parser.add_argument("--style_ref_paths", nargs="+", default=None, metavar="PATH",
                        help="Reference image path(s) for --reward style (Gram-matrix "
                             "style guidance). Required when --reward style is used; "
                             "applied uniformly to every config in the sweep, same as "
                             "--seed / --out_dir. Multiple paths are style-mixed "
                             "(Gram matrices averaged).")
    parser.add_argument("--lpips_net", default=None, choices=["alex", "vgg", "squeeze"],
                        help="LPIPS backbone for --reward lpips (default: RunConfig's "
                             "'alex', chosen for lowest activation memory -- see "
                             "RunConfig.lpips_net for the tradeoff against 'vgg'). "
                             "Applied uniformly to every config in the sweep, same as "
                             "--seed / --out_dir. No effect for other --reward values.")
    parser.add_argument("--flux_device", default="cuda:0",
                        help="Device for FLUX transformer + VAE (default: cuda:0)")
    parser.add_argument("--reward_device", default="auto",
                        help="Device for reward model. "
                             "'auto' uses a second GPU if available, else CPU.")
    parser.add_argument("--quiet",     action="store_true")
    # Diagnostics flags come from sweep_diagnostics.DIAGS, so adding a
    # diagnostic there gives run_sweep.py its --diag_<name> flag for free and
    # the two cannot drift. Imported inside a try: diagnostics are an optional
    # convenience, and a missing/broken sweep_diagnostics.py must not stop a
    # sweep from being launched at all.
    try:
        from sweep_diagnostics import add_diag_args as _add_diag_args
        _add_diag_args(parser, group_title="Post-sweep diagnostics (opt-in, run over --out_dir)")
    except Exception as _e:
        print(f"  note: diagnostics flags unavailable ({type(_e).__name__}: {_e})")
    parser.add_argument("--pbar_rewards", action="store_true",
                        help="Compute the reward shown in the ConicFlow progress bar. "
                             "OFF by default (2026-09-16): it costs a no-grad "
                             "endpoint rollout plus a decode and a reward-model "
                             "pass every outer step, and another decode + reward "
                             "per inner iteration -- about 4%% of a run at N=8 "
                             "for a display field. The bar itself still shows "
                             "step, control norm and mu.")
    parser.add_argument("--configs",   default=None,
                        help="Comma-separated indices into SWEEP_CONFIGS "
                             "(ignored when any --sweep_* flag is given)")

    # ── sweep axes (each accepts 1+ values; cartesian product is taken) ──────
    d = RunConfig()   # defaults
    parser.add_argument("--sweep_prompt",        nargs="+", metavar="PROMPT")
    parser.add_argument("--sweep_edit_prompt",   nargs="+", metavar="EDIT_PROMPT",
                        help="Reward-model target prompt(s); combined via cartesian "
                             "product with --sweep_prompt. Omit to reward against "
                             "the generation prompt itself.")
    parser.add_argument("--sweep_N",             nargs="+", type=int,   metavar="N")
    parser.add_argument("--sweep_H",             nargs="+", type=int,   metavar="H")
    parser.add_argument("--sweep_control_type",  nargs="+",
                        choices=["conicflow", "fmrg", "boffi_fmrg", "hardflow", "none"],
                        metavar="TYPE")
    parser.add_argument("--sweep_lam",           nargs="+", type=float, metavar="LAM")
    parser.add_argument("--sweep_n_inner",       nargs="+", type=int,   metavar="N_INNER")
    parser.add_argument("--sweep_lr_inner", "--lr_inner",
                        nargs="+", type=float, metavar="LR")
    parser.add_argument("--sweep_sample_mode",   nargs="+",
                        choices=["flow_map1", "flow_map2", "flow_matching"], metavar="MODE")
    # hardflow only. Defaults reproduce the authors' released implementation
    # (see RunConfig). n_inner / lr_inner are NOT duplicated here on purpose --
    # hardflow reuses those shared axes so one CSV column means one thing
    # across control types; the authors' values are n_inner=40, lr_inner=2.5.
    parser.add_argument("--sweep_hardflow_rho",        nargs="+", type=float, metavar="RHO",
                        help="hardflow only. Augmented-Lagrangian quadratic "
                             "penalty weight (authors' default 5.0).")
    parser.add_argument("--sweep_hardflow_rho_mult",   nargs="+", type=float, metavar="MULT",
                        help="hardflow only. Factor rho grows by when the "
                             "violation exceeds tolerance (authors' 2.0).")
    parser.add_argument("--sweep_hardflow_dual_every", nargs="+", type=int,   metavar="K",
                        help="hardflow only. Inner steps between dual updates "
                             "(authors' 8).")
    parser.add_argument("--sweep_hardflow_margin",     nargs="+", type=float, metavar="M",
                        help="hardflow only. Tightens every constraint by this "
                             "amount. The authors hard-code 0.02 and report "
                             "against the untightened threshold; undocumented "
                             "in their paper. Sweep it, do not leave it implicit.")
    parser.add_argument("--sweep_hardflow_activation", nargs="+", type=float, metavar="FRAC",
                        help="hardflow only. Fraction of sampling steps to skip "
                             "before enabling the constraint. Authors hard-code "
                             "0.5; default here is 0.0 (control from step 0) for "
                             "parity with ConicFlow.")
    parser.add_argument("--sweep_hardflow_reg",        nargs="+", type=float, metavar="REG",
                        help="hardflow only. Deviation-from-nominal-endpoint "
                             "penalty. MULTIPLIES (bigger = stronger), the "
                             "opposite of --sweep_lam which divides. Same "
                             "physical quantity as lam, but the conversion is "
                             "reg = N/(2*lam), NOT 1/lam -- it depends on N. "
                             "OMIT THIS FLAG and reg is DERIVED per config as "
                             "N/(2*lam), which is what matches HardFlow's "
                             "effort penalty to ConicFlow's per step -- that is the "
                             "default as of 2026-09-16 (it was a fixed 5e-4, "
                             "correct only at lam=10000/N=10). Supplying it "
                             "OVERRIDES the derivation and is never "
                             "recomputed: pass 0.0 to reproduce runs from "
                             "before 2026-09-09, which had no effort penalty.")
    parser.add_argument("--sweep_endpoint_surrogate", nargs="+",
                        choices=["rollout", "jump"], metavar="MODE",
                        help="conicflow only. How endpoint_map predicts the terminal "
                             "latent INSIDE the inner objective. 'rollout' "
                             "(default) integrates the remaining N-j-H steps -- "
                             "this is the n_inner*N(N-1)/2 term that makes ConicFlow "
                             "quadratic in N. 'jump' takes one flow-map call "
                             "with t_next=0 instead, making it linear. Only the "
                             "surrogate the optimizer differentiates changes; "
                             "the applied trajectory and everything scored from "
                             "it are unaffected. Measured at N=8/H=4: -33%% wall "
                             "clock, objective better at lam=1 and within the "
                             "noise floor at lam=10/100. Not yet re-checked at "
                             "N=10, hence opt-in.")
    parser.add_argument("--sweep_hardflow_endpoint_mode", nargs="+",
                        choices=["flow_map1", "flow_matching", "conicflow_rollout"],
                        metavar="MODE",
                        help="hardflow only. How the terminal estimate is taken. "
                             "flow_map1/flow_matching are single-NFE -- that is "
                             "the method's efficiency claim. conicflow_rollout is the "
                             "estimator ablation: run HardFlow on ConicFlow's full "
                             "endpoint_map rollout, so a head-to-head no longer "
                             "confounds the algorithm with the endpoint estimate.")
    parser.add_argument("--sweep_guidance_scale",nargs="+", type=float, metavar="CFG")
    parser.add_argument("--sweep_height",        nargs="+", type=int,   metavar="H_PX")
    parser.add_argument("--sweep_width",         nargs="+", type=int,   metavar="W_PX")
    parser.add_argument("--sweep_warmstart",     nargs="+", type=lambda x: x.lower() != "false",
                        metavar="BOOL")
    parser.add_argument("--sweep_n_alt",         nargs="+", type=int,   metavar="N_ALT",
                        help="ConicFlow only. Primal<->dual alternations per fixed window "
                             "before it shifts (the design plan). Default 1 = original "
                             "behavior. >1 requires constraints active (h + lr_dual>0) "
                             "-- see receding_horizon_flow_control's n_alt docstring "
                             "in control_loops.py.")
    parser.add_argument("--sweep_dual_adaptive", nargs="+",
                        type=lambda x: x.lower() != "false", metavar="BOOL",
                        help="ConicFlow only. Normalize the projected dual step per "
                             "constraint by a running RMS scale, and use the same "
                             "normalized constraint in the Lagrangian. Equalizes "
                             "dual pressure across constraints whose typical "
                             "violation magnitudes differ, and makes the objective "
                             "invariant to rescaling any h_c. Requires constraints "
                             "active (h + lr_dual>0), same as the other dual "
                             "options. Default False = plain dual ascent.")
    parser.add_argument("--sweep_skip_first_primal_step", nargs="+",
                        type=lambda x: x.lower() != "false", metavar="BOOL",
                        help="ConicFlow only. design plan option (b): one dual-ascent "
                             "update on the uncontrolled rollout's h before the "
                             "loop starts, instead of wasting step 0's primal solve "
                             "on a zero-gradient objective. Composable with a "
                             "nonzero --mu_init (option (a)). Requires constraints "
                             "active (h + lr_dual>0) when True -- see "
                             "receding_horizon_flow_control's skip_first_primal_step "
                             "docstring in control_loops.py.")
    parser.add_argument("--sweep_drift_detach",  nargs="+", type=lambda x: x.lower() != "false",
                        metavar="BOOL")
    parser.add_argument("--sweep_constraint_on", nargs="+",
                        choices=["horizon", "endpoint"], metavar="MODE",
                        help="ConicFlow only. Where to evaluate reward and constraint h(x): "
                             "'endpoint' (default -- flow-map to t=1) or 'horizon' (raw x_H).")

    # ── constraint flags (global, not sweep axes, except the threshold sweep) ──
    parser.add_argument("--constraint_prompts", nargs="+", default=[], metavar="PROMPT",
                        help="One or more CLIP constraint prompts (ConicFlow only). Each "
                             "enforces clip_score(image, prompt) >= threshold. "
                             f"At most {MAX_CONSTRAINTS}; more raises.")
    parser.add_argument("--constraint_thresholds", nargs="+", type=float, default=[],
                        metavar="THRESH",
                        help="Per-prompt CLIP thresholds paired with --constraint_prompts. "
                             "The count must match --constraint_prompts EXACTLY -- missing "
                             "thresholds are not defaulted to 0.0, since 'clip score >= 0' "
                             "is vacuously satisfied and would look active while doing "
                             "nothing. Used when --sweep_constraint_thresholds is not set. "
                             "h_i(z) = threshold_i - clip(z, prompt_i) <= 0.")
    parser.add_argument("--sweep_constraint_thresholds", nargs="+", metavar="THRESH_SPEC",
                        help="Sweep CLIP constraint thresholds (cartesian product axis). "
                             "Each spec is one float broadcast to all prompts (e.g. 0.3) "
                             "or comma-separated floats, one per --constraint_prompt "
                             "(e.g. 0.3,0.4). Requires --constraint_prompts.")
    # ── general-path constraint flags (Step 4's ConstraintSpec, any reward
    # type). Mutually exclusive with the legacy --constraint_prompts above:
    # specifying constraints two ways at once raises (see
    # _resolve_constraint_specs). Added 2026-08-26 -- before this, the
    # generalized builder had no CLI surface at all and Steps 6/7/8/10 could
    # not be launched from the command line.
    parser.add_argument("--constraint_types", nargs="+", default=[],
                        choices=[c for c in REWARD_CHOICES if c != "none"],
                        metavar="TYPE",
                        help="Reward type per constraint (ConicFlow only), e.g. "
                             "'style imagereward'. Pairs positionally with "
                             "--constraint_args / --constraint_thresholds / "
                             "--constraint_directions, all of which are "
                             "REQUIRED and must have the same length. Cannot "
                             "be combined with --constraint_prompts.")
    parser.add_argument("--constraint_args", nargs="+", default=[], metavar="ARG",
                        help="Per-constraint argument, positionally paired with "
                             "--constraint_types: a TEXT TARGET for "
                             "clip/hpsv2/pickscore/imagereward, a style "
                             "REFERENCE IMAGE PATH for style, and an empty "
                             "string '' for lpips (whose reference is always "
                             "the same-seed uncontrolled rollout).")
    parser.add_argument("--constraint_directions", nargs="+", default=[],
                        choices=list(VALID_CONSTRAINT_DIRECTIONS), metavar="DIR",
                        help="Inequality direction per constraint: 'ge' "
                             "(score >= threshold satisfied) or 'le' (score <= "
                             "threshold). NEVER inferred (plan 4.2). Note "
                             "_build_reward_fn returns higher-is-better for "
                             "EVERY type -- style negates its raw Gram "
                             "distance internally -- so style is 'ge' here "
                             "even though the plan describes style-distance "
                             "as an upper bound.")
    parser.add_argument("--constraint_normalize", nargs="+",
                        type=lambda x: x.lower() != "false", default=None,
                        metavar="BOOL",
                        help="Per-constraint: normalize the score via the Step 3 "
                             "calibration stats before comparing to the "
                             "threshold, making that threshold NORMALIZED-units. "
                             "Omit entirely to leave every constraint on raw "
                             "units (the default -- flipping it silently would "
                             "reinterpret existing raw thresholds).")
    parser.add_argument("--constraint_bind_mode", default="global",
                        choices=["global", "prompt"],
                        help="Normalizer bind mode when --constraint_normalize "
                             "is used (plan 3.4; 'global' is the provisional "
                             "recommended default). Requires the corresponding "
                             "entry to exist in --constraint_stats_json -- "
                             "build_normalizer RAISES rather than falling back.")
    parser.add_argument("--constraint_stats_json", default=DEFAULT_STATS_JSON,
                        help="StatsStore JSON backing --constraint_normalize "
                             f"(default: {DEFAULT_STATS_JSON}).")
    parser.add_argument("--lr_dual", type=float, default=0.0,
                        help="Dual-ascent step size for the Lagrange multipliers "
                             "(default 0 = frozen at mu_init, i.e. a fixed penalty). "
                             "Setting BOTH this and --mu_init to 0 with constraints "
                             "requested raises: the constraint term would be identically "
                             "zero, a silent no-op.")
    parser.add_argument("--hardflow_lr_alpha", type=float, default=None,
                        help="hardflow only. Lift the lr*reg >= 1 admissible floor: any "
                             "run whose configured lr_inner * reg would reach 1 uses "
                             "lr = ALPHA/reg instead (0 < ALPHA < 1; 0.5 is dead-beat at "
                             "the final step). Runs above the floor are unchanged. The "
                             "lr actually used is written to the hardflow_lr_eff column. "
                             "Off by default: the guard raises as before.")
    parser.add_argument("--mu_init", type=float, default=0.0,
                        help="Initial value for all Lagrange multipliers (default 0.0). "
                             "See --lr_dual for the both-zero guard.")
    parser.add_argument("--sweep_boffi_use_adam", nargs="+", type=lambda x: x.lower() != "false",
                        metavar="BOOL",
                        help="boffi_fmrg only. True (default) = upstream's Adam branch "
                             "(generate_aesthetic.py's recommended config). False = "
                             "upstream's gradient-descent fallback -- combine with "
                             "--sweep_boffi_grad_norm_mode normalize for the FMRG paper's "
                             "literal App. E.2 formula (u/||u|| * ||v||, undecayed step).")
    parser.add_argument("--sweep_boffi_grad_norm_mode", nargs="+",
                        choices=["none", "normalize"], metavar="MODE",
                        help="boffi_fmrg only. Only has an effect when "
                             "--sweep_boffi_use_adam false (dead parameter on the Adam branch).")

    # Let argparse accept comma-joined NEGATIVE threshold specs as VALUES
    # rather than misreading them as option strings.
    #
    # WHY (found 2026-08-26 while assembling the first real run commands):
    # argparse classifies an argument beginning with "-" as a value only if it
    # matches its own `_negative_number_matcher`, which is
    # `^-\d+$|^-\d*\.\d+$` -- a SINGLE number. So `--constraint_thresholds
    # -900.0 -1.0` works, but `--sweep_constraint_thresholds "-900,-1.0"` (the
    # comma-joined per-constraint form `_parse_threshold_sweep_spec` accepts,
    # and the form Step 10.4 needs to give two constraints different
    # thresholds) fails with a confusing "unrecognized arguments" error. This
    # bites specifically because style's thresholds are ALWAYS negative (its
    # plausible range is [-5000, 0]) and ImageReward's usually are, so the
    # Task 3 pair cannot express a threshold sweep without this.
    #
    # Widened to "a '-' followed by a digit, then only number-ish characters",
    # which cannot collide with any real flag (every option in this parser has
    # a letter after its dashes) and leaves argparse's behavior for genuine
    # options untouched.
    parser._negative_number_matcher = re.compile(r"^-\d[\d.,eE+-]*$")

    args = parser.parse_args()

    # Mutual exclusion between the two constraint mechanisms, checked here so
    # the message names the FLAGS rather than the RunConfig fields
    # _resolve_constraint_specs would report. Same rule, earlier and clearer.
    if args.constraint_types and args.constraint_prompts:
        parser.error(
            "--constraint_types and --constraint_prompts are mutually "
            "exclusive: --constraint_prompts is the legacy CLIP-only shorthand "
            "(equivalent to --constraint_types clip ... --constraint_directions "
            "ge), and specifying constraints both ways leaves it ambiguous "
            "which set is active. Use one."
        )
    # These four are a package: any one of them without --constraint_types is
    # a silent no-op (nothing reads them), which the standing rules forbid.
    for dep in ("constraint_args", "constraint_directions"):
        if getattr(args, dep) and not args.constraint_types:
            parser.error(f"--{dep} requires --constraint_types.")
    if args.constraint_normalize is not None and not args.constraint_types:
        parser.error("--constraint_normalize requires --constraint_types.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # HF auth
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)

    # Build config list: CLI sweep overrides SWEEP_CONFIGS
    configs = _build_configs_from_args(args)
    if configs:
        print(f"CLI sweep: {len(configs)} config(s) from cartesian product of --sweep_* args")
    else:
        configs = list(SWEEP_CONFIGS)
        if args.configs:
            idxs = [int(i) for i in args.configs.split(",")]
            configs = [SWEEP_CONFIGS[i] for i in idxs]
            print(f"Using SWEEP_CONFIGS indices {idxs}: {len(configs)} config(s)")
        else:
            print(f"Using all {len(configs)} SWEEP_CONFIGS entries")

    # Apply seed, out_dir, and constraint params to all configs
    for cfg in configs:
        cfg.seed = args.seed
        cfg.out_dir = args.out_dir
        if args.style_ref_paths:
            cfg.style_ref_paths = args.style_ref_paths
        if args.lpips_net:
            cfg.lpips_net = args.lpips_net
        # list(...) per cfg rather than sharing one list object across all of them
        cfg.constraint_prompts = list(args.constraint_prompts)
        # When --sweep_constraint_thresholds is given, _build_configs_from_args
        # has already set a per-config threshold list; don't clobber it.
        if args.sweep_constraint_thresholds is None:
            cfg.constraint_thresholds = list(args.constraint_thresholds)
        # General path (--constraint_types): convert this cfg's threshold
        # vector into ConstraintSpecs and CLEAR the legacy fields. Clearing
        # matters -- _resolve_constraint_specs raises when constraint_specs
        # and the legacy pair are BOTH populated, and the threshold list
        # above (or the sweep axis) legitimately populates the legacy field
        # on its way here. Done per-cfg rather than once, so
        # --sweep_constraint_thresholds sweeps the general path too (each
        # cfg already carries its own threshold vector at this point), which
        # is what Step 10.4's "vary the threshold until it breaks" needs.
        if args.constraint_types:
            cfg.constraint_specs = _specs_from_cli_lists(
                types=list(args.constraint_types),
                cargs=list(args.constraint_args),
                thresholds=list(cfg.constraint_thresholds),
                directions=list(args.constraint_directions),
                normalize=(list(args.constraint_normalize)
                           if args.constraint_normalize is not None else None),
                bind_mode=args.constraint_bind_mode,
                stats_json=args.constraint_stats_json,
            )
            cfg.constraint_prompts = []
            cfg.constraint_thresholds = []
        cfg.lr_dual = args.lr_dual
        cfg.mu_init = args.mu_init
        cfg.hardflow_lr_alpha = args.hardflow_lr_alpha
        # 2026-09-16: hardflow_reg now derives from N and lam unless supplied.
        # Done HERE because this loop is the one place both config sources --
        # the CLI cartesian product and SWEEP_CONFIGS -- funnel through, and it
        # runs before the pre-FLUX run_preflight pass below, which raises on an
        # unresolved None.
        _resolve_hardflow_reg(cfg)

    if args.reward == "style" and not args.style_ref_paths:
        parser.error("--reward style requires --style_ref_paths (one or more image paths)")

    # Validate every config's constraint settings BEFORE loading FLUX -- a bad
    # sweep should fail in seconds, not after a multi-minute model load. The
    # same checks run again inside _run_one, where SWEEP_CONFIGS entries that
    # never went through argparse are also covered.
    for _i, cfg in enumerate(configs):
        try:
            # Printing suppressed here (no-op log): the per-config resolved
            # values are printed for real at each run's actual start inside
            # _run_one, where "at run start" literally applies. This pass only
            # needs the RAISE, so a bad sweep fails before FLUX loads.
            run_preflight(cfg, log=lambda *a, **k: None)
        except ValueError as exc:
            raise SystemExit(f"config {_i}: {exc}") from exc

    flux_device, reward_device = _resolve_devices(args.flux_device, args.reward_device)
    if not args.quiet:
        print(f"Devices: FLUX={flux_device}  reward({args.reward})={reward_device}")
        # Topology BEFORE any model load, so a device problem is visible
        # before minutes are spent (see _report_cuda_topology).
        _report_cuda_topology(log=print)

    # Run sweep — CSV is written incrementally (one row per run, flushed to
    # disk immediately) so a crash/OOM/kill partway through a long sweep loses
    # at most the in-flight run, not every result computed so far.
    #
    # 2026-09-05: the CSV is now opened and its header written BEFORE the model
    # load, which used to sit above this block. A contention OOM during
    # FluxFlowMapSampler's `.to(cuda)` killed the process before summary.csv
    # existed, so the run left a completely empty results dir -- 44 of 475 dirs
    # under results/ are exactly that. Every aggregator globs summary.csv, so
    # those configs went missing from the analysis rather than showing up as
    # failures: zero of the 70 run_ok=False rows on disk are OOMs, but the 44
    # empty dirs are. Header-first plus the try below turns that into a visible
    # row.
    csv_path = out_dir / "summary.csv"

    # --resume: skip configs already recorded as successes in this out_dir.
    # NOTE the open mode below. Without --resume this is "w", which TRUNCATES --
    # so resuming into an existing dir without this branch would silently
    # destroy every result already computed there.
    _resume_done, _n_existing = (set(), 0)
    if args.resume:
        _resume_done, _n_existing = _plan_resume(csv_path, configs, args.reward)

    if _n_existing:
        csv_file = open(csv_path, "a", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=CSV_KEYS)   # header already there
    else:
        csv_file = open(csv_path, "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=CSV_KEYS)
        writer.writeheader()
        csv_file.flush()

    # Load sampler once
    lora_path = args.lora_path or os.path.join(
        FMRG_DIR, "checkpoints", "flux-flowmap-lora"
    )
    print(f"Loading FluxFlowMapSampler (quantize_mode={args.quantize})...")
    # Use the first prompt as the initial cached embedding; it will be
    # re-encoded in _run_one when a different prompt is encountered.
    first_prompt = configs[0].prompt if configs else "a photo"
    try:
        sampler = FluxFlowMapSampler(
            model_id=args.model_id,
            lora_path=lora_path,
            device=str(flux_device),
            dtype=torch.bfloat16,
            prompt=first_prompt,
            quantize_mode=args.quantize,
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        _is_oom = isinstance(e, torch.cuda.OutOfMemoryError)
        row = {k: "" for k in CSV_KEYS}
        row.update({
            "tag": "LOAD_FAILED", "run_ok": False,
            "error_type": type(e).__name__,
            "error_msg": str(e)[:500].replace("\n", " "),
            "reward": args.reward,
        })
        writer.writerow(row); csv_file.flush(); csv_file.close()
        (out_dir / "LOAD_FAILED.txt").write_text(
            f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        )
        print(f"\nFATAL: model load failed before any config ran: "
              f"{type(e).__name__}: {e}")
        if _is_oom:
            print("  This is a CUDA OOM during model load -- the card did not have "
                  "enough free memory. Check the launcher's memory gate against "
                  "the current measured requirement (see the memory-budget notes).")
        raise SystemExit(1)

    device = next(sampler.pipeline.transformer.parameters()).device
    dtype  = next(sampler.pipeline.transformer.parameters()).dtype
    low_memory = not args.fast

    # Freeze everything we never optimize. Measured 2026-09-05: under int8 the
    # transformer already has 0/1864 params requiring grad (torchao's quantized
    # weights are not trainable), but the VAE has 244 params / 83.8 M elements
    # that DO, and the VAE decode sits inside the differentiated graph. Both
    # control_loops.py and image_conicflow.py use `.backward()` rather than
    # `autograd.grad` (deliberately -- autograd.grad is incompatible with the
    # reentrant checkpointing diffusers defaults to), and `.backward()`
    # populates `.grad` on every leaf that wants it, which opt.zero_grad() never
    # clears because it only sees x_traj. Freezing is exact: gradients still
    # flow to the latent INPUT, which is all the control needs.
    _n_frozen = 0
    for _mod in (sampler.pipeline.transformer, sampler.pipeline.vae):
        for _p in _mod.parameters():
            if _p.requires_grad:
                _p.requires_grad_(False)
                _n_frozen += 1
    if _n_frozen and not args.quiet:
        print(f"  [mem] froze {_n_frozen} transformer/VAE params (never optimized)")

    n_rows = 0
    failed_tags = []
    save_failed_tags: list = []   # save-path failures whose RUN data was still recorded
    n_skipped = 0
    try:
        for run_idx, cfg in enumerate(configs):
            if _resume_done and _resume_key_from_cfg(cfg, args.reward) in _resume_done:
                n_skipped += 1
                print(f"\nRun {run_idx+1}/{len(configs)}: SKIPPED (--resume: already "
                      f"in summary.csv with run_ok=True)  control={cfg.control_type} "
                      f"N={cfg.N} H={cfg.H} lam={cfg.lam} seed={cfg.seed}")
                continue
            print(f"\n{'='*60}")
            print(f"Run {run_idx+1}/{len(configs)}: "
                  f"control={cfg.control_type} N={cfg.N} H={cfg.H} "
                  f"lam={cfg.lam} mode={cfg.sample_mode}")
            print(f"  prompt: {cfg.prompt[:60]}")
            if cfg.edit_prompt:
                print(f"  edit_prompt: {cfg.edit_prompt[:60]}")

            try:
                result = _run_one(
                    sampler, cfg, device, reward_device, dtype, quiet=args.quiet,
                    fast=args.fast, low_memory=low_memory, reward_model=args.reward,
                    reward_dtype=_REWARD_DTYPES[args.reward_dtype],
                    probe_h_repeat=args.probe_h_repeat,
                    probe_hardflow_duals=args.probe_hardflow_duals,
                    pbar_rewards=args.pbar_rewards,
                )
            except Exception as e:
                # 2026-08-21 (Phase A2e): a config that raises is no longer
                # silently dropped from the CSV -- still caught here (so the
                # remaining configs in the sweep still run), but the row is
                # written anyway with run_ok=False and the exception recorded,
                # so a sweep that produces fewer rows than requested is
                # visible in the CSV itself, not just in stdout. Previously
                # this was `print(...); continue`, which skipped the row
                # entirely and still exited 0.
                import traceback; traceback.print_exc()
                fail_tag = (
                    f"{run_idx:03d}__{cfg.control_type}__N{cfg.N}__H{cfg.H}"
                    f"__lam{cfg.lam}__{cfg.sample_mode}__FAILED"
                )
                print(f"  ERROR: {type(e).__name__}: {e}")
                row = {k: "" for k in CSV_KEYS}
                row.update({
                    "tag": fail_tag, "reward": args.reward, "seed": cfg.seed,
                    "prompt": cfg.prompt, "edit_prompt": cfg.edit_prompt or "",
                    "N": cfg.N, "H": cfg.H, "control_type": cfg.control_type, "lam": cfg.lam,
                    "n_inner": cfg.n_inner, "lr_inner": cfg.lr_inner, "sample_mode": cfg.sample_mode,
                    "warmstart": cfg.warmstart, "n_alt": cfg.n_alt,
                    "skip_first_primal_step": cfg.skip_first_primal_step,
                    "dual_adaptive": cfg.dual_adaptive,
                    "drift_detach": cfg.drift_detach,
                    "constraint_on": cfg.constraint_on,
                    "n_constraints": len(cfg.constraint_prompts),
                    "constraint_names": ",".join(_constraint_names(cfg)),
                    "lr_dual": cfg.lr_dual, "mu_init": cfg.mu_init,
                    # thr_i is known even on a failed run; h_i / mu_i are not and
                    # stay "" -- never 0 or NaN, which would read as measured.
                    **{f"thr_{i}": (cfg.constraint_thresholds[i]
                                    if i < len(cfg.constraint_thresholds) else "")
                       for i in range(MAX_CONSTRAINTS)},
                    "boffi_use_adam": cfg.boffi_use_adam,
                    "lpips_net": cfg.lpips_net,
                    "r_unc": float("nan"), "r_ctrl": float("nan"), "delta_r": float("nan"),
                    "ctrl_energy": float("nan"), "ctrl_norm_mean": float("nan"),
                    "ctrl_norm_max": float("nan"), "objective": float("nan"),
                    "time_unc_s": float("nan"), "time_ctrl_s": float("nan"),
                    "run_ok": False,
                    "error": f"{type(e).__name__}: {str(e)[:300]}",
                    "error_type": type(e).__name__,
                    "error_msg": str(e)[:500].replace("\n", " "),
                })
                writer.writerow(row)
                csv_file.flush()
                os.fsync(csv_file.fileno())
                failed_tags.append(fail_tag)
                # 2026-09-05: a CUDA OOM is not a per-config failure -- it says
                # the card is contended right now. Continuing marches the rest
                # of the sweep through the same contention (and through a CUDA
                # context that may already be poisoned), which is exactly how an
                # overnight run came back partial and contaminated: some configs
                # measured, some missing, nothing saying which. Stop by default
                # and let the launcher's retry/queue logic re-acquire a card.
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    if args.continue_on_oom:
                        print("  --continue_on_oom set: continuing into a "
                              "contended card at your own risk.")
                        gc.collect(); torch.cuda.empty_cache()
                        continue
                    print(f"\nABORTING SWEEP after CUDA OOM on run {run_idx+1}/"
                          f"{len(configs)}. {n_rows} config(s) completed and are "
                          f"in {csv_path}; the remaining "
                          f"{len(configs)-run_idx-1} were NOT run.\n"
                          f"  Re-launch when the card is free; pass "
                          f"--continue_on_oom to override.")
                    break
                continue

            # _save_run used to sit OUTSIDE any exception handling, so a crash
            # anywhere in the save path escaped the loop entirely: no data row,
            # no FAILED_ row, process dead -- which is exactly what happened on
            # 2026-08-26 (a NameError in _save_constraint_plot destroyed a
            # completed N=10/H=4 run, leaving summary.csv with only a header).
            # That also silently violated the standing rule that a failed run
            # still gets a run_ok row and is never skipped.
            #
            # At this point _run_one has ALREADY succeeded and every number is
            # in `result`, so a save-path failure must not cost the
            # measurement. Catch it, record it in the `error` column with an
            # unambiguous SAVE_FAILED prefix, and write the data row anyway.
            # run_ok reflects the RUN (which was fine); the error column
            # reflects the save. Loud: traceback printed, tag collected, and
            # reported again in the sweep summary.
            _save_err = None
            try:
                result["tag"] = _save_run(result, out_dir, run_idx)
            except Exception as _se:
                import traceback; traceback.print_exc()
                _save_err = f"SAVE_FAILED {type(_se).__name__}: {str(_se)[:280]}"
                result.setdefault(
                    "tag",
                    f"{run_idx:03d}__{cfg.control_type}__N{cfg.N}__H{cfg.H}"
                    f"__lam{cfg.lam}__{cfg.sample_mode}__SAVEFAIL",
                )
                print(f"\n  !! {_save_err}")
                print(f"  !! The RUN's numbers are intact and are being written "
                      f"to summary.csv below; only the save path failed.\n")
                save_failed_tags.append(result["tag"])

            row = {k: result.get(k, "") for k in CSV_KEYS}
            if _save_err:
                row["error"] = _save_err
            writer.writerow(row)
            csv_file.flush()
            os.fsync(csv_file.fileno())
            n_rows += 1

            if not args.quiet:
                energy = result["ctrl_energy"]
                energy_str = f"{energy:.4f}" if energy == energy else "n/a"
                obj = result["objective"]
                obj_str = f"{obj:.4f}" if obj == obj else "n/a"
                print(f"  r_unc={result['r_unc']:.4f}  r_ctrl={result['r_ctrl']:.4f}"
                      f"  Δr={result['delta_r']:+.4f}"
                      f"  Σ‖u‖²Δt={energy_str}"
                      f"  J={obj_str}"
                      f"  t_ctrl={result['time_ctrl_s']:.1f}s")
    finally:
        csv_file.close()

    if save_failed_tags:
        print(f"\n  !! {len(save_failed_tags)} run(s) COMPLETED but hit a "
              f"save-path error; their data rows were written anyway "
              f"(see the `error` column):")
        for t in save_failed_tags:
            print(f"       {t}")
    n_failed = len(failed_tags)
    # A config that was neither run nor skipped was cut short -- the OOM abort
    # above breaks the loop. Counting it explicitly means the tail of an
    # interrupted sweep is visible in the summary rather than being inferred
    # from a row count that looks merely "smaller than requested".
    n_not_run = len(configs) - n_rows - n_failed - n_skipped
    print(f"\n{'='*60}")
    print(f"SWEEP SUMMARY: {len(configs)} requested, {n_rows} completed ok, "
          f"{n_failed} failed, {n_skipped} skipped (--resume), "
          f"{n_not_run} not reached")
    if failed_tags:
        print("  failed configs:")
        for t in failed_tags:
            print(f"    - {t}")
    if n_not_run:
        print(f"  !! {n_not_run} config(s) were never attempted -- the sweep "
              f"stopped early. Re-launch with --resume to finish them.")
    print(f"\n{n_rows}/{len(configs)} runs completed. Results saved to {out_dir}/")
    print(f"CSV summary: {csv_path}")

    # Post-sweep diagnostics. Wrapped on purpose: these are analysis on top of a
    # sweep that is already written to disk, so a bug here must never turn a
    # completed sweep into a failed one. The exit code below reflects the RUNS.
    #
    # Note this only sees THIS out_dir. A cross-method diagnostic therefore needs
    # a single invocation that ran every method (one --sweep_control_type with
    # all three, on a shared lambda grid); auto_sweep.sh, which gives each method
    # its own out_dir, runs the same diagnostics itself once every method is done.
    try:
        from sweep_diagnostics import selected_diags as _selected_diags
        _which = _selected_diags(args)
    except Exception:
        _which = []
    # `n_rows or _n_existing`, not just n_rows: with --resume on an already
    # complete sweep nothing new runs, and that is precisely when re-running the
    # diagnostics over the existing CSV is useful.
    if _which and (n_rows or _n_existing):
        try:
            from sweep_diagnostics import run_diagnostics
            print(f"\n{'='*60}\nPOST-SWEEP DIAGNOSTICS: {', '.join(_which)}")
            run_diagnostics([str(out_dir)], _which, out_dir=str(out_dir))
        except Exception as e:
            import traceback; traceback.print_exc()
            flags = " ".join(f"--diag_{w}" for w in _which)
            print(f"  !! diagnostics failed ({type(e).__name__}: {e}) -- the sweep itself is fine, "
                  f"re-run them with: python sweep_diagnostics.py --runs {out_dir} {flags}")

    if n_failed or n_not_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
