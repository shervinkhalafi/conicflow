"""
reward_normalization.py -- Two-family reward normalization (the design plan).
======================================================================

Pins the INTERFACE Step 4's constraint builder depends on: what a normalizer
*is* (its call signature), whether Family A and Family B need different
shapes, and whether a normalizer is bound per-prompt or prompt-agnostic. Does
NOT implement Step 3.3's sampling loop -- this module only defines the schema
or that loop writes into and the builder reads out of.

Two families (decided 2026-08-24; see the design plan):

  Family A -- text-image pair scores (clip, hpsv2, pickscore, imagereward):
      normalized = (r - mean) / std
      The normalized reward has mean zero and standard deviation one.

  Family B -- distance-type (style, lpips):
      normalized = r / sqrt(E[r^2])          -- NOT mean-subtracted.
      Dividing by sqrt(E[r^2]) implicitly treats the mean as zero. The family's
      natural minimum is 0 by construction (a distance / negated distance),
      so centering at the mean is meaningless for it.

3.7 -- LPIPS is ill-posed under Family B, BY CONSTRUCTION, not as a bug here.
LPIPS's x_ref is the same-seed UNCONTROLLED output, so on uncontrolled
calibration samples LPIPS is exactly 0 for every sample: r_i = 0 for all i,
so sqrt(E[r^2]) = 0, and Family B's own formula divides by zero. This module
does NOT paper over that: Stats.compute() and Normalizer.__call__() both
RAISE rather than clamp/substitute/skip when a Family B scale is exactly (or
suspiciously close to) zero. Per the plan: "Do not pick one unilaterally --
raise it." The raise's message names the three options the plan lists
(exclude LPIPS from auto-normalization; redefine the calibration reference;
normalize against controlled samples at some fixed lambda) so the error
itself queues the question rather than only crashing.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Literal

from reward_ranges import REWARD_OBSERVED_PROVENANCE, observed_range

# ── Family registry ──────────────────────────────────────────────────────────
# clip/hpsv2/pickscore/imagereward: text-image pair scores (Family A).
# style/lpips: distance-type, minimum 0 by construction (Family B).
# Kept as an explicit registry (not inferred from name/sign) for the same
# reason 4.2's sign convention is explicit: getting this implicit is how a
# reward silently gets normalized under the wrong formula.
FAMILY_A = frozenset({"clip", "hpsv2", "pickscore", "imagereward"})
FAMILY_B = frozenset({"style", "lpips"})

GLOBAL_KEY = "GLOBAL"   # prompt-agnostic stats key, pooled across calibration prompts

BindMode = Literal["prompt", "global"]


def reward_family(reward_name: str) -> str:
    """Return "A" or "B". Raises on anything not in the registry -- a new
    reward type added to REWARD_CHOICES (run_sweep.py) without a family
    assignment here is a configuration gap, not something to guess at."""
    if reward_name in FAMILY_A:
        return "A"
    if reward_name in FAMILY_B:
        return "B"
    raise ValueError(
        f"reward {reward_name!r} has no Family A/B assignment in "
        f"reward_normalization.py's FAMILY_A/FAMILY_B registry. Add it "
        f"explicitly -- family is never inferred from the reward's name or "
        f"sign (see plan 4.2's sign-convention rationale for why implicit "
        f"is how this silently goes wrong)."
    )


# ── Family B scale function: injectable per 3.8 ─────────────────────────────
# sqrt(E[X^2]) was adopted for style/LPIPS as a provisional choice -- treat this
# as the working default, not a hardcoded formula. Step 4's builder (or this
# module's caller) can pass a different `family_b_scale_fn` to compute_stats
# without touching any call site.
def default_family_b_scale(values: List[float]) -> float:
    """sqrt(E[x^2]) over the raw (unnormalized) calibration values."""
    n = len(values)
    if n == 0:
        raise ValueError("default_family_b_scale: empty value list.")
    ms = sum(v * v for v in values) / n
    return math.sqrt(ms)


# A scale of exactly 0 is not "small" -- it means every calibration sample
# was identically 0 (LPIPS's structural failure mode, §3.7). Anything below
# this is ALSO treated as the same failure, since floating-point noise around
# a true zero could otherwise slip through as a huge but finite normalizer.
_FAMILY_B_ZERO_SCALE_ATOL = 1e-8

# ── Low-spread check (added 2026-08-25, on review) ──────────────────────────
# Distinct from, and layered ON TOP of, the exact-zero check above. That check
# catches division by (approximately) zero. This one catches the gap between
# "not zero" and "big enough to be a healthy discriminator" -- a reward whose
# calibration std (Family A) or rms (Family B) is small but nonzero passes the
# exact-zero check and then gets ITS OWN NOISE FLOOR AMPLIFIED by
# normalization: normalize(r) = (r-mean)/std means a raw deviation d becomes
# d/std in normalized units, so if std happens to be tiny, ordinary
# seed-to-seed generation noise (not a meaningful reward difference) reads as
# a large normalized swing -- and every threshold built on that normalized
# scale becomes meaningless.
#
# THRESHOLD, derived rather than picked: let f = scale / range_width, where
# range_width is the reward's own REWARD_OBSERVED_RANGE span (the range it
# ACTUALLY varies over -- NOT the analytic REWARD_PLAUSIBLE_RANGE, which is a
# wrong-value guard and, used here, made this check too strict; see
# reward_ranges.py's split rationale. Native,
# raw-scale plausible extent). A raw deviation equal to f * range_width
# normalizes to exactly 1.0 -- i.e. f is the fraction of the reward's ENTIRE
# plausible range that gets treated as "one full normalized standard
# deviation," the same unit Step 10.2 anchors thresholds in ("start at ~1
# std, expect success"). If f <= 0.01 (1% of the whole plausible range), a
# raw difference no bigger than 1% of everything this reward can possibly
# express is being treated as equivalent to a full threshold step -- at or
# below that point a plausible generation-noise-sized fluctuation (not a real
# reward difference) is indistinguishable, in normalized units, from the
# exact quantity the threshold-anchoring scheme in Step 10.2 is built around.
# 1% is therefore not an arbitrary round number: it is the point past which
# the normalized-threshold scheme's own unit stops being trustworthy.
#
# ImageReward is named explicitly because it is the exposure on record: two
# smoke-test rows, no lambda sweep, and it is half the Task 3 pair. If ITS
# calibration std comes back under this threshold, that is a finding that
# should change the Task 3 pair (to the CLIP+style fallback) BEFORE Step 10
# runs, not something to discover from uninterpretable threshold results
# after the fact.
MIN_SPREAD_FRACTION = 0.01


def _check_low_spread(reward_name: str, prompt_key: str, scale: float) -> None:
    """Raise if `scale` (std for Family A, rms for Family B) is nonzero but
    small relative to reward_name's own OBSERVED (attainable) range -- see
    reward_ranges.observed_range() and the split rationale there. Called
    from Stats.compute() AFTER the family-specific exact-zero check, so this
    never fires on the zero case -- that is a different failure, with its
    own message, already raised earlier.
    """
    lo, hi = observed_range(reward_name)
    range_width = hi - lo
    frac = scale / range_width
    if frac < MIN_SPREAD_FRACTION:
        raise ValueError(
            f"Low-spread calibration for reward={reward_name!r} at "
            f"prompt_key={prompt_key!r}: scale={scale:.6g} is only "
            f"{frac*100:.3g}% of this reward's OBSERVED (attainable) range "
            f"[{lo}, {hi}] (width={range_width:.6g}; provenance: "
            f"{REWARD_OBSERVED_PROVENANCE[reward_name]}). Below "
            f"{MIN_SPREAD_FRACTION*100:.0f}%, ordinary seed-to-seed "
            f"generation noise becomes indistinguishable, in NORMALIZED "
            f"units, from a full threshold step (Step 10.2 anchors "
            f"thresholds at ~1 normalized std) -- every threshold built on "
            f"this calibration would be measuring noise, not signal. NOT "
            f"clamped or worked around: if this is {reward_name!r} "
            f"specifically, this is exactly the kind of finding that should "
            f"swap the Task 3 pair to the CLIP+style fallback BEFORE Step "
            f"10 runs, not something to discover from uninterpretable "
            f"threshold-sweep results afterward. Recalibrate with more/"
            f"different samples to rule out an unlucky draw before "
            f"concluding the reward itself doesn't discriminate here."
        )


@dataclass
class Stats:
    """One (reward_name, prompt_key) calibration record.

    prompt_key is either a literal prompt string or GLOBAL_KEY. Exactly one
    of (mean, std) [Family A] or (rms,) [Family B] is populated, matching
    reward_family(reward_name) -- enforced in compute(), not left to the
    caller to get right.

    `seeds` is REQUIRED (no default) in Stats.compute() -- recorded so a
    later r_unc consistency check can tell whether a specific (prompt, seed)
    pair was actually seen during calibration, the same (prompt, seed)
    granularity run_sweep.py's own r_unc-drift guard already keys on. Not
    optional: a stats entry that can't say which seeds produced it can't
    later be checked against a specific run, which is the whole point of
    keeping it.
    """
    reward_name: str
    prompt_key: str
    family: str                # "A" or "B"
    n_samples: int
    seeds: List[int]
    mean: Optional[float] = None
    std: Optional[float] = None
    rms: Optional[float] = None
    computed_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    @staticmethod
    def compute(
        reward_name: str,
        prompt_key: str,
        values: List[float],
        seeds: List[int],
        family_b_scale_fn: Callable[[List[float]], float] = default_family_b_scale,
    ) -> "Stats":
        if not values:
            raise ValueError(
                f"Stats.compute({reward_name!r}, {prompt_key!r}): empty "
                f"values list -- refusing to record a stats entry with no "
                f"samples behind it."
            )
        if len(seeds) != len(values):
            raise ValueError(
                f"Stats.compute({reward_name!r}, {prompt_key!r}): "
                f"{len(values)} value(s) but {len(seeds)} seed(s) -- these "
                f"must be parallel lists (seeds[i] is the seed that produced "
                f"values[i]). A length mismatch here would silently "
                f"misattribute a value to the wrong seed in the recorded "
                f"stats -- same class of alignment bug the Step 1 mu_trace "
                f"work was careful to guard against."
            )
        family = reward_family(reward_name)
        n = len(values)
        if family == "A":
            mean = sum(values) / n
            var = sum((v - mean) ** 2 for v in values) / n
            std = math.sqrt(var)
            if std == 0.0:
                raise ValueError(
                    f"Family A reward {reward_name!r} at prompt_key="
                    f"{prompt_key!r}: std=0 over {n} sample(s) -- every "
                    f"calibration sample scored identically, so (r-mean)/std "
                    f"is 0/0. Not clamped: this means the calibration "
                    f"samples are degenerate (e.g. n=1, or the reward model "
                    f"is not actually varying with the image), and hiding it "
                    f"behind a fallback would silently launder that."
                )
            _check_low_spread(reward_name, prompt_key, std)
            return Stats(reward_name, prompt_key, "A", n, list(seeds), mean=mean, std=std)
        else:
            rms = family_b_scale_fn(values)
            if abs(rms) <= _FAMILY_B_ZERO_SCALE_ATOL:
                raise ValueError(
                    f"Family B reward {reward_name!r} at prompt_key="
                    f"{prompt_key!r}: sqrt(E[x^2])={rms:.3g} over {n} "
                    f"sample(s) -- Family B's normalize = r / scale divides "
                    f"by (approximately) zero. This is §3.7's LPIPS landmine "
                    f"(x_ref = the same-seed uncontrolled output, so LPIPS "
                    f"is exactly 0 on uncontrolled calibration samples by "
                    f"construction) and may also apply to `style` if its "
                    f"calibration images happen to already match the "
                    f"reference. DO NOT paper over this -- per the plan, "
                    f"pick one of: (a) exclude this reward from "
                    f"auto-normalization, (b) redefine the calibration "
                    f"reference so it isn't structurally zero, or (c) "
                    f"normalize against CONTROLLED samples at some fixed "
                    f"lambda instead of uncontrolled ones. This is a design "
                    f"decision, not something to make silently here."
                )
            _check_low_spread(reward_name, prompt_key, rms)
            return Stats(reward_name, prompt_key, "B", n, list(seeds), rms=rms)

    def normalize(self, r: float) -> float:
        if self.family == "A":
            return (r - self.mean) / self.std
        return r / self.rms


# ── Persistence: JSON on disk, keyed by (reward_name, prompt|GLOBAL) ────────
class StatsStore:
    """Loads/saves the calibration stats JSON. Per 3.5: write once, reload at
    run time -- "I would rather that we don't have to do this before every
    whenever we change the prompt." Overwriting an existing entry requires
    `overwrite=True` explicitly; the default raises, so a second calibration
    pass can't silently clobber the first without saying so.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._entries: Dict[str, Stats] = {}
        if self.path.exists():
            self._load()

    @staticmethod
    def _key(reward_name: str, prompt_key: str) -> str:
        return f"{reward_name}::{prompt_key}"

    def _load(self) -> None:
        raw = json.loads(self.path.read_text())
        for k, v in raw.items():
            self._entries[k] = Stats(**v)

    def save(self) -> None:
        raw = {k: asdict(v) for k, v in self._entries.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(raw, indent=2, sort_keys=True))

    def put(self, stats: Stats, overwrite: bool = False) -> None:
        k = self._key(stats.reward_name, stats.prompt_key)
        if k in self._entries and not overwrite:
            raise ValueError(
                f"StatsStore already has an entry for {k!r} "
                f"(n_samples={self._entries[k].n_samples}, computed_at="
                f"{self._entries[k].computed_at!r}). Pass overwrite=True to "
                f"replace it deliberately -- silent reclobbering is exactly "
                f"what per-run recomputation (3.5) exists to avoid."
            )
        self._entries[k] = stats

    def get(self, reward_name: str, prompt_key: str) -> Optional[Stats]:
        return self._entries.get(self._key(reward_name, prompt_key))

    def prompts_for(self, reward_name: str) -> List[str]:
        """All non-GLOBAL prompt_keys recorded for this reward."""
        return [
            s.prompt_key for k, s in self._entries.items()
            if s.reward_name == reward_name and s.prompt_key != GLOBAL_KEY
        ]


def compute_global_from_pooled(
    reward_name: str,
    per_prompt_values: Dict[str, List[float]],
    per_prompt_seeds: Dict[str, List[int]],
    family_b_scale_fn: Callable[[List[float]], float] = default_family_b_scale,
) -> Stats:
    """GLOBAL stats = pool the RAW values across all calibration prompts,
    then compute stats on the pooled list -- NOT an average of already-
    averaged per-prompt means/stds. Documented rather than assumed: pooling
    raw samples is the standard construction of a combined mean/std/rms and
    gives every individual sample equal weight regardless of which prompt it
    came from, whereas averaging per-prompt statistics would silently
    up-weight prompts calibrated with fewer samples. Either choice is
    defensible; this is the one made, recorded so it
    isn't re-derived differently later.

    `per_prompt_seeds` must have the same keys as `per_prompt_values`, each
    list parallel to (same length, same order as) its values list -- pooled
    the same way, so the resulting GLOBAL Stats.seeds stays index-aligned
    with its values internally (per-prompt seed provenance is what actually
    matters for the r_unc consistency check this is for; GLOBAL's pooled
    list is a concatenation across prompts, not itself prompt-specific).
    """
    if set(per_prompt_values) != set(per_prompt_seeds):
        raise ValueError(
            f"compute_global_from_pooled({reward_name!r}): per_prompt_values "
            f"and per_prompt_seeds have different prompt keys -- "
            f"{set(per_prompt_values) ^ set(per_prompt_seeds)} present in "
            f"only one of the two."
        )
    pooled_values: List[float] = []
    pooled_seeds: List[int] = []
    for prompt in per_prompt_values:
        vals, seeds = per_prompt_values[prompt], per_prompt_seeds[prompt]
        if len(vals) != len(seeds):
            raise ValueError(
                f"compute_global_from_pooled({reward_name!r}, {prompt!r}): "
                f"{len(vals)} value(s) but {len(seeds)} seed(s)."
            )
        pooled_values.extend(vals)
        pooled_seeds.extend(seeds)
    return Stats.compute(reward_name, GLOBAL_KEY, pooled_values, pooled_seeds, family_b_scale_fn)


# ── The Normalizer interface Step 4's builder consumes ──────────────────────
@dataclass
class Normalizer:
    """A bound, callable normalizer: normalizer(r) -> normalized r.

    This is the exact object Step 4.1 wants "injectable rather than
    hardcoded" into a constraint spec. Binding mode is fixed at construction
    (build_normalizer below), not mutable afterward -- rebinding to a new
    prompt means building a new Normalizer, never reassigning one in place,
    so a stale binding can't silently survive a prompt change.
    """
    reward_name: str
    bind_mode: BindMode
    prompt: Optional[str]     # only meaningful when bind_mode == "prompt"
    _stats: Stats

    def __call__(self, r: float) -> float:
        return self._stats.normalize(r)

    def __repr__(self) -> str:
        where = f"prompt={self.prompt!r}" if self.bind_mode == "prompt" else "GLOBAL"
        return f"Normalizer({self.reward_name!r}, {where}, family={self._stats.family})"


def build_normalizer(
    store: StatsStore,
    reward_name: str,
    bind_mode: BindMode,
    prompt: Optional[str] = None,
) -> Normalizer:
    """Build a Normalizer against a loaded StatsStore.

    RAISES if the required entry is absent -- never falls back from
    bind_mode="prompt" to GLOBAL silently, and never fabricates stats. A
    missing entry means the calibration pass (Step 3.3, still queued behind
    GPU) has not covered this (reward, prompt) yet; that has to be visible,
    not quietly patched over by reusing pooled stats the plan never said
    were an acceptable substitute for a specific prompt.
    """
    if bind_mode == "prompt":
        if prompt is None:
            raise ValueError("build_normalizer: bind_mode='prompt' requires `prompt`.")
        stats = store.get(reward_name, prompt)
        if stats is None:
            raise ValueError(
                f"No calibration stats for reward={reward_name!r} at "
                f"prompt={prompt!r}. bind_mode='prompt' does not fall back "
                f"to GLOBAL -- either calibrate this prompt (Step 3.3) or "
                f"build with bind_mode='global'."
            )
        return Normalizer(reward_name, "prompt", prompt, stats)
    elif bind_mode == "global":
        stats = store.get(reward_name, GLOBAL_KEY)
        if stats is None:
            raise ValueError(
                f"No GLOBAL calibration stats for reward={reward_name!r}. "
                f"Run the Step 3.3 calibration pass first."
            )
        return Normalizer(reward_name, "global", None, stats)
    else:
        raise ValueError(f"bind_mode must be 'prompt' or 'global', got {bind_mode!r}.")


# ── 3.6 cross-prompt diagnostic ─────────────────────────────────────────────
# Reports the spread; does NOT decide "substantially higher" for the caller.
# The plan reserves that judgment for a human ("flag it rather than silently
# averaging") -- this prints/returns the numbers and a clearly-labeled
# HEURISTIC flag, not a hard gate that changes behavior on its own.
_SPREAD_HEURISTIC_THRESHOLD = 1.0   # in units of the pooled GLOBAL scale; adjustable

# ── Headroom check (added 2026-08-25, on review) ────────────────────────────
# A DIFFERENT failure from the low-spread check above: a healthy std does not
# by itself guarantee there is room to push the reward further before hitting
# a hard bound the reward literally cannot exceed. Concrete trigger: a
# generation prompt whose uncontrolled samples already score near the top of
# a Family A reward's plausible range (e.g. an already-cheerful-looking scene
# scored against an "upbeat" edit_prompt) has little room left for guidance to
# improve it further, which breaks Step 10.2's threshold-anchoring
# ("~1 std, expect success ... break somewhere around 2") the same way a
# small std does, but for a different reason -- the ceiling, not the noise.
#
# FAMILY A ONLY. Family B's "ceiling" (the upper bound of REWARD_OBSERVED_RANGE,
# always 0.0 for style/lpips) is only ever approached when rms is small -- and
# rms IS the reported "spread" for Family B (there is no separate mean to be
# near the bound independently of the spread, since Family B has no mean
# subtraction). So a near-zero rms already trips the EXISTING low-spread
# check at Stats.compute() time; a separate headroom check for Family B would
# just be re-detecting the same condition under a different name.
#
# THRESHOLD: flag when the room remaining to either bound is under
# _HEADROOM_STD_MULTIPLE standard deviations. 2.0 is used elsewhere in this
# codebase's own plausibility reasoning as a "notably close" (not yet
# "provably impossible", which used 6 sigma in run_sweep.py's ImageReward
# bound) cutoff -- here it means: if the mean were perturbed upward by
# guidance, a value within ~2 std of where it already sits would run into the
# bound, which is a small, easily reached perturbation relative to the
# calibrated spread itself.
_HEADROOM_STD_MULTIPLE = 2.0


def _headroom_flag(stats: Stats, lo: float, hi: float) -> Optional[str]:
    """Family A only (see module comment above for why B doesn't need this).
    Returns "near_upper_bound", "near_lower_bound", or None.
    """
    if stats.family != "A":
        return None
    room_above = hi - stats.mean
    room_below = stats.mean - lo
    if room_above < _HEADROOM_STD_MULTIPLE * stats.std:
        return "near_upper_bound"
    if room_below < _HEADROOM_STD_MULTIPLE * stats.std:
        return "near_lower_bound"
    return None


def cross_prompt_spread_report(store: StatsStore, reward_name: str) -> dict:
    """Return {prompt: {"center": ..., "spread": ...}} for every calibrated
    prompt (center = mean for Family A, rms for Family B; spread = std for
    Family A, rms for Family B -- Family B's rms plays both roles, since it
    has no separate mean-subtracted spread concept), plus the GLOBAL entry
    and a `flagged` list of prompts whose CENTER differs from GLOBAL's by
    more than _SPREAD_HEURISTIC_THRESHOLD pooled-scale units. `flagged` is
    informational -- nothing in this module changes behavior based on it.

    The `spread` value is reported for visibility only: any Stats object
    that reached this store already passed Stats.compute()'s low-spread
    RAISE (see _check_low_spread) at calibration time, which is the actual
    enforcement point -- this report cannot flag a low-spread reward that
    already exists in the store, because a low-spread one would have raised
    before it was ever stored. This is deliberate: "raise loudly when
    std/rms is small" is implemented as an immediate calibration-time gate
    (fires during Step 3.3, the moment the data is produced), not as a
    report-time flag that could go unread until Step 10's threshold sweep
    already produced uninterpretable results.

    Caveat verified while testing this (2026-08-25): with exactly TWO
    equal-sized calibration prompts, a prompt's deviation from the pooled
    mean divided by the pooled std sits at (very close to) exactly 1.0
    regardless of how far apart the two prompts actually are -- the pooled
    std is itself inflated by the same separation being measured, so the
    ratio self-normalizes toward 1 in that specific symmetric case. Not a
    bug; a property of measuring spread against a std that already includes
    the thing being flagged. With 3+ prompts, or unequal prompt weights, this
    degeneracy does not occur (verified: a genuine outlier among 3+ prompts
    is flagged; the non-outliers are not). Calibrate 3+ prompts if this
    diagnostic needs to be trustworthy, not exactly 2.
    """
    global_stats = store.get(reward_name, GLOBAL_KEY)
    if global_stats is None:
        raise ValueError(f"cross_prompt_spread_report: no GLOBAL stats for {reward_name!r}.")
    scale = global_stats.std if global_stats.family == "A" else global_stats.rms
    global_center = global_stats.mean if global_stats.family == "A" else global_stats.rms
    lo, hi = observed_range(reward_name)

    per_prompt = {}
    flagged = []
    headroom_flagged = {}
    for p in store.prompts_for(reward_name):
        s = store.get(reward_name, p)
        center = s.mean if s.family == "A" else s.rms
        spread = s.std if s.family == "A" else s.rms
        per_prompt[p] = {"center": center, "spread": spread, "n_samples": s.n_samples}
        if scale and abs(center - global_center) / scale > _SPREAD_HEURISTIC_THRESHOLD:
            flagged.append(p)
        hf = _headroom_flag(s, lo, hi)
        if hf is not None:
            headroom_flagged[p] = hf

    return {
        "reward_name": reward_name,
        "global": {"center": global_center, "spread": scale, "n_samples": global_stats.n_samples},
        "per_prompt": per_prompt,
        "flagged": flagged,
        "heuristic_threshold_in_pooled_scale_units": _SPREAD_HEURISTIC_THRESHOLD,
        # {prompt: "near_upper_bound" | "near_lower_bound"} -- Family A only
        # (see _headroom_flag's module comment for why Family B doesn't need
        # its own entry here). A prompt appearing here has limited room left
        # to move in that direction regardless of how healthy its spread is
        # -- a DIFFERENT risk from `flagged` above (which is about prompts
        # disagreeing with each other) and from Stats.compute()'s low-spread
        # raise (which is about noise amplification, not ceiling proximity).
        "headroom_flagged": headroom_flagged,
    }
