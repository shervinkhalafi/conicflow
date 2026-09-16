"""
reward_ranges.py -- shared reward plausibility ranges (raw/native scale, BEFORE
any Step 3 normalization).

Single source of truth used by BOTH:
  - run_sweep.py's per-run plausibility guard (catches an isfinite-but-wrong
    reward value -- see the 2026-08-21 contamination audit note preserved
    below, which this dict's per-reward comments are lifted from verbatim).
  - reward_normalization.py's low-spread check (needs a reward's own nominal
    range to judge whether an observed calibration std/rms is "too small
    relative to" it -- see that module for the justification of the exact
    fraction used).

Extracted 2026-08-25 into its own module rather than duplicated across the
two files above, specifically to avoid the two-copies-drift failure class
this session has repeatedly found and fixed (the Step 1 merge itself; the
three independent guard call sites unified into run_sweep.py's
run_preflight()). reward_normalization.py is deliberately lightweight (no
torch/transformers/diffusers imports) so it can be imported by a future
calibration script without pulling in FLUX -- this module has zero
dependencies for the same reason, and run_sweep.py imports it too rather
than keeping its own copy.
"""

# native scale of run_sweep.py's r_image, i.e. the reward AFTER
# _build_reward_fn's own reward-vs-loss conversion -- not the *Reward
# classes' raw "loss". clip/hpsv2/pickscore are exact analytic bounds: clip
# and hpsv2 are raw cosine similarities of two L2-normalized vectors -> [-1,1];
# pickscore is logit_scale.exp()*cos_sim, and logit_scale.exp() was read
# directly off the loaded checkpoint (2026-08-21): exp(4.59375) =
# 98.8644790649414. imagereward has no hard analytic bound -- it's a
# z-scored regression output (checkpoint mean=0.16717362830052426,
# std=1.0333394966054072, read 2026-08-21); [-6,6] is a probabilistic
# 6-sigma cutoff (~1 - 2e-9 mass under a roughly-Gaussian assumption), not a
# proof, chosen to bias against false positives while still catching real
# breakage. Added after the 2026-08-21 contamination audit found a
# wrong-but-finite reward value (Category A) that an isfinite-only guard
# does not catch.
REWARD_PLAUSIBLE_RANGE = {
    "clip":        (-1.0, 1.0),
    "hpsv2":       (-1.0, 1.0),
    "pickscore":   (-98.8644790649414, 98.8644790649414),
    "imagereward": (-6.0, 6.0),
    # style: MPGD-faithful -- raw, UNnormalized Gram matrices (F^T F, no
    # 1/(C*H*W) or L2 scaling), so an earlier [-2,0] bound (which only held
    # under L2-normalized tokens) does not apply. Upper bound 0.0 IS exact
    # and provable regardless of normalization -- r_style =
    # -||G_gen - G_ref||_F and a Frobenius norm is never negative. Lower
    # bound -5000.0 is NOT a proof, unlike clip/hpsv2/pickscore's exact
    # bounds -- it's empirical, from n=4 real image pairs (2026-08-21):
    # observed ||G1-G2||_F ranged 0 (identical) to ~863 (very different
    # scenes, portrait vs. sunset). -5000 leaves ~6x headroom over the
    # largest observed value on a tiny sample, same spirit as imagereward's
    # probabilistic bound but with much less data behind it -- revisit once
    # more images are scored, same as imagereward's guard.
    "style":       (-5000.0, 0.0),
    # lpips: r = -LPIPS(gen, x_ref). Upper bound 0.0 IS exact and provable --
    # LPIPS is a sum of squared-difference terms over normalized network
    # activations, never negative, so -LPIPS <= 0 always (verified on CPU
    # 2026-08-22: LPIPS(x,x) == 0.0 exactly, net='alex'). Lower bound -2.0 is
    # NOT a proof, unlike the exact upper bound -- borrowed-empirical from a
    # DIFFERENT model/dataset (HardFlow Table V reports 0.026-0.144 for
    # their editing task, not ours) with ~14-77x headroom over that range,
    # same spirit as style's own n=4 empirical guard but weaker still since
    # it isn't even our own measurement yet.
    "lpips":       (-2.0, 0.0),
    # none: the "none" branch of _build_reward_fn returns literal
    # z.new_zeros(...), so (0.0, 0.0) is EXACT and provable -- not an
    # empirical or probabilistic bound like style/lpips/imagereward above.
    # Present so that --reward none (the mode Step 7 and Step 10 run in,
    # where steering is entirely via the constraints) does not KeyError out
    # of the plausibility guard in _run_one.
    "none":        (0.0, 0.0),
}


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVED (attainable) ranges -- a DIFFERENT question from the dict above
# ═══════════════════════════════════════════════════════════════════════════
# Added 2026-08-26. REWARD_PLAUSIBLE_RANGE above answers "is this value
# WRONG?" and therefore must use analytic/provable bounds: a CLIP cosine of
# 1.5 is impossible, while 0.5 is merely unusual, so [-1, 1] is exactly
# right for that guard and is left untouched.
#
# But two OTHER consumers were reading the same dict to answer a different
# question -- "is this calibration spread small relative to the range this
# reward actually varies over?" -- namely reward_normalization.py's
# _check_low_spread (divides the observed scale by the range width) and
# _headroom_flag (asks whether the mean sits within 2 std of a bound). For
# those, an analytic bound the reward never approaches is the wrong
# denominator, and it distorted both checks in opposite, misleading
# directions:
#
#   * _check_low_spread became too STRICT. CLIP's analytic width is 2.0 but
#     its measured attainable width is ~0.77, i.e. 2.6x narrower. Row 1's
#     CLIP calibration (std 0.0246) therefore reported 1.23% of range and
#     "nearly tripped" the 1% floor, when against the range CLIP actually
#     occupies it is ~3.1% -- comfortably clear. The near-miss was an
#     artifact of the denominator, not a property of the calibration.
#   * _headroom_flag became near-VACUOUS. With CLIP's analytic ceiling of
#     1.0 -- a value no real image approaches -- room-to-ceiling is always
#     enormous, so the check could essentially never fire for CLIP no matter
#     how close to its real ceiling a prompt sat. Row 1's reported "no
#     headroom flag" was thus not meaningful evidence of headroom.
#
# Hence the split. Same single-source-of-truth discipline as before (one
# module, imported by both consumers), just two dicts because there are two
# questions.
#
# EVERY entry is labeled measured-vs-assumed in REWARD_OBSERVED_PROVENANCE
# below, and observed_range() refuses to return an unlabeled entry -- so a
# guard that fires can always be traced to the quality of the number it
# fired against, which is the whole point of doing this before rows 2-3 are
# scored rather than after.
#
# DATA SOURCE for the "measured" entries: all 232 historical
# results/*/summary.csv files (2791 rows), pooling r_unc AND r_ctrl, since
# the attainable range must include what guidance can actually reach (Step
# 10.2 anchors thresholds ABOVE r_unc). Classified by value band, which is
# unambiguous here because the reward scales are disjoint by orders of
# magnitude; directory names were NOT used for classification because they
# proved unreliable (the phaseS5b_* style runs do not contain "style" in
# their names). 54 values in two bands (-71..-44, and 1.4..1.9) could not be
# attributed to a reward type with confidence and were EXCLUDED rather than
# folded in.
REWARD_OBSERVED_RANGE = {
    # MEASURED, strong. n=5453 pooled values, observed [0.00116, 0.77530];
    # rounded outward to (0.0, 0.80). Independently corroborates
    # pre-meeting §1's "unconstrained CLIP 0.276-0.339 across prompts" and
    # its stated ~0.12-0.42 uncontrolled span -- this measurement reproduces
    # that from the CSVs without having been fitted to it.
    "clip":        (0.0, 0.80),
    # ASSUMED. No usable data: hpsv2 is EXCLUDED from this project by locked
    # decision, so nothing here has ever calibrated it. Left at the analytic
    # cosine bound purely as a placeholder; do not trust a guard that fires
    # against this entry without measuring first.
    "hpsv2":       (-1.0, 1.0),
    # MEASURED but WEAK: n=4 only, observed [19.25889, 25.17626], widened to
    # (18.0, 27.0). Enough to establish the scale (~20, nowhere near the
    # analytic +-98.86) but not the true extremes.
    "pickscore":   (18.0, 27.0),
    # MIXED -- read the asymmetry before trusting a headroom result here.
    # r_image for imagereward returns the Z-SCORE itself: ImageRewardReward
    # .forward gives loss = 2 - z, and run_sweep.py's closure returns
    # 2.0 - loss = z. So its population std is 1.0 BY CONSTRUCTION, which is
    # why row 1's n=100 std came back 1.0041 -- that agreement is a design
    # property, not a lucky draw, and it is the strongest evidence we have
    # that the imagereward path is wired correctly.
    #   upper +1.9: MEASURED, but n=2 (max observed 1.87726).
    #   lower -4.3: STATISTICAL, not min/max -- row 1's n=100 mean (-1.2630)
    #     minus 3*std (1.0041). Row 1's raw per-sample dump does not exist on
    #     disk yet (it predates raw-value dumping), so a true observed
    #     minimum is not available until that rerun happens.
    # CONSEQUENCE, stated rather than buried: because the lower bound is
    # itself mean-3*std, _headroom_flag's near_lower_bound test (mean within
    # 2*std of the bound) CANNOT fire for imagereward by construction -- it
    # is weak-by-design for this reward until the raw dump supplies a real
    # minimum. Do not read "not flagged" as evidence of headroom for
    # imagereward specifically.
    "imagereward": (-4.3, 1.9),
    # MIXED, and the honest combination: ceiling PROVABLE, floor MEASURED.
    # 0.0 is exact (r_style = -||G_gen - G_ref||_F, zero iff the Gram
    # matrices match). Floor -1000.0 from n=34 pooled values, observed min
    # -980.29865 -- vs. the analytic dict's -5000.0, which was an n=4 guess
    # with 6x headroom and is 5x too wide. Note Family B does not use
    # _headroom_flag at all, so this entry only affects the low-spread
    # denominator.
    "style":       (-1000.0, 0.0),
    # MEASURED, and it independently CONFIRMS §3.7's landmine from data
    # rather than from reasoning: every historical uncontrolled lpips value
    # is exactly 0.0 (x_ref IS the same-seed uncontrolled output), and the
    # controlled ones span only [-0.00804, 0.0]. n=10. The analytic (-2.0,
    # 0.0) entry is ~250x too wide -- against it, essentially ANY real lpips
    # calibration would trip the 1% low-spread floor for the wrong reason.
    "lpips":       (-0.01, 0.0),
    # EXACT, and unreachable in practice: "none" has no FAMILY_A/FAMILY_B
    # assignment, so reward_family() raises before either consumer of this
    # dict is ever called with it. Present only for symmetry with the dict
    # above. Zero width -- observed_range() rejects it explicitly rather
    # than letting a caller divide by zero.
    "none":        (0.0, 0.0),
}

# Per-entry provenance. Deliberately a separate REQUIRED dict rather than an
# optional third tuple element: a missing label is then a KeyError at import
# time, not a silently-unlabeled number.
REWARD_OBSERVED_PROVENANCE = {
    "clip":        "MEASURED n=5453 (historical r_unc+r_ctrl; observed [0.00116, 0.77530])",
    "hpsv2":       "ASSUMED (no data; hpsv2 excluded from this project by locked decision)",
    "pickscore":   "MEASURED but WEAK n=4 (observed [19.25889, 25.17626])",
    "imagereward": ("MIXED: upper MEASURED n=2 (max 1.87726); lower STATISTICAL "
                    "from row-1 n=100 (mean -1.2630 - 3*std 1.0041), NOT an observed "
                    "minimum -- so near_lower_bound cannot fire by construction"),
    "style":       ("MIXED: ceiling 0.0 PROVABLE (Frobenius norm); floor MEASURED "
                    "n=34 (observed min -980.29865)"),
    "lpips":       ("MEASURED n=10 (observed [-0.00804, 0.0]; every uncontrolled "
                    "value exactly 0.0, empirically confirming the §3.7 landmine)"),
    "none":        "EXACT (literal zeros); unreachable -- no Family A/B assignment",
}

_UNLABELED = set(REWARD_OBSERVED_RANGE) ^ set(REWARD_OBSERVED_PROVENANCE)
if _UNLABELED:
    raise ImportError(
        f"reward_ranges.py: {sorted(_UNLABELED)} appear in exactly one of "
        f"REWARD_OBSERVED_RANGE / REWARD_OBSERVED_PROVENANCE. Every observed "
        f"range must carry an explicit measured-vs-assumed label -- an "
        f"unlabeled range is how a guard fires against a number nobody can "
        f"trace."
    )
del _UNLABELED


def observed_range(reward_name: str) -> tuple:
    """(lo, hi) attainable range for spread/headroom judgments.

    Use this -- NOT REWARD_PLAUSIBLE_RANGE -- wherever the question is "how
    big is this spread relative to how much the reward actually moves".
    Raises on an unknown reward and on a zero-width range, rather than
    letting a caller divide by zero.
    """
    if reward_name not in REWARD_OBSERVED_RANGE:
        raise KeyError(
            f"no observed range for reward {reward_name!r}. Add one to "
            f"reward_ranges.REWARD_OBSERVED_RANGE together with its "
            f"REWARD_OBSERVED_PROVENANCE label -- do not fall back to "
            f"REWARD_PLAUSIBLE_RANGE, which answers a different question "
            f"(is this value wrong?) and distorts both the low-spread and "
            f"headroom checks when used here."
        )
    lo, hi = REWARD_OBSERVED_RANGE[reward_name]
    if hi - lo <= 0.0:
        raise ValueError(
            f"observed range for {reward_name!r} has zero/negative width "
            f"({lo}, {hi}) -- cannot be used as a spread denominator. "
            f"Provenance: {REWARD_OBSERVED_PROVENANCE[reward_name]}"
        )
    return (lo, hi)
