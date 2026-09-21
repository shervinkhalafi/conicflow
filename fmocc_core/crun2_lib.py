"""Shared loader for the 2026-09-21 constrained run.

Two things both the table and the figure need to get right, factored here so
they cannot drift apart:

1. THE THREE ARMS LIVE IN TWO DIRECTORIES. pair<X>_mpc_s<N>/ holds BOTH MPC
   arms -- they were run in one process via --sweep_dual_adaptive false true
   to share a model load -- and they are told apart by the dual_adaptive CSV
   column, not by path. pair<X>_hardflow_s<N>/ holds the third.

2. ROWS ARE PAIRED TO IMAGES BY THE `tag` COLUMN, never by sorted order.
   `tag` is literally the artifact filename prefix, so the mapping is exact.
   Sorted-order pairing breaks in two ways we have already hit: --resume
   restarts the run-index numbering, and here two arms interleave in one
   directory. Mis-pairing puts the wrong numbers under the wrong picture,
   which is worse than no figure -- so a row whose image is missing is
   dropped loudly rather than matched to a neighbour.
"""
import csv
import glob
import os

ARM_ORDER = ['mpc', 'mpcad', 'hardflow']
ARM_LABEL = {'mpc': 'MPC', 'mpcad': 'MPC + adaptive dual', 'hardflow': 'HardFlow'}
PAIR_TXT = {
    'A': ('a portrait of a young man with straight hair', 'curly hair + old face'),
    'B': ('a portrait of a serious man with dark hair', 'smiling + white hair'),
}


def _f(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return float('nan')


def load_pair_seed(crun, pair, seed):
    """-> (arms, problems).  arms: arm -> {tau: row-dict}, sorted by tau."""
    arms = {a: {} for a in ARM_ORDER}
    problems = []

    for sub, is_mpc in (('mpc', True), ('hardflow', False)):
        d = os.path.join(crun, 'pair%s_%s_s%s' % (pair, sub, seed))
        path = os.path.join(d, 'summary.csv')
        if not os.path.exists(path):
            problems.append('%s: no summary.csv' % sub)
            continue
        for r in csv.DictReader(open(path)):
            if (r.get('run_ok') or '').lower() != 'true' or not r.get('thr_0'):
                continue
            adaptive = (r.get('dual_adaptive') or '').lower() == 'true'
            if is_mpc:
                arm = 'mpcad' if adaptive else 'mpc'
            else:
                arm = 'hardflow'
                if adaptive:
                    # Would mean the mpc-only guard did not hold. Flag it; do
                    # not quietly fold the row into the hardflow arm.
                    problems.append('hardflow row with dual_adaptive=True (%s)'
                                    % r.get('tag'))
                    continue
            tag = (r.get('tag') or '').strip()
            ctrl = os.path.join(d, tag + '__img_ctrl.png')
            unc = os.path.join(d, tag + '__img_unc.png')
            if not tag or not os.path.exists(ctrl):
                problems.append('%s: no image for tag %r' % (sub, tag))
                continue
            v0, v1 = _f(r, 'viol_0'), _f(r, 'viol_1')
            arms[arm][round(_f(r, 'thr_0'), 4)] = dict(
                tau=_f(r, 'thr_0'), v0=v0, v1=v1,
                feas=(v0 <= 0 and v1 <= 0),
                imbalance=abs(v0 - v1),
                E=_f(r, 'ctrl_energy'), J=_f(r, 'objective'),
                lpips=abs(_f(r, 'r_ctrl')),
                img_ctrl=ctrl,
                img_unc=unc if os.path.exists(unc) else None)

    # An mpc directory that produced only one of the two arms means
    # --sweep_dual_adaptive did not take, and every adaptive-vs-vanilla
    # conclusion would then be drawn from a single arm compared with itself.
    if arms['mpc'] and not arms['mpcad']:
        problems.append('mpc dir has no dual_adaptive=True rows '
                        '-- did --sweep_dual_adaptive take?')
    if arms['mpcad'] and not arms['mpc']:
        problems.append('mpc dir has no dual_adaptive=False rows')
    return arms, problems


def discover(crun):
    """-> sorted [(pair, seed)] present on disk."""
    found = set()
    for d in glob.glob(os.path.join(crun, 'pair*_s*')):
        if not os.path.isdir(d):
            continue
        b = os.path.basename(d)
        try:
            pair = b[4]
            seed = int(b.rsplit('_s', 1)[1])
        except (IndexError, ValueError):
            continue
        if pair in PAIR_TXT:
            found.add((pair, seed))
    return sorted(found)
