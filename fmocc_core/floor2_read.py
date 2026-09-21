"""Read results/floor2 and emit the seed ranking plus THRS_A / THRS_B.

    python floor2_read.py

FLOOR is measured today (r_unc under the new base prompts). CEILING is carried
over from the 09-20 unconstrained screen, because the edit prompts did not
change and the reachable CLIP is mostly a property of the arm's control
authority at fixed N and lambda. That carry-over is the one approximation
this script makes relative to the full probe2.sh, and it is stated in the
output rather than buried here.

SEEDS are ranked by the WORST (max) r_unc across the pair's two edit prompts:
a seed that starts far from "curly" but already looks old is no good, and the
max is what catches that. Lower is better -- it is Shervin's "someone who is
not already curly, not already old" criterion, measured.
"""
import csv
import glob
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, 'results', 'floor2')
if not os.path.isdir(ROOT):
    sys.exit('no results/floor2 under %s -- run floor2.sh first' % HERE)

BASE = {'A': 'a portrait of a young man with straight hair',
        'B': 'a portrait of a serious man with dark hair'}
EDITS = {'A': ('a face with curly hair', 'an old face'),
         'B': ('a smiling face', 'a face with white hair')}

# Measured 09-20 (results/cprep/B_*), reachable CLIP with that prompt as the
# REWARD. See crun.sh's header for provenance.
CEIL_HF = {'A': 0.487, 'B': 0.370}
CEIL_MPC = {'A': 0.553, 'B': 0.463}

# (prompt, edit) -> seed -> r_unc
vals = defaultdict(dict)
for f in glob.glob(os.path.join(ROOT, 's*', 'summary.csv')):
    for r in csv.DictReader(open(f)):
        if (r.get('run_ok') or '').lower() != 'true':
            continue
        try:
            vals[(r['prompt'].strip(), r['edit_prompt'].strip())][int(r['seed'])] \
                = float(r['r_unc'])
        except (TypeError, ValueError, KeyError):
            continue

if not vals:
    sys.exit('floor2 produced no run_ok rows -- check results/floor2/log_*.txt')

print('=' * 72)
print('SEED RANKING -- r_unc = CLIP(uncontrolled image, edit prompt)')
print('LOWER IS BETTER: the base image does not already have the attribute.')

best = {}
for pair in ('A', 'B'):
    e0, e1 = EDITS[pair]
    d0 = vals.get((BASE[pair], e0), {})
    d1 = vals.get((BASE[pair], e1), {})
    seeds = sorted(set(d0) & set(d1))
    if not seeds:
        print('\npair %s: no rows' % pair)
        continue
    print('\npair %s   base = "%s"' % (pair, BASE[pair]))
    print('  %-6s %-22s %-22s %-8s' % ('seed', e0[:21], e1[:21], 'worst'))
    rank = sorted((max(d0[s], d1[s]), s, d0[s], d1[s]) for s in seeds)
    for worst, s, a, b in rank:
        print('  %-6d %-22.4f %-22.4f %-8.4f' % (s, a, b, worst))
    print('  -> seed order: %s' % ' '.join(str(s) for _, s, _, _ in rank))
    best[pair] = rank

print('\n' + '=' * 72)
print('TAU GRID   (floor measured today; ceiling carried over from 09-20)')
for pair in ('A', 'B'):
    if pair not in best:
        continue
    # Floor across the seeds we would actually use: the grid has to clear the
    # highest r_unc among them, or that seed's constraint starts satisfied.
    floor = max(w for w, _, _, _ in best[pair])
    hf, mpc = CEIL_HF[pair], CEIL_MPC[pair]
    print('\npair %s:  floor(max r_unc over seeds) = %.4f   HF ceil %.3f   MPC ceil %.3f'
          % (pair, floor, hf, mpc))
    lo, hi = floor + 0.02, min(hf * 1.08, mpc)
    if hi <= lo:
        print('  !! ceiling %.4f is at or below floor+0.02 = %.4f.' % (hi, lo))
        print('     No informative tau: the uncontrolled image already scores')
        print('     as high as control can reach. Change the prompt rather')
        print('     than widening the grid -- and tell me, this needs a call.')
        continue
    grid = [lo + (hi - lo) * i / 3.0 for i in range(4)]
    print('  export THRS_%s="%s"' % (pair, ' '.join('%.3f' % g for g in grid)))
    n_below = sum(1 for g in grid if g < hf)
    print('  (HardFlow should start failing around row %d of 4)' % (n_below + 1))

print('\nNOTE: ceilings are from runs where CLIP was the REWARD. Here it is')
print('only a CONSTRAINT and LPIPS pulls against it, so real failures arrive')
print('EARLIER than the ceiling suggests -- the grid is placed low on purpose.')
