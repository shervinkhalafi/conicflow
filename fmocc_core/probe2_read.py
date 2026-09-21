"""Read results/probe2 and emit (a) the seed ranking and (b) the tau grid.

    python probe2_read.py

Two questions, one table each.

SEEDS. Shervin asked for seeds "where the difference would be even more
recognisable" -- i.e. where the uncontrolled image does NOT already have the
attribute. That is exactly a low r_unc, so seeds are ranked by the WORST (max)
r_unc across the pair's two edit prompts: a seed that starts far from "curly"
but already looks old is no good, and the max is what catches that.

TAU. A tau grid is only informative between the floor and the ceiling:
  floor   = max over seeds of r_unc  -- at or below this, the constraint is
            already satisfied before any control acts, mu stays at 0, and the
            run measures nothing.
  ceiling = the arm's reachable CLIP with that prompt as the REWARD. Above it
            the run is infeasible by construction.
HardFlow's ceiling is the binding one (it is the lower of the two), so the grid
is placed to straddle it: we want rows where HardFlow fails and MPC does not,
which is the comparison the figure is for. The printed grid spans
floor+delta .. min(hf_ceiling*1.08, mpc_ceiling), 4 points.

CAVEAT, stated because it bit us on 2026-09-19: these ceilings come from runs
where CLIP is the REWARD. In the real run CLIP is only a CONSTRAINT and LPIPS
pulls against it, so genuine failures arrive EARLIER than the ceiling suggests.
The grid is therefore deliberately placed a little low rather than centred.
"""
import csv
import glob
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, 'results', 'probe2')
if not os.path.isdir(ROOT):
    sys.exit('no results/probe2 under %s -- run probe2.sh first' % HERE)

LABEL = {
    'A0': ('A', 'a face with curly hair'),
    'A1': ('A', 'an old face'),
    'B0': ('B', 'a smiling face'),
    'B1': ('B', 'a face with white hair'),
}

# tag -> arm -> seed -> (r_unc, r_ctrl)
data = defaultdict(lambda: defaultdict(dict))
for d in sorted(glob.glob(os.path.join(ROOT, '*__*__s*'))):
    m = re.match(r'(A0|A1|B0|B1)__(\w+)__s(\d+)$', os.path.basename(d))
    if not m:
        continue
    tag, arm, seed = m.group(1), m.group(2), int(m.group(3))
    p = os.path.join(d, 'summary.csv')
    if not os.path.exists(p):
        continue
    for r in csv.DictReader(open(p)):
        if (r.get('run_ok') or '').lower() != 'true':
            continue
        try:
            data[tag][arm][seed] = (float(r['r_unc']), float(r['r_ctrl']))
        except (TypeError, ValueError):
            continue

if not data:
    sys.exit('probe2 produced no run_ok rows -- check results/probe2/log_*.txt')

# ── seeds ────────────────────────────────────────────────────────────────────
print('=' * 74)
print('SEED RANKING -- r_unc = CLIP(uncontrolled, edit prompt). LOWER IS BETTER')
print('(low = the base image does not already have the attribute)')
for pair, tags in (('A', ('A0', 'A1')), ('B', ('B0', 'B1'))):
    seeds = sorted(set(data[tags[0]].get('hardflow', {})) &
                   set(data[tags[1]].get('hardflow', {})))
    if not seeds:
        print('\npair %s: no hardflow probe rows' % pair)
        continue
    print('\npair %s   base = %s' % (pair, 'young/straight' if pair == 'A' else 'serious/dark'))
    print('  %-6s %-10s %-10s %-10s' %
          ('seed', LABEL[tags[0]][1][:9], LABEL[tags[1]][1][:9], 'worst'))
    rank = []
    for s in seeds:
        a = data[tags[0]]['hardflow'][s][0]
        b = data[tags[1]]['hardflow'][s][0]
        rank.append((max(a, b), s, a, b))
    for worst, s, a, b in sorted(rank):
        print('  %-6d %-10.4f %-10.4f %-10.4f' % (s, a, b, worst))
    print('  -> use, in order: %s' % ', '.join(str(s) for _, s, _, _ in sorted(rank)))

# ── tau ──────────────────────────────────────────────────────────────────────
print('\n' + '=' * 74)
print('TAU GRID')
for pair, tags in (('A', ('A0', 'A1')), ('B', ('B0', 'B1'))):
    floors, hf_ceils, mpc_ceils = [], [], []
    for t in tags:
        hf = data[t].get('hardflow', {})
        mp = data[t].get('mpc', {})
        if hf:
            floors.append(max(v[0] for v in hf.values()))
            hf_ceils.append(max(v[1] for v in hf.values()))
        if mp:
            mpc_ceils.append(max(v[1] for v in mp.values()))
    if not floors or not hf_ceils:
        print('\npair %s: incomplete probe' % pair)
        continue
    floor = max(floors)
    hf_c = min(hf_ceils)          # binding across the pair's two constraints
    mpc_c = min(mpc_ceils) if mpc_ceils else float('nan')
    print('\npair %s:  floor(max r_unc) = %.4f   HF ceiling = %.4f   MPC ceiling = %s'
          % (pair, floor, hf_c, '%.4f' % mpc_c if mpc_ceils else 'n/a'))
    lo = floor + 0.02
    hi = hf_c * 1.08
    if mpc_ceils:
        hi = min(hi, mpc_c)
    if hi <= lo:
        print('  !! ceiling (%.4f) is at or below the floor+0.02 (%.4f).' % (hi, lo))
        print('     There is no informative tau here -- the uncontrolled image')
        print('     already scores as high as control can reach. Change the')
        print('     prompt, do not widen the grid.')
        continue
    grid = [lo + (hi - lo) * i / 3.0 for i in range(4)]
    print('  THRS_%s="%s"' % (pair, ' '.join('%.3f' % g for g in grid)))
    print('  (straddles the HF ceiling at %.4f: expect HardFlow to start '
          'failing around point %d)'
          % (hf_c, 1 + sum(1 for g in grid if g < hf_c)))
