"""Three-way comparison for the 2026-09-21 run: MPC vs MPC+adaptive vs HardFlow.

    python crun2_analyze.py

Two separate questions, reported separately because they have different
answers and merging them hides both.

Q1 (Shervin's): does the adaptive dual help? Compare mpcad vs mpc at identical
   tau/seed/lambda. His predicted mechanism is specific -- with two
   constraints on different scales, plain dual ascent lets the larger-|h| one
   monopolise the dual pressure while the other is ignored and never comes
   down. So the mechanism check is whether the IMBALANCE |viol_0 - viol_1|
   shrinks, and the payoff check is whether that buys feasibility. They can
   disagree, and if they do that is the finding: report "it does what it
   claims, and it does not matter here" rather than calling it a win or a
   loss.

Q2 (the paper's): MPC vs HardFlow, restricted to thresholds where BOTH are
   feasible. HardFlow enforces by projection and will spend whatever energy
   that takes, so comparing energy where it failed is not like-for-like.

J = E/(2*lam) - r_ctrl is comparable throughout because lambda is fixed at
5000 across the whole run (unlike 09-20, where it was swept).
"""
import os
import sys

from crun2_lib import ARM_ORDER, ARM_LABEL, discover, load_pair_seed

HERE = os.path.dirname(os.path.abspath(__file__))
CRUN = os.path.join(HERE, 'results', 'crun2')
if not os.path.isdir(CRUN):
    sys.exit('no results/crun2 under %s' % HERE)

combos = discover(CRUN)
if not combos:
    sys.exit('no pair*_s* directories -- check results/crun2/log_*.txt')

q1, q2 = [], []
for pair, seed in combos:
    arms, problems = load_pair_seed(CRUN, pair, seed)
    for p in problems:
        print('!! pair %s seed %d -- %s' % (pair, seed, p))
    taus = sorted(set().union(*[set(v) for v in arms.values()]) or [])
    if not taus:
        continue
    print('\n== pair %s  seed %d  lam=5000' % (pair, seed))
    print('   %-7s | %s' % ('tau', ' | '.join('%-34s' % ARM_LABEL[a] for a in ARM_ORDER)))
    for t in taus:
        cells = []
        for a in ARM_ORDER:
            r = arms[a].get(t)
            cells.append('%-34s' % ('-' if r is None else
                                    '%s E=%-6.3g J=%-7.4f (%+.3f,%+.3f)'
                                    % ('FEAS' if r['feas'] else 'viol',
                                       r['E'], r['J'], r['v0'], r['v1'])))
        print('   %-7.3f | %s' % (t, ' | '.join(cells)))

        m, ad, hf = arms['mpc'].get(t), arms['mpcad'].get(t), arms['hardflow'].get(t)
        if m and ad:
            q1.append((pair, seed, t, m, ad))
        # Both MPC arms are compared against HardFlow. Which one is "the"
        # paper number depends on whether the adaptive dual is adopted, and
        # that is decided by Q1 -- so report both rather than pre-judging it.
        for arm, r in (('mpc', m), ('mpcad', ad)):
            if r and hf and r['feas'] and hf['feas']:
                q2.append((arm, pair, seed, t, r, hf))

print('\n' + '=' * 80)
print('Q1  ADAPTIVE DUAL -- mechanism, then payoff')
if not q1:
    print('  no tau where both MPC arms produced a row')
else:
    print('  %-5s %-5s %-7s %-11s %-11s %-9s %-9s %-9s %-9s' %
          ('pair', 'seed', 'tau', '|v0-v1|mpc', '|v0-v1|ad',
           'feas mpc', 'feas ad', 'J mpc', 'J ad'))
    imb = feas_gain = feas_loss = j_win = 0
    for pair, seed, t, m, ad in q1:
        imb += ad['imbalance'] < m['imbalance']
        feas_gain += (ad['feas'] and not m['feas'])
        feas_loss += (m['feas'] and not ad['feas'])
        j_win += ad['J'] < m['J']
        print('  %-5s %-5d %-7.3f %-11.4f %-11.4f %-9s %-9s %-9.4f %-9.4f' %
              (pair, seed, t, m['imbalance'], ad['imbalance'],
               m['feas'], ad['feas'], m['J'], ad['J']))
    n = len(q1)
    print('\n  MECHANISM  imbalance |v0-v1| reduced : %d / %d' % (imb, n))
    print('  PAYOFF     newly feasible (ad only)  : %d / %d' % (feas_gain, n))
    print('             feasibility LOST          : %d / %d' % (feas_loss, n))
    print('             objective J improved      : %d / %d' % (j_win, n))
    if imb > n * 0.6 and feas_gain == 0:
        print('\n  -> The heuristic is doing what it claims (imbalance down) without')
        print('     that changing feasibility here. Report it that way; it is not')
        print('     a win and not a failure.')

print('\n' + '=' * 80)
print('Q2  MPC vs HARDFLOW, thresholds where BOTH are feasible')
if not q2:
    print('  none -- no directly comparable point')
else:
    print('  %-9s %-5s %-5s %-7s %-9s %-9s %-8s %-9s %-9s' %
          ('mpc arm', 'pair', 'seed', 'tau', 'E_mpc', 'E_hf', 'HF/MPC',
           'J_mpc', 'J_hf'))
    tally = {}
    for arm, pair, seed, t, r, hf in q2:
        e, j, n = tally.get(arm, (0, 0, 0))
        tally[arm] = (e + (hf['E'] > r['E']), j + (r['J'] < hf['J']), n + 1)
        print('  %-9s %-5s %-5d %-7.3f %-9.4g %-9.4g %-8.2f %-9.4f %-9.4f' %
              (arm, pair, seed, t, r['E'], hf['E'],
               (hf['E'] / r['E'] if r['E'] else float('inf')), r['J'], hf['J']))
    print()
    for arm in ('mpc', 'mpcad'):
        if arm in tally:
            e, j, n = tally[arm]
            print('  %-20s cheaper on energy: %2d / %-3d   better on J: %2d / %d'
                  % (ARM_LABEL[arm], e, n, j, n))
