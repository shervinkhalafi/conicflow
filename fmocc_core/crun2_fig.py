"""One multi-row figure per (pair, seed): rows = tau, columns =
uncontrolled | MPC | MPC+adaptive | HardFlow.

    python crun2_fig.py            # every (pair, seed) on disk
    python crun2_fig.py A 42       # just one

Shervin's 09-20 request was uncontrolled | MPC | HardFlow; this is the same
layout with the adaptive-dual arm inserted as a third column, so his
comparison still reads left to right unchanged.

Rows are paired to images by the `tag` column (see crun2_lib), so the two MPC
arms sharing one directory is not a hazard and a --resume index restart is not
either. A tau missing from any arm is skipped rather than rendered with a gap.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from crun2_lib import ARM_ORDER, ARM_LABEL, PAIR_TXT, discover, load_pair_seed

HERE = os.path.dirname(os.path.abspath(__file__))
CRUN = os.path.join(HERE, 'results', 'crun2')


def build(pair, seed):
    arms, problems = load_pair_seed(CRUN, pair, seed)
    for p in problems:
        print('  !! pair %s s%d -- %s' % (pair, seed, p))
    taus = sorted(set(arms['mpc']) & set(arms['mpcad']) & set(arms['hardflow']))
    if not taus:
        print('  SKIP pair%s s%d -- no tau present in all three arms '
              '(mpc %d, mpcad %d, hardflow %d)'
              % (pair, seed, len(arms['mpc']), len(arms['mpcad']),
                 len(arms['hardflow'])))
        return

    n, ncol = len(taus), 1 + len(ARM_ORDER)
    fig, ax = plt.subplots(n, ncol, figsize=(3.4 * ncol, 4.1 * n), squeeze=False)
    for i, t in enumerate(taus):
        unc = arms['mpc'][t]['img_unc']
        if unc:
            ax[i][0].imshow(Image.open(unc))
        ax[i][0].set_title('uncontrolled', fontsize=10)
        ax[i][0].set_xlabel('LPIPS = 0 by construction', fontsize=8)
        ax[i][0].set_ylabel('tau = %.3f' % t, rotation=0, ha='right',
                            va='center', fontsize=13, fontweight='bold')
        for col, arm in enumerate(ARM_ORDER, start=1):
            r = arms[arm][t]
            ax[i][col].imshow(Image.open(r['img_ctrl']))
            # feasibility stated in words, never carried by colour alone
            ax[i][col].set_title(
                '%s  [%s]' % (ARM_LABEL[arm],
                              'feasible' if r['feas'] else 'INFEASIBLE'),
                fontsize=10, color='black' if r['feas'] else 'firebrick',
                fontweight='bold' if r['feas'] else 'normal')
            ax[i][col].set_xlabel(
                'J=%.4f   E=%.3g   LPIPS=%.4f\nviol = (%+.4f, %+.4f)'
                % (r['J'], r['E'], r['lpips'], r['v0'], r['v1']), fontsize=8)
        for a in ax[i]:
            a.set_xticks([])
            a.set_yticks([])

    base, edits = PAIR_TXT[pair]
    fig.suptitle(
        'Constrained editing -- pair %s,  seed %d,  lambda = 5000\n'
        'generated from "%s", constrained toward %s\n'
        'LPIPS reward + two CLIP constraints, N=8, H=2.  '
        'J = E/(2*lambda) - r_ctrl, minimised; viol <= 0 means satisfied'
        % (pair, seed, base, edits), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = os.path.join(CRUN, 'fig_pair%s_s%d.png' % (pair, seed))
    fig.savefig(out, dpi=88, bbox_inches='tight')
    plt.close(fig)
    print('  wrote %s  (%d thresholds)' % (out, n))


if not os.path.isdir(CRUN):
    sys.exit('no results/crun2 under %s' % HERE)
targets = ([(sys.argv[1], int(sys.argv[2]))] if len(sys.argv) == 3
           else discover(CRUN))
print('%d (pair, seed) combination(s)' % len(targets))
for p, s in targets:
    build(p, s)
