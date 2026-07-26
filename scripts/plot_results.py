"""Plot success rate vs training steps from LwH/DDPG run logs (thesis Fig 10).

Each run directory must contain a 'uav-v0_log.txt' written by test.py or
ddpg_baseline.py. Usage:

    python plot_results.py --runs logs/lwh_prior logs/a3c_noprior logs/ddpg \
                           --labels "LwH" "A3C (no prior)" "DDPG" \
                           --out results/fig10_reproduction.png
"""
import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r'training steps (\d+), reward episode ([-\d.eE]+), success_rate ([\d.eE]+)')


def parse_log(run_dir, env='uav-v0'):
    path = os.path.join(run_dir, '{}_log.txt'.format(env))
    steps, success = [], []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                success.append(float(m.group(3)))
    return steps, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True,
                        help='run directories containing uav-v0_log.txt')
    parser.add_argument('--labels', nargs='+', default=None)
    parser.add_argument('--env', default='uav-v0')
    parser.add_argument('--out', default=os.path.join('results', 'fig10_reproduction.png'))
    args = parser.parse_args()

    labels = args.labels if args.labels else [os.path.basename(r.rstrip('\\/')) for r in args.runs]
    assert len(labels) == len(args.runs), '--labels must match --runs'

    plt.figure(figsize=(7, 5))
    for run_dir, label in zip(args.runs, labels):
        steps, success = parse_log(run_dir, args.env)
        if not steps:
            print('warning: no data lines in', run_dir)
            continue
        plt.plot(steps, success, label=label, linewidth=1.5)
        print('{}: {} eval points, final success rate {:.2f}'.format(
            label, len(steps), success[-1]))

    plt.xlabel('training step')
    plt.ylabel('success rate')
    plt.title('Success rate vs training steps (thesis Fig 10 reproduction)')
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print('saved', args.out)


if __name__ == '__main__':
    main()
