#!/usr/bin/env python3
"""Export reproducible figures from accepted matched-overlap evidence (matplotlib)."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['svg.hashsalt'] = 'rk3588-overlap'
import matplotlib.pyplot as plt


def three_arm_plot(report, output, image_format):
    rows = [r for r in report['cases'] if r['accepted'] and r['k'] == 4096
            and r['m'] == 12 and r['n'] == 32 and r['b_elements'] == 32768
            and 'separate' in r['service_ns']]
    rows.sort(key=lambda r: r['pairs'])
    if not rows:
        raise ValueError('No accepted three-arm evidence')
    fig, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 4.4), layout='constrained', squeeze=False)
    choices = [('separate', 'Separate\n2 submits', '#8798a0'),
               ('a_then_b', 'Full-task chain\n1 submit', '#37658a'),
               ('overlap', 'Regcmd split\n1 submit', '#007d82')]
    for ax, row in zip(axes[0], rows):
        for index, (arm, label, color) in enumerate(choices):
            values = row['service_ns'][arm]
            value = values['median'] / 1000
            ax.bar(index, value, color=color, width=.65)
            ax.errorbar(index, value,
                        yerr=[[(values['median'] - values['p10']) / 1000],
                              [(values['p90'] - values['median']) / 1000]],
                        color='#333333', fmt='none', capsize=3, linewidth=1)
            ax.text(index, value * 1.015, f'{value:.1f}', ha='center', fontsize=9)
        ax.set_xticks(range(3), [c[1] for c in choices], fontsize=9)
        ax.set_title(f'{row["pairs"]} row tiles / M={row["pairs"] * row["m"]}\n'
                     f'Merge: {row["service_ns_merge_only_pct"]["median"]:.1f}%  |  '
                     f'Overlap vs chain: {row["service_ns_vs_a_then_b_pct"]["median"]:.1f}%', fontsize=10)
        ax.set_ylabel('Driver service envelope (µs)')
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.2)
        ax.set_ylim(bottom=0)
        ax.margins(y=.18)
    fig.suptitle('RK3588 Core 0 / identical A then B workload per panel / K=4096, N=32', fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    target = output.with_suffix('.' + image_format)
    options = {'metadata': {'Date': None}} if image_format == 'svg' else {}
    fig.savefig(target, dpi=180, **options)
    if image_format == 'svg':
        target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--format', choices=('svg', 'png'), default='svg')
    args = parser.parse_args()
    report = json.loads((args.results / 'analysis.json').read_text())
    three_arm_plot(report, args.output, args.format)


if __name__ == '__main__':
    main()
