#!/usr/bin/env python3
"""Numerical gates and paired speedup/hidden-budget estimates for matched workloads."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

LEGACY_ARMS = ('a', 'b', 'ba', 'serial', 'overlap', 'bb')
EIGHT_ARMS = (*LEGACY_ARMS, 'ab', 'a_then_b')
ARMS = (*EIGHT_ARMS, 'separate')


def dist(values):
    if not values:
        return None
    values = sorted(values)
    def quantile(q):
        pos = (len(values) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return dict(median=median(values), p10=quantile(.1), p90=quantile(.9))


def check_sample(row, case):
    if (row.get('type') != 'matched' or row.get('result') != 0 or row.get('valid') != 1
            or row.get('actual_mask') != 1 or row.get('service_ns', 0) <= 0
            or row.get('int_status', 0) & 0x300 == 0
            or row.get('irq_ok') != 1 or row.get('irq') != ([2, 0, 0] if row.get('arm') == 'separate' else [1, 0, 0])):
        return False
    if case.get('global_inputs') and row.get('global_inputs') != 1:
        return False
    if row.get('cache_lut', 0) != case.get('cache_lut', 0):
        return False
    if row.get('early_core', 0) != case.get('early_core', 0):
        return False
    arm, pairs = row.get('arm'), case['pairs']
    if arm not in ARMS or row.get('task_count') != pairs * (1 if arm in ('a', 'b') else 2):
        return False
    if row.get('arm_count', 6) == 9:
        submits = 2 if arm == 'separate' else 1
        parts = row.get('submissions', [])
        if row.get('submits') != submits or len(parts) != submits:
            return False
        if (sum(p.get('tasks', 0) for p in parts) != row['task_count']
                or sum(p.get('service_ns', 0) for p in parts) != row['service_ns']
                or any(p.get('result') != 0 or p.get('service_ns', 0) <= 0
                       or p.get('actual_mask') != 1 or p.get('int_status', 0) & 0x300 == 0 for p in parts)):
            return False
        if arm == 'separate' and any(p.get('tasks') != pairs for p in parts):
            return False
    for key in ('m', 'k', 'n', 'b_elements', 'pairs', 'warmups'):
        if row.get(key) != case[key]:
            return False
    if row.get('requested_reps') != case['reps']:
        return False
    checks = row.get('checks', [])
    if len(checks) != 2 * pairs:
        return False
    expected = {(i, kind) for i in range(pairs) for kind in ('mm', 'lut')}
    if {(c.get('tile'), c.get('kind')) for c in checks} != expected:
        return False
    for c in checks:
        used = arm != 'a' if c['kind'] == 'lut' else arm not in ('b', 'bb')
        count = case['b_elements'] if c['kind'] == 'lut' else case['m'] * case['n']
        if (c.get('used') != used or c.get('valid') != 1 or c.get('bad') != 0
                or c.get('elements') != count or c.get('nan') != (0 if used else count)
                or not isinstance(c.get('hash'), str) or len(c['hash']) != 16):
            return False
    return True


def submitted_pair_ok(path, name, pairs, early_core=False):
    def parse(arm):
        lines = (path / f'{name}.{arm}.submitted').read_text().splitlines()
        info = dict(word.split('=', 1) for word in lines[0].split()[1:])
        base = int(info['iova'], 16)
        words = [int(line.split()[1], 16) for line in lines[1:]]
        for i, word in enumerate(words):
            if word >> 48 == 0x101 and word & 0xffff == 0x10:
                relative = ((word >> 16) & 0xffffffff) - base
                words[i] = (word & 0xffff00000000ffff) | (relative << 16)
        return info, words
    try:
        si, serial = parse('serial')
        oi, overlap = parse('overlap')
    except (OSError, ValueError, IndexError):
        return False
    if (len(serial) != len(overlap) or int(si['hw_tasks']) != pairs * 2 or int(oi['hw_tasks']) != pairs * 2
            or int(si['streams']) != pairs * 2 or int(oi['streams']) != pairs * 2
            or int(si['count']) != len(serial) or int(oi['count']) != len(overlap)
            or si.get('pp') != '0' or oi.get('pp') != '0'):
        return False
    differences = [(a, b) for a, b in zip(serial, overlap) if a != b]
    def enable(mask):
        return (0x81 << 48) | (mask << 16) | 8
    if early_core:
        expected = [((0x201 << 48) | 0x1008, (0x201 << 48) | (1 << 16) | 0x1008),
                    ((0x801 << 48) | 0x3008, (0x801 << 48) | (1 << 16) | 0x3008),
                    (enable(0x1d), enable(0x18))] * pairs
    else:
        expected = [(enable(0x18), enable(0x1d)), (enable(0x1d), enable(0x18))] * pairs
    return differences == expected


def default_streams_ok(path, name, pairs, include_separate):
    """Check full A->B programs and, where present, identical two-Submit bodies."""
    def parse(suffix, tasks):
        lines = (path / f'{name}.{suffix}.submitted').read_text().splitlines()
        info = dict(word.split('=', 1) for word in lines[0].split()[1:])
        words = [int(line.split()[1], 16) for line in lines[1:]]
        if (int(info['hw_tasks']) != tasks or int(info['streams']) != tasks
                or int(info['count']) != len(words) or info.get('pp') != '0'):
            raise ValueError('bad task layout')
        programs, current = [], []
        for word in words:
            current.append(word)
            if word >> 48 == 0x81 and word & 0xffff == 8:
                programs.append(current)
                current = []
        if current or len(programs) != tasks:
            raise ValueError('missing task boundaries')
        return programs
    def kind(program):
        targets = {(w >> 48) & 0xff00 for w in program[:-4]}
        enable = (program[-1] >> 16) & 0xffffffff
        if 0x200 in targets and 0x800 in targets and 0x1000 in targets and enable == 0x1d:
            return 'mm'
        if 0x200 not in targets and 0x800 not in targets and 0x1000 in targets and enable == 0x18:
            return 'lut'
        return 'invalid'
    try:
        grouped = parse('a_then_b', 2 * pairs)
        interleaved = parse('ab', 2 * pairs)
        if ([kind(p) for p in grouped] != ['mm'] * pairs + ['lut'] * pairs
                or [kind(p) for p in interleaved] != ['mm', 'lut'] * pairs):
            return False
        if include_separate:
            separate = parse('separate.a', pairs) + parse('separate.b', pairs)
            if [p[:-4] for p in separate] != [p[:-4] for p in grouped]:
                return False
        return True
    except (OSError, ValueError, IndexError, KeyError):
        return False


def analyze_case(path, case):
    name = case['name']
    result = dict(case, accepted=False, errors=[])
    rows = [json.loads(line) for line in (path / f'{name}.jsonl').read_text().splitlines() if line.startswith('{')]
    samples = [r for r in rows if r.get('type') == 'matched']
    counts = {r.get('arm_count', 6) for r in samples}
    arms = ARMS if counts == {9} else EIGHT_ARMS if counts == {8} else LEGACY_ARMS
    if counts not in ({6}, {8}, {9}):
        result['errors'].append('inconsistent/unknown arm count')
    result['arm_count'] = len(arms)
    if case['exit'] or len(samples) != (case['reps'] + case['warmups']) * len(arms):
        result['errors'].append('incomplete/failed execution')
    if not all(check_sample(row, case) for row in samples):
        result['errors'].append('numerical/IRQ/sample gate failed')
    hashes = {}
    for row in samples:
        for c in row.get('checks', []):
            if c.get('used'):
                hashes.setdefault((c['tile'], c['kind']), set()).add(c.get('hash'))
    if any(len(values) != 1 or None in values for values in hashes.values()):
        result['errors'].append('output bits differ across arms/repetitions')
    if not submitted_pair_ok(path, name, case['pairs'], case.get('early_core', 0)):
        result['errors'].append('submitted streams differ beyond expected enable words')
    if len(arms) >= 8 and not default_streams_ok(path, name, case['pairs'], len(arms) == 9):
        result['errors'].append('default A->B stream order or separate-job bodies differ')
    rounds = {}
    for row in samples:
        if row.get('phase') == 'main':
            group = rounds.setdefault(row['rep'], {})
            if row['arm'] in group:
                result['errors'].append('duplicate sample')
            group[row['arm']] = row
    complete = [g for g in rounds.values() if set(g) == set(arms) and all(check_sample(r, case) for r in g.values())]
    if len(complete) != case['reps']:
        result['errors'].append('missing complete paired rounds')
    if result['errors']:
        return result
    result['accepted'] = True
    result['samples'] = len(samples)
    result['service_ns'] = {a: dist([g[a]['service_ns'] for g in complete]) for a in arms}
    result['ioctl_ns'] = {a: dist([g[a]['ioctl_ns'] for g in complete]) for a in arms}
    def differences(field, baseline, metric):
        return dist([metric(g[baseline][field], g['overlap'][field]) for g in complete])
    fields = ['service_ns', 'ioctl_ns']
    if all('submit_ns' in row for g in complete for row in g.values()):
        fields.append('submit_ns')
        result['submit_ns'] = {a: dist([g[a]['submit_ns'] for g in complete]) for a in arms}
    for field in fields:
        result[field + '_saving'] = differences(field, 'serial', lambda s, o: s - o)
        result[field + '_saving_pct'] = differences(field, 'serial', lambda s, o: 100 * (s - o) / s)
        result[field + '_speedup'] = differences(field, 'serial', lambda s, o: s / o)
        result[field + '_vs_ba_pct'] = differences(field, 'ba', lambda s, o: 100 * (s - o) / s)
        for baseline in ('ab', 'a_then_b', 'separate'):
            if baseline in arms:
                result[field + '_vs_' + baseline + '_pct'] = differences(field, baseline, lambda s, o: 100 * (s - o) / s)
                result[field + '_vs_' + baseline + '_saving_ns'] = differences(field, baseline, lambda s, o: s - o)
                result[field + '_vs_' + baseline + '_speedup'] = differences(field, baseline, lambda s, o: s / o)
    result['serial_increment_over_a_ns'] = dist([
        g['serial']['service_ns'] - g['a']['service_ns'] for g in complete])
    result['overlap_increment_over_a_ns'] = dist([
        g['overlap']['service_ns'] - g['a']['service_ns'] for g in complete])
    # BB-B removes one fixed submit envelope. It includes DPU register fetch/configuration.
    budget = [g['bb']['service_ns'] - g['b']['service_ns'] for g in complete]
    result['dpu_chain_budget_ns'] = dist(budget)
    result['hidden_dpu_budget_pct'] = dist([
        100 * (g['serial']['service_ns'] - g['overlap']['service_ns']) / b
        for g, b in zip(complete, budget) if b > 0])
    result['positive_budget_rounds'] = sum(b > 0 for b in budget)
    for baseline in ('ab', 'a_then_b', 'separate'):
        if baseline in arms:
            result['hidden_dpu_budget_vs_' + baseline + '_pct'] = dist([
                100 * (g[baseline]['service_ns'] - g['overlap']['service_ns']) / b
                for g, b in zip(complete, budget) if b > 0])
    result['primary_baseline'] = 'separate' if 'separate' in arms else 'a_then_b' if 'a_then_b' in arms else 'serial'
    if 'separate' in arms:
        for field in fields:
            result[field + '_merge_only_pct'] = dist([
                100 * (g['separate'][field] - g['a_then_b'][field]) / g['separate'][field] for g in complete])
            result[field + '_merge_only_saving_ns'] = dist([
                g['separate'][field] - g['a_then_b'][field] for g in complete])
        result['physical_submits'] = (case['reps'] + case['warmups']) * (len(arms) + 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    root = args.results
    cases = json.loads((root / 'cases.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    results = [analyze_case(root, case) for case in cases]
    errors = []
    if len(cases) != len(config['cases']):
        errors.append('campaign missing cases')
    for case, requested in zip(cases, config['cases']):
        if any(case.get(k) != v for k, v in requested.items()):
            errors.append(f'{case["name"]}: configuration differs from requested case')
    for label in ('initial-recovery', 'final'):
        if not (root / f'{label}.exit').exists() or int((root / f'{label}.exit').read_text()):
            errors.append(f'{label} failed')
    if '100% tests passed' not in (root / 'host-tests.log').read_text():
        errors.append('host tests failed')
    kernel_path = root / 'kernel-after.log'
    if not kernel_path.exists():
        errors.append('missing kernel log')
        kernel = ''
    else:
        kernel = kernel_path.read_text()
    kernel_lines = kernel.splitlines()
    fatal_lines = [line for line in kernel_lines if
                   ('iommu' in line.lower() and any(w in line.lower() for w in ('fault', 'error')))
                   or 'kernel panic' in line.lower() or 'BUG:' in line]
    if fatal_lines:
        errors.append('unexpected kernel/IOMMU fault')
    resets = sum('soft reset' in line for line in kernel_lines)
    if config.get('require_all', True) and resets != 1:
        errors.append(f'expected only initial reset, observed {resets}')
    accepted = [r for r in results if r['accepted']]
    report = dict(cases=results, errors=errors, complete=not errors, kernel_resets=resets, kernel_faults=fatal_lines,
                  accepted_cases=len(accepted), rejected_cases=len(results) - len(accepted))
    report['arm_definitions'] = {
        'separate': 'All complete MatMul tasks, then all complete DPU-only tasks; two blocking Submits.',
        'a_then_b': 'Same complete tasks and order as separate; one same-core chained Submit.',
        'overlap': 'One Submit, with CNA/CORE(MatMul)+DPU-only followed by DPU(MatMul) per tile.',
        'ab': 'Complete MatMul then complete DPU-only per tile; one chained Submit.',
        'ba': 'Complete DPU-only then complete MatMul per tile; one chained Submit.',
        'serial': 'Artificial same-layout delayed-launch control; not the default A-then-B path.'
    }
    report['ranking'] = [r['name'] for r in sorted(accepted, key=lambda r: r.get('service_ns_vs_separate_pct', r.get('service_ns_vs_a_then_b_pct', r['service_ns_saving_pct']))['median'], reverse=True)]
    report['evidence_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(root.iterdir()) if p.is_file() and p.name != 'analysis.json'}
    (root / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'MATCHED complete={report["complete"]} accepted={len(accepted)} rejected={report["rejected_cases"]}')
    for name in report['ranking'][:12]:
        r = next(r for r in results if r['name'] == name)
        if r['primary_baseline'] == 'separate':
            print(f'{name} separate_to_overlap={r["service_ns_vs_separate_pct"]["median"]:.2f}% '
                  f'chain_to_overlap={r["service_ns_vs_a_then_b_pct"]["median"]:.2f}% '
                  f'merge_only={r["service_ns_merge_only_pct"]["median"]:.2f}%')
        elif r['primary_baseline'] == 'a_then_b':
            print(f'{name} AB={r["service_ns_vs_ab_pct"]["median"]:.2f}% '
                  f'all_A_then_B={r["service_ns_vs_a_then_b_pct"]["median"]:.2f}%')
        print(f'{name} M/K/N={r["m"]}/{r["k"]}/{r["n"]} B={r["b_elements"]} pairs={r["pairs"]} '
              f'saving={r["service_ns_saving_pct"]["median"]:.2f}% '
              f'vs_BA={r["service_ns_vs_ba_pct"]["median"]:.2f}% '
              f'hidden={r["hidden_dpu_budget_pct"]["median"]:.1f}%')
    return int(bool(errors) or (config.get('require_all', True) and len(accepted) != len(results)))


if __name__ == '__main__':
    raise SystemExit(main())
