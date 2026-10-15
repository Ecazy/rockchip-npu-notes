#!/usr/bin/env python3
"""Run explicit matched-overlap cases, preserving failures and recovery evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('probe', type=Path)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {4})
    config = json.loads(args.config.read_text())
    output = args.output
    output.mkdir(exist_ok=True)
    (output / "cpu-affinity.json").write_text(json.dumps({"cpu_affinity": sorted(os.sched_getaffinity(0))}) + "\n")
    env = dict(os.environ, RKNPU_CORE_MASK='1', RG_PC_PP='0')
    env.pop('RG_DUMP_PATH', None)

    def run(label, arguments):
        with (output / f'{label}.jsonl').open('x') as stdout, (output / f'{label}.stderr').open('x') as stderr:
            try:
                result = subprocess.run([str(args.probe), *map(str, arguments)], env=env,
                                        stdout=stdout, stderr=stderr, timeout=180, check=False).returncode
            except subprocess.TimeoutExpired:
                result = 124
        (output / f'{label}.exit').write_text(str(result) + '\n')
        return result

    def reset(label):
        run(label + '-reset', ['direct-a', 1, 2, 0, output / (label + '-reset.regcmd')])
        return run(label + '-recovery', ['ab', 4, 1, 0, output / (label + '-recovery.regcmd')]) == 0

    if not reset('initial'):
        return 1
    cases = []
    for index, case in enumerate(config['cases']):
        name = f'case-{index:04d}'
        fields = [case[k] for k in ('m', 'k', 'n', 'b_elements', 'pairs', 'reps', 'warmups')]
        if not all(isinstance(v, int) and v >= 0 for v in fields):
            raise ValueError(case)
        env['RG_EARLY_CORE'] = str(case.get('early_core', 0))
        env['RG_CACHE_LUT'] = str(case.get('cache_lut', 0))
        status = run(name, ['matched', *fields, output / name])
        cases.append(dict(name=name, **case, exit=status))
        (output / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
        print(f'{name} M/K/N={case["m"]}/{case["k"]}/{case["n"]} B={case["b_elements"]} pairs={case["pairs"]} exit={status}', flush=True)
        if status and not reset(name):
            return 1
    if run('final', ['ab', 16, 1, 0, output / 'final.regcmd']):
        return 1
    return int(config.get('require_all', True) and any(c['exit'] for c in cases))


if __name__ == '__main__':
    raise SystemExit(main())
