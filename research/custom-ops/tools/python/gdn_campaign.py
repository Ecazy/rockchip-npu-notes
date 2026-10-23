#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Revalidate retrieved artifacts, paired mode equivalence and SHA256 manifest."""
import hashlib
import json
from pathlib import Path
import sys
import gdn_gate

def verify(root):
    report = {}
    for core in (1, 2, 4):
        for mode in ('chain', 'separate'):
            name = f'core-{core}-{mode}'
            report[name] = dict(numerical=gdn_gate.verify(root/name),
                                performance=gdn_gate.logs(root/f'{name}.log', 20, mode))
        for dv, dk in gdn_gate.SHAPES:
            for name in ('initial', 'k', 'q', 'v', 'gates', 'out', 'states'):
                suffix = f'{dv}x{dk}-{name}.bin'
                a = (root/f'core-{core}-chain'/suffix).read_bytes()
                b = (root/f'core-{core}-separate'/suffix).read_bytes()
                if a != b:
                    raise ValueError(f'chain/separate bytes differ: core {core}, {suffix}')
    alias = {}
    for core in (1, 2, 4):
        status = root/f'inplace-{core}.exit'
        if status.exists():
            code = int(status.read_text())
            rows = [json.loads(line) for line in (root/f'inplace-{core}.log').read_text().splitlines() if line.startswith('{')]
            if code == 0:
                expected = set(gdn_gate.SHAPES)
                if len(rows) != len(expected) or {(r['dv'], r['dk']) for r in rows} != expected:
                    raise ValueError('incomplete in-place diagnostics')
                if any(r['kind'] != 'inplace_diagnostic' or r['production_enabled'] or r['trials'] != 8 for r in rows):
                    raise ValueError('wrong alias diagnostic contract')
            alias[str(core)] = dict(exit_code=code, rows=rows)
    journal = (root/'kernel-interval.log').read_text()
    errors = [line for line in journal.splitlines()
              if any(x in line.lower() for x in ('rknpu', 'iommu'))
              and any(x in line.lower() for x in ('fault', 'timeout', 'failed', 'error'))]
    if errors:
        raise ValueError(f'kernel errors: {errors}')
    hashes = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'campaign.json'}
    return dict(status='PASS', mode_equivalence='bit-exact within each core',
                runs=report, inplace_diagnostic=alias, sha256=hashes)

if __name__ == '__main__':
    print(json.dumps(verify(Path(sys.argv[1])), indent=2))
