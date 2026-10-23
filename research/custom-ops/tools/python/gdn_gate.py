#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent FP64 recurrence and merged-old-state correction oracle; stdlib only."""
import json
import math
from pathlib import Path
import statistics
import struct
import sys

SHAPES = ((1, 1), (19, 37), (32, 32), (128, 128), (256, 256))
STEPS = 32

def read(root, dv, dk, name, n):
    data = (root / f"{dv}x{dk}-{name}.bin").read_bytes()
    if len(data) != n * 2:
        raise ValueError(f"wrong file size: {name}")
    values = struct.unpack(f"<{n}e", data)
    if not all(map(math.isfinite, values)):
        raise ValueError(f"nonfinite {name}")
    return values

def check(got, expected):
    if not math.isfinite(got) or abs(got-expected) > .0003 + .006*abs(expected):
        raise ValueError(f"numerical mismatch: {got} versus {expected}")
    return abs(got-expected)

def verify(root):
    result = []
    for dv, dk in SHAPES:
        n = dv*dk
        s = list(read(root, dv, dk, "initial", n))
        ks = read(root, dv, dk, "k", STEPS*dk)
        qs = read(root, dv, dk, "q", STEPS*dk)
        vs = read(root, dv, dk, "v", STEPS*dv)
        gates = read(root, dv, dk, "gates", STEPS*2)
        outputs = read(root, dv, dk, "out", STEPS*dv)
        states = read(root, dv, dk, "states", STEPS*n)
        peak_s = peak_o = 0.
        for t in range(STEPS):
            a, b = gates[t*2:t*2+2]
            k, q = ks[t*dk:(t+1)*dk], qs[t*dk:(t+1)*dk]
            kq = sum(x*y for x, y in zip(k, q))
            for r in range(dv):
                old = s[r*dk:(r+1)*dk]
                sk = sum(x*y for x, y in zip(old, k))
                sq = sum(x*y for x, y in zip(old, q))
                e = b*(vs[t*dv+r]-a*sk)
                new = [a*x+e*y for x, y in zip(old, k)]
                out = sum(x*y for x, y in zip(new, q))
                if abs(out-(a*sq+e*kq)) > 1e-12:
                    raise ValueError("merged old-state correction failed")
                peak_o = max(peak_o, check(outputs[t*dv+r], out))
                for j, val in enumerate(new):
                    peak_s = max(peak_s, check(states[t*n+r*dk+j], val))
                s[r*dk:(r+1)*dk] = new
        result.append(dict(dv=dv, dk=dk, steps=STEPS, state_peak_abs=peak_s, output_peak_abs=peak_o))
    return result

def logs(path, reps, mode):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
    if sum(r.get('kind') == 'result' and r.get('status') == 'PASS' for r in rows) != 1:
        raise ValueError("missing or duplicate final PASS")
    report = []
    for dv, dk in SHAPES:
        selected = [r for r in rows if (r.get('dv'), r.get('dk')) == (dv, dk)]
        for kind, arms in (("gemv", ("A", "A2", "B", "C", "A_pair")), ("outer", ("K1", "EW_expanded"))):
            for arm in arms:
                group = [r for r in selected if r['kind'] == kind and r['arm'] == arm]
                if sorted(r['rep'] for r in group) != list(range(reps)):
                    raise ValueError("missing/duplicate measurements")
                for r in group:
                    if any(not math.isfinite(v) or v < 0 for key, v in r.items() if key.endswith('_us')):
                        raise ValueError("invalid timing")
                    if kind == 'gemv':
                        if r['tasks'] != (0 if arm == 'C' else 2 if arm == 'A_pair' else 1) or r['submits'] != (0 if arm == 'C' else 2 if arm == 'A_pair' else 1):
                            raise ValueError("wrong submit/task count")
                        if r['cpu_kind'] != 'aarch64_neon_f16_storage_f32_acc':
                            raise ValueError("board CPU baseline is not NEON")
                report.append(dict(dv=dv, dk=dk, kind=kind, arm=arm, samples=len(group),
                                   median_us={key:statistics.median(r[key] for r in group)
                                              for key in group[0] if key.endswith('_us')}))
        resident = [r for r in selected if r['kind'] == 'resident_repeat']
        if len(resident) != 1 or resident[0]['steps'] != STEPS or resident[0]['state_exports_during_steps'] != 0 or resident[0]['bit_exact'] is not True:
            raise ValueError("missing continuous resident-state validation")
        scalar = [r for r in selected if r['kind'] == 'scalar_and_subtract']
        if len(scalar) != 1 or scalar[0]['scales'] != 4:
            raise ValueError("missing scalar/subtraction validation")
        group = [r for r in selected if r['kind'] == 'recurrent']
        if sorted(r['token'] for r in group) != list(range(STEPS)):
            raise ValueError("missing/duplicate recurrent tokens")
        if any(r['tasks'] != 6 or r['submits'] != (1 if mode == 'chain' else 6) for r in group):
            raise ValueError("wrong recurrent submit count")
    return report

if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.exit('usage: gdn_gate.py RESULT_DIR LOG REPS chain|separate')
    if sys.argv[4] not in ('chain', 'separate'):
        sys.exit('invalid mode')
    print(json.dumps(dict(status='PASS', numerical=verify(Path(sys.argv[1])),
                          performance=logs(Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4])), indent=2))
