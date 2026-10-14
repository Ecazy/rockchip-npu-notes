#!/usr/bin/env python3
"""Aggregate medians from three controlled runs; compare best vendor layout."""
import csv
import re
import statistics
from pathlib import Path
root = Path(__file__).parent / 'results'
pattern = re.compile(r'^(hidden|ffn_up|ffn_down)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)')
def read(prefix):
    values = {}
    for p in sorted(root.glob(prefix + '-[123].log')):
        for line in p.read_text().splitlines():
            m = pattern.match(line)
            if m:
                key = (m[1], int(m[2]), int(m[3]), int(m[4]))
                values.setdefault(key, []).append(float(m[5]))
    if not values or any(len(v) != 3 for v in values.values()):
        raise ValueError(f'{prefix}: expected three complete rounds')
    return {k: statistics.median(v) for k, v in values.items()}
open_times = read('controlled-open')
normal = read('controlled-off-normal')
native = read('controlled-off-native')
with (root / 'controlled-summary.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['layer', 'M', 'K', 'N', 'open_us', 'official_normal_us', 'official_native_us', 'official_best_layout', 'speedup'])
    for k, t in open_times.items():
        best = min(normal[k], native[k])
        layout = 'normal' if normal[k] <= native[k] else 'native'
        w.writerow([*k, t, normal[k], native[k], layout, best/t])
        print(f'{k}: open={t:.2f}us official_best={best:.2f}us ({layout}) speedup={best/t:.3f}x')
