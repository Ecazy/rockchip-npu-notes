#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measured Submit costs. Standard library, no extrapolation or invented queue."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CAMPAIGNS = (
    'cost-20261010-165249', 'concurrency-20261010-170013',
    'resident-20261010-171054', 'saturation-20261010-172338',
    'interference-20261010-170421', 'tail-20261010-173208', 'trace-20261010-173033')
SEED = 20261010

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def read_rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def quantile(hist, q):
    """Nearest-rank empirical quantile, integer ns (no precision interpolation)."""
    total = sum(hist.values())
    if not total:
        return None
    rank = max(1, math.ceil(q * total))
    n = 0
    for value, count in sorted(hist.items()):
        n += count
        if n >= rank:
            return value

def distribution(hist):
    count = sum(hist.values())
    return {'count': count, 'histogram_ns': [[k, v] for k, v in sorted(hist.items())],
            **{f'p{p}_ns': quantile(hist, p / 100) for p in (50, 90, 95, 99)}}

def confidence(block_hists, repeats=1000):
    """Resample independent randomized blocks, not within-block submissions."""
    medians = [quantile(h, .5) for _, h in sorted(block_hists.items())]
    if len(medians) < 2:
        return {'level': .95, 'method': 'block-bootstrap mean of block p50',
                'independent_blocks': len(medians), 'status': 'insufficient_blocks',
                'mean_block_p50_ns': medians[0] if medians else None, 'ci_ns': None,
                'seed': SEED, 'resamples': 0}
    rng = random.Random(SEED)
    samples = sorted(statistics.mean(rng.choices(medians, k=len(medians)))
                     for _ in range(repeats))
    return {'level': .95, 'method': 'percentile block-bootstrap mean of block p50',
            'independent_blocks': len(medians), 'status': 'measured',
            'mean_block_p50_ns': statistics.mean(medians),
            'ci_ns': [samples[math.floor(.025 * (repeats - 1))],
                      samples[math.ceil(.975 * (repeats - 1))]],
            'seed': SEED, 'resamples': repeats}

def condition(config, row, campaign):
    return {'campaign': campaign, 'workload': config['workload'],
            'shape_mkn': config['dims'], 'dtype': 'fp16',
            'programs_per_operation': config['programs'], 'mode': config['mode'],
            'submit_index': row['submit_index'], 'task_count': row['task_count'],
            'residency': config['path'], 'workers': config['workers'],
            'strategy': config['strategy'], 'requested_mask': row['requested_mask'],
            'actual_mask': row['actual_mask'], 'allowed_mask': config['allowed'],
            'start': config['start'], 'background': config.get('background'),
            'capture': config.get('capture', 'inline'),
            'affinity_base': config.get('affinity_base'), 'tracing': config['trace']}

def valid_submit(row, op):
    if row.get('warmup'):
        return 'warmup'
    if (row.get('result') != 0 or row.get('bad_count') != 0
            or row.get('guard_ok') is not True or op is None
            or op.get('result') != 0 or op.get('bad_count') != 0):
        return 'invalid'
    times = [row.get(k) for k in ('host_begin_ns', 'ioctl_begin_ns', 'ioctl_end_ns', 'host_done_ns')]
    service = row.get('driver_service_ns')
    if (any(type(t) is not int or t < 0 for t in times) or times != sorted(times)
            or type(service) is not int or not 0 < service <= times[2] - times[1]):
        raise ValueError('corrupt Submit timing')
    return 'valid'

def build(root=ROOT, campaigns=CAMPAIGNS):
    root = Path(root)
    sources, hardware, groups, census = {}, {}, {}, []
    def source(path):
        relative = path.relative_to(root).as_posix()
        if relative not in sources:
            sources[relative] = {'path': relative, 'sha256': digest(path), 'bytes': path.stat().st_size}
        return relative
    for name in campaigns:
        camp = root / 'research/custom-ops/results/2026-10-10' / ('device-cost-' + name)
        status = json.loads((camp / 'status.json').read_text())
        if status.get('status') != 'PASS':
            raise ValueError(f'campaign not PASS: {name}')
        env = json.loads((camp / 'environment-fixed.json').read_text())
        hardware[name] = {'chip': 'rk3588', 'board': 'Orange Pi 5 Plus (experiment identity)',
                          'serial': None, 'driver': env.get('driver'), 'uname': env.get('uname'),
                          'npu': env.get('npu'), 'dmc': env.get('dmc'),
                          'cpu_affinity': env.get('cpu_affinity'),
                          'environment_source': source(camp / 'environment-fixed.json'),
                          'memory_domain': 'shared system DDR BO; CBUF residency unobserved'}
        for filename in ('status.json', 'config.json', 'runs.jsonl', 'source-manifest.json',
                         'evidence-manifest.sha256'):
            if (camp / filename).exists():
                source(camp / filename)
        counts = Counter()
        seen = set()
        for run in read_rows(camp / 'runs.jsonl'):
            if run['exit_code'] != 0:
                raise ValueError(f'failed run in accepted campaign {name}')
            raw = camp / run['raw']
            if raw in seen:
                raise ValueError('duplicate raw run')
            seen.add(raw)
            raw_ref = source(raw)
            refs = [raw_ref]
            for filename in ('capture.trace', 'queue-bounds.json', 'trace-stats.json',
                             'kernel-audit.json', 'command.json', 'environment-samples.jsonl',
                             'background.jsonl'):
                p = raw.parent / filename
                if p.exists():
                    refs.append(source(p))
            rows = list(read_rows(raw))
            ops = {(r['worker'], r['operation_id']): r for r in rows if r.get('type') == 'operation'}
            for r in rows:
                if r.get('type') != 'submit':
                    continue
                op = ops.get((r['worker'], r['operation_id']))
                gate = valid_submit(r, op)
                counts[gate] += 1
                if gate != 'valid':
                    continue
                c = condition(run['config'], r, name)
                key = json.dumps(c, sort_keys=True, separators=(',', ':'))
                if key not in groups:
                    groups[key] = {'condition': c, 'service': Counter(), 'blocks': defaultdict(Counter),
                                   'metrics': defaultdict(Counter), 'refs': set(), 'memory': {}, 'features': {}}
                g = groups[key]
                g['service'][r['driver_service_ns']] += 1
                g['blocks'][run['block']][r['driver_service_ns']] += 1
                h0, h1, h2, h3 = (r[k] for k in ('host_begin_ns', 'ioctl_begin_ns', 'ioctl_end_ns', 'host_done_ns'))
                for metric, value in {'prepare_ns': h1-h0, 'ioctl_ns': h2-h1,
                                      'api_response_ns': h3-h0, 'post_submit_ns': h3-h2,
                                      'operation_upload_window_ns': op['upload_end_ns']-op['begin_ns'],
                                      'operation_post_window_ns': op['end_ns']-op['submit_end_ns']}.items():
                    if value < 0:
                        raise ValueError('negative measured operation component')
                    g['metrics'][metric][value] += 1
                for field in ('payload_bytes', 'allocated_tensor_bo_bytes'):
                    g['memory'][field] = max(g['memory'].get(field, 0), op[field])
                for field in ('physical_macs', 'estimated_bytes', 'command_bytes'):
                    g['features'][field] = max(g['features'].get(field, 0), r[field])
                g['refs'].update(refs)
        census.append({'campaign': name, 'raw_files': len(seen), **dict(counts)})
    cells = []
    for key, g in sorted(groups.items()):
        metrics = {}
        for name, hist in sorted(g['metrics'].items()):
            d = distribution(hist)
            d.pop('histogram_ns')  # service retains full empirical distribution
            metrics[name] = d
        cells.append({'id': hashlib.sha256(key.encode()).hexdigest()[:20],
                      'condition': g['condition'], 'hardware_id': g['condition']['campaign'],
                      'service': distribution(g['service']), 'confidence': confidence(g['blocks']),
                      'block_counts': {str(k): sum(h.values()) for k, h in sorted(g['blocks'].items())},
                      'host': metrics, 'memory': g['memory'], 'features': g['features'],
                      'sources': sorted(g['refs']), 'queue_ns': None,
                      'queue_scope': 'unobserved here; traced hulls retained in queue-bounds source',
                      'migration_cost_ns': None})
    return {'schema_version': 1, 'time_unit': 'ns',
            'service_scope': 'driver commit-to-IRQ envelope, not pure hardware execution',
            'operation_metric_scope': 'operation window repeated on each constituent Submit; never sum as job-exclusive time',
            'hardware': hardware, 'census': census, 'cells': cells}, {
                'schema_version': 1, 'sources': [sources[k] for k in sorted(sources)]}

def validate(document, manifest=None, root=ROOT, full_sources=False):
    if document.get('schema_version') != 1 or document.get('time_unit') != 'ns':
        raise ValueError('unsupported dataset schema')
    ids = set()
    n = 0
    for cell in document['cells']:
        if cell['id'] in ids or cell['hardware_id'] not in document['hardware']:
            raise ValueError('duplicate cell or missing hardware identity')
        ids.add(cell['id'])
        hist = cell['service']['histogram_ns']
        if any(type(t) is not int or t <= 0 or type(c) is not int or c <= 0 for t, c in hist):
            raise ValueError('invalid histogram')
        if len({t for t, _ in hist}) != len(hist):
            raise ValueError('duplicate histogram values')
        if sum(c for _, c in hist) != cell['service']['count'] or sum(cell['block_counts'].values()) != cell['service']['count']:
            raise ValueError('sample count mismatch')
        for p in (50, 90, 95, 99):
            if quantile(dict(hist), p/100) != cell['service'][f'p{p}_ns']:
                raise ValueError('quantile mismatch')
        n += cell['service']['count']
    if n != sum(c.get('valid', 0) for c in document['census']):
        raise ValueError('census mismatch')
    indexed = {s['path']: s for s in (manifest or {}).get('sources', [])}
    if manifest is not None:
        for cell in document['cells']:
            if any(p not in indexed for p in cell['sources']):
                raise ValueError('unindexed source reference')
    if full_sources:
        if manifest is None:
            raise ValueError('full verification requires source manifest')
        for s in indexed.values():
            path = (Path(root) / s['path']).resolve()
            if not path.is_relative_to(Path(root).resolve()) or not path.is_file() or digest(path) != s['sha256']:
                raise ValueError(f'missing or changed source: {s["path"]}')
    return {'cells': len(ids), 'submits': n, 'full_sources_verified': bool(full_sources), 'sources': len(indexed)}

class JobCostDataset:
    def __init__(self, path=HERE / 'data/job-costs.json'):
        self.document = json.loads(Path(path).read_text())
        validate(self.document)
    def match(self, **conditions):
        return [cell for cell in self.document['cells']
                if all(cell['condition'].get(k) == v for k, v in conditions.items())]
    def cost(self, condition, percentile=50):
        if percentile not in (50, 90, 95, 99):
            raise ValueError('unsupported percentile')
        candidates = [c for c in self.document['cells'] if c['condition'] == condition]
        if len(candidates) != 1:
            raise ValueError('unknown or ambiguous measured condition; no extrapolation')
        c = candidates[0]
        return {'duration_ns': c['service'][f'p{percentile}_ns'], 'cell_id': c['id'],
                'hardware': self.document['hardware'][c['hardware_id']], 'confidence': c['confidence'],
                'scope': self.document['service_scope'], 'evidence': 'measured'}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['build', 'validate'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--data', type=Path, default=HERE / 'data/job-costs.json')
    p.add_argument('--manifest', type=Path, default=HERE / 'data/source-manifest.json')
    p.add_argument('--full-sources', action='store_true')
    args = p.parse_args(argv)
    if args.command == 'build':
        doc, manifest = build(args.root)
        validate(doc, manifest)
        for path, data in ((args.data, doc), (args.manifest, manifest)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, sort_keys=True, separators=(',', ':'))+'\n')
    else:
        doc = json.loads(args.data.read_text()); manifest = json.loads(args.manifest.read_text())
    print(json.dumps(validate(doc, manifest, args.root, args.full_sources), sort_keys=True))

if __name__ == '__main__':
    main()
