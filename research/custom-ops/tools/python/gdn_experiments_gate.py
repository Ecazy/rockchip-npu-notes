#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent FP64 oracle and complete matched-experiment evidence gate."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys

SIZES = (128, 64, 256)
ARMS = ('cpu16', 'cpu32', 'npu_alloc', 'npu_reuse', 'beta_fold', 'merged', 'native_alpha')
STEPS, BLOCKS = 32, 5

def read(p, n, code):
    data = p.read_bytes()
    size = 2 if code == 'e' else 4
    if len(data) != n*size:
        raise ValueError(f'size mismatch {p}')
    values = struct.unpack(f'<{n}{code}', data)
    if not all(map(math.isfinite, values)):
        raise ValueError(f'nonfinite {p}')
    return values

def reference(root, n, b):
    prefix = root/f'n{n}-b{b}'
    def inp(name, count): return read(Path(str(prefix)+f'-{name}.bin'), count, 'e')
    s = list(inp('initial', n*n))
    ks, qs, vs, gs = inp('k', STEPS*n), inp('q', STEPS*n), inp('v', STEPS*n), inp('gates', STEPS*2)
    outputs = []
    for t in range(STEPS):
        k, q, v = ks[t*n:(t+1)*n], qs[t*n:(t+1)*n], vs[t*n:(t+1)*n]
        a, beta = gs[2*t:2*t+2]
        for r in range(n):
            row = s[r*n:(r+1)*n]
            d = [a*x for x in row]
            pred = sum(x*y for x, y in zip(d, k))
            e = beta*(v[r]-pred)
            row = [x+e*y for x, y in zip(d, k)]
            outputs.append(sum(x*y for x, y in zip(row, q)))
            s[r*n:(r+1)*n] = row
    return outputs, s

def error(got, ref, fp32=False):
    absolute = [abs(a-b) for a, b in zip(got, ref)]
    atol, rtol = (1e-6, 1e-5) if fp32 else (.0003, .006)
    if len(got) != len(ref) or any(not math.isfinite(a) or abs(a-b) > atol+rtol*abs(b) for a,b in zip(got, ref)):
        raise ValueError(f'numerical error max={max(absolute, default=0)} fp32={fp32}')
    return dict(max_abs=max(absolute), rmse=math.sqrt(sum(x*x for x in absolute)/len(ref)))

def expected_tasks(arm, n):
    return 0 if arm.startswith('cpu') else 5 if arm=='beta_fold' else 7 if arm=='merged' or (arm=='native_alpha' and n==256) else 6

def validate_rows(rows):
    if sum(r.get('kind')=='result' and r.get('status')=='PASS' for r in rows)!=1:
        raise ValueError('missing/duplicate PASS')
    backends=[r for r in rows if r['kind']=='backend']
    if len(backends)!=1 or backends[0]['cpu']!='aarch64-neon-f32-fma':
        raise ValueError('not a NEON board baseline')
    seen=set()
    for r in rows:
        if r['kind']!='step': continue
        key=tuple(r[k] for k in ('n','block','token','arm','instrumented'))
        if key in seen: raise ValueError('duplicate sample')
        seen.add(key)
        if r['tasks']!=expected_tasks(r['arm'],r['n']) or r['submits']!=(0 if r['arm'].startswith('cpu') else 1):
            raise ValueError('task/submit mismatch')
        if r['freq_before_khz']<=0 or r['freq_after_khz']<=0:
            raise ValueError('missing CPU frequency')
        if r.get('npu_freq_before_khz',0)<=0 or r.get('npu_freq_after_khz',0)<=0:
            raise ValueError('missing NPU frequency')
        if not math.isfinite(r['total_us']) or r['total_us']<=0:
            raise ValueError('invalid latency')
        if r['instrumented'] and not r['arm'].startswith('cpu'):
            p=r['profile']
            if any(not math.isfinite(v) or v<0 for v in p.values()): raise ValueError('invalid profile')
            if p['guard_us']<=0 or p['tasks']!=r['tasks']: raise ValueError('missing guard/tasks')
            if p['host_allocations']!=(1 if r['arm']=='npu_alloc' else 0): raise ValueError('unexpected host allocation')
            keys=('allocation_us','prepare_us','pack_k_us','pack_q_us','pack_v_us','pack_coeff_us','bo_sync_copy_us','emit_us','execute_api_us','guard_us','download_us')
            if sum(p[k] for k in keys)>p['total_us']+3: raise ValueError('overlapping top-level timers')
    expected={(n,b,t,a,p) for n in SIZES for b in range(BLOCKS) for t in range(STEPS) for a in ARMS for p in (0,1)}
    if seen!=expected: raise ValueError(f'missing/extra steps {len(seen)} != {len(expected)}')
    scalar=[r for r in rows if r['kind']=='scalar']
    expected={(n,v,r,a) for n in SIZES for v in (0,1) for r in range(32) for a in range(4)}
    if len(scalar)!=len(expected) or {(r['n'],r['vector'],r['rep'],r['arm']) for r in scalar}!=expected:
        raise ValueError('incomplete scalar experiment')
    if any(not math.isfinite(r['max_abs']) or r['max_abs']>.003 for r in scalar): raise ValueError('scalar mismatch')
    if any(r.get('freq_khz',0)<=0 or r.get('npu_freq_khz',0)<=0 for r in scalar): raise ValueError('missing scalar frequency')
    diag=[r for r in rows if r['kind']=='task_diagnostic']
    if len(diag)!=len(SIZES)*16 or {(r['n'],r['rep']) for r in diag}!={(n,r) for n in SIZES for r in range(16)}:
        raise ValueError('incomplete task diagnosis')
    if any(len(r['submit_wait_us'])!=6 or any(not math.isfinite(v) or v<=0 for v in r['submit_wait_us']) for r in diag):
        raise ValueError('invalid separate-task envelopes')

def summarize(rows):
    result=[]
    for n in SIZES:
        for arm in ARMS:
            g=[r for r in rows if r['kind']=='step' and r['n']==n and r['arm']==arm]
            plain=[r for r in g if not r['instrumented']]
            prof=[r for r in g if r['instrumented']]
            med=[statistics.median(r['total_us'] for r in plain if r['block']==b) for b in range(BLOCKS)]
            d=dict(n=n,arm=arm,samples=len(plain),block_median_us=med,median_us=statistics.median(r['total_us'] for r in plain),
                   instrumented_median_us=statistics.median(r['total_us'] for r in prof),
                   frequencies_khz=sorted({r[k] for r in plain for k in ('freq_before_khz','freq_after_khz')}))
            if not arm.startswith('cpu'):
                d['profile_median']={k:statistics.median(r['profile'][k] for r in prof) for k in prof[0]['profile']}
            result.append(d)
    scalar=[]
    for n in SIZES:
        for vector in (0,1):
            for arm in range(4):
                g=[r for r in rows if r['kind']=='scalar' and r['n']==n and r['vector']==vector and r['arm']==arm]
                scalar.append(dict(n=n,vector=vector,arm=arm,median_us={k:statistics.median(r[k] for r in g) for k in g[0] if k.endswith('_us')},max_abs=max(r['max_abs'] for r in g)))
    tasks=[dict(n=n,separate_submit_median_us=[statistics.median(r['submit_wait_us'][i] for r in rows if r['kind']=='task_diagnostic' and r['n']==n) for i in range(6)]) for n in SIZES]
    return dict(recurrence=result,scalar=scalar,task_diagnostic=tasks)

def run(root):
    result={}
    original=dict(part.split('=') for part in (root/'cpu-original.log').read_text().split())
    restored=(root/'cpu-restored.log').read_text().split()
    if restored!=[original['governor'],original['min'],original['max']]:
        raise ValueError('CPU policy not restored')
    npu_original=dict(part.split('=') for part in (root/'npu-original.log').read_text().split())
    npu_restored=(root/'npu-restored.log').read_text().split()
    if npu_restored!=[npu_original['governor'],npu_original['min'],npu_original['max']]:
        raise ValueError('NPU policy not restored')
    smoke=[json.loads(l) for l in (root/'native-smoke.log').read_text().splitlines() if l.startswith('{')]
    if sum(r.get('kind')=='result' and r.get('status')=='PASS' for r in smoke)!=1:
        raise ValueError('native smoke did not pass')
    samples=[r for r in smoke if r['kind']=='scalar']
    if len(samples)!=48 or {(r['n'],r['vector'],r['rep'],r['arm']) for r in samples}!={(n,v,r,2) for n in SIZES for v in (0,1) for r in range(8)}:
        raise ValueError('native smoke incomplete')
    refs={(n,b):reference(root/'core-1',n,b) for n in SIZES for b in range(BLOCKS)}
    for core in (1,2,4):
        folder=root/f'core-{core}'
        rows=[json.loads(l) for l in (root/f'core-{core}.log').read_text().splitlines() if l.startswith('{')]
        validate_rows(rows)
        numerical=[]
        for n in SIZES:
            for b in range(BLOCKS):
                prefix=f'n{n}-b{b}'
                for name in ('initial','k','q','v','gates'):
                    f=f'{prefix}-{name}.bin'
                    if (folder/f).read_bytes()!=(root/'core-1'/f).read_bytes(): raise ValueError('inputs differ across cores')
                for name,count in (('out',STEPS*n),('state',n*n)):
                    a=read(folder/f'{prefix}-p0-{name}.bin',len(ARMS)*count,'f')
                    if (folder/f'{prefix}-p0-{name}.bin').read_bytes()!=(folder/f'{prefix}-p1-{name}.bin').read_bytes():
                        raise ValueError('instrumentation changed numerics')
                    if a[2*count:3*count]!=a[3*count:4*count]: raise ValueError('reuse changed numerics')
                    for i,arm in enumerate(ARMS):
                        try: err=error(a[i*count:(i+1)*count],refs[n,b][0 if name=='out' else 1],arm=='cpu32')
                        except ValueError as ex: raise ValueError(f'core={core} n={n} block={b} arm={arm} {name}: {ex}') from ex
                        numerical.append(dict(n=n,block=b,arm=arm,tensor=name,**err))
        result[str(core)]=dict(**summarize(rows),numerical=numerical)
    journal=(root/'kernel-interval.log').read_text()
    bad=[l for l in journal.splitlines() if any(w in l.lower() for w in ('rknpu','iommu')) and any(w in l.lower() for w in ('error','fault','timeout','failed'))]
    if bad: raise ValueError(f'kernel errors {bad}')
    return dict(status='PASS',runs=result,sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.name!='experiments.json'})

if __name__=='__main__':
    print(json.dumps(run(Path(sys.argv[1])),indent=2))
