#!/usr/bin/env python3
"""Board-local campaigns: one external lock, fixed environment, immutable logs.

The v2 executable matrix replaces the unfinished DeepSeek v1 draft. Old evidence
and generators remain archived; no draft template is passed to a device runner.
"""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import random
import re
import signal
import subprocess
import threading
import time

from device_cost_analyze import analyze, require

HERE = pathlib.Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
TRACE = pathlib.Path('/sys/kernel/tracing')


def cell(workload, dims, programs=1, workers=1, strategy='fixed', mask=1,
         allowed=7, mode='chain', path='device_ready', start='continuous', background=None):
    label = f"{workload}-{'x'.join(map(str, dims))}-p{programs}-w{workers}-{strategy}-m{mask}-a{allowed}-{mode}-{path}-{start}-{background or 'none'}"
    return dict(id=label, workload=workload, dims=list(dims), programs=programs,
                workers=workers, strategy=strategy, mask=mask, allowed=allowed,
                mode=mode, path=path, start=start, background=background,
                affinity_base=0, trace=False, capture='immediate')


def matrix(suite):
    cells = []
    if suite == 'smoke':
        cells = [cell('lut', (64, 64, 64), 4), cell('matmul', (29, 37, 45)),
                 cell('mm_bias_hsig', (29, 37, 45)), cell('mm_bias_hsig', (256, 256, 256)),
                 cell('lut', (64, 64, 64), 4, 3, 'split')]
    elif suite == 'cost':
        for core in range(3):
            for rows, cols in [(8, 8), (29, 37), (32, 32), (64, 64), (128, 128)]:
                for pc in (1, 4, 16):
                    cells.append(cell('lut', (rows, cols, cols), pc, mask=1 << core))
            cells.append(cell('lut', (64, 64, 64), 64, mask=1 << core))
            for dims in [(16, 16, 16), (29, 37, 45), (64, 64, 64), (128, 128, 128),
                         (256, 256, 256), (1, 256, 64), (128, 32, 128)]:
                cells.append(cell('matmul', dims, mask=1 << core))
            for dims in [(16, 16, 16), (29, 37, 45), (64, 64, 64), (256, 256, 256)]:
                for mode in ('chain', 'separate'):
                    cells.append(cell('mm_bias_hsig', dims, mask=1 << core, mode=mode))
            for name in ('lut', 'matmul', 'mm_bias_hsig'):
                cells.append(cell(name, (64, 64, 64), mask=1 << core, path='host_io'))
    elif suite == 'concurrency':
        for name, dims, pc in [('lut', (32, 32, 32), 1), ('matmul', (128, 128, 128), 1),
                               ('mm_bias_hsig', (64, 64, 64), 1)]:
            for core in (1, 2, 4):
                cells.append(cell(name, dims, pc, mask=core))
            for workers in (2, 3, 4, 6, 8):
                for strategy, allowed in [('fixed', 7), ('split', 7), ('poll', 3), ('poll', 7), ('auto', 7)]:
                    cells.append(cell(name, dims, pc, workers, strategy, allowed=allowed))
            cells.append(cell(name, dims, pc, 3, 'split', start='burst'))
    elif suite == 'resident':
        cells = matrix('concurrency')
        for c in cells:
            c['capture'] = 'deferred'
    elif suite == 'resident-smoke':
        cells = [cell('lut', (32,32,32), workers=3, strategy='split'),
                 cell('matmul', (128,128,128), workers=3, strategy='fixed'),
                 cell('mm_bias_hsig', (64,64,64), workers=3, strategy='split')]
        for c in cells:
            c['capture'] = 'deferred'
    elif suite == 'saturation':
        for name in ('matmul','mm_bias_hsig'):
            dims=(256,256,256)
            for mask in (1,2,4):
                cells.append(cell(name,dims,mask=mask))
            for workers in (3,8):
                for strategy in ('fixed','split','auto'):
                    cells.append(cell(name,dims,workers=workers,strategy=strategy))
        for c in cells:
            c['capture']='deferred'
    elif suite == 'tail':
        for name, dims in [('lut', (32, 32, 32)), ('matmul', (256, 256, 256))]:
            cells.append(cell(name, dims))
            cells.append(cell(name, dims, workers=4, strategy='fixed'))
            cells.append(cell(name, dims, workers=4, strategy='split'))
        for c in cells:
            c['capture'] = 'deferred'
    elif suite == 'interference':
        for name in ('lut', 'matmul'):
            for core in (1, 2, 4):
                cells.append(cell(name, (64, 64, 64), 4 if name == 'lut' else 1, mask=core))
            for background in ('compute:1', 'compute:2', 'compute:4', 'stream:1', 'stream:2', 'stream:4'):
                cells.append(cell(name, (64, 64, 64), 4 if name == 'lut' else 1, background=background))
        for name in ('lut+matmul', 'matmul+lut'):
            cells.append(cell(name, (64, 64, 64), 4, 2, 'fixed'))
            for allowed in (3, 5, 6):
                cells.append(cell(name, (64, 64, 64), 4, 2, 'split', allowed=allowed))
        for name in ('lut', 'matmul'):
            for workers in (2, 3):
                cells.append(cell(name, (64, 64, 64), 4 if name == 'lut' else 1, workers, 'split'))
    elif suite == 'trace':
        cells = [cell('lut', (64, 64, 64), 16),
                 cell('lut', (64, 64, 64), 16, 4, 'fixed'),
                 cell('lut', (64, 64, 64), 16, 3, 'split')]
        for strategy in ('fixed','split'):
            c=cell('matmul',(256,256,256),workers=8,strategy=strategy)
            c['capture']='deferred';cells.append(c)
        for c in cells:
            c['trace'] = True
    else:
        raise ValueError('unknown suite')
    require(len({c['id'] for c in cells}) == len(cells), 'duplicate cell')
    return cells


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def source_manifest(out):
    files = sorted(p for p in PROJECT.rglob('*') if p.is_file() and
                   not any(part.startswith('build') or part in ('results', '__pycache__', '.git') for part in p.relative_to(PROJECT).parts))
    entries = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    dump(out/'source-manifest.json', entries)
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def environment():
    values = {'uname': subprocess.check_output(['uname', '-a'], text=True).strip(),
              'driver': pathlib.Path('/sys/kernel/debug/rknpu/version').read_text().strip()}
    for name, p in [('npu', pathlib.Path('/sys/class/devfreq/fdab0000.npu')),
                    ('dmc', pathlib.Path('/sys/class/devfreq/dmc'))]:
        values[name] = {k: (p/k).read_text().strip() for k in ('governor', 'cur_freq', 'min_freq', 'max_freq') if (p/k).exists()}
    values['cpu_affinity'] = list(os.sched_getaffinity(0))
    values['interrupts'] = pathlib.Path('/proc/interrupts').read_text()
    values['memory'] = pathlib.Path('/proc/meminfo').read_text()
    return values


class Sampler:
    def __init__(self, path):
        self.path, self.stop = path, threading.Event()
        self.thread = threading.Thread(target=self.run)

    def run(self):
        paths = [pathlib.Path('/sys/class/devfreq/fdab0000.npu/cur_freq'),
                 pathlib.Path('/sys/class/devfreq/dmc/cur_freq')]
        paths += list(pathlib.Path('/sys/class/thermal').glob('thermal_zone*/temp'))
        with self.path.open('w') as out:
            while not self.stop.is_set():
                row = {'mono_ns': time.monotonic_ns()}
                for p in paths:
                    try:
                        row[str(p)] = int(p.read_text().strip())
                    except (ValueError, OSError):
                        row[str(p)] = None
                out.write(json.dumps(row)+'\n'); out.flush()
                self.stop.wait(.25)

    def __enter__(self):
        self.thread.start(); return self

    def __exit__(self, *args):
        self.stop.set(); self.thread.join()


@contextlib.contextmanager
def controlled(out):
    with open('/tmp/rk3588-custom-ops.lock', 'w') as lock, open('/tmp/rknpu-global-trace.lock', 'w') as trace_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(trace_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require((TRACE/'current_tracer').read_text().strip() == 'nop' and
                (TRACE/'events/enable').read_text().strip() == '0', 'tracing already in use')
        require(pathlib.Path('/sys/kernel/debug/rknpu/version').read_text().strip() == 'RKNPU driver: v0.9.8', 'unsupported driver ABI')
        require(b'rockchip,rk3588' in pathlib.Path('/proc/device-tree/compatible').read_bytes(), 'campaign supports RK3588 only')
        saved, changed = {}, []
        dump(out/'environment-before.json', environment())
        try:
            npu = pathlib.Path('/sys/class/devfreq/fdab0000.npu')
            # Pin NPU min=max at the existing maximum, no voltage/overclock change.
            for p, value in [(npu/'min_freq', (npu/'max_freq').read_text().strip()),
                             (pathlib.Path('/sys/class/devfreq/dmc/governor'), 'performance')]:
                saved[str(p)] = p.read_text().strip(); changed.append(p)
                p.write_text(value+'\n')
                require(p.read_text().strip() == value, f'control readback failed {p}')
            for p in pathlib.Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_governor'):
                saved[str(p)] = p.read_text().strip(); changed.append(p);p.write_text('performance\n')
                require(p.read_text().strip() == 'performance', 'CPU governor readback')
            dump(out/'saved-state.json', saved)
            dump(out/'environment-fixed.json', environment())
            yield
        finally:
            errors = []
            for p in reversed(changed):
                try:
                    p.write_text(saved[str(p)]+'\n')
                    require(p.read_text().strip() == saved[str(p)], f'restore {p}')
                except Exception as e:
                    errors.append(str(e))
            dump(out/'restoration.json', {'restored': not errors, 'errors': errors})
            dump(out/'environment-after.json', environment())
            require(not errors, 'device control restoration failed')


@contextlib.contextmanager
def tracing(out):
    from schedule_capture import FUNCTIONS, ROOTS
    from schedule_kernel_audit import audit
    audit(out)
    names = ['current_tracer', 'tracing_on', 'trace_clock', 'buffer_size_kb', 'set_ftrace_filter', 'set_graph_function']
    saved = {k: (TRACE/k).read_text() for k in names}
    (out/'previous-trace.txt').write_text((TRACE/'trace').read_text())
    options = []
    try:
        (TRACE/'tracing_on').write_text('0\n')
        (TRACE/'current_tracer').write_text('function_graph\n')
        options = (TRACE/'trace_options').read_text().splitlines()
        for name, value in [('trace_clock', 'mono'), ('buffer_size_kb', '8192'),
                            ('set_ftrace_filter', '\n'.join(FUNCTIONS)), ('set_graph_function', '\n'.join(ROOTS))]:
            (TRACE/name).write_text(value+'\n')
        for option in ['funcgraph-abstime', 'funcgraph-proc', 'funcgraph-cpu', 'funcgraph-duration', 'funcgraph-tail', 'funcgraph-irqs', 'sleep-time']:
            (TRACE/'trace_options').write_text(option+'\n')
        (TRACE/'trace').write_text('\n'); (TRACE/'tracing_on').write_text('1\n')
        yield
    finally:
        (TRACE/'tracing_on').write_text('0\n')
        (out/'capture.trace').write_text((TRACE/'trace').read_text())
        dump(out/'trace-stats.json', {p.parent.name: p.read_text() for p in (TRACE/'per_cpu').glob('cpu*/stats')})
        for option in options:
            (TRACE/'trace_options').write_text(option+'\n')
        options_restored = (TRACE/'trace_options').read_text().splitlines() == options
        (TRACE/'current_tracer').write_text('nop\n')
        for name in ['set_ftrace_filter', 'set_graph_function']:
            (TRACE/name).write_text('\n'.join(x for x in saved[name].splitlines() if not x.startswith('#'))+'\n')
        (TRACE/'buffer_size_kb').write_text(saved['buffer_size_kb'])
        clock = saved['trace_clock'].split('[')[1].split(']')[0]
        (TRACE/'trace_clock').write_text(clock+'\n')
        (TRACE/'tracing_on').write_text(saved['tracing_on'])
        restored = options_restored and (TRACE/'current_tracer').read_text().strip() == 'nop'
        restored = restored and (TRACE/'tracing_on').read_text().strip() == saved['tracing_on'].strip()
        restored = restored and (TRACE/'buffer_size_kb').read_text().strip() == saved['buffer_size_kb'].strip()
        dump(out/'trace-restoration.json', {'restored': restored, 'saved': saved})
        require(restored, 'global tracer restoration failed')


def execute(binary, c, out):
    out.mkdir()
    cmd = [str(binary), c['workload'], *map(str, c['dims']), str(c['programs']), str(c['workers']),
           str(c['operations']), str(c['warmup']), str(c['seed']), c['mode'], c['path'], c['strategy'],
           str(c['mask']), str(c['allowed']), c['start'], str(c['affinity_base']), '2']
    dump(out/'command.json', cmd)
    env = dict(os.environ, ROCKET_CHIP='rk3588', COST_REQUEST_PREFIX=str(c['request_prefix']), COST_CAPTURE_MODE=c['capture'])
    if c['trace']:
        env['COST_TRACE_MARKER'] = str(TRACE/'trace_marker')
    background = None
    with (out/'raw.jsonl').open('w') as raw, (out/'stderr.txt').open('w') as err, (out/'background.jsonl').open('w') as bg:
        try:
            if c['background']:
                mode, count = c['background'].split(':')
                background = subprocess.Popen([str(binary.parent/'interference_load'), mode, count, str(c['timeout_s']+5), '256'], stdout=bg, stderr=err, start_new_session=True)
                deadline = time.monotonic()+10
                while time.monotonic() < deadline:
                    if background.poll() is not None:
                        raise RuntimeError('background exited before READY')
                    if (out/'background.jsonl').stat().st_size:
                        break
                    time.sleep(.05)
                else:
                    raise RuntimeError('background READY timeout')
                time.sleep(.5)
            with Sampler(out/'environment-samples.jsonl'), tracing(out) if c['trace'] else contextlib.nullcontext():
                proc = subprocess.Popen(cmd, stdout=raw, stderr=err, env=env, start_new_session=True)
                try:
                    code = proc.wait(timeout=c['timeout_s'])
                except BaseException:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                    raise
        finally:
            if background is not None:
                if background.poll() is None:
                    os.killpg(background.pid, signal.SIGTERM)
                try:
                    background.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(background.pid, signal.SIGKILL); background.wait()
                require(background.returncode == 0, 'background failed')
    return code


def journal(out, since):
    result = subprocess.run(['journalctl', '-k', '--since', since, '--no-pager'], capture_output=True, text=True)
    (out/'kernel-journal.txt').write_text(result.stdout+result.stderr)
    anomalies = [line for line in result.stdout.splitlines() if
                 re.search(r'rknpu|fdab.*iommu', line, re.I) and re.search(r'fault|timeout|reset|failed|error', line, re.I)]
    dump(out/'journal-status.json', {'exit_code': result.returncode, 'npu_anomalies': anomalies})
    return anomalies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=pathlib.Path)
    parser.add_argument('out', type=pathlib.Path)
    parser.add_argument('--suite', choices=('smoke', 'cost', 'concurrency', 'resident', 'resident-smoke', 'saturation', 'interference', 'trace', 'tail'), default='smoke')
    parser.add_argument('--blocks', type=int, default=5)
    parser.add_argument('--operations', type=int)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--timeout', type=int, default=120)
    args = parser.parse_args()
    if args.operations is None:
        args.operations = {'smoke':12,'trace':24,'tail':5000,'concurrency':600,'resident':600,'saturation':600,'resident-smoke':12}.get(args.suite,200)
    require(args.blocks >= 1 and args.operations >= 8 and args.warmup >= 0 and args.timeout >= 1, 'invalid budget')
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    cells = matrix(args.suite)
    for c in cells:
        c.update(operations=args.operations, warmup=args.warmup, seed=20261010, timeout_s=args.timeout)
        require(c['start'] != 'burst' or args.operations % c['workers'] == 0, 'burst operation budget must be divisible')
    plan = dict(schema_version=2, suite=args.suite, blocks=args.blocks, cells=cells,
                source_sha256=source_manifest(out), binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                claim='closed-loop controlled untraced service/Host cost; no queue assumption')
    from device_cost_schema import validate_config, enumerate_config
    validate_config(plan)
    plan['planned_budget'] = enumerate_config(plan)
    dump(out/'config.json', plan)
    command_start = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    try:
        with controlled(out), (out/'runs.jsonl').open('w') as ledger:
            rng = random.Random(20261010)
            for block in range(args.blocks):
                order = list(cells); rng.shuffle(order)
                for index, c in enumerate(order):
                    c = dict(c, request_prefix=block*len(cells)+index)
                    directory = out/f'block-{block:02d}-cell-{index:03d}'
                    code = execute(args.binary.resolve(), c, directory)
                    entry = dict(cell=c['id'], block=block, config=c, raw=str((directory/'raw.jsonl').relative_to(out)),
                                 raw_sha256=hashlib.sha256((directory/'raw.jsonl').read_bytes()).hexdigest(), exit_code=code)
                    ledger.write(json.dumps(entry)+'\n'); ledger.flush()
                    print(f"block={block} cell={index+1}/{len(cells)} exit={code} {c['id']}", flush=True)
                    require(code == 0, 'hardware run failed; evidence retained, no automatic retry')
        require(not journal(out, command_start), 'new kernel NPU fault/timeout/reset event')
        result = analyze(out)
        dump(out/'status.json', {'status': result['status']})
    except BaseException as e:
        dump(out/'status.json', {'status': 'FAIL', 'error': str(e)})
        raise
    finally:
        journal(out, command_start)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
