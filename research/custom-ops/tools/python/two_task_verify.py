#!/usr/bin/env python3
"""Independently verify real vendor ABI, IRQ, lifetime and numerical witnesses."""
from collections import Counter, defaultdict
import hashlib
import gzip
import json
import math
from pathlib import Path
import re
import struct
import sys


class Invalid(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Invalid(message)


def records(path):
    if not path.exists() and path.with_suffix(path.suffix + '.gz').exists():
        with gzip.open(path.with_suffix(path.suffix + '.gz'), 'rt') as stream:
            return [json.loads(line) for line in stream if line.strip()]
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def half(value):
    return struct.unpack('<e', struct.pack('<e', value))[0]


def hsig(value):
    return half(min(1, max(0, value / 6 + 0.5)))


def span(record, address, size, what):
    require(isinstance(size, int) and size > 0 and address >= record['dma']
            and address - record['dma'] + size <= record['size'], f'{what} outside BO')


def decode_raw(data):
    require(len(data) % 8 == 0, 'raw regcmd length')
    words = struct.unpack('<' + 'Q' * (len(data) // 8), data)
    require(len(words) >= 4, 'raw trailer missing')
    regs = {(w >> 48, w & 65535): (w >> 16) & 0xFFFFFFFF for w in words}
    trailer = [(w >> 48, w & 65535, (w >> 16) & 0xFFFFFFFF) for w in words[-4:]]
    require(trailer[0][:2] in ((0, 0), (0x101, 0x10)) and
            trailer[1][:2] == (0x101, 0x14) and trailer[2] == (0x41, 0, 0) and
            trailer[3] == (0x81, 8, 0x18), 'raw trailer shape')
    return dict(words=len(words), sptr_dpu=regs[(0x1001, 0x4004)],
                sptr_rdma=regs[(0x2001, 0x5004)], src=regs[(0x2001, 0x5018)],
                dst=regs[(0x1001, 0x4020)], data_bytes=(regs[(0x1001, 0x4030)] + 1) * 16,
                next=trailer[0][2], next_amount=trailer[1][2], enable=trailer[3][2])


def validate(path):
    probe = records(path / 'probe.jsonl')
    summaries = [r for r in probe if r['event'] == 'summary']
    require(len(summaries) == 1 and probe[-1] == summaries[0], 'missing/duplicate final summary')
    summary = summaries[0]
    rounds = [r for r in probe if r['event'] == 'round']
    require(summary['status'] == 'PASS' and summary['failures'] == 0, 'numerical run failed')
    require(summary['rounds'] == len(rounds) and len(rounds) >= 2, 'round count mismatch/truncated')
    require(summary['core_mask'] in (1, 2, 4) and summary['sptr'] in (14, 48), 'process configuration')
    mask, sptr = summary['core_mask'], summary['sptr']
    core = {1: 0, 2: 1, 4: 2}[mask]
    shapes = ((16, 32), (29, 45), (64, 64))
    for index, r in enumerate(rounds):
        require(r['round'] == index and (r['rows'], r['cols']) == shapes[index % 3]
                and r['pattern'] == index % 2 and r['pass'] is True, 'round order/shape/input alternation')
        require([a['arm'] for a in r['arms']] == list('ABCD'), 'missing/reordered arm')
        for a in r['arms']:
            expected = (2, 2, 3) if a['arm'] == 'B' else (1, 1, 1) if a['arm'] == 'A' else (1, 2, 3)
            require((a['submits'], a['tasks'], a['checks']) == expected, 'arm submit/task/check count')
            irq = [0, 0, 0]; irq[core] = expected[0]
            require(a['irq_delta'] == irq, 'IRQ completion count/core mismatch')
            require(a['pass'] is True and math.isfinite(a['max_abs']) and 0 <= a['max_abs'] <= 0.002,
                    'numerical error/nonfinite arm')
            require(a['padding_guard'] is True, 'physical cube padding corruption')
        require(r['checks'] == 10, 'round check count')
    require(summary['checks'] == len(rounds) * 10, 'summary check count')
    lines = [s for s in (path / 'probe.log').read_text().splitlines() if s.startswith('CHECK ')]
    require(len(lines) == summary['checks'] and all(re.search(r'bad=0 PASS$', s) for s in lines),
            'raw numerical log disagrees with summary')

    live, seen = {}, set()
    active = None
    stage = None
    completed = defaultdict(list)
    bank_counts = Counter()
    created = 0
    max_wall = 0
    trace = records(path / 'trace.jsonl')
    require(trace and trace[-1]['event'] == 'end', 'trace incomplete')
    for expected_id, event in enumerate(trace, 1):
        require(event['event_id'] == expected_id, 'trace event loss/reorder')
        kind = event['event']
        if kind == 'create':
            require(active is None and event['id'] not in seen and event['size'] > 0, 'allocation reuse/creation during submit')
            require(all((a['fd'], a['obj']) != (event['fd'], event['obj']) for a in live.values()), 'live object alias')
            seen.add(event['id']); live[event['id']] = dict(event, mapped=False); created += 1
        elif kind in ('map', 'mmap', 'unmap', 'destroy', 'sync'):
            require(active is None, f'{kind} during active submit (lifetime/intermediate CPU touch)')
            require(event['id'] in live, f'{kind} of non-live allocation')
            obj = live[event['id']]
            if 'fd' in event:
                require(event['fd'] == obj['fd'], 'memory fd identity mismatch')
            if kind == 'map':
                obj['map_offset'] = event['offset']
            elif kind == 'mmap':
                require(not obj['mapped'] and event['length'] >= obj['size'], 'mapping bounds/re-map')
                obj.update(mapped=True, address=event['address'], length=event['length'])
            elif kind == 'unmap':
                require(obj['mapped'] and event['address'] == obj['address'] and event['length'] == obj['length'], 'unmap identity/range')
                obj['mapped'] = False
            elif kind == 'destroy':
                require(event['result'] == 0 and event['obj'] == obj['obj'] and not obj['mapped'], 'destroy result/object/mapping')
                del live[event['id']]
            elif kind == 'sync':
                require(event['result'] == 0 and event['flags'] in (1, 2) and event['offset'] >= 0
                        and event['size'] >= 0 and event['offset'] + event['size'] <= obj['size'], 'sync bounds/flags')
        elif kind == 'stage':
            require(active is None, 'stage change while active')
            stage = (event['round'], event['arm'])
        elif kind == 'submit_begin':
            require(active is None and stage == (event['round'], event['arm']), 'submit stage/overlap')
            arm = event['arm']; wanted = 1 if arm in ('A', 'B') else 2
            require(arm in 'ABCD' and event['count'] == wanted and event['flags'] == 1 and
                    event['core_mask'] == mask and event['base'] == 0, 'submit configuration')
            require(event['task_size'] == 40 and event['task_bytes'] == wanted * 40
                    and event['task_offset'] == event['task_start'] * 40, 'descriptor ABI/stride')
            require(event['task_id'] in live, 'descriptor BO not live')
            obj = live[event['task_id']]
            require(obj['mapped'] and obj['obj'] == event['task_obj'] and obj['fd'] == event['fd']
                    and obj['domain'] == event['domain'] and event['task_offset'] >= 0
                    and event['task_offset'] + event['task_bytes'] <= obj['size'], 'descriptor BO bounds/identity')
            active = dict(event, programs=[], refs={event['task_id']}, subcore=False)
        elif kind == 'subcore':
            require(active and event['submit_id'] == active['submit_id'] and not active['subcore'], 'subcore event')
            require(event['index'] == core and event['start'] == active['task_start']
                    and event['count'] == active['count'], 'effective subcore range mismatch')
            active['subcore'] = True
        elif kind == 'program':
            require(active and event['submit_id'] == active['submit_id'] and event['valid'] is True, 'unresolved program')
            index = len(active['programs'])
            require(event['index'] == index and index < active['count'] and event['desc_bytes'] == 40
                    and event['desc_offset'] == active['task_offset'] + 40 * index, 'descriptor index/boundary')
            require(event['words'] == event['amount'] + 4 and event['words'] > 4
                    and event['fetch_bytes'] == ((event['words'] + 1) // 2) * 16, 'fetch length encoding')
            require(event['rc_id'] in live, 'regcmd BO not live')
            obj = live[event['rc_id']]
            require(obj['mapped'] and event['rc_offset'] >= 0 and event['regcmd'] % 16 == 0
                    and event['regcmd'] == obj['dma'] + event['rc_offset'], 'regcmd address/offset/alignment')
            span(obj, event['regcmd'], event['fetch_bytes'], 'regcmd fetch')
            active['refs'].add(event['rc_id'])
            rows, cols = shapes[active['round'] % 3]
            physical_rows = ((max(rows, 12) + 3) // 4) * 4
            physical_cols = ((cols + 31) // 32) * 32
            require(event['data_bytes'] == physical_rows * physical_cols * 2, 'cube dimensions/span mismatch')
            for name in ('src', 'dst'):
                require(event[name + '_id'] in live, f'{name} BO not live')
                obj = live[event[name + '_id']]
                require(obj['mapped'] and obj['fd'] == active['fd'] and obj['domain'] == active['domain'], 'operand domain/fd/map')
                span(obj, event[name], event['data_bytes'], name)
                active['refs'].add(event[name + '_id'])
            require(event['src_id'] != event['dst_id'] and event['enable'] == 0x18 and
                    event['int_mask'] == 0x300 and event['sptr_dpu'] == sptr and event['sptr_rdma'] == sptr,
                    'DPU enable/completion/group/operand alias')
            raw = path / f"program-{active['submit_id']:04d}-{index}.bin"
            if active['submit_id'] <= 15:
                require(raw.exists(), 'missing raw command witness')
                decoded = decode_raw(raw.read_bytes())
                require(all(event[key] == value for key, value in decoded.items()), 'raw regcmd contradicts metadata')
            active['programs'].append(event)
        elif kind == 'submit_end':
            require(active and event['submit_id'] == active['submit_id'] and active['subcore'], 'unpaired submit end')
            require(event['result'] == 0 and event['errno'] == 0 and event['hw_ns'] > 0 and
                    0 < event['wall_ns'] < 1_500_000_000 and event['live'] is True, 'failed/slow/no-work submit')
            require(event['descriptor_guard'] is True, 'task descriptor BO changed outside last int_status')
            require(active['refs'] <= live.keys(), 'BO freed before completion')
            programs = active['programs']
            require(len(programs) == active['count'], 'program count incomplete')
            for index, program in enumerate(programs):
                if index + 1 < len(programs):
                    following = programs[index + 1]
                    require(program['rc_id'] == following['rc_id'] and
                            following['rc_offset'] == program['rc_offset'] + program['fetch_bytes'] and
                            program['next'] == following['regcmd'] and
                            program['next_amount'] == (following['words'] + 1) // 2 - 1, 'chain address/next length/stride')
                else:
                    require(program['next'] == 0, 'terminal forward link not sealed')
            arm = active['arm']
            if arm in ('C', 'D'):
                first, second = programs
                if arm == 'C':
                    require(first['src_id'] != second['src_id'] and first['dst_id'] != second['dst_id'] and
                            first['dst_id'] != second['src_id'] and second['dst_id'] != first['src_id'], 'C not independent')
                else:
                    require(second['src_id'] == first['dst_id'] and second['src'] == first['dst'], 'D cube handoff not exact alias')
            require(event['last_status'] in (0x100, 0x200, 0x300), 'missing/wrong DPU completion event')
            active['last_status'] = event['last_status']; active['wall_ns'] = event['wall_ns']
            bank_counts[event['last_status']] += 1; max_wall = max(max_wall, event['wall_ns'])
            completed[(active['round'], arm)].append(active)
            active = None
        elif kind == 'end':
            require(active is None and not live and event['live_allocations'] == 0, 'unfinished work/leaked BOs')
        else:
            raise Invalid(f'unknown trace event: {kind}')
    require(active is None and len(completed) == len(rounds) * 4, 'missing stage execution')
    previous_a = None
    for r in rounds:
        number = r['round']
        for arm in 'ABCD':
            require(len(completed[(number, arm)]) == (2 if arm == 'B' else 1), 'actual submit count per arm')
        a = completed[(number, 'A')][0]['last_status']
        b0, b1 = [s['last_status'] for s in completed[(number, 'B')]]
        a0 = completed[(number, 'A')][0]['programs'][0]
        independent = [s['programs'][0] for s in completed[(number, 'B')]]
        require(independent[0]['src_id'] != independent[1]['src_id'] and
                independent[0]['dst_id'] != independent[1]['dst_id'] and
                independent[0]['dst_id'] != independent[1]['src_id'] and
                independent[1]['dst_id'] != independent[0]['src_id'], 'B not independent')
        for arm in 'BCD':
            first = completed[(number, arm)][0]['programs'][0]
            require((first['src_id'], first['src']) == (a0['src_id'], a0['src']), 'task0 input changed between arms')
        second = completed[(number, 'C')][0]['programs'][1]
        require((second['src_id'], second['src']) == (independent[1]['src_id'], independent[1]['src']), 'B/C independent input mismatch')
        if sptr == 48:
            require(all(s['last_status'] == 0x100 for arm in 'ABCD' for s in completed[(number, arm)]), 'fixed group completion not group0')
        else:
            require(a in (0x100, 0x200) and b0 in (0x100, 0x200) and b1 in (0x100, 0x200)
                    and b0 != a and b1 == a and (previous_a is None or a != previous_a), 'ping-pong completion groups did not alternate')
            for arm in 'CD':
                require(completed[(number, arm)][0]['last_status'] in (a, 0x300), 'even-task chain group continuity')
        previous_a = a

    witness_checks = 0
    for r in rounds[:3]:
        number, count = r['round'], r['rows'] * r['cols']
        def read(label):
            data = (path / f'round-{number:04d}-{label}.bin').read_bytes()
            require(len(data) == count * 2, 'tensor witness truncated')
            return data, struct.unpack('<' + 'e' * count, data)
        _, xa = read('input-a'); _, xb = read('input-b')
        ra = [hsig(x) for x in xa]; rb = [hsig(x) for x in xb]; rd = [hsig(x) for x in ra]
        a_bytes, _ = read('arm-A-task-0')
        for arm in 'ABCD':
            for index in range(1 if arm == 'A' else 2):
                data, got = read(f'arm-{arm}-task-{index}')
                ref = ra if index == 0 else rd if arm == 'D' else rb
                require(all(math.isfinite(g) and abs(g-e) <= 0.002 for g,e in zip(got,ref)), 'independent tensor witness oracle failed')
                if index == 0:
                    require(data == a_bytes, 'cross-arm task0 tensor mismatch')
                witness_checks += 1
    return dict(rounds=len(rounds), checks=summary['checks'], submits=len(rounds)*5,
                tasks=len(rounds)*7, core_mask=mask, sptr=sptr, created=created,
                last_status_counts={str(key):value for key,value in bank_counts.items()}, max_submit_wall_ns=max_wall,
                independent_tensor_witness_checks=witness_checks)


def verify_process(path):
    path = Path(path)
    try:
        stats = validate(path)
        errors = []
    except (Invalid, OSError, KeyError, TypeError, ValueError, IndexError, struct.error) as error:
        stats, errors = {}, [str(error)]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.iterdir())
              if p.is_file() and p.name != 'validation.json'} if path.exists() else {}
    return dict(status='FAIL' if errors else 'PASS', errors=errors, stats=stats, sha256=hashes)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: two_task_verify.py PROCESS_DIR')
    result = verify_process(sys.argv[1])
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['status'] == 'PASS' else 1)
