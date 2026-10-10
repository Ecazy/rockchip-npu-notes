#!/usr/bin/env python3
"""Replay prior single-device raw submit records through the simulator.

Evidence consumed
-----------------
* ``raw.jsonl`` (or a block directory holding it) with one ``submit`` record per
  blocking ioctl.  Required integer-ns fields: ``host_begin_ns``,
  ``ioctl_begin_ns``, ``ioctl_end_ns``, ``host_done_ns``, ``driver_service_ns``.
* Optional interval evidence: a block-local ``queue-bounds.json`` (written by
  the existing tracing analysis) or ``schedule_analysis.json``.  Each matched
  record then also carries ``driver_entry_ns``, ``enqueue_ns``, ``start_ns`` and
  ``completion_ns`` as ``[lo, hi]`` integer-ns hulls.

What is validated
-----------------
* Host order ``host_begin <= ioctl_begin <= ioctl_end <= host_done``.
* ``0 < service <= ioctl_end - ioctl_begin`` (service fits the blocking ioctl).
* When tracing is present: interval order and hulls, service consistent with
  ``completion``/``start``, ``completion <= ioctl_end``, queue interval equals
  the derived hull, and the observed per-core FIFO (enqueue order matches start
  order).  Any contradiction raises ``ValueError`` and stops the replay.

Accounting
----------
``response        = ioctl_end  - host_begin``
``api_response    = host_done  - host_begin``
``prepare         = ioctl_begin - host_begin``
``ioctl_blocking  = ioctl_end - ioctl_begin``
``ioctl_residual  = ioctl_blocking - service``   (UNKNOWN queue + host work)
``post            = host_done - ioctl_end``

``ioctl_end - service`` is explicitly **not** claimed to be an exact commit
time.  Untraced queueing is unobserved, so ``queue_ns`` stays ``null``; the
residual is reported as an unknown combined bucket, never promoted to a queue.
Traced records report the audited ``queue_ns`` interval only.

The observed release (``ioctl_begin_ns`` untraced, ``enqueue_ns[0]`` traced) and
the observed service are handed to the simulator as a single FIFO device.  The
resulting synthetic schedule is compared against the observed-compatible
schedule: service durations must match exactly and the synthetic interval must
fit inside the observed envelope.  This is an internal-consistency check, not a
prediction of unseen hardware schedules (see ``limits``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from simulator import validate_scenario, Simulator  # noqa: E402

SCHEMA_VERSION = 1
TOLERANCE_NS = 0  # audited hulls already encode timestamp quantisation
INTERVAL_FILES = ("queue-bounds.json", "schedule_analysis.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _int_field(record, name):
    value = record.get(name)
    require(_int(value) and value >= 0, f"request {record.get('request_id')}: invalid {name}")
    return value


def _interval_field(record, name):
    value = record.get(name)
    require(isinstance(value, list) and len(value) == 2, f"request {record.get('request_id')}: bad {name}")
    lo, hi = value
    require(_int(lo) and _int(hi) and 0 <= lo <= hi, f"request {record.get('request_id')}: disordered {name}")
    return lo, hi


def read_jsonl(path):
    records = []
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def validate_submit(record):
    require(record.get("type") == "submit", "non-submit record")
    require(record.get('result', 0) == 0 and record.get('bad_count', 0) == 0
            and record.get('guard_ok', True) is True, "failed Submit cannot be replayed")
    rid = record.get("request_id")
    require(_int(rid) and rid >= 0, "invalid request_id")
    host_begin = _int_field(record, "host_begin_ns")
    ioctl_begin = _int_field(record, "ioctl_begin_ns")
    ioctl_end = _int_field(record, "ioctl_end_ns")
    host_done = _int_field(record, "host_done_ns")
    service = _int_field(record, "driver_service_ns")
    require(host_begin <= ioctl_begin <= ioctl_end <= host_done,
            f"request {rid}: host clock order corrupted")
    require(0 < service <= ioctl_end - ioctl_begin,
            f"request {rid}: service does not fit the blocking ioctl")
    require(_int(record.get("task_count", 1)) and record.get("task_count", 1) >= 1,
            f"request {rid}: invalid task_count")
    return record


def _core_of(mask):
    require(_int(mask) and mask in (1, 2, 4), f"invalid actual_mask {mask!r}")
    return mask.bit_length() - 1


def validate_intervals(record):
    rid = record["request_id"]
    service = record["driver_service_ns"]
    ioctl_begin, ioctl_end = record["ioctl_begin_ns"], record["ioctl_end_ns"]
    entry = _interval_field(record, "driver_entry_ns")
    enqueue = _interval_field(record, "enqueue_ns")
    start = _interval_field(record, "start_ns")
    completion = _interval_field(record, "completion_ns")
    require(entry[0] >= ioctl_begin, f"request {rid}: driver entry before ioctl begin")
    require(enqueue[0] >= entry[0], f"request {rid}: enqueue before driver entry")
    require(start[0] >= enqueue[0], f"request {rid}: start before enqueue")
    require(completion[0] >= start[0], f"request {rid}: completion before start")
    require(completion[1] <= ioctl_end + TOLERANCE_NS,
            f"request {rid}: completion after host observation")
    require(completion[0] - start[1] <= service <= completion[1] - start[0],
            f"request {rid}: driver service inconsistent with trace hulls")
    derived_queue = [max(0, start[0] - enqueue[1]), max(0, start[1] - enqueue[0])]
    if "queue_ns" in record and record["queue_ns"] is not None:
        queue = _interval_field(record, "queue_ns")
        require(list(queue) == derived_queue,
                f"request {rid}: queue interval contradicts enqueue/start hulls")
    if "core" in record:
        require(_int(record["core"]) and record["core"] == _core_of(record["actual_mask"]),
                f"request {rid}: core does not match actual_mask")
    if "irq_core" in record:
        require(record["irq_core"] == _core_of(record["actual_mask"]),
                f"request {rid}: irq core does not match actual_mask")
    return derived_queue


def check_core_fifo(rows, intervals_present):
    """Observed per-core FIFO: enqueue order must equal start order when traced.

    Untraced records expose no timing, so FIFO is unobservable there and the
    check is skipped rather than guessed.
    """
    if not intervals_present:
        return {"observable": False, "reason": "untraced: core FIFO timing unobserved"}
    per_core = {}
    for row in rows:
        core = _core_of(row["actual_mask"])
        per_core.setdefault(core, []).append(row)
    ambiguous = 0
    for core, members in per_core.items():
        ordered = sorted(members, key=lambda r: r['start_ns'][0])
        enqueue_witness = 0
        for row in ordered:
            enqueue_witness = max(enqueue_witness, row['enqueue_ns'][0])
            require(enqueue_witness <= row['enqueue_ns'][1],
                    f'core {core}: enqueue order contradicts execution order')
        for previous, nxt in zip(ordered, ordered[1:]):
            require(previous['start_ns'][1] < nxt['start_ns'][0],
                    f'core {core}: execution order ambiguous')
            if previous['enqueue_ns'][1] >= nxt['enqueue_ns'][0]:
                ambiguous += 1
    return {'observable':True, 'cores':sorted(per_core), 'execution_order_unique':True,
            'unique_enqueue_order':ambiguous == 0, 'overlapping_enqueue_pairs':ambiguous,
            'scope':'definite execution order with feasible FIFO enqueue hulls; overlapping hull order not identified'}


def account(record, queue_interval):
    rid = record["request_id"]
    host_begin, ioctl_begin = record["host_begin_ns"], record["ioctl_begin_ns"]
    ioctl_end, host_done = record["ioctl_end_ns"], record["host_done_ns"]
    service = record["driver_service_ns"]
    ioctl_blocking = ioctl_end - ioctl_begin
    residual = ioctl_blocking - service
    require(residual >= 0, f"request {rid}: negative ioctl residual")
    response = ioctl_end - host_begin
    require(service <= response, f"request {rid}: service exceeds response")
    return {
        "request_id": rid,
        "worker": record.get("worker"),
        "core": _core_of(record["actual_mask"]),
        "task_count": record.get("task_count"),
        "host_begin_ns": host_begin,
        "ioctl_begin_ns": ioctl_begin,
        "ioctl_end_ns": ioctl_end,
        "host_done_ns": host_done,
        "prepare_ns": ioctl_begin - host_begin,
        "ioctl_blocking_ns": ioctl_blocking,
        "ioctl_service_ns": service,
        "ioctl_residual_ns": residual,
        "post_ns": host_done - ioctl_end,
        "response_ns": response,
        "api_response_ns": host_done - host_begin,
        "queue_ns": None if queue_interval is None else list(queue_interval),
        "queue_status": "unobserved" if queue_interval is None else "audited_interval_bound",
        # ioctl_end - service is NOT an exact commit timestamp.
        "residual_scope": "ioctl_residual = queue + host work, unresolved",
        "commit_estimate_ns": None,
    }


def _release_ns(record, intervals_present):
    if intervals_present:
        return record["enqueue_ns"][0]
    return record["ioctl_begin_ns"]


def build_synthetic(scenario_rows, device_id, intervals_present):
    """Run the observed release/service through the FIFO simulator.

    Returns ``{job_id: {start_ns, end_ns, service_ns}}``.
    """
    base = min(_release_ns(r, intervals_present) for r in scenario_rows)
    device_name = "replay-device"
    capacity = 1  # no tensors/workspace modeled in timing-only replay
    jobs = []
    for index, row in enumerate(scenario_rows):
        jobs.append({
            "id": f"submit-{index}",
            "arrival_ns": _release_ns(row, intervals_present) - base,
            "depends_on": [],
            "inputs": [],
            "outputs": [],
            "workspace_bytes": 0,
            "durations_ns": {device_name: row["driver_service_ns"]},
            "eligible_devices": [device_name],
            "task_count": row.get("task_count", 1),
        })
    document = {
        "schema_version": 1,
        "devices": [{"id": device_name, "memory_capacity_bytes": capacity, "measured": True}],
        "links": [],
        "tensors": [],
        "jobs": jobs,
    }
    scenario = validate_scenario(document)
    result = Simulator(scenario, "shortest_queue").run()
    require(result["status"] == "ok", f"synthetic replay failed: {result['failure']}")
    schedule = {}
    for index, job in enumerate(result["jobs"]):
        require(job.get("state", "done") == "done", "synthetic replay left unfinished jobs")
        schedule[f"submit-{index}"] = {
            "start_ns": job["exec_start_ns"] + base,
            "end_ns": job["end_ns"] + base,
            "service_ns": job["compute_ns"],
            "device": device_id,
        }
    return schedule


def replay_records(records, intervals=None, worker=None):
    submits = [r for r in records if r.get('type') == 'submit' and not r.get('warmup', False)]
    if worker is not None:
        submits = [r for r in submits if r.get('worker', 0) == worker]
    require(submits, 'no measured submit records')
    require(len({r['request_id'] for r in submits}) == len(submits), 'duplicate raw request IDs')
    bound_index = {}
    if intervals is not None:
        for row in intervals:
            if row.get('type') == 'submit':
                require(row['request_id'] not in bound_index, 'duplicate interval request IDs')
                bound_index[row['request_id']] = row
        require(bound_index, 'empty interval evidence')
    traced = intervals is not None
    observed = []
    for row in submits:
        validate_submit(row)
        merged = dict(row)
        if traced:
            require(row['request_id'] in bound_index, 'missing interval request')
            bound = bound_index[row['request_id']]
            for key in row.keys() & bound.keys():
                require(row[key] == bound[key], f"request {row['request_id']}: raw/interval mismatch {key}")
            merged.update(bound)
            validate_intervals(merged)
        observed.append(merged)
    fifo = check_core_fifo(observed, traced)
    observed.sort(key=lambda r: (_core_of(r['actual_mask']), r['start_ns'][0] if traced else _release_ns(r, False), r['request_id']))
    requests, total_service, lower_starts, lower_ends = [], 0, [], []
    # RK3588 core queues are timing resources within one physical shared-DDR device.
    for mask in sorted({r['actual_mask'] for r in observed}):
        rows = [r for r in observed if r['actual_mask'] == mask]
        lower = build_synthetic(rows, mask, traced)
        witness_rows, previous_end = [], 0
        for i, row in enumerate(rows):
            item = account(row, validate_intervals(row) if traced else None)
            syn = lower[f'submit-{i}']
            require(syn['service_ns'] == row['driver_service_ns'], 'simulator service mismatch')
            require(syn['end_ns'] <= row['ioctl_end_ns'], 'lower-bound replay exceeds observed response')
            item.update(synthetic_start_ns=syn['start_ns'], synthetic_end_ns=syn['end_ns'],
                        synthetic_compatible=True, synthetic_scope='earliest-release lower-bound reference')
            lower_starts.append(syn['start_ns']); lower_ends.append(syn['end_ns'])
            if traced:
                service = row['driver_service_ns']
                start = max(row['start_ns'][0], row['completion_ns'][0]-service,
                            row['enqueue_ns'][0], previous_end)
                latest = min(row['start_ns'][1], row['completion_ns'][1]-service)
                require(start <= latest, f"core {mask}: no FIFO timeline satisfies trace hulls")
                previous_end = start + service
                witness = dict(row, enqueue_ns=[start, start])
                witness_rows.append(witness)
                item.update(feasible_start_ns=start, feasible_end_ns=previous_end,
                            feasible_scope='one interval-consistent reconstruction; not exact hardware timestamps')
            requests.append(item); total_service += row['driver_service_ns']
        if traced:
            reconstructed = build_synthetic(witness_rows, mask, True)
            for i, row in enumerate(witness_rows):
                require(reconstructed[f'submit-{i}']['start_ns'] == row['enqueue_ns'][0],
                        'simulator interval-witness event order differs')
    return {'schema_version':1, 'source':None, 'worker':worker, 'kind':'traced' if traced else 'untraced',
            'requests':requests,
            'summary': {'requests':len(requests), 'cores': sorted({_core_of(r['actual_mask']) for r in observed}),
                        'service_sum_ns':total_service, 'service_median_ns':_median([r['driver_service_ns'] for r in observed]),
                        'observed_span_ns':max(r['ioctl_end_ns'] for r in observed)-min(r['ioctl_begin_ns'] for r in observed),
                        'synthetic_makespan_ns':max(lower_ends)-min(lower_starts),
                        'service_durations_match':True, 'synthetic_observed_compatible':True,
                        'interval_reconstruction_pass':True if traced else None,
                        'queue_observed':traced, 'core_fifo':fifo,
                        'ioctl_residual_median_ns':_median([r['ioctl_residual_ns'] for r in requests])},
            'limits':limits_text(traced) + ['Core queues share physical DDR; capacities are not inferred from timing.',
                     'The earliest-release schedule is a lower-bound reference, not a claim of observed start.',
                     'Traced replay additionally checks a feasible start/completion reconstruction within existing hulls.']}


def _median(values):
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) // 2


def limits_text(traced):
    common = [
        "Synthetic schedules are derived from the same observed service durations, so exact agreement validates accounting/FIFO, not predictive power.",
        "task_count and submit packaging are metadata; this replay does not reconstruct register commands or hardware start.",
    ]
    if traced:
        return common + [
            "queue_ns is an audited function-graph interval; ioctl_end-service is NOT an exact commit time.",
            "Some queue interval width is irreducible (microsecond trace quantisation); the residual still contains host work.",
        ]
    return common + [
        "Untraced queueing is unobserved: queue_ns is null and ioctl_residual is a combined queue+host bucket, never a queue estimate.",
        "The release proxy is ioctl_begin_ns; it is not an observed device enqueue time.",
    ]


def load_interval_evidence(block_dir):
    for name in INTERVAL_FILES:
        path = block_dir / name
        if path.exists():
            document = json.loads(path.read_text())
            require(isinstance(document, list), f"{name} must be a list of submit records")
            return document, str(path)
    return None, None


def resolve_block(root, want_trace=None):
    root = pathlib.Path(root)
    if root.is_file():
        return root.parent, None
    if (root / "raw.jsonl").exists():
        return root, None
    if (root / "runs.jsonl").exists():
        ledger = read_jsonl(root / "runs.jsonl")
        for entry in ledger:
            if entry.get("exit_code") != 0:
                continue
            config = entry.get("config", {})
            if config.get("workers") != 1 or config.get("strategy") != "fixed":
                continue
            if config.get("background"):
                continue
            if want_trace is True and not config.get("trace"):
                continue
            if want_trace is False and config.get("trace"):
                continue
            block = root / pathlib.Path(entry["raw"]).parent
            if (block / "raw.jsonl").exists():
                return block, config
        raise ValueError("no matching single-worker fixed block in campaign")
    raise ValueError(f"no raw.jsonl or runs.jsonl under {root}")


def replay(path, worker=None, want_trace=None):
    block, config = resolve_block(path, want_trace)
    records = read_jsonl(block / "raw.jsonl")
    intervals, interval_path = load_interval_evidence(block)
    report = replay_records(records, intervals, worker=worker)
    report["source"] = str(block)
    report['source_sha256'] = hashlib.sha256((block/'raw.jsonl').read_bytes()).hexdigest()
    if interval_path:
        report['interval_sha256'] = hashlib.sha256(pathlib.Path(interval_path).read_bytes()).hexdigest()
    report["config"] = config
    report["interval_evidence"] = interval_path
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replay single-device raw submit records")
    parser.add_argument("path", type=pathlib.Path,
                        help="raw.jsonl, a block directory, or a campaign directory")
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--trace", dest="want_trace", action="store_true", default=None,
                        help="require traced blocks when resolving a campaign")
    parser.add_argument("--no-trace", dest="want_trace", action="store_false",
                        help="require untraced blocks when resolving a campaign")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    report = replay(args.path, worker=args.worker, want_trace=args.want_trace)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    if not args.quiet:
        print(json.dumps({"source": report["source"], "kind": report["kind"],
                          "summary": report["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
