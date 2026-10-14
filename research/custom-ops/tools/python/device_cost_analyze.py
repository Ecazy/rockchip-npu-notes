#!/usr/bin/env python3
"""Strict v2 raw-evidence gate and isolated empirical service prediction.

Per-operation metrics and per-Submit metrics are deliberately distinct.
No tracing => queue is unobserved. Software stream bytes are not DDR counters.
"""
import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import statistics


def require(ok, why):
    if not ok:
        raise ValueError(why)


def integer(v):
    return isinstance(v, int) and not isinstance(v, bool)


def quantile(values, q):
    values = sorted(values)
    require(values, "empty quantile")
    index = (len(values)-1)*q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi]-values[lo])*(index-lo)


def distribution(values):
    return {"n": len(values), "p50": quantile(values, .5),
            "p95": quantile(values, .95), "p99": quantile(values, .99),
            "max": max(values)}


def validate_run(records, config):
    submits = [r for r in records if r["type"] == "submit"]
    operations = [r for r in records if r["type"] == "operation"]
    campaigns = [r for r in records if r["type"] == "campaign"]
    require(len(campaigns) == 1 and campaigns[0]["result"] == 0, "missing/failed campaign")
    require(all(r["type"] in ("submit", "operation", "campaign") for r in records), "unknown record")
    workers, budget, warmup = config["workers"], config["operations"], config["warmup"]
    require(campaigns[0]["operations"] == budget and campaigns[0]["workers"] == workers,
            "campaign configuration mismatch")
    require(campaigns[0]["warmup_per_worker"] == warmup, "warmup mismatch")
    require(len(operations) == budget+warmup*workers, "missing/extra operation")
    ids, opids = set(), set()
    byop = collections.defaultdict(list)
    for row in submits:
        for name in ('result','bad_count','requested_mask','worker','round','submit_index','submit_count','operation_id'):
            require(integer(row[name]), 'invalid integer '+name)
        rid = row["request_id"]
        require(integer(rid) and rid not in ids, "duplicate/invalid request ID")
        ids.add(rid)
        require(row["result"] == 0 and row["bad_count"] == 0 and row["guard_ok"] is True,
                "failed numeric/guard/submit")
        require(not isinstance(row['max_abs_error'],bool) and isinstance(row["max_abs_error"], (float, int)) and math.isfinite(row["max_abs_error"]), "nonfinite/error type")
        require(isinstance(row['warmup'],bool), 'invalid warmup flag')
        for name in ("host_begin_ns", "ioctl_begin_ns", "ioctl_end_ns", "host_done_ns", "driver_service_ns",
                     "command_bytes", "estimated_bytes", "logical_macs", "physical_macs", "task_count"):
            require(integer(row[name]) and row[name] >= 0, f"invalid {name}")
        begin, ib, ie, end = (row[k] for k in ("host_begin_ns", "ioctl_begin_ns", "ioctl_end_ns", "host_done_ns"))
        require(begin <= ib <= ie <= end, "Host clock order")
        require(0 < row["driver_service_ns"] <= ie-ib, "service outside blocking ioctl")
        require(row["command_bytes"] > 0 and 1 <= row["task_count"] <= 64, "empty program")
        require(integer(row["actual_mask"]) and row["actual_mask"] in (1, 2, 4), "invalid actual core")
        require(integer(row["int_status"]) and row["int_status"] & 0x300, "missing DPU completion")
        wi = row["worker"]
        require(integer(wi) and 0 <= wi < workers, "unknown worker")
        expected = {"fixed": config["mask"], "auto": 0}.get(config["strategy"])
        if config['strategy'] == 'split':
            allowed = [1 << i for i in range(3) if config['allowed'] & (1 << i)]
            expected = allowed[wi % len(allowed)]
        if config["strategy"] == "poll":
            allowed = [1 << i for i in range(3) if config["allowed"] & (1 << i)]
            expected = allowed[(wi+row["round"]) % len(allowed)]
        require(row["requested_mask"] == expected, "wrong policy request mask")
        require(expected == 0 or row["actual_mask"] == expected, "core binding mismatch")
        programs = program_features(config, wi)
        name = config['workload'].split('+')[wi % len(config['workload'].split('+'))]
        require(0 <= row['max_abs_error'] <= (.01 if name == 'matmul' else .002), 'score exceeds numeric contract')
        selected = programs if config['mode'] == 'chain' else [programs[row['submit_index']]]
        require(row['task_count'] == len(selected), 'actual task count differs from configured workload')
        for field in ('logical_macs', 'physical_macs', 'estimated_bytes'):
            require(row[field] == sum(p[field] for p in selected), f'wrong per-Submit {field}')
        byop[row["operation_id"]].append(row)
    measured = []
    for row in operations:
        for name in ('operation_id','result','bad_count','worker','round','begin_ns','upload_end_ns','submit_end_ns','end_ns','program_count','submit_count','completed_submits'):
            require(integer(row[name]), 'invalid operation integer '+name)
        require(not isinstance(row['max_abs_error'],bool) and isinstance(row['max_abs_error'],(int,float)) and math.isfinite(row['max_abs_error']), 'nonfinite operation error')
        oid = row["operation_id"]
        require(oid not in opids, "duplicate operation ID")
        opids.add(oid)
        require(row["result"] == 0 and row["bad_count"] == 0, "operation verification failed")
        require(isinstance(row['warmup'], bool), 'invalid warmup flag')
        require(row["begin_ns"] <= row["upload_end_ns"] <= row["submit_end_ns"] <= row["end_ns"], "operation clock order")
        group = byop.pop(oid, [])
        count = row["submit_count"]
        require(row["completed_submits"] == count and len(group) == count, "missing operation Submit")
        require(sorted(s["submit_index"] for s in group) == list(range(count)), "missing/duplicate Submit index")
        require(all(s["submit_count"] == count and s["warmup"] == row["warmup"] and
                    s["worker"] == row["worker"] and s["round"] == row["round"] for s in group), "operation association mismatch")
        require(sum(s["task_count"] for s in group) == row["program_count"], "program count mismatch")
        require(row['program_count'] == len(program_features(config, row['worker'])), 'configured program missing')
        require((count == 1 if config["mode"] == "chain" else count == row["program_count"]), "wrong Submit packaging")
        require(row["begin_ns"] <= min(s["host_begin_ns"] for s in group) and
                max(s["host_done_ns"] for s in group) <= row["end_ns"], "Submit outside operation")
        if row["warmup"] is False:
            measured.append(row)
    require(not byop and len(measured) == budget, "orphan/missing measured operation")
    for wi in range(workers):
        rows = [r for r in operations if r["worker"] == wi]
        expected = budget//workers+(wi < budget % workers)
        require(sum(r["warmup"] is False for r in rows) == expected and
                sum(r["warmup"] is True for r in rows) == warmup, "worker budget mismatch")
    require(len({(r["worker"], r["round"]) for r in operations}) == len(operations), "duplicate worker round")
    return [r for r in submits if not r["warmup"]], measured


def program_features(config, worker=0):
    m, k, n = config['dims']
    pm = (max(m, 12)+3)//4*4
    pk, pn = (k+31)//32*32, (n+31)//32*32
    name = config['workload'].split('+')[worker % len(config['workload'].split('+'))]
    if name == 'lut':
        require(pm*pn//8 < 8191, 'single LUT exceeds emitter limit')
        return [dict(logical_macs=0, physical_macs=0, estimated_bytes=4*pm*pn)]*config['programs']
    mm = dict(logical_macs=m*k*n, physical_macs=pm*pk*pn,
              estimated_bytes=2*(pm*pk+pn*pk+pm*pn))
    if name == 'matmul':
        return [mm]*(1 if '+' in config['workload'] else config['programs'])
    ew = dict(logical_macs=0, physical_macs=pm*pn*pn, estimated_bytes=6*pm*pn+2*pn*pn)
    luts = [dict(logical_macs=0, physical_macs=0, estimated_bytes=4*min(32768, pm*pn-offset)) for offset in range(0, pm*pn, 32768)]
    return [mm, ew, *luts]


def summarize_run(records, config):
    submits, operations = validate_run(records, config)
    span = max(r["end_ns"] for r in operations)-min(r["begin_ns"] for r in operations)
    require(span > 0, "zero measured window")
    per_worker = {}
    for wi in range(config["workers"]):
        rows = [r for r in operations if r["worker"] == wi]
        sub = [r for r in submits if r['worker'] == wi]
        ordered = sorted(rows, key=lambda r: r['round'])
        gaps = [(b['begin_ns']-a['end_ns'])/1000 for a,b in zip(ordered,ordered[1:])]
        per_worker[str(wi)] = {"n": len(rows), "latency_us": distribution([(r["end_ns"]-r["begin_ns"])/1000 for r in rows]),
                              'service_us': distribution([r['driver_service_ns']/1000 for r in sub]),
                              'think_us': distribution(gaps) if gaps else None,
                              'actual_masks': sorted({r['actual_mask'] for r in sub})}
    result = {"operations": len(operations), "submits": len(submits),
              "tasks": sum(r["task_count"] for r in submits), "makespan_ns": span,
              "operations_per_s": len(operations)*1e9/span, "submits_per_s": len(submits)*1e9/span,
              "operation_us": distribution([(r["end_ns"]-r["begin_ns"])/1000 for r in operations]),
              "response_us": distribution([(r["ioctl_end_ns"]-r["host_begin_ns"])/1000 for r in submits]),
              "api_response_us": distribution([(r["host_done_ns"]-r["host_begin_ns"])/1000 for r in submits]),
              "service_us": distribution([r["driver_service_ns"]/1000 for r in submits]),
              "prepare_us": distribution([(r["ioctl_begin_ns"]-r["host_begin_ns"])/1000 for r in submits]),
              "post_submit_us": distribution([(r["host_done_ns"]-r["ioctl_end_ns"])/1000 for r in submits]),
              "actual_core_counts": dict(collections.Counter(str(r["actual_mask"]) for r in submits)),
              "queue": None, "queue_reason": "untraced: no queue observation", "per_worker": per_worker,
              "max_abs_error": max(r["max_abs_error"] for r in operations),
              "payload_bytes_per_worker": max(r["payload_bytes"] for r in operations),
              "allocated_tensor_bo_bytes_per_worker": max(r["allocated_tensor_bo_bytes"] for r in operations)}
    # Jain on a common active interval; final equal quotas are not fairness.
    start = max(min(r["begin_ns"] for r in operations if r["worker"] == wi) for wi in range(config["workers"]))
    end = min(max(r["end_ns"] for r in operations if r["worker"] == wi) for wi in range(config["workers"]))
    counts = [sum(start <= r["begin_ns"] and r["end_ns"] <= end and r["worker"] == wi for r in operations) for wi in range(config["workers"])]
    result["steady_jain"] = sum(counts)**2/(len(counts)*sum(v*v for v in counts)) if end > start and all(counts) and "+" not in config["workload"] else None
    result["steady_counts"] = counts
    result['request_overlap_proxy_ns'] = max(0, end-start)
    result['request_overlap_fraction'] = max(0, end-start)/span
    first = [r for r in records if r['type']=='submit' and r['round']==0]
    result['process_first_submit'] = {'service_us': distribution([r['driver_service_ns']/1000 for r in first]),
                                     'api_response_us': distribution([(r['host_done_ns']-r['host_begin_ns'])/1000 for r in first]),
                                     'scope': 'first Submit per worker/process; hardware cache coldness unobserved'}
    return result, submits


def features(r):
    return [1.0, r["task_count"], r["physical_macs"]/1e6,
            r["estimated_bytes"]/1e4, r["command_bytes"]/1e3]


def fit(rows):
    n = len(features(rows[0])); matrix = [[0.0]*(n+1) for _ in range(n)]
    for r in rows:
        x, y = features(r), r["driver_service_ns"]/1000
        for i in range(n):
            for j in range(n):
                matrix[i][j] += x[i]*x[j]
            matrix[i][-1] += x[i]*y
    for i in range(n):
        matrix[i][i] += .01
    for i in range(n):
        pivot = max(range(i, n), key=lambda j: abs(matrix[j][i]))
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        require(abs(matrix[i][i]) > 1e-15, "singular regression")
        divisor = matrix[i][i]
        matrix[i] = [v/divisor for v in matrix[i]]
        for j in range(n):
            if j != i:
                scale = matrix[j][i]
                matrix[j] = [a-scale*b for a, b in zip(matrix[j], matrix[i])]
    return [matrix[i][-1] for i in range(n)]


def predict(coefficients, row):
    return max(.001, sum(a*b for a, b in zip(coefficients, features(row))))


def model_rows(rows):
    """Fit whole-geometry holdout; never randomly split repeat requests."""
    grouped = collections.defaultdict(list)
    for r in rows:
        if r["config"]["workers"] == 1 and r["config"]["strategy"] == "fixed" and not r["config"]["background"] and not r['config'].get('trace'):
            family = r["config"]["workload"]
            # separate bias/LUT are different program kinds, not all MatMul.
            program = "chain" if r["config"]["mode"] == "chain" else f"separate-{r['submit_index']}"
            grouped[(family, program, r["actual_mask"])].append(r)
    models = {}
    for key, group in grouped.items():
        shapes = sorted({tuple(r["config"]["dims"]) for r in group})
        if len(shapes) < 4:
            models[str(key)] = {"status": "insufficient_geometry", "shapes": len(shapes)}
            continue
        shuffled = list(shapes); random.Random(20261010).shuffle(shuffled)
        holdout = set(shuffled[:max(1, math.ceil(len(shapes)*.2))])
        train = [r for r in group if tuple(r["config"]["dims"]) not in holdout]
        test = [r for r in group if tuple(r["config"]["dims"]) in holdout]
        # Fit cell/block medians instead of giving long campaigns extra weight.
        cells = collections.defaultdict(list)
        for r in train:
            cells[(tuple(r["config"]["dims"]), r["task_count"], r["block"])].append(r)
        medians = []
        for values in cells.values():
            row = dict(values[0]); row["driver_service_ns"] = statistics.median(r["driver_service_ns"] for r in values)
            row['api_response_ns'] = statistics.median(r['host_done_ns']-r['host_begin_ns'] for r in values)
            medians.append(row)
        coefficients = fit(medians)
        host_medians = [dict(r, driver_service_ns=r['api_response_ns']) for r in medians]
        host_coefficients = fit(host_medians)
        errors = [abs(predict(coefficients, r)-r["driver_service_ns"]/1000)/(r["driver_service_ns"]/1000) for r in test]
        baseline = statistics.median(r["driver_service_ns"]/1000 for r in train)
        base_errors = [abs(baseline-r["driver_service_ns"]/1000)/(r["driver_service_ns"]/1000) for r in test]
        p50, p90 = quantile(errors, .5), quantile(errors, .9)
        host_errors = [abs(predict(host_coefficients, r)-(r['host_done_ns']-r['host_begin_ns'])/1000)/((r['host_done_ns']-r['host_begin_ns'])/1000) for r in test]
        last_block = max(r['block'] for r in group)
        earlier = [r for r in group if r['block'] < last_block]
        later = [r for r in group if r['block'] == last_block]
        temporal = None
        if earlier:
            time_coefficients = fit(earlier)
            time_errors = [abs(predict(time_coefficients, r)-r['driver_service_ns']/1000)/(r['driver_service_ns']/1000) for r in later]
            temporal = {'holdout_block': last_block, 'relative_error_p50': quantile(time_errors, .5), 'relative_error_p90': quantile(time_errors, .9)}
        models[str(key)] = {"status": "validated" if p50 <= .15 and p90 <= .30 else "experimental",
                            "coefficients": coefficients, "ridge_lambda": .01,
                            "workload": key[0], "program": key[1], "core_mask": key[2],
                            "api_response_coefficients": host_coefficients,
                            "api_response_relative_p50": quantile(host_errors, .5),
                            "api_response_relative_p90": quantile(host_errors, .9),
                            "feature_bounds": {name: [min(r[name] for r in group), max(r[name] for r in group)] for name in ['task_count','physical_macs','estimated_bytes','command_bytes']},
                            "unit": "microseconds", "train_shapes": [list(s) for s in shapes if s not in holdout],
                            "holdout_shapes": [list(s) for s in holdout], "holdout_requests": len(test),
                            "relative_error_p50": p50, "relative_error_p90": p90,
                            "absolute_error_mae_us": statistics.mean(abs(predict(coefficients, r)-r["driver_service_ns"]/1000) for r in test),
                            "constant_baseline_relative_p50": quantile(base_errors, .5),
                            "temporal_validation": temporal,
                            "scope": "isolated untraced driver service envelope only"}
    return {"schema_version": 2, "features": ["intercept", "tasks", "physical_macs/1e6", "estimated_bytes/1e4", "command_bytes/1e3"],
            "models": models, "queue_model": None}


class DeviceCostModel:
    """Bounded offline prediction. Experimental status is propagated to callers."""
    def __init__(self, path):
        self.document = json.loads(pathlib.Path(path).read_text())
        require(self.document['schema_version'] == 2, 'model version')

    def predict(self, workload, core_mask, program='chain', **features):
        key = str((workload, program, core_mask))
        model = self.document['models'].get(key)
        require(model and model['status'] != 'insufficient_geometry', 'unsupported model family')
        for name, bounds in model['feature_bounds'].items():
            require(name in features and bounds[0] <= features[name] <= bounds[1], 'out_of_domain '+name)
        return dict(driver_service_us=predict(model['coefficients'], features),
                    api_response_us=predict(model['api_response_coefficients'], features),
                    status=model['status'], queue_us=None,
                    scope='isolated calibrated feature domain; unseen feature combinations unverified')


def analyze(root):
    root = pathlib.Path(root)
    ledger = [json.loads(line) for line in (root/"runs.jsonl").read_text().splitlines()]
    require(ledger, "empty run ledger")
    summaries, allrows = [], []
    for entry in ledger:
        require(entry["exit_code"] == 0, "campaign contains failed run")
        raw = root/entry['raw']
        if 'raw_sha256' in entry:
            require(hashlib.sha256(raw.read_bytes()).hexdigest() == entry['raw_sha256'], 'raw evidence hash mismatch')
        records = [json.loads(line) for line in raw.read_text().splitlines()]
        summary, rows = summarize_run(records, entry["config"])
        if entry['config'].get('trace'):
            from schedule_analyze import analyze as trace_analyze, summary as trace_summary
            trace_root = (root/entry['raw']).parent
            converted = []
            for r in records:
                if r['type'] == 'submit':
                    converted.append(dict(r, id=r['request_id'], bad=r['bad_count'], allowed_mask=entry['config']['allowed']))
            bounds = trace_analyze(converted, (trace_root/'capture.trace').read_text(), json.loads((trace_root/'trace-stats.json').read_text()))
            (trace_root/'queue-bounds.json').write_text(json.dumps(bounds, indent=2)+'\n')
            summary['queue'] = trace_summary([r for r in bounds if not r['warmup']])
            summary['queue_reason'] = 'controlled audited function graph, interval bounds'
        summary.update({"cell": entry["cell"], "block": entry["block"], "config": entry["config"]})
        summaries.append(summary)
        for row in rows:
            row.update({"config": entry["config"], "block": entry["block"], "cell": entry["cell"]})
            allrows.append(row)
    # In addition to per-run gates, demand the entire planned ledger exactly.
    plan = json.loads((root/"config.json").read_text())
    expected = {(c["id"], b) for c in plan["cells"] for b in range(plan["blocks"])}
    actual = [(r["cell"], r["block"]) for r in summaries]
    require(len(actual) == len(set(actual)) and set(actual) == expected, "incomplete/duplicate campaign matrix")
    require(json.loads((root/"restoration.json").read_text())["restored"] is True, "device configuration not restored")
    global_ids = [r['request_id'] for r in allrows]
    # Earliest smoke used run-local IDs; newer producers explicitly prefix them.
    if all('request_prefix' in e['config'] for e in ledger):
        require(len(global_ids) == len(set(global_ids)), 'request ID collision across runs')
    result = {"schema_version": 2, "status": "PASS", "runs": summaries,
              "measured_submits": len(allrows), "source_sha256": plan["source_sha256"],
              "claim_boundary": "untraced performance and separate audited tracing diagnostics; service is driver envelope"}
    if (root/'journal-status.json').exists():
        journal = json.loads((root/'journal-status.json').read_text())
        require(not journal.get('npu_anomalies'), 'kernel NPU anomaly')
        result['kernel_journal_available'] = journal['exit_code'] == 0
        if (root/'review-journal-status.json').exists():
            review = json.loads((root/'review-journal-status.json').read_text())
            require(review['exit_code']==0 and not review['npu_anomalies'], 'supplemental kernel observation failed')
            result['kernel_journal_available']=True
            result['kernel_journal_source']='supplemental read-only time-window review'
    result['comparisons'] = comparisons(summaries)
    result['policy_comparisons'] = policy_comparisons(summaries)
    if plan['suite'] == 'interference':
        calibration = interference_calibration(summaries)
        (root/'interference-model.json').write_text(json.dumps(calibration, indent=2)+'\n')
    (root/"analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    model = model_rows(allrows)
    model['ledger_sha256'] = hashlib.sha256((root/'runs.jsonl').read_bytes()).hexdigest()
    model['raw_manifest_sha256'] = hashlib.sha256(json.dumps({e['raw']: hashlib.sha256((root/e['raw']).read_bytes()).hexdigest() for e in ledger},sort_keys=True).encode()).hexdigest()
    (root/"model.json").write_text(json.dumps(model, indent=2)+"\n")
    return result


def bootstrap(values, seed=20261010):
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000)]
    return [quantile(means, .025), quantile(means, .975)]


def policy_comparisons(summaries):
    """Same worker count/CPU placement/workload/budget; vary NPU policy only."""
    fixed = {}
    def key(run):
        c=run['config']
        return (c['workload'],tuple(c['dims']),c['programs'],c['workers'],c['operations'],c['mode'],c['path'],c['start'],c.get('capture','immediate'),run['block'])
    for run in summaries:
        if run['config']['strategy']=='fixed' and run['config']['mask']==1 and not run['config']['background']:
            fixed[key(run)]=run
    pairs=collections.defaultdict(list)
    for run in summaries:
        if run['config']['strategy']!='fixed' and key(run) in fixed:
            base=fixed[key(run)]
            pairs[run['cell']].append((base['makespan_ns']/run['makespan_ns'],run['service_us']['p50']/base['service_us']['p50']))
    return [dict(cell=cell,matched_blocks=len(values),same_worker_speedup=statistics.mean(v[0] for v in values),
                 speedup_ci95=bootstrap([v[0] for v in values]),service_ratio=statistics.mean(v[1] for v in values)) for cell,values in pairs.items()]


def comparisons(summaries):
    baseline = {}
    for run in summaries:
        c = run['config']
        if c['workers'] == 1 and c['strategy'] == 'fixed' and not c['background'] and not c.get('trace'):
            key = (c['workload'], tuple(c['dims']), c['programs'], c['mask'], c['mode'], c['path'], run['block'])
            baseline[key] = run
    cells = collections.defaultdict(list)
    for run in summaries:
        if not run['config'].get('trace'):
            cells[run['cell']].append(run)
    result = []
    for cid, runs in cells.items():
        c = runs[0]['config']; names = c['workload'].split('+')
        for wi in range(c['workers']):
            name = names[wi % len(names)]
            ratios, response_ratios, speedups = [], [], []
            for run in runs:
                worker = run['per_worker'][str(wi)]
                masks = worker['actual_masks']
                if len(masks) != 1:
                    continue  # AUTO/poll require per-actual-core conditional baselines.
                pc = 1 if name == 'matmul' and len(names) > 1 else c['programs']
                base = baseline.get((name, tuple(c['dims']), pc, masks[0], c['mode'], c['path'], run['block']))
                if base:
                    ratios.append(worker['service_us']['p50']/base['service_us']['p50'])
                    response_ratios.append(worker['latency_us']['p50']/base['operation_us']['p50'])
                    if len(names) == 1:
                        speedups.append(base['makespan_ns']/run['makespan_ns'])
            if ratios:
                result.append(dict(cell=cid, worker=wi, matched_blocks=len(ratios),
                    service_slowdown=statistics.mean(ratios), service_slowdown_ci95=bootstrap(ratios),
                    operation_slowdown=statistics.mean(response_ratios),
                    equal_work_speedup=statistics.mean(speedups) if speedups else None,
                    equal_work_speedup_ci95=bootstrap(speedups) if speedups else None,
                    warning='request overlap is a Host proxy; DDR/CBUF cause is unproven'))
    return result


def interference_calibration(summaries):
    """Multipliers for measured conditions with one complete block held out.
    No unseen victim/competitor or shape generalization is claimed.
    """
    byblock = collections.defaultdict(list)
    for run in summaries:
        byblock[run['block']].append(run)
    last = max(byblock)
    training = comparisons([r for r in summaries if r['block'] < last])
    testing = {(r['cell'], r['worker']): r for r in comparisons(byblock[last])}
    rows = []
    for train in training:
        test = testing.get((train['cell'], train['worker']))
        if test:
            rows.append(dict(cell=train['cell'],worker=train['worker'],service_multiplier=train['service_slowdown'],
                             multiplier_ci95=train['service_slowdown_ci95'],holdout_multiplier=test['service_slowdown'],
                             holdout_relative_error=abs(train['service_slowdown']-test['service_slowdown'])/test['service_slowdown']))
    errors = [r['holdout_relative_error'] for r in rows]
    # Additional exploratory combination holdout: use heterogeneous NPU pairs
    # to predict homogeneous pairs, and 1/4 streaming threads to predict 2.
    configs = {r['cell']: r['config'] for r in summaries}
    cross_train = collections.defaultdict(list)
    for r in training:
        c=configs[r['cell']]
        if '+' in c['workload'] and c['strategy']=='split':
            family=c['workload'].split('+')[r['worker']%2]
            cross_train[family].append(r['service_slowdown'])
    combinations=[]
    for test in comparisons(summaries):
        c=configs[test['cell']]
        if '+' not in c['workload'] and c['workers']>=2 and c['strategy']=='split' and not c['background'] and cross_train[c['workload']]:
            predicted=statistics.mean(cross_train[c['workload']])
            combinations.append(dict(cell=test['cell'],worker=test['worker'],kind='unseen_homogeneous_NPU_pair',
                predicted_multiplier=predicted,observed_multiplier=test['service_slowdown'],
                relative_error=abs(predicted-test['service_slowdown'])/test['service_slowdown']))
        if c['background']=='stream:2':
            alternatives=[]
            for count in (1,4):
                match=next((r for r in training if configs[r['cell']]['workload']==c['workload'] and configs[r['cell']]['background']==f'stream:{count}'),None)
                if match:alternatives.append(match['service_slowdown'])
            if len(alternatives)==2:
                predicted=alternatives[0]+(alternatives[1]-alternatives[0])/3
                combinations.append(dict(cell=test['cell'],worker=test['worker'],kind='unseen_stream_thread_count_2',
                    predicted_multiplier=predicted,observed_multiplier=test['service_slowdown'],
                    relative_error=abs(predicted-test['service_slowdown'])/test['service_slowdown']))
    combo_errors=[r['relative_error'] for r in combinations]
    return dict(schema_version=2,status='calibrated_conditions_only',holdout_block=last,
                heldout_combinations=len({r['cell'] for r in combinations}),relative_error_p50=quantile(errors,.5) if errors else None,
                relative_error_p90=quantile(errors,.9) if errors else None,conditions=rows,
                exploratory_combination_validation=combinations,
                combination_relative_p50=quantile(combo_errors,.5) if combo_errors else None,
                combination_relative_p90=quantile(combo_errors,.9) if combo_errors else None,
                validation_note='exploratory post-hoc fixed constant/linear baselines, not pre-registered generalization proof',
                scope='64x64 measured conditions; unknown shapes, loads and counterfactual DDR bandwidth unsupported')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=pathlib.Path)
    args = parser.parse_args()
    result = analyze(args.root)
    print(json.dumps({k: result[k] for k in ("status", "measured_submits", "claim_boundary")}, indent=2))
