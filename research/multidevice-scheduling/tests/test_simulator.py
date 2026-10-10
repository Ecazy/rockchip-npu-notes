#!/usr/bin/env python3
"""Unit tests for the multi-device deterministic scheduler simulator."""

import copy
import sys
import unittest
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from simulator import (  # noqa: E402
    DeadlockError,
    POLICIES,
    SimulationError,
    load_scenario,
    simulate,
    simulate_all,
    transfer_ns,
    validate_scenario,
)

HERE = _Path(__file__).resolve().parent
EXAMPLE = HERE.parent / "examples" / "two_device_scenario.json"


def doc(devices=None, links=None, tensors=None, jobs=None):
    """A minimal valid one-device scenario, overridable per test."""
    return {
        "schema_version": 1,
        "devices": devices if devices is not None else [
            {"id": "d0", "memory_capacity_bytes": 1_000_000, "measured": True}],
        "links": links if links is not None else [
            {"src": "host", "dst": "d0",
             "bandwidth_bytes_per_s": 1_000_000_000, "latency_ns": 5}],
        "tensors": tensors if tensors is not None else [
            {"id": "x", "size_bytes": 100, "source": "host"}],
        "jobs": jobs if jobs is not None else [
            {"id": "j0", "arrival_ns": 0, "depends_on": [], "inputs": ["x"],
             "outputs": [], "workspace_bytes": 0, "durations_ns": {"d0": 1000},
             "eligible_devices": ["d0"], "task_count": 1}],
    }


def job(jid, inputs=(), outputs=(), deps=(), duration=10, device="d0",
        arrival=0, workspace=0):
    return {
        "id": jid, "arrival_ns": arrival, "depends_on": list(deps),
        "inputs": list(inputs), "outputs": list(outputs),
        "workspace_bytes": workspace, "durations_ns": {device: duration},
        "eligible_devices": [device], "task_count": 1,
    }


class TransferTests(unittest.TestCase):
    def test_ceiling_and_latency(self):
        self.assertEqual(transfer_ns(60, 1_000_000_000, 5), 65)
        self.assertEqual(transfer_ns(0, 1_000_000_000, 7), 7)
        self.assertEqual(transfer_ns(1, 3, 0), 333_333_334)
        self.assertEqual(transfer_ns(1_000_000_000, 1_000_000_000, 0), 1_000_000_000)

    def test_rejects_bad_inputs(self):
        for args in ((1, 0, 0), (-1, 1, 0), (1, 1, -1), (True, 1, 0)):
            with self.subTest(args=args), self.assertRaises(SimulationError):
                transfer_ns(*args)


class ValidationTests(unittest.TestCase):
    def test_minimal_scenario_ok(self):
        scenario = validate_scenario(doc())
        self.assertEqual(list(scenario.devices), ["d0"])
        self.assertEqual(scenario.jobs["j0"].durations, {"d0": 1000})

    def test_schema_and_duplicates(self):
        with self.assertRaises(SimulationError):
            validate_scenario({**doc(), "schema_version": 2})
        bad = doc(devices=[{"id": "d0", "memory_capacity_bytes": 10, "measured": True},
                           {"id": "d0", "memory_capacity_bytes": 10, "measured": True}])
        with self.assertRaises(SimulationError):
            validate_scenario(bad)

    def test_unknown_link_endpoint(self):
        bad = doc(links=[{"src": "host", "dst": "nope",
                          "bandwidth_bytes_per_s": 1, "latency_ns": 0}])
        with self.assertRaises(SimulationError):
            validate_scenario(bad)

    def test_unknown_dependency_and_producer(self):
        bad = doc(tensors=[{"id": "x", "size_bytes": 10, "source": "host"}],
                  jobs=[job("j0", inputs=["x"], deps=["ghost"])])
        with self.assertRaises(SimulationError):
            validate_scenario(bad)
        bad = doc(tensors=[{"id": "x", "size_bytes": 10, "producer": "nobody"}],
                  jobs=[job("j0", inputs=["x"])])
        with self.assertRaises(SimulationError):
            validate_scenario(bad)

    def test_output_requires_declared_producer(self):
        bad = doc(tensors=[{"id": "x", "size_bytes": 10, "producer": "other"}],
                  jobs=[job("j0", outputs=["x"], duration=5)])
        with self.assertRaises(SimulationError):
            validate_scenario(bad)

    def test_cycle_rejected(self):
        bad = doc(
            tensors=[{"id": "x", "size_bytes": 10, "source": "host"}],
            jobs=[job("j0", inputs=["x"], deps=["j1"]),
                  job("j1", inputs=["x"], deps=["j0"])])
        with self.assertRaises(SimulationError):
            validate_scenario(bad)

    def test_missing_measured_duration_rejected(self):
        bad = doc(jobs=[{"id": "j0", "arrival_ns": 0, "depends_on": [],
                         "inputs": ["x"], "outputs": [], "workspace_bytes": 0,
                         "durations_ns": {}, "eligible_devices": ["d0"]}])
        with self.assertRaises(SimulationError):
            validate_scenario(bad)

    def test_oversize_footprint_rejected(self):
        bad = doc(
            devices=[{"id": "d0", "memory_capacity_bytes": 100, "measured": True}],
            tensors=[{"id": "x", "size_bytes": 200, "source": "host"}],
            jobs=[job("j0", inputs=["x"])])
        with self.assertRaises(SimulationError):
            validate_scenario(bad)

    def test_host_input_requires_link(self):
        bad = doc(links=[], tensors=[{"id": "x", "size_bytes": 10, "source": "host"}],
                  jobs=[job("j0", inputs=["x"])])
        with self.assertRaises(SimulationError):
            validate_scenario(bad)


class TimelineTests(unittest.TestCase):
    def test_single_job_timeline(self):
        result = simulate(validate_scenario(doc()), "round_robin")
        self.assertEqual(result["status"], "ok")
        entry = result["jobs"][0]
        self.assertEqual(entry["exec_start_ns"], 0)
        self.assertEqual(entry["transfer_bytes"], 100)
        self.assertEqual(entry["transfer_time_ns"], 105)
        self.assertEqual(entry["compute_start_ns"], 105)
        self.assertEqual(entry["compute_ns"], 1000)
        self.assertEqual(entry["end_ns"], 1105)
        self.assertEqual(result["makespan_ns"], 1105)

    def test_dependency_serialises_jobs(self):
        scenario = doc(
            tensors=[{"id": "x", "size_bytes": 10, "source": "host"},
                     {"id": "t", "size_bytes": 10, "producer": "j0"}],
            jobs=[job("j0", inputs=["x"], outputs=["t"], duration=100),
                  job("j1", inputs=["t"], deps=["j0"], duration=50)])
        result = simulate(validate_scenario(scenario), "round_robin")
        first, second = result["jobs"]
        self.assertEqual(first["end_ns"], 15 + 100)
        self.assertEqual(second["ready_ns"], first["end_ns"])
        # The produced tensor stays on d0, so j1 needs no host transfer.
        self.assertEqual(second["transfer_time_ns"], 0)
        self.assertEqual(second["end_ns"], first["end_ns"] + 50)

    def test_link_transfers_serialise(self):
        scenario = doc(
            tensors=[{"id": "a", "size_bytes": 1000, "source": "host"},
                     {"id": "b", "size_bytes": 1000, "source": "host"}],
            links=[{"src": "host", "dst": "d0",
                    "bandwidth_bytes_per_s": 1_000_000_000, "latency_ns": 0}],
            jobs=[job("j0", inputs=["a", "b"], duration=10)])
        result = simulate(validate_scenario(scenario), "round_robin")
        entry = result["jobs"][0]
        self.assertEqual(entry["transfer_bytes"], 2000)
        self.assertEqual(entry["transfer_time_ns"], 2000)
        self.assertEqual(entry["compute_start_ns"], 2000)
        self.assertEqual(result["links"]["host->d0"]["busy_ns"], 2000)
        self.assertEqual(result["links"]["host->d0"]["transfers"], 2)

    def test_equal_time_events_keep_insertion_order(self):
        scenario = doc(
            devices=[{"id": "d0", "memory_capacity_bytes": 1000, "measured": True},
                     {"id": "d1", "memory_capacity_bytes": 1000, "measured": True}],
            links=[{"src": "host", "dst": "d0", "bandwidth_bytes_per_s": 1, "latency_ns": 0},
                   {"src": "host", "dst": "d1", "bandwidth_bytes_per_s": 1, "latency_ns": 0}],
            tensors=[{"id": "x", "size_bytes": 1, "source": "host"}],
            jobs=[{"id": "j0", "arrival_ns": 0, "depends_on": [], "inputs": [],
                   "outputs": [], "workspace_bytes": 0, "durations_ns": {"d0": 100},
                   "eligible_devices": ["d0"], "task_count": 1},
                  {"id": "j1", "arrival_ns": 0, "depends_on": [], "inputs": [],
                   "outputs": [], "workspace_bytes": 0, "durations_ns": {"d1": 100},
                   "eligible_devices": ["d1"], "task_count": 1}])
        result = simulate(validate_scenario(scenario), "round_robin")
        done = [e for e in result["events"] if e["type"] == "compute_done"]
        self.assertEqual([e["time_ns"] for e in done], [100, 100])
        self.assertEqual([e["job"] for e in done], ["j0", "j1"])

    def test_deterministic_across_policies(self):
        scenario = load_scenario(EXAMPLE)
        first = simulate_all(scenario)
        second = simulate_all(scenario)
        for policy in POLICIES:
            self.assertEqual(first[policy], second[policy], policy)
            self.assertEqual(first[policy]["status"], "ok")
            self.assertEqual(len(first[policy]["jobs"]), len(scenario.jobs))

    def test_round_robin_spreads_two_devices(self):
        result = simulate(load_scenario(EXAMPLE), "round_robin")
        devices = {entry["device"] for entry in result["jobs"]}
        self.assertEqual(devices, {"rk3588", "rk182x"})


class MemoryTests(unittest.TestCase):
    @staticmethod
    def deadlock_scenario():
        # Every job fits at validation time, but when ``h`` is admitted the
        # still-referenced homes ``t`` (60) and ``p`` (40) are already resident:
        # 100 + 60 > 130 and neither can be evicted (``t`` is an input, ``p`` has
        # a future consumer on the delayed job ``c``).
        return doc(
            devices=[{"id": "d0", "memory_capacity_bytes": 130, "measured": True}],
            links=[],
            tensors=[{"id": "t", "size_bytes": 60, "producer": "g"},
                     {"id": "p", "size_bytes": 40, "producer": "g2"},
                     {"id": "o", "size_bytes": 60, "producer": "h"}],
            jobs=[job("g", outputs=["t"]),
                  job("g2", outputs=["p"]),
                  job("k", inputs=["t"]),
                  job("h", inputs=["t"], outputs=["o"]),
                  job("c", inputs=["p"], arrival=10_000_000)])

    def test_runtime_deadlock_reported(self):
        scenario = self.deadlock_scenario()
        result = simulate(validate_scenario(scenario), "round_robin")
        self.assertEqual(result["status"], "failed")
        self.assertIn("deadlock", result["failure"])
        with self.assertRaises(DeadlockError):
            simulate(validate_scenario(scenario), "round_robin", strict=True)

    def test_referenced_home_tensor_never_evicted(self):
        result = simulate(validate_scenario(self.deadlock_scenario()), "round_robin")
        self.assertFalse(any(e["type"] == "evict" for e in result["events"]))

    def test_unreferenced_home_tensor_is_evicted(self):
        scenario = doc(
            devices=[{"id": "d0", "memory_capacity_bytes": 100, "measured": True}],
            links=[{"src": "host", "dst": "d0",
                    "bandwidth_bytes_per_s": 1_000_000_000, "latency_ns": 0}],
            tensors=[{"id": "r", "size_bytes": 40, "producer": "p"},
                     {"id": "u", "size_bytes": 40, "source": "host"}],
            jobs=[job("p", outputs=["r"]),
                  job("q", inputs=["u"], workspace=40)])
        result = simulate(validate_scenario(scenario), "round_robin", strict=True)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(any(e["type"] == "evict" and e.get("tensor") == "r"
                            for e in result["events"]))

    def test_workspace_counts_toward_footprint(self):
        scenario = doc(
            devices=[{"id": "d0", "memory_capacity_bytes": 50, "measured": True}],
            links=[],
            tensors=[{"id": "t", "size_bytes": 40, "producer": "g"}],
            jobs=[job("g", outputs=["t"]),
                  job("k", inputs=["t"], workspace=20)])
        with self.assertRaises(SimulationError):
            validate_scenario(scenario)


class CliTests(unittest.TestCase):
    def test_example_main(self):
        import contextlib
        import io
        from simulator import main
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(EXAMPLE), "--policy", "round_robin"]), 0)
            self.assertEqual(main([str(EXAMPLE), "--policy", "all", "--events"]), 0)


class SchedulingRegressionTests(unittest.TestCase):
    def independent(self):
        return {'schema_version':1,'devices':[{'id':'A','memory_capacity_bytes':100},{'id':'B','memory_capacity_bytes':100}],
                'jobs':[{'id':str(i),'arrival_ns':0,'durations_ns':{'A':10,'B':10}} for i in range(3)]}
    def test_running_job_count_and_order_invariant_reuse(self):
        s=validate_scenario(self.independent())
        for policy in ('shortest_queue','earliest_finish'):
            r=simulate(s,policy)
            self.assertEqual([j['device'] for j in r['jobs']],['A','B','A'])
            self.assertEqual(r,simulate(s,policy))
            self.assertEqual(r['makespan_ns'],20)
    def test_arrival_and_additive_response_identity(self):
        d=self.independent();d['jobs'][0]['arrival_ns']=50
        r=simulate(validate_scenario(d),'round_robin')
        self.assertGreaterEqual(r['jobs'][0]['exec_start_ns'],50)
        for j in r['jobs']:
            self.assertEqual(j['end_ns']-j['arrival_ns'],sum(j[k] for k in
                ('dependency_wait_ns','dispatch_wait_ns','queue_wait_ns','transfer_wait_ns','compute_ns')))
    def test_host_staging_route_timeline_and_budget(self):
        d={'schema_version':1,'meta':{'host_copy_bandwidth_bytes_per_s':10**9,'host_staging_capacity_bytes':10},
           'devices':[{'id':i,'memory_capacity_bytes':100} for i in ('A','B')],
           'tensors':[{'id':'x','size_bytes':10,'producer':'first'}],
           'links':[{'src':'A','dst':'host','bandwidth_bytes_per_s':10**9,'latency_ns':2},
                    {'src':'host','dst':'B','bandwidth_bytes_per_s':10**9,'latency_ns':3}],
           'jobs':[{'id':'first','outputs':['x'],'durations_ns':{'A':5}},
                   {'id':'last','inputs':['x'],'durations_ns':{'B':7}}]}
        r=simulate(validate_scenario(d),'round_robin',strict=True)
        self.assertEqual(r['makespan_ns'],47) # 5 + 12 download + 10 copy + 13 upload + 7
        self.assertEqual(r['host_staging']['peak_reserved_bytes'],10)
        self.assertEqual(r['host_staging']['final_reserved_bytes'],0)
        for dev in r['devices'].values():self.assertLessEqual(dev['peak_used_bytes'],dev['memory_capacity_bytes'])
        starts=[e['time_ns'] for e in r['events'] if e['type']=='transfer_begin']
        self.assertEqual(starts,[5,17,27])
        d['meta']['host_staging_capacity_bytes']=9
        self.assertEqual(simulate(validate_scenario(d),'round_robin')['status'],'failed')
    def test_duplicate_input_and_implicit_data_cycle_reject(self):
        d=self.independent();d['tensors']=[{'id':'x','size_bytes':1,'source':'host'}]
        d['jobs'][0]['inputs']=['x','x']
        with self.assertRaises(SimulationError):validate_scenario(d)

class ResidencyReuseRegressionTests(unittest.TestCase):
    def test_future_consumers_reuse_host_uploaded_weights(self):
        d={'schema_version':1,'devices':[{'id':'A','memory_capacity_bytes':100}],
           'tensors':[{'id':'w','size_bytes':10,'source':'host'}],
           'links':[{'src':'host','dst':'A','bandwidth_bytes_per_s':10**9,'latency_ns':0}],
           'jobs':[{'id':'one','inputs':['w'],'durations_ns':{'A':5}},
                   {'id':'two','inputs':['w'],'depends_on':['one'],'durations_ns':{'A':5}}]}
        r=simulate(validate_scenario(d),'round_robin',strict=True)
        self.assertEqual(r['makespan_ns'],20)
        self.assertEqual(r['links']['host->A']['transfers'],1)
        self.assertEqual(r['devices']['A']['final_used_bytes'],0)

class TransientMemoryRegressionTests(unittest.TestCase):
    def test_source_retained_during_transfer_and_admission_retries(self):
        d={'schema_version':1,'devices':[{'id':'A','memory_capacity_bytes':15},{'id':'B','memory_capacity_bytes':20}],
           'tensors':[{'id':'x','size_bytes':10,'producer':'producer'}],
           'links':[{'src':'A','dst':'B','bandwidth_bytes_per_s':10**9,'latency_ns':0}],
           'jobs':[{'id':'producer','outputs':['x'],'durations_ns':{'A':5}},
                   {'id':'later','depends_on':['producer'],'workspace_bytes':10,'durations_ns':{'A':5}},
                   {'id':'consumer','inputs':['x'],'durations_ns':{'B':5}}]}
        r=simulate(validate_scenario(d),'round_robin',strict=True)
        self.assertEqual(r['makespan_ns'],25)
        self.assertEqual(r['jobs'][1]['dispatch_wait_ns'],15)
        release=[e['time_ns'] for e in r['events'] if e['type']=='release' and e['device']=='A']
        self.assertEqual(release,[20])


if __name__ == "__main__":
    unittest.main()
