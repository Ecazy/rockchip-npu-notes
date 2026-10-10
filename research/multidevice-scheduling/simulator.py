#!/usr/bin/env python3
"""Deterministic multi-device discrete-event scheduler simulator.

Design contract
---------------
* Integer nanoseconds everywhere.  No floating point in the event loop.
* Events are processed in the stable order ``(time_ns, priority, sequence)``.
  The sequence counter is a global insertion counter, so equal-time events keep
  insertion order for a given priority.  ``compute_done`` (priority 0) is
  processed before ``transfer_done`` (priority 1); scheduling is always run as an
  explicit step after an event, never inside the heap comparison.
* Every device owns an independent FIFO job queue.  A device is busy from the
  moment it starts the head job (its input transfers included) until that job's
  service finishes.  Compute and transfer therefore never overlap on a device
  and are never double counted.
* Directed links have ``bandwidth_bytes_per_s`` and ``latency_ns``.  A link is a
  single serial resource (FIFO contention); a transfer occupies
  ``latency + ceil(bytes*1e9/bandwidth)`` and starts at ``max(now,
  link.next_free_ns)``.
* Memory is accounted per device in bytes.  Inputs, outputs and temporary
  workspace all occupy capacity.  A produced tensor has one non-evictable
  "home" device; copies pulled onto another device are replicas that may be
  evicted (and re-fetched later).  Data that still has a future consumer on its
  home device is never deleted.  Exhausting capacity without an evictable
  candidate is reported as a deadlock/failure, never silently ignored.
* ``task_count`` is metadata only: a submit / task-chain is indivisible.  There
  are no register-command (regcmd) assumptions.

Policies (four explicit baselines, all constrained by device eligibility,
per-device measured duration, memory, link existence and DAG dependencies):

``round_robin``
    Ready jobs are ordered by ``ready_seq``; a global device cursor is advanced.
    For each job the first statically feasible device at or after the cursor is
    chosen; the cursor moves past it.  Tie: lowest device index.

``shortest_queue``
    Minimise ``(len(device.fifo) + busy, device_index)`` for the first feasible ready Job.

``earliest_finish``
    Minimise ``(estimated_finish_ns, device_index, job.ready_seq)`` where
    ``estimated_finish = estimated_device_free + estimated_job_duration`` and
    ``estimated_device_free = max(now, device.free_ns) + sum(estimated_job_
    duration of queued jobs)``.  The job duration estimate includes the
    measured service plus the transfer estimate of non-resident inputs.

``locality_aware``
    Minimise ``(non_resident_input_bytes, estimated_finish_ns, device_index,
    job.ready_seq)``: keep data where it already is, then prefer earliest
    finish.

An unknown measured duration for a device makes that device ineligible; a job
with no measured duration on any eligible device is rejected at validation
(never treated as zero).  A dependency cycle or an invalid link is rejected.
"""

from __future__ import annotations

import argparse
import copy
import heapq
import json
import pathlib
import sys
from dataclasses import dataclass, field

POLICIES = ("round_robin", "shortest_queue", "earliest_finish", "locality_aware")
HOST = "host"
SCHEMA_VERSION = 1

COMPUTE_DONE = 0
TRANSFER_DONE = 1


class SimulationError(Exception):
    """Scenario / scheduling error that fails the run."""


class DeadlockError(SimulationError):
    """No legal progress can be made (memory, links or dependencies)."""


def require(condition, message):
    if not condition:
        raise SimulationError(message)


def _int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def transfer_ns(size_bytes, bandwidth_bytes_per_s, latency_ns):
    """Integer transfer time: latency plus a ceiling of bytes/bandwidth."""
    require(_int(size_bytes) and size_bytes >= 0, "invalid transfer size")
    require(_int(bandwidth_bytes_per_s) and bandwidth_bytes_per_s > 0,
            "invalid bandwidth")
    require(_int(latency_ns) and latency_ns >= 0, "invalid latency")
    payload = (size_bytes * 1_000_000_000 + bandwidth_bytes_per_s - 1) // bandwidth_bytes_per_s
    return latency_ns + payload


# ---------------------------------------------------------------------------
# Scenario model
# ---------------------------------------------------------------------------

@dataclass
class Tensor:
    id: str
    size_bytes: int
    source: str | None = None       # "host" for externally supplied tensors
    producer: str | None = None     # job id for produced tensors


@dataclass
class Job:
    id: str
    arrival_ns: int
    depends_on: tuple
    inputs: tuple
    outputs: tuple
    workspace_bytes: int
    durations: dict
    eligible: tuple
    task_count: int

    # runtime state
    state: str = "waiting"          # waiting | ready | queued | transferring | running | done
    arrived: bool = False
    ready_seq: int = -1
    ready_ns: int = 0
    assigned_ns: int = 0
    exec_start_ns: int = 0
    compute_start_ns: int = 0
    end_ns: int = 0
    device: str | None = None
    transfer_bytes: int = 0
    transfer_time_ns: int = 0
    pending_transfers: int = 0
    last_transfer_end_ns: int = 0


@dataclass
class Device:
    id: str
    memory_capacity_bytes: int
    measured: bool
    note: str = ""
    free_ns: int = 0
    fifo: list = field(default_factory=list)
    busy: bool = False
    locked: set = field(default_factory=set)
    used_bytes: int = 0
    peak_bytes: int = 0
    busy_ns: int = 0
    compute_ns: int = 0
    transfer_ns: int = 0
    jobs_done: list = field(default_factory=list)


@dataclass
class Link:
    src: str
    dst: str
    bandwidth_bytes_per_s: int
    latency_ns: int
    next_free_ns: int = 0
    transfers: int = 0
    bytes: int = 0
    busy_ns: int = 0

    @property
    def key(self):
        return (self.src, self.dst)


@dataclass
class Scenario:
    devices: dict
    links: dict
    tensors: dict
    jobs: dict
    job_order: list
    meta: dict


def load_scenario(path):
    document = json.loads(pathlib.Path(path).read_text())
    return validate_scenario(document)


def validate_scenario(document):
    require(isinstance(document, dict), "scenario must be an object")
    require(document.get("schema_version") == SCHEMA_VERSION,
            "unsupported scenario schema_version")

    devices = {}
    for raw in document.get("devices", []):
        require(isinstance(raw, dict), "device entry must be an object")
        did = raw.get("id")
        require(isinstance(did, str) and did and did != HOST and did not in devices,
                f"invalid/duplicate device id {did!r}")
        capacity = raw.get("memory_capacity_bytes")
        require(_int(capacity) and capacity > 0, f"device {did} capacity")
        require(isinstance(raw.get("measured", False), bool), f"device {did} measured flag")
        devices[did] = Device(id=did, memory_capacity_bytes=capacity,
                              measured=raw.get("measured", False),
                              note=raw.get("note", ""))
    require(devices, "scenario needs at least one device")

    links = {}
    for raw in document.get("links", []):
        require(isinstance(raw, dict), "link entry must be an object")
        src, dst = raw.get("src"), raw.get("dst")
        require(isinstance(src, str) and isinstance(dst, str) and src != dst,
                f"invalid link endpoints {src!r}->{dst!r}")
        for endpoint in (src, dst):
            require(endpoint == HOST or endpoint in devices,
                    f"link references unknown endpoint {endpoint!r}")
        bandwidth, latency = raw.get("bandwidth_bytes_per_s"), raw.get("latency_ns")
        require(_int(bandwidth) and bandwidth > 0, f"link {src}->{dst} bandwidth")
        require(_int(latency) and latency >= 0, f"link {src}->{dst} latency")
        require((src, dst) not in links, f"duplicate link {src}->{dst}")
        links[(src, dst)] = Link(src, dst, bandwidth, latency)

    jobs_raw = document.get("jobs")
    require(isinstance(jobs_raw, list) and jobs_raw, "scenario needs jobs")
    producer_of = {}
    job_ids = []
    for raw in jobs_raw:
        require(isinstance(raw, dict), "job entry must be an object")
        jid = raw.get("id")
        require(isinstance(jid, str) and jid and jid not in producer_of,
                f"invalid/duplicate job id {jid!r}")
        job_ids.append(jid)
        producer_of[jid] = raw

    tensors = {}
    for raw in document.get("tensors", []):
        require(isinstance(raw, dict), "tensor entry must be an object")
        tid = raw.get("id")
        require(isinstance(tid, str) and tid and tid not in tensors,
                f"invalid/duplicate tensor id {tid!r}")
        size = raw.get("size_bytes")
        require(_int(size) and size > 0, f"tensor {tid} size")
        source, producer = raw.get("source"), raw.get("producer")
        require((source == HOST) != (producer is not None),
                f"tensor {tid} must have exactly one of source=host/producer")
        if producer is not None:
            require(producer in producer_of, f"tensor {tid} unknown producer")
        tensors[tid] = Tensor(id=tid, size_bytes=size, source=source, producer=producer)

    jobs = {}
    for raw in jobs_raw:
        jid = raw["id"]
        arrival = raw.get("arrival_ns", 0)
        require(_int(arrival) and arrival >= 0, f"job {jid} arrival")
        deps = raw.get("depends_on", [])
        require(isinstance(deps, list) and all(isinstance(d, str) for d in deps),
                f"job {jid} depends_on")
        require(jid not in deps, f"job {jid} depends on itself")
        for d in deps:
            require(d in producer_of, f"job {jid} unknown dependency {d!r}")
        inputs = raw.get("inputs", [])
        outputs = raw.get("outputs", [])
        require(isinstance(inputs, list) and isinstance(outputs, list),
                f"job {jid} inputs/outputs")
        require(len(set(inputs)) == len(inputs) and len(set(outputs)) == len(outputs),
                f"job {jid} duplicate tensor references")
        require(not set(inputs).intersection(outputs), f"job {jid} in-place unsupported")
        for t in inputs:
            require(t in tensors, f"job {jid} unknown input {t!r}")
            producer = tensors[t].producer
            if producer is not None and producer not in deps:
                deps = deps + [producer]
        for t in outputs:
            require(t in tensors and tensors[t].producer == jid,
                    f"job {jid} output {t!r} not declared with this producer")
        workspace = raw.get("workspace_bytes", 0)
        require(_int(workspace) and workspace >= 0, f"job {jid} workspace")
        durations = raw.get("durations_ns", {})
        require(isinstance(durations, dict) and durations, f"job {jid} durations")
        for did, duration in durations.items():
            require(did in devices, f"job {jid} duration for unknown device {did!r}")
            require(_int(duration) and duration > 0, f"job {jid} duration {did}")
        eligible = raw.get("eligible_devices", sorted(durations))
        require(isinstance(eligible, list) and eligible, f"job {jid} eligible_devices")
        for did in eligible:
            require(did in devices, f"job {jid} eligible unknown device {did!r}")
        with_duration = [d for d in eligible if d in durations]
        require(with_duration, f"job {jid} has no measured duration on any eligible device")
        eligible = tuple(with_duration)
        task_count = raw.get("task_count", 1)
        require(_int(task_count) and task_count >= 1, f"job {jid} task_count")
        for d in deps:
            require(producer_of[d], f"job {jid} dependency")
        jobs[jid] = Job(id=jid, arrival_ns=arrival, depends_on=tuple(deps),
                        inputs=tuple(inputs), outputs=tuple(outputs),
                        workspace_bytes=workspace, durations=dict(durations),
                        eligible=eligible, task_count=task_count)

    for tensor in tensors.values():
        if tensor.producer is not None:
            require(tensor.id in jobs[tensor.producer].outputs,
                    f"tensor {tensor.id} missing from declared producer outputs")
    _reject_cycles(jobs)
    _validate_footprint(jobs, tensors, devices)
    _validate_external_links(jobs, tensors, links)

    meta = dict(document.get("meta", {}))
    return Scenario(devices=devices, links=links, tensors=tensors, jobs=jobs,
                    job_order=job_ids, meta=meta)


def _reject_cycles(jobs):
    indegree = {jid: 0 for jid in jobs}
    children = {jid: [] for jid in jobs}
    for jid, job in jobs.items():
        for dep in job.depends_on:
            indegree[jid] += 1
            children[dep].append(jid)
    queue = [jid for jid in jobs if indegree[jid] == 0]
    seen = 0
    while queue:
        node = queue.pop()
        seen += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    require(seen == len(jobs), "dependency cycle in job DAG")


def _job_min_bytes(job, tensors):
    return (sum(tensors[t].size_bytes for t in job.inputs)
            + sum(tensors[t].size_bytes for t in job.outputs)
            + job.workspace_bytes)


def _validate_footprint(jobs, tensors, devices):
    for jid, job in jobs.items():
        need = _job_min_bytes(job, tensors)
        if not any(need <= devices[d].memory_capacity_bytes for d in job.eligible):
            raise SimulationError(
                f"job {jid} needs {need} bytes, exceeds every eligible device capacity")


def _validate_external_links(jobs, tensors, links):
    for jid, job in jobs.items():
        for tid in job.inputs:
            tensor = tensors[tid]
            if tensor.source == HOST:
                if not any((HOST, d) in links for d in job.eligible):
                    raise SimulationError(
                        f"job {jid} input {tid} from host but no host link eligible")


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    def __init__(self, scenario, policy):
        require(policy in POLICIES, f"unknown policy {policy!r}")
        # Deep-copy so a Simulator never mutates the caller's Scenario and the
        # same Scenario can be replayed by several policies deterministically.
        self.scenario = copy.deepcopy(scenario)
        scenario = self.scenario
        self.policy = policy
        self.now = 0
        self.seq = 0
        self.heap = []
        self.rr_index = 0
        self.ready_counter = 0
        self.events = []
        self.failure = None
        self.completed = 0

        self.devices = scenario.devices
        self.links = scenario.links
        self.tensors = scenario.tensors
        self.jobs = scenario.jobs
        self.home = {}
        self.resident = {tid: set() for tid in self.tensors}
        self.consumers_by_device = {tid: {} for tid in self.tensors}
        self.remaining_total = {tid: 0 for tid in self.tensors}
        self.host_used = 0
        self.host_peak = 0
        self.host_copy_free_ns = 0
        self.staged = {}
        self.last_access = {tid: 0 for tid in self.tensors}
        self.access_counter = 0
        for job in self.jobs.values():
            for tid in job.inputs:
                self.remaining_total[tid] += 1

    # -- event queue --------------------------------------------------------

    def _push(self, time_ns, priority, kind, payload):
        entry = (time_ns, priority, self.seq, kind, payload)
        self.seq += 1
        heapq.heappush(self.heap, entry)

    def _log(self, kind, **fields):
        event = {"time_ns": self.now, "type": kind}
        event.update(fields)
        self.events.append(event)

    # -- public API ---------------------------------------------------------

    def run(self, strict=False):
        try:
            self._run()
        except DeadlockError as error:
            self.failure = str(error)
        if strict and self.failure:
            raise DeadlockError(self.failure)
        return self.result()

    def _run(self):
        for jid in self.scenario.job_order:
            job = self.jobs[jid]
            if job.arrival_ns == 0:
                job.arrived = True
            else:
                self._push(job.arrival_ns, TRANSFER_DONE, "arrival", job)
        for jid in self.scenario.job_order:
            job = self.jobs[jid]
            if job.arrived and not job.depends_on:
                self._make_ready(job, job.arrival_ns)
        self._schedule()
        while self.heap:
            time_ns, _priority, _seq, kind, payload = heapq.heappop(self.heap)
            self.now = time_ns
            if kind == "compute_done":
                self._compute_done(payload)
            elif kind == "transfer_done":
                self._transfer_done(*payload)
            elif kind == "transfer_begin":
                self._log("transfer_begin", **payload)
            elif kind == "arrival":
                payload.arrived = True
                if all(self.jobs[d].state == "done" for d in payload.depends_on):
                    self._make_ready(payload, self.now)
                self._schedule()
            else:  # pragma: no cover - defensive
                raise SimulationError(f"unknown event {kind!r}")
            for device in self.devices.values():
                self._pump(device)
            self._schedule()
        if self.completed < len(self.jobs):
            pending = sorted(j.id for j in self.jobs.values() if j.state != "done")
            raise DeadlockError(f"deadlock: unfinished jobs {pending}")

    # -- scheduling ---------------------------------------------------------

    def _make_ready(self, job, when):
        job.state = "ready"
        job.ready_ns = when
        job.ready_seq = self.ready_counter
        self.ready_counter += 1
        self._log("ready", job=job.id, ready_ns=when)

    def _schedule(self):
        """Greedily assign ready jobs to device FIFO queues."""
        while True:
            choice = self._select_assignment()
            if choice is None:
                return
            job, device, index = choice
            job.state = "queued"
            job.assigned_ns = self.now
            job.device = device.id
            self.rr_index = (index + 1) % len(self.devices)
            device.fifo.append(job)
            self._log("enqueue", job=job.id, device=device.id,
                      queue_depth=len(device.fifo))
            self._pump(device)

    def _ready_jobs(self):
        return [j for j in self.jobs.values() if j.state == "ready"]

    def _device_estimate_ns(self, device):
        estimate = max(self.now, device.free_ns)
        for queued in device.fifo:
            estimate += self._estimate_job_ns(queued, device, estimate)
        return estimate

    def _route(self, source, target):
        if (source, target) in self.links:
            return [self.links[(source, target)]]
        if source != HOST and (source, HOST) in self.links and (HOST, target) in self.links:
            bandwidth = self.scenario.meta.get('host_copy_bandwidth_bytes_per_s')
            require(_int(bandwidth) and bandwidth > 0,
                    'Host staging requires configured host_copy_bandwidth_bytes_per_s')
            return [self.links[(source, HOST)], self.links[(HOST, target)]]
        return []

    def _estimate_job_ns(self, job, device, available_ns=None):
        base = self.now if available_ns is None else available_ns
        finish = base
        links_available = {}
        for tid in job.inputs:
            if device.id in self.resident[tid]:
                continue
            source = HOST if self.tensors[tid].source == HOST else self.home.get(tid)
            route = self._route(source, device.id)
            end = base
            for index, link in enumerate(route):
                if index == 1:
                    end = max(end, self.host_copy_free_ns) + transfer_ns(
                        self.tensors[tid].size_bytes,
                        self.scenario.meta['host_copy_bandwidth_bytes_per_s'], 0)
                begin = max(end, link.next_free_ns, links_available.get(link.key, 0))
                end = begin + transfer_ns(self.tensors[tid].size_bytes,
                                         link.bandwidth_bytes_per_s, link.latency_ns)
                links_available[link.key] = end
            finish = max(finish, end)
        return finish - base + job.durations[device.id]

    def _resident_input_bytes(self, job, device):
        return sum(self.tensors[t].size_bytes for t in job.inputs
                   if device.id not in self.resident[t])

    def _static_ok(self, job, device):
        if device.id not in job.eligible or device.id not in job.durations:
            return False
        # Worst-case footprint (all inputs as replicas + outputs + workspace)
        # must fit; anything above that can never run on this device.  Replica
        # eviction is handled at admission time.
        if _job_min_bytes(job, self.tensors) > device.memory_capacity_bytes:
            return False
        for tid in job.inputs:
            if device.id in self.resident[tid]:
                continue
            tensor = self.tensors[tid]
            source = HOST if tensor.source == HOST else self.home.get(tid)
            if source is None:
                return False
            if source != device.id and not self._route(source, device.id):
                return False
        if not device.busy and not device.fifo and not self._admission_possible(job, device):
            return False
        return True

    def _admission_possible(self, job, device):
        missing = [t for t in job.inputs if device.id not in self.resident[t]]
        required = (sum(self.tensors[t].size_bytes for t in missing)
                    + sum(self.tensors[t].size_bytes for t in job.outputs) + job.workspace_bytes)
        reclaimable = sum(self.tensors[t].size_bytes for t, holders in self.resident.items()
                          if device.id in holders and t not in device.locked
                          and t not in job.inputs and t not in job.outputs
                          and not (self.home.get(t) == device.id and self.remaining_total[t] > 0))
        staging = sum(self.tensors[t].size_bytes for t in missing
                      if len(self._route(HOST if self.tensors[t].source == HOST else self.home.get(t), device.id)) == 2)
        capacity = self.scenario.meta.get('host_staging_capacity_bytes')
        if capacity is not None:
            require(_int(capacity) and capacity > 0, 'invalid Host staging capacity')
            if self.host_used + staging > capacity:
                return False
        return device.used_bytes + required - reclaimable <= device.memory_capacity_bytes

    def _select_assignment(self):
        ready = self._ready_jobs()
        if not ready:
            return None
        if self.policy == "round_robin":
            return self._select_round_robin(ready)
        best = None
        best_key = None
        for job in sorted(ready, key=lambda j: j.ready_seq):
            for index, device in enumerate(self.devices.values()):
                if not self._static_ok(job, device):
                    continue
                key = self._policy_key(job, device, index)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (job, device, index)
            if best is not None:
                return best
        return best

    def _select_round_robin(self, ready):
        devices = list(self.devices.values())
        ordered = sorted(ready, key=lambda j: j.ready_seq)
        for job in ordered:
            for offset in range(len(devices)):
                index = (self.rr_index + offset) % len(devices)
                device = devices[index]
                if self._static_ok(job, device):
                    return (job, device, index)
        return None

    def _policy_key(self, job, device, index):
        if self.policy == "shortest_queue":
            return (len(device.fifo) + int(device.busy), index, job.ready_seq)
        if self.policy == "earliest_finish":
            available = self._device_estimate_ns(device)
            estimate = available + self._estimate_job_ns(job, device, available)
            return (estimate, index, job.ready_seq)
        if self.policy == "locality_aware":
            transfer_bytes = self._resident_input_bytes(job, device)
            available = self._device_estimate_ns(device)
            finish = available + self._estimate_job_ns(job, device, available)
            return (transfer_bytes, finish, index, job.ready_seq)
        raise SimulationError(f"no key for policy {self.policy!r}")

    # -- device execution ---------------------------------------------------

    def _pump(self, device):
        if device.busy or not device.fifo:
            return
        job = device.fifo[0]
        if not self._admission_possible(job, device) or not self._admit(job, device):
            return
        device.fifo.pop(0)
        device.busy = True
        job.state = "transferring"
        job.exec_start_ns = self.now
        device.free_ns = self.now
        self._log("exec_start", job=job.id, device=device.id)
        missing = [tid for tid in job.inputs if device.id not in self.resident[tid]]
        if not missing:
            job.state = "running"
            job.compute_start_ns = self.now
            duration = job.durations[device.id]
            device.free_ns = self.now + duration
            self._log("compute_begin", job=job.id, device=device.id)
            self._push(self.now + duration, COMPUTE_DONE, "compute_done", job)
            return
        job.pending_transfers = len(missing)
        for tid in missing:
            tensor = self.tensors[tid]
            source = HOST if tensor.source == HOST else self.home[tid]
            route = self._route(source, device.id)
            size = tensor.size_bytes
            end = self.now
            if len(route) == 2:
                self.host_used += size
                self.host_peak = max(self.host_peak, self.host_used)
                self.staged[(job.id, tid)] = size
            for index, link in enumerate(route):
                if index == 1:
                    copy_begin = max(end, self.host_copy_free_ns)
                    copy_duration = transfer_ns(size, self.scenario.meta['host_copy_bandwidth_bytes_per_s'], 0)
                    end = copy_begin + copy_duration
                    self.host_copy_free_ns = end
                    job.transfer_time_ns += copy_duration
                    self._push(copy_begin, 2, 'transfer_begin', dict(job=job.id, tensor=tid,
                               src=HOST, dst=HOST, bytes=size, scope='host_copy', end_ns=end))
                start = max(end, link.next_free_ns)
                duration = transfer_ns(size, link.bandwidth_bytes_per_s, link.latency_ns)
                end = start + duration
                link.next_free_ns = end
                link.transfers += 1
                link.bytes += size
                link.busy_ns += duration
                device.transfer_ns += duration
                job.transfer_bytes += size
                job.transfer_time_ns += duration
                self._log('transfer_reserve', job=job.id, tensor=tid, src=link.src,
                          dst=link.dst, bytes=size, start_ns=start, end_ns=end)
                self._push(start, 2, 'transfer_begin', dict(job=job.id, tensor=tid,
                           src=link.src, dst=link.dst, bytes=size, end_ns=end))
            device.free_ns = max(device.free_ns, end + job.durations[device.id])
            self._push(end, TRANSFER_DONE, 'transfer_done', (job, device, tid, end))

    def _admit(self, job, device):
        """Reserve inputs+outputs+workspace, evicting safe replicas if needed."""
        missing = [t for t in job.inputs if device.id not in self.resident[t]]
        required = sum(self.tensors[t].size_bytes for t in missing)
        required += sum(self.tensors[t].size_bytes for t in job.outputs)
        required += job.workspace_bytes
        while device.used_bytes + required > device.memory_capacity_bytes:
            candidate = self._evict_candidate(job, device, missing)
            if candidate is None:
                return False
            self._release(candidate, device, reason="evict")
        device.used_bytes += required
        device.peak_bytes = max(device.peak_bytes, device.used_bytes)
        for tid in missing:
            self.access_counter += 1
            self.last_access[tid] = self.access_counter
        device.locked = set(job.inputs) | set(job.outputs)
        return True

    def _evict_candidate(self, job, device, missing):
        options = []
        for tid, holders in self.resident.items():
            if device.id not in holders:
                continue
            if tid in device.locked:
                continue
            if tid in job.inputs or tid in job.outputs:
                continue
            if tid in missing:
                continue
            if self.home.get(tid) == device.id and self.remaining_total[tid] > 0:
                continue  # home copy still referenced by future consumers
            options.append(tid)
        if not options:
            return None
        # Least-recently-used, stable by tensor id.
        return min(options, key=lambda tid: (self.last_access[tid], tid))

    def _transfer_done(self, job, device, tid, when):
        self.host_used -= self.staged.pop((job.id, tid), 0)
        self.resident[tid].add(device.id)
        self.access_counter += 1
        self.last_access[tid] = self.access_counter
        self._log("transfer_done", job=job.id, tensor=tid, dst=device.id)
        job.pending_transfers -= 1
        if job.pending_transfers == 0:
            job.state = "running"
            job.compute_start_ns = when
            duration = job.durations[device.id]
            device.free_ns = when + duration
            self._log("compute_begin", job=job.id, device=device.id)
            self._push(when + duration, COMPUTE_DONE, "compute_done", job)

    def _compute_done(self, job):
        device = self.devices[job.device]
        job.end_ns = self.now
        job.state = "done"
        device.busy = False
        device.free_ns = self.now
        device.busy_ns += self.now - job.exec_start_ns
        device.compute_ns += job.durations[device.id]
        device.jobs_done.append(job.id)
        self.completed += 1
        # Release inputs whose consumers on this device are exhausted.
        for tid in job.inputs:
            self.remaining_total[tid] -= 1
            if self.remaining_total[tid] == 0:
                for holder in list(self.resident[tid]):
                    self._release(tid, self.devices[holder], reason='release')
            # Otherwise keep copies resident for future consumers. Remote replicas
            # remain safely evictable; the producer's last live home is pinned.
        # Outputs become home residents of this device.
        for tid in job.outputs:
            self.home[tid] = device.id
            self.resident[tid].add(device.id)
            self.access_counter += 1
            self.last_access[tid] = self.access_counter
        device.used_bytes -= job.workspace_bytes
        device.locked = set()
        self._log("compute_done", job=job.id, device=device.id, end_ns=self.now)
        for dependent in self.scenario.job_order:
            child = self.jobs[dependent]
            if child.state == "waiting" and child.arrived and all(
                    self.jobs[dep].state == "done" for dep in child.depends_on):
                self._make_ready(child, self.now)
        self._pump(device)
        self._schedule()

    def _release(self, tid, device, reason):
        if device.id in self.resident[tid]:
            self.resident[tid].discard(device.id)
            device.used_bytes -= self.tensors[tid].size_bytes
            self._log(reason, tensor=tid, device=device.id)

    # -- results ------------------------------------------------------------

    def result(self):
        jobs = []
        for jid in self.scenario.job_order:
            job = self.jobs[jid]
            if job.state != "done":
                jobs.append({"id": jid, "state": job.state, "device": job.device})
                continue
            jobs.append({
                "id": jid,
                "device": job.device,
                "task_count": job.task_count,
                "arrival_ns": job.arrival_ns,
                "ready_ns": job.ready_ns,
                "assigned_ns": job.assigned_ns,
                "exec_start_ns": job.exec_start_ns,
                "compute_start_ns": job.compute_start_ns,
                "end_ns": job.end_ns,
                "dependency_wait_ns": job.ready_ns - job.arrival_ns,
                "dispatch_wait_ns": job.assigned_ns - job.ready_ns,
                "queue_wait_ns": job.exec_start_ns - job.assigned_ns,
                "transfer_wait_ns": job.compute_start_ns - job.exec_start_ns,
                "compute_ns": job.end_ns - job.compute_start_ns,
                "transfer_bytes": job.transfer_bytes,
                "transfer_time_ns": job.transfer_time_ns,
                "workspace_bytes": job.workspace_bytes,
                "inputs": list(job.inputs),
                "outputs": list(job.outputs),
            })
        devices = {}
        for did, device in self.devices.items():
            devices[did] = {
                "memory_capacity_bytes": device.memory_capacity_bytes,
                "measured": device.measured,
                "note": device.note,
                "busy_ns": device.busy_ns,
                "compute_ns": device.compute_ns,
                "transfer_ns": device.transfer_ns,
                "jobs": device.jobs_done,
                "peak_used_bytes": device.peak_bytes,
                "final_used_bytes": device.used_bytes,
            }
        links = {}
        for (src, dst), link in self.links.items():
            links[f"{src}->{dst}"] = {
                "bandwidth_bytes_per_s": link.bandwidth_bytes_per_s,
                "latency_ns": link.latency_ns,
                "transfers": link.transfers,
                "bytes": link.bytes,
                "busy_ns": link.busy_ns,
            }
        done = [j for j in jobs if j.get("state", "done") == "done"]
        makespan = max((j["end_ns"] for j in done), default=0) if done else None
        return {
            "status": "failed" if self.failure else ("ok" if self.completed == len(self.jobs) else "failed"),
            "failure": self.failure,
            "policy": self.policy,
            "makespan_ns": makespan,
            "jobs": jobs,
            "devices": devices,
            "host_staging": {'peak_reserved_bytes':self.host_peak,
                             'final_reserved_bytes':self.host_used,
                             'capacity_bytes':self.scenario.meta.get('host_staging_capacity_bytes'),
                             'scope':'conservative reservation for entire download/copy/upload route'},
            "links": links,
            "events": self.events,
        }


def simulate(scenario, policy, strict=False):
    return Simulator(scenario, policy).run(strict=strict)


def simulate_all(scenario, strict=False):
    return {policy: simulate(scenario, policy, strict=strict) for policy in POLICIES}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Multi-device deterministic scheduler simulator")
    parser.add_argument("scenario", type=pathlib.Path)
    parser.add_argument("--policy", default="all", choices=("all",) + POLICIES)
    parser.add_argument("--json", type=pathlib.Path, default=None)
    parser.add_argument("--events", action="store_true", help="include the event log")
    args = parser.parse_args(argv)
    scenario = load_scenario(args.scenario)
    policies = POLICIES if args.policy == "all" else (args.policy,)
    output = {}
    for policy in policies:
        result = simulate(scenario, policy)
        if not args.events:
            result.pop("events", None)
        output[policy] = result
    text = json.dumps(output, indent=2, sort_keys=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    print(text)
    return 0 if all(r["status"] == "ok" for r in output.values()) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (SimulationError, OSError, ValueError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
