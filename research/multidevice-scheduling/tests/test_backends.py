#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contract tests for the multi-device backend adapters.

These tests are hardware-free.  They exercise:

* the uniform interface (capability tri-state, ownership, token lifetime, bounds),
* the in-memory mock adapter's submit/wait and timeout behaviour,
* the RK3588 adapter against an *injected* bridge double that mirrors the real
  ``libco_backend_bridge`` Python surface, including fail-closed behaviour when the
  bridge is absent,
* the RK182X adapter's unavailable state and its injected runtime contract checks.

When the real bridge has been built (see ``backends/bridge/CMakeLists.txt``) the loader
test additionally dlopen()s it and checks the ABI.  It never claims hardware evidence:
executing on silicon is not possible here, and the test only notes success/failure.
"""

import ctypes.util  # noqa: F401  (kept for parity with the loader's search)
import itertools
import sys
import unittest
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from backends import (  # noqa: E402
    BRIDGE_ENV,  # noqa: E402,F401 (documented env var)
    MAX_TRANSFER_BYTES,
    BackendMismatch,
    BufferOwnershipError,
    Capability,
    CapabilityState,
    DeviceUnavailable,
    Evidence,
    InvalidPayload,
    MemoryBoundsError,
    MemoryInfo,
    OperationTimeout,
    RK182XBackend,
    RK182XPayload,
    RK3588Backend,
    RK3588Payload,
    TokenState,
    UnknownSDKBehavior,
    UnsupportedOperation,
    find_bridge_library,
)
from backends.mock import FakeClock, MockBackend, MockPayload  # noqa: E402
from backends.rk182x import ContractOnlyRKNN3Runtime, RKNN3RuntimeContract  # noqa: E402
from backends.rk3588 import (  # noqa: E402
    BRIDGE_ABI_VERSION,
    BridgeCallError,
    CtypesBridge,
    MODE_CHAIN,
    MODE_SEPARATE,
)


# --------------------------------------------------------------------------- doubles


class FakeBridge:
    """Bridge double with the same Python surface as :class:`CtypesBridge`."""

    CO_SLACK = 32768
    CO_GUARD = 64
    _STATUS = {-1: "E_SHAPE", -2: "E_TILING", -3: "E_NOMEM", -4: "E_DEVICE",
               -5: "E_UNSUPPORTED"}

    def __init__(self, *, device_name="rk3588", abi=BRIDGE_ABI_VERSION,
                 fail_open=False, execution_error=None):
        self._device_name = device_name
        self._abi = abi
        self.fail_open = fail_open
        self.execution_error = execution_error
        self.opened = False
        self.closed = False
        self.executed = []
        self.batches = []
        self.scheduled = []
        self._buffers = {}
        self._jobs = {}
        self._ids = itertools.count(1)

    # loader surface
    def abi_version(self):
        return self._abi

    def device_name(self):
        return self._device_name

    def open_device(self):
        if self.fail_open:
            raise BridgeCallError("open", -2, "ENOENT")
        self.opened = True
        return 7

    def close_device(self, fd):
        self.opened = False
        self.closed = True

    def _buf(self, handle):
        b = self._buffers.get(handle)
        if b is None:
            raise BridgeCallError("buffer", -1, "E_SHAPE")
        return b

    def buffer_alloc(self, fd, payload):
        handle = next(self._ids)
        self._buffers[handle] = {
            "payload": payload,
            "alloc": payload + self.CO_SLACK + self.CO_GUARD,
            "data": bytearray(payload),
            "tensor": None,
        }
        return handle

    def buffer_free(self, handle):
        self._buffers.pop(handle, None)

    def buffer_payload(self, handle):
        return self._buf(handle)["payload"]

    def buffer_allocated(self, handle):
        return self._buf(handle)["alloc"]

    def tensor_bytes(self, handle):
        t = self._buf(handle)["tensor"]
        return 0 if t is None else t.get("bytes", 0)

    def _set_tensor(self, handle, **info):
        self._buf(handle)["tensor"] = info

    def buffer_set_matrix(self, handle, rows, cols, type_):
        self._set_tensor(handle, kind="matrix", rows=rows, cols=cols)

    def buffer_set_input(self, handle, rows, cols):
        self._set_tensor(handle, kind="input", rows=rows, cols=cols)

    def buffer_set_weight(self, handle, outputs, inputs):
        self._set_tensor(handle, kind="weight", outputs=outputs, inputs=inputs)

    def buffer_set_linear(self, handle, elements):
        self._set_tensor(handle, kind="linear", elements=elements)

    def upload(self, handle, offset, data):
        b = self._buf(handle)
        if offset < 0 or offset + len(data) > b["payload"]:
            raise BridgeCallError("upload", -1, "E_SHAPE")
        b["data"][offset:offset + len(data)] = data

    def download(self, handle, offset, size):
        b = self._buf(handle)
        if offset < 0 or offset + size > b["payload"]:
            raise BridgeCallError("download", -1, "E_SHAPE")
        return bytes(b["data"][offset:offset + size])

    def job_create(self, out_handle, out_offset):
        self._buf(out_handle)
        job = next(self._ids)
        self._jobs[job] = {"out": out_handle, "out_offset": out_offset,
                           "reads": [], "words": (), "flags": 0}
        return job

    def job_add_read(self, job, buffer, offset):
        self._buf(buffer)
        self._jobs[job]["reads"].append((buffer, offset))

    def job_set_words(self, job, words, flags):
        self._jobs[job]["words"] = tuple(words)
        self._jobs[job]["flags"] = flags

    def job_validate(self, job):
        j = self._jobs.get(job)
        if j is None or not j["words"]:
            raise BridgeCallError("job_validate", -1, "E_SHAPE")

    def job_free(self, job):
        self._jobs.pop(job, None)

    def execute(self, fd, job, mode):
        if self.execution_error is not None:
            raise BridgeCallError("execute", self.execution_error,
                                  self.status_name(self.execution_error))
        self.executed.append((job, mode))

    def execute_batch(self, fd, jobs, mode):
        self.batches.append((tuple(jobs), mode))

    def schedule_execute(self, fd, job, mask, request_id):
        self.scheduled.append((job, mask, request_id))
        return {"host_begin_ns": 10, "ioctl_begin_ns": 11, "ioctl_end_ns": 20,
                "host_done_ns": 30, "driver_service_ns": 9, "actual_mask": mask or 1,
                "task_count": 1, "int_status": 0x100, "result": 0}

    def status_name(self, code):
        return self._STATUS.get(code, "E_UNKNOWN")


class NoBridge:
    """A factory that always fails, simulating a missing library."""

    def __init__(self, message="libco_backend_bridge not found"):
        self._message = message

    def __call__(self):
        raise DeviceUnavailable(self._message)


def _make_rk3588(bridge):
    return RK3588Backend(bridge_factory=lambda: bridge, clock=FakeClock())


def _configured_output(backend, payload_bytes=64):
    buf = backend.alloc_buffer(payload_bytes, "output")
    backend.configure_tensor(buf, "matrix", rows=4, cols=4, type=0)
    return buf


# --------------------------------------------------------------------------- interface


class InterfaceContractTests(unittest.TestCase):
    def test_capability_tristate(self):
        c = FakeClock()
        b = MockBackend(clock=c)
        self.assertIs(b.capability_state(Capability.SUBMIT), CapabilityState.AVAILABLE)
        self.assertIs(b.capability_state(Capability.PEER_TO_PEER_TRANSFER),
                      CapabilityState.UNAVAILABLE)
        b.require(Capability.SUBMIT)
        with self.assertRaises(UnsupportedOperation):
            b.require(Capability.PEER_TO_PEER_TRANSFER)

    def test_memoryinfo_fields_are_explicit(self):
        m = MemoryInfo(backend="x", owned_payload_bytes=4, owned_allocated_bytes=8,
                       live_buffers=1, reserved_bytes=None, device_total_bytes=None,
                       system_total_bytes=None)
        self.assertFalse(m.totals_known)
        self.assertIsNone(m.device_total_bytes)

    def test_payload_backend_is_class_bound(self):
        self.assertEqual(MockPayload().backend, "mock")
        self.assertEqual(RK3588Payload().backend, "rk3588")
        self.assertEqual(RK182XPayload().backend, "rk182x")


# --------------------------------------------------------------------------- mock


class MockContractTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.backend = MockBackend(clock=self.clock, capacity_bytes=1 << 20)
        self.backend.open()

    def test_submit_wait_and_token_identity(self):
        job = self.backend.create_job(MockPayload(kind="k", duration_ns=100))
        t1 = self.backend.submit(job)
        t2 = self.backend.submit(self.backend.create_job(MockPayload(duration_ns=0)))
        self.assertNotEqual(t1.token_id, t2.token_id)
        self.assertIs(t1.state, TokenState.PENDING)
        self.assertIs(t2.state, TokenState.COMPLETE)
        self.clock.advance(100)
        self.assertIs(self.backend.wait(t1), TokenState.COMPLETE)
        self.assertIsNotNone(t1.completed_ns)

    def test_token_lifetime_single_use(self):
        job = self.backend.create_job(MockPayload(duration_ns=0))
        token = self.backend.submit(job)
        self.backend.release_token(token)
        self.assertIs(token.state, TokenState.CONSUMED)
        with self.assertRaises(BufferOwnershipError):
            self.backend.wait(token)
        with self.assertRaises(BufferOwnershipError):
            self.backend.release_token(token)

    def test_timeout(self):
        job = self.backend.create_job(MockPayload(duration_ns=1000))
        token = self.backend.submit(job)
        with self.assertRaises(OperationTimeout):
            self.backend.wait(token, timeout_ns=10)
        self.assertIs(token.state, TokenState.PENDING)
        self.assertIsNone(token.error)
        self.assertEqual(self.backend.wait(token), TokenState.COMPLETE)

    def test_mock_failure_is_reported(self):
        job = self.backend.create_job(MockPayload(fail=True))
        with self.assertRaises(Exception):
            self.backend.submit(job)

    def test_transfer_roundtrip_and_bounds(self):
        buf = self.backend.alloc_buffer(8, "x")
        self.assertEqual(self.backend.upload(buf, 2, b"abcd"), 4)
        self.assertEqual(self.backend.download(buf, 0, 8), b"\x00\x00abcd\x00\x00")
        with self.assertRaises(MemoryBoundsError):
            self.backend.upload(buf, 6, b"abcd")
        with self.assertRaises(MemoryBoundsError):
            self.backend.download(buf, 0, 9)
        with self.assertRaises(MemoryBoundsError):
            self.backend.upload(buf, -1, b"a")

    def test_transfer_cap(self):
        # The per-call cap is independent of buffer capacity, so use a mock large enough
        # to hold the oversized buffer and prove upload (not alloc) rejects it.
        big = MockBackend(clock=FakeClock(),
                          capacity_bytes=MAX_TRANSFER_BYTES + (1 << 20))
        big.open()
        buf = big.alloc_buffer(MAX_TRANSFER_BYTES + 16, "big")
        too_big = b"\x00" * (MAX_TRANSFER_BYTES + 1)
        with self.assertRaises(MemoryBoundsError):
            big.upload(buf, 0, too_big)

    def test_buffer_ownership(self):
        buf = self.backend.alloc_buffer(8)
        self.backend.free_buffer(buf)
        self.assertTrue(buf.released)
        with self.assertRaises(BufferOwnershipError):
            self.backend.free_buffer(buf)
        with self.assertRaises(BufferOwnershipError):
            self.backend.upload(buf, 0, b"a")

    def test_allocation_capacity(self):
        with self.assertRaises(MemoryBoundsError):
            self.backend.alloc_buffer((1 << 20) + 1)

    def test_query_memory_owned_vs_unknown(self):
        self.backend.alloc_buffer(10, "a")
        self.backend.alloc_buffer(20, "b")
        info = self.backend.query_memory()
        self.assertEqual(info.owned_payload_bytes, 30)
        self.assertEqual(info.live_buffers, 2)
        self.assertEqual(info.device_total_bytes, 1 << 20)  # mock capacity is configured
        self.assertIsNone(info.system_total_bytes)          # never fabricated

    def test_wrong_backend_payload_and_handle(self):
        with self.assertRaises(BackendMismatch):
            self.backend.create_job(RK3588Payload(words=(1,)))
        rk_job = None
        other = _make_rk3588(FakeBridge())
        other.open()
        out = _configured_output(other)
        rk_job = other.create_job(RK3588Payload(words=(1, 2, 3), output=(out, 0)))
        with self.assertRaises(BackendMismatch):
            self.backend.submit(rk_job)
        self.assertNotEqual(rk_job.payload.backend, "mock")

    def test_wrong_backend_buffer(self):
        # A genuinely cross-ABI buffer: one allocated by the RK3588 adapter, handed to the
        # mock adapter.  The opaque backend tag is what makes this fail closed.
        other = _make_rk3588(FakeBridge())
        other.open()
        foreign = other.alloc_buffer(8)
        with self.assertRaises(BackendMismatch):
            self.backend.upload(foreign, 0, b"a")


# --------------------------------------------------------------------------- rk3588


class RK3588AdapterTests(unittest.TestCase):
    def test_fails_closed_without_bridge(self):
        backend = RK3588Backend(bridge_factory=NoBridge())
        for cap in (Capability.SUBMIT, Capability.WAIT, Capability.UPLOAD,
                    Capability.DOWNLOAD, Capability.QUERY_MEMORY):
            self.assertIs(backend.capability_state(cap), CapabilityState.UNAVAILABLE)
        with self.assertRaises(DeviceUnavailable):
            backend.open()
        with self.assertRaises((DeviceUnavailable, UnsupportedOperation)):
            backend.alloc_buffer(8)

    def test_abi_mismatch_fails_closed(self):
        backend = RK3588Backend(bridge_factory=lambda: FakeBridge(abi=99))
        self.assertIs(backend.capability_state(Capability.SUBMIT),
                      CapabilityState.UNAVAILABLE)
        with self.assertRaises(DeviceUnavailable):
            backend.open()

    def test_open_alloc_transfer_query(self):
        bridge = FakeBridge()
        backend = _make_rk3588(bridge)
        backend.open()
        self.assertTrue(bridge.opened)
        buf = backend.alloc_buffer(32, "io")
        backend.upload(buf, 0, b"hello")
        self.assertEqual(backend.download(buf, 0, 5), b"hello")
        info = backend.query_memory()
        self.assertEqual(info.owned_payload_bytes, 32)
        self.assertGreaterEqual(info.owned_allocated_bytes, 32)
        # The custom-ops path does not publish device/system totals.
        self.assertIsNone(info.device_total_bytes)
        self.assertIsNone(info.system_total_bytes)
        self.assertFalse(info.totals_known)
        backend.close()
        self.assertTrue(bridge.closed)

    def test_synchronous_submit_and_wait(self):
        bridge = FakeBridge()
        backend = _make_rk3588(bridge)
        backend.open()
        out = _configured_output(backend)
        payload = RK3588Payload(words=(0x11, 0x22, 0x33), output=(out, 0),
                                mode=MODE_SEPARATE)
        job = backend.create_job(payload)
        token = backend.submit(job)
        # Synchronous: already complete, and wait only observes.
        self.assertIs(token.state, TokenState.COMPLETE)
        self.assertIs(backend.wait(token), TokenState.COMPLETE)
        self.assertEqual(len(bridge.executed), 1)
        self.assertEqual(bridge.executed[0][1], MODE_SEPARATE)
        # Async completion is honestly unavailable.
        self.assertIs(backend.capability_state(Capability.ASYNC_COMPLETION),
                      CapabilityState.UNAVAILABLE)
        self.assertEqual(backend.identity.evidence, Evidence.BUILD)

    def test_chain_mode_passed_through(self):
        bridge = FakeBridge()
        backend = _make_rk3588(bridge)
        backend.open()
        out = _configured_output(backend)
        job = backend.create_job(RK3588Payload(words=(1,), output=(out, 0),
                                               mode=MODE_CHAIN))
        backend.submit(job)
        self.assertEqual(bridge.executed[0][1], MODE_CHAIN)

    def test_batch_and_profiled_submit(self):
        bridge = FakeBridge()
        backend = _make_rk3588(bridge)
        backend.open()
        jobs = []
        for _ in range(2):
            out = _configured_output(backend)
            jobs.append(backend.create_job(
                RK3588Payload(words=(1, 2), output=(out, 0), mode=MODE_SEPARATE)))
        backend.submit_batch(jobs, MODE_SEPARATE)
        self.assertEqual(len(bridge.batches), 1)
        token, profile = backend.submit_profiled(jobs[0], core_mask=1, request_id=42)
        self.assertIs(token.state, TokenState.COMPLETE)
        self.assertEqual(profile["actual_mask"], 1)
        self.assertEqual(profile["driver_service_ns"], 9)

    def test_register_command_payload_requires_words(self):
        bridge = FakeBridge()
        backend = _make_rk3588(bridge)
        backend.open()
        out = _configured_output(backend)
        with self.assertRaises(InvalidPayload):
            backend.create_job(RK3588Payload(words=(), output=(out, 0)))

    def test_engine_error_maps_to_backend_error(self):
        bridge = FakeBridge(execution_error=-4)
        backend = _make_rk3588(bridge)
        backend.open()
        out = _configured_output(backend)
        job = backend.create_job(RK3588Payload(words=(1,), output=(out, 0)))
        from backends import BackendError
        with self.assertRaises(BackendError):
            backend.submit(job)

    def test_library_payload_rejected(self):
        bridge = FakeBridge()
        backend = _make_rk3588(bridge)
        backend.open()
        with self.assertRaises(BackendMismatch):
            backend.create_job(RK182XPayload(model_name="m.rknn3"))
        # The adapter never advertises the RKNN3 ABI.
        self.assertIs(backend.capability_state(Capability.RKNN3_MODEL),
                      CapabilityState.UNAVAILABLE)

    def test_buffer_ownership_enforced(self):
        bridge = FakeBridge()
        backend = _make_rk3588(bridge)
        backend.open()
        buf = backend.alloc_buffer(8)
        backend.free_buffer(buf)
        with self.assertRaises(BufferOwnershipError):
            backend.upload(buf, 0, b"a")

    def test_open_failure_is_not_faked(self):
        bridge = FakeBridge(fail_open=True)
        backend = _make_rk3588(bridge)
        with self.assertRaises(DeviceUnavailable):
            backend.open()


# --------------------------------------------------------------------------- rk182x


class RK182XAdapterTests(unittest.TestCase):
    def test_unavailable_without_runtime(self):
        backend = RK182XBackend()
        self.assertEqual(backend.identity.evidence, Evidence.CONTRACT)
        for cap in (Capability.SUBMIT, Capability.WAIT, Capability.UPLOAD,
                    Capability.DOWNLOAD, Capability.QUERY_MEMORY,
                    Capability.RKNN3_MODEL):
            self.assertIs(backend.capability_state(cap), CapabilityState.UNAVAILABLE)
        # Never accepts RK3588 regcmd.
        self.assertIs(backend.capability_state(Capability.REGCMD_PROGRAM),
                      CapabilityState.UNAVAILABLE)
        # Host staging is declared; P2P is not.
        self.assertIs(backend.capability_state(Capability.HOST_STAGED_MIGRATION),
                      CapabilityState.UNAVAILABLE)
        self.assertIs(backend.capability_state(Capability.PEER_TO_PEER_TRANSFER),
                      CapabilityState.UNAVAILABLE)
        backend.open()
        with self.assertRaises(UnsupportedOperation):
            backend.alloc_buffer(16)
        with self.assertRaises(UnsupportedOperation):
            backend.query_memory()

    def test_contract_only_runtime_binds_but_does_not_execute(self):
        backend = RK182XBackend(runtime=ContractOnlyRKNN3Runtime())
        info = backend._bound_info
        self.assertEqual(info["platform"], "rk182x")
        self.assertTrue(info["contract_only"])
        self.assertTrue(backend.runtime_bound())
        self.assertTrue(backend.runtime_contract_only())
        self.assertIs(backend.capability_state(Capability.SUBMIT),
                      CapabilityState.UNAVAILABLE)
        backend.open()
        with self.assertRaises(UnsupportedOperation):
            backend.alloc_buffer(16)

    def test_contract_only_runtime_refuses_submit(self):
        backend = RK182XBackend(runtime=ContractOnlyRKNN3Runtime())
        payload = RK182XPayload(model_name="qwen.rknn3")
        # create_job itself is gated because execution is unavailable.
        backend.open()
        with self.assertRaises(UnsupportedOperation):
            backend.create_job(payload)

    def test_wrong_platform_rejected(self):
        runtime = ContractOnlyRKNN3Runtime(platform_name="rk3588")
        with self.assertRaises(BackendMismatch):
            RK182XBackend(runtime=runtime)

    def test_old_version_rejected(self):
        runtime = ContractOnlyRKNN3Runtime(api_version_value=(0, 9, 0))
        with self.assertRaises(UnsupportedOperation):
            RK182XBackend(runtime=runtime)

    def test_malformed_version_unknown(self):
        class Bad:
            def api_version(self):
                return ("one",)

            def platform(self):
                return "rk182x"

            def capabilities(self):
                return {}

        with self.assertRaises(UnknownSDKBehavior):
            RK182XBackend(runtime=Bad())

    def test_missing_entry_point_rejected(self):
        runtime = ContractOnlyRKNN3Runtime(
            capability_map={"rknn3_run": True, "rknn3_create_mem": False,
                            "rknn3_mem_sync": True})
        with self.assertRaises(UnsupportedOperation):
            RK182XBackend(runtime=runtime)

    def test_rk3588_payload_rejected(self):
        backend = RK182XBackend()
        backend.open()
        with self.assertRaises(BackendMismatch):
            backend.create_job(RK3588Payload(words=(1, 2, 3)))
        with self.assertRaises(BackendMismatch):
            backend.create_job(MockPayload())

    def test_custom_contract(self):
        contract = RKNN3RuntimeContract(required_api_major=2, required_platform="rk182x")
        backend = RK182XBackend(contract=contract)
        self.assertEqual(backend._contract.required_api_major, 2)
        self.assertIn("rknn3_run", backend._contract.required_entry_points)

    def test_migration_plan_is_host_staged_only(self):
        plan = RK182XBackend().migration_plan()
        self.assertTrue(plan["host_staged"])
        self.assertFalse(plan["peer_to_peer"])
        self.assertIn("unmeasured", plan["timing_scope"])


# --------------------------------------------------------------------------- build


class BridgeBuildTests(unittest.TestCase):
    def test_bridge_sources_present(self):
        bridge_dir = _PKG_ROOT / "backends" / "bridge"
        self.assertTrue((bridge_dir / "co_backend_bridge.c").is_file())
        self.assertTrue((bridge_dir / "CMakeLists.txt").is_file())

    def test_real_bridge_loads_if_built(self):
        path = find_bridge_library()
        if path is None:
            self.skipTest(
                "libco_backend_bridge not built; see backends/bridge/CMakeLists.txt"
            )
        bridge = CtypesBridge(str(path))
        self.assertEqual(bridge._lib.cobs_abi_version(), BRIDGE_ABI_VERSION)
        name = bridge.device_name()
        self.assertIsInstance(name, (str, type(None)))
        # Host ABI checks do not open hardware; smoke.py is the explicit device gate.
        self.assertTrue(callable(bridge.open_device))



class BackendLifetimeRegressionTests(unittest.TestCase):
    def test_untouched_allocations_consume_capacity(self):
        b=MockBackend(clock=FakeClock(),capacity_bytes=10);b.open();b.alloc_buffer(8)
        with self.assertRaises(MemoryBoundsError):b.alloc_buffer(8)
        b.close()
    def test_pending_timeout_stays_live_and_close_rejects(self):
        b=MockBackend(clock=FakeClock());b.open();j=b.create_job(MockPayload(duration_ns=100));t=b.submit(j)
        with self.assertRaises(OperationTimeout):b.wait(t,timeout_ns=1)
        self.assertEqual(t.state,TokenState.PENDING)
        for op in (lambda:b.close(),lambda:b.release_token(t),lambda:b.release_job(j),lambda:b.submit(j)):
            with self.assertRaises(BufferOwnershipError):op()
        b.wait(t);b.close();b.open()
        with self.assertRaises(BufferOwnershipError):b.wait(t)
        b.close()
    def test_closed_and_host_staged_transfer(self):
        from backends.interface import transfer_between
        a=MockBackend(clock=FakeClock());b=MockBackend(clock=FakeClock())
        with self.assertRaises(DeviceUnavailable):a.alloc_buffer(8)
        a.open();b.open();x=a.alloc_buffer(8);y=b.alloc_buffer(8)
        a.transfer(x,direction='upload',data=b'12345678')
        r=transfer_between(a,x,b,y,length=8)
        self.assertEqual(r['bytes'],8)
        self.assertEqual(b.transfer(y,direction='download',length=8),b'12345678')
        with self.assertRaises(BufferOwnershipError):b.transfer(x,direction='download',length=1)
        with self.assertRaises(MemoryBoundsError):transfer_between(a,x,b,y,length=9)
        a.close();b.close()
    def test_prepared_job_pins_buffer_and_stale_handles(self):
        b=_make_rk3588(FakeBridge());b.open();out=_configured_output(b)
        job=b.create_job(RK3588Payload(words=(1,2,3),output=(out,0)))
        with self.assertRaises(BufferOwnershipError):b.free_buffer(out)
        with self.assertRaises(BufferOwnershipError):b.configure_tensor(out,'matrix',rows=32,cols=32)
        b.release_job(job);b.free_buffer(out);b.close()
        with self.assertRaises(DeviceUnavailable):b.download(out,0,1)
    def test_rk182x_incomplete_executable_binding_rejected(self):
        runtime=ContractOnlyRKNN3Runtime(contract_only=False)
        with self.assertRaises(UnsupportedOperation):RK182XBackend(runtime=runtime)
        runtime=ContractOnlyRKNN3Runtime(api_version_value=(2,0,0))
        with self.assertRaises(UnsupportedOperation):RK182XBackend(runtime=runtime)
        runtime=ContractOnlyRKNN3Runtime(capability_map={'rknn3_run':1})
        with self.assertRaises(UnknownSDKBehavior):RK182XBackend(runtime=runtime)
    def test_native_invalid_ids_and_old_abi_fail_closed(self):
        path=find_bridge_library()
        if path is None:self.skipTest('native bridge not built')
        lib=CtypesBridge(str(path))._lib
        self.assertEqual(lib.cobs_buffer_payload(1),0)
        self.assertNotEqual(lib.cobs_buffer_free(1),0)
        self.assertNotEqual(lib.cobs_guard_check(1),0)
        self.assertNotEqual(lib.cobs_job_validate(1),0)
        self.assertNotEqual(lib.cobs_execute(-1,1,99),0)


if __name__ == "__main__":
    unittest.main()
