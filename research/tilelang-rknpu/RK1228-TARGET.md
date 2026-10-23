# RK1228 independent target admission contract (pending hardware identification)

_Status: not started against real hardware.  No RK1228 device ID, driver ABI,
memory-layout facts, register recipes or acceptance numbers are asserted here._

The RK1228 work must reuse the **target-neutral RKIR, graph legalization and
planner common layer**.  It must not inherit any RK3588 register recipe, core
mask, uAPI descriptor, memory-transfer assumption or board-admission result.

## 1. Identify before defining the profile

On the actual RK1228 device, capture and commit sanitized evidence for:

1. board/model identifier, accelerator vendor/device ID and PCIe/platform path;
2. kernel module name/version and the exact userspace uAPI header or source
   commit used to build the runtime;
3. device node(s) and permissions; IOMMU/DMA address width and alignment;
4. buffer allocation, mapping, cache maintenance and synchronization calls;
5. submit descriptor layout, queue/doorbell model, completion IRQ/error codes
   and timeout/reset behavior;
6. core/partition topology, frequency/power controls and allowed scheduling
   policy;
7. supported input/output dtypes and accumulator/writeback behavior.

Until (1)-(7) are proven, `chip_id`, `driver_abi_id`, `runtime_abi_id`, hashes
and capability entries remain **unset**.  Do not create a placeholder manifest
that can be selected by a normal compile.

## 2. Separate profile and capability manifest

RK1228 requires its own `TargetProfile` document and capability IDs, with at
least:

* a target-specific `chip_id`, `driver_abi_id`, `runtime_abi_id`, layout and
  template version set;
* a target-specific capability manifest hash and evidence references;
* a target-specific `execution_policy`, `numeric_policy`, budgets and
  `primitive_extent_limit`;
* capability `validation_domain` and `uncovered_risks` populated from RK1228
  experiments, never copied from RK3588.

Capability IDs or aliases may be shared only after proving the emitted task
sequence and numeric semantics are identical.  Sharing an ID must not imply
that RK3588 register generation or runtime evidence transfers.

## 3. Runtime identity and fail-closed selection

The RK1228 runtime/library must fail closed when compiled for another target:

* plans carry a target profile hash plus target runtime ABI in the descriptor;
* the runtime refuses a plan whose target identity differs from its own;
* the Python loader verifies the profile manifest and library identity before
  BO allocation or device open;
* no chip-name environment variable alone may enable a target.

The existing RK3588 C ABI version is not evidence that RK1228 shares the same
descriptor or submit ABI.  A separate runtime ABI version is required if any
field, size or addressing rule differs.

## 4. Separate memory-transfer contract

RK1228 must define, independently:

* logical host binding → physical tile packing rules;
* scalar/channel/constant expansion and upload-once semantics;
* buffer ownership, alignment, cacheability, IOMMU/address limits and sync;
* RAW/WAR/WAW and completion ordering for chained versus separate submits;
* guard behavior for non-finite inputs/outputs and failed submissions;
* byte-accounting definitions for upload/download and expanded constants.

Reuse RKIR Value/Op semantics and the common planner interfaces only.  A new
emitter/template family is expected unless RK1228 hardware is proven to accept
the RK3588 custom-ops programs byte-for-byte.

## 5. Admission gates

Minimum RK1228 acceptance, mirroring but not inheriting the RK3588 campaign:

1. identify and hash the runtime/driver/toolchain;
2. validate the target profile hash on every plan;
3. compile and execute MatMul, projection, staged-FP16 FFN, constant broadcast
   and explicit guarded ragged tails;
4. run chain and separate modes, with independent numeric references;
5. run non-finite input/constant guards and prove output non-publication;
6. verify all supported core/partition policies independently;
7. archive sanitized logs without credentials;
8. state the tested shape/layout/numeric domain and every remaining risk.

Only after these gates pass may RK1228-specific capability status move from
`unsupported`/`experimental` toward `enabled`.
