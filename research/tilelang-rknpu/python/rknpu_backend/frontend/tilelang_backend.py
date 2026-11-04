# SPDX-License-Identifier: GPL-3.0-or-later
"""Opt-in TileLang backend registration for the RKNPU lowering pilot.

What the installed build actually offers (TileLang 0.1.15, vendored TVM 0.25.dev0;
recorded with file:line in results/codex-pass-65/backend-registration/):

    tilelang/backend/module.py:28    BackendModule          the registration manifest
    tilelang/backend/module.py:238   register_backend       validates and installs it
    tilelang/backend/module.py:310   create_backend_context resolves target -> context
    tilelang/backend/pass_pipeline/pipeline.py:11
                                     PassPipeline           callable (IRModule, Target)
    tilelang/backend/device_codegen.py:31
                                     DeviceCodegen          build / build_without_compile
    tilelang/backend/execution_backend.py:18
                                     ExecutionBackendSpec   execution policy value
    tilelang/backend/target.py:45    register_target_normalizer  name -> target spec
    tilelang/engine/lower.py:136     mod = context.lower(mod)    the compile call site

Target *kinds* are not part of that Python surface: they are registered in C++
with TVM_REGISTER_TARGET_KIND (3rdparty/tvm/src/target/target_kind.cc,
src/ascend/target.cc:36), so a Python package can adopt an existing kind but
cannot create one.  This build has no "rknpu" kind - Target("rknpu") raises
ValueError: Target kind "rknpu" is not defined - and this module does not fake
one.  It registers the RKNPU backend on the external-device kind "ext_dev" and
claims only targets explicitly marked with the target key "rknpu"::

    {"kind": "ext_dev", "keys": ["rknpu"]}

"rknpu" therefore works as a target *name* through a registered target
normalizer that always resolves to that marked target; it is not a new kind.
Targets that do not carry the key are left alone, so "ext_dev" itself stays
unclaimed (create_backend_context still refuses a bare ext_dev target).

What the registered lowering does::

    context = create_backend_context("rknpu")
    context.lower(module)      -> tilelang_lowering.compile_tilelang_ir(module PrimFunc)
                                  full-logical GEMM:
                                    TIR proof -> target rk_* TIR -> task consumer
                                    -> existing planner resources/submits -> RKPlan
                                  guarded-tail GEMM:
                                    normalise -> pass_localise -> pass_instantiate
                                    -> tasks_from_module (ordered rk_* operations)
                                    -> compile_from_tasks / build_from_tasks -> RKPlan

Both forms are explicit conversions over the real TIR.  The dispatcher never
falls back from one form to the other after a rejection.

The product is an rknpu_backend.Plan; this pilot emits no device C source, no
runtime, and maps no GPU thread/shared/warp semantics onto RKNPU hardware.  The
plan is the artifact of ONE compilation and it travels with the IR that produced
it: the lowering returns the module carrying

    rknpu_plan_json   canonical JSON of plan.to_dict(), hash included
    rknpu_plan_hash   the plan_hash of that document
    rknpu_stages      the stages that ran
    rknpu_tasks       the number of instantiated tasks
    rknpu_target_key  the target key this backend owns

on the module AND on its PrimFunc: tirx.transform.Filter keeps module attributes
on the empty device module, while a function-level pass keeps only the function,
so both copies are needed for the artifact to survive to codegen.  The registered
DeviceCodegen re-attaches that artifact instead of refusing, so
tilelang.engine.lower(module, target="rknpu") returns a CompiledArtifact whose
kernel_source IS the canonical plan JSON.  It still refuses a module that carries
no plan artifact, i.e. one this pipeline did not lower.

The engine's call site (tilelang/engine/lower.py:191) reads
codegen_mod.inspect_source() after device codegen, and this build's tvm.IRModule
defines no inspect_source, so register_rknpu_backend() installs that missing
method as an explicit, documented shim - see install_inspect_source.

Reading the product never goes through a process global: plan_from_module,
plan_from_artifact and compile_to_plan return it, and serialize it the way
save_plan/load_plan already do.  last_lowering() is a diagnostic log only.

Programs outside the closed full-logical GEMM and guarded-tail FP16 GEMM forms
are rejected by the selected chain's own reject() diagnostics; the pipeline never
uses the structured prototype; the generic adapter extractor is retired.

Importing this module registers nothing; call register_rknpu_backend(), or use
the registered() context manager for a scoped registration.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ..errors import RK_E_UNSUPPORTED, reject

RKNPU_TARGET_KIND = "ext_dev"
RKNPU_TARGET_KEY = "rknpu"
RKNPU_TARGET_NAME = "rknpu"
RKNPU_BACKEND_NAME = "rknpu"
RKNPU_EXECUTION_BACKEND = "rknpu_plan"
# BackendModule requires pipeline.name == target_kind (module.py:55-57), so the
# pipeline carries the kind name; the RKNPU identity is the backend name, the
# target key and the pipeline function.
RKNPU_PIPELINE_NAME = RKNPU_TARGET_KIND

_REQUIRED_API = ("BackendModule", "DeviceCodegen", "ExecutionBackendSpec",
                 "PassPipeline", "get_backend", "list_backends",
                 "register_backend", "register_target_normalizer")

# The plan travels on the IR it produced: one canonical document plus its hash,
# written to the module and to its PrimFunc (see _annotate).
RKNPU_PLAN_JSON_ATTR = "rknpu_plan_json"
RKNPU_PLAN_HASH_ATTR = "rknpu_plan_hash"
RKNPU_STAGES_ATTR = "rknpu_stages"
RKNPU_TASKS_ATTR = "rknpu_tasks"
RKNPU_TARGET_KEY_ATTR = "rknpu_target_key"


def backend_api() -> Any:
    """Return the installed tilelang.backend package, or reject.

    The import is deferred so that importing this module never needs TileLang.
    """
    try:
        import tilelang  # noqa: F401  (loads the vendored TVM)
        from tilelang import backend as api
    except Exception as err:  # noqa: BLE001 - the installed build decides
        reject(RK_E_UNSUPPORTED,
               "the installed TileLang exposes no tilelang.backend registry, so "
               "there is nothing to register the RKNPU chain with",
               stage="backend", actual=f"{type(err).__name__}: {err}",
               required=list(_REQUIRED_API))
    missing = [name for name in _REQUIRED_API if not hasattr(api, name)]
    if missing:
        reject(RK_E_UNSUPPORTED,
               "the installed tilelang.backend registry is incomplete",
               stage="backend", required=missing,
               actual=sorted(name for name in dir(api)
                             if not name.startswith("_")))
    return api


def available() -> bool:
    """Whether this build exposes the backend registry this module registers into."""
    try:
        backend_api()
    except Exception:  # noqa: BLE001 - any failure means "not registrable here"
        return False
    return True


def rknpu_target() -> dict[str, Any]:
    """The target specification this backend owns: ext_dev marked "rknpu"."""
    return {"kind": RKNPU_TARGET_KIND, "keys": [RKNPU_TARGET_KEY]}


def is_rknpu_target(target: Any) -> bool:
    """Whether target is the marked external-device target this backend owns."""
    kind = getattr(getattr(target, "kind", None), "name", None)
    if kind != RKNPU_TARGET_KIND:
        return False
    return RKNPU_TARGET_KEY in tuple(getattr(target, "keys", ()) or ())


def _normalise_target(target: Any) -> Any:
    """Map the "rknpu" target name onto the marked ext_dev target, else None."""
    if isinstance(target, dict):
        if str(target.get("kind", "")).strip().lower() != RKNPU_TARGET_NAME:
            return None
        extra = sorted(set(target) - {"kind"})
        if extra:
            reject(RK_E_UNSUPPORTED,
                   "the rknpu target name takes no options: the backend owns "
                   "ext_dev targets marked with the rknpu key",
                   stage="backend", actual=extra)
        return rknpu_target()
    if isinstance(target, str) and target.strip().lower() == RKNPU_TARGET_NAME:
        return rknpu_target()
    return None


@dataclass(frozen=True)
class Lowering:
    """One run of the registered lowering pipeline."""

    target: str
    plan_hash: str
    stages: tuple[tuple[str, str], ...]
    facts: dict[str, Any]
    tasks: tuple[dict[str, Any], ...]
    plan: Any = field(repr=False, default=None)
    entry: Any = field(repr=False, default=None)

    def to_dict(self) -> dict[str, Any]:
        """The record as plain data (the plan object itself is not included)."""
        return {"target": self.target, "plan_hash": self.plan_hash,
                "stages": [list(stage) for stage in self.stages],
                "facts": self.facts, "tasks": len(self.tasks),
                "entry": f"{self.entry.__module__}.{self.entry.__qualname__}"}


_LOWERINGS: list[Lowering] = []


def lowerings() -> tuple[Lowering, ...]:
    """Every lowering this process ran through the registered pipeline."""
    return tuple(_LOWERINGS)


def last_lowering() -> Lowering | None:
    """The most recent registered lowering, if any."""
    return _LOWERINGS[-1] if _LOWERINGS else None


def clear_lowerings() -> None:
    """Forget the recorded lowerings (evidence only; the plans stay valid)."""
    _LOWERINGS.clear()


def _only_prim_func(mod: Any) -> Any:
    from tvm import IRModule, tirx

    if not isinstance(mod, IRModule):
        reject(RK_E_UNSUPPORTED,
               "the RKNPU pipeline lowers a tvm.IRModule", stage="backend",
               actual=type(mod).__name__)
    functions = list(mod.functions.items())
    if len(functions) != 1:
        reject(RK_E_UNSUPPORTED,
               "the RKNPU pilot lowers exactly one kernel per module",
               stage="backend", actual=[name for name, _ in functions])
    name, func = functions[0]
    if not isinstance(func, tirx.PrimFunc):
        reject(RK_E_UNSUPPORTED,
               "the RKNPU pipeline lowers a PrimFunc", stage="backend",
               actual=f"{name}: {type(func).__name__}")
    return func


def _require_rknpu_target(target: Any) -> None:
    """Reject any target this backend does not own."""
    if is_rknpu_target(target):
        return
    reject(RK_E_UNSUPPORTED,
           "the RKNPU pipeline only lowers ext_dev targets marked with the "
           "rknpu key", stage="backend",
           actual=str(target).replace("\n", " "),
           required=rknpu_target()["keys"])


def _as_prim_func(value: Any) -> Any:
    """The single PrimFunc of a module, or the PrimFunc passed in."""
    from tvm import IRModule, tirx

    if isinstance(value, IRModule):
        return _only_prim_func(value)
    if isinstance(value, tirx.PrimFunc):
        return value
    reject(RK_E_UNSUPPORTED,
           "the RKNPU pipeline compiles a PrimFunc or a single-function "
           "IRModule", stage="backend", actual=type(value).__name__)


def _attr_keys(attrs: Any) -> list[str]:
    """The attribute names of a DictAttrs, sorted; [] when there is none."""
    if attrs is None:
        return []
    return sorted(str(key) for key in attrs.keys())


def _attr_text(attrs: Any, key: str) -> str | None:
    """The string value of one attribute, or None when it is not set."""
    if attrs is None or key not in attrs:
        return None
    return str(attrs[key])


def _carried(mod: Any) -> tuple[str | None, str | None]:
    """The (plan JSON, plan hash) artifact a module or a PrimFunc carries."""
    if mod is None:
        return None, None
    attrs = getattr(mod, "attrs", None)
    plan_json = _attr_text(attrs, RKNPU_PLAN_JSON_ATTR)
    if plan_json is not None:
        return plan_json, _attr_text(attrs, RKNPU_PLAN_HASH_ATTR)
    functions = getattr(mod, "functions", None)
    for _, func in (functions.items() if functions is not None else ()):
        func_attrs = getattr(func, "attrs", None)
        plan_json = _attr_text(func_attrs, RKNPU_PLAN_JSON_ATTR)
        if plan_json is not None:
            return plan_json, _attr_text(func_attrs, RKNPU_PLAN_HASH_ATTR)
    return None, None


def plan_json_from_module(mod: Any) -> str | None:
    """The serialized RKPlan this module or PrimFunc carries, else None."""
    return _carried(mod)[0]


def plan_from_json(plan_json: Any) -> Any:
    """The Plan a canonical plan document encodes, with its hash re-checked.

    Plan.from_dict recomputes plan_hash over the canonical bytes, so a document
    that does not round-trip is rejected here instead of being trusted.
    """
    from ..plan import Plan
    from ..serialize import loads

    if not isinstance(plan_json, str) or not plan_json.strip():
        reject(RK_E_UNSUPPORTED,
               "an RKPlan artifact is canonical JSON text", stage="backend",
               actual=type(plan_json).__name__)
    return Plan.from_dict(loads(plan_json))


def plan_from_module(mod: Any) -> Any:
    """The Plan carried by a module this pipeline lowered (no global lookup)."""
    plan_json = plan_json_from_module(mod)
    if plan_json is None:
        reject(RK_E_UNSUPPORTED,
               "this module carries no RKPlan artifact, so the RKNPU pipeline "
               "did not lower it", stage="backend",
               actual=_attr_keys(getattr(mod, "attrs", None)))
    return plan_from_json(plan_json)


def plan_from_artifact(artifact: Any) -> Any:
    """The Plan a tilelang CompiledArtifact carries: its source, else its IR."""
    kernel_source = getattr(artifact, "kernel_source", None)
    if isinstance(kernel_source, str) and kernel_source.strip():
        return plan_from_json(kernel_source)
    for name in ("device_mod", "host_mod"):
        plan_json = plan_json_from_module(getattr(artifact, name, None))
        if plan_json is not None:
            return plan_from_json(plan_json)
    reject(RK_E_UNSUPPORTED,
           "this artifact carries no RKPlan artifact: neither its kernel_source "
           "nor its modules were produced by the RKNPU pipeline",
           stage="backend", actual=type(artifact).__name__)


def _annotate(module: Any, *, plan_json: str, plan_hash: str, stages: Any,
              tasks: Any) -> Any:
    """module carrying the plan artifact on the module and on its PrimFunc."""
    from tvm import IRModule

    functions = {name: func.with_attr(RKNPU_PLAN_JSON_ATTR, plan_json)
                 .with_attr(RKNPU_PLAN_HASH_ATTR, plan_hash)
                 for name, func in module.functions.items()}
    annotated = IRModule(functions, module.attrs)
    return (annotated
            .with_attr(RKNPU_PLAN_JSON_ATTR, plan_json)
            .with_attr(RKNPU_PLAN_HASH_ATTR, plan_hash)
            .with_attr(RKNPU_STAGES_ATTR,
                       ",".join(stage.name for stage in stages))
            .with_attr(RKNPU_TASKS_ATTR, len(tasks))
            .with_attr(RKNPU_TARGET_KEY_ATTR, RKNPU_TARGET_KEY))


@dataclass(frozen=True)
class PlanCompilation:
    """One explicit RKNPU compilation product; nothing here is a global."""

    plan: Any = field(repr=False)
    plan_json: str = field(repr=False)
    plan_hash: str
    stages: tuple[Any, ...] = field(repr=False)
    facts: Any = field(repr=False)
    tasks: tuple[dict[str, Any], ...] = field(repr=False)
    module: Any = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """The record as plain data (the plan object itself is not included)."""
        return {"plan_hash": self.plan_hash, "plan_bytes": len(self.plan_json),
                "stages": [stage.to_dict() for stage in self.stages],
                "facts": self.facts.to_dict(), "tasks": len(self.tasks),
                "module_attrs": _attr_keys(getattr(self.module, "attrs", None))}


def _pilot_options() -> Any:
    """The numeric contract the pilot chain proves."""
    from ..options import CompileOptions

    return CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=False)


def compile_to_plan(prim_func_or_module: Any, *, profile: Any = None,
                    options: Any = None, simplify: bool = False,
                    name: str = "rknpu-pass") -> PlanCompilation:
    """Run the RKNPU chain and return its product, attached to the module.

    The explicit compilation entry: the Plan together with the canonical JSON it
    round-trips through, the hash, the stages, the proven facts, the tasks and
    the IRModule carrying the artifact.  Nothing is recorded in _LOWERINGS - the
    product is the return value.
    """
    from tvm import IRModule

    from ..capabilities import default_profile
    from ..serialize import canonical_json
    from . import tilelang_lowering

    if isinstance(prim_func_or_module, IRModule):
        module_attrs = getattr(prim_func_or_module, "attrs", None)
        module_requests = [
            {"key": str(key), "value": str(module_attrs[key])}
            for key in (module_attrs.keys() if module_attrs is not None else ())
            if str(key).startswith("rknpu.")]
        if module_requests:
            reject(RK_E_UNSUPPORTED,
                   "module-level rknpu declarations are not supported; attach "
                   "the request to the PrimFunc or a supported IR annotation "
                   "scope", stage="backend", actual=module_requests,
                   required=["PrimFunc.attrs", "SBlock/For/Call annotations"])
    prim_func = _as_prim_func(prim_func_or_module)
    plan, stages, facts, tasks = tilelang_lowering.compile_tilelang_ir(
        prim_func, profile=profile or default_profile(),
        options=options or _pilot_options(), simplify=simplify, name=name)
    plan_json = canonical_json(plan.to_dict())
    return PlanCompilation(
        plan=plan, plan_json=plan_json, plan_hash=plan.plan_hash,
        stages=tuple(stages), facts=facts, tasks=tuple(tasks),
        module=_annotate(IRModule({"main": prim_func}), plan_json=plan_json,
                         plan_hash=plan.plan_hash, stages=stages, tasks=tasks))


def lower_module(mod: Any, target: Any) -> Any:
    """The registered lowering over one supported full-logical or guarded GEMM.

    Returns the module annotated with the artifact of THIS compilation: the
    canonical plan JSON and its hash, on the module and on its PrimFunc.  The
    plan is a product of the returned module (plan_from_module) and of the
    engine's CompiledArtifact (plan_from_artifact); _LOWERINGS only records it
    for diagnostics.  No device code is produced.
    """
    from . import tilelang_lowering

    _require_rknpu_target(target)
    compilation = compile_to_plan(mod)
    _LOWERINGS.append(Lowering(
        target=str(target).replace("\n", " "),
        plan_hash=compilation.plan_hash,
        stages=tuple((stage.name, stage.kind) for stage in compilation.stages),
        facts=compilation.facts.to_dict(), tasks=compilation.tasks,
        plan=compilation.plan,
        entry=tilelang_lowering.entry_for(_as_prim_func(mod))))
    return compilation.module


def emit_plan_artifact(mod: Any, target: Any) -> Any:
    """The registered device codegen: return the module carrying its RKPlan.

    Device codegen is where this pilot's product is re-attached instead of being
    refused.  A module the RKNPU pipeline lowered already carries the canonical
    plan JSON, which the engine reads back through inspect_source(); a module
    without that artifact was not lowered here, and that is the one real
    refusal: there is still no RKNPU C source, no runtime, and no mapping of GPU
    thread/shared/warp semantics onto RKNPU hardware.
    """
    _require_rknpu_target(target)
    plan_json, plan_hash = _carried(mod)
    if plan_json is None:
        reject(RK_E_UNSUPPORTED,
               "the RKNPU device codegen emits the RKPlan artifact of a module "
               "the RKNPU pipeline lowered, and this module carries none: no "
               "RKNPU C source, no runtime, and no mapping of GPU "
               "thread/shared/warp semantics onto RKNPU hardware",
               stage="codegen",
               actual={"target":
                       getattr(getattr(target, "kind", None), "name", str(target)),
                       "module_attrs": _attr_keys(getattr(mod, "attrs", None))},
               alternative="rknpu_pass.compile_guarded_tail(...) then save_plan")
    if plan_hash is None:
        plan_hash = plan_from_json(plan_json).plan_hash
    return (mod.with_attr(RKNPU_PLAN_JSON_ATTR, plan_json)
            .with_attr(RKNPU_PLAN_HASH_ATTR, plan_hash)
            .with_attr(RKNPU_TARGET_KEY_ATTR, RKNPU_TARGET_KEY))


def refuse_device_codegen(mod: Any, target: Any) -> Any:
    """The device codegen name the manifest used before the plan travelled.

    Same registered entry: it emits the carried artifact, and still refuses a
    module that carries none (see emit_plan_artifact).
    """
    return emit_plan_artifact(mod, target)


_MANIFEST: Any = None


def backend_module() -> Any:
    """The RKNPU BackendModule for this build, built once and cached."""
    global _MANIFEST
    if _MANIFEST is None:
        api = backend_api()
        _MANIFEST = api.BackendModule(
            name=RKNPU_BACKEND_NAME,
            target_kinds=(RKNPU_TARGET_KIND,),
            supports_target=is_rknpu_target,
            pipelines={RKNPU_TARGET_KIND: api.PassPipeline(
                RKNPU_PIPELINE_NAME, lower_module)},
            device_codegens={RKNPU_TARGET_KIND: api.DeviceCodegen(
                RKNPU_BACKEND_NAME, build=emit_plan_artifact,
                build_without_compile=emit_plan_artifact)},
            execution_backends=(api.ExecutionBackendSpec(
                RKNPU_EXECUTION_BACKEND, supports_target=is_rknpu_target,
                enable_host_codegen=False, enable_device_compile=False),),
        )
    return _MANIFEST


def inspect_source_shim(self: Any) -> str:
    """tvm.IRModule.inspect_source for a build that does not define it.

    tilelang.engine.lower (0.1.15 over the vendored TVM 0.25.dev0) calls
    codegen_mod.inspect_source() at tilelang/engine/lower.py:191, and this build's
    tvm.IRModule defines no such method: hasattr(tvm.IRModule, "inspect_source") is
    False even after importing tilelang.backend, so the engine cannot finish a
    compilation for any backend here, this one included.

    register_rknpu_backend() installs this function as that missing method, and
    only when the installed build lacks it, so one compile entry can complete.  It
    returns the canonical RKPlan JSON the module carries - the same text
    plan_from_module parses, and this backend's kernel_source - and it raises a
    clear RK_E_UNSUPPORTED for a module that carries no plan artifact instead of
    returning invented source.
    """
    plan_json = plan_json_from_module(self)
    if plan_json is None:
        reject(RK_E_UNSUPPORTED,
               "inspect_source is a shim installed by "
               "rknpu_backend.frontend.tilelang_backend.register_rknpu_backend, "
               "and it only answers for a module the RKNPU pipeline lowered; this "
               "module carries no RKPlan artifact", stage="backend",
               actual=_attr_keys(getattr(self, "attrs", None)))
    return plan_json


def install_inspect_source() -> bool:
    """Install the shim when this build lacks tvm.IRModule.inspect_source."""
    from tvm import IRModule

    if hasattr(IRModule, "inspect_source"):
        return False
    IRModule.inspect_source = inspect_source_shim
    return True


def remove_inspect_source() -> bool:
    """Remove the shim this module installed, if it is still the installed one."""
    from tvm import IRModule

    if IRModule.__dict__.get("inspect_source") is not inspect_source_shim:
        return False
    delattr(IRModule, "inspect_source")
    return True


def register_rknpu_backend() -> Any:
    """Register the RKNPU backend and its "rknpu" target name; idempotent.

    Also installs the tvm.IRModule.inspect_source shim this build's engine needs
    (see install_inspect_source).  Returns the registered BackendModule.  Call it
    explicitly: importing this module registers nothing.
    """
    api = backend_api()
    backend = api.register_backend(backend_module())
    api.register_target_normalizer(RKNPU_TARGET_NAME, _normalise_target,
                                  override=True)
    install_inspect_source()
    return backend


def is_registered() -> bool:
    """Whether this process registered this backend and this target name."""
    if not available():
        return False
    api = backend_api()
    backend = backend_module()
    if api.list_backends().get(RKNPU_BACKEND_NAME) is not backend:
        return False
    from tilelang.backend import target as target_api

    spec = getattr(target_api, "_TARGET_NORMALIZERS", {}).get(RKNPU_TARGET_NAME)
    return spec is not None and spec.normalize is _normalise_target


def unregister_rknpu_backend() -> bool:
    """Remove the registration again; returns whether anything was removed.

    TileLang exposes register_backend but no public unregistration
    (backend/module.py:238 has no counterpart), so this restores the two private
    registry dicts it registered into, and only for its own declaration.
    """
    if not available():
        return False
    from tilelang.backend import module as module_api
    from tilelang.backend import target as target_api

    backend = backend_module()
    removed = False
    backends = getattr(module_api, "_BACKENDS", None)
    if isinstance(backends, dict) and backends.get(RKNPU_BACKEND_NAME) is backend:
        backends.pop(RKNPU_BACKEND_NAME)
        removed = True
    index = getattr(module_api, "_TARGET_KIND_INDEX", None)
    if isinstance(index, dict):
        owned = index.get(RKNPU_TARGET_KIND, [])
        if RKNPU_BACKEND_NAME in owned:
            owned.remove(RKNPU_BACKEND_NAME)
            removed = True
            if not owned:
                index.pop(RKNPU_TARGET_KIND, None)
    normalizers = getattr(target_api, "_TARGET_NORMALIZERS", None)
    if isinstance(normalizers, dict):
        spec = normalizers.get(RKNPU_TARGET_NAME)
        if spec is not None and spec.normalize is _normalise_target:
            normalizers.pop(RKNPU_TARGET_NAME)
            removed = True
    if remove_inspect_source():
        removed = True
    return removed


@contextlib.contextmanager
def registered() -> Iterator[Any]:
    """Register for the duration of the block and restore the prior state."""
    already = is_registered()
    backend = register_rknpu_backend()
    try:
        yield backend
    finally:
        if not already:
            unregister_rknpu_backend()
