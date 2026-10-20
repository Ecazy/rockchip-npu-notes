# SPDX-License-Identifier: GPL-3.0-or-later
"""Command line entry point.  ``validate`` and ``explain`` never import kernels."""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .api import compile_rkir, compile_tilelang, explain_plan, load_plan, save_plan
from .capabilities import load_profile
from .errors import (BackendError, RK_E_BUSY, RK_E_DEVICE, RK_E_NOMEM,
                     RK_E_POISONED, RK_E_UNSUPPORTED, code_name)
from .ir import Module
from .options import CompileOptions
from .serialize import canonical_json, load_json_safe, loads


def _profile(name: str):
    if os.path.exists(name):
        return load_profile(name)
    from .capabilities import DEFAULT_PROFILE_PATH
    profile = load_profile(DEFAULT_PROFILE_PATH)
    if name not in {"rk3588_vendor_co_v1", profile.chip_id}:
        raise BackendError(RK_E_UNSUPPORTED,
                           f"unknown profile {name!r}", stage="cli")
    return profile


def _options(args: argparse.Namespace) -> CompileOptions:
    return CompileOptions(
        numeric_mode=args.numeric_mode,
        allow_approx_lut=bool(args.allow_approx_lut),
        fallback=args.fallback,
        submit_mode=args.submit_mode,
        tile_limit=args.tile_limit,
        device_budget_bytes=args.device_budget_bytes,
        host_budget_bytes=args.host_budget_bytes,
    )


def _load_module_from_file(path: str, symbol: str) -> Any:
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise BackendError(RK_E_UNSUPPORTED, f"kernel file not found: {path}",
                           stage="frontend")
    module_name = f"_rknpu_kernel_{file_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise BackendError(RK_E_UNSUPPORTED,
                           f"cannot import kernel file {path}", stage="frontend")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise BackendError(RK_E_UNSUPPORTED,
                           f"error importing kernel file: {exc}",
                           stage="frontend") from exc
    if not hasattr(module, symbol):
        raise BackendError(RK_E_UNSUPPORTED,
                           f"kernel symbol {symbol!r} not found", stage="frontend")
    return getattr(module, symbol)


def _write_plan(result: Any, out: str) -> None:
    save_plan(result.plan, out)


def _cmd_compile_tilelang(args: argparse.Namespace) -> int:
    prim_func = _load_module_from_file(args.file, args.symbol)
    result = compile_tilelang(prim_func, profile=_profile(args.profile),
                              options=_options(args))
    _write_plan(result, args.out)
    return 0


def _cmd_compile_rkir(args: argparse.Namespace) -> int:
    module = Module.from_dict(load_json_safe(args.path))
    result = compile_rkir(module, profile=_profile(args.profile),
                          options=_options(args))
    _write_plan(result, args.out)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    plan = load_plan(args.path)
    # Plan.__post_init__ validates the Python model and plan hash.  Ask the C
    # verifier to independently validate the descriptor when the shared library is
    # available; no device is opened by this path.
    try:
        from .runtime.ffi import default_library
        lib = default_library()
        cplan = lib.create_plan(plan)
        cplan.destroy()
    except BackendError:
        raise
    print(f"OK {plan.plan_hash}")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    plan = load_plan(args.path)
    report = explain_plan(plan)
    if args.format == "json":
        sys.stdout.write(canonical_json(report) + "\n")
    else:
        sys.stdout.write(_explain_text(report))
    return 0


def _explain_text(report: dict[str, Any]) -> str:
    counts = report["counts"]
    accounting = report["accounting"]
    lines = [
        f"plan {report['name']} {report['plan_hash']}",
        f"tasks={counts['tasks']} submits={counts['submits']} "
        f"data_bos={counts['data_bos']}",
        f"device_request_bytes={accounting['device_request_bytes']} "
        f"host_bytes={accounting['host_bytes']}",
    ]
    for submit in report["submits"]:
        lines.append(f"submit {submit['id']}: mode={submit['mode']} "
                     f"tasks={submit['task_ids']}")
    return "\n".join(lines) + "\n"


def _load_bindings(path: str) -> dict[str, np.ndarray]:
    obj = load_json_safe(path)
    if not isinstance(obj, dict):
        raise BackendError(RK_E_UNSUPPORTED, "bindings JSON must be an object",
                           stage="cli")
    result: dict[str, np.ndarray] = {}
    for name, value in obj.items():
        if isinstance(value, str):
            npy_path = value
        elif isinstance(value, dict) and "npy" in value:
            npy_path = value["npy"]
        else:
            raise BackendError(RK_E_UNSUPPORTED,
                               f"binding {name!r} must map to an .npy path",
                               stage="cli")
        if not os.path.isabs(npy_path):
            npy_path = os.path.join(os.path.dirname(os.path.abspath(path)), npy_path)
        try:
            array = np.load(npy_path, allow_pickle=False)
        except Exception as exc:
            raise BackendError(RK_E_UNSUPPORTED,
                               f"cannot load {npy_path}: {exc}", stage="cli") from exc
        if not isinstance(array, np.ndarray):
            raise BackendError(RK_E_UNSUPPORTED,
                               f"{npy_path} did not contain one ndarray", stage="cli")
        result[name] = array
    return result


def _atomic_save_npy(path: str, array: np.ndarray) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".rkout-", suffix=".npy", dir=directory)
    os.close(fd)
    try:
        # np.save appends .npy only when the name has no suffix.
        np.save(tmp, array, allow_pickle=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _cmd_run(args: argparse.Namespace) -> int:
    plan = load_plan(args.path)
    arrays = _load_bindings(args.bindings)
    inputs = {b.name: arrays[b.name] for b in plan.bindings
              if b.role == "input" and b.name in arrays}
    constants = {b.name: arrays[b.name] for b in plan.bindings
                 if b.role == "constant" and b.name in arrays}
    from .runtime.ffi import default_library
    lib = default_library()
    # rocket_open/rocket_close are exported by the same vendor-backed library.
    lib.cdll.rocket_open.argtypes = []
    lib.cdll.rocket_open.restype = ctypes.c_int
    lib.cdll.rocket_close.argtypes = [ctypes.c_int]
    lib.cdll.rocket_close.restype = ctypes.c_int
    fd = int(lib.cdll.rocket_open())
    if fd < 0:
        raise BackendError(RK_E_DEVICE, "rocket_open failed; no supported RKNPU fd",
                           stage="cli", native_status=fd)
    ws = None
    try:
        from .runtime.bindings import Workspace
        ws = Workspace.create(plan, fd=fd, constants=constants)
        result = ws.run(inputs)
        for name, array in result.items():
            _atomic_save_npy(os.path.join(args.output_dir, name + ".npy"), array)
        return 0
    finally:
        if ws is not None:
            try:
                ws.close()
            except BackendError:
                pass
        try:
            lib.cdll.rocket_close(fd)
        except Exception:
            pass


def _add_compile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default="rk3588_vendor_co_v1")
    parser.add_argument("--numeric-mode", choices=["strict", "rk_f16_tiled"],
                        default="strict")
    parser.add_argument("--allow-approx-lut", action="store_true")
    parser.add_argument("--fallback", choices=["error", "host_prepare"],
                        default="error")
    parser.add_argument("--submit-mode", choices=["chain", "separate"],
                        default="chain")
    parser.add_argument("--tile-limit", type=int, default=256)
    parser.add_argument("--device-budget-bytes", type=int,
                        default=256 * 1024 * 1024)
    parser.add_argument("--host-budget-bytes", type=int,
                        default=256 * 1024 * 1024)
    parser.add_argument("--out", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rknpu_backend.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("compile-tilelang")
    p.add_argument("--file", required=True)
    p.add_argument("--symbol", default="kernel")
    _add_compile_args(p)
    p.set_defaults(func=_cmd_compile_tilelang)
    p = sub.add_parser("compile-rkir")
    p.add_argument("path")
    _add_compile_args(p)
    p.set_defaults(func=_cmd_compile_rkir)
    p = sub.add_parser("validate")
    p.add_argument("path")
    p.set_defaults(func=_cmd_validate)
    p = sub.add_parser("explain")
    p.add_argument("path")
    p.add_argument("--format", choices=["json", "text"], default="text")
    p.set_defaults(func=_cmd_explain)
    p = sub.add_parser("run")
    p.add_argument("path")
    p.add_argument("--bindings", required=True)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except BackendError as exc:
        for diag in exc.diagnostics:
            sys.stderr.write(diag.format() + "\n")
        resource_codes = {RK_E_DEVICE, RK_E_NOMEM, RK_E_BUSY, RK_E_POISONED}
        return 3 if exc.code in resource_codes else 2
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
