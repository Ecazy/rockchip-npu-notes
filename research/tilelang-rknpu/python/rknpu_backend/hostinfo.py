# SPDX-License-Identifier: GPL-3.0-or-later
"""Small, dependency-free CPU/BLAS environment recorders for timing evidence."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np

_ENV_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
_BLAS_THREAD_SYMBOLS = (
    "openblas_get_num_threads64_",
    "openblas_get_num_threads",
    "scipy_openblas_get_num_threads64_",
    "scipy_openblas_get_num_threads",
    "MKL_Get_Max_Threads",
)


def _process_threads() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("Threads:"):
                return int(line.split(":", 1)[1].strip())
    except (OSError, ValueError):
        pass
    return None


def _loaded_blas_libraries() -> list[str]:
    found: list[str] = []
    try:
        lines = Path("/proc/self/maps").read_text().splitlines()
    except OSError:
        return found
    for line in lines:
        lower = line.lower()
        if not any(token in lower for token in ("openblas", "mkl_rt", "blis")):
            continue
        path = line.split()[-1]
        if path.startswith("/") and path not in found:
            found.append(path)
    return sorted(found)


def blas_threads() -> list[dict[str, Any]]:
    """Read current BLAS pool sizes from loaded libraries where symbols exist."""
    rows: list[dict[str, Any]] = []
    for path in _loaded_blas_libraries():
        row: dict[str, Any] = {"library": Path(path).name, "threads": None,
                               "symbol": None}
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            rows.append(row)
            continue
        for name in _BLAS_THREAD_SYMBOLS:
            fn = getattr(lib, name, None)
            if fn is None:
                continue
            try:
                fn.argtypes = []
                fn.restype = ctypes.c_int
                row["threads"] = int(fn())
                row["symbol"] = name
                break
            except Exception:
                continue
        rows.append(row)
    return rows


def cpu_frequency_mhz() -> dict[str, float | None]:
    values: list[float] = []
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("cpu MHz"):
                values.append(float(line.split(":", 1)[1]))
    except (OSError, ValueError):
        pass
    if not values:
        for p in sorted(Path("/sys/devices/system/cpu").glob(
                "cpu[0-9]*/cpufreq/scaling_cur_freq")):
            try:
                values.append(float(p.read_text().strip()) / 1000.0)
            except (OSError, ValueError):
                continue
    return {"min_mhz": min(values) if values else None,
            "max_mhz": max(values) if values else None,
            "mean_mhz": sum(values) / len(values) if values else None}


def numpy_blas_summary() -> dict[str, Any]:
    config = getattr(np.__config__, "CONFIG", {})
    blas = config.get("Build Dependencies", {}).get("blas", {}) \
        if isinstance(config, dict) else {}
    return {
        "numpy_version": np.__version__,
        "blas_name": blas.get("name"),
        "blas_version": blas.get("version"),
        "blas_openblas_configuration": blas.get("openblas configuration"),
    }


def thread_config() -> dict[str, Any]:
    """Record the actual CPU/BLAS view available to this process."""
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(int(c) for c in os.sched_getaffinity(0))
        except OSError:
            affinity = None
    return {
        "cpu_count": os.cpu_count(),
        "cpu_affinity_count": len(affinity) if affinity is not None else None,
        "cpu_affinity": affinity,
        "process_threads": _process_threads(),
        "cpu_thread_env": {key: os.environ.get(key) for key in _ENV_KEYS},
        "blas": blas_threads(),
        "numpy": numpy_blas_summary(),
        "cpuinfo": cpu_frequency_mhz(),
    }
