# SPDX-License-Identifier: GPL-3.0-or-later
"""Strict JSON parsing and canonical serialization.

Python owns JSON.  The C runtime never parses it, never opens a file, and never
trusts a Python-computed ``verified`` flag.  Duplicate keys, NaN/Infinity, string
numbers and bool-in-place-of-int are rejected before a dataclass is constructed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from typing import Any

from .errors import RK_E_SCHEMA, reject


def _reject_constant(token: str) -> None:
    reject(RK_E_SCHEMA, f"non-finite JSON number {token!r} is not allowed",
           stage="schema")


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            reject(RK_E_SCHEMA, f"duplicate JSON key {key!r}", stage="schema")
        out[key] = value
    return out


def loads(text: str | bytes) -> Any:
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as exc:
            reject(RK_E_SCHEMA, f"JSON is not valid UTF-8: {exc}", stage="schema")
    try:
        obj = json.loads(text, object_pairs_hook=_object_pairs,
                         parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        reject(RK_E_SCHEMA, f"invalid JSON at {exc.lineno}:{exc.colno}: {exc.msg}",
               stage="schema")
    _validate_json_values(obj)
    return obj


def _validate_json_values(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                reject(RK_E_SCHEMA, f"non-string key at {path}", stage="schema")
            _validate_json_values(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_values(child, f"{path}[{index}]")
    elif isinstance(value, float):
        if not math.isfinite(value):
            reject(RK_E_SCHEMA, f"non-finite float at {path}", stage="schema")
    elif value is None or isinstance(value, (str, bool, int)):
        return
    else:
        reject(RK_E_SCHEMA, f"unsupported JSON value at {path}", stage="schema")


def canonical_json(obj: Any) -> str:
    _validate_json_values(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def canonical_bytes(obj: Any) -> bytes:
    return canonical_json(obj).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_bytes(obj))



def load_json_safe(path: str | os.PathLike[str]) -> Any:
    """Load one JSON document with duplicate-key and non-finite checks."""
    with open(path, "r", encoding="utf-8") as fh:
        return loads(fh.read())


def load_module(path: str):
    from .ir import Module

    with open(path, "r", encoding="utf-8") as fh:
        return Module.from_dict(loads(fh.read()))


def save_module(module: Any, path: str) -> None:
    _atomic_write(path, canonical_json(module.to_dict()).encode("utf-8"))


def load_plan(path: str):
    from .plan import Plan

    with open(path, "r", encoding="utf-8") as fh:
        return Plan.from_dict(loads(fh.read()))


def save_plan(plan: Any, path: str) -> None:
    _atomic_write(path, canonical_json(plan.to_dict()).encode("utf-8"))


def _atomic_write(path: str, data: bytes) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".rkplan-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
