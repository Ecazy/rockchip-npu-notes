#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Assert that the test interpreter is isolated to the locked dependency tree.

`pip --target` creates a fresh directory but does not stop the interpreter from
also seeing the OS ``dist-packages``; a declared dependency such as ``jsonschema``
could therefore be satisfied by the system copy without anyone noticing.  This
check fails the run unless:

* no path outside the agreed environment contributes packages (specifically no
  ``/usr/lib/python3/dist-packages`` and no foreign ``site-packages``);
* every imported third-party module resolves inside ``TILELANG_DEPS_DIR``;
* every installed distribution reports a location inside that directory;
* the distribution list matches the locked name/version set exactly.

It prints a machine-readable summary and exits non-zero on any violation.
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import os
import pathlib
import sys

MODULES = ("numpy", "jsonschema", "pytest", "tilelang", "tvm", "tvm_ffi", "z3",
           "torch", "attrs", "referencing", "rpds")
FORBIDDEN_FRAGMENTS = ("/usr/lib/python3/dist-packages", "/usr/lib/python3.12/dist-packages",
                       "/usr/local/lib/python", "/home/maoyu/.local/lib")


def fail(message: str, problems: list[str]) -> None:
    problems.append(message)


def main() -> int:
    deps_env = os.environ.get("TILELANG_DEPS_DIR")
    lock_env = os.environ.get("TILELANG_LOCK_FILE")
    if not deps_env or not lock_env:
        raise SystemExit(
            "TILELANG_DEPS_DIR and TILELANG_LOCK_FILE must be set "
            "(scripts/run-verification.sh exports both; a direct call must too)")
    deps = pathlib.Path(deps_env).resolve()
    lock_path = pathlib.Path(lock_env)
    problems: list[str] = []

    locked: dict[str, str] = {}
    for line in lock_path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        spec = line.split()[0]
        name, version = spec.split("==", 1)
        locked[name.lower().replace("_", "-")] = version
    if not locked:
        return fail("empty dependency lock", problems) or 1

    # 1. sys.path must not expose foreign package roots.
    foreign = [p for p in sys.path
               if p and not p.startswith(str(deps))
               and any(frag in p for frag in FORBIDDEN_FRAGMENTS)]
    if foreign:
        fail(f"foreign package paths on sys.path: {foreign}", problems)

    # 2. Every imported third-party module must come from the locked tree.
    module_origins: dict[str, str] = {}
    for name in MODULES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - isolation report
            fail(f"{name} is not importable in the isolated interpreter: {exc}",
                 problems)
            continue
        origin = getattr(module, "__file__", None) or "<builtin>"
        module_origins[name] = origin
        if origin != "<builtin>" and not pathlib.Path(origin).resolve().is_relative_to(deps):
            fail(f"{name} resolves outside the locked tree: {origin}", problems)

    # 3. Installed distributions must live in the locked tree and match the lock.
    installed: dict[str, str] = {}
    for dist in md.distributions(path=[str(deps)]):
        name = (dist.metadata["Name"] or "").lower().replace("_", "-")
        installed[name] = dist.version
        location = str(pathlib.Path(str(dist.locate_file(""))).resolve())
        if not pathlib.Path(location).is_relative_to(deps):
            fail(f"distribution {name} is installed outside the locked tree: {location}",
                 problems)
    missing = sorted(set(locked) - set(installed))
    extra = sorted(set(installed) - set(locked))
    version_drift = {name: (locked[name], installed[name]) for name in
                     set(locked) & set(installed) if locked[name] != installed[name]}
    if missing:
        fail(f"locked distributions missing from the environment: {missing}", problems)
    if extra:
        fail(f"distributions installed but not locked: {extra}", problems)
    if version_drift:
        fail(f"version drift against the lock: {version_drift}", problems)

    # 4. The OS-level site directories must not be reachable at all.
    try:
        import site
        os_paths = [p for p in (getattr(site, "getsitepackages", lambda: [])())
                    if p and any(frag in p for frag in FORBIDDEN_FRAGMENTS)]
        if os_paths:
            fail(f"interpreter still reports OS site directories: {os_paths}", problems)
    except Exception:  # noqa: BLE001 - best effort probe
        pass

    summary = {
        "deps_dir": str(deps),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "sys_prefix": sys.prefix,
        "no_site_flag": bool(sys.flags.no_site),
        "locked_distributions": len(locked),
        "installed_distributions": len(installed),
        "module_origins": module_origins,
        "foreign_paths": foreign,
        "problems": problems,
    }
    print(json.dumps(summary, indent=2))
    if problems:
        raise SystemExit(f"isolation check failed: {problems}")
    print("ISOLATION OK: every third-party module comes from the locked tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
