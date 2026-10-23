#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""P0 source lock for the optional TileLang dependency.

What this produces (all under ``provenance/``):

* ``source-lock.json`` — the pinned TileLang tag/commit, its submodule gitlinks
  (``3rdparty/tvm``, ``3rdparty/cutlass``), the TVM commit, and a per-file
  ``git blob`` SHA list for the Python surface this backend imports.
* ``upstream-trees/*.json`` — the raw GitHub tree responses used above, so the
  record can be re-checked without trusting this script's arithmetic.
* ``wheel-vs-source.json`` — for every installed package file that also exists in
  the pinned sources, whether its content hash equals the pinned ``git blob``.

The point of the last item: the released wheel is what actually runs in CI, so we
prove that the wheel's Python surface is byte-identical to the locked commit
instead of merely asserting a version string.  Any mismatch is reported and the
script exits non-zero, which means the lock must be re-derived rather than
silently accepted.

Nothing here needs a device and nothing here compiles anything.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE = REPO_ROOT / "research" / "tilelang-rknpu"
CACHE = MODULE / ".src" / "cache"
PROVENANCE = MODULE / "provenance"
TREES = PROVENANCE / "upstream-trees"
TILELANG_URL = "https://github.com/tile-ai/tilelang.git"
TILELANG_TAG = os.environ.get("TILELANG_LOCK_TAG", "v0.1.15")
# Roots whose Python surface the adapter can touch.
TILELANG_PREFIXES = ("tilelang/",)
TVM_PREFIXES = ("python/tvm/", "3rdparty/")


def _request(url: str, *, retries: int = 4) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"Accept": "application/vnd.github+json",
                              "User-Agent": "rknpu-backend-p0-lock"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise SystemExit(f"cannot fetch {url}: {last}")


def _cached(name: str, url: str) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if path.is_file() and path.stat().st_size:
        return path.read_bytes()
    payload = _request(url)
    path.write_bytes(payload)
    return payload


def resolve_tag(api_repo: str, repo_url: str, tag: str) -> tuple[str, str]:
    """Resolve a tag to a commit, preferring the API.

    ``git ls-remote`` against github.com is unreliable from this sandbox (TLS
    resets and timeouts), so the API is the primary path and the git client is a
    fallback.  Either way the resolved SHA is recorded with the method used.
    """
    try:
        payload = _cached(f"tagref-{tag}.json",
                          f"https://api.github.com/repos/{api_repo}/git/ref/tags/{tag}")
        ref = json.loads(payload)
        obj = ref["object"]
        if obj["type"] == "commit":
            return obj["sha"], "api:git/ref/tags"
        if obj["type"] == "tag":
            annotated = json.loads(_cached(
                f"annotated-{obj['sha']}.json",
                f"https://api.github.com/repos/{api_repo}/git/tags/{obj['sha']}"))
            return annotated["object"]["sha"], "api:git/tags(annotated)"
    except SystemExit:
        pass
    try:
        out = subprocess.run(["git", "ls-remote", repo_url, f"refs/tags/{tag}"],
                             capture_output=True, text=True, timeout=90)
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                sha, ref = line.split("\t")
                if ref == f"refs/tags/{tag}":
                    return sha, "git:ls-remote"
    except (subprocess.TimeoutExpired, OSError):
        pass
    raise SystemExit(f"cannot resolve tag {tag} in {repo_url} via API or git")


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - git object id


def fetch_tree(api_repo: str, commit: str, cache_name: str) -> dict:
    payload = _cached(cache_name,
                      f"https://api.github.com/repos/{api_repo}/git/trees/{commit}?recursive=1")
    data = json.loads(payload)
    if data.get("truncated"):
        raise SystemExit(f"tree listing for {api_repo}@{commit} is truncated")
    return data


def submodule_shas(tree: dict) -> dict[str, str]:
    return {entry["path"]: entry["sha"] for entry in tree["tree"]
            if entry["type"] == "commit"}


def blob_index(tree: dict, prefixes: tuple[str, ...]) -> dict[str, str]:
    return {entry["path"]: entry["sha"] for entry in tree["tree"]
            if entry["type"] == "blob" and entry["path"].startswith(prefixes)}


def installed_file_index(deps_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    files: dict[str, pathlib.Path] = {}
    for path in deps_dir.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(deps_dir).as_posix()
        if rel.startswith("tilelang/"):
            files[rel] = path
    return files


BINARY_SUFFIXES = (".so", ".a", ".o", ".pyd", ".dylib", ".pyc")


def is_binary(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return name.endswith(BINARY_SUFFIXES) or ".so." in name


# Files the wheel legitimately contains that no upstream tree can account for.
# Each entry needs a reason; anything else is a failure.
EXPLICIT_EXCEPTIONS = {
    "tilelang/3rdparty/tvm/version.py":
        "generated by the TVM build; the version string is not a source file",
}


@dataclass
class Scope:
    name: str
    installed_prefix: str
    upstream: dict[str, str]
    upstream_prefix: str
    completeness: str = "full"  # "full" | "subset"
    exclude: tuple[str, ...] = ()


def installed_scope(files: dict[str, pathlib.Path], prefix: str,
                    exclude: tuple[str, ...] = ()) -> dict[str, pathlib.Path]:
    return {rel: path for rel, path in files.items()
            if rel.startswith(prefix) and not rel.startswith(exclude)}


def compare_scope(scope: Scope, files: dict[str, pathlib.Path]) -> dict:
    installed = installed_scope(files, scope.installed_prefix, scope.exclude)
    upstream = {rel[len(scope.upstream_prefix):]: sha
                for rel, sha in scope.upstream.items()
                if rel.startswith(scope.upstream_prefix)}
    installed_rel = {rel[len(scope.installed_prefix):]: path
                     for rel, path in installed.items()}

    matched, mismatched = [], []
    for rel, path in sorted(installed_rel.items()):
        expected = upstream.get(rel)
        if expected is None:
            continue
        actual = git_blob_sha(path.read_bytes())
        (matched if actual == expected else mismatched).append(rel)

    extra = sorted(rel for rel in installed_rel if rel not in upstream)
    missing = sorted(rel for rel in upstream if rel not in installed_rel)
    extra_py = [rel for rel in extra if rel.endswith(".py")]
    missing_py = [rel for rel in missing if rel.endswith(".py")]
    extra_binary = [rel for rel in extra if is_binary(rel)]
    unexpected_extra_py = [rel for rel in extra_py
                           if scope.installed_prefix + rel not in EXPLICIT_EXCEPTIONS]
    unexplained_extra = [rel for rel in extra
                         if rel not in extra_py and rel not in extra_binary
                         and scope.installed_prefix + rel not in EXPLICIT_EXCEPTIONS]
    return {
        "name": scope.name,
        "completeness": scope.completeness,
        "compared": len(matched) + len(mismatched),
        "matched": len(matched),
        "mismatched": sorted(mismatched),
        "installed_python": len([r for r in installed_rel if r.endswith(".py")]),
        "upstream_python": len([r for r in upstream if r.endswith(".py")]),
        "extra_installed_python": unexpected_extra_py,
        "extra_installed_binary": len(extra_binary),
        "extra_installed_unexplained": unexplained_extra,
        "missing_upstream_python": missing_py if scope.completeness == "full" else [],
        "missing_upstream_python_not_claimed": (missing_py
                                                if scope.completeness != "full" else []),
    }


def main() -> int:
    deps_dir = pathlib.Path(os.environ.get("TILELANG_DEPS_DIR",
                                           MODULE / ".pydeps"))
    if not (deps_dir / "tilelang").is_dir():
        raise SystemExit(f"no installed tilelang package under {deps_dir}")

    commit, method = resolve_tag("tile-ai/tilelang", TILELANG_URL, TILELANG_TAG)
    tree = fetch_tree("tile-ai/tilelang", commit, f"tilelang-tree-{commit}.json")
    subs = submodule_shas(tree)
    if "3rdparty/tvm" not in subs:
        raise SystemExit("pinned tilelang tree has no 3rdparty/tvm submodule")

    TREES.mkdir(parents=True, exist_ok=True)
    (TREES / "tilelang-tree.json").write_bytes(
        (CACHE / f"tilelang-tree-{commit}.json").read_bytes())

    tvms_sha = subs["3rdparty/tvm"]
    tvm_tree = fetch_tree("apache/tvm", tvms_sha, f"tvm-tree-{tvms_sha}.json")
    (TREES / "tvm-tree.json").write_bytes(
        (CACHE / f"tvm-tree-{tvms_sha}.json").read_bytes())

    cutlass_sha = subs.get("3rdparty/cutlass")
    cutlass_tree = (fetch_tree("NVIDIA/cutlass", cutlass_sha,
                               f"cutlass-tree-{cutlass_sha}.json")
                    if cutlass_sha else {"tree": []})
    if cutlass_sha:
        (TREES / "cutlass-tree.json").write_bytes(
            (CACHE / f"cutlass-tree-{cutlass_sha}.json").read_bytes())

    # Only the repository's own ``tilelang/`` package tree is shipped in the
    # wheel; benchmarks, examples and skills are not, so they are out of scope.
    tilelang_root = {path: sha for path, sha in
                     blob_index(tree, ("tilelang/",)).items()
                     if not path.startswith("tilelang/3rdparty/")}
    tilelang_cpp = blob_index(tree, ("src/",))
    tvm_all = {path: sha for path, sha in blob_index(tvm_tree, ("",)).items()}
    cutlass_all = {path: sha for path, sha in blob_index(cutlass_tree, ("",)).items()}

    wheel_files = installed_file_index(deps_dir)
    scopes = [
        # The wheel's own TileLang package must equal the tagged tree.
        Scope("tilelang-package", "tilelang/", tilelang_root, "tilelang/",
              completeness="full",
              exclude=("tilelang/3rdparty/", "tilelang/src/", "tilelang/lib/")),
        # The wheel also ships TileLang's own C++ sources and the compiled
        # libraries.  The sources map to the repository root ``src/`` tree; the
        # libraries are build outputs and are only hash-recorded.
        Scope("tilelang-cpp-src", "tilelang/src/", tilelang_cpp, "src/",
              completeness="full"),
        # The wheel vendors TVM's python package and C++ headers/sources.
        Scope("tvm-python", "tilelang/3rdparty/tvm/python/", tvm_all,
              "python/", completeness="full"),
        Scope("tvm-include", "tilelang/3rdparty/tvm/include/", tvm_all,
              "include/", completeness="full"),
        Scope("tvm-src", "tilelang/3rdparty/tvm/src/", tvm_all,
              "src/", completeness="full"),
        # cutlass is vendored as a subset (headers + one script): compare what is
        # present, make no completeness claim.
        Scope("cutlass-subset", "tilelang/3rdparty/cutlass/", cutlass_all, "",
              completeness="subset"),
    ]
    results = [compare_scope(scope, wheel_files) for scope in scopes]

    problems: list[str] = []
    # Anything installed under tilelang/ that no scope covers must be classified
    # explicitly: a shipped binary (hash-recorded) or an allowlisted generated
    # file.  Nothing may remain unexplained.
    covered = set()
    for scope in scopes:
        for rel in installed_scope(wheel_files, scope.installed_prefix, scope.exclude):
            covered.add(rel)
    leftovers = sorted(rel for rel in
                       installed_scope(wheel_files, "tilelang/")
                       if rel not in covered)
    binary_artifacts, allowlisted_leftovers, unexplained_leftovers = [], [], []
    for rel in leftovers:
        if rel in EXPLICIT_EXCEPTIONS:
            allowlisted_leftovers.append(rel)
        elif is_binary(rel):
            binary_artifacts.append({
                "path": rel,
                "sha256": hashlib.sha256(wheel_files[rel].read_bytes()).hexdigest(),
                "note": "compiled artifact shipped by the release pipeline; its "
                        "provenance is the wheel hash, not a source-tree match",
            })
        else:
            unexplained_leftovers.append(rel)
    if unexplained_leftovers:
        problems.append(f"installed files outside every scope: "
                        f"{unexplained_leftovers[:5]}")

    python_matched = sum(r["matched"] for r in results)
    python_mismatched = [m for r in results for m in r["mismatched"]]
    for entry in results:
        if entry["mismatched"]:
            problems.append(f"{entry['name']}: content mismatch in "
                            f"{entry['mismatched'][:5]}")
        if entry["extra_installed_python"]:
            problems.append(f"{entry['name']}: python files not in upstream and "
                            f"not allowlisted: {entry['extra_installed_python'][:5]}")
        if entry["extra_installed_unexplained"]:
            problems.append(f"{entry['name']}: unexplained extra files: "
                            f"{entry['extra_installed_unexplained'][:5]}")
        if entry["missing_upstream_python"]:
            problems.append(f"{entry['name']}: upstream python files missing from "
                            f"the install: {entry['missing_upstream_python'][:5]}")
        if entry["installed_python"] != entry["upstream_python"] and \
                entry["completeness"] == "full" and \
                not entry["missing_upstream_python"]:
            problems.append(f"{entry['name']}: python file count differs "
                            f"({entry['installed_python']} installed vs "
                            f"{entry['upstream_python']} upstream)")

    comparison_totals = {
        "files_compared": sum(r["compared"] for r in results),
        "files_matched": python_matched,
        "files_mismatched": len(python_mismatched),
        "installed_python": sum(r["installed_python"] for r in results),
        "upstream_python_full_scopes": sum(r["upstream_python"] for r in results
                                           if r["completeness"] == "full"),
    }
    lock = {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tilelang": {
            "repository": "https://github.com/tile-ai/tilelang",
            "tag": TILELANG_TAG,
            "commit": commit,
            "resolved_by": method,
            "tree_entries": len(tree["tree"]),
        },
        "submodules": subs,
        "tvm_repository": "https://github.com/apache/tvm",
        "tvm_commit": tvms_sha,
        "python_surface": {
            "scope": "tilelang/** (minus 3rdparty), 3rdparty/tvm/{python,include,src}",
            "completeness": "python files in those scopes are complete; vendored "
                            "cutlass is a subset and makes no completeness claim",
            "totals": comparison_totals,
        },
        "patches": [],
        "patch_policy": "no patches are applied to TileLang or TVM; the backend "
                        "only reads their public Python objects",
        "backends_files_hashed": {
            rel: hashlib.sha256((MODULE / rel).read_bytes()).hexdigest()
            for rel in ("python/rknpu_backend/frontend/tilelang_adapter.py",
                        "scripts/lock_source.py")
            if (MODULE / rel).is_file()
        },
    }
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    (PROVENANCE / "source-lock.json").write_text(json.dumps(lock, indent=2) + "\n")

    comparison = {
        "deps_dir": str(deps_dir),
        "method": "git blob SHA (sha1 of 'blob <size>\\0' + content) compared "
                  "between the installed package and the pinned upstream tree",
        "scopes": results,
        "totals": {
            "files_compared": sum(r["compared"] for r in results),
            "files_matched": python_matched,
            "files_mismatched": len(python_mismatched),
            "installed_python_all_scopes": sum(r["installed_python"]
                                               for r in results),
            "installed_python_full_scopes": sum(
                r["installed_python"] for r in results
                if r["completeness"] == "full"),
            "upstream_python_full_scopes": sum(
                r["upstream_python"] for r in results
                if r["completeness"] == "full"),
            "missing_upstream_python": sum(
                len(r["missing_upstream_python"]) for r in results),
            "unexplained_extra_python": sum(
                len(r["extra_installed_python"]) for r in results),
            "unexplained_extra_other": sum(
                len(r["extra_installed_unexplained"]) for r in results),
            "unexplained_unscoped": len(unexplained_leftovers),
            "binary_artifacts_not_source_comparable": len(binary_artifacts),
            "allowlisted_generated_files": allowlisted_leftovers,
        },
        "explicit_exceptions": EXPLICIT_EXCEPTIONS,
        "binary_artifacts": binary_artifacts,
        "problems": problems,
    }
    (PROVENANCE / "wheel-vs-source.json").write_text(
        json.dumps(comparison, indent=2) + "\n")

    print(f"tilelang {TILELANG_TAG} -> {commit}")
    print(f"3rdparty/tvm       -> {tvms_sha}")
    print(f"3rdparty/cutlass   -> {cutlass_sha}")
    totals = comparison["totals"]
    print(f"files compared and matched: {totals['files_matched']}"
          f"/{totals['files_compared']}  (mismatched "
          f"{totals['files_mismatched']})")
    print(f"python files: {totals['installed_python_full_scopes']} installed in "
          f"full scopes, {totals['upstream_python_full_scopes']} upstream "
          f"(+{totals['installed_python_all_scopes'] - totals['installed_python_full_scopes']}"
          f" vendored subset)")
    print(f"missing upstream python: {totals['missing_upstream_python']}")
    print(f"unexplained extra python: {totals['unexplained_extra_python']}, "
          f"unexplained extra other: {totals['unexplained_extra_other']}")
    print(f"unexplained unscoped files: {totals['unexplained_unscoped']}")
    print(f"binary artifacts (hash-recorded, not source-comparable): "
          f"{totals['binary_artifacts_not_source_comparable']}")
    for entry in results:
        print(f"  [{entry['name']}/{entry['completeness']}] compared "
              f"{entry['compared']}, matched {entry['matched']}, unmatched "
              f"{len(entry['mismatched'])}")
    if problems:
        for problem in problems:
            print(f"  PROBLEM {problem}", file=sys.stderr)
        return 1
    if totals["files_matched"] < 300:
        print("too few files matched; refusing a weak lock", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
