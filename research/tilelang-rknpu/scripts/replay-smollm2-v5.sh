#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Executable SmolLM2 V5 replay entry.  See `--help`.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
export PYTHONPATH="$HERE:$HERE/python:$HERE/.pydeps-p0:$HERE/.src/p0-build-src/tilelang:$HERE/.src/p0-source-env${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$HERE/.src/p0-build-tilelang/lib:$HERE/.src/p0-build-tvmffi-python/lib:$HERE/.pydeps-p0/z3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TILELANG_CACHE_DIR="$HERE/.src/p0-tilelang-cache"
export RKNPU_CHIP="${RKNPU_CHIP:-rk3588}"
export RKNPU_PLAN_LIB="${RKNPU_PLAN_LIB:-$HERE/../custom-ops/build-host-mvp/librknpu_plan.so}"
export RKNPU_PLAN_MOCK_LIB="${RKNPU_PLAN_MOCK_LIB:-$HERE/../custom-ops/build-host-mvp/librknpu_plan_mock.so}"
exec python3 "$HERE/tools/replay_smollm2_v5.py" "$@"
