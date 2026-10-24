# SPDX-License-Identifier: GPL-3.0-or-later
"""Child process: model-level failure must not publish a result."""
import gzip
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rknpu_backend import load_plan
from rknpu_backend.models import MLPRunner, load_weight_npz
from rknpu_backend.runtime.ffi import LibRKPlan


def read_idx(path, rank):
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">" + "I" * rank, f.read(4 * rank))
        return np.frombuffer(f.read(), np.uint8).reshape(dims)


def main() -> int:
    lib = LibRKPlan(os.environ["RKNPU_PLAN_MOCK_LIB"])
    plan = load_plan(str(ROOT / "models" / "compiled" /
                         "mnist_mlp_b1.rkplan.json"))
    weights = load_weight_npz(ROOT / "models" / "mnist_mlp.npz")
    x = read_idx(ROOT / ".data" / "mnist" / "t10k-images-idx3-ubyte.gz", 3)
    runner = MLPRunner(lib, plan, weights, 1, fd=0)
    before = int(lib.mock_control("rk_mock_execs"))
    lib.mock_control("rk_mock_fail_at", before + 1)
    out = {"run_code": "NO_ERROR", "close_code": "NO_ERROR",
           "execs_before": before}
    try:
        runner.run(x[:1])
        out["published"] = True
    except Exception as exc:
        code = getattr(exc, "diagnostics", None)
        out["run_code"] = code[0].code if code else type(exc).__name__
        out["published"] = False
    out["execs_after"] = int(lib.mock_control("rk_mock_execs"))
    try:
        runner.close()
    except Exception as exc:
        code = getattr(exc, "diagnostics", None)
        out["close_code"] = code[0].code if code else type(exc).__name__
    print(json.dumps(out, sort_keys=True))
    return 0 if (not out["published"] and out["run_code"] != "NO_ERROR") else 1


if __name__ == "__main__":
    raise SystemExit(main())
