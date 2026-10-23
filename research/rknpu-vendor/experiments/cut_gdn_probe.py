#!/usr/bin/env python3
"""Cut the GDN graph at a named tensor, convert, run on the board and compare.

Exposing many tensors as extra graph outputs made rknn mis-execute the model,
so intermediate checks are done by *pruning* the graph at one tensor and
building a single-output model instead.

  python3 cut_gdn_probe.py --onnx /tmp/gdn_cycles/real_opt0/gdn_layer.onnx \
      --inputs /tmp/qwen35_gdn_inputs.npz --out-dir /tmp/gdn_cuts \
      --cuts core_trim gated_silu out_proj [--fp16] [--optimization-level 0]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx

HERE = Path(__file__).resolve().parent
BOARD = "/mnt/ssd/workspace/gdn-probe"
PYLIBS = "/tmp/pylibs"


def run(cmd, env=None):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)


def board(cmd: str) -> str:
    return run(["python3", "/tmp/board.py", cmd], env={"PYTHONPATH": PYLIBS, "PATH": "/usr/bin:/bin"}).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cuts", nargs="+", required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--optimization-level", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = {"PYTHONPATH": "/tmp/gdnpy", "PATH": "/usr/bin:/bin"}

    import onnxruntime as ort

    full = onnx.load(str(args.onnx))
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    x = np.load(args.inputs)["x"][0:1].astype(np.float32)
    known = {o.name for o in sess.get_outputs()}
    want = [c for c in args.cuts if c not in known]
    names = [o.name for o in sess.get_outputs()] + want
    # add the cut tensors as extra outputs so ORT produces the reference
    for name in want:
        full.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    tmp = args.out_dir / "with_ref.onnx"
    onnx.save(full, str(tmp))
    ref_sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    ref = dict(zip([o.name for o in ref_sess.get_outputs()], ref_sess.run(None, {"x": x})))

    report = {}
    for cut in args.cuts:
        cut_dir = args.out_dir / cut
        cut_dir.mkdir(exist_ok=True)
        model_path = cut_dir / f"{cut}.onnx"
        onnx.utils.extract_model(str(args.onnx), str(model_path), ["x"], [cut])
        out_rknn = cut_dir / f"{cut}.rknn"
        run([
            sys.executable, str(HERE / "convert_gdn_layer_rknn.py"),
            "--onnx", str(model_path), "--inputs", str(args.inputs),
            "--out", str(out_rknn), "--calib-samples", "4",
            "--optimization-level", str(args.optimization_level),
        ] + ([] if args.fp16 else ["--fp16"]), env=env)
        x[0].astype(np.float32).tofile(cut_dir / "in0.bin")
        board(f"mkdir -p {BOARD}")
        board_put = run(["python3", "/tmp/board.py", "--put", str(out_rknn), f"{BOARD}/{cut}.rknn"],
                        env={"PYTHONPATH": PYLIBS, "PATH": "/usr/bin:/bin"})
        run(["python3", "/tmp/board.py", "--put", str(cut_dir / "in0.bin"), f"{BOARD}/{cut}_in0.bin"],
            env={"PYTHONPATH": PYLIBS, "PATH": "/usr/bin:/bin"})
        board(
            f"cd {BOARD} && export LD_LIBRARY_PATH=/mnt/ssd/workspace/runtimes/mars-overhead/rknn/lib:$LD_LIBRARY_PATH && "
            f"./gdn_feed {cut}.rknn {cut} {cut}_in0.bin"
        )
        run(["python3", "/tmp/board.py", "--get", f"{BOARD}/{cut}_out00.bin", str(cut_dir / "board.bin")],
            env={"PYTHONPATH": PYLIBS, "PATH": "/usr/bin:/bin"})
        got = np.fromfile(cut_dir / "board.bin", dtype=np.float32)
        refv = ref[cut].ravel()
        if got.size != refv.size:
            report[cut] = {"error": f"size {got.size} vs {refv.size}"}
            continue
        a = refv.astype(np.float64)
        b = got.astype(np.float64)
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        aa, bb = a - a.mean(), b - b.mean()
        corr = float((aa * bb).sum() / np.sqrt((aa * aa).sum() * (bb * bb).sum()))
        ratio = float(np.median(b[np.abs(a) > 1e-6] / a[np.abs(a) > 1e-6])) if np.any(np.abs(a) > 1e-6) else float("nan")
        report[cut] = {
            "corr": corr,
            "ratio": ratio,
            "ref_rms": float(np.sqrt((a**2).mean())),
            "got_rms": float(np.sqrt((b**2).mean())),
        }

    (args.out_dir / "cut_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
