#!/usr/bin/env python3
"""Build -> convert -> board run -> numeric compare, for one GDN config.

Host side: build_gdn_layer_onnx.py + convert_gdn_layer_rknn.py.
Board side: upload the .rknn and the input, run gdn_feed, fetch the outputs,
then correlate against the numpy oracle.

  python3 gdn_cycle.py --tag small --hidden 512 --num-k-heads 2 --num-v-heads 4 \
      --head-k-dim 64 --head-v-dim 64 --seq 196
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE
BOARD = "/mnt/ssd/workspace/gdn-probe"
BOARD_PY = "/tmp/board.py"
PYLIBS = "/tmp/pylibs"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def board(cmd: str) -> str:
    env = {"PYTHONPATH": PYLIBS, "PATH": "/usr/bin:/bin"}
    return run(["python3", BOARD_PY, cmd], env=env).stdout


def board_put(local: Path, remote: str) -> None:
    env = {"PYTHONPATH": PYLIBS, "PATH": "/usr/bin:/bin"}
    run(["python3", BOARD_PY, "--put", str(local), remote], env=env)


def board_get(remote: str, local: Path) -> None:
    env = {"PYTHONPATH": PYLIBS, "PATH": "/usr/bin:/bin"}
    run(["python3", BOARD_PY, "--get", remote, str(local)], env=env)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--num-k-heads", type=int, default=2)
    ap.add_argument("--num-v-heads", type=int, default=4)
    ap.add_argument("--head-k-dim", type=int, default=64)
    ap.add_argument("--head-v-dim", type=int, default=64)
    ap.add_argument("--seq", type=int, default=196)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--value-scale", type=float, default=1.0)
    ap.add_argument("--input-npz", type=Path, default=None,
                    help="use real activations (npz with x [N,1,L,H]) for calibration and validation")
    args = ap.parse_args()

    work = Path("/tmp/gdn_cycles") / args.tag
    work.mkdir(parents=True, exist_ok=True)
    env = {"PYTHONPATH": "/tmp/gdnpy", "PATH": "/usr/bin:/bin"}

    build_cmd = [
        sys.executable, str(HERE / "build_gdn_layer_onnx.py"),
        "--out-dir", str(work),
        "--hidden", str(args.hidden),
        "--num-k-heads", str(args.num_k_heads),
        "--num-v-heads", str(args.num_v_heads),
        "--head-k-dim", str(args.head_k_dim),
        "--head-v-dim", str(args.head_v_dim),
        "--seq", str(args.seq),
        "--value-scale", str(args.value_scale),
    ]
    if args.weights:
        build_cmd += ["--weights", str(args.weights)]
    run(build_cmd, env=env)

    # a calibration set large enough for the mixed-input models
    ref = dict(np.load(work / "gdn_reference.npz"))
    if args.input_npz:
        real = np.load(args.input_npz)["x"].astype(np.float32)
        sys.path.insert(0, str(HERE))
        from build_gdn_layer_onnx import GDNConfig, reference_layer

        cfg = GDNConfig(
            hidden=args.hidden, num_k_heads=args.num_k_heads, num_v_heads=args.num_v_heads,
            head_k_dim=args.head_k_dim, head_v_dim=args.head_v_dim, seq=args.seq,
        )
        weights = np.load(args.weights) if args.weights else None
        if weights is None:
            raise SystemExit("--input-npz needs --weights so the oracle can be recomputed")
        sys.path.insert(0, str(HERE))
        y_ref, st_ref = reference_layer(cfg, {k: weights[k] for k in weights.files}, real[0:1])
        np.savez(work / "gdn_reference.npz", x=real[0:1], y=y_ref, final_state=st_ref)
        ref = dict(np.load(work / "gdn_reference.npz"))
    x = ref["x"]
    calib = work / "inputs.npz"
    np.savez(calib, x=np.repeat(x, 4, axis=0)[:4])

    model = work / f"gdn_{args.tag}.rknn"
    conv_cmd = [
        sys.executable, str(HERE / "convert_gdn_layer_rknn.py"),
        "--onnx", str(work / "gdn_layer.onnx"),
        "--inputs", str(calib),
        "--out", str(model),
        "--calib-samples", str(min(4, x.shape[0])),
        "--optimization-level", str(args.opt_level),
    ]
    if not args.int8:
        conv_cmd.append("--fp16")
    run(conv_cmd, env=env)

    x[0].astype(np.float32).tofile(work / "in0.bin")

    board(f"mkdir -p {BOARD}")
    board_put(model, f"{BOARD}/{args.tag}.rknn")
    board_put(work / "in0.bin", f"{BOARD}/{args.tag}_in0.bin")
    out = board(
        f"cd {BOARD} && export LD_LIBRARY_PATH=/mnt/ssd/workspace/runtimes/mars-overhead/rknn/lib:$LD_LIBRARY_PATH && "
        f"./gdn_feed {args.tag}.rknn {args.tag} {args.tag}_in0.bin"
    )
    print(out)
    board_get(f"{BOARD}/{args.tag}_out00.bin", work / "board_y.bin")

    y = np.fromfile(work / "board_y.bin", dtype=np.float32)
    ref_y = ref["y"].astype(np.float32)
    if y.size != ref_y.size:
        print(f"SIZE MISMATCH {y.size} vs {ref_y.size}")
        return 1
    y = y.reshape(ref_y.shape)

    def corr(a, b):
        a = a.ravel().astype(np.float64)
        b = b.ravel().astype(np.float64)
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        if a.size < 2:
            return float("nan")
        a = a - a.mean()
        b = b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d else float("nan")

    report = {
        "tag": args.tag,
        "quantized": args.int8,
        "optimization_level": args.opt_level,
        "corr": corr(ref_y, y),
        "max_abs": float(np.nanmax(np.abs(ref_y - y))),
        "ref_rms": float(np.sqrt((ref_y**2).mean())),
        "got_rms": float(np.sqrt(np.nanmean(y**2))),
        "finite": float(np.isfinite(y).mean()),
    }
    (work / "cycle_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
