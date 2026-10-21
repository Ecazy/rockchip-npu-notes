#!/usr/bin/env python3
"""Validate the RK3588 SDPA op6 math-phase mapping.

Evidence sources:
  * RKNN_LOG_LEVEL=5 / RKNN_DUMP_* output from the board;
  * post-submit /dev/mem workspace dumps at selected op6 prefix lengths;
  * regcmd address chain from sdpa_npuop_timeline.csv.

The script recomputes CPU SDPA from the dumped Q/K/V, extracts the relevant
native-layout slices, and records correlations for:
  QK score, softmax row-max, row-max subtraction, FP16 exp/LUT, row-sum,
  softmax probability, and PV output.

It deliberately reports the layout transform used for each comparison and
masks C-lane padding before correlation.  The math mapping is only marked
validated where the corresponding tensor is matched by the dumped bytes.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

from sdpa_operation_map import classify_operation


def load_npy(dump_dir: Path, prefix: str) -> np.ndarray:
    hits = sorted(glob.glob(str(dump_dir / f"{prefix}*.npy")))
    if not hits:
        raise FileNotFoundError(f"missing {prefix}*.npy in {dump_dir}")
    return np.load(hits[0])


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


def native_score_layout(mat: np.ndarray) -> np.ndarray:
    """Logical [196,196] -> DPU native padded [13,16,196] order."""
    p = np.zeros((208, 196), dtype=np.float32)
    p[: mat.shape[0], :] = mat
    return p.reshape(13, 16, 196).transpose(0, 2, 1).ravel()


def native_fp16_layout(mat: np.ndarray) -> np.ndarray:
    """Logical [196,196] -> FP16 native padded [25,8,196] order."""
    p = np.zeros((200, 196), dtype=np.float32)
    p[: mat.shape[0], :] = mat
    return p.reshape(25, 8, 196).transpose(0, 2, 1).ravel()


def interior_mask(layout_fn, shape: tuple[int, int]) -> np.ndarray:
    """Positions belonging to the real 196x196 matrix, excluding C padding."""
    return layout_fn(np.ones(shape, dtype=np.float32)) > 0.5


def corr_masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return corr(np.asarray(a)[mask], np.asarray(b)[mask])


def native_output_layout(mat: np.ndarray) -> np.ndarray:
    """Logical [196,64] -> DPU native [4,16,196] order."""
    return mat.T.reshape(4, 16, 196).transpose(0, 2, 1).ravel()


def i16_stride8(raw: np.ndarray, count: int = 196) -> np.ndarray:
    """Read the row-vector layout used by the reduction tasks."""
    vals = np.frombuffer(raw[: count * 16].tobytes(), dtype="<i2")
    return vals[::8][: count].astype(np.float32)


def read_slice(prefix_dir: Path, name: str, addr: int, size: int) -> np.ndarray:
    path = prefix_dir / name
    if not path.exists():
        # ws_44 was captured separately; allow a flat sibling file.
        path = prefix_dir.parent / name
    data = np.fromfile(path, dtype=np.int8)
    off = addr - 0xFFE33000
    return data[off : off + size]


def read_fp16_slice(prefix_dir: Path, name: str, addr: int, count: int) -> np.ndarray:
    """Read `count` little-endian FP16 values from a workspace dump."""
    path = prefix_dir / name
    if not path.exists():
        path = prefix_dir.parent / name
    data = np.fromfile(path, dtype=np.uint8)
    off = addr - 0xFFE33000
    return np.frombuffer(data[off : off + 2 * count].tobytes(), dtype="<f2").astype(np.float32)


def reg_value(key_registers: str, name: str) -> int | None:
    m = re.search(rf"{re.escape(name)}=0x([0-9a-fA-F]+)", key_registers)
    return int(m.group(1), 16) if m else None


def build_dependencies(timeline: Path, out_csv: Path) -> list[dict]:
    rows = [r for r in csv.DictReader(timeline.open()) if r.get("op_idx") == "6"]
    task_csv = timeline.with_name("sdpa_task_timeline.csv")
    class_by_task = {}
    phase_by_task = {}
    if task_csv.exists():
        for r in csv.DictReader(task_csv.open()):
            if r.get("op_idx") == "6":
                task_idx = int(r["submit_task_order"])
                class_by_task[task_idx] = r.get("task_class", "")
                phase_by_task[task_idx] = classify_operation(r)
    by_task: dict[int, list[dict]] = {}
    for r in rows:
        by_task.setdefault(int(r["submit_task_order"]), []).append(r)

    read_names = {
        "CNA_FEATURE_DATA_ADDR",
        "CNA_DCOMP_ADDR0",
        "DPU_RDMA_RDMA_SRC_BASE_ADDR",
        "DPU_RDMA_RDMA_BS_BASE_ADDR",
        "DPU_RDMA_RDMA_BN_BASE_ADDR",
        "DPU_RDMA_RDMA_EW_BASE_ADDR",
    }
    out: list[dict] = []
    # H0 and H1 are the representative first/repeated head patterns.
    for head, start, end in [("head0", 1, 25), ("head1", 26, 50)]:
        last_writer: dict[int, tuple[int, int]] = {}
        for task_idx in range(start, end + 1):
            rs = by_task.get(task_idx, [])
            if not rs:
                continue
            dst = None
            reads: list[int] = []
            for r in rs:
                name = r["reg_name"]
                value = int(r["value"], 16)
                if name == "DPU_DST_BASE_ADDR":
                    dst = value
                elif name in read_names and value:
                    reads.append(value)
            producer = []
            for a in sorted(set(reads)):
                if a in last_writer:
                    ptask, prel = last_writer[a]
                    producer.append(f"t{ptask}/rel{prel}")
            out.append({
                "head": head,
                "task_index": task_idx,
                "head_task_order": rs[0]["head_task_order"],
                "math_phase": phase_by_task.get(task_idx, ""),
                "task_class": class_by_task.get(task_idx, ""),
                "write_addr": f"0x{dst:08x}" if dst is not None else "",
                "read_addrs": ";".join(f"0x{x:08x}" for x in sorted(set(reads))),
                "exact_producers": ";".join(producer),
            })
            if dst is not None:
                last_writer[dst] = (task_idx, int(rs[0]["head_task_order"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    return out


def build_head_address_map(timeline: Path, out_csv: Path) -> list[dict]:
    """Check the graph-region/per-head-stride formula for all 16 heads."""
    task_csv = timeline.with_name("sdpa_task_timeline.csv")
    task_rows = [
        r for r in csv.DictReader(task_csv.open()) if r.get("op_idx") == "6"
    ]
    op_rows = [
        r for r in csv.DictReader(timeline.open()) if r.get("op_idx") == "6"
    ]
    regs_by_task: dict[int, dict[str, str]] = {}
    for r in op_rows:
        regs_by_task.setdefault(int(r["submit_task_order"]), {})[r["reg_name"]] = r["value"]

    by_head: dict[str, list[dict]] = {}
    for r in task_rows:
        by_head.setdefault(r.get("head_group", ""), []).append(r)

    q_base = 0xFFF28000
    pv_base = 0xFFF59000
    stride = 0x3100
    out: list[dict] = []
    for head in range(16):
        hr = by_head.get(f"head{head}", [])
        qk = next(r for r in hr if classify_operation(r) == "qk_matmul")
        pv = next(r for r in hr if classify_operation(r) == "pv_matmul")
        qk_idx = int(qk["submit_task_order"])
        pv_idx = int(pv["submit_task_order"])
        q_read = regs_by_task[qk_idx].get("CNA_FEATURE_DATA_ADDR", "")
        pv_write = regs_by_task[pv_idx].get("DPU_DST_BASE_ADDR", "")
        q_expected = f"0x{q_base + head * stride:08x}"
        pv_expected = f"0x{pv_base + head * stride:08x}"
        out.append({
            "head": head,
            "qk_task": qk_idx,
            "qk_head_task_order": qk["head_task_order"],
            "q_tp_rs_read": q_read,
            "q_tp_rs_expected": q_expected,
            "q_match": int(q_read.lower() == q_expected),
            "pv_task": pv_idx,
            "pv_head_task_order": pv["head_task_order"],
            "mm_sdpa_write": pv_write,
            "mm_sdpa_expected": pv_expected,
            "pv_match": int(pv_write.lower() == pv_expected),
        })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rknn-dump-dir", required=True, type=Path)
    ap.add_argument("--prefix-dir", required=True, type=Path)
    ap.add_argument("--timeline", required=True, type=Path)
    ap.add_argument("--out-npz", required=True, type=Path)
    ap.add_argument("--out-validation", required=True, type=Path)
    ap.add_argument("--out-deps", required=True, type=Path)
    ap.add_argument("--out-head-map", type=Path, default=None)
    args = ap.parse_args()
    if args.out_head_map is None:
        args.out_head_map = args.out_deps.with_name("sdpa_head_address_map.csv")

    Q8 = load_npy(args.rknn_dump_dir, "000_InputOperator_Q")[0]
    K8 = load_npy(args.rknn_dump_dir, "001_InputOperator_K")[0]
    V8 = load_npy(args.rknn_dump_dir, "002_InputOperator_V")[0]
    out6 = load_npy(args.rknn_dump_dir, "006_")  # internal [16,64,1,196]
    out8 = load_npy(args.rknn_dump_dir, "008_OutputOperator")  # final [1,16,196,64]

    qscale, qzp = 0.0365723446, -2
    kscale, kzp = 0.0341316387, 0
    vscale, vzp = 0.0351534113, 2
    oscale, ozp = 0.00686752424, 1
    Qf = (Q8.astype(np.float32) - qzp) * qscale
    Kf = (K8.astype(np.float32) - kzp) * kscale
    Vf = (V8.astype(np.float32) - vzp) * vscale
    scores = (Qf @ Kf.transpose(0, 2, 1)) * 0.125
    rowmax = scores.max(axis=-1, keepdims=True)
    diff = scores - rowmax
    exp = np.exp(diff)
    rowsum = exp.sum(axis=-1, keepdims=True)
    probs = exp / rowsum
    y = probs @ Vf
    out_float = (out8[0].astype(np.float32) - ozp) * oscale

    # Slices actually produced by representative tasks.
    h0_qk = read_slice(args.prefix_dir, "ws_31.bin", 0xFFEA30C0, 13 * 16 * 196)
    h1_qk = read_slice(args.prefix_dir, "ws_32.bin", 0xFFEA30C0, 13 * 16 * 196)
    h1_diff = read_fp16_slice(args.prefix_dir, "ws_39.bin", 0xFFED5980, 25 * 8 * 196)
    h1_exp = read_fp16_slice(args.prefix_dir, "ws_40.bin", 0xFFEADC40, 25 * 8 * 196)
    h1_rowmax = read_slice(args.prefix_dir, "ws_38.bin", 0xFFEC1B00, 196 * 16)
    h1_rowsum = read_slice(args.prefix_dir, "ws_41.bin", 0xFFED40C0, 196 * 16)
    h0_prob = read_slice(args.prefix_dir, "ws_43.bin", 0xFFEFCA00, 13 * 16 * 196)
    h1_prob = read_slice(args.prefix_dir, "ws_44.bin", 0xFFEFCA00, 13 * 16 * 196)
    h0_pv = read_slice(args.prefix_dir, "ws_31.bin", 0xFFF59000, 4 * 16 * 196)
    h1_pv = read_slice(args.prefix_dir, "ws_51.bin", 0xFFF5C100, 4 * 16 * 196)

    score_mask = interior_mask(native_score_layout, scores[1].T.shape)
    fp16_mask = interior_mask(native_fp16_layout, scores[1].T.shape)
    records = [
        dict(stage="QK matmul", head=0, task_index=5, prefix="ws_31",
             address="0xffea30c0", expected="scores[head0]", layout="score.T -> C1WC2, interior",
             corr=corr_masked(h0_qk, native_score_layout(scores[0].T), score_mask)),
        dict(stage="QK matmul", head=1, task_index=31, prefix="ws_32",
             address="0xffea30c0", expected="scores[head1]", layout="score.T -> C1WC2, interior",
             corr=corr_masked(h1_qk, native_score_layout(scores[1].T), score_mask)),
        dict(stage="Softmax reduce (row max)", head=1, task_index=37, prefix="ws_38",
             address="0xffec1b00", expected="scores.max(-1)", layout="int16 stride 8",
             corr=corr(i16_stride8(h1_rowmax), rowmax[1].ravel())),
        dict(stage="Softmax subtract (row max)", head=1, task_index=38, prefix="ws_39",
             address="0xffed5980", expected="scores - max(scores) [head1]",
             layout="float16 C1WC2 (C2=8), interior",
             corr=corr_masked(h1_diff, native_fp16_layout(diff[1].T), fp16_mask)),
        dict(stage="Softmax exp/LUT", head=1, task_index=39, prefix="ws_40",
             address="0xffeadc40", expected="exp(scores - max(scores)) [head1]",
             layout="float16 C1WC2 (C2=8), interior",
             corr=corr_masked(h1_exp, native_fp16_layout(exp[1].T), fp16_mask)),
        dict(stage="Softmax exp/LUT (dump-to-dump)", head=1, task_index=39,
             prefix="ws_40", address="0xffeadc40",
             expected="exp(task38 FP16 dump)", layout="float16 interior",
             corr=corr(np.exp(h1_diff[fp16_mask]), h1_exp[fp16_mask])),
        dict(stage="Softmax normalize (row sum)", head=1, task_index=40, prefix="ws_41",
             address="0xffed40c0", expected="exp.sum(-1)", layout="int16 stride 8",
             corr=corr(i16_stride8(h1_rowsum), rowsum[1].ravel())),
        dict(stage="Softmax normalize (prob)", head=0, task_index=18, prefix="ws_43",
             address="0xffefca00", expected="softmax(scores)[head0]", layout="prob.T -> C1WC2, interior",
             corr=corr_masked(h0_prob, native_score_layout(probs[0].T), score_mask)),
        dict(stage="Softmax normalize (prob)", head=1, task_index=43, prefix="ws_44",
             address="0xffefca00", expected="softmax(scores)[head1]", layout="prob.T -> C1WC2, interior",
             corr=corr_masked(h1_prob, native_score_layout(probs[1].T), score_mask)),
        dict(stage="PV matmul", head=0, task_index=25, prefix="ws_31",
             address="0xfff59000", expected="probs @ V [head0]", layout="C1WC2",
             corr=corr(h0_pv, native_output_layout(y[0]))),
        dict(stage="PV matmul", head=1, task_index=50, prefix="ws_51",
             address="0xfff5c100", expected="probs @ V [head1]", layout="C1WC2",
             corr=corr(h1_pv, native_output_layout(y[1]))),
        dict(stage="whole model", head=-1, task_index=-1, prefix="rknn-dump",
             address="final output", expected="CPU SDPA output",
             layout="[1,16,196,64]", corr=corr(out_float, y)),
    ]
    args.out_validation.parent.mkdir(parents=True, exist_ok=True)
    with args.out_validation.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        for r in records:
            r = dict(r)
            r["corr"] = f"{r['corr']:.9f}"
            w.writerow(r)

    np.savez_compressed(
        args.out_npz,
        Q8=Q8, K8=K8, V8=V8, out6=out6, out8=out8,
        scores0=scores[0], scores1=scores[1],
        rowmax1=rowmax[1].ravel(), exp1=exp[1],
        rowsum1=rowsum[1].ravel(), probs0=probs[0], probs1=probs[1],
        y0=y[0], y1=y[1], out_float=out_float,
        h0_qk=h0_qk, h1_qk=h1_qk, h1_diff=h1_diff, h1_exp=h1_exp,
        h1_rowmax=h1_rowmax, h1_rowsum=h1_rowsum,
        h0_prob=h0_prob, h1_prob=h1_prob, h0_pv=h0_pv, h1_pv=h1_pv,
        score_mask=score_mask, fp16_mask=fp16_mask,
    )
    deps = build_dependencies(args.timeline, args.out_deps)
    head_map = build_head_address_map(args.timeline, args.out_head_map)

    print(f"wrote {args.out_validation}")
    print(f"wrote {args.out_npz}")
    print(f"wrote {args.out_deps} ({len(deps)} rows)")
    print(
        f"wrote {args.out_head_map} "
        f"({sum(r['q_match'] for r in head_map)}/16 Q matches, "
        f"{sum(r['pv_match'] for r in head_map)}/16 PV matches)"
    )
    for r in records:
        print(f"{r['stage']:32s} head={r['head']:2d} task={r['task_index']:3d} corr={r['corr']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
