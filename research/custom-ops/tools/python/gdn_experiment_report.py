#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render the four GDN experiments from a gated campaign directory.

The script only reads campaign artifacts (experiments.json, raw core logs and
the saved FP32 views of FP16 outputs/state). It intentionally does not rerun
the oracle; `gdn_experiments_gate.py` is the acceptance gate. This renderer
keeps the hardware-envelope and separate-submit caveats next to the numbers so
they are not mistaken for pure NPU execution time.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
from pathlib import Path
from typing import Iterable

CPU_ARMS = ("cpu16", "cpu32")
NPU_ARMS = ("npu_alloc", "npu_reuse", "beta_fold", "merged", "native_alpha")
ALL_ARMS = CPU_ARMS + NPU_ARMS
ARM_LABEL = {
    "cpu16": "CPU-FP16",
    "cpu32": "CPU-FP32",
    "npu_alloc": "NPU-FP16 six Task (default alloc)",
    "npu_reuse": "NPU-FP16 six Task (reuse)",
    "beta_fold": "NPU-FP16 beta-fold",
    "merged": "NPU-FP16 merged k/q",
    "native_alpha": "NPU-FP16 native alpha scale",
}
SCALAR_LABEL = {
    0: "A diagonal MatMul",
    1: "B expanded EW Mul",
    2: "C native scalar scale",
    3: "D CPU resident scale",
}
TASK_LABEL = (
    "alpha scale",
    "p = D k",
    "r = v - p",
    "e = beta r",
    "S' = D + e k^T",
    "o = S' q",
)


def med(values: Iterable[float]) -> float:
    return statistics.median(list(values))


def fmt(x: float, digits: int = 2) -> str:
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def read_rows(root: Path) -> list[dict]:
    rows = []
    for log in sorted(root.glob("core-*.log")):
        for line in log.read_text().splitlines():
            if line.startswith("{"):
                rows.append(json.loads(line))
    if not rows or not any(r.get("kind") == "result" and r.get("status") == "PASS" for r in rows):
        raise ValueError("campaign does not contain a PASS result")
    return rows


def plain_steps(rows: list[dict], arm: str, n: int) -> list[dict]:
    return [r for r in rows if r.get("kind") == "step" and r.get("arm") == arm
            and r.get("n") == n and not r.get("instrumented")]


def instrumented_steps(rows: list[dict], arm: str, n: int) -> list[dict]:
    return [r for r in rows if r.get("kind") == "step" and r.get("arm") == arm
            and r.get("n") == n and r.get("instrumented")]


def core_step_medians(rows: list[dict], arm: str, n: int) -> list[float]:
    root = Path(rows[0]["_root"])
    result = []
    for log in sorted(root.glob("core-*.log")):
        core_rows = [json.loads(line) for line in log.read_text().splitlines()
                     if line.startswith("{")]
        values = [r["total_us"] for r in core_rows
                  if r.get("kind") == "step" and r.get("arm") == arm
                  and r.get("n") == n and not r.get("instrumented")]
        if values:
            result.append(med(values))
    return result


def step_median(rows: list[dict], arm: str, n: int) -> float:
    """Median of the three per-core medians used by the report tables."""
    return med(core_step_medians(rows, arm, n))


def recurrence_table(rows: list[dict], sizes: list[int]) -> str:
    # The combined log loses the core tag. Read each core separately so the
    # robustness range is a real per-core median range, not a resampling range.
    per_core: dict[tuple[str, int, int], list[float]] = {}
    root = Path(rows[0]["_root"])
    for core in sorted({int(p.stem.split("-")[1]) for p in root.glob("core-*.log")}):
        core_rows = [json.loads(line) for line in (root / f"core-{core}.log").read_text().splitlines()
                     if line.startswith("{")]
        for arm in ALL_ARMS:
            for n in sizes:
                values = [r["total_us"] for r in core_rows
                          if r.get("kind") == "step" and r.get("arm") == arm
                          and r.get("n") == n and not r.get("instrumented")]
                if values:
                    per_core[(arm, n, core)] = med(values)
    lines = [
        "| n | arm | median us | per-core median range us | samples |",
        "|---:|---|---:|---:|---:|",
    ]
    for n in sizes:
        for arm in ALL_ARMS:
            vals = [v for (a, nn, c), v in per_core.items() if a == arm and nn == n]
            if not vals:
                continue
            lines.append(f"| {n} | {ARM_LABEL[arm]} | {fmt(med(vals))} | "
                         f"{fmt(min(vals))}..{fmt(max(vals))} | {len(vals) * 160} |")
    return "\n".join(lines)


def comparison_table(rows: list[dict], sizes: list[int]) -> str:
    lines = [
        "| n | CPU-FP16 us | CPU-FP32 us | NPU default us | NPU reuse us | NPU default / CPU-FP16 | NPU reuse / CPU-FP16 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in sizes:
        c16 = step_median(rows, "cpu16", n)
        c32 = step_median(rows, "cpu32", n)
        alloc = step_median(rows, "npu_alloc", n)
        reuse = step_median(rows, "npu_reuse", n)
        lines.append(f"| {n} | {fmt(c16)} | {fmt(c32)} | {fmt(alloc)} | {fmt(reuse)} | "
                     f"{alloc / c16:.1f}x | {reuse / c16:.1f}x |")
    return "\n".join(lines)


def optimization_table(rows: list[dict], sizes: list[int]) -> str:
    lines = [
        "| n | default alloc us | reuse six Task us | beta-fold us | merged k/q us | native alpha us | reuse vs default | beta-fold vs reuse | merged vs reuse | native-alpha vs reuse |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in sizes:
        cells = {arm: step_median(rows, arm, n) for arm in NPU_ARMS}
        base, reuse = cells["npu_alloc"], cells["npu_reuse"]
        def delta(arm: str) -> str:
            return f"{(cells[arm] / reuse - 1) * 100:+.1f}%"
        lines.append(f"| {n} | {fmt(base)} | {fmt(reuse)} | "
                     f"{fmt(cells['beta_fold'])} | {fmt(cells['merged'])} | "
                     f"{fmt(cells['native_alpha'])} | "
                     f"{(reuse / base - 1) * 100:+.1f}% | "
                     f"{delta('beta_fold')} | {delta('merged')} | {delta('native_alpha')} |")
    return "\n".join(lines)


def profile_table(rows: list[dict], sizes: list[int]) -> str:
    fields = ("total_us", "allocation_us", "prepare_us", "pack_k_us", "pack_q_us",
              "pack_v_us", "pack_coeff_us", "bo_sync_copy_us", "emit_us",
              "execute_api_us", "guard_us", "download_us")
    lines = [
        "| n | arm | total | host alloc | prep | pack k | pack q | pack v | pack coeff | BO sync/copy | emit | execute API | guard | download | other |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in sizes:
        for arm in NPU_ARMS:
            smp = instrumented_steps(rows, arm, n)
            if not smp:
                continue
            p = {k: med(r["profile"][k] for r in smp) for k in fields}
            other = med(r["profile"]["total_us"] -
                        sum(r["profile"][k] for k in fields if k != "total_us")
                        for r in smp)
            lines.append(
                f"| {n} | {ARM_LABEL[arm]} | {fmt(p['total_us'], 1)} | "
                f"{fmt(p['allocation_us'], 1)} | {fmt(p['prepare_us'], 1)} | "
                f"{fmt(p['pack_k_us'], 1)} | {fmt(p['pack_q_us'], 1)} | "
                f"{fmt(p['pack_v_us'], 1)} | {fmt(p['pack_coeff_us'], 1)} | "
                f"{fmt(p['bo_sync_copy_us'], 1)} | {fmt(p['emit_us'], 1)} | "
                f"{fmt(p['execute_api_us'], 1)} | {fmt(p['guard_us'], 1)} | "
                f"{fmt(p['download_us'], 1)} | {fmt(other, 1)} |"
            )
    return "\n".join(lines)


def executor_table(rows: list[dict], sizes: list[int]) -> str:
    lines = [
        "| n | arm | execute_api | command alloc | command build | submit+wait | output sync |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for n in sizes:
        for arm in NPU_ARMS:
            smp = instrumented_steps(rows, arm, n)
            if not smp:
                continue
            p = smp[0]["profile"]
            vals = {
                name: med(r["profile"][name] for r in smp)
                for name in ("execute_api_us", "command_allocation_us", "command_us",
                             "submit_wait_us", "output_sync_us")
            }
            lines.append(
                f"| {n} | {ARM_LABEL[arm]} | {fmt(vals['execute_api_us'], 1)} | "
                f"{fmt(vals['command_allocation_us'], 1)} | {fmt(vals['command_us'], 1)} | "
                f"{fmt(vals['submit_wait_us'], 1)} | {fmt(vals['output_sync_us'], 1)} |"
            )
    return "\n".join(lines)


def scalar_table(rows: list[dict], sizes: list[int]) -> str:
    lines = [
        "| n | geometry | arm | median total us | prepare+upload us | execute API us | guard us | max abs |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for n in sizes:
        for vector, geometry in ((0, "state matrix"), (1, "beta vector")):
            for arm in range(4):
                smp = [r for r in rows if r.get("kind") == "scalar" and r.get("n") == n
                       and r.get("vector") == vector and r.get("arm") == arm]
                if not smp:
                    continue
                lines.append(
                    f"| {n} | {geometry} | {SCALAR_LABEL[arm]} | "
                    f"{fmt(med(r['total_us'] for r in smp))} | "
                    f"{fmt(med(r['prepare_us'] + r['upload_us'] for r in smp))} | "
                    f"{fmt(med(r['execute_api_us'] for r in smp))} | "
                    f"{fmt(med(r['guard_us'] for r in smp))} | "
                    f"{max(r['max_abs'] for r in smp):.3g} |"
                )
    return "\n".join(lines)


def task_table(rows: list[dict], sizes: list[int]) -> str:
    lines = [
        "| n | " + " | ".join(f"task {i + 1}: {name}" for i, name in enumerate(TASK_LABEL)) + " |",
        "|---:|" + "---:|" * len(TASK_LABEL),
    ]
    for n in sizes:
        smp = [r for r in rows if r.get("kind") == "task_diagnostic" and r.get("n") == n]
        vals = [med(r["submit_wait_us"][i] for r in smp) for i in range(len(TASK_LABEL))]
        lines.append(f"| {n} | " + " | ".join(fmt(v, 1) for v in vals) + " |")
    return "\n".join(lines)


def read_f32(path: Path, count: int) -> tuple[float, ...]:
    data = path.read_bytes()
    if len(data) != count * 4:
        raise ValueError(f"unexpected result size: {path}")
    vals = struct.unpack(f"<{count}f", data)
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"nonfinite result: {path}")
    return vals


def numerical_table(experiments: dict, root: Path, sizes: list[int]) -> tuple[str, str]:
    lines = [
        "| n | arm | output max abs | output RMSE max | state max abs | state RMSE max |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for n in sizes:
        for arm in ALL_ARMS:
            errs = [e for run in experiments["runs"].values() for e in run["numerical"]
                    if e["n"] == n and e["arm"] == arm]
            out = [e for e in errs if e["tensor"] == "out"]
            state = [e for e in errs if e["tensor"] == "state"]
            if not out or not state:
                continue
            lines.append(
                f"| {n} | {ARM_LABEL[arm]} | {max(e['max_abs'] for e in out):.6g} | "
                f"{max(e['rmse'] for e in out):.6g} | {max(e['max_abs'] for e in state):.6g} | "
                f"{max(e['rmse'] for e in state):.6g} |"
            )
    # FP16-recomposition comparison against the default six-task output. These
    # are integer/bit comparisons over the saved FP32 views of FP16 values.
    cmp_lines = [
        "| n | arm | output mismatches / values | max abs vs six Task |",
        "|---:|---|---:|---:|",
    ]
    for n in sizes:
        for arm in ALL_ARMS:
            total = mismatch = 0
            max_diff = 0.0
            for path in sorted(root.glob("core-*/n*-b*-p0-out.bin")):
                # core-*/nN-bB-p0-out.bin: per arm [32*n] FP16-derived floats.
                fields = path.name.split("-")
                if int(fields[0][1:]) != n:
                    continue
                values = read_f32(path, len(ALL_ARMS) * 32 * n)
                arm_count = 32 * n
                ref = values[2 * arm_count:3 * arm_count]
                got = values[ALL_ARMS.index(arm) * arm_count:(ALL_ARMS.index(arm) + 1) * arm_count]
                for a, b in zip(got, ref):
                    total += 1
                    mismatch += a != b
                    max_diff = max(max_diff, abs(a - b))
            cmp_lines.append(f"| {n} | {ARM_LABEL[arm]} | {mismatch} / {total} | {max_diff:.6g} |")
    return "\n".join(lines), "\n".join(cmp_lines)


def metadata(root: Path) -> str:
    env = (root / "environment.log").read_text(errors="replace")
    original = (root / "cpu-original.log").read_text(errors="replace").strip()
    restored = (root / "cpu-restored.log").read_text(errors="replace").strip()
    control = (root / "cpu-control.log").read_text(errors="replace").strip()
    npu_original = (root / "npu-original.log").read_text(errors="replace").strip()
    npu_restored = (root / "npu-restored.log").read_text(errors="replace").strip()
    npu_control = (root / "npu-control.log").read_text(errors="replace").strip()
    binary = (root / "binary.sha256").read_text(errors="replace").strip()
    return "\n".join([
        "```text",
        env.strip(),
        "--- CPU policy before / after ---",
        original,
        restored,
        "--- CPU control ---",
        control,
        "--- NPU devfreq policy before / after ---",
        npu_original,
        npu_restored,
        "--- NPU control ---",
        npu_control,
        "--- executable ---",
        binary,
        "```",
    ])


def frequency_summary(rows: list[dict]) -> str:
    steps = [r for r in rows if r.get("kind") == "step"]
    scalars = [r for r in rows if r.get("kind") == "scalar"]
    cpu = sorted({r[k] for r in steps for k in ("freq_before_khz", "freq_after_khz")})
    npu = sorted({r[k] for r in steps for k in ("npu_freq_before_khz", "npu_freq_after_khz")})
    scpu = sorted({r["freq_khz"] for r in scalars})
    snpu = sorted({r["npu_freq_khz"] for r in scalars})
    return (f"Observed per-sample CPU frequency: {cpu} kHz; NPU frequency: {npu} kHz. "
            f"Scalar benchmark: CPU {scpu} kHz, NPU {snpu} kHz. Every recurrence step "
            "records before/after samples; the campaign scripts attempt to pin both "
            "clocks at the existing maxima and restore the original policies on exit.")


def render(root: Path) -> str:
    experiments = json.loads((root / "experiments.json").read_text())
    if experiments.get("status") != "PASS":
        raise ValueError("experiments.json is not PASS")
    rows = read_rows(root)
    # Attach the source directory for helpers that need per-core logs.
    for row in rows:
        row["_root"] = str(root)
    sizes = sorted({r["n"] for r in rows if r.get("kind") == "step"})
    ntab = numerical_table(experiments, root, sizes)
    return "\n\n".join([
        "# GDN CPU/NPU recurrence experiment report",
        f"Generated from `{root}`; acceptance gate: **PASS**.",
        "## Environment and frequency control\n\n" + metadata(root),
        "## Experiment 1: end-to-end CPU-resident recurrence vs NPU six-Task chain\n\n"
        + frequency_summary(rows) + "\n\n"
        "CPU-FP16 uses FP16 state/intermediates with FP32 NEON FMA and rounds D, p, r, e, "
        "S_next and o at the same task boundaries. CPU-FP32 keeps state and intermediates "
        "in FP32. Times are medians over 32 tokens x 5 interleaved blocks; the range is "
        "the spread of the three per-core medians, not a confidence interval.\n\n"
        + recurrence_table(rows, sizes)
        + "\n\n" + comparison_table(rows, sizes)
        + "\n\nThe ratio is an end-to-end API envelope comparison. It is not a claim "
          "that the NPU operation is intrinsically that many times slower: the NPU "
          "number includes host packing, submit/wait, guard checks and output download, "
          "while the CPU number starts and ends with CPU-resident state.",
        "## Experiment 2: six-Task chain cost decomposition\n\n"
        "`execute_api_us` is a host envelope: command allocation/build, blocking submit/wait "
        "and executor output sync. It is not pure hardware execution time. Guard is a "
        "host-side BO integrity scan over all resident buffers. Pack fields are host scatter "
        "into mapped BO storage; BO sync/copy is prep+fini around that access. The default "
        "arm allocates coefficient staging per token; the reuse arm preallocates it and the "
        "command BO in create and updates only dynamic contents.\n\n"
        + profile_table(rows, sizes),
        "### Executor envelope inside `execute_api_us`\n\n" + executor_table(rows, sizes),
        "### Separate-submit task diagnostic\n\n"
        "These are six independent submissions on one context, not per-task device execution "
        "inside the production chain. Each value contains host command build, submission, "
        "blocking wait and executor sync, so it is useful only for locating large envelopes. "
        "The 128/256 shapes use the same six-task semantics; 256 has more alpha-scale sub-tasks "
        "when the native scale arm is used.\n\n" + task_table(rows, sizes),
        "## Experiment 3: scalar scaling alternatives (A/B/C/D)\n\n"
        "A and B include explicit coefficient matrix construction and upload; the native C arm "
        "uses the verified scalar BN multiplier probe and no coefficient materialization. "
        "NPU output download and validation are outside these scalar timings for all NPU arms; "
        "D times 64 CPU-resident repetitions and reports the mean. The state-matrix geometry is "
        "the alpha path; the vector geometry is the beta K=1 path.\n\n"
        + scalar_table(rows, sizes),
        "## Experiment 4: merged k/q path in the complete recurrence\n\n"
        "The merged arm computes `[p,u]=D[k,q]` from the same old state, forms the residual, "
        "updates the state, and applies `o=u+e(k^T q)` with `k^T q` computed on the host. It "
        "is compared end-to-end, not only as an isolated GEMV. The beta-fold arm evaluates "
        "`e=beta*v-D*(beta*k)` with pre-scaled host inputs and drops the separate beta Task.\n\n"
        + optimization_table(rows, sizes),
        "## Numerical acceptance against the independent FP64 oracle\n\n"
        + ntab[0],
        "### FP16 recomposition / arm-to-arm output comparison\n\n"
        "The reference column is the default six-Task `npu_alloc` output. Mismatches are "
        "FP16-valued outputs that differ in the saved FP32 view; this isolates re-association "
        "of calculations that are equal in exact arithmetic.\n\n" + ntab[1],
        "## Caveats\n\n"
        "- CPU and NPU medians answer different cost questions; CPU-resident state avoids "
        "the device-to-host boundary and must not be presented as NPU latency.\n"
        "- Device execution, host command construction and blocking waits are not separated "
        "by hardware counters in this campaign. Do not call `submit_wait_us` pure compute.\n"
        "- The task diagnostic changes synchronization structure versus the chained arm.\n"
        "- The merged and native-alpha paths are experimental recipes, not new general DPU "
        "broadcast or aliasing guarantees.\n"
        "- Report only the tested shapes: one head, batch 1, dk=dv in {64,128,256}, fixed "
        "FP16 inputs and up to 32 recurrence steps per block.\n",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    text = render(args.campaign.resolve())
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
