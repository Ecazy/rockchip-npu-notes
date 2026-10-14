#!/usr/bin/env python3
"""Independent file-to-file operator gate using only Python's standard library.

No NPU layout helper is used by the oracle: raw X[M,K], W[N,K], b[N] -> equation.
CPU reference rounds each K partial accumulation and bias to FP16, per the operator ABI.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess


SHAPES = ((1, 1, 1), (4, 32, 32), (29, 37, 45), (37, 65, 33), (1, 256, 1), (256, 1, 256),
          (257, 37, 45), (29, 37, 257), (257, 257, 257), (513, 33, 289))


def f16(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def fill(count, seed, scale):
    # Independent generator from the C probe, deliberately using non-dyadic scales.
    values = []
    for _ in range(count):
        seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
        values.append(f16((((seed >> 8) & 0xFFFF) / 65535 * 2 - 1) * scale))
    return values


def write_half(path, values):
    path.write_bytes(struct.pack("<" + "e" * len(values), *values))


def oracle(m, k, n, x, w, bias):
    out = []
    for row in range(m):
        for col in range(n):
            dot = 0.0
            for k0 in range(0, k, 256):
                partial = math.fsum(x[row * k + q] * w[col * k + q]
                                    for q in range(k0, min(k0 + 256, k)))
                dot = f16(dot + partial)
            affine = f16(f16(dot) + bias[col])
            out.append(f16(min(1.0, max(0.0, affine / 6.0 + 0.5))))
    return out


def run(binary, root):
    root.mkdir(parents=True, exist_ok=True)
    checks = []
    commands = []
    for m, k, n in SHAPES:
        stem = f"cli-{m}-{k}-{n}"
        x, w, b = fill(m * k, 303, 1.7), fill(n * k, 911, 0.21), fill(n, 409, 4.7)
        inputs = [root / f"{stem}-{name}.bin" for name in ("x", "w", "bias")]
        for path, values in zip(inputs, (x, w, b)):
            write_half(path, values)
        ref = oracle(m, k, n, x, w, b)
        baseline = None
        for mode in ("separate", "chain"):
            output = root / f"{stem}-{mode}-y.bin"
            command = [str(binary), str(m), str(k), str(n), *map(str, inputs), str(output), mode, "3"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
            commands.append({"argv": command, "exit": result.returncode,
                             "stdout": result.stdout, "stderr": result.stderr})
            if result.returncode:
                raise RuntimeError(f"operator failed: {command}: {result.stdout}{result.stderr}")
            data = output.read_bytes()
            if len(data) != m * n * 2:
                raise RuntimeError(f"wrong output length: {output}")
            got = struct.unpack("<" + "e" * (m * n), data)
            error = max(abs(a - b) for a, b in zip(got, ref))
            if not all(math.isfinite(v) for v in got) or error > 0.002:
                raise RuntimeError(f"CPU oracle mismatch {stem}/{mode}: {error}")
            tasks = 0
            for m0 in range(0, m, 256):
                rows = ((max(min(256, m - m0), 12) + 3) // 4) * 4
                for n0 in range(0, n, 256):
                    cols = ((min(256, n - n0) + 31) // 32) * 32
                    tasks += (k + 255) // 256 + 1 + (rows * cols + 32767) // 32768
            output_tiles = ((m + 255) // 256) * ((n + 255) // 256)
            expected_submits = output_tiles * 3 if mode == "chain" else tasks * 3
            match = re.search(r"repeat=3 submits=(\d+) tasks=(\d+) PASS", result.stdout)
            if not match or (int(match[1]), int(match[2])) != (expected_submits, tasks * 3):
                raise RuntimeError(f"submit counters differ: {result.stdout}")
            if mode == "separate":
                baseline = data
            elif data != baseline:
                raise RuntimeError(f"CLI mode byte mismatch: {stem}")
            record = {"shape": [m, k, n], "mode": mode, "max_abs": error, "pass": True}
            checks.append(record)
            print(f"CLI_CHECK {stem} {mode} max_abs={error:.9g} PASS")

    # Invalid-size/nonfinite inputs and bad shape must leave an existing output intact.
    xfile, wfile, bfile = (root / f"negative-{name}.bin" for name in ("x", "w", "b"))
    write_half(wfile, [1]); write_half(bfile, [0])
    output = root / "negative-output.bin"
    marker = b"existing-output-preserved"
    negative = [(b"", "1"), (struct.pack("<ee", 1, 2), "1"), (struct.pack("<e", math.nan), "1"),
                (struct.pack("<e", 1), "0"), (struct.pack("<e", 1), "4097")]
    for index, (data, m) in enumerate(negative):
        xfile.write_bytes(data);output.write_bytes(marker)
        command = [str(binary), m, "1", "1", str(xfile), str(wfile), str(bfile), str(output)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        commands.append({"argv": command, "exit": result.returncode,
                         "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode != 1 or output.read_bytes() != marker:
            raise RuntimeError(f"invalid input accepted / output overwritten: {index}")
        checks.append({"negative_case": index, "pass": True})
        print(f"CLI_CHECK invalid-{index} preserves-output PASS")
    report = {"status": "PASS", "checks": checks, "commands": commands,
              "core_mask": os.environ.get("RKNPU_CORE_MASK", "AUTO")}
    (root / "cli-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"CLI_SUMMARY checks={len(checks)} status=PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    run(args.binary.resolve(), args.results)
