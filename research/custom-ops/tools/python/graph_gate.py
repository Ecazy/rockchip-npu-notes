#!/usr/bin/env python3
"""Independent FP16 tile-aware SwiGLU reference; no graph planner/layout helpers."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
from cli_gate import f16, fill, write_half

SHAPES = ((1, 17, 33), (13, 257, 289), (257, 257, 289), (513, 33, 257))


def linear(m, k, n, x, w, bias=None):
    out = []
    for r in range(m):
        for c in range(n):
            acc = 0.0
            for k0 in range(0, k, 256):
                partial = math.fsum(x[r*k+q]*w[c*k+q] for q in range(k0, min(k0+256, k)))
                acc = f16(acc+partial)
            out.append(f16(acc+bias[c]) if bias is not None else acc)
    return out


def oracle(m, d, h, x, wg, wu, wd, bias, gamma=None):
    projected = x
    if gamma is not None:
        projected = []
        for r in range(m):
            squares = [f16(x[r*d+c]**2) for c in range(d)]
            total = sum(struct.unpack("<f", struct.pack("<f", math.fsum(squares[k0:k0+256])))[0]
                        for k0 in range(0, d, 256))
            scale = 1/math.sqrt(total/d+1e-5)
            projected.extend(f16(x[r*d+c]*f16(gamma[c]*scale)) for c in range(d))
    gate = linear(m, d, h, projected, wg, bias)
    up = linear(m, d, h, projected, wu)
    gated = [f16(f16(a*f16(1/(1+math.exp(-a))))*b) for a, b in zip(gate, up)]
    down = linear(m, h, d, gated, wd)
    return [f16(a+b) for a, b in zip(x, down)]


def geometry(m, cols):
    for m0 in range(0, m, 256):
        rows = (max(12, min(256, m-m0))+3)//4*4
        for c0 in range(0, cols, 256):
            channels = (min(256, cols-c0)+31)//32*32
            yield rows, channels


def counts(m, d, h, bias):
    hidden = list(geometry(m, h))
    output = list(geometry(m, d))
    tasks = len(hidden)*(2*((d+255)//256)+int(bias)) + len(output)*((h+255)//256)
    tasks += sum((r*c+32767)//32768+1 for r, c in hidden)  # SiLU
    tasks += len(hidden)+len(output)  # gating multiply and residual
    submits = 4*len(hidden)+2*len(output)
    return tasks, submits, len(output)


def run(binary, root, normalize=False):
    root.mkdir(parents=True, exist_ok=False)
    reports=[]
    for m, d, h in SHAPES:
        stem=f"ffn-{m}-{d}-{h}"
        x=fill(m*d, 123, 0.7)
        wg=fill(h*d, 451, 0.06)
        wu=fill(h*d, 911, 0.07)
        wd=fill(d*h, 771, 0.05)
        bias=fill(h, 153, 0.2)
        gamma=[f16(1+v) for v in fill(d, 229, 0.3)] if normalize else None
        gamma_path=root/f"{stem}-gamma.bin"
        if gamma is not None: write_half(gamma_path,gamma)
        paths=[root/f"{stem}-{name}.bin" for name in ("x", "wg", "wu", "wd", "bias")]
        for path, values in zip(paths, (x, wg, wu, wd, bias)):
            write_half(path, values)
        for use_bias in (False, True):
            ref=oracle(m, d, h, x, wg, wu, wd, bias if use_bias else None, gamma)
            baseline=None
            for mode in ("separate", "chain"):
                output=root/f"{stem}-b{int(use_bias)}-{mode}.bin"
                args=[str(binary), str(m), str(d), str(h), *map(str,paths[:4]),
                      str(paths[4]) if use_bias else "-", str(output), mode, "3"]
                if normalize: args.extend(["rmsnorm",str(gamma_path)])
                proc=subprocess.run(args, capture_output=True, text=True, timeout=90)
                if proc.returncode:
                    raise RuntimeError(f"graph failed {args}: {proc.stdout}{proc.stderr}")
                raw=output.read_bytes()
                if len(raw)!=m*d*2:
                    raise RuntimeError("output byte count mismatch")
                got=struct.unpack(f"<{m*d}e", raw)
                error=max(abs(a-b) for a,b in zip(got,ref))
                if not all(math.isfinite(v) for v in got) or error>0.003:
                    raise RuntimeError(f"oracle mismatch {stem} {mode}: {error}")
                match=re.search(r"repeat=3 submits=(\d+) tasks=(\d+) uploads=(\d+) downloads=(\d+) PASS",proc.stdout)
                tasks,submits,boundaries=counts(m,d,h,use_bias)
                if normalize:
                    tasks+=3*boundaries
                    submits+=2*boundaries
                transfers=boundaries*(2 if normalize else 1)
                expected=(3*(submits if mode=="chain" else tasks),3*tasks,3*transfers,3*transfers)
                if not match or tuple(map(int,match.groups()))!=expected:
                    raise RuntimeError(f"execution/transfer count mismatch: {proc.stdout} expected {expected}")
                if mode=="separate":
                    baseline=raw
                elif baseline!=raw:
                    raise RuntimeError("chain/separate byte mismatch")
                reports.append({"shape":[m,d,h],"rmsnorm":normalize,"bias":use_bias,"mode":mode,"max_abs":error,
                                "stdout":proc.stdout,"stderr":proc.stderr,"argv":args,"pass":True})
                print(f"GRAPH_CHECK {stem} bias={use_bias} {mode} max_abs={error:.9g} PASS",flush=True)
    # Nonfinite/truncated inputs must preserve an existing output file.
    marker=b"preserve-existing-output"
    output=root/"negative-output.bin"
    for data in (b"",struct.pack("<e",math.nan)):
        output.write_bytes(marker)
        bad=root/"negative-x.bin";bad.write_bytes(data)
        args=[str(binary),"1","17","33",str(bad),*map(str,paths[1:4]),"-",str(output)]
        proc=subprocess.run(args,capture_output=True,text=True,timeout=10)
        if proc.returncode!=1 or output.read_bytes()!=marker:
            raise RuntimeError("invalid input changed output")
    artifacts={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(root.iterdir()) if p.is_file()}
    (root/"graph-report.json").write_text(json.dumps({"status":"PASS","checks":reports,"negative_checks":2,
        "artifacts":artifacts},indent=2)+"\n")
    print(f"GRAPH_SUMMARY checks={len(reports)} negatives=2 PASS",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("binary",type=Path)
    parser.add_argument("results",type=Path)
    parser.add_argument("--rmsnorm",action="store_true")
    args=parser.parse_args()
    run(args.binary.resolve(),args.results,args.rmsnorm)
