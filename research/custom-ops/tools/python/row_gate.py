#!/usr/bin/env python3
"""Independent mathematical oracles for hybrid cross-tile RMSNorm/Softmax."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
from cli_gate import f16, fill, write_half

CASES = (
    ("rms", 1, 1, "zero"), ("rms", 13, 257, "random"),
    ("rms", 257, 513, "random"), ("rms", 13, 4096, "random"),
    ("hsig-rms", 13, 257, "random"),
    ("softmax", 1, 1, "mixed"), ("softmax", 13, 257, "mixed"),
    ("softmax", 257, 513, "mixed"), ("softmax", 13, 4096, "mixed"),
    ("softmax", 13, 4096, "tail"), ("hsig-softmax", 13, 257, "random"),
)


def inputs(m, n, distribution):
    if distribution == "random":
        return fill(m*n, 591, 2.0)
    if distribution == "zero":
        return [0.0]*(m*n)
    if distribution == "tail":
        # Many nonzero tails near the old LUT floor and below it, with a peak in
        # the ragged/final tile. Their combined denominator mass matters.
        return [f16(0 if c == n-1 else -9.5 if c % 2 else -17) for r in range(m) for c in range(n)]
    out=[]
    for r in range(m):
        for c in range(n):
            value = (-math.inf if r%6==0 else
                     1000.0 if r%6==1 else
                     (0.0 if c==n-1 else -40.0) if r%6==2 else
                     (c%31-15) if r%6==3 else
                     -math.inf if r%6==4 and c%3 else
                     (c%29-14)+1000 if r%6==5 else c%29-14)
            out.append(f16(value))
    return out


def reference(kind, m, n, x, gamma):
    if kind.startswith("hsig-"):
        x=[f16(min(1,max(0,v/6+0.5))) for v in x]
    out=[]
    for r in range(m):
        row=x[r*n:(r+1)*n]
        if kind.endswith("rms"):
            scale=1/math.sqrt(math.fsum(v*v for v in row)/n+1e-5)
            out.extend(v*gamma[c]*scale for c,v in enumerate(row))
        else:
            peak=max(row)
            values=[0.0]*n if peak == -math.inf else [math.exp(v-peak) for v in row]
            total=math.fsum(values)
            out.extend(v/total if total else 0.0 for v in values)
    return out


def counts(kind,m,n):
    tiles=((m+255)//256)*((n+255)//256)
    chunks=[]
    for m0 in range(0,m,256):
        rows=(max(12,min(256,m-m0))+3)//4*4
        for n0 in range(0,n,256):
            cols=(min(256,n-n0)+31)//32*32
            chunks.append((rows*cols+32767)//32768)
    soft=kind.endswith("softmax")
    tasks=sum(chunks)+3*tiles if soft else 3*tiles
    submits=2*tiles
    if kind.startswith("hsig-"):
        tasks+=sum(chunks);submits+=tiles
    return tasks,submits,tiles*(4 if soft else 2),2*tiles


def run(binary, root):
    root.mkdir(parents=True,exist_ok=False)
    reports=[]
    for kind,m,n,distribution in CASES:
        stem=f"{kind}-{m}-{n}-{distribution}"
        x=inputs(m,n,distribution)
        gamma=[f16(1+v) for v in fill(n,197,0.4)] if kind.endswith("rms") else None
        xfile=root/f"{stem}-x.bin";write_half(xfile,x)
        gammafile=root/f"{stem}-gamma.bin"
        if gamma is not None:write_half(gammafile,gamma)
        ref=reference(kind,m,n,x,gamma)
        baseline=None
        for mode in ("separate","chain"):
            output=root/f"{stem}-{mode}-y.bin"
            args=[str(binary),kind,str(m),str(n),str(xfile),str(gammafile) if gamma is not None else "-",str(output),mode,"3"]
            proc=subprocess.run(args,capture_output=True,text=True,timeout=90)
            if proc.returncode:raise RuntimeError(f"row graph failed {args}: {proc.stdout}{proc.stderr}")
            raw=output.read_bytes()
            if len(raw)!=m*n*2:raise RuntimeError("output size mismatch")
            got=struct.unpack(f"<{m*n}e",raw)
            error=max(abs(a-b) for a,b in zip(got,ref))
            tol=0.004 if gamma is not None else 0.002
            if not all(math.isfinite(v) for v in got) or error>tol:
                raise RuntimeError(f"oracle mismatch {stem}/{mode}: {error}")
            mass_error=0
            if gamma is None:
                for r in range(m):
                    expected=0 if not any(ref[r*n:(r+1)*n]) else 1
                    mass_error=max(mass_error,abs(math.fsum(got[r*n:(r+1)*n])-expected))
                    if any(v<0 for v in got[r*n:(r+1)*n]):raise RuntimeError("negative probability")
                if mass_error>0.002:raise RuntimeError(f"row mass mismatch {stem}: {mass_error}")
                if distribution == "tail" and max(abs(got[(r+1)*n-1]-ref[(r+1)*n-1]) for r in range(m))>0.001:
                    raise RuntimeError("long tail denominator incorrect")
            match=re.search(r"repeat=3 submits=(\d+) tasks=(\d+) uploads=(\d+) downloads=(\d+)",proc.stdout)
            tasks,submits,uploads,downloads=counts(kind,m,n)
            expected=(3*(submits if mode=="chain" else tasks),3*tasks,3*uploads,3*downloads)
            if not match or tuple(map(int,match.groups()))!=expected:
                raise RuntimeError(f"counter mismatch {proc.stdout} expected {expected}")
            if mode=="separate":baseline=raw
            elif baseline!=raw:raise RuntimeError("chain/separate mismatch")
            reports.append({"kind":kind,"shape":[m,n],"distribution":distribution,"mode":mode,
                "max_abs":error,"row_mass_error":mass_error,"stdout":proc.stdout,"stderr":proc.stderr,"argv":args,"pass":True})
            print(f"ROW_CHECK {stem} {mode} max_abs={error:.9g} row_mass_error={mass_error:.9g} PASS",flush=True)
    marker=b"preserve-existing-output"
    output=root/"negative-y.bin";bad=root/"negative-x.bin"
    negative_count=0
    for kind,n,value in (("softmax",1,math.inf),("softmax",1,math.nan),("rms",1,256.0)):
        write_half(bad,[value]*n)
        gammafile=root/"negative-gamma.bin";write_half(gammafile,[1]*n)
        output.write_bytes(marker)
        args=[str(binary),kind,"1",str(n),str(bad),str(gammafile) if kind=="rms" else "-",str(output)]
        proc=subprocess.run(args,capture_output=True,text=True,timeout=20)
        if proc.returncode!=1 or output.read_bytes()!=marker:raise RuntimeError("invalid numerical input changed output")
        negative_count+=1
    artifacts={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(root.iterdir()) if p.is_file()}
    (root/"row-report.json").write_text(json.dumps({"status":"PASS","checks":reports,"negative_checks":negative_count,
        "artifacts":artifacts},indent=2)+"\n")
    print(f"ROW_SUMMARY checks={len(reports)} negatives={negative_count} PASS",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("binary",type=Path)
    parser.add_argument("results",type=Path)
    args=parser.parse_args()
    run(args.binary.resolve(),args.results)
