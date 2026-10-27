#!/usr/bin/env python3
"""Same-program two-runner control for the full-model direct-group probe."""
import argparse, gzip, json, statistics, struct, time
from pathlib import Path
import numpy as np
from rknpu_backend.hybrid import HybridRunner, load_hybrid_program
from rknpu_backend.models import load_weight_npz
from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi

def read_idx(path):
    with gzip.open(path, 'rb') as f:
        f.read(4); dims=struct.unpack('>III',f.read(12))
        return np.frombuffer(f.read(),dtype=np.uint8).reshape(dims)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plans-dir',type=Path,required=True); ap.add_argument('--weights-dir',type=Path,required=True); ap.add_argument('--lib',required=True); ap.add_argument('--data-dir',type=Path,required=True); ap.add_argument('--batch',type=int,default=8); ap.add_argument('--limit',type=int,default=64); ap.add_argument('--rounds',type=int,default=1); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    imgs=read_idx(args.data_dir/'t10k-images-idx3-ubyte.gz')[:args.limit]
    prog=load_hybrid_program(args.plans_dir/f'tiny_vit_b{args.batch}.rkhybrid.json',plans_dir=args.plans_dir)
    w=load_weight_npz(args.weights_dir/'tiny_vit.npz'); lib=LibRKPlan(args.lib); configure_rocket_abi(lib.cdll); fd=int(lib.cdll.rocket_open())
    a=b=None
    try:
        t0=time.perf_counter(); a=HybridRunner(lib,prog,w,weights_path=args.weights_dir/'tiny_vit.npz',fd=fd,execution_mode='graph'); sa=(time.perf_counter()-t0)*1e6
        t0=time.perf_counter(); b=HybridRunner(lib,prog,w,weights_path=args.weights_dir/'tiny_vit.npz',fd=fd,execution_mode='graph'); sb=(time.perf_counter()-t0)*1e6
        a.run(imgs[:args.batch],stats=None); b.run(imgs[:args.batch],stats=None)
        samples={'a':[],'b':[]}; bit=True; calls=args.limit//args.batch
        for r in range(args.rounds):
          for ci in range(calls):
            st=ci*args.batch; en=st+args.batch; order=('a','b') if (ci+r)%2==0 else ('b','a'); out={}; tim={}
            for n in order:
                t0=time.perf_counter(); out[n]=a.run(imgs[st:en],stats=None).outputs['logits'] if n=='a' else b.run(imgs[st:en],stats=None).outputs['logits']; tim[n]=(time.perf_counter()-t0)*1e6
            if not np.array_equal(out['a'],out['b']): bit=False
            for n in ('a','b'): samples[n].append(tim[n])
        def st(v): return {'median_us':float(statistics.median(v)),'p10_us':float(sorted(v)[(len(v)-1)//10]),'min_us':float(min(v)),'max_us':float(max(v)),'n':len(v)}
        p={'bitwise':bit,'setup_us':{'a':sa,'b':sb},'timing':{k:st(v) for k,v in samples.items()},'paired_a_over_b':st([x/y for x,y in zip(samples['a'],samples['b'])])}
        args.out.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n'); print(json.dumps(p,indent=2,sort_keys=True))
    finally:
        if a: a.close()
        if b: b.close()
        lib.cdll.rocket_close(fd)
if __name__=='__main__': raise SystemExit(main())
