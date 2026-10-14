#!/usr/bin/env python3
"""Independent projected-QKV SDPA/RoPE oracles and cache transaction witnesses."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def half(path,n):
    raw=path.read_bytes()
    if len(raw)!=n*2:raise ValueError(f"wrong byte count: {path}")
    return struct.unpack(f"<{n}e",raw)


def rope(x,n,h,d,rd,style,base,theta):
    if not rd:return list(x)
    out=[]
    for t in range(n):
        for head in range(h):
            row=x[(t*h+head)*d:(t*h+head+1)*d]
            for i in range(d):
                if i>=rd:out.append(row[i]);continue
                pair=(i^1) if style==0 else (i+rd//2 if i<rd//2 else i-rd//2)
                freq=i//2 if style==0 else i%(rd//2)
                negative=(i%2==0) if style==0 else i<rd//2
                angle=(base+t)*theta**(-2*freq/rd)
                out.append(row[i]*math.cos(angle)+(-row[pair] if negative else row[pair])*math.sin(angle))
    return out


def reference(c,q,k,v,mask):
    n,h,hv,d=c['tokens'],c['heads'],c['kv_heads'],c['head_dim']
    qr=rope(q,n,h,d,c['rotary_dim'],c['style'],c['position_base'],c['theta'])
    kr=rope(k,n,hv,d,c['rotary_dim'],c['style'],c['position_base'],c['theta'])
    out=[]
    for t in range(n):
        for head in range(h):
            hk=head//(h//hv)
            scores=[]
            for j in range(n):
                if (c['causal'] and j>t) or (c['window'] and j+c['window']<=t):scores.append(-math.inf);continue
                dot=math.fsum(qr[(t*h+head)*d+i]*kr[(j*hv+hk)*d+i] for i in range(d))
                scores.append(dot/math.sqrt(d)+mask[t*n+j])
            peak=max(scores)
            e=[0.0]*n if peak==-math.inf else [math.exp(s-peak) for s in scores]
            denom=math.fsum(e)
            for i in range(d):out.append(math.fsum((p/denom if denom else 0)*v[(j*hv+hk)*d+i] for j,p in enumerate(e)))
    return out,kr


def geometry(m,n):
    for m0 in range(0,m,256):
        rows=(max(12,min(256,m-m0))+3)//4*4
        for n0 in range(0,n,256):
            cols=(min(256,n-n0)+31)//32*32
            yield rows,cols


def counts(m,cap,h,hv,d,rd):
    mt=(m+255)//256;nt=(cap+255)//256;st=mt*nt
    chunks=sum((r*c+32767)//32768 for r,c in geometry(m,cap))
    rope_tasks=3*mt*(h+hv) if rd else 0
    rope_submits=mt*(h+hv) if rd else 0
    tasks=h*(2*st+chunks+3*st)+rope_tasks
    submits=h*(3*st+mt)+rope_submits
    row_classes=2 if m>256 and m%256 else 1
    uploads=h*(mt+3*st)+2*hv*row_classes*nt
    downloads=h*(mt+st)
    if rd:uploads+=hv*mt+3*mt*(h+hv);downloads+=hv*mt
    return tasks,submits,uploads,downloads


def verify(root,mode):
    records=[]
    for ident in range(6):
        c=json.loads((root/f'case-{ident}.json').read_text())
        if c['status']!='PASS' or not c['rollback']:raise ValueError("incomplete case/rollback")
        n,h,hv,d=c['tokens'],c['heads'],c['kv_heads'],c['head_dim'];qs=n*h*d;ks=n*hv*d
        q=half(root/f'case-{ident}-q.bin',qs);k=half(root/f'case-{ident}-k.bin',ks);v=half(root/f'case-{ident}-v.bin',ks)
        mask=half(root/f'case-{ident}-mask.bin',n*n)
        expected,rotated=reference(c,q,k,v,mask)
        errors={}
        for name in ('stream','graph'):
            got=half(root/f'case-{ident}-{name}-y.bin',qs)
            if not all(math.isfinite(x) for x in got):raise ValueError("nonfinite attention result")
            error=max(abs(a-b) for a,b in zip(got,expected))
            if error>0.004:raise ValueError(f"SDPA mismatch case {ident}/{name}: {error}")
            if any(got[(2*h)*d:(3*h)*d]):raise ValueError("fully masked row is not zero")
            errors[name+'_max_abs']=error
        cached=half(root/f'case-{ident}-cache-k.bin',ks)
        cache_error=max(abs(a-b) for a,b in zip(cached,rotated))
        for idx,value in enumerate(cached):
            if not c['rotary_dim'] or idx%d>=c['rotary_dim']:
                if value!=k[idx]:raise ValueError("unrotated K dimension changed")
        if cache_error>0.002:raise ValueError(f"RoPE cache mismatch {ident}: {cache_error}")
        if (root/f'case-{ident}-cache-v.bin').read_bytes()!=(root/f'case-{ident}-v.bin').read_bytes():
            raise ValueError("V cache differs from inputs")
        for field in ('k','v'):
            if (root/f'case-{ident}-before-fail-{field}.bin').read_bytes()!=(root/f'case-{ident}-after-fail-{field}.bin').read_bytes():
                raise ValueError("failed append modified committed cache")
        if set((root/f'case-{ident}-fail-y.bin').read_bytes())!={0x23}:raise ValueError("failed append modified output")
        task,submit,up,down=counts(c['max_query'],c['capacity'],h,hv,d,c['rotary_dim'])
        if len(c['runs'])!=3*len(c['steps']):raise ValueError("step repetitions incomplete")
        for rep in range(3):
            old=0
            for idx,tok in enumerate(c['steps']):
                run=c['runs'][rep*len(c['steps'])+idx]
                if (run['repeat'],run['old'],run['tokens'],run['length'])!=(rep,old,tok,old+tok):raise ValueError("cache length progression")
                if (run['submits'],run['tasks'],run['uploads'],run['downloads'])!=((submit if mode=='chain' else task),task,up,down):
                    raise ValueError("stream task/transfer count mismatch")
                old+=tok
            if old!=n:raise ValueError("step token coverage")
        task,submit,up,down=counts(n,n,h,hv,d,c['rotary_dim'])
        qt=((n+255)//256)*((h*d+255)//256);kt=((n+255)//256)*((hv*d+255)//256);maskt=((n+255)//256)**2
        expected_graph=((submit if mode=='chain' else task),task,qt+2*kt+maskt+up+qt,down+qt+2*kt+maskt+qt)
        if len(c['graph_runs'])!=3:raise ValueError("graph repetitions incomplete")
        for run in c['graph_runs']:
            if tuple(run[key] for key in ('submits','tasks','uploads','downloads'))!=expected_graph:
                raise ValueError("graph task/transfer count mismatch")
        records.append({'case':ident,'tokens':n,'heads':h,'kv_heads':hv,**errors,'cache_rope_max_abs':cache_error,'rollback':True})
    rx=half(root/'case-99-rope-x.bin',13*4*100);ry=half(root/'case-99-rope-y.bin',13*4*100)
    rr=rope(rx,13,4,100,96,1,17,10000)
    rope_error=max(abs(a-b) for a,b in zip(ry,rr))
    if rope_error>0.002:raise ValueError("cross-tile RoPE mismatch")
    rx2=half(root/'case-98-rope-x.bin',257*16*19);ry2=half(root/'case-98-rope-y.bin',257*16*19)
    rr2=rope(rx2,257,16,19,18,0,1000,10000)
    rope_error2=max(abs(a-b) for a,b in zip(ry2,rr2))
    if rope_error2>0.002:raise ValueError("multi-row/channel-tile interleaved RoPE mismatch")
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir()) if p.is_file() and p.name!='attention-report.json'}
    return {'status':'PASS','mode':mode,'checks':records,'cross_tile_rope_max_abs':max(rope_error,rope_error2),'artifacts':hashes}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('results',type=Path);parser.add_argument('mode',choices=('chain','separate'))
    args=parser.parse_args();report=verify(args.results,args.mode);target=args.results/'attention-report.json'
    if target.exists():
        if json.loads(target.read_text())!=report:raise ValueError("existing report differs")
    else:target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks'],indent=2));print('ATTENTION mathematical gate PASS')
