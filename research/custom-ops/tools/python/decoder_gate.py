#!/usr/bin/env python3
"""Independent double decoder-block equation, cache integrity and geometry checks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from attention_gate import half, rope, counts as attention_counts, geometry


def linear(x,w,m,k,n):
    out=[]
    for r in range(m):
        for c in range(n):out.append(math.fsum(x[r*k+i]*w[c*k+i] for i in range(k)))
    return out


def rms(x,gamma,m,n,eps):
    out=[]
    for r in range(m):
        inv=1/math.sqrt(math.fsum(v*v for v in x[r*n:(r+1)*n])/n+eps)
        out.extend(x[r*n+c]*gamma[c]*inv for c in range(n))
    return out


def reference(c,root,ident):
    n,dm,ff,h,hv,d=c['tokens'],c['model_dim'],c['hidden_dim'],c['heads'],c['kv_heads'],c['head_dim']
    qw,kw=h*d,hv*d
    def load(name,size):return half(root/f'case-{ident}-{name}.bin',size)
    x=load('x',n*dm);mask=load('mask',n*n);norm1=load('norm1',dm);norm2=load('norm2',dm)
    wq=load('wq',qw*dm);wk=load('wk',kw*dm);wv=load('wv',kw*dm);wo=load('wo',dm*qw)
    wg=load('wgate',ff*dm);wu=load('wup',ff*dm);wd=load('wdown',dm*ff)
    z=rms(x,norm1,n,dm,c['epsilon'])
    q=linear(z,wq,n,dm,qw);k=linear(z,wk,n,dm,kw);v=linear(z,wv,n,dm,kw)
    qr=rope(q,n,h,d,c['rotary_dim'],c['style'],c['position_base'],c['theta'])
    kr=rope(k,n,hv,d,c['rotary_dim'],c['style'],c['position_base'],c['theta'])
    context=[]
    for t in range(n):
        for head in range(h):
            hk=head//(h//hv);s=[]
            for j in range(n):
                if j>t or (c['window'] and j+c['window']<=t):s.append(-math.inf)
                else:s.append(math.fsum(qr[(t*h+head)*d+i]*kr[(j*hv+hk)*d+i] for i in range(d))/math.sqrt(d)+mask[t*n+j])
            peak=max(s);e=[0.0]*n if peak==-math.inf else [math.exp(value-peak) for value in s];denom=math.fsum(e)
            context.extend(math.fsum((p/denom if denom else 0)*v[(j*hv+hk)*d+i] for j,p in enumerate(e)) for i in range(d))
    projected=linear(context,wo,n,qw,dm);residual=[a+b for a,b in zip(x,projected)]
    z=rms(residual,norm2,n,dm,c['epsilon']);gate=linear(z,wg,n,dm,ff);up=linear(z,wu,n,dm,ff)
    product=[a/(1+math.exp(-a))*b for a,b in zip(gate,up)]
    down=linear(product,wd,n,ff,dm)
    return [a+b for a,b in zip(residual,down)],kr,v


def node_counts(m,dm,ff,qw,kw):
    def tiles(n):return ((m+255)//256)*((n+255)//256)
    def mm(k,n):return tiles(n)*((k+255)//256)
    pre_tasks=3*tiles(dm)+mm(dm,qw)+2*mm(dm,kw)
    pre_submits=2*tiles(dm)+tiles(qw)+2*tiles(kw)
    lut=sum((r*c+32767)//32768 for r,c in geometry(m,ff))
    post_tasks=mm(qw,dm)+tiles(dm)+3*tiles(dm)+2*mm(dm,ff)+lut+2*tiles(ff)+mm(ff,dm)+tiles(dm)
    post_submits=tiles(dm)+tiles(dm)+2*tiles(dm)+2*tiles(ff)+tiles(ff)+tiles(ff)+tiles(dm)+tiles(dm)
    # CPU materialization of each RMSNorm plus graph input/output boundaries.
    uploads=tiles(dm)*4+tiles(qw)
    downloads=tiles(dm)*3+tiles(qw)+2*tiles(kw)
    return pre_tasks+post_tasks,pre_submits+post_submits,uploads,downloads


def verify(root,mode):
    results=[]
    for ident in range(3):
        c=json.loads((root/f'case-{ident}.json').read_text())
        if c['status']!='PASS' or not c['post_ffn_rollback']:raise ValueError('incomplete block/rollback case')
        n,dm,ff,h,hv,d=c['tokens'],c['model_dim'],c['hidden_dim'],c['heads'],c['kv_heads'],c['head_dim'];qw=h*d;kw=hv*d
        expected,kref,vref=reference(c,root,ident);data={};errors={}
        for name in ('baseline','optimized','full'):
            got=half(root/f'case-{ident}-{name}-y.bin',n*dm)
            if not all(math.isfinite(x) for x in got):raise ValueError('nonfinite block output')
            error=max(abs(a-b) for a,b in zip(got,expected))
            if error>0.008:raise ValueError(f'block equation mismatch {ident}/{name}: {error}')
            data[name]=got;errors[name+'_max_abs']=error
        cache_error=0
        for name in ('baseline','optimized'):
            k=half(root/f'case-{ident}-{name}-k.bin',n*kw);v=half(root/f'case-{ident}-{name}-v.bin',n*kw)
            cache_error=max(cache_error,max(abs(a-b) for a,b in zip(k,kref)),max(abs(a-b) for a,b in zip(v,vref)))
        if cache_error>0.006:raise ValueError('projected/rotated cache mismatch')
        for name in ('k','v'):
            if (root/f'case-{ident}-before-fail-{name}.bin').read_bytes()!=(root/f'case-{ident}-after-fail-{name}.bin').read_bytes():
                raise ValueError('post-FFN error modified committed prefix')
        if set((root/f'case-{ident}-fail-y.bin').read_bytes())!={0x23}:raise ValueError('post-FFN error published output')
        if len(c['runs'])!=6*len(c['steps']):raise ValueError('step repeat coverage')
        times={'baseline':[],'optimized':[]}
        for rep in range(3):
            old=0;last_geom=None;switches=0
            for step,tok in enumerate(c['steps']):
                found=[r for r in c['runs'] if r['repeat']==rep and r['step']==step]
                if {r['path'] for r in found}!={'baseline','optimized'} or len(found)!=2:raise ValueError('paired coverage')
                for run in found:
                    optimized=run['path']=='optimized';m=1 if optimized and tok==1 else c['max_query']
                    if (run['old'],run['tokens'],run['length'],run['query_rows'])!=(old,tok,old+tok,m):raise ValueError('cache/geometry progression')
                    base=node_counts(m,dm,ff,qw,kw);att=attention_counts(m,c['capacity'],h,hv,d,c['rotary_dim'])
                    task=base[0]+att[0];submit=base[1]+att[1];up=base[2]+att[2];down=base[3]+att[3]
                    if (run['submits'],run['tasks'],run['uploads'],run['downloads'])!=((submit if mode=='chain' else task),task,up,down):
                        raise ValueError(f'block task/transfer count mismatch {ident}: {run}, expected {(submit,task,up,down)}')
                    if run['score_elements']!=m*c['capacity']*h:raise ValueError('padded score work differs')
                    if optimized:
                        # Initial canonical geometry is max_query, including the first M=1 transition.
                        previous=c['max_query'] if last_geom is None else last_geom
                        if previous!=m:switches+=1
                        if run['cache_switches']!=switches:raise ValueError('KV copied without a geometry switch')
                        last_geom=m
                    elif run['cache_switches']:raise ValueError('baseline switched cache')
                    if tok==1:times[run['path']].append(run['api_ns'])
                old+=tok
            if old!=n:raise ValueError('token coverage')
        if len(c['full_runs'])!=3:raise ValueError('full graph repetitions')
        delta=max(abs(a-b) for a,b in zip(data['optimized'],data['baseline']))
        if delta>0.006:raise ValueError('decode geometry changed block semantics')
        results.append({'case':ident,'tokens':n,'model_dim':dm,'hidden_dim':ff,**errors,'cache_max_abs':cache_error,
            'optimized_vs_baseline_max_abs':delta,'post_ffn_rollback':True,
            'baseline_single_token_api_ns_median':statistics.median(times['baseline']),
            'optimized_single_token_api_ns_median':statistics.median(times['optimized']),
            'single_token_api_ratio':statistics.median(times['baseline'])/statistics.median(times['optimized']),
            'logical_score_reduction':c['max_query']})
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir()) if p.is_file() and p.name!='decoder-report.json'}
    return {'status':'PASS','mode':mode,'checks':results,'artifacts':hashes}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('results',type=Path);parser.add_argument('mode',choices=('chain','separate'));args=parser.parse_args()
    report=verify(args.results,args.mode);target=args.results/'decoder-report.json'
    if target.exists():
        if json.loads(target.read_text())!=report:raise ValueError('existing report differs')
    else:target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks'],indent=2));print('DECODER block mathematical gate PASS')
