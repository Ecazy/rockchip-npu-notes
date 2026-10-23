// SPDX-License-Identifier: GPL-3.0-or-later
/* BO/execution mocks are confined to this test binary. Packing, emitters, graph
 * analysis and execution orchestration remain production code. */
#include "co_graph.h"
#include "co_decoder.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks,failures,live,allocs,fail_alloc,execs,fail_exec;
static uint32_t next=1;static size_t resident;
#define EXPECT(e) do {checks++;if(!(e)){fprintf(stderr,"FAIL %d %s\n",__LINE__,#e);failures++;}} while(0)
int __wrap_rocket_bo_alloc32(int fd,size_t size,rocket_bo *bo) {
    (void)fd;if(++allocs==fail_alloc)return ROCKET_E_NOMEM;
    if(!live)next=1;
    memset(bo,0,sizeof(*bo));bo->ptr=calloc(1,size);if(!bo->ptr)return ROCKET_E_NOMEM;
    bo->size=size;bo->handle=next++;bo->dma_address=(uint64_t)bo->handle*0x400000;
    live++;resident+=size;return 0;
}
void __wrap_rocket_bo_free(int fd,rocket_bo *bo) {(void)fd;free(bo->ptr);live--;resident-=bo->size;memset(bo,0,sizeof(*bo));}
int __wrap_rocket_bo_prep(int fd,rocket_bo *bo,int dir,uint64_t ns) {(void)fd;(void)bo;(void)dir;(void)ns;return 0;}
int __wrap_rocket_bo_fini(int fd,rocket_bo *bo) {(void)fd;(void)bo;return 0;}
int __wrap_rocket_batched_submit_supported(void) {return 1;}
int __real_co_emit_ew(co_program *,co_view,co_view,co_view,co_view,int);
int __wrap_co_emit_ew(co_program *p,co_view x,co_view a,co_view w,co_view y,int mul) {
    int rc=__real_co_emit_ew(p,x,a,w,y,mul);if(!rc&&mul)p->flags=0x10000;return rc;
}
int __real_co_emit_lut(co_program *,co_view,co_view,enum co_act);
int __wrap_co_emit_lut(co_program *p,co_view x,co_view y,enum co_act act) {
    int rc=__real_co_emit_lut(p,x,y,act);if(!rc)p->flags=(unsigned)act;return rc;
}
static _Float16 *data(co_view v) {return (_Float16 *)((char *)v.buffer->bo.ptr+v.offset);}
int __wrap_co_execute(int fd,const co_program *p,unsigned count,enum co_mode mode) {
    (void)fd;(void)mode;if(++execs==fail_exec)return ROCKET_E_DEVICE;
    for(unsigned i=0;i<count;i++) {
        co_view x=p[i].reads[0],y=p[i].output;_Float16 *src=data(x),*dst=data(y);
        if(p[i].nreads==1) {
            for(size_t q=0;q<y.tensor.bytes/2;q++) {
                double v=src[q],a=p[i].flags==CO_SIGMOID?1/(1+exp(-v)):
                    p[i].flags==CO_GELU_GATE?0.5*(1+erf(v/sqrt(2))):
                    p[i].flags==CO_EXP_NEG4?exp(v):fmin(1,fmax(0,v/6+0.5));
                dst[q]=(_Float16)a;
            }
        } else {
            co_view w=p[i].reads[1];_Float16 *weight=data(w);
            for(unsigned r=0;r<y.tensor.physical_rows;r++)for(unsigned c=0;c<y.tensor.physical_cols;c++) {
                double v=0;
                for(unsigned q=0;q<x.tensor.physical_cols;q++)
                    v+=(double)src[co_index(&x.tensor,r,q)]*weight[co_index(&w.tensor,c,q)];
                if(p[i].nreads==3) {
                    co_view a=p[i].reads[2];double operand=data(a)[co_index(&a.tensor,r,c)];
                    v=p[i].flags==0x10000?v*operand:v+operand;
                }
                if(y.tensor.type==CO_F32)((float *)dst)[co_index(&y.tensor,r,c)]=(float)v;
                else {
                    _Float16 narrowed=(_Float16)v;
                    if(p[i].flags==0x10000&&p[i].reads[0].buffer==p[i].reads[2].buffer&&
                       isinf((double)narrowed))narrowed=65504;
                    dst[co_index(&y.tensor,r,c)]=narrowed;
                }
            }
        }
    }
    return 0;
}
static _Float16 fill(unsigned i,unsigned seed) {return (_Float16)(((int)((i*17+seed)%31)-15)/128.0);}
static void linear_ref(unsigned m,unsigned k,unsigned n,const _Float16 *x,const _Float16 *w,const _Float16 *bias,_Float16 *y) {
    for(unsigned r=0;r<m;r++)for(unsigned c=0;c<n;c++) {
        _Float16 acc=0;
        for(unsigned k0=0;k0<k;k0+=256) {
            double v=acc;unsigned end=k-k0>256?k0+256:k;
            for(unsigned q=k0;q<end;q++)v+=(double)x[(size_t)r*k+q]*w[(size_t)c*k+q];
            acc=(_Float16)v;
        }
        y[(size_t)r*n+c]=bias?(_Float16)(acc+bias[c]):acc;
    }
}
static void execution_case(unsigned m,unsigned d,unsigned h) {
    _Float16 *x=malloc((size_t)m*d*2),*wg=malloc((size_t)h*d*2),*wu=malloc((size_t)h*d*2),
        *wd=malloc((size_t)d*h*2),*bias=malloc(h*2),*gate=malloc((size_t)m*h*2),
        *up=malloc((size_t)m*h*2),*ref=malloc((size_t)m*d*2),*y=malloc((size_t)m*d*2);
    EXPECT(x&&wg&&wu&&wd&&bias&&gate&&up&&ref&&y);if(!x||!wg||!wu||!wd||!bias||!gate||!up||!ref||!y)exit(1);
    for(size_t i=0;i<(size_t)m*d;i++)x[i]=fill(i,2);
    for(size_t i=0;i<(size_t)h*d;i++){wg[i]=fill(i,3);wu[i]=fill(i,7);}
    for(size_t i=0;i<(size_t)d*h;i++)wd[i]=fill(i,11);
    for(unsigned i=0;i<h;i++)bias[i]=fill(i,17);
    co_graph_node nodes[]={
        {.op=CO_GRAPH_INPUT,.rows=m,.cols=d},
        {.op=CO_GRAPH_LINEAR,.a=0,.cols=h,.weights=wg,.bias=bias},
        {.op=CO_GRAPH_LINEAR,.a=0,.cols=h,.weights=wu},
        {.op=CO_GRAPH_SILU,.a=1}, {.op=CO_GRAPH_MUL,.a=3,.b=2},
        {.op=CO_GRAPH_LINEAR,.a=4,.cols=d,.weights=wd}, {.op=CO_GRAPH_ADD,.a=5,.b=0},
        {.op=(enum co_graph_op)99}};
    co_graph_desc desc={nodes,8,6,0};co_graph_plan p;
    EXPECT(!co_graph_plan_init(&desc,&p));EXPECT(p.steps==7&&p.inputs==1&&p.slots<7);
    EXPECT(p.tensors[0].last==6&&p.tensors[0].slot!=p.tensors[5].slot);
    linear_ref(m,d,h,x,wg,bias,gate);linear_ref(m,d,h,x,wu,NULL,up);
    for(size_t i=0;i<(size_t)m*h;i++) {
        _Float16 sigmoid=(_Float16)(1/(1+exp(-(double)gate[i])));
        gate[i]=(_Float16)(gate[i]*sigmoid);gate[i]=(_Float16)(gate[i]*up[i]);
    }
    linear_ref(m,h,d,gate,wd,NULL,ref);
    for(size_t i=0;i<(size_t)m*d;i++)ref[i]=(_Float16)(ref[i]+x[i]);
    for(unsigned mode=0;mode<2;mode++) {
        co_graph *g=NULL;EXPECT(!co_graph_compile(3,&desc,(enum co_mode)mode,&g));if(!g)exit(1);
        EXPECT(resident==p.device_bytes);co_graph_input input={0,x};
        EXPECT(!co_graph_run(g,&input,1,y));EXPECT(!memcmp(y,ref,(size_t)m*d*2));
        unsigned before=execs;input.node=1;
        EXPECT(co_graph_run(g,&input,1,y)==ROCKET_E_SHAPE);EXPECT(execs==before);input.node=0;
        memset(y,0x23,(size_t)m*d*2);fail_exec=execs+2;
        EXPECT(co_graph_run(g,&input,1,y)==ROCKET_E_DEVICE);fail_exec=0;
        for(size_t i=0;i<(size_t)m*d*2;i++)EXPECT(((unsigned char *)y)[i]==0x23);
        EXPECT(!co_graph_run(g,&input,1,y));EXPECT(!memcmp(y,ref,(size_t)m*d*2));
        /* A second run with different inputs must produce a different, correct
         * result: the plan is re-runnable, not a one-shot buffer image. */
        for(size_t i=0;i<(size_t)m*d;i++)x[i]=(_Float16)((double)x[i]*0.5+0.125);
        linear_ref(m,d,h,x,wg,bias,gate);linear_ref(m,d,h,x,wu,NULL,up);
        for(size_t i=0;i<(size_t)m*h;i++) {
            _Float16 sig=(_Float16)(1/(1+exp(-(double)gate[i])));
            gate[i]=(_Float16)(gate[i]*sig);gate[i]=(_Float16)(gate[i]*up[i]);
        }
        linear_ref(m,h,d,gate,wd,NULL,ref);
        for(size_t i=0;i<(size_t)m*d;i++)ref[i]=(_Float16)(ref[i]+x[i]);
        _Float16 first_out=y[0];
        EXPECT(!co_graph_run(g,&input,1,y));EXPECT(!memcmp(y,ref,(size_t)m*d*2));
        EXPECT(y[0]!=first_out);
        /* and the failure path still leaves the buffer untouched afterwards */
        memset(y,0x23,(size_t)m*d*2);fail_exec=execs+2;
        EXPECT(co_graph_run(g,&input,1,y)==ROCKET_E_DEVICE);fail_exec=0;
        for(size_t i=0;i<(size_t)m*d*2;i++)EXPECT(((unsigned char *)y)[i]==0x23);
        co_graph_destroy(g);EXPECT(!live&&!resident);
    }
    desc.device_budget=p.device_bytes-1;EXPECT(co_graph_plan_init(&desc,&p)==ROCKET_E_TILING);
    desc.device_budget=0;
    for(unsigned offset=1;offset<=15;offset++) {
        fail_alloc=allocs+offset;co_graph *g=(void *)1;
        EXPECT(co_graph_compile(3,&desc,CO_CHAIN,&g)==ROCKET_E_NOMEM);
        EXPECT(!g&&!live&&!resident);fail_alloc=0;
    }
    free(x);free(wg);free(wu);free(wd);free(bias);free(gate);free(up);free(ref);free(y);
}
static void row_case(enum co_graph_op op,unsigned m,unsigned n) {
    _Float16 *x=malloc((size_t)m*n*2),*y=malloc((size_t)m*n*2),*gamma=malloc(n*2);
    EXPECT(x&&y&&gamma);if(!x||!y||!gamma)exit(1);
    for(unsigned c=0;c<n;c++)gamma[c]=(_Float16)(0.5+(c%5)*0.125);
    for(unsigned r=0;r<m;r++)for(unsigned c=0;c<n;c++) {
        double v=op==CO_GRAPH_RMSNORM?(r%3?((int)(c%17)-8)/8.0:0):
            (r%5==0?-INFINITY:r%5==1?1000:r%5==2?(c==n-1?0:-40):((int)(c%29)-14));
        x[(size_t)r*n+c]=(_Float16)v;
    }
    co_graph_node nodes[]={{.op=CO_GRAPH_INPUT,.rows=m,.cols=n,
        .flags=op==CO_GRAPH_SOFTMAX?CO_GRAPH_ALLOW_NEG_INF:0},
        {.op=op,.a=0,.gamma=gamma,.epsilon=1e-5f}};
    co_graph_desc d={nodes,2,1,0};co_graph_plan p;EXPECT(!co_graph_plan_init(&d,&p));
    EXPECT(p.hybrid_nodes==1&&p.statistic_downloads==(size_t)m*((n+255)/256));
    for(unsigned mode=0;mode<2;mode++) {
        co_graph *g=NULL;EXPECT(!co_graph_compile(3,&d,(enum co_mode)mode,&g));if(!g)exit(1);
        EXPECT(resident==p.device_bytes);co_graph_input in={0,x};EXPECT(!co_graph_run(g,&in,1,y));
        for(unsigned r=0;r<m;r++) {
            double sum=0,max=-INFINITY;
            for(unsigned c=0;c<n;c++)if(x[(size_t)r*n+c]>max)max=x[(size_t)r*n+c];
            for(unsigned c=0;c<n;c++)sum+=op==CO_GRAPH_RMSNORM?
                (double)x[(size_t)r*n+c]*x[(size_t)r*n+c]:(max==-INFINITY?0:exp(x[(size_t)r*n+c]-max));
            double got_sum=0;
            for(unsigned c=0;c<n;c++) {
                double expected=op==CO_GRAPH_RMSNORM?x[(size_t)r*n+c]*gamma[c]/sqrt(sum/n+1e-5):
                    (sum?exp(x[(size_t)r*n+c]-max)/sum:0);
                EXPECT(isfinite((double)y[(size_t)r*n+c])&&fabs(y[(size_t)r*n+c]-expected)<0.004);
                got_sum+=y[(size_t)r*n+c];
            }
            if(op==CO_GRAPH_SOFTMAX)EXPECT(fabs(got_sum-(sum?1:0))<0.002);
        }
        _Float16 saved=x[0];x[0]=INFINITY;unsigned before=execs;
        EXPECT(co_graph_run(g,&in,1,y)==ROCKET_E_SHAPE&&execs==before);x[0]=saved;
        fail_exec=execs+2;memset(y,0x23,(size_t)m*n*2);
        EXPECT(co_graph_run(g,&in,1,y)==ROCKET_E_DEVICE);fail_exec=0;
        for(size_t i=0;i<(size_t)m*n*2;i++)EXPECT(((unsigned char *)y)[i]==0x23);
        co_graph_destroy(g);EXPECT(!live&&!resident);
    }
    unsigned bos=2+(op==CO_GRAPH_RMSNORM?3:5)+2*((n>256&&n%256)?2:1);
    for(unsigned offset=1;offset<=bos;offset++) {
        fail_alloc=allocs+offset;co_graph *g=(void *)1;
        EXPECT(co_graph_compile(3,&d,CO_CHAIN,&g)==ROCKET_E_NOMEM);
        EXPECT(!g&&!live&&!resident);fail_alloc=0;
    }
    if(op==CO_GRAPH_RMSNORM) {
        nodes[1].epsilon=0;EXPECT(co_graph_plan_init(&d,&p)==ROCKET_E_SHAPE);nodes[1].epsilon=1e-5f;
        nodes[0].flags=CO_GRAPH_ALLOW_NEG_INF;EXPECT(co_graph_plan_init(&d,&p)==ROCKET_E_SHAPE);
        nodes[0].flags=0;
        co_graph *overflow_graph=NULL;EXPECT(!co_graph_compile(3,&d,CO_CHAIN,&overflow_graph));
        memset(x,0,(size_t)m*n*2);x[0]=256;memset(y,0x23,(size_t)m*n*2);
        co_graph_input overflow_input={0,x};
        EXPECT(co_graph_run(overflow_graph,&overflow_input,1,y)==ROCKET_E_SHAPE);
        for(size_t i=0;i<(size_t)m*n*2;i++)EXPECT(((unsigned char *)y)[i]==0x23);
        co_graph_destroy(overflow_graph);
        gamma[0]=NAN;co_graph *g=(void *)1;
        EXPECT(co_graph_compile(3,&d,CO_CHAIN,&g)==ROCKET_E_SHAPE);EXPECT(!g&&!live);
    }
    free(x);free(y);free(gamma);
}
static void rope_math(const _Float16 *x,_Float16 *y,unsigned rows,unsigned heads,unsigned d,unsigned rd,
                      enum co_rope_style style,unsigned base,float theta) {
    for(unsigned t=0;t<rows;t++)for(unsigned h=0;h<heads;h++)for(unsigned i=0;i<d;i++) {
        size_t at=((size_t)t*heads+h)*d+i;
        if(!rd||i>=rd){y[at]=x[at];continue;}
        unsigned pair=style==CO_ROPE_INTERLEAVED?(i^1u):(i<rd/2?i+rd/2:i-rd/2);
        unsigned freq=style==CO_ROPE_INTERLEAVED?i/2:i%(rd/2);
        int neg=style==CO_ROPE_INTERLEAVED?!(i&1):i<rd/2;
        double angle=(base+t)*pow(theta,-2.0*freq/rd);
        double rotated=x[((size_t)t*heads+h)*d+pair]*(neg?-1:1);
        y[at]=(_Float16)(x[at]*cos(angle)+rotated*sin(angle));
    }
}
static void attention_math(unsigned n,unsigned hq,unsigned hv,unsigned d,unsigned rd,enum co_rope_style style,
                           unsigned base,unsigned window,const _Float16 *q,const _Float16 *k,const _Float16 *v,
                           const _Float16 *mask,_Float16 *y) {
    _Float16 *qr=malloc((size_t)n*hq*d*2),*kr=malloc((size_t)n*hv*d*2);double *p=malloc(n*sizeof(*p));
    EXPECT(qr&&kr&&p);if(!qr||!kr||!p)exit(1);
    rope_math(q,qr,n,hq,d,rd,style,base,10000);rope_math(k,kr,n,hv,d,rd,style,base,10000);
    for(unsigned t=0;t<n;t++)for(unsigned h=0;h<hq;h++) {
        unsigned hk=h/(hq/hv);double peak=-INFINITY;
        for(unsigned j=0;j<n;j++) {
            double dot=0;for(unsigned i=0;i<d;i++)dot+=(double)qr[((size_t)t*hq+h)*d+i]*kr[((size_t)j*hv+hk)*d+i];
            p[j]=(j>t||(window&&j+window<=t))?-INFINITY:dot/sqrt(d)+(mask?mask[(size_t)t*n+j]:0);
            if(p[j]>peak)peak=p[j];
        }
        double sum=0;for(unsigned j=0;j<n;j++){p[j]=peak==-INFINITY?0:exp(p[j]-peak);sum+=p[j];}
        for(unsigned i=0;i<d;i++) {
            double value=0;for(unsigned j=0;j<n;j++)value+=(sum?p[j]/sum:0)*v[((size_t)j*hv+hk)*d+i];
            y[((size_t)t*hq+h)*d+i]=(_Float16)value;
        }
    }
    free(qr);free(kr);free(p);
}
static void rope_case(unsigned m,unsigned heads,unsigned d,unsigned rd,enum co_rope_style style) {
    unsigned width=heads*d;size_t elements=(size_t)m*width;
    _Float16 *x=malloc(elements*2),*y=malloc(elements*2),*ref=malloc(elements*2);
    EXPECT(x&&y&&ref);if(!x||!y||!ref)exit(1);
    for(size_t i=0;i<elements;i++)x[i]=fill(i,29);
    rope_math(x,ref,m,heads,d,rd,style,17,10000);
    co_graph_node nodes[]={{.op=CO_GRAPH_INPUT,.rows=m,.cols=width},
        {.op=CO_GRAPH_ROPE,.a=0,.heads=heads,.head_dim=d,.rotary_dim=rd,.theta=10000,.rope_style=style,.position_base=17}};
    co_graph_desc desc={nodes,2,1,0};co_graph_plan plan;EXPECT(!co_graph_plan_init(&desc,&plan));
    for(unsigned mode=0;mode<2;mode++) {
        co_graph *g=NULL;EXPECT(!co_graph_compile(3,&desc,(enum co_mode)mode,&g));if(!g)exit(1);
        EXPECT(resident==plan.device_bytes);co_graph_input in={0,x};EXPECT(!co_graph_run(g,&in,1,y));
        for(size_t i=0;i<elements;i++)EXPECT(fabs((double)y[i]-ref[i])<0.001);
        co_graph_destroy(g);EXPECT(!live&&!resident);
    }
    nodes[1].rotary_dim=rd-1;EXPECT(co_graph_plan_init(&desc,&plan)==ROCKET_E_SHAPE);nodes[1].rotary_dim=rd;
    nodes[1].position_base=UINT32_MAX;EXPECT(co_graph_plan_init(&desc,&plan)==ROCKET_E_SHAPE);
    free(x);free(y);free(ref);
}
static void attention_case(enum co_rope_style style) {
    enum { N=7,HQ=4,HV=2,D=6,RD=4 };
    _Float16 q[N*HQ*D],k[N*HV*D],v[N*HV*D],mask[N*N],ref[N*HQ*D],y[N*HQ*D];
    for(unsigned i=0;i<N*HQ*D;i++)q[i]=fill(i,3);
    for(unsigned i=0;i<N*HV*D;i++){k[i]=fill(i,9);v[i]=fill(i,13);}
    for(unsigned t=0;t<N;t++)for(unsigned j=0;j<N;j++)mask[t*N+j]=t==2?-INFINITY:(j%3?0:-0.25f);
    attention_math(N,HQ,HV,D,RD,style,17,3,q,k,v,mask,ref);
    for(unsigned mode=0;mode<2;mode++) {
        co_attention_config cfg={.max_query=3,.capacity=N,.heads=HQ,.kv_heads=HV,.head_dim=D,
            .rotary_dim=RD,.theta=10000,.style=style,.causal=1,.window=3};
        co_attention_plan p;EXPECT(!co_attention_plan_init(&cfg,&p));co_attention *c=NULL;
        unsigned alloc_before=allocs;
        EXPECT(!co_attention_create(3,&cfg,(enum co_mode)mode,&c));if(!c)exit(1);
        unsigned bos=allocs-alloc_before;EXPECT(resident==p.device_bytes);EXPECT(!co_attention_reset(c,17));
        _Float16 stepmask[3*N];
        for(unsigned t=0;t<3;t++)for(unsigned j=0;j<3;j++)stepmask[t*3+j]=mask[t*N+j];
        EXPECT(!co_attention_step(c,3,q,k,v,stepmask,y));EXPECT(co_attention_cache_length(c)==3);
        _Float16 beforek[3*HV*D],beforev[3*HV*D],afterk[3*HV*D],afterv[3*HV*D];
        EXPECT(!co_attention_cache_export(c,beforek,beforev,3));
        for(unsigned t=0;t<3;t++)for(unsigned j=0;j<6;j++)stepmask[t*6+j]=mask[(t+3)*N+j];
        for(unsigned offset=1;offset<=7;offset+=6) {
            fail_exec=execs+offset;memset(y+3*HQ*D,0x23,3*HQ*D*2);
            EXPECT(co_attention_step(c,3,q+3*HQ*D,k+3*HV*D,v+3*HV*D,stepmask,y+3*HQ*D)==ROCKET_E_DEVICE);
            fail_exec=0;EXPECT(co_attention_cache_length(c)==3);
            EXPECT(!co_attention_cache_export(c,afterk,afterv,3));
            EXPECT(!memcmp(beforek,afterk,sizeof(beforek))&&!memcmp(beforev,afterv,sizeof(beforev)));
            for(unsigned i=0;i<3*HQ*D*2;i++)EXPECT(((unsigned char *)(y+3*HQ*D))[i]==0x23);
        }
        EXPECT(!co_attention_step(c,3,q+3*HQ*D,k+3*HV*D,v+3*HV*D,stepmask,y+3*HQ*D));
        EXPECT(!co_attention_step(c,1,q+6*HQ*D,k+6*HV*D,v+6*HV*D,mask+6*N,y+6*HQ*D));
        EXPECT(co_attention_cache_length(c)==N);
        for(unsigned i=0;i<N*HQ*D;i++)EXPECT(fabs((double)y[i]-ref[i])<0.003);
        EXPECT(co_attention_step(c,1,q,k,v,NULL,y)==ROCKET_E_SHAPE);
        EXPECT(co_attention_reset(c,UINT32_MAX)==ROCKET_E_SHAPE&&co_attention_cache_length(c)==N);
        _Float16 allk[N*HV*D],allv[N*HV*D],kref[N*HV*D];
        EXPECT(!co_attention_cache_export(c,allk,allv,N));EXPECT(!memcmp(allv,v,sizeof(v)));
        rope_math(k,kref,N,HV,D,RD,style,17,10000);
        for(unsigned i=0;i<N*HV*D;i++)EXPECT(fabs((double)allk[i]-kref[i])<0.001);
        EXPECT(!co_attention_reset(c,0)&&co_attention_cache_length(c)==0);
        co_attention_destroy(c);EXPECT(!live&&!resident);
        /* Every BO allocation failure must release the partially built contexts. */
        if(!mode)for(unsigned off=1;off<=bos;off++) {
            fail_alloc=allocs+off;c=(void *)1;
            EXPECT(co_attention_create(3,&cfg,CO_CHAIN,&c)==ROCKET_E_NOMEM);EXPECT(!c&&!live&&!resident);fail_alloc=0;
        }
        cfg.device_budget=p.device_bytes-1;EXPECT(co_attention_plan_init(&cfg,&p)==ROCKET_E_TILING);
    }
    co_graph_node nodes[]={{.op=CO_GRAPH_INPUT,.rows=N,.cols=HQ*D},
        {.op=CO_GRAPH_INPUT,.rows=N,.cols=HV*D},{.op=CO_GRAPH_INPUT,.rows=N,.cols=HV*D},
        {.op=CO_GRAPH_INPUT,.rows=N,.cols=N,.flags=CO_GRAPH_ALLOW_NEG_INF},
        {.op=CO_GRAPH_ATTENTION,.a=0,.b=1,.c=2,.mask_node=3,.has_mask=1,.heads=HQ,.kv_heads=HV,.head_dim=D,
         .rotary_dim=RD,.theta=10000,.rope_style=style,.position_base=17,.causal=1,.window=3},
        {.op=CO_GRAPH_ADD,.a=4,.b=0}};
    co_graph_desc desc={nodes,6,5,0};co_graph_plan gp;EXPECT(!co_graph_plan_init(&desc,&gp));
    co_graph *g=NULL;EXPECT(!co_graph_compile(3,&desc,CO_CHAIN,&g));if(!g)exit(1);
    EXPECT(resident==gp.device_bytes);co_graph_input inputs[]={{0,q},{1,k},{2,v},{3,mask}};
    EXPECT(!co_graph_run(g,inputs,4,y));
    for(unsigned i=0;i<N*HQ*D;i++)EXPECT(fabs((double)y[i]-(_Float16)(ref[i]+q[i]))<0.003);
    EXPECT(!co_graph_run(g,inputs,4,y));co_graph_destroy(g);EXPECT(!live&&!resident);
}
static void multiple_outputs_case(void) {
    _Float16 x[13*33],a[13*33],b[13*33],c[13*33];
    for(unsigned i=0;i<13*33;i++)x[i]=fill(i,3);
    co_graph_node nodes[]={{.op=CO_GRAPH_INPUT,.rows=13,.cols=33},
        {.op=CO_GRAPH_HSIG,.a=0},{.op=CO_GRAPH_ADD,.a=1,.b=0},{.op=CO_GRAPH_MUL,.a=1,.b=1}};
    co_graph_desc desc={nodes,4,1,0};unsigned ids[]={1,2,3};co_graph_plan p;
    EXPECT(!co_graph_plan_many(&desc,ids,3,&p));EXPECT(p.steps==4&&p.noutputs==3);
    for(unsigned i=1;i<4;i++)EXPECT(p.tensors[i].last==p.steps);
    co_graph *g=NULL;EXPECT(!co_graph_compile_many(3,&desc,ids,3,CO_CHAIN,&g));if(!g)exit(1);
    EXPECT(resident==p.device_bytes);co_graph_input input={0,x};co_graph_output outputs[]={{3,c},{1,a},{2,b}};
    EXPECT(!co_graph_run_many(g,&input,1,outputs,3));
    for(unsigned i=0;i<13*33;i++) {
        _Float16 v=(_Float16)(x[i]/6.0+0.5);
        EXPECT(a[i]==v&&b[i]==(_Float16)(v+x[i])&&c[i]==(_Float16)(v*v));
    }
    unsigned before=execs;outputs[0].data=a;
    EXPECT(co_graph_run_many(g,&input,1,outputs,3)==ROCKET_E_SHAPE&&execs==before);outputs[0].data=c;
    memset(a,0x23,sizeof(a));memset(b,0x23,sizeof(b));memset(c,0x23,sizeof(c));fail_exec=execs+p.chain_submits;
    EXPECT(co_graph_run_many(g,&input,1,outputs,3)==ROCKET_E_DEVICE);fail_exec=0;
    for(unsigned i=0;i<sizeof(a);i++)EXPECT(((unsigned char *)a)[i]==0x23&&((unsigned char *)b)[i]==0x23&&((unsigned char *)c)[i]==0x23);
    EXPECT(co_graph_run(g,&input,1,a)==ROCKET_E_SHAPE);co_graph_destroy(g);EXPECT(!live&&!resident);
    ids[2]=1;EXPECT(co_graph_plan_many(&desc,ids,3,&p)==ROCKET_E_SHAPE);
}
static void norm_ref(unsigned m,unsigned n,const _Float16 *x,const _Float16 *gamma,_Float16 *y) {
    for(unsigned r=0;r<m;r++) {
        double total=0;
        for(unsigned start=0;start<n;start+=256) {
            double partial=0;unsigned end=n-start>256?start+256:n;
            for(unsigned c=start;c<end;c++)partial+=(_Float16)(x[(size_t)r*n+c]*x[(size_t)r*n+c]);
            total+=(float)partial;
        }
        double inv=1/sqrt(total/n+1e-5);
        for(unsigned c=0;c<n;c++)y[(size_t)r*n+c]=(_Float16)(x[(size_t)r*n+c]*(_Float16)(inv*gamma[c]));
    }
}
static void decoder_case(void) {
    enum {N=7,DM=17,FF=33,HQ=4,HV=2,D=6,QW=HQ*D,KW=HV*D};
    _Float16 x[N*DM],n1[DM],n2[DM],wq[QW*DM],wk[KW*DM],wv[KW*DM],wo[DM*QW],wg[FF*DM],wu[FF*DM],wd[DM*FF];
    for(unsigned i=0;i<N*DM;i++)x[i]=fill(i,13)*4;
    for(unsigned i=0;i<DM;i++){n1[i]=1+fill(i,7);n2[i]=1+fill(i,11);}
    for(unsigned i=0;i<QW*DM;i++)wq[i]=fill(i,3);
    for(unsigned i=0;i<KW*DM;i++){wk[i]=fill(i,9);wv[i]=fill(i,19);}
    for(unsigned i=0;i<DM*QW;i++)wo[i]=fill(i,23);
    for(unsigned i=0;i<FF*DM;i++){wg[i]=fill(i,17)*2;wu[i]=fill(i,29)*2;}
    for(unsigned i=0;i<DM*FF;i++)wd[i]=fill(i,31)*2;
    co_decoder_weights weights={n1,n2,wq,wk,wv,wo,wg,wu,wd};
    _Float16 z[N*DM],q[N*QW],k[N*KW],v[N*KW],a[N*QW],r[N*DM],gate[N*FF],up[N*FF],ref[N*DM],y[N*DM],full[N*DM];
    norm_ref(N,DM,x,n1,z);linear_ref(N,DM,QW,z,wq,NULL,q);linear_ref(N,DM,KW,z,wk,NULL,k);linear_ref(N,DM,KW,z,wv,NULL,v);
    attention_math(N,HQ,HV,D,4,CO_ROPE_SPLIT_HALF,17,3,q,k,v,NULL,a);
    linear_ref(N,QW,DM,a,wo,NULL,r);for(unsigned i=0;i<N*DM;i++)r[i]=(_Float16)(r[i]+x[i]);
    norm_ref(N,DM,r,n2,z);linear_ref(N,DM,FF,z,wg,NULL,gate);linear_ref(N,DM,FF,z,wu,NULL,up);
    for(unsigned i=0;i<N*FF;i++)gate[i]=(_Float16)((_Float16)(gate[i]*(_Float16)(1/(1+exp(-(double)gate[i]))))*up[i]);
    linear_ref(N,FF,DM,gate,wd,NULL,ref);for(unsigned i=0;i<N*DM;i++)ref[i]=(_Float16)(ref[i]+r[i]);
    co_decoder_config cfg={.model_dim=DM,.hidden_dim=FF,.epsilon=1e-5f,
        .attention={.max_query=3,.capacity=N,.heads=HQ,.kv_heads=HV,.head_dim=D,.rotary_dim=4,
         .theta=10000,.style=CO_ROPE_SPLIT_HALF,.causal=1,.window=3}};
    co_decoder_plan p;EXPECT(!co_decoder_plan_init(&cfg,&p));EXPECT(p.geometries==2&&p.rows[1]==1&&p.pre[0].noutputs==3);
    for(unsigned mode=0;mode<2;mode++) {
        co_decoder *c=NULL;unsigned before_alloc=allocs;
        EXPECT(!co_decoder_create(3,&cfg,&weights,(enum co_mode)mode,&c));if(!c)exit(1);
        unsigned bos=allocs-before_alloc;EXPECT(resident==p.device_bytes);EXPECT(!co_decoder_reset(c,17));
        EXPECT(!co_decoder_step(c,3,x,NULL,y));EXPECT(co_decoder_cache_length(c)==3&&co_decoder_last_query_rows(c)==3);
        _Float16 ck[3*KW],cv[3*KW],ak[3*KW],av[3*KW];EXPECT(!co_decoder_cache_export(c,ck,cv,3));
        fail_exec=execs+p.pre[1].chain_submits+p.attention[1].chain_submits_per_step+p.post[1].chain_submits-1;
        memset(y+3*DM,0x23,DM*2);
        EXPECT(co_decoder_step(c,1,x+3*DM,NULL,y+3*DM)==ROCKET_E_DEVICE);fail_exec=0;
        EXPECT(co_decoder_cache_length(c)==3&&co_decoder_last_query_rows(c)==3&&co_decoder_cache_switches(c)==0);
        EXPECT(!co_decoder_cache_export(c,ak,av,3));EXPECT(!memcmp(ck,ak,sizeof(ck))&&!memcmp(cv,av,sizeof(cv)));
        for(unsigned i=0;i<DM*2;i++)EXPECT(((unsigned char *)(y+3*DM))[i]==0x23);
        EXPECT(!co_decoder_step(c,1,x+3*DM,NULL,y+3*DM));EXPECT(co_decoder_last_query_rows(c)==1&&co_decoder_cache_switches(c)==1);
        EXPECT(!co_decoder_step(c,3,x+4*DM,NULL,y+4*DM));EXPECT(co_decoder_cache_length(c)==N&&co_decoder_cache_switches(c)==2);
        for(unsigned i=0;i<N*DM;i++)EXPECT(fabs((double)y[i]-ref[i])<0.005);
        EXPECT(resident==p.device_bytes);
        co_decoder_destroy(c);EXPECT(!live&&!resident);
        if(!mode)for(unsigned off=1;off<=bos;off++) {
            fail_alloc=allocs+off;c=(void *)1;
            EXPECT(co_decoder_create(3,&cfg,&weights,CO_CHAIN,&c)==ROCKET_E_NOMEM);EXPECT(!c&&!live&&!resident);fail_alloc=0;
        }
    }
    cfg.attention.max_query=N;cfg.disable_one_row_decode=1;co_decoder *batch=NULL;
    EXPECT(!co_decoder_create(3,&cfg,&weights,CO_CHAIN,&batch));EXPECT(!co_decoder_reset(batch,17));
    EXPECT(!co_decoder_step(batch,N,x,NULL,full));
    for(unsigned i=0;i<N*DM;i++)EXPECT(fabs((double)full[i]-y[i])<0.004);
    co_decoder_destroy(batch);EXPECT(!live&&!resident);
    /* A real coefficient-overflow error after attention must roll back KV too. */
    for(unsigned i=0;i<DM;i++)n2[i]=65504;
    cfg.attention.max_query=3;cfg.disable_one_row_decode=0;co_decoder *bad=NULL;
    EXPECT(!co_decoder_create(3,&cfg,&weights,CO_CHAIN,&bad));memset(full,0x23,sizeof(full));
    EXPECT(co_decoder_step(bad,1,x,NULL,full)==ROCKET_E_SHAPE&&co_decoder_cache_length(bad)==0);
    for(unsigned i=0;i<sizeof(full);i++)EXPECT(((unsigned char *)full)[i]==0x23);
    co_decoder_destroy(bad);EXPECT(!live&&!resident);
    cfg.device_budget=p.device_bytes-1;EXPECT(co_decoder_plan_init(&cfg,&p)==ROCKET_E_TILING);
}
int main(void) {
    co_graph_node nodes[]={{.op=CO_GRAPH_ADD,.a=1,.b=2},
        {.op=CO_GRAPH_INPUT,.rows=13,.cols=33},{.op=CO_GRAPH_INPUT,.rows=13,.cols=33}};
    co_graph_desc desc={nodes,3,0,0};co_graph_plan p;
    EXPECT(!co_graph_plan_init(&desc,&p));EXPECT(p.order[2]==0&&p.inputs==2);
    _Float16 a[13*33],b[13*33],y[13*33];
    for(unsigned i=0;i<13*33;i++){a[i]=fill(i,2);b[i]=fill(i,9);}
    co_graph *g=NULL;EXPECT(!co_graph_compile(3,&desc,CO_CHAIN,&g));
    co_graph_input inputs[]={{2,b},{1,a}};
    EXPECT(!co_graph_run(g,inputs,2,y));
    for(unsigned i=0;i<13*33;i++)EXPECT(y[i]==(_Float16)(a[i]+b[i]));
    unsigned before=execs;inputs[0].node=1;
    EXPECT(co_graph_run(g,inputs,2,y)==ROCKET_E_SHAPE);EXPECT(execs==before);
    co_graph_destroy(g);EXPECT(!live&&!resident);
    nodes[2].cols=32;EXPECT(co_graph_plan_init(&desc,&p)==ROCKET_E_SHAPE);
    nodes[2].cols=33;nodes[1].op=CO_GRAPH_SILU;nodes[1].a=0;
    EXPECT(co_graph_plan_init(&desc,&p)==ROCKET_E_SHAPE);
    nodes[1].a=3;EXPECT(co_graph_plan_init(&desc,&p)==ROCKET_E_SHAPE);
    nodes[1].op=(enum co_graph_op)99;EXPECT(co_graph_plan_init(&desc,&p)==ROCKET_E_UNSUPPORTED);
    execution_case(13,257,289);execution_case(257,17,33);
    row_case(CO_GRAPH_RMSNORM,257,513);row_case(CO_GRAPH_SOFTMAX,257,513);
    row_case(CO_GRAPH_SOFTMAX,13,4096);
    rope_case(13,4,100,96,CO_ROPE_SPLIT_HALF);rope_case(257,16,19,18,CO_ROPE_INTERLEAVED);
    attention_case(CO_ROPE_INTERLEAVED);attention_case(CO_ROPE_SPLIT_HALF);
    multiple_outputs_case();decoder_case();
    printf("GRAPH host checks=%u failures=%u\n",checks,failures);return failures?1:0;
}
