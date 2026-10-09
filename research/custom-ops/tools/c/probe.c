// SPDX-License-Identifier: GPL-3.0-or-later
#include "custom_ops.h"
#include "npu_matmul.h"
#include "rocket_chain.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define CHECK(call) do { int status_=(call); if(status_) { \
    fprintf(stderr,"FAIL %s:%d %s = %d\n",__FILE__,__LINE__,#call,status_); rc=1;goto cleanup; } } while(0)
#define EXPECT(expr) do { if(!(expr)) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#expr); \
    rc=1;goto cleanup; } } while(0)
static unsigned checks, failures;
static uint32_t mix(uint32_t x) { x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16); }
static _Float16 value(unsigned seed,size_t i,double scale) {
    return (_Float16)(((int)(mix(seed^(uint32_t)i)%31)-15)*scale);
}
static uint64_t hash(const void *p,size_t n) {
    const unsigned char *s=p;uint64_t h=14695981039346656037ull;
    for(size_t i=0;i<n;i++){h^=s[i];h*=1099511628211ull;}return h;
}
static void artifact(const char *label,const void *p,size_t bytes) {
    const char *dir=getenv("CO_ARTIFACT_DIR");if(!dir)return;
    char path[1024];int len=snprintf(path,sizeof(path),"%s/%s.bin",dir,label);
    if(len<0||(size_t)len>=sizeof(path)){fprintf(stderr,"artifact path too long\n");exit(1);}
    FILE *f=fopen(path,"wb");if(!f){perror(path);exit(1);}
    if(fwrite(p,1,bytes,f)!=bytes||fclose(f)){fprintf(stderr,"artifact write failed\n");exit(1);}
}
static int score(const char *label,const _Float16 *g,const _Float16 *r,size_t n,double tol) {
    double max=0;size_t bad=0,nf=0;
    for(size_t i=0;i<n;i++) {
        double d=fabs((double)g[i]-(double)r[i]);
        if(!isfinite((double)g[i]))nf++;
        int differs=tol==0?memcmp(g+i,r+i,sizeof(*g))!=0:d>tol;
        if(!isfinite(d)||differs) {if(bad<3)printf("  %s i=%zu got=%g ref=%g\n",label,i,(double)g[i],(double)r[i]);bad++;}
        if(isfinite(d)&&d>max)max=d;
    }
    printf("CHECK %s n=%zu max_abs=%.9g tol=%.9g bad=%zu nonfinite=%zu hash=%016llx %s\n",
        label,n,max,tol,bad,nf,(unsigned long long)hash(g,n*2),bad?"FAIL":"PASS");
    checks++;if(bad)failures++;return bad?1:0;
}
static int make_view(int fd,co_buffer *b,co_view *v,co_tensor t) {
    int rc=co_buffer_alloc(fd,t.bytes,b);if(rc)return rc;
    rc=co_view_init(v,b,0,&t);if(rc)co_buffer_free(b);return rc;
}
static _Float16 hs(double x) { return (_Float16)fmin(1,fmax(0,x/6+0.5)); }
static void oracle_mm(unsigned m,unsigned k,unsigned n,const _Float16 *x,const _Float16 *w,_Float16 *o) {
    for(unsigned i=0;i<m;i++)for(unsigned j=0;j<n;j++) {
        double a=0;for(unsigned q=0;q<k;q++)a+=(double)x[(size_t)i*k+q]*w[(size_t)j*k+q];
        o[(size_t)i*n+j]=(_Float16)a;
    }
}

static int host(void) {
    int rc=0;co_tensor x,w,o;co_program p[2];
    CHECK(co_tensor_matrix(&x,29,37,CO_F16));
    CHECK(co_tensor_weight(&w,45,37));CHECK(co_tensor_matrix(&o,29,45,CO_F16));
    EXPECT(x.physical_rows==32&&x.physical_cols==64);
    /* Cross-check against the upstream indices, not a round trip through our own helper. */
    for(unsigned r=0;r<x.physical_rows;r++)for(unsigned c=0;c<x.physical_cols;c++)
        EXPECT(co_index(&x,r,c)==(size_t)feature_data(x.physical_cols,x.physical_rows,1,8,c+1,r+1,1));
    for(unsigned r=0;r<w.physical_rows;r++)for(unsigned c=0;c<w.physical_cols;c++)
        EXPECT(co_index(&w,r,c)==(size_t)weight_fp16(w.physical_cols,r+1,c+1));
    EXPECT(co_tensor_matrix(&x,0,32,CO_F16)<0);
    EXPECT(co_tensor_matrix(&x,257,32,CO_F16)<0);
    CHECK(co_tensor_matrix(&x,29,37,CO_F16));
    co_buffer bx={.bo={.handle=1,.dma_address=0x100000,.size=131072},.fd=3,.payload=131072};
    co_buffer bw=bx,bo=bx;bw.bo.handle=2;bw.bo.dma_address=0x200000;bo.bo.handle=3;bo.bo.dma_address=0x300000;
    co_view vx,vw,vo;
    CHECK(co_view_init(&vx,&bx,16,&x));CHECK(co_view_init(&vw,&bw,0,&w));CHECK(co_view_init(&vo,&bo,0,&o));
    EXPECT(co_view_init(&vx,&bx,1,&x)<0);
    EXPECT(co_view_init(&vx,&bx,bx.payload-16,&x)<0);
    CHECK(co_view_init(&vx,&bx,16,&x));
    co_tensor malformed=x;malformed.bytes=1;EXPECT(co_view_init(&vx,&bx,0,&malformed)<0);
    co_buffer high=bx;high.bo.dma_address=UINT32_MAX-15;
    EXPECT(co_view_init(&vx,&high,0,&x)<0);
    CHECK(co_view_init(&vx,&bx,16,&x));
    CHECK(co_emit_mm(&p[0],vx,vw,vo));
    EXPECT(co_emit_mm(&p[1],vx,vw,vx)<0);
    co_tensor padded_weight=w; padded_weight.physical_cols=96;
    padded_weight.bytes=(size_t)padded_weight.physical_rows*96*2;
    co_view mismatched_weight;
    CHECK(co_view_init(&mismatched_weight,&bw,0,&padded_weight));
    EXPECT(co_emit_mm(&p[1],vx,mismatched_weight,vo)<0);
    co_buffer b2=bo;b2.bo.handle=4;b2.bo.dma_address=0x400000;co_view v2;
    CHECK(co_view_init(&v2,&b2,0,&o));CHECK(co_emit_lut(&p[1],vo,v2,CO_HARDSIGMOID));
    uint64_t packed[CO_MAX_WORDS*2]={0};
    rocket_bo rb={.handle=5,.dma_address=0x500000,.ptr=packed,.size=sizeof(packed)};
    rocket_task_desc tasks[2];size_t next=rkt_chain_words(p[0].count);
    CHECK(rkt_chain_pack_at(&rb,tasks,0,0,next,p[0].words,p[0].count,p[1].count));
    CHECK(rkt_chain_pack_at(&rb,tasks,1,next,next,p[1].words,p[1].count,p[1].count));
    CHECK(rkt_chain_verify(&rb,tasks,2));
    /* Negative check: a wrong successor fetch length must be rejected. */
    tasks[1].regcmd_count-=2;EXPECT(rkt_chain_verify(&rb,tasks,2)<0);
    co_linear_hsig *invalid=(co_linear_hsig *)(uintptr_t)1;
    EXPECT(co_linear_hsig_create(-1,29,37,45,NULL,NULL,CO_CHAIN,&invalid)<0);
    EXPECT(invalid==NULL);
    EXPECT(co_linear_hsig_run(NULL,NULL,NULL)<0);
    co_linear_hsig_destroy(NULL);
    printf("HOST layout, ragged padding, address/view bounds, alias rejection, mixed-length chain PASS\n");
cleanup:return rc;
}

static int composite(int fd,unsigned m,unsigned k,unsigned n,unsigned seed,int chained) {
    int rc=0;co_tensor tx,tw,to,ti;co_buffer b[9]={0};co_view v[9];co_program p[3];
    size_t nx=(size_t)m*k,nw=(size_t)n*k,ny=(size_t)m*n;
    _Float16 *x=malloc(nx*2),*w=malloc(nw*2),*bias=malloc(n*2),*ref=malloc(ny*2),
        *aff=malloc(ny*2),*final=malloc(ny*2),*got=malloc(ny*2),*baseline=malloc(ny*2);
    EXPECT(x&&w&&bias&&ref&&aff&&final&&got&&baseline);
    CHECK(co_tensor_matrix(&tx,m,k,CO_F16));CHECK(co_tensor_weight(&tw,n,k));
    CHECK(co_tensor_matrix(&to,m,n,CO_F16));
    CHECK(co_tensor_weight(&ti,to.physical_cols,to.physical_cols));
    CHECK(make_view(fd,&b[0],&v[0],tx));CHECK(make_view(fd,&b[1],&v[1],tw));
    for(unsigned i=2;i<8;i++)CHECK(make_view(fd,&b[i],&v[i],to));
    CHECK(make_view(fd,&b[8],&v[8],ti));
    for(size_t i=0;i<nx;i++)x[i]=value(seed,i,1.0/16);
    for(size_t i=0;i<nw;i++)w[i]=value(seed+103,i,1.0/64);
    for(unsigned j=0;j<n;j++)bias[j]=value(seed+277,j,1.0/4);
    oracle_mm(m,k,n,x,w,ref);
    for(size_t i=0;i<ny;i++){aff[i]=(_Float16)((float)ref[i]+(float)bias[i%n]);final[i]=hs((double)aff[i]);}
    CHECK(co_upload_f16(&v[0],x));CHECK(co_upload_f16(&v[1],w));CHECK(co_bias_upload(&v[2],bias));
    CHECK(co_identity_upload(&v[8]));
    CHECK(co_emit_mm(&p[0],v[0],v[1],v[3]));
    CHECK(co_emit_ew(&p[1],v[3],v[2],v[8],v[4],0));
    CHECK(co_emit_lut(&p[2],v[4],v[5],CO_HARDSIGMOID));
    printf("COMPOSITE m=%u k=%u n=%u seed=%u mode=%s words=%u/%u/%u\n",m,k,n,seed,chained?"chain":"separate",p[0].count,p[1].count,p[2].count);
    if(!chained) {
        /* Stage oracle gates. CPU reads are for validation, followed by a fresh no-touch run. */
        for(unsigned i=3;i<=5;i++)CHECK(co_sentinel(&b[i]));
        CHECK(co_execute(fd,p,1,CO_SEPARATE));CHECK(co_download(&v[3],got));CHECK(score("matmul",got,ref,ny,0.004));
        CHECK(co_execute(fd,p+1,1,CO_SEPARATE));CHECK(co_download(&v[4],got));CHECK(score("bias",got,aff,ny,0.008));
        CHECK(co_execute(fd,p+2,1,CO_SEPARATE));CHECK(co_download(&v[5],got));CHECK(score("activation",got,final,ny,0.002));
        /* Explicit host round trip of both intermediate cubes. */
        CHECK(co_copy(&v[6],&v[3]));CHECK(co_emit_ew(&p[1],v[6],v[2],v[8],v[4],0));
        CHECK(co_execute(fd,p+1,1,CO_SEPARATE));CHECK(co_copy(&v[7],&v[4]));
        CHECK(co_emit_lut(&p[2],v[7],v[5],CO_HARDSIGMOID));CHECK(co_execute(fd,p+2,1,CO_SEPARATE));
        CHECK(co_download(&v[5],baseline));
        CHECK(co_emit_ew(&p[1],v[3],v[2],v[8],v[4],0));CHECK(co_emit_lut(&p[2],v[4],v[5],CO_HARDSIGMOID));
    } else {
        CHECK(co_execute(fd,p,3,CO_SEPARATE));CHECK(co_download(&v[5],baseline));
    }
    for(unsigned i=3;i<=5;i++)CHECK(co_sentinel(&b[i]));
    rocket_submit_counters_reset();
    CHECK(co_execute(fd,p,3,chained?CO_CHAIN:CO_SEPARATE));
    uint64_t submits=rocket_submit_ioctl_count(),tasks=rocket_submit_task_count();
    EXPECT(submits==(chained?1u:3u)&&tasks==3);
    CHECK(co_download(&v[5],got));CHECK(score("composite-oracle",got,final,ny,0.002));
    CHECK(score(chained?"chain-vs-separate":"resident-vs-host-roundtrip",got,baseline,ny,0));
    CHECK(co_download(&v[3],got));CHECK(score("resident-intermediate-mm",got,ref,ny,0.004));
    CHECK(co_download(&v[4],got));CHECK(score("resident-intermediate-bias",got,aff,ny,0.008));
    for(unsigned i=0;i<9;i++)CHECK(co_guard_check(&b[i]));
    char label[128];snprintf(label,sizeof(label),"composite-%u-%u-%u-%u-%s",m,k,n,seed,chained?"chain":"separate");
    CHECK(co_download(&v[5],got));artifact(label,got,ny*2);
    snprintf(label,sizeof(label),"input-%u-%u-%u-%u",m,k,n,seed);artifact(label,x,nx*2);
    snprintf(label,sizeof(label),"weights-%u-%u-%u-%u",m,k,n,seed);artifact(label,w,nw*2);
    snprintf(label,sizeof(label),"bias-%u-%u-%u-%u",m,k,n,seed);artifact(label,bias,n*2);
    for(unsigned i=0;i<3;i++){snprintf(label,sizeof(label),"regcmd-%u-%u-%u-%u-%u",m,k,n,seed,i);artifact(label,p[i].words,p[i].count*8);}
    printf("EXEC submits=%llu tasks=%llu guards=PASS intermediate_cpu_copy=0\n",(unsigned long long)submits,(unsigned long long)tasks);
cleanup:
    for(unsigned i=0;i<9;i++)co_buffer_free(&b[i]);
    free(x);free(w);free(bias);free(ref);free(aff);free(final);free(got);free(baseline);return rc;
}

static int independent_chain(int fd,unsigned seed,int homogeneous) {
    int rc=0;co_buffer b[9]={0};co_view v[9];co_program p[3];
    unsigned dims[3][3]={{4,32,32},{29,37,45},{64,128,64}};
    if(homogeneous)for(unsigned i=0;i<3;i++){dims[i][0]=64;dims[i][1]=128;dims[i][2]=64;}
    _Float16 *x[3]={0},*w[3]={0},*ref[3]={0},*got[3]={0},*baseline[3]={0};
    for(unsigned i=0;i<3;i++) {
        unsigned m=dims[i][0],k=dims[i][1],n=dims[i][2];co_tensor tx,tw,to;
        x[i]=malloc((size_t)m*k*2);w[i]=malloc((size_t)n*k*2);
        ref[i]=malloc((size_t)m*n*2);got[i]=malloc((size_t)m*n*2);baseline[i]=malloc((size_t)m*n*2);
        EXPECT(x[i]&&w[i]&&ref[i]&&got[i]&&baseline[i]);
        for(size_t j=0;j<(size_t)m*k;j++)x[i][j]=value(seed+i,j,1.0/16);
        if(homogeneous&&seed==157&&i==0)memset(x[i],0,(size_t)m*k*2);
        for(size_t j=0;j<(size_t)n*k;j++)w[i][j]=value(seed+i+997,j,1.0/64);
        oracle_mm(m,k,n,x[i],w[i],ref[i]);
        CHECK(co_tensor_matrix(&tx,m,k,CO_F16));CHECK(co_tensor_weight(&tw,n,k));CHECK(co_tensor_matrix(&to,m,n,CO_F16));
        CHECK(make_view(fd,&b[i*3],&v[i*3],tx));CHECK(make_view(fd,&b[i*3+1],&v[i*3+1],tw));
        CHECK(make_view(fd,&b[i*3+2],&v[i*3+2],to));CHECK(co_upload_f16(&v[i*3],x[i]));CHECK(co_upload_f16(&v[i*3+1],w[i]));
        CHECK(co_emit_mm(&p[i],v[i*3],v[i*3+1],v[i*3+2]));
    }
    CHECK(co_execute(fd,p,3,CO_SEPARATE));
    for(unsigned i=0;i<3;i++){CHECK(co_download(&v[i*3+2],baseline[i]));CHECK(co_sentinel(&b[i*3+2]));}
    rocket_submit_counters_reset();CHECK(co_execute(fd,p,3,CO_CHAIN));
    EXPECT(rocket_submit_ioctl_count()==1&&rocket_submit_task_count()==3);
    for(unsigned i=0;i<3;i++) {
        CHECK(co_download(&v[i*3+2],got[i]));size_t n=(size_t)dims[i][0]*dims[i][2];
        CHECK(score("independent-chain-oracle",got[i],ref[i],n,0.004));
        CHECK(score("independent-chain-vs-separate",got[i],baseline[i],n,0));
    }
    for(unsigned i=0;i<9;i++)CHECK(co_guard_check(&b[i]));
    printf("INDEPENDENT_CHAIN seed=%u homogeneous=%d independent inputs/weights outputs=PASS\n",seed,homogeneous);
cleanup:
    for(unsigned i=0;i<9;i++)co_buffer_free(&b[i]);
    for(unsigned i=0;i<3;i++){free(x[i]);free(w[i]);free(ref[i]);free(got[i]);free(baseline[i]);}return rc;
}

static int primitives(int fd) {
    int rc=0;co_tensor t,id,wt,red;co_buffer b[7]={0};co_view v[7];co_program p;
    const unsigned m=29,n=45;size_t nn=(size_t)m*n;
    _Float16 *a=malloc(nn*2),*operand=malloc(nn*2),*ref=malloc(nn*2),*got=malloc(nn*2);float *sum=malloc(m*4);
    EXPECT(a&&operand&&ref&&got&&sum);
    CHECK(co_tensor_matrix(&t,m,n,CO_F16));CHECK(co_tensor_weight(&id,t.physical_cols,t.physical_cols));
    CHECK(co_tensor_weight(&wt,16,n));CHECK(co_tensor_matrix(&red,m,1,CO_F32));
    for(unsigned i=0;i<4;i++)CHECK(make_view(fd,&b[i],&v[i],t));
    CHECK(make_view(fd,&b[4],&v[4],id));CHECK(make_view(fd,&b[5],&v[5],wt));CHECK(make_view(fd,&b[6],&v[6],red));
    for(size_t i=0;i<nn;i++){a[i]=value(19,i,1.0/8);operand[i]=value(331,i,1.0/8);}
    CHECK(co_upload_f16(&v[0],a));CHECK(co_upload_f16(&v[1],operand));CHECK(co_identity_upload(&v[4]));
    for(int mul=0;mul<2;mul++) {
        for(size_t i=0;i<nn;i++)ref[i]=(_Float16)(mul?(float)a[i]*(float)operand[i]:(float)a[i]+(float)operand[i]);
        CHECK(co_sentinel(&b[2]));CHECK(co_emit_ew(&p,v[0],v[1],v[4],v[2],mul));CHECK(co_execute(fd,&p,1,CO_SEPARATE));
        CHECK(co_download(&v[2],got));CHECK(score(mul?"ew-mul":"ew-add",got,ref,nn,0));
        artifact(mul?"ew-mul-output":"ew-add-output",got,nn*2);
    }
    artifact("ew-input-a",a,nn*2);artifact("ew-input-b",operand,nn*2);
    for(int kind=0;kind<2;kind++) {
        for(size_t i=0;i<nn;i++)a[i]=(_Float16)(-8+16.0*i/(nn-1));
        for(size_t i=0;i<nn;i++)ref[i]=kind?(_Float16)(1/(1+exp(-(double)a[i]))):hs((double)a[i]);
        CHECK(co_upload_f16(&v[0],a));CHECK(co_sentinel(&b[2]));CHECK(co_emit_lut(&p,v[0],v[2],kind?CO_SIGMOID:CO_HARDSIGMOID));
        CHECK(co_execute(fd,&p,1,CO_SEPARATE));CHECK(co_download(&v[2],got));CHECK(score(kind?"sigmoid":"hardsigmoid",got,ref,nn,0.002));
        artifact(kind?"sigmoid-output":"hardsigmoid-output",got,nn*2);
    }
    artifact("lut-input",a,nn*2);
    for(int gelu=0;gelu<2;gelu++) {
        for(size_t i=0;i<nn;i++) {
            double x=a[i],gate=gelu?0.5*(1+erf(x/sqrt(2.0))):1/(1+exp(-x));
            ref[i]=(_Float16)(x*gate);
        }
        CHECK(co_sentinel(&b[2]));CHECK(co_gated_activation(fd,v[0],v[2],gelu,CO_SEPARATE));
        CHECK(co_download(&v[2],got));CHECK(score(gelu?"gelu":"silu",got,ref,nn,0.02));
        CHECK(co_copy(&v[3],&v[2]));
        CHECK(co_sentinel(&b[2]));CHECK(co_gated_activation(fd,v[0],v[2],gelu,CO_CHAIN));
        CHECK(co_download(&v[2],got));CHECK(score(gelu?"gelu":"silu",got,ref,nn,0.02));
        CHECK(co_download(&v[3],operand));CHECK(score("gated-chain-vs-separate",got,operand,nn,0));
        artifact(gelu?"gelu-output":"silu-output",got,nn*2);
    }
    for(int rsqrt=0;rsqrt<2;rsqrt++) {
        for(size_t i=0;i<nn;i++) {
            a[i]=(_Float16)(rsqrt?0.25+3.75*i/(nn-1):-4+8.0*i/(nn-1));
            ref[i]=(_Float16)(rsqrt?1/sqrt((double)a[i]):exp((double)a[i]));
        }
        CHECK(co_upload_f16(&v[0],a));CHECK(co_sentinel(&b[2]));
        CHECK(co_emit_lut(&p,v[0],v[2],rsqrt?CO_RSQRT_BOUNDED:CO_EXP_BOUNDED));
        CHECK(co_execute(fd,&p,1,CO_SEPARATE));CHECK(co_download(&v[2],got));
        CHECK(score(rsqrt?"rsqrt-bounded":"exp-bounded",got,ref,nn,rsqrt?0.003:0.0625));
        artifact(rsqrt?"rsqrt-bounded-input":"exp-bounded-input",a,nn*2);
        artifact(rsqrt?"rsqrt-bounded-output":"exp-bounded-output",got,nn*2);
    }
    for(size_t i=0;i<nn;i++)a[i]=value(49,i,1.0/8);
    CHECK(co_upload_f16(&v[0],a));
    for(int mean=0;mean<2;mean++) {
        CHECK(co_ones_upload(&v[5],n,mean));CHECK(co_sentinel(&b[6]));CHECK(co_emit_mm(&p,v[0],v[5],v[6]));
        CHECK(co_execute(fd,&p,1,CO_SEPARATE));CHECK(co_download(&v[6],sum));
        double max=0;size_t bad=0;
        for(unsigned r=0;r<m;r++){double s=0;for(unsigned c=0;c<n;c++)s+=(double)a[(size_t)r*n+c];if(mean)s/=n;
            double d=fabs(sum[r]-s);if(!isfinite(d)||d>(mean?0.001:0.00001))bad++;if(isfinite(d)&&d>max)max=d;}
        printf("CHECK feature-%s rows=%u cols=%u max_abs=%.9g bad=%zu %s\n",mean?"mean":"sum",m,n,max,bad,bad?"FAIL":"PASS");
        checks++;if(bad)failures++;EXPECT(!bad);
        artifact(mean?"feature-mean-output":"feature-sum-output",sum,m*4);
    }
    artifact("feature-reduce-input",a,nn*2);
    for(unsigned i=0;i<7;i++)CHECK(co_guard_check(&b[i]));
cleanup:for(unsigned i=0;i<7;i++)co_buffer_free(&b[i]);free(a);free(operand);free(ref);free(got);free(sum);return rc;
}

static int pooling(int fd,int global) {
    int rc=0;const unsigned c=17,h=8,w=8,kh=global?8:2,kw=kh,oh=h/kh,ow=w/kw;
    co_tensor ti,to;co_buffer b[2]={0};co_view v[2];co_program p;
    _Float16 in[h*w*c],out[oh*ow*c],ref[oh*ow*c];
    CHECK(co_tensor_spatial(&ti,c,h,w));CHECK(co_tensor_spatial(&to,c,oh,ow));
    CHECK(make_view(fd,&b[0],&v[0],ti));CHECK(make_view(fd,&b[1],&v[1],to));
    for(unsigned i=0;i<h*w*c;i++)in[i]=value(199,i,1.0/8);
    CHECK(co_upload_f16(&v[0],in));
    for(int avg=0;avg<2;avg++) {
        for(unsigned y=0;y<oh;y++)for(unsigned x=0;x<ow;x++)for(unsigned ch=0;ch<c;ch++) {
            double a=avg?0:-INFINITY;
            for(unsigned dy=0;dy<kh;dy++)for(unsigned dx=0;dx<kw;dx++) {
                double q=in[((size_t)(y*kh+dy)*w+x*kw+dx)*c+ch];a=avg?a+q:fmax(a,q);
            }
            ref[((size_t)y*ow+x)*c+ch]=(_Float16)(avg?a/(kh*kw):a);
        }
        CHECK(co_sentinel(&b[1]));CHECK(co_emit_pool(&p,v[0],v[1],h,w,kh,kw,avg));
        CHECK(co_execute(fd,&p,1,CO_SEPARATE));CHECK(co_download(&v[1],out));
        CHECK(score(global?(avg?"global-avgpool":"global-maxpool"):(avg?"avgpool":"maxpool"),out,ref,oh*ow*c,avg?0.002:0));
        artifact(global?(avg?"global-avgpool-output":"global-maxpool-output"):(avg?"avgpool-output":"maxpool-output"),out,oh*ow*c*2);
        for(unsigned i=0;i<2;i++)CHECK(co_guard_check(&b[i]));
    }
    artifact("pool-input",in,sizeof(in));
cleanup:for(unsigned i=0;i<2;i++)co_buffer_free(&b[i]);return rc;
}

static int reformat(int fd) {
    int rc=0;co_tensor t;co_buffer b[2]={0};co_view v[2];co_program p;
    _Float16 in[512],out[512],ref[512];
    CHECK(co_tensor_linear(&t,512));
    CHECK(make_view(fd,&b[0],&v[0],t));CHECK(make_view(fd,&b[1],&v[1],t));
    for(unsigned i=0;i<512;i++)in[i]=(_Float16)(i+1); /* unique, nonzero and exactly representable */
    CHECK(co_upload_f16(&v[0],in));CHECK(co_sentinel(&b[1]));
    CHECK(co_emit_reformat_8x8x8(&p,v[0],v[1]));CHECK(co_execute(fd,&p,1,CO_SEPARATE));
    CHECK(co_download(&v[1],out));
    for(unsigned a=0;a<8;a++)for(unsigned b0=0;b0<8;b0++)for(unsigned c=0;c<8;c++)
        ref[(c*8+a)*8+b0]=in[(a*8+b0)*8+c];
    for(unsigned i=0;i<16;i++)printf("REFORMAT_SAMPLE %u %.0f\n",i,(double)out[i]);
    CHECK(score("reformat-perm-201",out,ref,512,0));
    CHECK(co_guard_check(&b[0]));CHECK(co_guard_check(&b[1]));
    artifact("reformat-input",in,sizeof(in));artifact("reformat-output",out,sizeof(out));
    artifact("reformat-regcmd",p.words,p.count*8);
cleanup:for(unsigned i=0;i<2;i++)co_buffer_free(&b[i]);return rc;
}

static int normalization(int fd) {
    int rc=0;const unsigned m=29,n=45;co_tensor t;co_buffer b[2]={0};co_view v[2];
    _Float16 x[m*n],got[m*n],ref[m*n],baseline[m*n],gamma[n],beta[n];
    CHECK(co_tensor_matrix(&t,m,n,CO_F16));
    CHECK(make_view(fd,&b[0],&v[0],t));CHECK(make_view(fd,&b[1],&v[1],t));
    for(unsigned i=0;i<m*n;i++)x[i]=value(977,i,1.0/8);
    for(unsigned i=0;i<n;i++){gamma[i]=(_Float16)(0.5+((i*13)%17)/16.0);beta[i]=value(879,i,1.0/16);}
    CHECK(co_upload_f16(&v[0],x));
    for(int ln=0;ln<2;ln++) {
        for(unsigned r=0;r<m;r++) {
            double mu=0,ss=0;
            for(unsigned c=0;c<n;c++){double z=x[r*n+c];mu+=z;ss+=z*z;}
            mu=ln?mu/n:0;double inv=1/sqrt(ss/n-mu*mu+1e-5);
            for(unsigned c=0;c<n;c++)ref[r*n+c]=(_Float16)(((double)x[r*n+c]-mu)*inv*gamma[c]+(ln?beta[c]:0));
        }
        for(int chain=0;chain<2;chain++) {
            CHECK(co_sentinel(&b[1]));
            CHECK(co_norm_hybrid(fd,v[0],v[1],gamma,ln?beta:NULL,1e-5,ln,chain?CO_CHAIN:CO_SEPARATE));
            CHECK(co_download(&v[1],got));
            CHECK(score(ln?"layernorm-hybrid":"rmsnorm-hybrid",got,ref,m*n,0.008));
            if(!chain)memcpy(baseline,got,sizeof(got));
            else CHECK(score("norm-chain-vs-separate",got,baseline,m*n,0));
            CHECK(co_guard_check(&b[0]));CHECK(co_guard_check(&b[1]));
        }
        artifact(ln?"layernorm-output":"rmsnorm-output",got,sizeof(got));
    }
    artifact("norm-input",x,sizeof(x));artifact("norm-gamma",gamma,sizeof(gamma));artifact("norm-beta",beta,sizeof(beta));
    printf("NORM host row-statistics/rsqrt + coefficient materialization; input cube stays resident\n");
cleanup:for(unsigned i=0;i<2;i++)co_buffer_free(&b[i]);return rc;
}

static int resident_operator(int fd,unsigned m,unsigned k,unsigned n) {
    int rc=0;co_linear_hsig *ctx=NULL;co_tensor tx,ty;
    co_buffer bx={0},by={0};co_view vx,vy;
    size_t nx=(size_t)m*k,nw=(size_t)n*k,ny=(size_t)m*n;
    _Float16 *x=malloc(nx*2),*w=malloc(nw*2),*bias=malloc(n*2),*ref=malloc(ny*2),
        *got=malloc(ny*2),*cube=malloc(ny*2),*baseline=malloc(ny*4*2);
    EXPECT(x&&w&&bias&&ref&&got&&cube&&baseline);
    CHECK(co_tensor_matrix(&tx,m,k,CO_F16));CHECK(co_tensor_matrix(&ty,m,n,CO_F16));
    CHECK(co_buffer_alloc(fd,tx.bytes+32,&bx));CHECK(co_view_init(&vx,&bx,16,&tx));
    CHECK(co_buffer_alloc(fd,ty.bytes+32,&by));CHECK(co_view_init(&vy,&by,16,&ty));
    for(size_t i=0;i<nw;i++)w[i]=value(889,i,1.0/64);
    for(unsigned i=0;i<n;i++)bias[i]=value(1779,i,1.0/4);
    unsigned tasks=2+(ty.physical_rows*ty.physical_cols>32768?2:1);
    for(unsigned mode=0;mode<2;mode++) {
        CHECK(co_linear_hsig_create(fd,m,k,n,w,bias,mode?CO_CHAIN:CO_SEPARATE,&ctx));
        for(unsigned rep=0;rep<4;rep++) {
            for(size_t i=0;i<nx;i++)x[i]=value(977+rep,i,1.0/16);
            oracle_mm(m,k,n,x,w,ref);
            for(size_t i=0;i<ny;i++)ref[i]=hs((double)(_Float16)((float)ref[i]+(float)bias[i%n]));
            rocket_submit_counters_reset();CHECK(co_linear_hsig_run(ctx,x,got));
            EXPECT(rocket_submit_ioctl_count()==(mode?1u:tasks)&&rocket_submit_task_count()==tasks);
            CHECK(score("resident-context-oracle",got,ref,ny,0.002));
            if(!mode)memcpy(baseline+(size_t)rep*ny,got,ny*2);
            else CHECK(score("resident-context-modes",got,baseline+(size_t)rep*ny,ny,0));
            CHECK(co_upload_f16(&vx,x));
            rocket_submit_counters_reset();CHECK(co_linear_hsig_run_cube(ctx,vx,vy));
            EXPECT(rocket_submit_ioctl_count()==(mode?1u:tasks)&&rocket_submit_task_count()==tasks);
            CHECK(co_download(&vy,cube));
            CHECK(score("resident-context-cube-oracle",cube,ref,ny,0.002));
            CHECK(score("resident-context-cube-vs-host",cube,got,ny,0));
            /* Offset views must leave the prefix and suffix inside each payload untouched. */
            for(unsigned j=0;j<2;j++) {
                co_buffer *b=j?&by:&bx;co_tensor *t=j?&ty:&tx;
                CHECK(rocket_bo_prep(fd,&b->bo,0,2000000000ull));
                unsigned char *p=b->bo.ptr;
                for(unsigned z=0;z<16;z++)EXPECT(p[z]==0&&p[16+t->bytes+z]==0);
                CHECK(rocket_bo_fini(fd,&b->bo));CHECK(co_guard_check(b));
            }
            char name[128];snprintf(name,sizeof(name),"resident-context-%u-%u-%u-%u-%u",m,k,n,mode,rep);
            artifact(name,got,ny*2);
        }
        rocket_submit_counters_reset();
        EXPECT(co_linear_hsig_run_cube(ctx,vx,vx)<0);
        _Float16 saved=x[0];x[0]=(_Float16)NAN;
        EXPECT(co_linear_hsig_run(ctx,x,got)<0);x[0]=saved;
        EXPECT(rocket_submit_ioctl_count()==0);
        co_linear_hsig_destroy(ctx);ctx=NULL;
    }
    printf("RESIDENT_OPERATOR m=%u k=%u n=%u changing-inputs=4 tasks=%u offset-views=PASS\n",m,k,n,tasks);
cleanup:
    co_linear_hsig_destroy(ctx);co_buffer_free(&bx);co_buffer_free(&by);
    free(x);free(w);free(bias);free(ref);free(got);free(cube);free(baseline);return rc;
}

int main(int argc,char **argv) {
    if(argc!=2){fprintf(stderr,"usage: %s host|composite|chain|primitives|pool|reformat|norm|operator|all\n",argv[0]);return 1;}
    if(!strcmp(argv[1],"host"))return host();
    int fd=rocket_open();if(fd<0){fprintf(stderr,"SKIP no NPU: %d\n",fd);return 77;}
    int rc=0,all=!strcmp(argv[1],"all");
    if(all||!strcmp(argv[1],"composite")) {
        const unsigned d[][3]={{4,32,32},{16,64,32},{29,37,45},{64,128,64}};
        for(unsigned i=0;i<4&&!rc;i++)rc=composite(fd,d[i][0],d[i][1],d[i][2],17+i,0);
    }
    if(!rc&&(all||!strcmp(argv[1],"chain"))) {
        for(unsigned i=0;i<8&&!rc;i++) {
            rc=independent_chain(fd,57+i,0);
            if(!rc)rc=independent_chain(fd,157+i,1);
            if(!rc)rc=composite(fd,i%2?29:64,i%2?37:128,i%2?45:64,67+i,1);
        }
    }
    if(!rc&&(all||!strcmp(argv[1],"primitives")))rc=primitives(fd);
    if(!rc&&(all||!strcmp(argv[1],"pool"))){rc=pooling(fd,0);if(!rc)rc=pooling(fd,1);}
    if(!rc&&(all||!strcmp(argv[1],"reformat")))rc=reformat(fd);
    if(!rc&&(all||!strcmp(argv[1],"norm")))rc=normalization(fd);
    if(!rc&&(all||!strcmp(argv[1],"operator"))) {
        const unsigned shapes[][3]={{1,1,1},{4,32,32},{29,37,45},{37,65,33},{1,256,1},{256,1,256}};
        for(unsigned i=0;i<6&&!rc;i++)rc=resident_operator(fd,shapes[i][0],shapes[i][1],shapes[i][2]);
    }
    if(!all&&strcmp(argv[1],"composite")&&strcmp(argv[1],"chain")&&strcmp(argv[1],"primitives")&&strcmp(argv[1],"pool")&&strcmp(argv[1],"reformat")&&strcmp(argv[1],"norm")&&strcmp(argv[1],"operator"))rc=1;
    rocket_close(fd);
    printf("SUMMARY checks=%u failures=%u status=%s\n",checks,failures,rc?"FAIL":"PASS");
    return rc;
}
