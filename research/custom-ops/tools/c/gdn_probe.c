// SPDX-License-Identifier: GPL-3.0-or-later
#define _POSIX_C_SOURCE 200809L
#include "co_gdn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#ifdef __aarch64__
#include <arm_neon.h>
#define CPU_KIND "aarch64_neon_f16_storage_f32_acc"
#else
#define CPU_KIND "portable_f32_acc_not_SIMD_evidence"
#endif
#define CHECK(x) do{int z=(x);if(z){fprintf(stderr,"line %d rc=%d\n",__LINE__,z);rc=z;goto done;}}while(0)
#define NEED(x) do{if(!(x)){fprintf(stderr,"line %d: %s\n",__LINE__,#x);rc=1;goto done;}}while(0)
static unsigned uploads,downloads,fail_after_execute;
int __real_co_upload_f16(co_view *,const _Float16 *);
int __wrap_co_upload_f16(co_view *v,const _Float16 *x){uploads++;return __real_co_upload_f16(v,x);}
int __real_co_download(co_view *,void *);
int __wrap_co_download(co_view *v,void *x){downloads++;return __real_co_download(v,x);}
int __real_co_execute(int,const co_program *,unsigned,enum co_mode);
int __wrap_co_execute(int fd,const co_program *p,unsigned n,enum co_mode m){
    int rc=__real_co_execute(fd,p,n,m);
    return !rc&&fail_after_execute?ROCKET_E_DEVICE:rc;
}
typedef struct {co_buffer b;co_view v;} tensor;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
static int make(int fd,tensor *t,unsigned r,unsigned c,int weight){
    co_tensor d;int rc=weight?co_tensor_weight(&d,r,c):co_tensor_matrix(&d,r,c,CO_F16);
    if(!rc)rc=co_buffer_alloc(fd,d.bytes,&t->b);
    if(!rc)rc=co_view_init(&t->v,&t->b,0,&d);
    return rc;
}
static void fill(_Float16 *x,size_t n,unsigned seed,float scale){
    for(size_t i=0;i<n;i++){seed=1664525u*seed+1013904223u;x[i]=(_Float16)(((int)(seed>>16)-32768)/32768.0f*scale);}
}
static void cpu_gemv(const _Float16 *s,const _Float16 *k,float *out,unsigned dv,unsigned dk){
    for(unsigned r=0;r<dv;r++){
        unsigned j=0;float sum=0;
#ifdef __aarch64__
        float32x4_t a=vdupq_n_f32(0);
        for(;j+4<=dk;j+=4)a=vfmaq_f32(a,vcvt_f32_f16(vld1_f16((const float16_t *)(s+(size_t)r*dk+j))),vcvt_f32_f16(vld1_f16((const float16_t *)(k+j))));
        sum=vaddvq_f32(a);
#endif
        for(;j<dk;j++)sum+=(float)s[(size_t)r*dk+j]*(float)k[j];
        out[r]=sum;
    }
}
static int near(const _Float16 *y,const double *ref,size_t n,double *peak){
    *peak=0;
    for(size_t i=0;i<n;i++){
        double e=fabs((double)y[i]-ref[i]);if(e>*peak)*peak=e;
        if(!isfinite((double)y[i])||e>0.0003+0.006*fabs(ref[i]))return 1;
    }return 0;
}
/* Separate CPU packing from BO sync/copy. Upload is shared-memory CPU memcpy +
 * prep/fini; not a PCIe transfer or pure DMA measurement. Staging allocation is
 * outside timed interval and reused. */
static int upload(tensor *t,const _Float16 *src,_Float16 *packed,double *pack,double *copy){
    double start=now();memset(packed,0,t->v.tensor.bytes);
    for(unsigned r=0;r<t->v.tensor.rows;r++)for(unsigned c=0;c<t->v.tensor.cols;c++)
        packed[co_index(&t->v.tensor,r,c)]=src[(size_t)r*t->v.tensor.cols+c];
    *pack+=now()-start;start=now();
    int rc=rocket_bo_prep(t->b.fd,&t->b.bo,1,2000000000ull);
    if(!rc){memcpy(t->b.bo.ptr,packed,t->v.tensor.bytes);rc=rocket_bo_fini(t->b.fd,&t->b.bo);}
    *copy+=now()-start;return rc;
}
static int save(const char *root,unsigned dv,unsigned dk,const char *name,const void *p,size_t n){
    char path[1024];if(snprintf(path,sizeof(path),"%s/%ux%u-%s",root,dv,dk,name)>=(int)sizeof(path))return 1;
    FILE *f=fopen(path,"wb");if(!f)return 1;int rc=fwrite(p,1,n,f)!=n;if(fclose(f))rc=1;return rc;
}
static int benchmark(int fd,unsigned dv,unsigned dk,unsigned reps,enum co_mode mode){
    int rc=0;size_t ns=(size_t)dv*dk;
    _Float16 *s=calloc(ns,2),*bridge=calloc(ns,2),*k=calloc(dk*2,2),*q=calloc(dk,2),*y=calloc(dv*2,2),*expanded=calloc(ns*2,2),*stage=calloc(256*256,2);
    double *ref=calloc(ns+dv*2,sizeof(double));float *cy=calloc(dv,sizeof(float));
    tensor t[10]={0};co_program p[2];double peak=0;
    NEED(s&&bridge&&k&&q&&y&&expanded&&stage&&ref&&cy);
    fill(s,ns,17,0.25f);fill(q,dk,19,0.2f);
    CHECK(make(fd,&t[0],dv,dk,0));CHECK(make(fd,&t[1],1,dk,1));CHECK(make(fd,&t[2],dv,1,0));
    CHECK(make(fd,&t[3],2,dk,1));CHECK(make(fd,&t[4],dv,2,0));
    CHECK(make(fd,&t[5],1,dk,0));CHECK(make(fd,&t[6],dv,dk,1));CHECK(make(fd,&t[7],1,dv,0));
    CHECK(co_upload_f16(&t[0].v,s));
    printf("{\"kind\":\"geometry\",\"dv\":%u,\"dk\":%u,\"A_weight_bytes\":%zu,\"B_state_weight_bytes\":%zu,\"A_state_feature_bytes\":%zu,\"A_physical_mac\":%zu,\"B_physical_mac\":%zu}\n",dv,dk,t[1].v.tensor.bytes,t[6].v.tensor.bytes,t[0].v.tensor.bytes,(size_t)t[0].v.tensor.physical_rows*t[0].v.tensor.physical_cols*t[2].v.tensor.physical_cols,(size_t)t[5].v.tensor.physical_rows*t[5].v.tensor.physical_cols*t[7].v.tensor.physical_cols);
    /* Rotate route order each token. All routes use identical changing k and S;
     * S is made feature-resident before timing, the GDN-A starting condition.
     * B pays download+pack of that state on every iteration, never static weights. */
    for(unsigned rep=0;rep<reps+2;rep++){
        fill(k,dk,31+rep,0.2f);fill(s,ns,17+rep,0.25f);CHECK(co_upload_f16(&t[0].v,s));
        memcpy(k+dk,q,dk*2);
        for(unsigned r=0;r<dv;r++){ref[r]=0;ref[dv+r]=0;for(unsigned j=0;j<dk;j++){ref[r]+=(double)s[(size_t)r*dk+j]*k[j];ref[dv+r]+=(double)s[(size_t)r*dk+j]*q[j];}}
        for(unsigned order=0;order<5;order++){
            unsigned arm=(order+rep)%5;double pack=0,copy=0,read=0,execute=0,total=now(),st;
            rocket_submit_counters_reset();
            if(arm==0||arm==1){unsigned wi=arm?3:1,yi=arm?4:2;
                CHECK(upload(&t[wi],k,stage,&pack,&copy));
                CHECK(co_emit_mm(&p[0],t[0].v,t[wi].v,t[yi].v));st=now();CHECK(co_execute(fd,p,1,mode));execute=now()-st;
                st=now();CHECK(co_download(&t[yi].v,y));read=now()-st;
                total=now()-total;
                if(!arm)NEED(!near(y,ref,dv,&peak));
                else for(unsigned r=0;r<dv;r++){NEED(!near(y+2*r,ref+r,1,&peak));NEED(!near(y+2*r+1,ref+dv+r,1,&peak));}
            }else if(arm==2){
                st=now();CHECK(co_download(&t[0].v,bridge));read=now()-st;
                CHECK(upload(&t[6],bridge,stage,&pack,&copy));CHECK(upload(&t[5],k,stage,&pack,&copy));
                CHECK(co_emit_mm(&p[0],t[5].v,t[6].v,t[7].v));st=now();CHECK(co_execute(fd,p,1,mode));execute=now()-st;
                st=now();CHECK(co_download(&t[7].v,y));read+=now()-st;total=now()-total;NEED(!near(y,ref,dv,&peak));
            }else if(arm==4){
                for(unsigned j=0;j<2;j++){
                    CHECK(upload(&t[1],j?q:k,stage,&pack,&copy));
                    CHECK(co_emit_mm(&p[0],t[0].v,t[1].v,t[2].v));
                    st=now();CHECK(co_execute(fd,p,1,mode));execute+=now()-st;
                    st=now();CHECK(co_download(&t[2].v,y+j*dv));read+=now()-st;
                }
                total=now()-total;NEED(!near(y,ref,dv*2,&peak));
            }else{
                /* CPU-resident competitive baseline; separately time device
                 * bridge so residency boundaries can be compared explicitly. */
                st=now();
                for(unsigned batch=0;batch<64;batch++){
                    cpu_gemv(s,k,cy,dv,dk);
                    __asm__ volatile("" : : "r"(cy) : "memory");
                }
                execute=(now()-st)/64;total=execute;
                for(unsigned r=0;r<dv;r++)NEED(fabs(cy[r]-ref[r])<1e-5);
                st=now();CHECK(co_download(&t[0].v,bridge));read=now()-st;
            }
            if(rep>=2)printf("{\"kind\":\"gemv\",\"dv\":%u,\"dk\":%u,\"rep\":%u,\"arm\":\"%s\",\"pack_us\":%.3f,\"upload_sync_copy_us\":%.3f,\"read_or_bridge_us\":%.3f,\"execute_api_us\":%.3f,\"total_us\":%.3f,\"tasks\":%llu,\"submits\":%llu,\"cpu_kind\":\"%s\"}\n",dv,dk,rep-2,(const char *[]){"A","A2","B","C","A_pair"}[arm],pack,copy,read,execute,total,(unsigned long long)rocket_submit_task_count(),(unsigned long long)rocket_submit_ioctl_count(),CPU_KIND);
        }
    }
    /* Outer product: K=1 versus explicit same-shape EW operands. */
    for(unsigned i=0;i<10;i++)if(t[i].b.bo.handle)CHECK(co_guard_check(&t[i].b));
    for(unsigned i=0;i<10;i++)co_buffer_free(&t[i].b);
    CHECK(make(fd,&t[0],dv,1,0));CHECK(make(fd,&t[1],dk,1,1));CHECK(make(fd,&t[2],dv,dk,0));
    CHECK(make(fd,&t[3],dv,dk,0));CHECK(make(fd,&t[4],dv,dk,0));CHECK(make(fd,&t[5],dv,dk,0));
    unsigned pc=t[3].v.tensor.physical_cols;CHECK(make(fd,&t[6],pc,pc,1));CHECK(co_identity_upload(&t[6].v));
    for(unsigned rep=0;rep<reps+2;rep++){
        fill(y,dv,89+rep,0.4f);fill(k,dk,97+rep,0.3f);
        for(unsigned r=0;r<dv;r++)for(unsigned c=0;c<dk;c++)ref[(size_t)r*dk+c]=(double)y[r]*k[c];
        for(unsigned order=0;order<2;order++){
            unsigned arm=(rep+order)%2;double pack=0,copy=0,expand=0,exec=0,read=0,total=now(),st;
            if(!arm){CHECK(upload(&t[0],y,stage,&pack,&copy));CHECK(upload(&t[1],k,stage,&pack,&copy));CHECK(co_emit_mm(p,t[0].v,t[1].v,t[2].v));}
            else{
                st=now();for(unsigned r=0;r<dv;r++)for(unsigned c=0;c<dk;c++){expanded[(size_t)r*dk+c]=y[r];expanded[ns+(size_t)r*dk+c]=k[c];}expand=now()-st;
                CHECK(upload(&t[3],expanded,stage,&pack,&copy));CHECK(upload(&t[4],expanded+ns,stage,&pack,&copy));CHECK(co_emit_ew(p,t[3].v,t[4].v,t[6].v,t[5].v,1));
            }
            st=now();CHECK(co_execute(fd,p,1,mode));exec=now()-st;st=now();CHECK(co_download(&t[arm?5:2].v,bridge));read=now()-st;total=now()-total;
            NEED(!near(bridge,ref,ns,&peak));
            if(rep>=2)printf("{\"kind\":\"outer\",\"dv\":%u,\"dk\":%u,\"rep\":%u,\"arm\":\"%s\",\"expand_us\":%.3f,\"pack_us\":%.3f,\"upload_sync_copy_us\":%.3f,\"execute_api_us\":%.3f,\"read_us\":%.3f,\"total_us\":%.3f,\"upload_bytes\":%zu,\"logical_mac\":%zu,\"physical_conv_mac\":%zu}\n",dv,dk,rep-2,arm?"EW_expanded":"K1",expand,pack,copy,exec,read,total,arm?t[3].v.tensor.bytes+t[4].v.tensor.bytes:t[0].v.tensor.bytes+t[1].v.tensor.bytes,ns,(size_t)t[0].v.tensor.physical_rows*t[2].v.tensor.physical_cols*(arm?pc:32));
        }
    }
    for(unsigned i=0;i<10;i++)if(t[i].b.bo.handle)CHECK(co_guard_check(&t[i].b));
done:for(unsigned i=0;i<10;i++)co_buffer_free(&t[i].b);
    free(s);free(bridge);free(k);free(q);free(y);free(expanded);free(stage);free(ref);free(cy);return rc;
}
static int recurrence(int fd,const char *root,unsigned dv,unsigned dk,enum co_mode mode){
    int rc=0;enum {STEPS=32};size_t ns=(size_t)dv*dk;co_gdn *c=NULL;
    _Float16 *s=calloc(ns,2),*initial=calloc(ns,2),*k=calloc(STEPS*dk,2),*q=calloc(STEPS*dk,2),*v=calloc(STEPS*dv,2),*out=calloc(STEPS*dv,2),*states=calloc(STEPS*ns,2);
    _Float16 gates[STEPS*2];double *ref=calloc(ns,sizeof(double)),*o=calloc(dv,sizeof(double)),peak;
    NEED(s&&initial&&k&&q&&v&&out&&states&&ref&&o);
    fill(initial,ns,301,0.1f);fill(k,STEPS*dk,307,0.08f);fill(q,STEPS*dk,311,0.15f);fill(v,STEPS*dv,313,0.3f);
    memcpy(s,initial,ns*2);for(size_t i=0;i<ns;i++)ref[i]=initial[i];
    CHECK(co_gdn_create(fd,dv,dk,mode,&c));CHECK(co_gdn_reset(c,initial));
    CHECK(save(root,dv,dk,"initial.bin",initial,ns*2));CHECK(save(root,dv,dk,"k.bin",k,STEPS*dk*2));CHECK(save(root,dv,dk,"q.bin",q,STEPS*dk*2));CHECK(save(root,dv,dk,"v.bin",v,STEPS*dv*2));
    for(unsigned t=0;t<STEPS;t++){
        _Float16 a=(_Float16)(t%9==0?0:t%9==1?1:0.95f),b=(_Float16)(t%7==0?0:t%7==1?1:0.6f);gates[2*t]=a;gates[2*t+1]=b;
        /* Ideal FP64 recurrence, no reuse of device result as reference. */
        for(unsigned r=0;r<dv;r++){
            double sk=0;for(unsigned j=0;j<dk;j++)sk+=a*ref[(size_t)r*dk+j]*k[t*dk+j];
            double e=b*((double)v[t*dv+r]-sk);o[r]=0;
            for(unsigned j=0;j<dk;j++){ref[(size_t)r*dk+j]=a*ref[(size_t)r*dk+j]+e*k[t*dk+j];o[r]+=ref[(size_t)r*dk+j]*q[t*dk+j];}
        }
        uploads=downloads=0;rocket_submit_counters_reset();double st=now();CHECK(co_gdn_step(c,k+t*dk,q+t*dk,v+t*dv,a,b,out+t*dv));double elapsed=now()-st;
        NEED(uploads==6&&downloads==1);
        NEED(rocket_submit_task_count()==6&&rocket_submit_ioctl_count()==(mode==CO_CHAIN?1u:6u));
        NEED(!near(out+t*dv,o,dv,&peak));
        CHECK(co_gdn_export(c,s));NEED(!near(s,ref,ns,&peak));memcpy(states+t*ns,s,ns*2);
        printf("{\"kind\":\"recurrent\",\"dv\":%u,\"dk\":%u,\"token\":%u,\"total_us\":%.3f,\"state_peak_abs\":%.8g,\"tasks\":6,\"submits\":%u}\n",dv,dk,t,elapsed,peak,mode==CO_CHAIN?1:6);
    }
    /* Repeat without exporting any intermediate state. All tokens consume only
     * the prior device result; output equality checks catch stale state reuse. */
    CHECK(co_gdn_reset(c,initial));
    for(unsigned t=0;t<STEPS;t++){
        _Float16 got[256];uploads=downloads=0;
        CHECK(co_gdn_step(c,k+t*dk,q+t*dk,v+t*dv,gates[2*t],gates[2*t+1],got));
        NEED(uploads==6&&downloads==1);NEED(!memcmp(got,out+t*dv,dv*2));
    }
    CHECK(co_gdn_export(c,s));NEED(!memcmp(s,states+(STEPS-1)*ns,ns*2));
    printf("{\"kind\":\"resident_repeat\",\"dv\":%u,\"dk\":%u,\"steps\":%u,\"state_exports_during_steps\":0,\"bit_exact\":true}\n",dv,dk,STEPS);
    /* Rejected scalar must neither mutate state nor caller output. */
    _Float16 rejected[256];for(unsigned i=0;i<dv;i++)rejected[i]=123;
    NEED(co_gdn_step(c,k,q,v,NAN,1,rejected)==ROCKET_E_SHAPE);for(unsigned i=0;i<dv;i++)NEED(rejected[i]==123);
    CHECK(co_gdn_export(c,s));NEED(!memcmp(s,states+(STEPS-1)*ns,ns*2));
    /* Simulate a late completion error after hardware already wrote scratch/new
     * state. The API must poison the context, preserve caller output, and reset. */
    fail_after_execute=1;
    int failed=co_gdn_step(c,k,q,v,1,1,rejected);fail_after_execute=0;
    NEED(failed==ROCKET_E_DEVICE);for(unsigned i=0;i<dv;i++)NEED(rejected[i]==123);
    NEED(co_gdn_export(c,s)==ROCKET_E_SHAPE);
    NEED(co_gdn_step(c,k,q,v,1,1,rejected)==ROCKET_E_SHAPE);
    CHECK(co_gdn_reset(c,initial));CHECK(co_gdn_export(c,s));NEED(!memcmp(s,initial,ns*2));
    CHECK(save(root,dv,dk,"gates.bin",gates,sizeof(gates)));CHECK(save(root,dv,dk,"out.bin",out,STEPS*dv*2));CHECK(save(root,dv,dk,"states.bin",states,STEPS*ns*2));
done:co_gdn_destroy(c);free(s);free(initial);free(k);free(q);free(v);free(out);free(states);free(ref);free(o);return rc;
}
/* Research-only isolated alias diagnostic. Never used by co_gdn_step.
 * Start from a legal out-of-place program, rebind only the WDMA address and
 * execution metadata. A matching result qualifies only this exact geometry. */
static int inplace(int fd,unsigned dv,unsigned dk,enum co_mode mode){
    int rc=0;size_t n=(size_t)dv*dk;tensor t[4]={0};co_program p;
    _Float16 *s=calloc(n,2),*y=calloc(n,2),*e=calloc(dv,2),*k=calloc(dk,2);
    double *ref=calloc(n,sizeof(double)),peak=0;unsigned mismatches=0;
    NEED(s&&y&&e&&k&&ref);
    CHECK(make(fd,&t[0],dv,1,0));CHECK(make(fd,&t[1],dk,1,1));
    CHECK(make(fd,&t[2],dv,dk,0));CHECK(make(fd,&t[3],dv,dk,0));
    for(unsigned rep=0;rep<8;rep++){
        fill(s,n,41+rep,.2f);fill(e,dv,51+rep,.3f);fill(k,dk,61+rep,.2f);
        for(unsigned r=0;r<dv;r++)for(unsigned j=0;j<dk;j++)ref[(size_t)r*dk+j]=s[(size_t)r*dk+j]+(double)e[r]*k[j];
        CHECK(co_upload_f16(&t[0].v,e));CHECK(co_upload_f16(&t[1].v,k));CHECK(co_upload_f16(&t[2].v,s));
        NEED(co_emit_mm_acc(&p,t[0].v,t[1].v,t[2].v,t[2].v)==ROCKET_E_SHAPE);
        CHECK(co_emit_mm_acc(&p,t[0].v,t[1].v,t[2].v,t[3].v));
        CHECK(co_execute(fd,&p,1,mode));CHECK(co_download(&t[3].v,y));NEED(!near(y,ref,n,&peak));
        unsigned changed=0;
        for(unsigned i=0;i<p.count;i++)if((p.words[i]&0xffff)==0x4020){
            p.words[i]=(p.words[i]&0xffff00000000ffffull)|((uint64_t)t[2].b.bo.dma_address<<16);changed++;
        }
        NEED(changed==1);p.output=t[2].v;
        CHECK(co_execute(fd,&p,1,mode));CHECK(co_download(&t[2].v,y));
        if(near(y,ref,n,&peak))mismatches++;
        for(unsigned i=0;i<4;i++)CHECK(co_guard_check(&t[i].b));
    }
    printf("{\"kind\":\"inplace_diagnostic\",\"dv\":%u,\"dk\":%u,\"trials\":8,\"mismatches\":%u,\"production_enabled\":false}\n",dv,dk,mismatches);
done:for(unsigned i=0;i<4;i++)co_buffer_free(&t[i].b);free(s);free(y);free(e);free(k);free(ref);return rc;
}
static int scalar_checks(int fd,unsigned dv,unsigned dk,enum co_mode mode){
    int rc=0;size_t n=(size_t)dv*dk;tensor t[8]={0};co_program p;
    _Float16 *x=calloc(n,2),*coeff=calloc(n,2),*out=calloc(n,2),*w=calloc(256*256,2);
    double *ref=calloc(n,sizeof(double)),peak;
    NEED(x&&coeff&&out&&w&&ref);fill(x,n,409,.4f);
    for(unsigned i=0;i<5;i++)CHECK(make(fd,&t[i],dv,dk,0));
    unsigned pc=t[0].v.tensor.physical_cols;
    CHECK(make(fd,&t[5],dk,dk,1));CHECK(make(fd,&t[6],pc,pc,1));CHECK(make(fd,&t[7],pc,pc,1));
    CHECK(co_identity_upload(&t[6].v));
    for(unsigned j=0;j<pc;j++)w[(size_t)j*pc+j]=-1;
    CHECK(co_upload_f16(&t[7].v,w));CHECK(co_upload_f16(&t[0].v,x));
    const _Float16 scales[]={0,1,-.5,.9375};
    for(unsigned a=0;a<4;a++){
        memset(w,0,256*256*2);for(unsigned j=0;j<dk;j++)w[(size_t)j*dk+j]=scales[a];
        for(size_t i=0;i<n;i++){coeff[i]=scales[a];ref[i]=(double)x[i]*scales[a];}
        CHECK(co_upload_f16(&t[5].v,w));CHECK(co_upload_f16(&t[1].v,coeff));
        CHECK(co_emit_mm(&p,t[0].v,t[5].v,t[2].v));CHECK(co_execute(fd,&p,1,mode));CHECK(co_download(&t[2].v,out));NEED(!near(out,ref,n,&peak));
        CHECK(co_emit_ew(&p,t[0].v,t[1].v,t[6].v,t[3].v,1));CHECK(co_execute(fd,&p,1,mode));CHECK(co_download(&t[3].v,out));NEED(!near(out,ref,n,&peak));
        for(size_t i=0;i<n;i++)ref[i]=coeff[i]-(double)x[i];
        CHECK(co_emit_ew(&p,t[0].v,t[1].v,t[7].v,t[4].v,0));CHECK(co_execute(fd,&p,1,mode));CHECK(co_download(&t[4].v,out));NEED(!near(out,ref,n,&peak));
    }
    for(unsigned i=0;i<8;i++)CHECK(co_guard_check(&t[i].b));
    printf("{\"kind\":\"scalar_and_subtract\",\"dv\":%u,\"dk\":%u,\"scales\":4,\"diagonal_weight_bytes\":%zu,\"expanded_scalar_bytes\":%zu,\"native_scalar_broadcast_tested\":false}\n",dv,dk,t[5].v.tensor.bytes,t[1].v.tensor.bytes);
done:for(unsigned i=0;i<8;i++)co_buffer_free(&t[i].b);free(x);free(coeff);free(out);free(w);free(ref);return rc;
}
static int host(void){
    int rc=0;co_tensor s,w,o;CHECK(co_tensor_matrix(&s,19,37,CO_F16));CHECK(co_tensor_weight(&w,1,37));CHECK(co_tensor_matrix(&o,19,1,CO_F16));
    co_buffer b[3]={ {.bo={.handle=1,.dma_address=0x100000},.fd=3,.payload=s.bytes}, {.bo={.handle=2,.dma_address=0x200000},.fd=3,.payload=w.bytes}, {.bo={.handle=3,.dma_address=0x300000},.fd=3,.payload=o.bytes} };
    co_view vs,vw,vo;co_program p;CHECK(co_view_init(&vs,b,0,&s));CHECK(co_view_init(&vw,b+1,0,&w));CHECK(co_view_init(&vo,b+2,0,&o));CHECK(co_emit_mm(&p,vs,vw,vo));
    NEED(co_emit_mm(&p,vs,vw,vs)==ROCKET_E_SHAPE);
    /* Merged old-state GEMV needs rank-one correction to read updated state. */
    double state[6]={1,2,3,4,5,6},k[3]={.2,-.3,.4},q[3]={.4,.1,-.2};
    for(unsigned r=0;r<2;r++){
        double sk=0,sq=0,kq=0,direct=0,a=.8,b=.6,v=.2;
        for(unsigned j=0;j<3;j++){sk+=state[r*3+j]*k[j];sq+=state[r*3+j]*q[j];kq+=k[j]*q[j];}
        double e=b*(v-a*sk);for(unsigned j=0;j<3;j++)direct+=(a*state[r*3+j]+e*k[j])*q[j];
        NEED(fabs(direct-(a*sq+e*kq))<1e-12);
    }
    puts("GDN host geometry and merged-output identity PASS (no hardware claim)");
done:return rc;
}
int main(int argc,char **argv){
    if(argc==2&&!strcmp(argv[1],"host"))return host();
    if(argc==2&&!strcmp(argv[1],"inplace")){
        int fd=rocket_open();if(fd<0)return 77;int rc=0;
        const unsigned shapes[][2]={{1,1},{19,37},{32,32},{128,128},{256,256}};
        for(unsigned i=0;i<5&&!rc;i++)rc=inplace(fd,shapes[i][0],shapes[i][1],CO_CHAIN);
        rocket_close(fd);return rc?1:0;
    }
    if(argc!=4||(strcmp(argv[2],"chain")&&strcmp(argv[2],"separate"))){fprintf(stderr,"usage: %s host | NEW_RESULT_DIR chain|separate REPETITIONS\n",argv[0]);return 2;}
    char *end=NULL;unsigned long reps=strtoul(argv[3],&end,10);if(!end||*end||reps<1||reps>10000)return 2;
    if(mkdir(argv[1],0755))return 2;
    int fd=rocket_open();if(fd<0)return 77;int rc=0;enum co_mode mode=!strcmp(argv[2],"chain")?CO_CHAIN:CO_SEPARATE;
    const unsigned shapes[][2]={{1,1},{19,37},{32,32},{128,128},{256,256}};
    for(unsigned i=0;i<5&&!rc;i++){
        rc=scalar_checks(fd,shapes[i][0],shapes[i][1],mode);
        if(!rc)rc=benchmark(fd,shapes[i][0],shapes[i][1],(unsigned)reps,mode);
        if(!rc)rc=recurrence(fd,argv[1],shapes[i][0],shapes[i][1],mode);
    }
    rocket_close(fd);if(!rc)puts("{\"kind\":\"result\",\"status\":\"PASS\"}");return rc?1:0;
}
