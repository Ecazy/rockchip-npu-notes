// SPDX-License-Identifier: GPL-3.0-or-later
#include "co_gdn_cpu.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif
struct co_gdn_cpu {unsigned dv,dk,cur;enum co_gdn_cpu_precision precision;void *s[2],*d;float k[256],q[256];int ready;};
const char *co_gdn_cpu_backend(void){
#ifdef __aarch64__
    return "aarch64-neon-f32-fma";
#else
    return "portable-f32-fma";
#endif
}
static int finite_h(const _Float16 *x,size_t n){if(!x)return 0;for(size_t i=0;i<n;i++)if(!isfinite((float)x[i]))return 0;return 1;}
void co_gdn_cpu_destroy(co_gdn_cpu *c){if(c){free(c->s[0]);free(c->s[1]);free(c->d);free(c);}}
int co_gdn_cpu_create(unsigned dv,unsigned dk,enum co_gdn_cpu_precision p,co_gdn_cpu **out){
    if(!out)return ROCKET_E_SHAPE;
    *out=NULL;if(!dv||!dk||dv>256||dk>256||(p!=CO_GDN_CPU_F16&&p!=CO_GDN_CPU_F32))return ROCKET_E_SHAPE;
    co_gdn_cpu *c=calloc(1,sizeof(*c));if(!c)return ROCKET_E_NOMEM;
    c->dv=dv;c->dk=dk;c->precision=p;size_t bytes=(size_t)dv*dk*(p==CO_GDN_CPU_F16?2:4);
    c->s[0]=malloc(bytes);c->s[1]=malloc(bytes);c->d=malloc(bytes);
    if(!c->s[0]||!c->s[1]||!c->d){co_gdn_cpu_destroy(c);return ROCKET_E_NOMEM;}*out=c;return 0;
}
int co_gdn_cpu_reset(co_gdn_cpu *c,const _Float16 *s){
    if(!c||!finite_h(s,(size_t)c->dv*c->dk))return ROCKET_E_SHAPE;
    size_t n=(size_t)c->dv*c->dk;
    if(c->precision==CO_GDN_CPU_F16)memcpy(c->s[0],s,n*2);
    else for(size_t i=0;i<n;i++)((float *)c->s[0])[i]=s[i];
    c->cur=0;c->ready=1;return 0;
}
int co_gdn_cpu_export(const co_gdn_cpu *c,void *s){
    if(!c||!s||!c->ready)return ROCKET_E_SHAPE;
    memcpy(s,c->s[c->cur],(size_t)c->dv*c->dk*(c->precision==CO_GDN_CPU_F16?2:4));return 0;
}
void co_gdn_cpu_scale_f16(const _Float16 *x,_Float16 *y,size_t n,_Float16 a){
    size_t i=0;
#ifdef __aarch64__
    for(;i+4<=n;i+=4){float32x4_t v=vcvt_f32_f16(vld1_f16((const float16_t *)(x+i)));
        vst1_f16((float16_t *)(y+i),vcvt_f16_f32(vmulq_n_f32(v,(float)a)));}
#endif
    for(;i<n;i++)y[i]=(_Float16)((float)x[i]*(float)a);
}
static float dot(const void *x,const float *k,unsigned n,int half){
    unsigned j=0;float sum=0;
#ifdef __aarch64__
    float32x4_t acc=vdupq_n_f32(0);
    for(;j+4<=n;j+=4){float32x4_t a=half?vcvt_f32_f16(vld1_f16((const float16_t *)x+j)):vld1q_f32((const float *)x+j);
        acc=vfmaq_f32(acc,a,vld1q_f32(k+j));}sum=vaddvq_f32(acc);
#endif
    for(;j<n;j++)sum=fmaf(half?(float)((const _Float16 *)x)[j]:((const float *)x)[j],k[j],sum);
    return sum;
}
int co_gdn_cpu_step(co_gdn_cpu *c,const _Float16 *k,const _Float16 *q,const _Float16 *v,_Float16 alpha,_Float16 beta,void *out){
    if(!c||!out||!c->ready||!finite_h(k,c->dk)||!finite_h(q,c->dk)||!finite_h(v,c->dv)||
       !isfinite((float)alpha)||!isfinite((float)beta)||alpha<0||alpha>1||beta<0||beta>1)return ROCKET_E_SHAPE;
    unsigned n=c->dk,half=c->precision==CO_GDN_CPU_F16,bytes=half?2:4;size_t count=(size_t)c->dv*n;
    for(unsigned j=0;j<n;j++){c->k[j]=k[j];c->q[j]=q[j];}
    if(half)co_gdn_cpu_scale_f16(c->s[c->cur],c->d,count,alpha);
    else {size_t j=0;float *d=c->d,*s=c->s[c->cur];
#ifdef __aarch64__
        for(;j+4<=count;j+=4)vst1q_f32(d+j,vmulq_n_f32(vld1q_f32(s+j),(float)alpha));
#endif
        for(;j<count;j++)d[j]=s[j]*(float)alpha;
    }
    float outs[256];
    for(unsigned r=0;r<c->dv;r++){
        const void *d=(char *)c->d+(size_t)r*n*bytes;
        void *sn=(char *)c->s[1-c->cur]+(size_t)r*n*bytes;
        float p=dot(d,c->k,n,half);if(half)p=(float)(_Float16)p;
        float residual=(float)v[r]-p;if(half)residual=(float)(_Float16)residual;
        float e=(float)beta*residual;if(half)e=(float)(_Float16)e;
        unsigned j=0;
#ifdef __aarch64__
        for(;j+4<=n;j+=4){float32x4_t dv=half?vcvt_f32_f16(vld1_f16((const float16_t *)d+j)):vld1q_f32((const float *)d+j);
            float32x4_t sv=vfmaq_n_f32(dv,vld1q_f32(c->k+j),e);
            if(half)vst1_f16((float16_t *)sn+j,vcvt_f16_f32(sv));else vst1q_f32((float *)sn+j,sv);
        }
#endif
        for(;j<n;j++){float val=fmaf(e,c->k[j],half?(float)((const _Float16 *)d)[j]:((const float *)d)[j]);
            if(half)((_Float16 *)sn)[j]=(_Float16)val;else ((float *)sn)[j]=val;}
        float y=dot(sn,c->q,n,half);outs[r]=half?(float)(_Float16)y:y;
        if(!isfinite(outs[r])){c->ready=0;return ROCKET_E_DEVICE;}
    }
    if(half)for(unsigned r=0;r<c->dv;r++)((_Float16 *)out)[r]=(_Float16)outs[r];
    else memcpy(out,outs,c->dv*4);
    c->cur=1-c->cur;return 0;
}
