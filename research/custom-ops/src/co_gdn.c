// SPDX-License-Identifier: GPL-3.0-or-later
#include "co_gdn.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
enum { S0,S1,D,P,R,E,O,V,WK,WQ,WO,WA,WB,WN,PU,WPU,WNEG,WU,ES,WS,AC,ID,NB };
struct co_gdn {
    int fd,ready;unsigned dv,dk,current;enum co_mode mode;co_gdn_config cfg;
    co_buffer b[NB];co_view v[NB];co_execute_workspace exec;
    _Float16 *coeff;size_t coeff_count;co_program programs[12];
    _Float16 kwork[512],vwork[256],tmp[256];
};
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
static int finite_array(const _Float16 *p,size_t n){if(!p)return 0;for(size_t i=0;i<n;i++)if(!isfinite((float)p[i]))return 0;return 1;}
static int diagonal(co_view *v,_Float16 scale){
    unsigned n=v->tensor.cols;_Float16 *w=calloc((size_t)n*n,2);if(!w)return ROCKET_E_NOMEM;
    for(unsigned i=0;i<n;i++)w[(size_t)i*n+i]=scale;
    int rc=co_upload_f16(v,w);free(w);return rc;
}
void co_gdn_destroy(co_gdn *c){if(c){co_execute_workspace_destroy(&c->exec);for(unsigned i=0;i<NB;i++)co_buffer_free(&c->b[i]);free(c->coeff);free(c);}}
int co_gdn_create(int fd,unsigned dv,unsigned dk,enum co_mode mode,co_gdn **out){
    co_gdn_config cfg={0,CO_GDN_SIX,CO_GDN_DIAGONAL};return co_gdn_create_ex(fd,dv,dk,mode,&cfg,out);
}
int co_gdn_create_ex(int fd,unsigned dv,unsigned dk,enum co_mode mode,const co_gdn_config *cfg,co_gdn **out){
    if(!out)return ROCKET_E_SHAPE;
    *out=NULL;
    if(fd<0||!dv||!dk||dv>256||dk>256||!cfg||(mode!=CO_CHAIN&&mode!=CO_SEPARATE)||
       (cfg->reuse!=0&&cfg->reuse!=1)||cfg->graph<CO_GDN_SIX||cfg->graph>CO_GDN_MERGED||
       cfg->alpha<CO_GDN_DIAGONAL||cfg->alpha>CO_GDN_NATIVE)return ROCKET_E_SHAPE;
    co_gdn *c=calloc(1,sizeof(*c));if(!c)return ROCKET_E_NOMEM;
    c->fd=fd;c->dv=dv;c->dk=dk;c->mode=mode;c->cfg=*cfg;
    co_tensor t;int rc=0;
    for(unsigned i=0;i<NB&&!rc;i++){
        /* Allocate only buffers used by the selected recipe; guard counts fair. */
        if((i==PU||i==WPU||i==WNEG||i==WU||i==ES||i==WS)&&cfg->graph!=CO_GDN_MERGED)continue;
        if(i==WK&&cfg->graph==CO_GDN_MERGED)continue;
        if(i==WQ&&cfg->graph==CO_GDN_MERGED)continue;
        if((i==P||i==WN)&&cfg->graph==CO_GDN_MERGED)continue;
        if((i==R||i==WB)&&cfg->graph==CO_GDN_BETA_FOLD)continue;
        if(i==WA&&cfg->alpha!=CO_GDN_DIAGONAL)continue;
        if((i==AC||i==ID)&&cfg->alpha!=CO_GDN_EXPANDED)continue;
        if(i<=D||i==AC)rc=co_tensor_matrix(&t,dv,dk,CO_F16);
        else if(i<=V||i==ES)rc=co_tensor_matrix(&t,dv,1,CO_F16);
        else if(i==WK||i==WQ)rc=co_tensor_weight(&t,1,dk);
        else if(i==WO)rc=co_tensor_weight(&t,dk,1);
        else if(i==WA)rc=co_tensor_weight(&t,dk,dk);
        else if(i==WB||i==WS)rc=co_tensor_weight(&t,1,1);
        else if(i==PU)rc=co_tensor_matrix(&t,dv,2,CO_F16);
        else if(i==WPU)rc=co_tensor_weight(&t,2,dk);
        else if(i==WNEG||i==WU)rc=co_tensor_weight(&t,1,2);
        else if(i==ID)rc=co_tensor_weight(&t,(dk+31)/32*32,(dk+31)/32*32);
        else rc=co_tensor_weight(&t,32,32);
        if(!rc)rc=co_buffer_alloc(fd,t.bytes,&c->b[i]);
        if(!rc)rc=co_view_init(&c->v[i],&c->b[i],0,&t);
    }
    if(!rc&&c->b[WN].bo.handle)rc=diagonal(&c->v[WN],-1);
    if(!rc&&c->b[ID].bo.handle)rc=co_identity_upload(&c->v[ID]);
    if(!rc&&cfg->graph==CO_GDN_MERGED){_Float16 a[2]={-1,0},b[2]={0,1};rc=co_upload_f16(&c->v[WNEG],a);if(!rc)rc=co_upload_f16(&c->v[WU],b);}
    c->coeff_count=(size_t)dk*dk;if((size_t)dv*dk>c->coeff_count)c->coeff_count=(size_t)dv*dk;
    if(!rc&&cfg->reuse){c->coeff=calloc(c->coeff_count,2);if(!c->coeff)rc=ROCKET_E_NOMEM;else rc=co_execute_workspace_init(fd,12,&c->exec);}
    if(rc){co_gdn_destroy(c);return rc;}*out=c;return 0;
}
int co_gdn_reset(co_gdn *c,const _Float16 *s){
    if(!c||!finite_array(s,(size_t)c->dv*c->dk))return ROCKET_E_SHAPE;
    c->ready=0;int rc=co_upload_f16(&c->v[S0],s);if(!rc){c->current=S0;c->ready=1;}return rc;
}
int co_gdn_export(co_gdn *c,_Float16 *s){if(!c||!s||!c->ready)return ROCKET_E_SHAPE;return co_download(&c->v[c->current],s);}
/* Direct BO packing, same as co_upload_f16: no extra staging copy. Instrument
 * prep/fini separately from scatter. Unprofiled legacy calls retain original seam. */
static int upload(co_view *v,const _Float16 *src,double *pack,co_gdn_profile *pr,int reuse){
    if(!pr&&!reuse)return co_upload_f16(v,src);
    double st=pr?now():0;int rc=rocket_bo_prep(v->buffer->fd,&v->buffer->bo,1,2000000000ull);
    if(pr)pr->bo_sync_copy_us+=now()-st;
    if(rc)return rc;
    st=pr?now():0;_Float16 *d=(_Float16 *)((char *)v->buffer->bo.ptr+v->offset);
    /* These immutable-shape input-only BOs start zeroed at create and are never
     * device outputs; every logical element is replaced, padding stays zero. */
    if(!reuse)memset(d,0,v->tensor.bytes);
    for(unsigned r=0;r<v->tensor.rows;r++)for(unsigned j=0;j<v->tensor.cols;j++)d[co_index(&v->tensor,r,j)]=src[(size_t)r*v->tensor.cols+j];
    if(pr)*pack+=now()-st;
    st=pr?now():0;rc=rocket_bo_fini(v->buffer->fd,&v->buffer->bo);
    if(pr)pr->bo_sync_copy_us+=now()-st;
    return rc;
}
static int diagonal_update(co_view *v,_Float16 a,co_gdn_profile *pr){
    double st=pr?now():0;int rc=rocket_bo_prep(v->buffer->fd,&v->buffer->bo,1,2000000000ull);
    if(pr)pr->bo_sync_copy_us+=now()-st;
    if(rc)return rc;
    st=pr?now():0;_Float16 *d=(_Float16 *)v->buffer->bo.ptr;
    /* Weight is zero-initialized once and input-only. Keep the static zeros;
     * update precisely the dynamic diagonal in the already packed layout. */
    for(unsigned j=0;j<v->tensor.cols;j++)d[co_index(&v->tensor,j,j)]=a;
    if(pr)pr->pack_coeff_us+=now()-st;
    st=pr?now():0;rc=rocket_bo_fini(v->buffer->fd,&v->buffer->bo);
    if(pr)pr->bo_sync_copy_us+=now()-st;
    return rc;
}
static int scale_programs(co_program *p,unsigned *count,co_view x,co_view y,_Float16 a){
    size_t elements=x.tensor.bytes/2;unsigned off=0;
    while(elements){unsigned part=elements>32768?32768:(unsigned)elements;co_tensor t;co_view xv,yv;
        int rc=co_tensor_linear(&t,part);if(rc)return rc;
        if((rc=co_view_init(&xv,x.buffer,x.offset+off*2,&t))||(rc=co_view_init(&yv,y.buffer,y.offset+off*2,&t)))return rc;
        rc=co_emit_scale_native(p+(*count)++,xv,yv,a);if(rc)return rc;off+=part;elements-=part;
    }return 0;
}
int co_gdn_step(co_gdn *c,const _Float16 *k,const _Float16 *q,const _Float16 *v,_Float16 a,_Float16 b,_Float16 *out){return co_gdn_step_profiled(c,k,q,v,a,b,out,NULL);}
int co_gdn_step_profiled(co_gdn *c,const _Float16 *k,const _Float16 *q,const _Float16 *v,_Float16 alpha,_Float16 beta,_Float16 *out,co_gdn_profile *pr){
    double total=pr?now():0;if(pr)memset(pr,0,sizeof(*pr));
    if(!c||!out||!c->ready||!finite_array(k,c->dk)||!finite_array(q,c->dk)||!finite_array(v,c->dv)||
       !isfinite((float)alpha)||!isfinite((float)beta)||alpha<0||alpha>1||beta<0||beta>1)return ROCKET_E_SHAPE;
    int rc=0;unsigned next=c->current==S0?S1:S0,count=0;co_program *p=c->programs;
    _Float16 *coeff=c->coeff;double st=pr?now():0,dummy=0;_Float16 kq=0;
#define DO(x) do{rc=(x);if(rc)goto fail;}while(0)
#define UP(i,x,field) DO(upload(&c->v[i],x,pr?&pr->field:&dummy,pr,c->cfg.reuse))
    if(!coeff){coeff=malloc(c->coeff_count*2);if(pr){pr->host_allocations++;pr->host_allocation_us+=now()-st;}if(!coeff){rc=ROCKET_E_NOMEM;goto fail;}}
    st=pr?now():0;
    const _Float16 *ki=k,*vi=v;
    if(c->cfg.graph==CO_GDN_BETA_FOLD){
        for(unsigned j=0;j<c->dk;j++)c->kwork[j]=(_Float16)((float)beta*(float)k[j]);
        for(unsigned r=0;r<c->dv;r++)c->vwork[r]=(_Float16)((float)beta*(float)v[r]);
        ki=c->kwork;vi=c->vwork;
    }
    if(c->cfg.graph==CO_GDN_MERGED){
        memcpy(c->kwork,k,c->dk*2);memcpy(c->kwork+c->dk,q,c->dk*2);
        float dot=0;for(unsigned j=0;j<c->dk;j++)dot=fmaf((float)k[j],(float)q[j],dot);kq=(_Float16)dot;
    }
    if(pr)pr->prepare_us+=now()-st;
    if(c->cfg.graph==CO_GDN_MERGED){UP(WPU,c->kwork,pack_k_us);UP(WS,&kq,pack_coeff_us);}
    else {UP(WK,ki,pack_k_us);UP(WQ,q,pack_q_us);}
    UP(WO,k,pack_k_us);UP(V,vi,pack_v_us);
    st=pr?now():0;
    if(c->cfg.alpha==CO_GDN_DIAGONAL&&!c->cfg.reuse){memset(coeff,0,(size_t)c->dk*c->dk*2);for(unsigned j=0;j<c->dk;j++)coeff[(size_t)j*c->dk+j]=alpha;}
    else if(c->cfg.alpha==CO_GDN_EXPANDED)for(size_t i=0;i<(size_t)c->dv*c->dk;i++)coeff[i]=alpha;
    if(pr)pr->prepare_us+=now()-st;
    if(c->cfg.alpha==CO_GDN_DIAGONAL){if(c->cfg.reuse)DO(diagonal_update(&c->v[WA],alpha,pr));else {UP(WA,coeff,pack_coeff_us);}}
    else if(c->cfg.alpha==CO_GDN_EXPANDED){UP(AC,coeff,pack_coeff_us);}
    if(c->cfg.graph!=CO_GDN_BETA_FOLD){UP(WB,&beta,pack_coeff_us);}
    st=pr?now():0;
    if(c->cfg.alpha==CO_GDN_DIAGONAL)DO(co_emit_mm(&p[count++],c->v[c->current],c->v[WA],c->v[D]));
    else if(c->cfg.alpha==CO_GDN_EXPANDED)DO(co_emit_ew(&p[count++],c->v[c->current],c->v[AC],c->v[ID],c->v[D],1));
    else DO(scale_programs(p,&count,c->v[c->current],c->v[D],alpha));
    if(c->cfg.graph==CO_GDN_MERGED){
        DO(co_emit_mm(&p[count++],c->v[D],c->v[WPU],c->v[PU]));
        DO(co_emit_mm_acc(&p[count++],c->v[PU],c->v[WNEG],c->v[V],c->v[R]));
    }else{
        DO(co_emit_mm(&p[count++],c->v[D],c->v[WK],c->v[P]));
        DO(co_emit_ew(&p[count++],c->v[P],c->v[V],c->v[WN],c->v[c->cfg.graph==CO_GDN_BETA_FOLD?E:R],0));
    }
    if(c->cfg.graph!=CO_GDN_BETA_FOLD)DO(co_emit_mm(&p[count++],c->v[R],c->v[WB],c->v[E]));
    DO(co_emit_mm_acc(&p[count++],c->v[E],c->v[WO],c->v[D],c->v[next]));
    if(c->cfg.graph==CO_GDN_MERGED){
        DO(co_emit_mm(&p[count++],c->v[E],c->v[WS],c->v[ES]));
        DO(co_emit_mm_acc(&p[count++],c->v[PU],c->v[WU],c->v[ES],c->v[O]));
    }else DO(co_emit_mm(&p[count++],c->v[next],c->v[WQ],c->v[O]));
    if(pr){pr->emit_us=now()-st;pr->tasks=count;}
    st=pr?now():0;
    if(!pr&&!c->cfg.reuse)DO(co_execute(c->fd,p,count,c->mode));
    else if(pr&&c->mode==CO_SEPARATE){
        for(unsigned i=0;i<count;i++){co_execute_profile ep;
            DO(co_execute_profiled(c->fd,p+i,1,CO_SEPARATE,c->cfg.reuse?&c->exec:NULL,&ep));
            pr->task_submit_wait_us[i]=ep.submit_wait_us;
            pr->executor.allocation_us+=ep.allocation_us;pr->executor.command_us+=ep.command_us;
            pr->executor.submit_wait_us+=ep.submit_wait_us;pr->executor.output_sync_us+=ep.output_sync_us;
        }
    }else DO(co_execute_profiled(c->fd,p,count,c->mode,c->cfg.reuse?&c->exec:NULL,pr?&pr->executor:NULL));
    if(pr)pr->execute_api_us=now()-st;
    st=pr?now():0;for(unsigned i=0;i<NB;i++)if(c->b[i].bo.handle)DO(co_guard_check(&c->b[i]));if(pr)pr->guard_us=now()-st;
    st=pr?now():0;DO(co_download(&c->v[O],c->tmp));if(pr)pr->download_us=now()-st;
    if(!finite_array(c->tmp,c->dv)){rc=ROCKET_E_DEVICE;goto fail;}
    memcpy(out,c->tmp,c->dv*2);c->current=next;
fail:
    st=pr?now():0;if(!c->cfg.reuse)free(coeff);if(pr){pr->host_allocation_us+=now()-st;pr->total_us=now()-total;}
    if(rc)c->ready=0;
    return rc;
#undef DO
#undef UP
}
