// SPDX-License-Identifier: GPL-3.0-or-later
#include "co_attention.h"
#include "co_row_ops.h"
#include "co_tiled.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { A_INPUT,A_ROTATED,A_SCORE,A_PROB,A_OUTPUT,A_COUNT };
struct co_attention {
    int fd;enum co_mode mode;co_attention_config cfg;co_attention_plan plan;
    unsigned length;uint32_t position_base;
    co_buffer buffers[A_COUNT];co_view *views[A_COUNT];
    co_linear *qk,*pv;co_rowop *softmax;co_rope *rope;
    _Float16 *keys,*values,*wk,*wv,*head,*result;
};
int co_attention_plan_init(const co_attention_config *cfg,co_attention_plan *out) {
    if(!out)return ROCKET_E_SHAPE;
    memset(out,0,sizeof(*out));
    if(!cfg||!cfg->capacity||cfg->capacity>CO_MAX_K||!cfg->max_query||cfg->max_query>cfg->capacity||
       !cfg->head_dim||cfg->head_dim>256||!cfg->heads||cfg->heads>CO_MAX_N/cfg->head_dim||
       !cfg->kv_heads||cfg->heads%cfg->kv_heads||
       (cfg->causal!=0&&cfg->causal!=1)||cfg->window>cfg->capacity||(cfg->window&&!cfg->causal)||
       (cfg->scale!=0&&(!isfinite(cfg->scale)||cfg->scale<=0))||
       (cfg->style!=CO_ROPE_INTERLEAVED&&cfg->style!=CO_ROPE_SPLIT_HALF))return ROCKET_E_SHAPE;
    co_attention_plan p={0};unsigned m=cfg->max_query,n=cfg->capacity,d=cfg->head_dim;
    co_linear_plan qk,pv;co_row_plan sm;co_rope_plan rp={0};
    int rc=co_linear_plan_init_ex(&qk,m,d,n,0,CO_LINEAR_NONE,SIZE_MAX);if(rc)return rc;
    rc=co_linear_plan_init_ex(&pv,m,n,d,0,CO_LINEAR_NONE,SIZE_MAX);if(rc)return rc;
    rc=co_row_plan_init(&sm,CO_ROW_SOFTMAX,m,n);if(rc)return rc;
    if(cfg->rotary_dim) {
        co_rope_config r={m,1,d,cfg->rotary_dim,cfg->theta,cfg->style};
        rc=co_rope_plan_init(&r,&rp);if(rc)return rc;
    }
    size_t head,score;rc=co_tiles_size(m,d,&head);if(rc)return rc;
    rc=co_tiles_size(m,n,&score);if(rc)return rc;
    p.device_bytes=qk.device_bytes+pv.device_bytes+sm.device_bytes+rp.device_bytes+
        2*(head+CO_SLACK+CO_GUARD)+2*(score+CO_SLACK+CO_GUARD);
    if(cfg->rotary_dim)p.device_bytes+=head+CO_SLACK+CO_GUARD;
    p.host_bytes=2ull*n*cfg->kv_heads*d*2+2ull*n*d*2+(size_t)m*d*2+(size_t)m*cfg->heads*d*2+rp.host_bytes;
    unsigned mt=co_blocks(m),nt=co_blocks(n),rcs=qk.row_classes;
    p.tasks_per_step=(qk.tasks+pv.tasks+sm.tasks)*cfg->heads+rp.tasks*(cfg->heads+cfg->kv_heads);
    p.chain_submits_per_step=(qk.chain_submits+pv.chain_submits+sm.chain_submits)*cfg->heads+
        rp.chain_submits*(cfg->heads+cfg->kv_heads);
    p.upload_calls=(unsigned long long)cfg->heads*(mt+3*mt*nt)+2ull*cfg->kv_heads*rcs*nt;
    p.download_calls=(unsigned long long)cfg->heads*(mt+mt*nt);
    if(cfg->rotary_dim) {
        p.upload_calls+=cfg->kv_heads*mt+rp.upload_calls*(cfg->heads+cfg->kv_heads);
        p.download_calls+=cfg->kv_heads*mt;
    }
    p.host_activation_reads=3ull*cfg->heads*m*n+rp.host_activation_reads*(cfg->heads+cfg->kv_heads);
    p.score_writes=(size_t)cfg->heads*m*n;
    p.statistic_downloads=(size_t)cfg->heads*m*nt;
    p.activation_downloads=(size_t)m*d*(cfg->heads+(cfg->rotary_dim?cfg->kv_heads:0));
    p.dynamic_weight_elements=2ull*cfg->kv_heads*rcs*n*d;
    p.host_upload_elements=p.dynamic_weight_elements+(size_t)cfg->heads*m*d+3ull*cfg->heads*m*n;
    if(cfg->rotary_dim)p.host_upload_elements+=(size_t)cfg->kv_heads*m*d+rp.host_upload_elements*(cfg->heads+cfg->kv_heads);
    if(p.device_bytes>(cfg->device_budget?cfg->device_budget:CO_LINEAR_DEVICE_BUDGET))return ROCKET_E_TILING;
    *out=p;return 0;
}
void co_attention_destroy(co_attention *c) {
    if(!c)return;
    co_linear_destroy(c->qk);co_linear_destroy(c->pv);co_rowop_destroy(c->softmax);co_rope_destroy(c->rope);
    for(unsigned i=0;i<A_COUNT;i++){free(c->views[i]);co_buffer_free(&c->buffers[i]);}
    free(c->keys);free(c->values);free(c->wk);free(c->wv);free(c->head);free(c->result);free(c);
}
int co_attention_create(int fd,const co_attention_config *cfg,enum co_mode mode,co_attention **out) {
    if(!out)return ROCKET_E_SHAPE;
    *out=NULL;if(fd<0||(mode!=CO_CHAIN&&mode!=CO_SEPARATE))return ROCKET_E_SHAPE;
    co_attention_plan p;int rc=co_attention_plan_init(cfg,&p);if(rc)return rc;
    if(mode==CO_CHAIN&&!rocket_batched_submit_supported())return ROCKET_E_UNSUPPORTED;
    co_attention *c=calloc(1,sizeof(*c));if(!c)return ROCKET_E_NOMEM;
    c->fd=fd;c->cfg=*cfg;c->mode=mode;c->plan=p;
    if(!c->cfg.scale)c->cfg.scale=(float)(1/sqrt(cfg->head_dim));
    size_t cache=(size_t)cfg->capacity*cfg->kv_heads*cfg->head_dim;
    size_t weight=(size_t)cfg->capacity*cfg->head_dim;
    c->keys=calloc(cache,2);c->values=calloc(cache,2);c->wk=calloc(weight,2);c->wv=calloc(weight,2);
    c->head=malloc((size_t)cfg->max_query*cfg->head_dim*2);
    c->result=malloc((size_t)cfg->max_query*cfg->heads*cfg->head_dim*2);
    if(!c->keys||!c->values||!c->wk||!c->wv||!c->head||!c->result){rc=ROCKET_E_NOMEM;goto fail;}
    for(unsigned i=0;i<A_COUNT;i++) {
        if(i==A_ROTATED&&!cfg->rotary_dim)continue;
        unsigned cols=(i==A_SCORE||i==A_PROB)?cfg->capacity:cfg->head_dim;
        rc=co_tiles_alloc(fd,cfg->max_query,cols,&c->buffers[i],&c->views[i]);if(rc)goto fail;
    }
    rc=co_linear_create(fd,cfg->max_query,cfg->head_dim,cfg->capacity,c->wk,NULL,CO_LINEAR_NONE,mode,&c->qk);
    if(!rc)rc=co_linear_create(fd,cfg->max_query,cfg->capacity,cfg->head_dim,c->wv,NULL,CO_LINEAR_NONE,mode,&c->pv);
    if(!rc)rc=co_rowop_create(fd,CO_ROW_SOFTMAX,cfg->max_query,cfg->capacity,NULL,0,mode,&c->softmax);
    if(rc)goto fail;
    if(cfg->rotary_dim) {
        co_rope_config r={cfg->max_query,1,cfg->head_dim,cfg->rotary_dim,cfg->theta,cfg->style};
        rc=co_rope_create(fd,&r,mode,&c->rope);if(rc)goto fail;
    }
    *out=c;return 0;
fail:co_attention_destroy(c);return rc;
}
int co_attention_reset(co_attention *c,uint32_t base) {
    if(!c||base>UINT32_MAX-(c->cfg.capacity+c->cfg.max_query-2))return ROCKET_E_SHAPE;
    size_t count=(size_t)c->cfg.capacity*c->cfg.kv_heads*c->cfg.head_dim;
    memset(c->keys,0,count*2);memset(c->values,0,count*2);c->length=0;c->position_base=base;return 0;
}
unsigned co_attention_cache_length(const co_attention *c){return c?c->length:0;}
int co_attention_cache_export(const co_attention *c,_Float16 *k,_Float16 *v,unsigned cap) {
    if(!c||cap<c->length)return ROCKET_E_SHAPE;
    if(!c->length)return 0;
    size_t bytes=(size_t)c->length*c->cfg.kv_heads*c->cfg.head_dim*2;
    if(!k||!v)return ROCKET_E_SHAPE;
    uintptr_t ka=(uintptr_t)k,va=(uintptr_t)v;
    if(ka>UINTPTR_MAX-bytes||va>UINTPTR_MAX-bytes||(ka<va+bytes&&va<ka+bytes))return ROCKET_E_SHAPE;
    memcpy(k,c->keys,bytes);memcpy(v,c->values,bytes);return 0;
}
static int finite_vector(const _Float16 *x,size_t count,int mask) {
    if(!x)return 0;
    for(size_t i=0;i<count;i++)if(!isfinite((double)x[i])&&!(mask&&x[i]==-INFINITY))return 0;
    return 1;
}
static void gather_head(co_attention *c,const _Float16 *x,unsigned tokens,unsigned heads,unsigned h) {
    unsigned d=c->cfg.head_dim;
    memset(c->head,0,(size_t)c->cfg.max_query*d*2);
    for(unsigned t=0;t<tokens;t++)memcpy(c->head+(size_t)t*d,x+((size_t)t*heads+h)*d,d*2);
}
static int upload_head(co_attention *c) {
    unsigned d=c->cfg.head_dim;
    for(unsigned mi=0;mi<co_blocks(c->cfg.max_query);mi++) {
        int rc=co_upload_f16(&c->views[A_INPUT][mi],c->head+(size_t)mi*256*d);if(rc)return rc;
    }
    return 0;
}
static int download_head(co_attention *c,unsigned which) {
    unsigned d=c->cfg.head_dim;
    for(unsigned mi=0;mi<co_blocks(c->cfg.max_query);mi++) {
        int rc=co_download(&c->views[which][mi],c->head+(size_t)mi*256*d);if(rc)return rc;
    }
    return 0;
}
static int mask_scores(co_attention *c,unsigned tokens,unsigned old,const _Float16 *mask) {
    unsigned nt=co_blocks(c->cfg.capacity),length=old+tokens;
    for(unsigned mi=0;mi<co_blocks(c->cfg.max_query);mi++)for(unsigned ni=0;ni<nt;ni++) {
        co_view v=c->views[A_SCORE][mi*nt+ni];int rc=rocket_bo_prep(c->fd,&v.buffer->bo,1,2000000000ull);if(rc)return rc;
        _Float16 *data=(_Float16 *)((char *)v.buffer->bo.ptr+v.offset);
        for(unsigned r=0;r<v.tensor.rows;r++)for(unsigned col=0;col<v.tensor.cols;col++) {
            unsigned t=mi*256+r,key=ni*256+col;size_t at=co_index(&v.tensor,r,col);
            double score=data[at];
            int disabled=t>=tokens||key>=length||(c->cfg.causal&&key>old+t)||
                (c->cfg.window&&key+c->cfg.window<=old+t);
            if(disabled)data[at]=-INFINITY;
            else {
                double extra=mask?(double)mask[(size_t)t*length+key]:0;
                if(extra==-INFINITY)data[at]=-INFINITY;
                else {
                    double value=score*c->cfg.scale+extra;
                    data[at]=(_Float16)value;
                    if(!isfinite(value)||!isfinite((double)data[at]))rc=ROCKET_E_SHAPE;
                }
            }
        }
        int fini=rocket_bo_fini(c->fd,&v.buffer->bo);if(!rc)rc=fini;if(rc)return rc;
    }
    return 0;
}
static int update_head_weights(co_attention *c,unsigned hk,unsigned length) {
    unsigned d=c->cfg.head_dim,n=c->cfg.capacity,hv=c->cfg.kv_heads;
    memset(c->wk,0,(size_t)n*d*2);memset(c->wv,0,(size_t)n*d*2);
    for(unsigned t=0;t<length;t++)for(unsigned q=0;q<d;q++) {
        c->wk[(size_t)t*d+q]=c->keys[((size_t)t*hv+hk)*d+q];
        c->wv[(size_t)q*n+t]=c->values[((size_t)t*hv+hk)*d+q];
    }
    int rc=co_linear_update_weights(c->qk,c->wk,NULL);
    if(!rc)rc=co_linear_update_weights(c->pv,c->wv,NULL);
    return rc;
}
int co_attention_step(co_attention *c,unsigned tokens,const _Float16 *q,const _Float16 *k,
                      const _Float16 *v,const _Float16 *mask,_Float16 *out) {
    if(!c||!out||!tokens||tokens>c->cfg.max_query||tokens>c->cfg.capacity-c->length)return ROCKET_E_SHAPE;
    unsigned d=c->cfg.head_dim,hq=c->cfg.heads,hv=c->cfg.kv_heads,old=c->length,length=old+tokens;
    if(!finite_vector(q,(size_t)tokens*hq*d,0)||!finite_vector(k,(size_t)tokens*hv*d,0)||
       !finite_vector(v,(size_t)tokens*hv*d,0)||(mask&&!finite_vector(mask,(size_t)tokens*length,1)))return ROCKET_E_SHAPE;
    int rc=0;unsigned mt=co_blocks(c->cfg.max_query),nt=co_blocks(c->cfg.capacity);
    memcpy(c->values+(size_t)old*hv*d,v,(size_t)tokens*hv*d*2);
    for(unsigned hk=0;hk<hv;hk++) {
        if(c->rope) {
            gather_head(c,k,tokens,hv,hk);rc=upload_head(c);
            if(!rc)rc=co_rope_run(c->rope,c->views[A_INPUT],mt,c->views[A_ROTATED],mt,c->position_base+old);
            if(!rc)rc=download_head(c,A_ROTATED);
            if(rc)goto fail;
            for(unsigned t=0;t<tokens;t++)memcpy(c->keys+((size_t)(old+t)*hv+hk)*d,c->head+(size_t)t*d,d*2);
        } else for(unsigned t=0;t<tokens;t++)memcpy(c->keys+((size_t)(old+t)*hv+hk)*d,k+((size_t)t*hv+hk)*d,d*2);
    }
    for(unsigned h=0;h<hq;h++) {
        unsigned hk=h/(hq/hv);
        if(h%(hq/hv)==0){rc=update_head_weights(c,hk,length);if(rc)goto fail;}
        gather_head(c,q,tokens,hq,h);rc=upload_head(c);if(rc)goto fail;
        co_view *query=c->views[A_INPUT];
        if(c->rope) {
            rc=co_rope_run(c->rope,query,mt,c->views[A_ROTATED],mt,c->position_base+old);if(rc)goto fail;
            query=c->views[A_ROTATED];
        }
        rc=co_linear_run_tiled(c->qk,query,mt,c->views[A_SCORE],mt*nt);
        if(!rc)rc=mask_scores(c,tokens,old,mask);
        if(!rc)rc=co_rowop_run(c->softmax,c->views[A_SCORE],mt*nt,c->views[A_PROB],mt*nt);
        if(!rc)rc=co_linear_run_tiled(c->pv,c->views[A_PROB],mt*nt,c->views[A_OUTPUT],mt);
        if(!rc)rc=download_head(c,A_OUTPUT);
        if(rc)goto fail;
        for(unsigned t=0;t<tokens;t++)for(unsigned i=0;i<d;i++) {
            _Float16 value=c->head[(size_t)t*d+i];if(!isfinite((double)value)){rc=ROCKET_E_DEVICE;goto fail;}
            c->result[((size_t)t*hq+h)*d+i]=value;
        }
    }
    for(unsigned i=0;i<A_COUNT;i++)if(c->buffers[i].bo.handle){rc=co_guard_check(&c->buffers[i]);if(rc)goto fail;}
    c->length=length;memcpy(out,c->result,(size_t)tokens*hq*d*2);return 0;
fail:
    memset(c->keys+(size_t)old*hv*d,0,(size_t)tokens*hv*d*2);
    memset(c->values+(size_t)old*hv*d,0,(size_t)tokens*hv*d*2);return rc;
}

int co_attention_truncate(co_attention *c,unsigned length) {
    if(!c||length>c->length)return ROCKET_E_SHAPE;
    size_t stride=(size_t)c->cfg.kv_heads*c->cfg.head_dim;
    memset(c->keys+(size_t)length*stride,0,(c->length-length)*stride*2);
    memset(c->values+(size_t)length*stride,0,(c->length-length)*stride*2);
    c->length=length;return 0;
}
int co_attention_cache_clone(co_attention *dst,const co_attention *src) {
    if(!dst||!src||dst->cfg.capacity<src->length||dst->cfg.kv_heads!=src->cfg.kv_heads||
       dst->cfg.head_dim!=src->cfg.head_dim||dst->cfg.rotary_dim!=src->cfg.rotary_dim||
       (src->cfg.rotary_dim&&(dst->cfg.theta!=src->cfg.theta||dst->cfg.style!=src->cfg.style))||
       src->position_base>UINT32_MAX-(dst->cfg.capacity+dst->cfg.max_query-2))return ROCKET_E_SHAPE;
    if(dst==src)return 0;
    size_t stride=(size_t)src->cfg.kv_heads*src->cfg.head_dim;
    memset(dst->keys,0,(size_t)dst->cfg.capacity*stride*2);
    memset(dst->values,0,(size_t)dst->cfg.capacity*stride*2);
    memcpy(dst->keys,src->keys,(size_t)src->length*stride*2);
    memcpy(dst->values,src->values,(size_t)src->length*stride*2);
    dst->length=src->length;dst->position_base=src->position_base;return 0;
}
