// SPDX-License-Identifier: GPL-3.0-or-later
#include "co_decoder.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct geometry {
    unsigned rows;co_graph *pre,*post;co_attention *attention;
    _Float16 *x,*q,*k,*v,*context,*y;
};
struct co_decoder {
    co_decoder_config cfg;co_decoder_plan plan;
    struct geometry geometry[2];unsigned canonical,last_rows,switches;
};
static const unsigned pre_outputs[]={2,3,4};
static void build_graphs(const co_decoder_config *cfg,const co_decoder_weights *w,unsigned rows,
                         co_graph_node pre[5],co_graph_node post[11]) {
    unsigned q=cfg->attention.heads*cfg->attention.head_dim,kv=cfg->attention.kv_heads*cfg->attention.head_dim;
    co_graph_node a[]={
        {.op=CO_GRAPH_INPUT,.rows=rows,.cols=cfg->model_dim},
        {.op=CO_GRAPH_RMSNORM,.a=0,.gamma=w->norm1,.epsilon=cfg->epsilon},
        {.op=CO_GRAPH_LINEAR,.a=1,.cols=q,.weights=w->wq},
        {.op=CO_GRAPH_LINEAR,.a=1,.cols=kv,.weights=w->wk},
        {.op=CO_GRAPH_LINEAR,.a=1,.cols=kv,.weights=w->wv}};
    co_graph_node b[]={
        {.op=CO_GRAPH_INPUT,.rows=rows,.cols=cfg->model_dim},
        {.op=CO_GRAPH_INPUT,.rows=rows,.cols=q},
        {.op=CO_GRAPH_LINEAR,.a=1,.cols=cfg->model_dim,.weights=w->wo},
        {.op=CO_GRAPH_ADD,.a=2,.b=0},
        {.op=CO_GRAPH_RMSNORM,.a=3,.gamma=w->norm2,.epsilon=cfg->epsilon},
        {.op=CO_GRAPH_LINEAR,.a=4,.cols=cfg->hidden_dim,.weights=w->wgate},
        {.op=CO_GRAPH_LINEAR,.a=4,.cols=cfg->hidden_dim,.weights=w->wup},
        {.op=CO_GRAPH_SILU,.a=5},{.op=CO_GRAPH_MUL,.a=7,.b=6},
        {.op=CO_GRAPH_LINEAR,.a=8,.cols=cfg->model_dim,.weights=w->wdown},
        {.op=CO_GRAPH_ADD,.a=9,.b=3}};
    memcpy(pre,a,sizeof(a));memcpy(post,b,sizeof(b));
}
int co_decoder_plan_init(const co_decoder_config *cfg,co_decoder_plan *out) {
    if(!out)return ROCKET_E_SHAPE;
    memset(out,0,sizeof(*out));
    if(!cfg||!cfg->model_dim||cfg->model_dim>CO_MAX_N||!cfg->hidden_dim||cfg->hidden_dim>CO_MAX_N||
       !isfinite(cfg->epsilon)||cfg->epsilon<=0||!cfg->attention.causal||
       (cfg->disable_one_row_decode!=0&&cfg->disable_one_row_decode!=1))return ROCKET_E_SHAPE;
    co_decoder_plan p={0};co_attention_plan validation;
    int rc=co_attention_plan_init(&cfg->attention,&validation);if(rc)return rc;
    p.geometries=cfg->attention.max_query>1&&!cfg->disable_one_row_decode?2:1;
    p.rows[0]=cfg->attention.max_query;p.rows[1]=1;
    static const _Float16 placeholder=0;
    co_decoder_weights w={&placeholder,&placeholder,&placeholder,&placeholder,&placeholder,&placeholder,&placeholder,&placeholder,&placeholder};
    for(unsigned i=0;i<p.geometries;i++) {
        co_graph_node pre[5],post[11];build_graphs(cfg,&w,p.rows[i],pre,post);
        co_graph_desc a={pre,5,2,SIZE_MAX},b={post,11,10,SIZE_MAX};
        rc=co_graph_plan_many(&a,pre_outputs,3,&p.pre[i]);if(rc)return rc;
        rc=co_graph_plan_init(&b,&p.post[i]);if(rc)return rc;
        co_attention_config ac=cfg->attention;ac.max_query=p.rows[i];
        rc=co_attention_plan_init(&ac,&p.attention[i]);if(rc)return rc;
        p.device_bytes+=p.pre[i].device_bytes+p.post[i].device_bytes+p.attention[i].device_bytes;
        p.tasks[i]=p.pre[i].tasks+p.post[i].tasks+p.attention[i].tasks_per_step;
        p.chain_submits[i]=p.pre[i].chain_submits+p.post[i].chain_submits+p.attention[i].chain_submits_per_step;
        p.upload_calls[i]=p.pre[i].tensors[0].tiles+p.pre[i].host_upload_calls+p.attention[i].upload_calls+
            p.post[i].tensors[0].tiles+p.post[i].tensors[1].tiles+p.post[i].host_upload_calls;
        p.download_calls[i]=p.pre[i].statistic_download_calls+p.pre[i].activation_download_calls+
            p.attention[i].download_calls+p.post[i].statistic_download_calls+p.post[i].activation_download_calls+p.post[i].tensors[10].tiles;
        for(unsigned j=0;j<3;j++)p.download_calls[i]+=p.pre[i].tensors[pre_outputs[j]].tiles;
        p.score_elements[i]=(size_t)p.rows[i]*cfg->attention.capacity*cfg->attention.heads;
    }
    if(p.device_bytes>(cfg->device_budget?cfg->device_budget:CO_LINEAR_DEVICE_BUDGET))return ROCKET_E_TILING;
    *out=p;return 0;
}
void co_decoder_destroy(co_decoder *c) {
    if(!c)return;
    for(unsigned i=0;i<2;i++) {
        struct geometry *g=&c->geometry[i];co_graph_destroy(g->pre);co_graph_destroy(g->post);co_attention_destroy(g->attention);
        free(g->x);free(g->q);free(g->k);free(g->v);free(g->context);free(g->y);
    }
    free(c);
}
int co_decoder_create(int fd,const co_decoder_config *cfg,const co_decoder_weights *w,enum co_mode mode,co_decoder **out) {
    if(!out)return ROCKET_E_SHAPE;
    *out=NULL;
    if(fd<0||!w||!w->norm1||!w->norm2||!w->wq||!w->wk||!w->wv||!w->wo||!w->wgate||!w->wup||!w->wdown||
       (mode!=CO_CHAIN&&mode!=CO_SEPARATE))return ROCKET_E_SHAPE;
    co_decoder_plan p;int rc=co_decoder_plan_init(cfg,&p);if(rc)return rc;
    co_decoder *c=calloc(1,sizeof(*c));if(!c)return ROCKET_E_NOMEM;c->cfg=*cfg;c->plan=p;
    unsigned q=cfg->attention.heads*cfg->attention.head_dim,kv=cfg->attention.kv_heads*cfg->attention.head_dim;
    for(unsigned i=0;i<p.geometries;i++) {
        struct geometry *g=&c->geometry[i];g->rows=p.rows[i];
        size_t mb=(size_t)g->rows*cfg->model_dim*2,qb=(size_t)g->rows*q*2,kb=(size_t)g->rows*kv*2;
        g->x=malloc(mb);g->q=malloc(qb);g->k=malloc(kb);g->v=malloc(kb);g->context=malloc(qb);g->y=malloc(mb);
        if(!g->x||!g->q||!g->k||!g->v||!g->context||!g->y){rc=ROCKET_E_NOMEM;goto fail;}
        co_graph_node pre[5],post[11];build_graphs(cfg,w,g->rows,pre,post);
        co_graph_desc a={pre,5,2,SIZE_MAX},b={post,11,10,SIZE_MAX};
        rc=co_graph_compile_many(fd,&a,pre_outputs,3,mode,&g->pre);if(rc)goto fail;
        rc=co_graph_compile(fd,&b,mode,&g->post);if(rc)goto fail;
        co_attention_config ac=cfg->attention;ac.max_query=g->rows;
        rc=co_attention_create(fd,&ac,mode,&g->attention);if(rc)goto fail;
    }
    *out=c;return 0;
fail:co_decoder_destroy(c);return rc;
}
int co_decoder_reset(co_decoder *c,uint32_t base) {
    if(!c)return ROCKET_E_SHAPE;
    int rc=co_attention_reset(c->geometry[0].attention,base);if(rc)return rc;
    if(c->plan.geometries==2){rc=co_attention_reset(c->geometry[1].attention,base);if(rc)return rc;}
    c->canonical=0;c->last_rows=0;c->switches=0;return 0;
}
unsigned co_decoder_cache_length(const co_decoder *c){return c?co_attention_cache_length(c->geometry[c->canonical].attention):0;}
unsigned co_decoder_last_query_rows(const co_decoder *c){return c?c->last_rows:0;}
unsigned co_decoder_cache_switches(const co_decoder *c){return c?c->switches:0;}
int co_decoder_cache_export(const co_decoder *c,_Float16 *k,_Float16 *v,unsigned cap) {
    return c?co_attention_cache_export(c->geometry[c->canonical].attention,k,v,cap):ROCKET_E_SHAPE;
}
static int finite_vector(const _Float16 *x,size_t n,int mask) {
    if(!x)return 0;
    for(size_t i=0;i<n;i++)if(!isfinite((double)x[i])&&!(mask&&x[i]==-INFINITY))return 0;
    return 1;
}
int co_decoder_step(co_decoder *c,unsigned tokens,const _Float16 *x,const _Float16 *mask,_Float16 *y) {
    if(!c||!y||!tokens||tokens>c->cfg.attention.max_query)return ROCKET_E_SHAPE;
    unsigned old=co_decoder_cache_length(c);
    if(tokens>c->cfg.attention.capacity-old||!finite_vector(x,(size_t)tokens*c->cfg.model_dim,0)||
       (mask&&!finite_vector(mask,(size_t)tokens*(old+tokens),1)))return ROCKET_E_SHAPE;
    unsigned pick=tokens==1&&c->plan.geometries==2?1:0;struct geometry *g=&c->geometry[pick];
    if(pick!=c->canonical) {
        int rc=co_attention_cache_clone(g->attention,c->geometry[c->canonical].attention);if(rc)return rc;
    }
    memset(g->x,0,(size_t)g->rows*c->cfg.model_dim*2);memcpy(g->x,x,(size_t)tokens*c->cfg.model_dim*2);
    co_graph_input input={0,g->x};co_graph_output outputs[]={{2,g->q},{3,g->k},{4,g->v}};
    int rc=co_graph_run_many(g->pre,&input,1,outputs,3);if(rc)return rc;
    size_t qw=c->cfg.attention.heads*c->cfg.attention.head_dim;
    memset(g->context,0,(size_t)g->rows*qw*2);
    rc=co_attention_step(g->attention,tokens,g->q,g->k,g->v,mask,g->context);
    if(!rc) {
        co_graph_input post[]={{0,g->x},{1,g->context}};rc=co_graph_run(g->post,post,2,g->y);
    }
    if(rc) {
        int rollback=co_attention_truncate(g->attention,old);
        return rollback?rollback:rc;
    }
    if(pick!=c->canonical)c->switches++;
    c->canonical=pick;c->last_rows=g->rows;
    memcpy(y,g->y,(size_t)tokens*c->cfg.model_dim*2);return 0;
}
