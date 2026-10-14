// SPDX-License-Identifier: GPL-3.0-or-later
#include "co_rope.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { P_ROT,P_COS,P_SIN,P_MUL0,P_MUL1,P_COUNT };
struct co_rope {
    int fd;enum co_mode mode;co_rope_config cfg;co_rope_plan plan;
    co_buffer scratch[P_COUNT],identity[2];co_view ids[2];
    _Float16 *input,*rot,*cos,*sin;double *frequency;
};
static unsigned blocks(unsigned n){return (n+255)/256;}
static unsigned extent(unsigned n,unsigned i){unsigned t=n-i*256;return t>256?256:t;}
int co_rope_plan_init(const co_rope_config *cfg,co_rope_plan *out) {
    if(!out)return ROCKET_E_SHAPE;
    memset(out,0,sizeof(*out));
    if(!cfg||!cfg->rows||cfg->rows>CO_MAX_M||!cfg->heads||!cfg->head_dim||cfg->head_dim>256||
       cfg->heads>CO_MAX_N/cfg->head_dim||!cfg->rotary_dim||cfg->rotary_dim>cfg->head_dim||
       cfg->rotary_dim%2||!isfinite(cfg->theta)||cfg->theta<=0||
       (cfg->style!=CO_ROPE_INTERLEAVED&&cfg->style!=CO_ROPE_SPLIT_HALF))return ROCKET_E_SHAPE;
    unsigned cols=cfg->heads*cfg->head_dim;
    co_rope_plan p={.cols=cols,.tiles=blocks(cfg->rows)*blocks(cols),.column_classes=(cols>256&&cols%256)?2:1};
    co_tensor t;int rc=co_tensor_matrix(&t,extent(cfg->rows,0),extent(cols,0),CO_F16);if(rc)return rc;
    p.device_bytes=P_COUNT*(t.bytes+CO_SLACK+CO_GUARD);
    for(unsigned cl=0;cl<p.column_classes;cl++) {
        rc=co_tensor_matrix(&t,extent(cfg->rows,0),cl?cols%256:extent(cols,0),CO_F16);if(rc)return rc;
        co_tensor id;rc=co_tensor_weight(&id,t.physical_cols,t.physical_cols);if(rc)return rc;
        p.device_bytes+=id.bytes+CO_SLACK+CO_GUARD;
    }
    p.host_bytes=(size_t)extent(cfg->rows,0)*cols*2+3*256u*256u*2+(size_t)cfg->rotary_dim/2*sizeof(double);
    p.tasks=3ull*p.tiles;p.chain_submits=p.tiles;p.upload_calls=3ull*p.tiles;
    p.host_activation_reads=(size_t)cfg->rows*cols;p.host_upload_elements=3*p.host_activation_reads;
    *out=p;return 0;
}
void co_rope_destroy(co_rope *c) {
    if(!c)return;
    for(unsigned i=0;i<P_COUNT;i++)co_buffer_free(&c->scratch[i]);
    for(unsigned i=0;i<2;i++)co_buffer_free(&c->identity[i]);
    free(c->input);free(c->rot);free(c->cos);free(c->sin);free(c->frequency);free(c);
}
int co_rope_create(int fd,const co_rope_config *cfg,enum co_mode mode,co_rope **out) {
    if(!out)return ROCKET_E_SHAPE;
    *out=NULL;if(fd<0||(mode!=CO_CHAIN&&mode!=CO_SEPARATE))return ROCKET_E_SHAPE;
    co_rope_plan plan;int rc=co_rope_plan_init(cfg,&plan);if(rc)return rc;
    if(mode==CO_CHAIN&&!rocket_batched_submit_supported())return ROCKET_E_UNSUPPORTED;
    co_rope *c=calloc(1,sizeof(*c));if(!c)return ROCKET_E_NOMEM;
    c->fd=fd;c->cfg=*cfg;c->plan=plan;c->mode=mode;
    c->input=malloc((size_t)extent(cfg->rows,0)*plan.cols*2);
    c->rot=malloc(256u*256u*2);c->cos=malloc(256u*256u*2);c->sin=malloc(256u*256u*2);
    c->frequency=malloc((size_t)cfg->rotary_dim/2*sizeof(double));
    if(!c->input||!c->rot||!c->cos||!c->sin||!c->frequency){rc=ROCKET_E_NOMEM;goto fail;}
    for(unsigned i=0;i<cfg->rotary_dim/2;i++)c->frequency[i]=pow(cfg->theta,-2.0*i/cfg->rotary_dim);
    co_tensor t;rc=co_tensor_matrix(&t,extent(cfg->rows,0),extent(plan.cols,0),CO_F16);if(rc)goto fail;
    for(unsigned i=0;i<P_COUNT;i++){rc=co_buffer_alloc(fd,t.bytes,&c->scratch[i]);if(rc)goto fail;}
    for(unsigned cl=0;cl<plan.column_classes;cl++) {
        rc=co_tensor_matrix(&t,extent(cfg->rows,0),cl?plan.cols%256:extent(plan.cols,0),CO_F16);if(rc)goto fail;
        co_tensor id;rc=co_tensor_weight(&id,t.physical_cols,t.physical_cols);if(rc)goto fail;
        rc=co_buffer_alloc(fd,id.bytes,&c->identity[cl]);if(rc)goto fail;
        rc=co_view_init(&c->ids[cl],&c->identity[cl],0,&id);if(rc)goto fail;
        rc=co_identity_upload(&c->ids[cl]);if(rc)goto fail;
    }
    *out=c;return 0;
fail:co_rope_destroy(c);return rc;
}
static int compatible(co_view v,const co_tensor *t,int fd) {
    co_view check;
    return !co_view_init(&check,v.buffer,v.offset,&v.tensor)&&v.buffer->fd==fd&&
        v.tensor.type==t->type&&v.tensor.layout==t->layout&&v.tensor.rows==t->rows&&v.tensor.cols==t->cols&&
        v.tensor.physical_rows==t->physical_rows&&v.tensor.physical_cols==t->physical_cols&&v.tensor.atom==t->atom&&v.tensor.bytes==t->bytes;
}
static int overlap(co_view a,co_view b) {
    uint64_t aa=a.buffer->bo.dma_address+a.offset,bb=b.buffer->bo.dma_address+b.offset;
    return (a.buffer->fd==b.buffer->fd&&a.buffer->bo.handle==b.buffer->bo.handle&&
        a.offset<b.offset+b.tensor.bytes&&b.offset<a.offset+a.tensor.bytes)||
        (aa<bb+b.tensor.bytes&&bb<aa+a.tensor.bytes);
}
int co_rope_run(co_rope *c,const co_view *x,unsigned nx,const co_view *y,unsigned ny,uint32_t base) {
    if(!c||!x||!y||nx!=c->plan.tiles||ny!=nx||base>UINT32_MAX-(c->cfg.rows-1))return ROCKET_E_SHAPE;
    unsigned nt=blocks(c->plan.cols);
    for(unsigned i=0;i<nx;i++) {
        co_tensor t;int rc=co_tensor_matrix(&t,extent(c->cfg.rows,i/nt),extent(c->plan.cols,i%nt),CO_F16);if(rc)return rc;
        if(!compatible(x[i],&t,c->fd)||!compatible(y[i],&t,c->fd))return ROCKET_E_SHAPE;
    }
    for(unsigned i=0;i<ny;i++) {
        for(unsigned j=0;j<nx;j++)if(overlap(x[j],y[i]))return ROCKET_E_SHAPE;
        for(unsigned j=0;j<i;j++)if(overlap(y[j],y[i]))return ROCKET_E_SHAPE;
    }
    for(unsigned mi=0;mi<blocks(c->cfg.rows);mi++) {
        unsigned rows=extent(c->cfg.rows,mi);
        /* Gather a complete logical row block so a pair may cross a tile boundary. */
        for(unsigned ni=0;ni<nt;ni++) {
            co_view v=x[mi*nt+ni];int rc=rocket_bo_prep(c->fd,&v.buffer->bo,0,2000000000ull);if(rc)return rc;
            const _Float16 *src=(const _Float16 *)((const char *)v.buffer->bo.ptr+v.offset);
            for(unsigned r=0;r<rows;r++)for(unsigned q=0;q<v.tensor.cols;q++) {
                _Float16 value=src[co_index(&v.tensor,r,q)];
                if(!isfinite((double)value))rc=ROCKET_E_SHAPE;
                c->input[(size_t)r*c->plan.cols+ni*256+q]=value;
            }
            int fini=rocket_bo_fini(c->fd,&v.buffer->bo);if(!rc)rc=fini;if(rc)return rc;
        }
        for(unsigned ni=0;ni<nt;ni++) {
            co_view in=x[mi*nt+ni],out=y[mi*nt+ni],v[P_COUNT];
            unsigned cl=(c->plan.column_classes==2&&ni==nt-1)?1:0;
            for(unsigned r=0;r<rows;r++)for(unsigned q=0;q<in.tensor.cols;q++) {
                unsigned col=ni*256+q,d=col%c->cfg.head_dim;size_t at=(size_t)r*in.tensor.cols+q;
                if(d>=c->cfg.rotary_dim){c->cos[at]=1;c->sin[at]=0;c->rot[at]=0;continue;}
                unsigned half=c->cfg.rotary_dim/2,paired,index;int negative;
                if(c->cfg.style==CO_ROPE_INTERLEAVED){paired=d^1u;index=d/2;negative=!(d&1);}
                else {paired=d<half?d+half:d-half;index=d%half;negative=d<half;}
                double angle=((double)base+mi*256+r)*c->frequency[index];
                c->cos[at]=(_Float16)cos(angle);c->sin[at]=(_Float16)sin(angle);
                _Float16 source=c->input[(size_t)r*c->plan.cols+col-d+paired];
                c->rot[at]=negative?-source:source;
            }
            int rc=0;for(unsigned i=0;i<P_COUNT&&!rc;i++)rc=co_view_init(&v[i],&c->scratch[i],0,&in.tensor);
            if(!rc)rc=co_upload_f16(&v[P_ROT],c->rot);
            if(!rc)rc=co_upload_f16(&v[P_COS],c->cos);
            if(!rc)rc=co_upload_f16(&v[P_SIN],c->sin);
            co_program p[3];
            if(!rc)rc=co_emit_ew(&p[0],in,v[P_COS],c->ids[cl],v[P_MUL0],1);
            if(!rc)rc=co_emit_ew(&p[1],v[P_ROT],v[P_SIN],c->ids[cl],v[P_MUL1],1);
            if(!rc)rc=co_emit_ew(&p[2],v[P_MUL0],v[P_MUL1],c->ids[cl],out,0);
            if(!rc)rc=co_execute(c->fd,p,3,c->mode);
            if(rc)return rc;
        }
    }
    for(unsigned i=0;i<P_COUNT;i++){int rc=co_guard_check(&c->scratch[i]);if(rc)return rc;}
    for(unsigned cl=0;cl<c->plan.column_classes;cl++){int rc=co_guard_check(&c->identity[cl]);if(rc)return rc;}
    for(unsigned i=0;i<nx;i++){int rc=co_guard_check(x[i].buffer);if(!rc)rc=co_guard_check(y[i].buffer);if(rc)return rc;}
    return 0;
}
