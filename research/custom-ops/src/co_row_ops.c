// SPDX-License-Identifier: GPL-3.0-or-later
#include "co_row_ops.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { R_TMP,R_COEFF,R_SHIFT,R_EXP,R_STATS,R_COUNT };
struct co_rowop {
    int fd;enum co_row_kind kind;enum co_mode mode;float eps;co_row_plan plan;
    co_buffer buffers[R_COUNT],identity[2],ones[2];co_view ids[2],ws[2];
    _Float16 *gamma,*host,*coeff;float *stats;double *sum,*max;
};
static unsigned blocks(unsigned n){return (n+255)/256;}
static unsigned extent(unsigned n,unsigned i){unsigned t=n-i*256;return t>256?256:t;}
static int descriptors(enum co_row_kind kind,unsigned m,unsigned n,co_tensor *t) {
    unsigned mr=m>256?256:m,nc=n>256?256:n;
    int rc=co_tensor_matrix(&t[R_TMP],mr,nc,CO_F16);if(rc)return rc;
    t[R_COEFF]=t[R_TMP];t[R_SHIFT]=t[R_TMP];
    rc=co_tensor_input(&t[R_EXP],mr,n);if(rc)return rc;
    rc=co_tensor_matrix(&t[R_STATS],mr,1,CO_F32);if(rc)return rc;
    if(kind==CO_ROW_RMSNORM){t[R_SHIFT].bytes=0;t[R_EXP].bytes=0;}
    return 0;
}
int co_row_plan_init(co_row_plan *out,enum co_row_kind kind,unsigned m,unsigned n) {
    if(!out)return ROCKET_E_SHAPE;
    memset(out,0,sizeof(*out));
    if((kind!=CO_ROW_RMSNORM&&kind!=CO_ROW_SOFTMAX)||!m||m>CO_MAX_M||!n||n>CO_MAX_N)
        return ROCKET_E_SHAPE;
    co_row_plan p={.rows=m,.cols=n,.tiles=blocks(m)*blocks(n),
        .column_classes=(n>256&&n%256)?2:1};
    co_tensor t[R_COUNT];int rc=descriptors(kind,m,n,t);if(rc)return rc;
    for(unsigned i=0;i<R_COUNT;i++)if(t[i].bytes)p.device_bytes+=t[i].bytes+CO_SLACK+CO_GUARD;
    for(unsigned c=0;c<p.column_classes;c++) {
        unsigned cols=c?n%256:extent(n,0);co_tensor id,w,x;
        rc=co_tensor_matrix(&x,extent(m,0),cols,CO_F16);if(rc)return rc;
        rc=co_tensor_weight(&id,x.physical_cols,x.physical_cols);if(rc)return rc;
        rc=co_tensor_weight(&w,16,cols);if(rc)return rc;
        p.device_bytes+=id.bytes+w.bytes+2*(CO_SLACK+CO_GUARD);
    }
    p.chain_submits=2ull*p.tiles;
    p.statistic_downloads=(size_t)m*blocks(n);
    p.host_uploads=(size_t)m*n*(kind==CO_ROW_SOFTMAX?3:1);
    if(kind==CO_ROW_SOFTMAX)p.host_activation_reads=(size_t)m*n*2;
    for(unsigned mi=0;mi<blocks(m);mi++)for(unsigned ni=0;ni<blocks(n);ni++) {
        co_tensor x;rc=co_tensor_matrix(&x,extent(m,mi),extent(n,ni),CO_F16);if(rc)return rc;
        p.tasks+=kind==CO_ROW_RMSNORM?3:(x.bytes/2+32767)/32768+3;
    }
    *out=p;return 0;
}
void co_rowop_destroy(co_rowop *c) {
    if(!c)return;
    for(unsigned i=0;i<R_COUNT;i++)co_buffer_free(&c->buffers[i]);
    for(unsigned i=0;i<2;i++){co_buffer_free(&c->identity[i]);co_buffer_free(&c->ones[i]);}
    free(c->gamma);free(c->host);free(c->coeff);free(c->stats);free(c->sum);free(c->max);free(c);
}
int co_rowop_create(int fd,enum co_row_kind kind,unsigned m,unsigned n,
                    const _Float16 *gamma,float eps,enum co_mode mode,co_rowop **out) {
    if(!out)return ROCKET_E_SHAPE;
    *out=NULL;
    if(fd<0||(mode!=CO_CHAIN&&mode!=CO_SEPARATE))return ROCKET_E_SHAPE;
    co_row_plan plan;int rc=co_row_plan_init(&plan,kind,m,n);if(rc)return rc;
    if(kind==CO_ROW_RMSNORM) {
        if(!gamma||!isfinite(eps)||eps<=0)return ROCKET_E_SHAPE;
        for(unsigned i=0;i<n;i++)if(!isfinite((double)gamma[i]))return ROCKET_E_SHAPE;
    }
    if(mode==CO_CHAIN&&!rocket_batched_submit_supported())return ROCKET_E_UNSUPPORTED;
    co_rowop *c=calloc(1,sizeof(*c));if(!c)return ROCKET_E_NOMEM;
    c->fd=fd;c->kind=kind;c->mode=mode;c->eps=eps;c->plan=plan;
    c->host=malloc(256u*256u*2);c->coeff=malloc(256u*256u*2);
    c->stats=malloc(256*sizeof(*c->stats));c->sum=malloc(256*sizeof(*c->sum));c->max=malloc(256*sizeof(*c->max));
    if(kind==CO_ROW_RMSNORM)c->gamma=malloc((size_t)n*2);
    if(!c->host||!c->coeff||!c->stats||!c->sum||!c->max||(kind==CO_ROW_RMSNORM&&!c->gamma)) {
        rc=ROCKET_E_NOMEM;goto fail;
    }
    if(c->gamma)memcpy(c->gamma,gamma,(size_t)n*2);
    co_tensor t[R_COUNT];rc=descriptors(kind,m,n,t);if(rc)goto fail;
    for(unsigned i=0;i<R_COUNT;i++)if(t[i].bytes) {
        rc=co_buffer_alloc(fd,t[i].bytes,&c->buffers[i]);if(rc)goto fail;
    }
    for(unsigned cl=0;cl<plan.column_classes;cl++) {
        unsigned nc=cl?n%256:extent(n,0);co_tensor x,id,w;
        rc=co_tensor_matrix(&x,extent(m,0),nc,CO_F16);if(rc)goto fail;
        rc=co_tensor_weight(&id,x.physical_cols,x.physical_cols);if(rc)goto fail;
        rc=co_tensor_weight(&w,16,nc);if(rc)goto fail;
        rc=co_buffer_alloc(fd,id.bytes,&c->identity[cl]);if(rc)goto fail;
        rc=co_buffer_alloc(fd,w.bytes,&c->ones[cl]);if(rc)goto fail;
        rc=co_view_init(&c->ids[cl],&c->identity[cl],0,&id);if(rc)goto fail;
        rc=co_view_init(&c->ws[cl],&c->ones[cl],0,&w);if(rc)goto fail;
        rc=co_identity_upload(&c->ids[cl]);if(rc)goto fail;
        rc=co_ones_upload(&c->ws[cl],nc,0);if(rc)goto fail;
    }
    *out=c;return 0;
fail:co_rowop_destroy(c);return rc;
}
static int compatible(co_view v,const co_tensor *t,int fd) {
    co_view check;
    return !co_view_init(&check,v.buffer,v.offset,&v.tensor)&&v.buffer->fd==fd&&
        v.tensor.type==t->type&&v.tensor.layout==t->layout&&v.tensor.rows==t->rows&&v.tensor.cols==t->cols&&
        v.tensor.physical_rows==t->physical_rows&&v.tensor.physical_cols==t->physical_cols&&
        v.tensor.atom==t->atom&&v.tensor.bytes==t->bytes;
}
static int overlap(co_view a,co_view b) {
    uint64_t aa=a.buffer->bo.dma_address+a.offset,bb=b.buffer->bo.dma_address+b.offset;
    return (a.buffer->fd==b.buffer->fd&&a.buffer->bo.handle==b.buffer->bo.handle&&
        a.offset<b.offset+b.tensor.bytes&&b.offset<a.offset+a.tensor.bytes)||
        (aa<bb+b.tensor.bytes&&bb<aa+a.tensor.bytes);
}
static int max_scan(co_rowop *c,const co_view *x,unsigned mi) {
    unsigned rows=extent(c->plan.rows,mi),nt=blocks(c->plan.cols);
    for(unsigned r=0;r<rows;r++)c->max[r]=-INFINITY;
    for(unsigned ni=0;ni<nt;ni++) {
        co_view v=x[mi*nt+ni];int rc=rocket_bo_prep(c->fd,&v.buffer->bo,0,2000000000ull);if(rc)return rc;
        const _Float16 *src=(const _Float16 *)((const char *)v.buffer->bo.ptr+v.offset);
        for(unsigned r=0;r<rows;r++)for(unsigned q=0;q<v.tensor.cols;q++) {
            double value=src[co_index(&v.tensor,r,q)];
            if(isnan(value)||value==INFINITY)rc=ROCKET_E_SHAPE;
            if(value>c->max[r])c->max[r]=value;
        }
        int fini=rocket_bo_fini(c->fd,&v.buffer->bo);if(!rc)rc=fini;if(rc)return rc;
    }
    return 0;
}
static int shifted_upload(co_rowop *c,co_view x,co_view shift,co_view coeff) {
    int rc=rocket_bo_prep(c->fd,&x.buffer->bo,0,2000000000ull);if(rc)return rc;
    const _Float16 *src=(const _Float16 *)((const char *)x.buffer->bo.ptr+x.offset);
    for(unsigned r=0;r<x.tensor.rows;r++)for(unsigned q=0;q<x.tensor.cols;q++) {
        size_t at=(size_t)r*x.tensor.cols+q;double v=src[co_index(&x.tensor,r,q)];
        double delta=v-c->max[r];
        if(v==-INFINITY||c->max[r]==-INFINITY) {c->host[at]=0;c->coeff[at]=0;}
        else {
            unsigned band=(unsigned)floor(-delta/4);
            if(band>=5){c->host[at]=0;c->coeff[at]=0;}
            else {c->host[at]=(_Float16)(delta+4*band);c->coeff[at]=(_Float16)exp(-4.0*band);}
        }
    }
    rc=rocket_bo_fini(c->fd,&x.buffer->bo);if(rc)return rc;
    rc=co_upload_f16(&shift,c->host);if(!rc)rc=co_upload_f16(&coeff,c->coeff);return rc;
}
static int lut(co_program *p,unsigned *count,co_view x,co_view y) {
    for(size_t base=0;base<x.tensor.bytes;base+=65536) {
        size_t bytes=x.tensor.bytes-base;if(bytes>65536)bytes=65536;
        co_tensor t;co_view a,b;int rc=co_tensor_linear(&t,(unsigned)(bytes/2));
        if(!rc)rc=co_view_init(&a,x.buffer,x.offset+base,&t);
        if(!rc)rc=co_view_init(&b,y.buffer,y.offset+base,&t);
        if(!rc)rc=co_emit_lut(&p[(*count)++],a,b,CO_EXP_NEG4);
        if(rc)return rc;
    }
    return 0;
}
int co_rowop_run(co_rowop *c,const co_view *x,unsigned nx,const co_view *y,unsigned ny) {
    if(!c||!x||!y||nx!=c->plan.tiles||ny!=nx)return ROCKET_E_SHAPE;
    unsigned nt=blocks(c->plan.cols);
    for(unsigned i=0;i<nx;i++) {
        co_tensor t;int rc=co_tensor_matrix(&t,extent(c->plan.rows,i/nt),extent(c->plan.cols,i%nt),CO_F16);
        if(rc)return rc;
        if(!compatible(x[i],&t,c->fd)||!compatible(y[i],&t,c->fd))return ROCKET_E_SHAPE;
    }
    for(unsigned i=0;i<ny;i++) {
        for(unsigned j=0;j<nx;j++)if(overlap(x[j],y[i]))return ROCKET_E_SHAPE;
        for(unsigned j=0;j<i;j++)if(overlap(y[j],y[i]))return ROCKET_E_SHAPE;
    }
    for(unsigned mi=0;mi<blocks(c->plan.rows);mi++) {
        unsigned rows=extent(c->plan.rows,mi);
        memset(c->sum,0,rows*sizeof(*c->sum));
        if(c->kind==CO_ROW_SOFTMAX){int rc=max_scan(c,x,mi);if(rc)return rc;}
        for(unsigned ni=0;ni<nt;ni++) {
            unsigned cl=(c->plan.column_classes==2&&ni==nt-1)?1:0;
            co_view in=x[mi*nt+ni],tmp,coeff,stats,expview={0},shift={0};co_tensor st;
            int rc=co_view_init(&tmp,&c->buffers[R_TMP],0,&in.tensor);
            if(!rc)rc=co_view_init(&coeff,&c->buffers[R_COEFF],0,&in.tensor);
            if(!rc)rc=co_tensor_matrix(&st,rows,1,CO_F32);
            if(!rc)rc=co_view_init(&stats,&c->buffers[R_STATS],0,&st);
            if(rc)return rc;
            co_program p[4];unsigned count=0;
            co_view reduced=tmp;
            if(c->kind==CO_ROW_RMSNORM)rc=co_emit_ew(&p[count++],in,in,c->ids[cl],tmp,1);
            else {
                rc=co_view_init(&shift,&c->buffers[R_SHIFT],0,&in.tensor);
                if(!rc)rc=co_view_init(&expview,&c->buffers[R_EXP],(size_t)ni*256*in.tensor.physical_rows*2,&in.tensor);
                if(!rc)rc=shifted_upload(c,in,shift,coeff);
                if(!rc)rc=lut(p,&count,shift,tmp);
                if(!rc)rc=co_emit_ew(&p[count++],tmp,coeff,c->ids[cl],expview,1);
                reduced=expview;
            }
            if(!rc)rc=co_emit_mm(&p[count++],reduced,c->ws[cl],stats);
            if(!rc)rc=co_execute(c->fd,p,count,c->mode);
            if(!rc)rc=co_download(&stats,c->stats);
            if(rc)return rc;
            for(unsigned r=0;r<rows;r++) {
                /* Vendor FP16 output conversion can SATURATE a square at 65504
                 * rather than produce Inf. A per-tile energy reaching this bound
                 * is conservatively rejected: finite statistics alone cannot
                 * distinguish a legitimate high sum from a saturated element. */
                if(!isfinite(c->stats[r])||c->stats[r]<0||
                   (c->kind==CO_ROW_RMSNORM&&c->stats[r]>=65504))return ROCKET_E_SHAPE;
                c->sum[r]+=c->stats[r];
            }
        }
        for(unsigned ni=0;ni<nt;ni++) {
            unsigned cl=(c->plan.column_classes==2&&ni==nt-1)?1:0;
            co_view in=x[mi*nt+ni],out=y[mi*nt+ni],coeff,source=in;
            int rc=co_view_init(&coeff,&c->buffers[R_COEFF],0,&in.tensor);if(rc)return rc;
            if(c->kind==CO_ROW_SOFTMAX) {
                rc=co_view_init(&source,&c->buffers[R_EXP],(size_t)ni*256*in.tensor.physical_rows*2,&in.tensor);
                if(rc)return rc;
            }
            for(unsigned r=0;r<rows;r++)for(unsigned q=0;q<in.tensor.cols;q++) {
                double a=c->kind==CO_ROW_RMSNORM?(double)c->gamma[ni*256+q]/sqrt(c->sum[r]/c->plan.cols+c->eps):
                    (c->sum[r]>0?1/c->sum[r]:0);
                c->coeff[(size_t)r*in.tensor.cols+q]=(_Float16)a;
                if(!isfinite((double)c->coeff[(size_t)r*in.tensor.cols+q]))return ROCKET_E_SHAPE;
            }
            rc=co_upload_f16(&coeff,c->coeff);if(rc)return rc;
            co_program p;rc=co_emit_ew(&p,source,coeff,c->ids[cl],out,1);
            if(!rc)rc=co_execute(c->fd,&p,1,c->mode);
            if(rc)return rc;
        }
    }
    for(unsigned i=0;i<R_COUNT;i++)if(c->buffers[i].bo.handle){int rc=co_guard_check(&c->buffers[i]);if(rc)return rc;}
    for(unsigned i=0;i<c->plan.column_classes;i++) {
        int rc=co_guard_check(&c->identity[i]);if(!rc)rc=co_guard_check(&c->ones[i]);if(rc)return rc;
    }
    for(unsigned i=0;i<nx;i++){int rc=co_guard_check(x[i].buffer);if(!rc)rc=co_guard_check(y[i].buffer);if(rc)return rc;}
    return 0;
}
