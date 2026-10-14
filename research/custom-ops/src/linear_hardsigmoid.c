// SPDX-License-Identifier: GPL-3.0-or-later
#include "custom_ops.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Two ping-pong accumulators (L_ACC0/L_ACC1) replace the single L_MM cube: the
 * first K tile writes one without accumulate, every later tile fuses its conv
 * result with the prior accumulator through the DPU EW add. Weight tiles are packed
 * once, one <=256-wide BO per tile, and all live until the context is destroyed. */
enum { L_INPUT, L_BIAS, L_IDENTITY, L_AFFINE, L_OUTPUT, L_ACC0, L_ACC1, L_COUNT };
typedef struct co_linear_tile co_linear_tile;
struct co_linear_tile {
    int fd;
    unsigned m, k, n, ktiles;
    enum co_mode mode;
    int has_bias;
    enum co_linear_post post;
    co_buffer buffers[L_COUNT];
    co_view views[L_COUNT];
    co_buffer wtiles[CO_K_TILES];
    co_view wviews[CO_K_TILES];
    co_program programs[CO_MAX_PROGRAMS];
    unsigned nprograms;
};

static int finite_vector(const _Float16 *x, size_t n) {
    if (!x) return 0;
    for (size_t i=0; i<n; i++) if (!isfinite((double)x[i])) return 0;
    return 1;
}

static int compatible(co_view x, co_view expected, int fd) {
    co_view checked;
    const co_tensor *a=&x.tensor, *b=&expected.tensor;
    return !co_view_init(&checked,x.buffer,x.offset,a) && x.buffer->fd==fd &&
        a->type==b->type && a->layout==b->layout && a->rows==b->rows && a->cols==b->cols &&
        a->physical_rows==b->physical_rows && a->physical_cols==b->physical_cols &&
        a->atom==b->atom && a->bytes==b->bytes;
}

/* Derive a co_view for the K tile [k0,k0+kt) of the caller's full input cube x.
 * Only the descriptor (and its offset) is built; no BO payload is read or copied. */
static int input_tile_view(co_view x, unsigned k0, unsigned kt, co_view *out) {
    co_tensor slice; size_t off;
    int rc=co_tensor_input_slice(&slice,&x.tensor,k0,kt,&off);
    if (rc) return rc;
    return co_view_init(out,x.buffer,x.offset+off,&slice);
}

static int emit_tiled(co_linear_tile *c, const co_view *xs, co_view y,
                       co_program *p, unsigned *count) {
    unsigned idx=0;int rc=0;
    co_view prior={0};
    for (unsigned i=0;i<c->ktiles;i++) {
        co_view dst=(i&1)?c->views[L_ACC1]:c->views[L_ACC0];
        if (i==c->ktiles-1 && !c->has_bias && c->post==CO_LINEAR_NONE) dst=y;
        rc=i?co_emit_mm_acc(&p[idx],xs[i],c->wviews[i],prior,dst):
             co_emit_mm(&p[idx],xs[i],c->wviews[i],dst);
        if (rc) return rc;
        idx++;prior=dst;
    }
    if (c->has_bias) {
        co_view dst=c->post==CO_LINEAR_NONE?y:c->views[L_AFFINE];
        rc=co_emit_ew(&p[idx++],prior,c->views[L_BIAS],c->views[L_IDENTITY],dst,0);
        if (rc) return rc;
        prior=dst;
    }
    if (c->post==CO_LINEAR_HSIG) {
        unsigned elements=y.tensor.physical_rows*y.tensor.physical_cols;
        for (unsigned base=0;base<elements;base+=32768) {
            unsigned len=elements-base;if (len>32768) len=32768;
            co_tensor t;co_view src,dst;
            rc=co_tensor_linear(&t,len);
            if (!rc) rc=co_view_init(&src,prior.buffer,prior.offset+(size_t)base*2,&t);
            if (!rc) rc=co_view_init(&dst,y.buffer,y.offset+(size_t)base*2,&t);
            if (!rc) rc=co_emit_lut(&p[idx++],src,dst,CO_HARDSIGMOID);
            if (rc) return rc;
        }
    }
    *count=idx;return 0;
}

static int emit(co_linear_tile *c, co_view x, co_view y, co_program *p, unsigned *count) {
    co_view xs[CO_K_TILES];
    for (unsigned i=0;i<c->ktiles;i++) {
        int rc=input_tile_view(x,i*CO_K_TILE,co_k_tile_length(c->k,i),&xs[i]);
        if (rc) return rc;
    }
    return emit_tiled(c,xs,y,p,count);
}

static int pack_weight_tiles(co_linear_tile *c, const _Float16 *weights) {
    _Float16 *tile=malloc((size_t)c->n*CO_K_TILE*2);
    if (!tile) return ROCKET_E_NOMEM;
    int rc=0;
    for (unsigned i=0; i<c->ktiles && !rc; i++) {
        unsigned k0=i*CO_K_TILE, kt=co_k_tile_length(c->k,i);
        co_tensor wt;
        rc=co_tensor_weight(&wt,c->n,kt);
        if (!rc && !c->wtiles[i].bo.handle) rc=co_buffer_alloc(c->fd,wt.bytes,&c->wtiles[i]);
        if (!rc) rc=co_view_init(&c->wviews[i],&c->wtiles[i],0,&wt);
        if (rc) break;
        for (unsigned r=0; r<c->n; r++)
            for (unsigned q=0; q<kt; q++)
                tile[(size_t)r*kt+q]=weights[(size_t)r*c->k+k0+q];
        rc=co_upload_f16(&c->wviews[i],tile);
    }
    free(tile);
    return rc;
}

static void co_linear_tile_destroy(co_linear_tile *c) {
    if (!c) return;
    for (unsigned i=0; i<L_COUNT; i++) co_buffer_free(&c->buffers[i]);
    for (unsigned i=0; i<CO_K_TILES; i++) co_buffer_free(&c->wtiles[i]);
    free(c);
}

static int co_linear_tile_create(int fd, unsigned m, unsigned k, unsigned n,
                          const _Float16 *weights, const _Float16 *bias,
                          enum co_linear_post post, enum co_mode mode, co_linear_tile **out) {
    if (!out) return ROCKET_E_SHAPE;
    *out=NULL;
    co_tensor input,output,identity;
    if (fd<0 || (mode!=CO_SEPARATE && mode!=CO_CHAIN) ||
        !m || m>256 || !k || k>CO_MAX_K || !n || n>256 ||
        co_tensor_input(&input,m,k) || co_tensor_matrix(&output,m,n,CO_F16) ||
        co_tensor_weight(&identity,output.physical_cols,output.physical_cols) ||
        !finite_vector(weights,(size_t)n*k) || (bias && !finite_vector(bias,n))) return ROCKET_E_SHAPE;
    if (mode==CO_CHAIN && !rocket_batched_submit_supported()) return ROCKET_E_UNSUPPORTED;
    co_linear_tile *c=calloc(1,sizeof(*c));
    if (!c) return ROCKET_E_NOMEM;
    c->fd=fd;c->m=m;c->k=k;c->n=n;c->mode=mode;c->ktiles=co_k_tiles(k);
    c->has_bias=bias!=NULL;c->post=post;
    co_tensor desc[L_COUNT]={input,output,identity,output,output,output,output};
    int rc=0;
    for (unsigned i=0; i<L_COUNT; i++) {
        if ((rc=co_buffer_alloc(fd,desc[i].bytes,&c->buffers[i])) ||
            (rc=co_view_init(&c->views[i],&c->buffers[i],0,&desc[i]))) goto failed;
    }
    if ((rc=pack_weight_tiles(c,weights)) ||
        (rc=bias?co_bias_upload(&c->views[L_BIAS],bias):0) ||
        (rc=co_identity_upload(&c->views[L_IDENTITY])) ||
        (rc=emit(c,c->views[L_INPUT],c->views[L_OUTPUT],c->programs,&c->nprograms))) goto failed;
    *out=c;return 0;
failed:co_linear_tile_destroy(c);return rc;
}

static int prepare_outputs(co_linear_tile *c, co_view y) {
    int rc=co_sentinel(&c->buffers[L_ACC0]);
    if (!rc) rc=co_sentinel(&c->buffers[L_ACC1]);
    if (!rc) rc=co_sentinel(&c->buffers[L_AFFINE]);
    /* Preserve other views sharing the caller's output BO: only stamp y's slice. */
    if (!rc) rc=rocket_bo_prep(c->fd,&y.buffer->bo,1,2000000000ull);
    if (!rc) {
        memset((char *)y.buffer->bo.ptr+y.offset,0xff,y.tensor.bytes);
        rc=rocket_bo_fini(c->fd,&y.buffer->bo);
    }
    return rc;
}

static int guarded_execute(co_linear_tile *c, const co_program *p, unsigned count, co_view y) {
    int rc=prepare_outputs(c,y);
    if (!rc) rc=co_execute(c->fd,p,count,c->mode);
    for (unsigned i=0; !rc && i<L_COUNT; i++) rc=co_guard_check(&c->buffers[i]);
    for (unsigned i=0; !rc && i<CO_K_TILES; i++)
        if (c->wtiles[i].bo.handle) rc=co_guard_check(&c->wtiles[i]);
    if (!rc && y.buffer!=&c->buffers[L_OUTPUT]) rc=co_guard_check(y.buffer);
    return rc;
}

static int co_linear_tile_run(co_linear_tile *c, const _Float16 *x, _Float16 *y) {
    if (!c || !y || !finite_vector(x,(size_t)c->m*c->k)) return ROCKET_E_SHAPE;
    int rc=co_upload_f16(&c->views[L_INPUT],x);
    if (!rc) rc=guarded_execute(c,c->programs,c->nprograms,c->views[L_OUTPUT]);
    if (!rc) rc=co_download(&c->views[L_OUTPUT],y);
    if (!rc && !finite_vector(y,(size_t)c->m*c->n)) rc=ROCKET_E_DEVICE;
    return rc;
}

static int co_linear_tile_run_cube(co_linear_tile *c, co_view x, co_view y) {
    if (!c || !compatible(x,c->views[L_INPUT],c->fd) ||
        !compatible(y,c->views[L_OUTPUT],c->fd)) return ROCKET_E_SHAPE;
    if (x.buffer==y.buffer && x.offset<y.offset+y.tensor.bytes &&
        y.offset<x.offset+x.tensor.bytes) return ROCKET_E_SHAPE;
    for (unsigned i=0;i<L_COUNT;i++)
        if (x.buffer==&c->buffers[i] || y.buffer==&c->buffers[i]) return ROCKET_E_SHAPE;
    for (unsigned i=0;i<c->ktiles;i++)
        if (x.buffer==&c->wtiles[i] || y.buffer==&c->wtiles[i]) return ROCKET_E_SHAPE;
    co_program p[CO_MAX_PROGRAMS];unsigned count;
    int rc=emit(c,x,y,p,&count);
    if (!rc) rc=guarded_execute(c,p,count,y);
    if (!rc) rc=co_guard_check(x.buffer);
    /* No host tensor read, including validation; caller scores y independently. */
    return rc;
}

/* The full operator reuses one context per N tile and distinct row extent.
 * Repeated 256-row blocks therefore do not duplicate resident weights/scratch. */
struct co_linear_hsig {
    co_linear_plan plan;
    co_linear_tile **tiles;
    _Float16 *input, *output;
};

static co_linear_tile *context_at(co_linear_hsig *c, unsigned mi, unsigned ni) {
    unsigned row_class=(c->plan.row_classes==2 && mi==c->plan.mtiles-1)?1:0;
    return c->tiles[row_class*c->plan.ntiles+ni];
}

void co_linear_hsig_destroy(co_linear_hsig *c) {
    if (!c) return;
    if (c->tiles) for (unsigned i=0;i<c->plan.row_classes*c->plan.ntiles;i++)
        co_linear_tile_destroy(c->tiles[i]);
    free(c->tiles);free(c->input);free(c->output);free(c);
}

int co_linear_create(int fd, unsigned m, unsigned k, unsigned n,
                          const _Float16 *weights, const _Float16 *bias,
                          enum co_linear_post post, enum co_mode mode, co_linear **out) {
    if (!out) return ROCKET_E_SHAPE;
    *out=NULL;
    co_linear_plan plan;
    if (fd<0 || (mode!=CO_CHAIN && mode!=CO_SEPARATE)) return ROCKET_E_SHAPE;
    int rc=co_linear_plan_init_ex(&plan,m,k,n,bias!=NULL,post,0);
    if (rc) return rc;
    if (!finite_vector(weights,(size_t)n*k) || (bias && !finite_vector(bias,n))) return ROCKET_E_SHAPE;
    if (mode==CO_CHAIN && !rocket_batched_submit_supported()) return ROCKET_E_UNSUPPORTED;
    co_linear_hsig *c=calloc(1,sizeof(*c));
    if (!c) return ROCKET_E_NOMEM;
    c->plan=plan;
    c->tiles=calloc(plan.row_classes*plan.ntiles,sizeof(*c->tiles));
    c->input=malloc((size_t)(m<256?m:256)*k*sizeof(*c->input));
    c->output=malloc((size_t)(m<256?m:256)*(n<256?n:256)*sizeof(*c->output));
    if (!c->tiles || !c->input || !c->output) {rc=ROCKET_E_NOMEM;goto failed;}
    for (unsigned row_class=0;row_class<plan.row_classes;row_class++) {
        unsigned mi=row_class?plan.mtiles-1:0;
        for (unsigned ni=0;ni<plan.ntiles;ni++) {
            co_linear_tile_info info;
            rc=co_linear_plan_tile(&plan,mi*plan.ntiles+ni,&info);
            if (rc) goto failed;
            /* W's N slices are contiguous in the host row-major boundary. */
            rc=co_linear_tile_create(fd,info.rows,k,info.cols,
                weights+(size_t)info.n0*k,bias?bias+info.n0:NULL,post,mode,
                &c->tiles[row_class*plan.ntiles+ni]);
            if (rc) goto failed;
        }
    }
    *out=c;return 0;
failed:co_linear_hsig_destroy(c);return rc;
}

int co_linear_hsig_run(co_linear_hsig *c, const _Float16 *x, _Float16 *y) {
    if (!c || !y || !finite_vector(x,(size_t)c->plan.m*c->plan.k)) return ROCKET_E_SHAPE;
    /* Stage the whole result so overlapping host arrays and late device failures
     * cannot corrupt subsequent inputs or partially publish a result. */
    size_t bytes=(size_t)c->plan.m*c->plan.n*sizeof(*y);
    _Float16 *result=malloc(bytes);
    if (!result) return ROCKET_E_NOMEM;
    int rc=0;
    for (unsigned mi=0;mi<c->plan.mtiles && !rc;mi++) {
        unsigned rows=context_at(c,mi,0)->m;
        memcpy(c->input,x+(size_t)mi*256*c->plan.k,(size_t)rows*c->plan.k*sizeof(*x));
        for (unsigned ni=0;ni<c->plan.ntiles && !rc;ni++) {
            co_linear_tile *tile=context_at(c,mi,ni);
            rc=co_linear_tile_run(tile,c->input,c->output);
            if (!rc) for (unsigned r=0;r<rows;r++)
                memcpy(result+((size_t)mi*256+r)*c->plan.n+ni*256,
                       c->output+(size_t)r*tile->n,(size_t)tile->n*sizeof(*y));
        }
    }
    if (!rc) memcpy(y,result,bytes);
    free(result);return rc;
}

static int overlap(co_view a, co_view b) {
    /* Reject aliases even when represented by separate co_buffer wrappers. */
    if (a.buffer->bo.handle==b.buffer->bo.handle && a.buffer->fd==b.buffer->fd)
        return a.offset<b.offset+b.tensor.bytes && b.offset<a.offset+a.tensor.bytes;
    uint64_t aa=a.buffer->bo.dma_address+a.offset,bb=b.buffer->bo.dma_address+b.offset;
    return aa<bb+b.tensor.bytes && bb<aa+a.tensor.bytes;
}

int co_linear_hsig_run_tiles(co_linear_hsig *c, const co_view *x, unsigned nx,
                             const co_view *y, unsigned ny) {
    if (!c || !x || !y || nx!=c->plan.mtiles || ny!=c->plan.output_tiles)
        return ROCKET_E_SHAPE;
    /* Validate the entire collection before the first submission. */
    for (unsigned mi=0;mi<nx;mi++) {
        co_linear_tile *tile=context_at(c,mi,0);
        if (!compatible(x[mi],tile->views[L_INPUT],tile->fd)) return ROCKET_E_SHAPE;
    }
    for (unsigned i=0;i<ny;i++) {
        co_linear_tile *tile=context_at(c,i/c->plan.ntiles,i%c->plan.ntiles);
        if (!compatible(y[i],tile->views[L_OUTPUT],tile->fd)) return ROCKET_E_SHAPE;
        for (unsigned j=0;j<nx;j++) if (overlap(x[j],y[i])) return ROCKET_E_SHAPE;
        for (unsigned j=0;j<i;j++) if (overlap(y[j],y[i])) return ROCKET_E_SHAPE;
    }
    for (unsigned i=0;i<ny;i++) {
        int rc=co_linear_tile_run_cube(context_at(c,i/c->plan.ntiles,i%c->plan.ntiles),
                                     x[i/c->plan.ntiles],y[i]);
        if (rc) return rc;
    }
    return 0;
}

int co_linear_hsig_run_cube(co_linear_hsig *c, co_view x, co_view y) {
    if (!c) return ROCKET_E_SHAPE;
    if (c->plan.output_tiles!=1) return ROCKET_E_UNSUPPORTED;
    return co_linear_hsig_run_tiles(c,&x,1,&y,1);
}

int co_linear_hsig_create(int fd, unsigned m, unsigned k, unsigned n,
                          const _Float16 *w, const _Float16 *b,
                          enum co_mode mode, co_linear_hsig **out) {
    if (!b) {if (out) *out=NULL;return ROCKET_E_SHAPE;}
    return co_linear_create(fd,m,k,n,w,b,CO_LINEAR_HSIG,mode,out);
}
int co_linear_run(co_linear *c,const _Float16 *x,_Float16 *y) {
    return co_linear_hsig_run(c,x,y);
}
void co_linear_destroy(co_linear *c) {co_linear_hsig_destroy(c);}

int co_linear_run_tiled(co_linear *c,const co_view *x,unsigned nx,
                        const co_view *y,unsigned ny) {
    if (!c || !x || !y || nx!=c->plan.mtiles*c->plan.ktiles || ny!=c->plan.output_tiles)
        return ROCKET_E_SHAPE;
    for (unsigned i=0;i<nx;i++) {
        co_linear_tile *tile=context_at(c,i/c->plan.ktiles,0);
        co_tensor t;co_view expected;
        int rc=co_tensor_matrix(&t,tile->m,co_k_tile_length(c->plan.k,i%c->plan.ktiles),CO_F16);
        if (rc) return rc;
        expected=(co_view){.tensor=t};
        if (!compatible(x[i],expected,tile->fd)) return ROCKET_E_SHAPE;
    }
    for (unsigned i=0;i<ny;i++) {
        co_linear_tile *tile=context_at(c,i/c->plan.ntiles,i%c->plan.ntiles);
        if (!compatible(y[i],tile->views[L_OUTPUT],tile->fd)) return ROCKET_E_SHAPE;
        for (unsigned j=0;j<nx;j++) if (overlap(x[j],y[i])) return ROCKET_E_SHAPE;
        for (unsigned j=0;j<i;j++) if (overlap(y[j],y[i])) return ROCKET_E_SHAPE;
    }
    for (unsigned i=0;i<ny;i++) {
        co_linear_tile *tile=context_at(c,i/c->plan.ntiles,i%c->plan.ntiles);
        co_program p[CO_MAX_PROGRAMS];unsigned count=0;
        int rc=emit_tiled(tile,x+(i/c->plan.ntiles)*c->plan.ktiles,y[i],p,&count);
        if (!rc) rc=guarded_execute(tile,p,count,y[i]);
        if (rc) return rc;
    }
    for (unsigned i=0;i<nx;i++) {int rc=co_guard_check(x[i].buffer);if (rc) return rc;}
    return 0;
}

int co_linear_update_weights(co_linear *c,const _Float16 *weights,const _Float16 *bias) {
    if (!c || !finite_vector(weights,(size_t)c->plan.n*c->plan.k) ||
        (c->plan.has_bias?!finite_vector(bias,c->plan.n):bias!=NULL)) return ROCKET_E_SHAPE;
    for (unsigned cl=0;cl<c->plan.row_classes;cl++)for(unsigned ni=0;ni<c->plan.ntiles;ni++) {
        co_linear_tile *t=c->tiles[cl*c->plan.ntiles+ni];
        int rc=pack_weight_tiles(t,weights+(size_t)ni*256*c->plan.k);
        if (!rc && bias) rc=co_bias_upload(&t->views[L_BIAS],bias+ni*256);
        if (rc) return rc;
    }
    return 0;
}
