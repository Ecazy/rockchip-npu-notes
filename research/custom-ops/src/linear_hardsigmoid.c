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
struct co_linear_hsig {
    int fd;
    unsigned m, k, n, ktiles;
    enum co_mode mode;
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

static int emit(co_linear_hsig *c, co_view x, co_view y, co_program *p, unsigned *count) {
    unsigned idx=0;
    co_view xs;
    int rc=input_tile_view(x,0,co_k_tile_length(c->k,0),&xs);
    if (!rc) rc=co_emit_mm(&p[idx],xs,c->wviews[0],c->views[L_ACC0]);
    if (rc) return rc;
    idx++;
    for (unsigned i=1; i<c->ktiles; i++) {
        rc=input_tile_view(x,i*CO_K_TILE,co_k_tile_length(c->k,i),&xs);
        if (rc) return rc;
        /* prior is where tile i-1 landed; never write the same BO we read. */
        co_view prior=(i&1)?c->views[L_ACC0]:c->views[L_ACC1];
        co_view dst  =(i&1)?c->views[L_ACC1]:c->views[L_ACC0];
        rc=co_emit_mm_acc(&p[idx],xs,c->wviews[i],prior,dst);
        if (rc) return rc;
        idx++;
    }
    co_view acc=((c->ktiles-1)&1)?c->views[L_ACC1]:c->views[L_ACC0];
    rc=co_emit_ew(&p[idx],acc,c->views[L_BIAS],c->views[L_IDENTITY],c->views[L_AFFINE],0);
    if (rc) return rc;
    idx++;
    unsigned elements=y.tensor.physical_rows*y.tensor.physical_cols;
    if (elements<=32768) {
        rc=co_emit_lut(&p[idx++],c->views[L_AFFINE],y,CO_HARDSIGMOID);
    } else {
        /* LUT is flat elementwise: contiguous slices preserve the producer cube.
         * A 256x256 cube exceeds the DPU width field; two <=32768-element
         * programs cover it without detile/repack or dropping logical elements. */
        for (unsigned base=0; !rc && base<elements; base+=32768) {
            unsigned n=elements-base; if (n>32768) n=32768;
            co_tensor t;co_view src,dst;
            rc=co_tensor_linear(&t,n);
            if (!rc) rc=co_view_init(&src,c->views[L_AFFINE].buffer,
                                     c->views[L_AFFINE].offset+(size_t)base*2,&t);
            if (!rc) rc=co_view_init(&dst,y.buffer,y.offset+(size_t)base*2,&t);
            if (!rc) rc=co_emit_lut(&p[idx++],src,dst,CO_HARDSIGMOID);
        }
    }
    *count=idx;
    return rc;
}

static int pack_weight_tiles(co_linear_hsig *c, const _Float16 *weights) {
    _Float16 *tile=malloc((size_t)c->n*CO_K_TILE*2);
    if (!tile) return ROCKET_E_NOMEM;
    int rc=0;
    for (unsigned i=0; i<c->ktiles && !rc; i++) {
        unsigned k0=i*CO_K_TILE, kt=co_k_tile_length(c->k,i);
        co_tensor wt;
        rc=co_tensor_weight(&wt,c->n,kt);
        if (!rc) rc=co_buffer_alloc(c->fd,wt.bytes,&c->wtiles[i]);
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

void co_linear_hsig_destroy(co_linear_hsig *c) {
    if (!c) return;
    for (unsigned i=0; i<L_COUNT; i++) co_buffer_free(&c->buffers[i]);
    for (unsigned i=0; i<CO_K_TILES; i++) co_buffer_free(&c->wtiles[i]);
    free(c);
}

int co_linear_hsig_create(int fd, unsigned m, unsigned k, unsigned n,
                          const _Float16 *weights, const _Float16 *bias,
                          enum co_mode mode, co_linear_hsig **out) {
    if (!out) return ROCKET_E_SHAPE;
    *out=NULL;
    co_tensor input,output,identity;
    if (fd<0 || (mode!=CO_SEPARATE && mode!=CO_CHAIN) ||
        !m || m>256 || !k || k>CO_MAX_K || !n || n>256 ||
        co_tensor_input(&input,m,k) || co_tensor_matrix(&output,m,n,CO_F16) ||
        co_tensor_weight(&identity,output.physical_cols,output.physical_cols) ||
        !finite_vector(weights,(size_t)n*k) || !finite_vector(bias,n)) return ROCKET_E_SHAPE;
    if (mode==CO_CHAIN && !rocket_batched_submit_supported()) return ROCKET_E_UNSUPPORTED;
    co_linear_hsig *c=calloc(1,sizeof(*c));
    if (!c) return ROCKET_E_NOMEM;
    c->fd=fd;c->m=m;c->k=k;c->n=n;c->mode=mode;c->ktiles=co_k_tiles(k);
    co_tensor desc[L_COUNT]={input,output,identity,output,output,output,output};
    int rc=0;
    for (unsigned i=0; i<L_COUNT; i++) {
        if ((rc=co_buffer_alloc(fd,desc[i].bytes,&c->buffers[i])) ||
            (rc=co_view_init(&c->views[i],&c->buffers[i],0,&desc[i]))) goto failed;
    }
    if ((rc=pack_weight_tiles(c,weights)) ||
        (rc=co_bias_upload(&c->views[L_BIAS],bias)) ||
        (rc=co_identity_upload(&c->views[L_IDENTITY])) ||
        (rc=emit(c,c->views[L_INPUT],c->views[L_OUTPUT],c->programs,&c->nprograms))) goto failed;
    *out=c;return 0;
failed:co_linear_hsig_destroy(c);return rc;
}

static int prepare_outputs(co_linear_hsig *c, co_view y) {
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

static int guarded_execute(co_linear_hsig *c, const co_program *p, unsigned count, co_view y) {
    int rc=prepare_outputs(c,y);
    if (!rc) rc=co_execute(c->fd,p,count,c->mode);
    for (unsigned i=0; !rc && i<L_COUNT; i++) rc=co_guard_check(&c->buffers[i]);
    for (unsigned i=0; !rc && i<CO_K_TILES; i++)
        if (c->wtiles[i].bo.handle) rc=co_guard_check(&c->wtiles[i]);
    if (!rc && y.buffer!=&c->buffers[L_OUTPUT]) rc=co_guard_check(y.buffer);
    return rc;
}

int co_linear_hsig_run(co_linear_hsig *c, const _Float16 *x, _Float16 *y) {
    if (!c || !y || !finite_vector(x,(size_t)c->m*c->k)) return ROCKET_E_SHAPE;
    int rc=co_upload_f16(&c->views[L_INPUT],x);
    if (!rc) rc=guarded_execute(c,c->programs,c->nprograms,c->views[L_OUTPUT]);
    if (!rc) rc=co_download(&c->views[L_OUTPUT],y);
    if (!rc && !finite_vector(y,(size_t)c->m*c->n)) rc=ROCKET_E_DEVICE;
    return rc;
}

int co_linear_hsig_run_cube(co_linear_hsig *c, co_view x, co_view y) {
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
