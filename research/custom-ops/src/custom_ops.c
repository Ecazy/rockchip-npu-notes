// SPDX-License-Identifier: GPL-3.0-or-later
#include "custom_ops.h"
#include "npu_matmul.h"
#include "npu_activation.h"
#include "npu_pool.h"
#include "rocket_chain.h"
#include "rocket_hw_profile.h"
#include "reformat_template.h"
#include "co_execute_internal.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

static unsigned round_up(unsigned x, unsigned a) { return (x+a-1)/a*a; }
static unsigned width(enum co_type t) { return t == CO_F16 ? 2 : 4; }

int co_tensor_matrix(co_tensor *t, unsigned m, unsigned n, enum co_type type) {
    if (!t || !m || !n || m>256 || n>256 || (type!=CO_F16 && type!=CO_F32))
        return ROCKET_E_SHAPE;
    *t=(co_tensor){.type=type,.layout=CO_CUBE,.rows=m,.cols=n,
        .physical_rows=round_up(m<12?12:m,4),.physical_cols=round_up(n,32),
        .atom=type==CO_F16?8:4};
    t->bytes=(size_t)t->physical_rows*t->physical_cols*width(type);
    return 0;
}
int co_tensor_input(co_tensor *t, unsigned m, unsigned k) {
    if (!t || !m || m>256 || !k || k>CO_MAX_K) return ROCKET_E_SHAPE;
    *t=(co_tensor){.type=CO_F16,.layout=CO_CUBE,.rows=m,.cols=k,
        .physical_rows=round_up(m<12?12:m,4),.physical_cols=round_up(k,32),.atom=8};
    t->bytes=(size_t)t->physical_rows*t->physical_cols*2;
    return 0;
}
unsigned co_k_tiles(unsigned k) {
    return k ? (k + CO_K_TILE - 1u) / CO_K_TILE : 0;
}
unsigned co_k_tile_length(unsigned k, unsigned index) {
    unsigned k0=index*CO_K_TILE;
    if (k0>=k) return 0;
    unsigned rem=k-k0;
    return rem>CO_K_TILE?CO_K_TILE:rem;
}
size_t co_k_tile_offset(const co_tensor *full, unsigned k0) {
    if (!full || full->layout!=CO_CUBE) return 0;
    return (size_t)k0*full->physical_rows*2;
}
int co_tensor_input_slice(co_tensor *slice, const co_tensor *full,
                          unsigned k0, unsigned kt, size_t *offset) {
    if (!slice || !full || full->layout!=CO_CUBE || full->type!=CO_F16 || full->atom!=8 ||
        !full->rows || !full->cols || full->physical_rows<12 || full->physical_rows%4 ||
        full->physical_cols%32 || k0%CO_K_TILE || !kt || kt>CO_K_TILE || k0>=full->cols ||
        k0+kt>full->cols || k0+round_up(kt,32)>full->physical_cols) return ROCKET_E_SHAPE;
    unsigned pc=round_up(kt,32);
    co_tensor s={.type=CO_F16,.layout=CO_CUBE,.rows=full->rows,.cols=kt,
        .physical_rows=full->physical_rows,.physical_cols=pc,.atom=8,
        .bytes=(size_t)full->physical_rows*pc*2};
    if (s.bytes>full->bytes) return ROCKET_E_SHAPE;
    *slice=s;
    if (offset) *offset=co_k_tile_offset(full,k0);
    return 0;
}
int co_tensor_weight(co_tensor *t, unsigned n, unsigned k) {
    if (!t || !n || !k || n>256 || k>256) return ROCKET_E_SHAPE;
    *t=(co_tensor){.type=CO_F16,.layout=CO_WEIGHT,.rows=n,.cols=k,
        .physical_rows=round_up(n,32),.physical_cols=round_up(k,32),.atom=32};
    t->bytes=(size_t)t->physical_rows*t->physical_cols*2;
    return 0;
}
int co_tensor_spatial(co_tensor *t,unsigned c,unsigned h,unsigned w) {
    if(!t||!c||c>256||!h||!w||h>63||w>63||h*w>256)return ROCKET_E_SHAPE;
    *t=(co_tensor){.type=CO_F16,.layout=CO_CUBE,.rows=h*w,.cols=c,
        .physical_rows=h*w,.physical_cols=round_up(c,32),.atom=8};
    t->bytes=(size_t)t->physical_rows*t->physical_cols*2;return 0;
}
int co_tensor_linear(co_tensor *t,unsigned n) {
    if(!t||!n||n>65520)return ROCKET_E_SHAPE;
    *t=(co_tensor){.type=CO_F16,.layout=CO_LINEAR,.rows=1,.cols=n,
        .physical_rows=1,.physical_cols=n,.atom=1,.bytes=(size_t)n*2};return 0;
}
size_t co_index(const co_tensor *t, unsigned r, unsigned c) {
    if(t->layout==CO_LINEAR)return (size_t)r*t->physical_cols+c;
    if (t->layout==CO_WEIGHT)
        return (((size_t)(r/16)*(t->physical_cols/32)+c/32)*16+r%16)*32+c%32;
    return ((size_t)(c/t->atom)*t->physical_rows+r)*t->atom+c%t->atom;
}
int co_buffer_alloc(int fd, size_t payload, co_buffer *b) {
    if (!b || !payload || payload>SIZE_MAX-CO_SLACK-CO_GUARD) return ROCKET_E_SHAPE;
    memset(b,0,sizeof(*b)); b->fd=fd; b->payload=payload;
    int rc=rocket_bo_alloc32(fd,payload+CO_SLACK+CO_GUARD,&b->bo);
    if (rc) return rc;
    if ((rc=rocket_bo_prep(fd,&b->bo,1,0))) { co_buffer_free(b); return rc; }
    memset(b->bo.ptr,0,b->bo.size);
    memset((char *)b->bo.ptr+payload+CO_SLACK,0xa5,CO_GUARD);
    rc=rocket_bo_fini(fd,&b->bo);
    if (rc) co_buffer_free(b);
    return rc;
}
void co_buffer_free(co_buffer *b) {
    if (b && b->bo.handle) rocket_bo_free(b->fd,&b->bo);
    if (b) memset(b,0,sizeof(*b));
}
int co_view_init(co_view *v, co_buffer *b, size_t offset, const co_tensor *t) {
    if (!v || !b || !t || !b->bo.handle || !t->bytes ||
        (t->type!=CO_F16&&t->type!=CO_F32)||(t->layout!=CO_CUBE&&t->layout!=CO_WEIGHT&&t->layout!=CO_LINEAR)||
        !t->rows||!t->cols||t->rows>t->physical_rows||t->cols>t->physical_cols||
        t->physical_rows>256||t->physical_cols>(t->layout==CO_LINEAR?65520u:
            (t->layout==CO_WEIGHT?256u:CO_MAX_K))||
        t->atom!=(t->layout==CO_LINEAR?1:(t->layout==CO_WEIGHT?32:(t->type==CO_F16?8:4)))||
        t->physical_cols%t->atom||
        (t->layout==CO_WEIGHT&&(t->type!=CO_F16||t->physical_rows%16))||
        t->bytes!=(size_t)t->physical_rows*t->physical_cols*width(t->type)||
        (offset&15) || offset>b->payload ||
        t->bytes>b->payload-offset || b->bo.dma_address>UINT32_MAX ||
        offset>UINT32_MAX-b->bo.dma_address ||
        t->bytes-1>UINT32_MAX-(b->bo.dma_address+offset)) return ROCKET_E_SHAPE;
    *v=(co_view){b,offset,*t}; return 0;
}
static int valid(co_view v) {
    co_view check;
    return !co_view_init(&check,v.buffer,v.offset,&v.tensor);
}
static uint32_t address(co_view v) { return (uint32_t)(v.buffer->bo.dma_address+v.offset); }
static int begin(co_view *v,int write) {
    if (!v || !valid(*v)) return ROCKET_E_SHAPE;
    return rocket_bo_prep(v->buffer->fd,&v->buffer->bo,write,2000000000ull);
}
int co_upload_f16(co_view *v, const _Float16 *src) {
    if (!src || !v || v->tensor.type!=CO_F16) return ROCKET_E_SHAPE;
    int rc=begin(v,1); if(rc) return rc;
    _Float16 *dst=(_Float16 *)((char *)v->buffer->bo.ptr+v->offset);
    memset(dst,0,v->tensor.bytes);
    /* Two-byte element copies: memcpy() with a constant small size is a call in
     * an unoptimised build, and this loop runs once per logical element. */
    for(unsigned r=0;r<v->tensor.rows;r++) {
        size_t row=(size_t)r*v->tensor.cols;
        for(unsigned c=0;c<v->tensor.cols;c++)
            dst[co_index(&v->tensor,r,c)]=src[row+c];
    }
    return rocket_bo_fini(v->buffer->fd,&v->buffer->bo);
}
/* Fast exact download for the canonical CUBE FP16 layout.  The generic loop
 * walks every element and evaluates co_index() per element.  CUBE stores eight
 * consecutive logical columns as one contiguous 16-byte atom inside each row,
 * so an atom-wise copy preserves the exact row-major output order while removing
 * the inner index arithmetic.  The logical rows/cols loop is unchanged, so
 * padding is neither read nor published. */
static void download_cube_f16_rowmajor(const co_tensor *t, const char *src,
                                       void *dst) {
    const uint16_t *s = (const uint16_t *)src;
    uint16_t *d = (uint16_t *)dst;
    for (unsigned r = 0; r < t->rows; r++) {
        for (unsigned c0 = 0; c0 < t->cols; c0 += 8u) {
            unsigned n = t->cols - c0 < 8u ? t->cols - c0 : 8u;
            const uint16_t *atom =
                s + (((size_t)(c0 / 8u) * t->physical_rows + r) * 8u);
            uint16_t *out = d + (size_t)r * t->cols + c0;
            if (n == 8u) {
                memcpy(out, atom, 16u);
            } else {
                memcpy(out, atom, (size_t)n * 2u);
            }
        }
    }
}

int co_download(co_view *v, void *dst) {
    if (!dst) return ROCKET_E_SHAPE;
    int rc=begin(v,0); if(rc) return rc;
    unsigned b=width(v->tensor.type);
    char *src=(char *)v->buffer->bo.ptr+v->offset;
    if(v->tensor.layout==CO_CUBE && v->tensor.type==CO_F16 &&
       v->tensor.atom==8) {
        download_cube_f16_rowmajor(&v->tensor,src,dst);
        return rocket_bo_fini(v->buffer->fd,&v->buffer->bo);
    }
    /* Same elements, same order, same bytes as the memcpy loop; only the copy
     * itself is specialised on the element width (2 for FP16, 4 for FP32). */
    for(unsigned r=0;r<v->tensor.rows;r++) {
        size_t drow=(size_t)r*v->tensor.cols;
        for(unsigned c=0;c<v->tensor.cols;c++) {
            size_t idx=co_index(&v->tensor,r,c);
            if(b==2) ((uint16_t *)dst)[drow+c]=((const uint16_t *)src)[idx];
            else if(b==4) ((uint32_t *)dst)[drow+c]=((const uint32_t *)src)[idx];
            else memcpy((char *)dst+(drow+c)*b,src+idx*b,b);
        }
    }
    return rocket_bo_fini(v->buffer->fd,&v->buffer->bo);
}
int co_sentinel(co_buffer *b) {
    if (!b || !b->bo.handle) return ROCKET_E_SHAPE;
    int rc=rocket_bo_prep(b->fd,&b->bo,1,2000000000ull); if(rc) return rc;
    memset(b->bo.ptr,0xff,b->payload); /* NaN in both f16 and f32 */
    memset((char *)b->bo.ptr+b->payload+CO_SLACK,0xa5,CO_GUARD);
    return rocket_bo_fini(b->fd,&b->bo);
}
/* Exact equality check over a byte run, words at a time.  Coverage is identical
 * to a byte loop (every byte in [p,p+n) must equal `value`); only the failure
 * detection strategy differs, so a corrupt slack/guard byte is still caught.
 * CO_SLACK is 32 KiB, which made the byte loop the single largest Host cost of a
 * small plan call, larger than either cache-sync ioctl. */
static int bytes_all_equal(const unsigned char *p, unsigned char value, size_t n) {
    const uint64_t want = 0x0101010101010101ull * (uint64_t)value;
    size_t i = 0;
    for (; i + sizeof(uint64_t) <= n; i += sizeof(uint64_t)) {
        uint64_t word;
        memcpy(&word, p + i, sizeof(word));
        if (word != want) return 0;
    }
    for (; i < n; i++) if (p[i] != value) return 0;
    return 1;
}
int co_guard_check(co_buffer *b) {
    if (!b || !b->bo.handle) return ROCKET_E_SHAPE;
    int rc=rocket_bo_prep(b->fd,&b->bo,0,2000000000ull); if(rc) return rc;
    unsigned char *p=(unsigned char *)b->bo.ptr+b->payload+CO_SLACK;
    int guard_ok=bytes_all_equal(p,0xa5,CO_GUARD);
    p=(unsigned char *)b->bo.ptr+b->payload;
    int slack_ok=bytes_all_equal(p,0x00,CO_SLACK);
    if(!guard_ok||!slack_ok)rc=ROCKET_E_DEVICE;
    int end=rocket_bo_fini(b->fd,&b->bo); return rc?rc:end;
}
static int same(co_view a,co_view b) {
    const co_tensor *x=&a.tensor,*y=&b.tensor;
    return valid(a)&&valid(b)&&x->type==y->type&&x->layout==y->layout&&
        x->rows==y->rows&&x->cols==y->cols&&x->physical_rows==y->physical_rows&&
        x->physical_cols==y->physical_cols&&x->atom==y->atom&&x->bytes==y->bytes;
}
static int overlap(co_view a, co_view b) {
    return a.buffer==b.buffer && a.offset<b.offset+b.tensor.bytes && b.offset<a.offset+a.tensor.bytes;
}
int co_copy(co_view *dst,co_view *src) {
    if(!dst||!src||!same(*dst,*src)||overlap(*dst,*src)) return ROCKET_E_SHAPE;
    void *tmp=malloc(src->tensor.bytes); if(!tmp) return ROCKET_E_NOMEM;
    int rc=begin(src,0);
    if(!rc) { memcpy(tmp,(char *)src->buffer->bo.ptr+src->offset,src->tensor.bytes);
        rc=rocket_bo_fini(src->buffer->fd,&src->buffer->bo); }
    if(!rc) { rc=begin(dst,1); if(!rc) {
        memcpy((char *)dst->buffer->bo.ptr+dst->offset,tmp,dst->tensor.bytes);
        rc=rocket_bo_fini(dst->buffer->fd,&dst->buffer->bo); } }
    free(tmp); return rc;
}
int co_bias_upload(co_view *v,const _Float16 *bias) {
    if(!v||!bias||v->tensor.type!=CO_F16||v->tensor.layout!=CO_CUBE) return ROCKET_E_SHAPE;
    _Float16 *a=malloc((size_t)v->tensor.rows*v->tensor.cols*2);
    if(!a) return ROCKET_E_NOMEM;
    for(unsigned r=0;r<v->tensor.rows;r++) memcpy(a+(size_t)r*v->tensor.cols,bias,v->tensor.cols*2);
    int rc=co_upload_f16(v,a); free(a); return rc;
}
int co_identity_upload(co_view *v) {
    if(!v||v->tensor.layout!=CO_WEIGHT||v->tensor.rows!=v->tensor.cols) return ROCKET_E_SHAPE;
    unsigned n=v->tensor.rows;
    _Float16 *a=calloc((size_t)n*n,2); if(!a) return ROCKET_E_NOMEM;
    for(unsigned i=0;i<n;i++) a[(size_t)i*n+i]=1;
    int rc=co_upload_f16(v,a); free(a); return rc;
}
int co_ones_upload(co_view *v,unsigned k,int mean) {
    if(!v||!k||v->tensor.layout!=CO_WEIGHT||v->tensor.rows!=16||k!=v->tensor.cols)
        return ROCKET_E_SHAPE;
    _Float16 *a=calloc((size_t)16*k,2); if(!a) return ROCKET_E_NOMEM;
    for(unsigned i=0;i<k;i++) a[i]=(_Float16)(mean?1.0/(double)k:1.0);
    int rc=co_upload_f16(v,a); free(a); return rc;
}
static int finish_mm_op(co_program *p,co_view x,co_view w,co_view o,co_view *operand,int mul,int ew_op) {
    if(!p||!valid(x)||!valid(w)||!valid(o)||x.tensor.type!=CO_F16||w.tensor.type!=CO_F16||
       x.tensor.layout!=CO_CUBE||w.tensor.layout!=CO_WEIGHT||o.tensor.layout!=CO_CUBE||
       (operand?x.tensor.physical_cols!=w.tensor.cols:x.tensor.cols!=w.tensor.cols)||
       x.tensor.physical_cols!=w.tensor.physical_cols||
       /* Primitive emitters stay single-tile even though a full input cube may be wider. */
       x.tensor.physical_cols>256||x.tensor.physical_rows>256||
       w.tensor.physical_cols>256||w.tensor.physical_rows>256||o.tensor.physical_cols>256||
       x.tensor.physical_rows<12||x.tensor.physical_rows%4||x.tensor.physical_cols%32||
       o.tensor.rows!=x.tensor.rows||
       o.tensor.physical_rows!=x.tensor.physical_rows||
       o.tensor.physical_cols!=w.tensor.physical_rows||overlap(x,o)||overlap(w,o)) return ROCKET_E_SHAPE;
    if(operand&&(!same(x,*operand)||!same(x,o)||overlap(*operand,o))) return ROCKET_E_SHAPE;
    if(!operand && o.tensor.cols!=w.tensor.rows && !(o.tensor.cols==1&&w.tensor.rows==16))
        return ROCKET_E_SHAPE;
    memset(p,0,sizeof(*p));
    matmul_params_t a={.m=x.tensor.physical_rows,.k=x.tensor.physical_cols,.n=o.tensor.physical_cols,
        .input_dma=address(x),.weights_dma=address(w),.output_dma=address(o),
        .tasks=p->words,.fp32tofp16=o.tensor.type==CO_F16,.accumulate=operand!=NULL,
        .ew_mul=mul,.ew_op=(uint8_t)ew_op,
        .add_dma=operand?address(*operand):0};
    int rc=gen_matmul_fp16(&a); if(rc) return rc;
    p->count=a.task_count;p->reads[0]=x;p->reads[1]=w;p->nreads=2;p->output=o;
    if(operand) p->reads[p->nreads++]=*operand;
    return p->count&&p->count<=CO_MAX_WORDS?0:ROCKET_E_TILING;
}
static int finish_mm(co_program *p,co_view x,co_view w,co_view o,co_view *operand,int mul) {
    return finish_mm_op(p,x,w,o,operand,mul,0);
}
int co_emit_mm(co_program *p,co_view x,co_view w,co_view o) { return finish_mm(p,x,w,o,NULL,0); }
int co_emit_mm_acc(co_program *p,co_view x,co_view w,co_view prior,co_view o) {
    if(!p||!valid(x)||!valid(w)||!valid(prior)||!valid(o)||
       x.tensor.type!=CO_F16||w.tensor.type!=CO_F16||prior.tensor.type!=CO_F16||o.tensor.type!=CO_F16||
       x.tensor.layout!=CO_CUBE||w.tensor.layout!=CO_WEIGHT||
       prior.tensor.layout!=CO_CUBE||o.tensor.layout!=CO_CUBE||
       x.tensor.cols!=w.tensor.cols||x.tensor.physical_cols!=w.tensor.physical_cols||
       /* One K tile only, and never an alias of the partial it accumulates into. */
       x.tensor.physical_cols>256||x.tensor.physical_rows>256||
       w.tensor.physical_cols>256||w.tensor.physical_rows>256||o.tensor.physical_cols>256||
       x.tensor.physical_rows<12||x.tensor.physical_rows%4||x.tensor.physical_cols%32||
       !same(prior,o)||o.tensor.rows!=x.tensor.rows||
       o.tensor.physical_rows!=x.tensor.physical_rows||
       o.tensor.physical_cols!=w.tensor.physical_rows||
       /* The prior and the destination are the ping-pong pair: distinct BOs, so the
        * ERDMA never reads the surface the WDMA is writing. */
       prior.buffer==o.buffer||overlap(prior,o)||overlap(x,o)||overlap(x,prior)||
       overlap(w,o)||overlap(w,prior)||
       (o.tensor.cols!=w.tensor.rows&&!(o.tensor.cols==1&&w.tensor.rows==16))) return ROCKET_E_SHAPE;
    memset(p,0,sizeof(*p));
    matmul_params_t a={.m=x.tensor.physical_rows,.k=x.tensor.physical_cols,.n=o.tensor.physical_cols,
        .input_dma=address(x),.weights_dma=address(w),.output_dma=address(o),
        .tasks=p->words,.fp32tofp16=1,.accumulate=1,.ew_mul=0,.add_dma=address(prior)};
    int rc=gen_matmul_fp16(&a); if(rc) return rc;
    p->count=a.task_count;p->reads[0]=x;p->reads[1]=w;p->reads[2]=prior;p->nreads=3;p->output=o;
    return p->count&&p->count<=CO_MAX_WORDS?0:ROCKET_E_TILING;
}
int co_emit_ew(co_program *p,co_view x,co_view b,co_view id,co_view o,int mul) {
    if(mul!=0&&mul!=1) return ROCKET_E_SHAPE;
    if(id.tensor.rows!=x.tensor.physical_cols || id.tensor.cols!=x.tensor.physical_cols)
        return ROCKET_E_SHAPE;
    return finish_mm(p,x,id,o,&b,mul);
}
int co_emit_ew_max(co_program *p,co_view x,co_view b,co_view id,co_view o) {
    if(id.tensor.rows!=x.tensor.physical_cols || id.tensor.cols!=x.tensor.physical_cols)
        return ROCKET_E_SHAPE;
    return finish_mm_op(p,x,id,o,&b,0,2); /* ew_op=2 => EW ALU MAX */
}
int co_emit_lut(co_program *p,co_view x,co_view o,enum co_act kind) {
    if(!p||!same(x,o)||x.tensor.type!=CO_F16||
       (x.tensor.layout!=CO_CUBE&&x.tensor.layout!=CO_LINEAR)||overlap(x,o)||
       (kind!=CO_HARDSIGMOID&&kind!=CO_SIGMOID&&kind!=CO_GELU_GATE&&
        kind!=CO_EXP_BOUNDED&&kind!=CO_RSQRT_BOUNDED&&kind!=CO_EXP_NEG4)) return ROCKET_E_SHAPE;
    unsigned n=x.tensor.physical_rows*x.tensor.physical_cols;
    if(n/8>=8191) return ROCKET_E_TILING;
    uint16_t lut[1026];
    for(unsigned bank=0;bank<2;bank++) for(unsigned i=0;i<=512;i++) {
        double v=(bank?(double)i:(double)i-512)*32.0/(kind==CO_EXP_NEG4?4096.0:2596.0);
        /* Output converter decodes q/32768 times S: S=64 for bounded exp,
         * S=4 for positive-domain rsqrt; unit activations keep S=1. */
        double y=kind==CO_HARDSIGMOID?fmin(1,fmax(0,v/6+0.5)):
            kind==CO_SIGMOID?1/(1+exp(-v)):
            kind==CO_GELU_GATE?0.5*(1+erf(v/sqrt(2.0))):
            /* Clamp BELOW the declared rsqrt input domain, so interpolation
             * around x=0.25 samples the smooth function on BOTH sides. */
            kind==CO_EXP_BOUNDED?exp(v)/64.0:
            kind==CO_EXP_NEG4?exp(fmin(0,v)):1/(4*sqrt(fmax(0.125,v)));
        long q=lround(y*32768); if(q<1)q=1; if(q>32767)q=32767;
        lut[bank*513+i]=(uint16_t)q;
    }
    memset(p,0,sizeof(*p));
    lut_act_params_t a={.input_dma=address(x),.output_dma=address(o),.n=(int)n,.lut=lut,
        .bn_mul_operand=kind==CO_EXP_NEG4?0x6c00:0x6912,.bn_alu_cfg=0x80000000,.out_cvt_offset=1,.out_cvt_scale=1,
        .out_cvt_minus_exp=kind==CO_EXP_BOUNDED?9:(kind==CO_RSQRT_BOUNDED?13:15),
        .out_cvt_cvt_type=1,.lut_le_start=0xffffc000,.lut_lo_end=0x4000,
        .le_slope_scale=23107,.le_slope_shift=22,.le_index_select=5,.lo_index_select=5,.tasks=p->words};
    int rc=gen_lut_activation_fp16(&a); if(rc) return rc;
    p->count=(uint32_t)a.task_count;p->reads[0]=x;p->nreads=1;p->output=o;
    return p->count&&p->count<=CO_MAX_WORDS?0:ROCKET_E_TILING;
}
int co_emit_scale_native(co_program *p,co_view x,co_view o,_Float16 scalar){
    if(!p||!same(x,o)||x.tensor.type!=CO_F16||
       (x.tensor.layout!=CO_CUBE&&x.tensor.layout!=CO_LINEAR)||
       x.buffer->fd!=o.buffer->fd||x.buffer->bo.handle==o.buffer->bo.handle||
       x.tensor.bytes/2>32768||!isfinite((float)scalar))return ROCKET_E_SHAPE;
    uint16_t lut[1026]={0},bits;memcpy(&bits,&scalar,2);
    uint64_t words[CO_MAX_WORDS];
    lut_act_params_t a={.input_dma=address(x),.output_dma=address(o),
        .n=(int)(x.tensor.bytes/2),.lut=lut,.bn_mul_operand=bits,
        .out_cvt_scale=1,.tasks=words};
    int rc=gen_lut_activation_fp16(&a);if(rc)return rc;
    memset(p,0,sizeof(*p));
    for(int i=0;i<a.task_count;i++){
        uint16_t reg=(uint16_t)words[i];uint32_t val=(uint32_t)(words[i]>>16);
        /* LUT is bypassed: omit table access AND LUT control registers. */
        if(reg>=0x4100&&reg<=0x41ff)continue;
        if(reg==0x4060)val=(1u<<6)|(1u<<1); /* BN active, ALU/RELU bypass, MUL scalar */
        if(reg==0x4070)val=(1u<<9)|(1u<<8)|(1u<<7)|(1u<<1)|1u;
        p->words[p->count++]=(words[i]&0xffff00000000ffffull)|((uint64_t)val<<16);
    }
    p->reads[0]=x;p->nreads=1;p->output=o;return 0;
}

int co_emit_reformat_8x8x8(co_program *p,co_view x,co_view o) {
    if(!p||!same(x,o)||x.tensor.layout!=CO_LINEAR||x.tensor.type!=CO_F16||
       x.tensor.cols!=512||overlap(x,o))return ROCKET_E_SHAPE;
    memset(p,0,sizeof(*p));
    for(unsigned i=0;i<69;i++) {
        uint64_t word=co_reformat_template[i];uint16_t reg=(uint16_t)word;
        if(reg==0x4020)word=(word&0xffff00000000ffffull)|((uint64_t)address(o)<<16);
        if(reg==0x5018)word=(word&0xffff00000000ffffull)|((uint64_t)address(x)<<16);
        p->words[p->count++]=word;
    }
    p->words[p->count++]=0;
    p->words[p->count++]=((uint64_t)0x101<<48)|0x14;
    p->words[p->count++]=((uint64_t)0x41<<48);
    p->words[p->count++]=((uint64_t)0x81<<48)|((uint64_t)0x18<<16)|8;
    p->reads[0]=x;p->nreads=1;p->output=o;return 0;
}
int co_emit_pool(co_program *p,co_view x,co_view o,unsigned h,unsigned w,unsigned kh,unsigned kw,int avg) {
    if(!p||!valid(x)||!valid(o)||x.tensor.type!=CO_F16||o.tensor.type!=CO_F16||
       x.tensor.layout!=CO_CUBE||o.tensor.layout!=CO_CUBE||!h||!w||h>63||w>63||
       !kh||!kw||kh>16||kw>16||h%kh||w%kw||(avg!=0&&avg!=1)||(avg && (kh<2||kw<2))||
       x.tensor.rows!=h*w||o.tensor.rows!=(h/kh)*(w/kw)||x.tensor.cols!=o.tensor.cols||
       x.tensor.physical_cols!=o.tensor.physical_cols||
       x.tensor.physical_rows!=h*w||o.tensor.physical_rows!=(h/kh)*(w/kw)||overlap(x,o))
        return ROCKET_E_SHAPE;
    memset(p,0,sizeof(*p));
    pool_params_t a={.c=x.tensor.physical_cols,.ih=h,.iw=w,.oh=h/kh,.ow=w/kw,
        .kh=kh,.kw=kw,.stride_y=kh,.stride_x=kw,.method=avg?POOL_METHOD_AVG:POOL_METHOD_MAX,
        .recip_w=ppu_recip_kernel_fp16(kw),.recip_h=ppu_recip_kernel_fp16(kh),
        .input_dma=address(x),.output_dma=address(o),.tasks=p->words};
    int rc=gen_pool_fp16(&a);if(rc)return rc;
    p->count=a.task_count;p->flags=ROCKET_JOB_PPU_DONE;p->reads[0]=x;p->nreads=1;p->output=o;
    return 0;
}
static int add_handle(uint32_t *h,unsigned *n,uint32_t value) {
    for(unsigned i=0;i<*n;i++) if(h[i]==value) return 0;
    if(*n>=CO_MAX_PROGRAMS*3+1)return ROCKET_E_SHAPE;
    h[(*n)++]=value;return 0;
}
static double co_clock_us(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
int co_execute_workspace_init(int fd,unsigned n,co_execute_workspace *w){
    if(!w||fd<0||!n||n>CO_MAX_PROGRAMS)return ROCKET_E_SHAPE;
    memset(w,0,sizeof(*w));w->fd=fd;
    return rocket_bo_alloc32(fd,(size_t)n*rkt_chain_words(CO_MAX_WORDS)*8,&w->commands);
}
void co_execute_workspace_destroy(co_execute_workspace *w){
    if(w&&w->commands.handle)rocket_bo_free(w->fd,&w->commands);
    if(w)memset(w,0,sizeof(*w));
}
/* Whole-array validation shared by every submission and by command-plan
 * construction.  It must run before CO_SEPARATE fans out: otherwise a later
 * invalid program would only be discovered after the earlier programs had
 * already been submitted. */
static int validate_program_set(int fd,const co_program *p,unsigned n,
                                 enum co_mode mode,
                                 co_execute_workspace *workspace) {
    if(workspace&&(workspace->fd!=fd||!workspace->commands.handle))return ROCKET_E_SHAPE;
    if(fd<0||!p||!n||n>CO_MAX_PROGRAMS||(mode!=CO_SEPARATE&&mode!=CO_CHAIN))return ROCKET_E_SHAPE;
    if(strcmp(rocket_hw_current()->name,"rk3588"))return ROCKET_E_UNSUPPORTED;
    if(mode==CO_CHAIN&&!rocket_batched_submit_supported())return ROCKET_E_UNSUPPORTED;
    /* Mixing PPU/DPU completion classes is intentionally not supported. */
    for(unsigned i=0;i<n;i++) {
        if(!p[i].count||p[i].count>CO_MAX_WORDS||p[i].nreads>3||!valid(p[i].output)||
           p[i].output.buffer->fd!=fd)return ROCKET_E_SHAPE;
        if(mode==CO_CHAIN&&p[i].flags!=p[0].flags)return ROCKET_E_UNSUPPORTED;
        if(workspace&&workspace->commands.handle==p[i].output.buffer->bo.handle)return ROCKET_E_SHAPE;
        for(unsigned j=0;j<p[i].nreads;j++)
            if(!valid(p[i].reads[j])||p[i].reads[j].buffer->fd!=fd||
               (workspace&&workspace->commands.handle==p[i].reads[j].buffer->bo.handle))return ROCKET_E_SHAPE;
    }
    return 0;
}

/* Shared packing path.  This is the single place that clears the command BO,
 * copies regcmd words, writes chain links, verifies the chain and builds the
 * handle lists.  The creation-time cache calls it once; the uncached path calls
 * it once per submit.  Keeping one implementation is what makes the cached and
 * control command bytes comparable by construction. */
static int command_plan_build_impl(int fd,const co_program *p,unsigned n,
                                   enum co_mode mode,
                                   co_execute_workspace *workspace,
                                   co_command_plan *plan,
                                   co_command_times *times,int validate) {
    int detail=times?times->detail:0;
    if(times){memset(times,0,sizeof(*times));times->detail=detail;}
    if(!plan)return ROCKET_E_SHAPE;
    /* Avoid a full 3 KiB struct clear on the per-submit control path: every
     * field and every array entry below ``n``/``n_sync`` is initialised here. */
    plan->fd=fd;plan->programs=p;plan->mode=mode;plan->n_tasks=n;
    plan->commands=(rocket_bo){0};plan->owns_commands=0;
    plan->n_in=0;plan->n_out=0;plan->flags=0;plan->words=0;plan->n_sync=0;
    int rc=0;
    if(validate) {
        rc=validate_program_set(fd,p,n,mode,workspace);
        if(rc)return rc;
    }
    if(mode==CO_SEPARATE&&n>1)return ROCKET_E_SHAPE; /* execute_impl recurses first */
    size_t offs[CO_MAX_PROGRAMS],words=0;
    for(unsigned i=0;i<n;i++){offs[i]=words;words+=rkt_chain_words(p[i].count);}
    double start=times?co_clock_us():0;
    if(workspace){if(workspace->commands.size<words*8)return ROCKET_E_SHAPE;plan->commands=workspace->commands;}
    else {rc=rocket_bo_alloc32(fd,words*8,&plan->commands);plan->owns_commands=1;}
    if(times)times->allocation_us=co_clock_us()-start;
    if(rc)return rc;
    plan->words=words;
    /* The public command_us interval is one continuous interval from before
     * PREP through the handle lists.  Sub-intervals are recorded only for the
     * independent breakdown probe; the production path performs just the two
     * outer timestamps so it keeps the original attribution overhead. */
    double command_start=times?co_clock_us():0;
    double part_start=(times&&detail)?co_clock_us():0;
    if((rc=rocket_bo_prep(fd,&plan->commands,1,0)))goto bad;
    if(times&&detail)times->sync_us+=co_clock_us()-part_start;
    if(times&&detail)part_start=co_clock_us();
    memset(plan->commands.ptr,0,words*8);
    if(times&&detail)times->clear_us=co_clock_us()-part_start;
    if(times&&detail)part_start=co_clock_us();
    for(unsigned i=0;i<n;i++) {
        if(mode==CO_CHAIN)rc=rkt_chain_pack_at(&plan->commands,plan->tasks,(int)i,offs[i],
            i+1<n?offs[i+1]:offs[i],p[i].words,p[i].count,i+1<n?p[i+1].count:p[i].count);
        else {memcpy(plan->commands.ptr,p[i].words,p[i].count*8);
              plan->tasks[i]=(rocket_task_desc){(uint32_t)plan->commands.dma_address,p[i].count};}
        if(rc)break;
    }
    if(times&&detail)times->pack_us=co_clock_us()-part_start;
    if(times&&detail)part_start=co_clock_us();
    if(!rc&&mode==CO_CHAIN)rc=rkt_chain_verify(&plan->commands,plan->tasks,(int)n);
    if(times&&detail)times->verify_us=co_clock_us()-part_start;
    part_start=(times&&detail)?co_clock_us():0;
    {
        int r=rocket_bo_fini(fd,&plan->commands);
        if(times&&detail)times->sync_us+=co_clock_us()-part_start;
        if(!rc)rc=r;
    }
    if(rc)goto bad;
    if(times&&detail)part_start=co_clock_us();
    unsigned ni=0,no=0;
    for(unsigned i=0;i<n;i++)add_handle(plan->outs,&no,p[i].output.buffer->bo.handle);
    for(unsigned i=0;i<n;i++)for(unsigned j=0;j<p[i].nreads;j++) {
        uint32_t h=p[i].reads[j].buffer->bo.handle;int written=0;
        for(unsigned k=0;k<no;k++)if(plan->outs[k]==h)written=1;
        /* A device intermediate belongs only in out_handles, never both lists. */
        if(!written)add_handle(plan->ins,&ni,h);
    }
    add_handle(plan->ins,&ni,plan->commands.handle);
    plan->n_in=ni;plan->n_out=no;
    plan->flags=p[0].flags|(mode==CO_CHAIN?ROCKET_JOB_BATCHED:0);
    for(unsigned i=0;i<n;i++) {
        co_buffer *buffer=p[i].output.buffer;
        plan->program_outputs[i]=buffer;
    }
    /* Record the original post-submit dedup set, preserving the provider's
     * full-object identity guard: handle is the BO identity within fd, but only
     * identical backing objects may merge a prep/fini pair. */
    for(unsigned i=0;i<n;i++) {
        co_buffer *buffer=p[i].output.buffer;
        uint32_t handle=buffer->bo.handle;
        int duplicate=0;
        for(unsigned j=0;j<plan->n_sync;j++) {
            co_buffer *seen=plan->sync_buffers[j];
            if(seen->fd==buffer->fd&&seen->bo.handle==handle&&
               seen->bo.obj_addr==buffer->bo.obj_addr&&
               seen->bo.size==buffer->bo.size){duplicate=1;break;}
        }
        if(!duplicate)plan->sync_buffers[plan->n_sync++]=buffer;
    }
    if(times&&detail)times->handles_us=co_clock_us()-part_start;
    if(times)times->total_us=co_clock_us()-command_start;
    return 0;
bad:
    co_command_plan_release(plan);
    return rc;
}

int co_command_plan_build(int fd,const co_program *p,unsigned n,enum co_mode mode,
                          co_execute_workspace *workspace,co_command_plan *plan,
                          co_command_times *times) {
    return command_plan_build_impl(fd,p,n,mode,workspace,plan,times,1);
}

void co_command_plan_release(co_command_plan *plan) {
    if(!plan)return;
    if(plan->owns_commands&&plan->commands.handle)rocket_bo_free(plan->fd,&plan->commands);
    memset(plan,0,sizeof(*plan));
}

int co_command_plan_submit(co_command_plan *plan,int sync_command,
                           co_execute_profile *prof) {
    if(!plan||!plan->n_tasks||plan->fd<0||!plan->commands.handle)return ROCKET_E_SHAPE;
    int rc=0;
    double start=prof?co_clock_us():0;
    if(sync_command) {
        rc=rocket_bo_prep(plan->fd,&plan->commands,1,0);
        if(!rc)rc=rocket_bo_fini(plan->fd,&plan->commands);
    }
    if(prof)prof->command_us+=co_clock_us()-start;
    if(rc)return rc;
    start=prof?co_clock_us():0;
    rc=rocket_submit_tasks_flags(plan->fd,plan->tasks,plan->n_tasks,
                                 plan->ins,plan->n_in,plan->outs,plan->n_out,
                                 plan->flags);
    if(prof)prof->submit_wait_us+=co_clock_us()-start;
    start=prof?co_clock_us():0;
    if(!rc) {
#ifdef CO_OUTPUT_SYNC_REFERENCE
        /* Board A/B control: preserve the historical per-program output sync. */
        for(unsigned i=0;i<plan->n_tasks;i++) {
            rocket_bo *b=&plan->program_outputs[i]->bo;
            rc=rocket_bo_prep(plan->fd,b,0,2000000000ull);if(rc)break;
            rc=rocket_bo_fini(plan->fd,b);if(rc)break;
        }
#else
        for(unsigned i=0;i<plan->n_sync;i++) {
            rocket_bo *b=&plan->sync_buffers[i]->bo;
            rc=rocket_bo_prep(plan->fd,b,0,2000000000ull);if(rc)break;
            rc=rocket_bo_fini(plan->fd,b);if(rc)break;
        }
#endif
    }
    if(prof)prof->output_sync_us+=co_clock_us()-start;
    return rc;
}

int co_execute_cached_profiled(int fd,const co_program *p,unsigned n,
                               enum co_mode mode,co_execute_workspace *workspace,
                               co_command_plan *plan,
                               co_execute_profile *prof,int sync_command) {
    if(prof)memset(prof,0,sizeof(*prof));
    if(!workspace||!plan||workspace->fd!=fd||
       workspace->commands.handle!=plan->commands.handle||
       plan->fd!=fd||plan->mode!=mode||plan->n_tasks!=n||
       plan->programs!=p)return ROCKET_E_SHAPE;
    /* The cached programs are supplied so the caller's eligibility stays
     * explicit; the packed plan itself is what this entry executes.  A cheap
     * identity check prevents a stale plan from being paired with a different
     * program array. */
    if(!p||!n)return ROCKET_E_SHAPE;
    return co_command_plan_submit(plan,sync_command,prof);
}

int co_execute_plan_profiled(const co_program *p,unsigned n,enum co_mode mode,
                             co_command_plan *plan,co_execute_profile *prof,
                             int sync_command) {
    if(prof)memset(prof,0,sizeof(*prof));
    if(!plan||!p||!n||plan->programs!=p||plan->mode!=mode||
       plan->n_tasks!=n||!plan->commands.handle)return ROCKET_E_SHAPE;
    return co_command_plan_submit(plan,sync_command,prof);
}

static int execute_impl(int fd,const co_program *p,unsigned n,enum co_mode mode,
                       co_execute_workspace *workspace,co_execute_profile *prof) {
    int rc=validate_program_set(fd,p,n,mode,workspace);
    if(rc)return rc;
    /* CO_SEPARATE submits one program per call, but the whole array must be
     * validated first: a later invalid program cannot be allowed to leave the
     * earlier ones already on the device. */
    if(mode==CO_SEPARATE&&n>1) {
        for(unsigned i=0;i<n;i++){rc=execute_impl(fd,p+i,1,mode,workspace,prof);if(rc)return rc;}return 0;
    }
    co_command_plan plan;
    co_command_times times;
    memset(&times,0,sizeof(times));
    rc=command_plan_build_impl(fd,p,n,mode,workspace,&plan,
                               prof?&times:NULL,0);
    if(prof)prof->allocation_us+=times.allocation_us;
    if(rc)return rc;
    if(prof)prof->command_us+=times.total_us;
    rc=co_command_plan_submit(&plan,0,prof);
    if(!workspace) {
        double start=co_clock_us();
        co_command_plan_release(&plan);
        if(prof)prof->allocation_us+=co_clock_us()-start;
    }
    return rc;
}

int co_execute(int fd,const co_program *p,unsigned n,enum co_mode mode){
    return execute_impl(fd,p,n,mode,NULL,NULL);
}
int co_execute_profiled(int fd,const co_program *p,unsigned n,enum co_mode mode,
                        co_execute_workspace *w,co_execute_profile *prof){
    if(prof)memset(prof,0,sizeof(*prof));
    return execute_impl(fd,p,n,mode,w,prof);
}

int co_gated_activation(int fd,co_view x,co_view o,int gelu,enum co_mode mode) {
    if(!same(x,o)||x.tensor.type!=CO_F16||x.tensor.layout!=CO_CUBE||overlap(x,o)||
       (gelu!=0&&gelu!=1))return ROCKET_E_SHAPE;
    co_tensor id;int rc=co_tensor_weight(&id,x.tensor.physical_cols,x.tensor.physical_cols);
    if(rc)return rc;
    co_buffer gate={0},identity={0};co_view vg,vi;co_program p[2];
    if((rc=co_buffer_alloc(fd,x.tensor.bytes,&gate))||
       (rc=co_buffer_alloc(fd,id.bytes,&identity)))goto end;
    if((rc=co_view_init(&vg,&gate,0,&x.tensor))||(rc=co_view_init(&vi,&identity,0,&id))||
       (rc=co_identity_upload(&vi)))goto end;
    if((rc=co_emit_lut(&p[0],x,vg,gelu?CO_GELU_GATE:CO_SIGMOID))||
       (rc=co_emit_ew(&p[1],x,vg,vi,o,1))||(rc=co_execute(fd,p,2,mode)))goto end;
    if((rc=co_guard_check(&gate)))goto end;
    rc=co_guard_check(&identity);
end:co_buffer_free(&gate);co_buffer_free(&identity);return rc;
}

int co_norm_hybrid(int fd,co_view x,co_view o,const _Float16 *gamma,const _Float16 *beta,
                   float eps,int ln,enum co_mode mode) {
    if(!gamma||!same(x,o)||x.tensor.type!=CO_F16||x.tensor.layout!=CO_CUBE||
       overlap(x,o)||!isfinite(eps)||eps<=0||(ln!=0&&ln!=1))return ROCKET_E_SHAPE;
    const unsigned m=x.tensor.rows,n=x.tensor.cols,np=x.tensor.physical_cols;
    co_tensor id,w,stats;
    int rc=co_tensor_weight(&id,np,np);if(rc)return rc;
    if((rc=co_tensor_weight(&w,16,n)))return rc;
    if((rc=co_tensor_matrix(&stats,m,1,CO_F32)))return rc;
    co_buffer b[8]={0};co_view v[8];co_program p[3];
    co_tensor desc[8]={x.tensor,stats,stats,x.tensor,x.tensor,x.tensor,id,w};
    float *ms=malloc(m*4),*mean=calloc(m,4);
    _Float16 *a=malloc((size_t)m*n*2),*bias=malloc((size_t)m*n*2);
    if(!ms||!mean||!a||!bias){rc=ROCKET_E_NOMEM;goto end;}
    for(unsigned i=0;i<8;i++) {
        if((rc=co_buffer_alloc(fd,desc[i].bytes,&b[i])))goto end;
        if((rc=co_view_init(&v[i],&b[i],0,&desc[i])))goto end;
    }
    if((rc=co_identity_upload(&v[6]))||(rc=co_ones_upload(&v[7],n,0)))goto end;
    if((rc=co_emit_ew(&p[0],x,x,v[6],v[0],1)))goto end;
    if((rc=co_emit_mm(&p[1],v[0],v[7],v[1])))goto end;
    if(ln&&((rc=co_emit_mm(&p[2],x,v[7],v[2]))))goto end;
    if((rc=co_execute(fd,p,ln?3:2,mode)))goto end;
    if((rc=co_download(&v[1],ms)))goto end;
    if(ln&&((rc=co_download(&v[2],mean))))goto end;
    for(unsigned r=0;r<m;r++) {
        double mu=ln?(double)mean[r]/n:0;
        double variance=(double)ms[r]/n-mu*mu;
        /* E[x^2] uses an fp16 square; no exact-variance claim for nearly constant rows. */
        if(!isfinite(variance)||variance<0){rc=ROCKET_E_SHAPE;goto end;}
        double scale=1/sqrt(variance+eps);
        for(unsigned c=0;c<n;c++) {
            double s=scale*(double)gamma[c];
            double z=(beta?(double)beta[c]:0)-mu*s;
            a[(size_t)r*n+c]=(_Float16)s;bias[(size_t)r*n+c]=(_Float16)z;
            if(!isfinite((double)a[(size_t)r*n+c])||!isfinite((double)bias[(size_t)r*n+c])) {
                rc=ROCKET_E_SHAPE;goto end;
            }
        }
    }
    if((rc=co_upload_f16(&v[3],a))||(rc=co_upload_f16(&v[4],bias)))goto end;
    if((rc=co_emit_ew(&p[0],x,v[3],v[6],v[5],1)))goto end;
    if((rc=co_emit_ew(&p[1],v[5],v[4],v[6],o,0)))goto end;
    if((rc=co_execute(fd,p,2,mode)))goto end;
    for(unsigned i=0;i<8;i++)if((rc=co_guard_check(&b[i])))goto end;
end:for(unsigned i=0;i<8;i++)co_buffer_free(&b[i]);free(ms);free(mean);free(a);free(bias);return rc;
}
