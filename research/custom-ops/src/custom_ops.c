// SPDX-License-Identifier: GPL-3.0-or-later
#include "custom_ops.h"
#include "npu_matmul.h"
#include "npu_activation.h"
#include "npu_pool.h"
#include "rocket_chain.h"
#include "rocket_hw_profile.h"
#include "reformat_template.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

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
    for(unsigned r=0;r<v->tensor.rows;r++) for(unsigned c=0;c<v->tensor.cols;c++)
        dst[co_index(&v->tensor,r,c)]=src[(size_t)r*v->tensor.cols+c];
    return rocket_bo_fini(v->buffer->fd,&v->buffer->bo);
}
int co_download(co_view *v, void *dst) {
    if (!dst) return ROCKET_E_SHAPE;
    int rc=begin(v,0); if(rc) return rc;
    unsigned b=width(v->tensor.type);
    char *src=(char *)v->buffer->bo.ptr+v->offset;
    for(unsigned r=0;r<v->tensor.rows;r++) for(unsigned c=0;c<v->tensor.cols;c++)
        memcpy((char *)dst+((size_t)r*v->tensor.cols+c)*b,src+co_index(&v->tensor,r,c)*b,b);
    return rocket_bo_fini(v->buffer->fd,&v->buffer->bo);
}
int co_sentinel(co_buffer *b) {
    if (!b || !b->bo.handle) return ROCKET_E_SHAPE;
    int rc=rocket_bo_prep(b->fd,&b->bo,1,2000000000ull); if(rc) return rc;
    memset(b->bo.ptr,0xff,b->payload); /* NaN in both f16 and f32 */
    memset((char *)b->bo.ptr+b->payload+CO_SLACK,0xa5,CO_GUARD);
    return rocket_bo_fini(b->fd,&b->bo);
}
int co_guard_check(co_buffer *b) {
    if (!b || !b->bo.handle) return ROCKET_E_SHAPE;
    int rc=rocket_bo_prep(b->fd,&b->bo,0,2000000000ull); if(rc) return rc;
    unsigned char *p=(unsigned char *)b->bo.ptr+b->payload+CO_SLACK;
    for(unsigned i=0;i<CO_GUARD;i++) if(p[i]!=0xa5) rc=ROCKET_E_DEVICE;
    p=(unsigned char *)b->bo.ptr+b->payload;
    for(unsigned i=0;i<CO_SLACK;i++)if(p[i]!=0)rc=ROCKET_E_DEVICE;
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
static int finish_mm(co_program *p,co_view x,co_view w,co_view o,co_view *operand,int mul) {
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
        .ew_mul=mul,.add_dma=operand?address(*operand):0};
    int rc=gen_matmul_fp16(&a); if(rc) return rc;
    p->count=a.task_count;p->reads[0]=x;p->reads[1]=w;p->nreads=2;p->output=o;
    if(operand) p->reads[p->nreads++]=*operand;
    return p->count&&p->count<=CO_MAX_WORDS?0:ROCKET_E_TILING;
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
int co_emit_lut(co_program *p,co_view x,co_view o,enum co_act kind) {
    if(!p||!same(x,o)||x.tensor.type!=CO_F16||
       (x.tensor.layout!=CO_CUBE&&x.tensor.layout!=CO_LINEAR)||overlap(x,o)||
       (kind!=CO_HARDSIGMOID&&kind!=CO_SIGMOID&&kind!=CO_GELU_GATE&&
        kind!=CO_EXP_BOUNDED&&kind!=CO_RSQRT_BOUNDED)) return ROCKET_E_SHAPE;
    unsigned n=x.tensor.physical_rows*x.tensor.physical_cols;
    if(n/8>=8191) return ROCKET_E_TILING;
    uint16_t lut[1026];
    for(unsigned bank=0;bank<2;bank++) for(unsigned i=0;i<=512;i++) {
        double v=(bank?(double)i:(double)i-512)*32.0/2596.0;
        /* Output converter decodes q/32768 times S: S=64 for bounded exp,
         * S=4 for positive-domain rsqrt; unit activations keep S=1. */
        double y=kind==CO_HARDSIGMOID?fmin(1,fmax(0,v/6+0.5)):
            kind==CO_SIGMOID?1/(1+exp(-v)):
            kind==CO_GELU_GATE?0.5*(1+erf(v/sqrt(2.0))):
            /* Clamp BELOW the declared rsqrt input domain, so interpolation
             * around x=0.25 samples the smooth function on BOTH sides. */
            kind==CO_EXP_BOUNDED?exp(v)/64.0:1/(4*sqrt(fmax(0.125,v)));
        long q=lround(y*32768); if(q<1)q=1; if(q>32767)q=32767;
        lut[bank*513+i]=(uint16_t)q;
    }
    memset(p,0,sizeof(*p));
    lut_act_params_t a={.input_dma=address(x),.output_dma=address(o),.n=(int)n,.lut=lut,
        .bn_mul_operand=0x6912,.bn_alu_cfg=0x80000000,.out_cvt_offset=1,.out_cvt_scale=1,
        .out_cvt_minus_exp=kind==CO_EXP_BOUNDED?9:(kind==CO_RSQRT_BOUNDED?13:15),
        .out_cvt_cvt_type=1,.lut_le_start=0xffffc000,.lut_lo_end=0x4000,
        .le_slope_scale=23107,.le_slope_shift=22,.le_index_select=5,.lo_index_select=5,.tasks=p->words};
    int rc=gen_lut_activation_fp16(&a); if(rc) return rc;
    p->count=(uint32_t)a.task_count;p->reads[0]=x;p->nreads=1;p->output=o;
    return p->count&&p->count<=CO_MAX_WORDS?0:ROCKET_E_TILING;
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
int co_execute(int fd,const co_program *p,unsigned n,enum co_mode mode) {
    if(fd<0||!p||!n||n>CO_MAX_PROGRAMS||(mode!=CO_SEPARATE&&mode!=CO_CHAIN))return ROCKET_E_SHAPE;
    if(strcmp(rocket_hw_current()->name,"rk3588"))return ROCKET_E_UNSUPPORTED;
    if(mode==CO_CHAIN&&!rocket_batched_submit_supported())return ROCKET_E_UNSUPPORTED;
    /* Mixing PPU/DPU completion classes is intentionally not supported. */
    for(unsigned i=0;i<n;i++) {
        if(!p[i].count||p[i].count>CO_MAX_WORDS||p[i].nreads>3||!valid(p[i].output)||
           p[i].output.buffer->fd!=fd)return ROCKET_E_SHAPE;
        if(mode==CO_CHAIN&&p[i].flags!=p[0].flags)return ROCKET_E_UNSUPPORTED;
        for(unsigned j=0;j<p[i].nreads;j++)
            if(!valid(p[i].reads[j])||p[i].reads[j].buffer->fd!=fd)return ROCKET_E_SHAPE;
    }
    if(mode==CO_SEPARATE&&n>1) {
        for(unsigned i=0;i<n;i++){int rc=co_execute(fd,p+i,1,mode);if(rc)return rc;}return 0;
    }
    size_t offs[CO_MAX_PROGRAMS],words=0;
    for(unsigned i=0;i<n;i++){offs[i]=words;words+=rkt_chain_words(p[i].count);}
    rocket_bo rcbo={0};int rc=rocket_bo_alloc32(fd,words*8,&rcbo);if(rc)return rc;
    rocket_task_desc tasks[CO_MAX_PROGRAMS];
    if((rc=rocket_bo_prep(fd,&rcbo,1,0)))goto end;
    memset(rcbo.ptr,0,rcbo.size);
    for(unsigned i=0;i<n;i++) {
        if(mode==CO_CHAIN)rc=rkt_chain_pack_at(&rcbo,tasks,(int)i,offs[i],i+1<n?offs[i+1]:offs[i],
            p[i].words,p[i].count,i+1<n?p[i+1].count:p[i].count);
        else {memcpy(rcbo.ptr,p[i].words,p[i].count*8);tasks[i]=(rocket_task_desc){(uint32_t)rcbo.dma_address,p[i].count};}
        if(rc)break;
    }
    if(!rc&&mode==CO_CHAIN)rc=rkt_chain_verify(&rcbo,tasks,(int)n);
    {int r=rocket_bo_fini(fd,&rcbo);if(!rc)rc=r;}if(rc)goto end;
    uint32_t ins[CO_MAX_PROGRAMS*3+1],outs[CO_MAX_PROGRAMS*3+1];unsigned ni=0,no=0;
    for(unsigned i=0;i<n;i++)add_handle(outs,&no,p[i].output.buffer->bo.handle);
    for(unsigned i=0;i<n;i++)for(unsigned j=0;j<p[i].nreads;j++) {
        uint32_t h=p[i].reads[j].buffer->bo.handle;int written=0;
        for(unsigned k=0;k<no;k++)if(outs[k]==h)written=1;
        /* A device intermediate belongs only in out_handles, never both lists. */
        if(!written)add_handle(ins,&ni,h);
    }
    add_handle(ins,&ni,rcbo.handle);
    rc=rocket_submit_tasks_flags(fd,tasks,n,ins,ni,outs,no,p[0].flags|(mode==CO_CHAIN?ROCKET_JOB_BATCHED:0));
    if(!rc)for(unsigned i=0;i<n;i++) {
        rocket_bo *b=&p[i].output.buffer->bo;
        rc=rocket_bo_prep(fd,b,0,2000000000ull);if(rc)break;
        rc=rocket_bo_fini(fd,b);if(rc)break;
    }
end: rocket_bo_free(fd,&rcbo);return rc;
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
