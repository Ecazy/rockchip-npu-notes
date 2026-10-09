// SPDX-License-Identifier: GPL-3.0-or-later
/* Host-only structural tests for the multi-K HardSigmoid operator's K tiling.
 * No NPU and no BO payload is touched: descriptors, canonical slice offsets and
 * index equivalence, tile coverage, and emitter/alias rejection are checked.
 * Runtime accumulation itself is exercised on hardware, not here. */
#include "custom_ops.h"
#include <stdio.h>
#include <string.h>

static unsigned checks, failures;
#define EXPECT(expr) do { checks++; if(!(expr)) { \
    fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#expr); failures++; } } while(0)

static co_buffer fake(size_t payload, uint32_t handle, uint32_t dma) {
    co_buffer b; memset(&b,0,sizeof(b));
    b.fd=3; b.payload=payload;
    b.bo.handle=handle; b.bo.dma_address=dma;
    b.bo.size=payload+CO_SLACK+CO_GUARD;
    return b;
}

/* Canonical K-slice layout: for every tile, the slice descriptor and its byte
 * offset reproduce the full cube's element addresses exactly, tiles are
 * contiguous, and their logical lengths cover K with no gap or overlap. */
static int slice_case(unsigned m, unsigned k) {
    co_tensor full;
    EXPECT(co_tensor_input(&full,m,k)==0);
    if (full.layout!=CO_CUBE || full.type!=CO_F16) return 1;
    const size_t base=16; /* exercise a 16 B aligned external offset view */
    co_buffer bf=fake(full.bytes+base+64,1,0x100000);
    co_view vf;
    EXPECT(co_view_init(&vf,&bf,base,&full)==0);
    unsigned nt=co_k_tiles(k), total=0;
    size_t expect_off=0;
    EXPECT(nt>=1 && nt<=CO_K_TILES);
    for (unsigned i=0; i<nt; i++) {
        unsigned k0=i*CO_K_TILE, kt=co_k_tile_length(k,i);
        EXPECT(kt>=1 && kt<=CO_K_TILE);
        EXPECT(k0%CO_K_TILE==0);
        total+=kt;
        co_tensor slice; size_t off=0;
        EXPECT(co_tensor_input_slice(&slice,&full,k0,kt,&off)==0);
        EXPECT(off==expect_off);
        EXPECT(off==(size_t)k0*full.physical_rows*2);
        co_view vs;
        EXPECT(co_view_init(&vs,&bf,base+off,&slice)==0);
        EXPECT(vs.offset==base+off);
        for (unsigned r=0; r<m; r++)
            for (unsigned c=0; c<kt; c++)
                EXPECT(co_index(&full,r,k0+c)==off/2+co_index(&slice,r,c));
        expect_off=off+slice.bytes;
    }
    EXPECT(total==k);
    EXPECT(expect_off==full.bytes);   /* no gap and no over-read past the cube */
    printf("SLICE m=%u k=%u tiles=%u physical_cols=%u bytes=%zu PASS\n",
           m,k,nt,full.physical_cols,full.bytes);
    return 0;
}

int main(void) {
    /* Raised dimension bounds and full-input cube. */
    co_tensor t;
    EXPECT(co_tensor_input(&t,13,CO_MAX_K)==0);
    EXPECT(t.physical_rows==16 && t.physical_cols==4096);
    EXPECT(co_tensor_input(&t,12,4096+1)!=0);      /* K>4096 rejected */
    EXPECT(co_tensor_input(&t,257,256)!=0);        /* M>256 rejected */
    EXPECT(co_tensor_input(&t,0,256)!=0);
    EXPECT(co_tensor_matrix(&t,12,257,CO_F16)!=0); /* primitive input stays <=256 */
    EXPECT(co_tensor_weight(&t,64,257)!=0);        /* weight tile stays <=256 */

    /* Canonical ragged K splits at the acceptance dimensions. */
    const unsigned ks[]={257,513,769,1024,2049,4096};
    const unsigned ms[]={13,1,29,13,29,1};
    for (unsigned i=0; i<sizeof(ks)/sizeof(ks[0]); i++) slice_case(ms[i],ks[i]);
    slice_case(16,255);
    slice_case(16,256);
    slice_case(16,512);

    /* Slice descriptor rejects a tile wider than the primitive width and a
     * misaligned start channel. */
    co_tensor full,s;
    EXPECT(co_tensor_input(&full,16,512)==0);
    size_t off=0;
    EXPECT(co_tensor_input_slice(&s,&full,0,257,&off)!=0);
    EXPECT(co_tensor_input_slice(&s,&full,1,256,&off)!=0);
    EXPECT(co_tensor_input_slice(&s,&full,0,512,&off)!=0);
    EXPECT(co_tensor_input_slice(&s,&full,0,0,&off)!=0);
    EXPECT(co_tensor_input_slice(&s,&full,512,1,&off)!=0); /* beyond cols */
    EXPECT(co_tensor_input_slice(&s,&full,256,256,&off)==0);
    EXPECT(co_tensor_input_slice(&s,&full,0,256,&off)==0);

    /* Primitive emitter refuses a full-width input cube (Kt>256). */
    co_tensor xi,wt,ot,pi;
    EXPECT(co_tensor_input(&xi,16,512)==0);
    EXPECT(co_tensor_weight(&wt,32,256)==0);
    EXPECT(co_tensor_matrix(&ot,16,32,CO_F16)==0);
    co_buffer bx=fake(xi.bytes+32,1,0x100000);
    co_buffer bw=fake(wt.bytes,2,0x200000);
    co_buffer bo=fake(ot.bytes,3,0x300000);
    co_view vx,vw,vo;
    EXPECT(co_view_init(&vx,&bx,0,&xi)==0);
    EXPECT(co_view_init(&vw,&bw,0,&wt)==0);
    EXPECT(co_view_init(&vo,&bo,0,&ot)==0);
    co_program p;
    EXPECT(co_emit_mm(&p,vx,vw,vo)!=0); /* Kt=512 must be rejected */

    /* Accumulation entry: valid ping-pong pair accepted, aliases rejected. */
    co_tensor wi;
    EXPECT(co_tensor_input(&xi,16,256)==0);
    EXPECT(co_tensor_weight(&wi,32,256)==0);
    EXPECT(co_tensor_matrix(&ot,16,32,CO_F16)==0);
    EXPECT(co_tensor_matrix(&pi,16,32,CO_F16)==0);
    co_buffer bx2=fake(xi.bytes+32,10,0x100000);
    co_buffer bwt=fake(wi.bytes,11,0x200000);
    co_buffer bp=fake(pi.bytes,12,0x300000);
    co_buffer bout=fake(ot.bytes,13,0x400000);
    co_view vx2,vwt,vp,vout;
    EXPECT(co_view_init(&vx2,&bx2,0,&xi)==0);
    EXPECT(co_view_init(&vwt,&bwt,0,&wi)==0);
    EXPECT(co_view_init(&vp,&bp,0,&pi)==0);
    EXPECT(co_view_init(&vout,&bout,0,&ot)==0);
    EXPECT(co_emit_mm_acc(&p,vx2,vwt,vp,vout)==0);
    EXPECT(p.count>0 && p.nreads==3 && p.reads[2].buffer==&bp && p.output.buffer==&bout);

    /* prior == out (same BO/offset) */
    EXPECT(co_emit_mm_acc(&p,vx2,vwt,vout,vout)!=0);
    /* same BO, non-overlapping offsets is still not the required ping-pong pair */
    co_buffer bshared=fake(2*pi.bytes+64,14,0x500000);
    co_view vp2,vout2;
    EXPECT(co_view_init(&vp2,&bshared,0,&pi)==0);
    EXPECT(co_view_init(&vout2,&bshared,pi.bytes,&ot)==0);
    EXPECT(co_emit_mm_acc(&p,vx2,vwt,vp2,vout2)!=0);
    /* x aliases out */
    co_buffer bxo=fake(xi.bytes+ot.bytes+64,15,0x600000);
    co_view vx3,vout3;
    EXPECT(co_view_init(&vx3,&bxo,0,&xi)==0);
    EXPECT(co_view_init(&vout3,&bxo,0,&ot)==0);
    EXPECT(co_emit_mm_acc(&p,vx3,vwt,vp,vout3)!=0);
    /* Kt>256 input rejected on the accumulate entry too */
    EXPECT(co_emit_mm_acc(&p,vx,vwt,vp,vout)!=0);

    printf("K-TILING host checks=%u failures=%u %s\n",checks,failures,failures?"FAIL":"PASS");
    return failures?1:0;
}
