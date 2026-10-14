// SPDX-License-Identifier: GPL-3.0-or-later
#include "custom_ops.h"
#include <string.h>

static unsigned extent(unsigned total, unsigned i) {
    unsigned left=total-i*256u;
    return left>256u?256u:left;
}

int co_linear_plan_init_ex(co_linear_plan *out, unsigned m, unsigned k, unsigned n,
                           int has_bias, enum co_linear_post post, size_t budget) {
    if (!out) return ROCKET_E_SHAPE;
    memset(out,0,sizeof(*out));
    if ((has_bias!=0 && has_bias!=1) || (post!=CO_LINEAR_NONE && post!=CO_LINEAR_HSIG) ||
        !m || m>CO_MAX_M || !k || k>CO_MAX_K || !n || n>CO_MAX_N)
        return ROCKET_E_SHAPE;
    co_linear_plan p={.m=m,.k=k,.n=n,.mtiles=(m+255)/256,
        .ktiles=co_k_tiles(k),.ntiles=(n+255)/256,.has_bias=(unsigned)has_bias,.post=(unsigned)post};
    p.output_tiles=p.mtiles*p.ntiles;
    p.row_classes=(p.mtiles>1 && m%256)?2:1;
    p.chain_submits=p.output_tiles;
    for (unsigned mi=0;mi<p.mtiles;mi++) {
        for (unsigned ni=0;ni<p.ntiles;ni++) {
            co_tensor x,y,id,w;
            unsigned rows=extent(m,mi),cols=extent(n,ni);
            if (co_tensor_input(&x,rows,k) || co_tensor_matrix(&y,rows,cols,CO_F16) ||
                co_tensor_weight(&id,y.physical_cols,y.physical_cols)) return ROCKET_E_SHAPE;
            unsigned programs=p.ktiles+has_bias+(post==CO_LINEAR_HSIG?(y.bytes/2+32767)/32768:0);
            if (programs>CO_MAX_PROGRAMS) return ROCKET_E_TILING;
            p.tasks+=programs;
            /* Identical full M blocks reuse the same N contexts and weights. */
            if (mi && rows==256) continue;
            size_t bytes=x.bytes+5*y.bytes+id.bytes;
            for (unsigned ki=0;ki<p.ktiles;ki++) {
                if (co_tensor_weight(&w,cols,co_k_tile_length(k,ki))) return ROCKET_E_SHAPE;
                bytes+=w.bytes;
            }
            bytes+=(size_t)(7+p.ktiles)*(CO_SLACK+CO_GUARD);
            if (bytes>SIZE_MAX-p.device_bytes) return ROCKET_E_TILING;
            p.device_bytes+=bytes;
        }
    }
    if (p.device_bytes>(budget?budget:CO_LINEAR_DEVICE_BUDGET)) return ROCKET_E_TILING;
    *out=p;
    return 0;
}

int co_linear_plan_tile(const co_linear_plan *p, unsigned index, co_linear_tile_info *out) {
    if (!p || !out) return ROCKET_E_SHAPE;
    co_linear_plan canonical;
    if (co_linear_plan_init_ex(&canonical,p->m,p->k,p->n,p->has_bias,(enum co_linear_post)p->post,SIZE_MAX) ||
        p->mtiles!=canonical.mtiles || p->ntiles!=canonical.ntiles ||
        p->ktiles!=canonical.ktiles || p->output_tiles!=canonical.output_tiles ||
        index>=canonical.output_tiles) return ROCKET_E_SHAPE;
    unsigned mi=index/p->ntiles,ni=index%p->ntiles;
    co_tensor y;
    unsigned rows=extent(p->m,mi),cols=extent(p->n,ni);
    if (co_tensor_matrix(&y,rows,cols,CO_F16)) return ROCKET_E_SHAPE;
    *out=(co_linear_tile_info){mi*256,ni*256,rows,cols,p->ktiles,
        p->ktiles+p->has_bias+(p->post==CO_LINEAR_HSIG?(unsigned)(y.bytes/2+32767)/32768:0)};
    return 0;
}

int co_linear_plan_init(co_linear_plan *out, unsigned m, unsigned k, unsigned n, size_t budget) {
    return co_linear_plan_init_ex(out,m,k,n,1,CO_LINEAR_HSIG,budget);
}
