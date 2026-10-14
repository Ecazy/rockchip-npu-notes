// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef CO_TILED_INTERNAL_H
#define CO_TILED_INTERNAL_H
#include "custom_ops.h"
#include <stdlib.h>
static inline unsigned co_blocks(unsigned n){return (n+255)/256;}
static inline unsigned co_extent(unsigned n,unsigned i){unsigned t=n-i*256;return t>256?256:t;}
static inline int co_tiles_size(unsigned m,unsigned n,size_t *bytes) {
    if(!m||m>CO_MAX_M||!n||n>CO_MAX_N||!bytes)return ROCKET_E_SHAPE;
    *bytes=0;
    for(unsigned mi=0;mi<co_blocks(m);mi++)for(unsigned ni=0;ni<co_blocks(n);ni++) {
        co_tensor t;int rc=co_tensor_matrix(&t,co_extent(m,mi),co_extent(n,ni),CO_F16);if(rc)return rc;
        *bytes+=t.bytes;
    }
    return 0;
}
static inline int co_tiles_alloc(int fd,unsigned m,unsigned n,co_buffer *b,co_view **views) {
    size_t bytes;int rc=co_tiles_size(m,n,&bytes);if(rc)return rc;
    *views=calloc(co_blocks(m)*co_blocks(n),sizeof(co_view));if(!*views)return ROCKET_E_NOMEM;
    rc=co_buffer_alloc(fd,bytes,b);if(rc)return rc;
    size_t off=0;unsigned idx=0;
    for(unsigned mi=0;mi<co_blocks(m);mi++)for(unsigned ni=0;ni<co_blocks(n);ni++) {
        co_tensor t;rc=co_tensor_matrix(&t,co_extent(m,mi),co_extent(n,ni),CO_F16);if(rc)return rc;
        rc=co_view_init(&(*views)[idx++],b,off,&t);if(rc)return rc;
        off+=t.bytes;
    }
    return 0;
}
/* Caller supplies one <=256x256 scratch row-major tile. */
static inline int co_tiles_read(const co_view *v,unsigned m,unsigned n,_Float16 *out,_Float16 *scratch) {
    unsigned idx=0;
    for(unsigned mi=0;mi<co_blocks(m);mi++)for(unsigned ni=0;ni<co_blocks(n);ni++) {
        co_view t=v[idx++];int rc=co_download(&t,scratch);if(rc)return rc;
        for(unsigned r=0;r<t.tensor.rows;r++)for(unsigned q=0;q<t.tensor.cols;q++)
            out[((size_t)mi*256+r)*n+ni*256+q]=scratch[(size_t)r*t.tensor.cols+q];
    }
    return 0;
}
static inline int co_tiles_write(co_view *v,unsigned m,unsigned n,const _Float16 *in,_Float16 *scratch) {
    unsigned idx=0;
    for(unsigned mi=0;mi<co_blocks(m);mi++)for(unsigned ni=0;ni<co_blocks(n);ni++) {
        co_view *t=&v[idx++];
        for(unsigned r=0;r<t->tensor.rows;r++)for(unsigned q=0;q<t->tensor.cols;q++)
            scratch[(size_t)r*t->tensor.cols+q]=in[((size_t)mi*256+r)*n+ni*256+q];
        int rc=co_upload_f16(t,scratch);if(rc)return rc;
    }
    return 0;
}
#endif
