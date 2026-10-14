// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef CO_DECODER_H
#define CO_DECODER_H
#include "co_graph.h"
typedef struct {
    unsigned model_dim,hidden_dim;
    float epsilon;
    co_attention_config attention; /* causal self-attention; max_query=chunk bound */
    size_t device_budget;
    int disable_one_row_decode; /* 0=compile a dedicated single-token geometry */
} co_decoder_config;
typedef struct {
    const _Float16 *norm1,*norm2;
    const _Float16 *wq,*wk,*wv,*wo,*wgate,*wup,*wdown;
} co_decoder_weights;
typedef struct {
    unsigned geometries,rows[2];size_t device_bytes;
    co_graph_plan pre[2],post[2];co_attention_plan attention[2];
    unsigned long long tasks[2],chain_submits[2],upload_calls[2],download_calls[2];
    size_t score_elements[2];
} co_decoder_plan;
/* Bias-free, pre-RMSNorm, RoPE/GQA + SwiGLU decoder block. Seven row-major FP16
 * matrices: Wq[Hq*D,model], Wk/Wv[Hkv*D,model], Wo[model,Hq*D],
 * Wgate/Wup[hidden,model], Wdown[model,hidden]; norm1/2[model]. */
int co_decoder_plan_init(const co_decoder_config *,co_decoder_plan *);
typedef struct co_decoder co_decoder;
int co_decoder_create(int fd,const co_decoder_config *,const co_decoder_weights *,enum co_mode,co_decoder **);
void co_decoder_destroy(co_decoder *);
int co_decoder_reset(co_decoder *,uint32_t position_base);
unsigned co_decoder_cache_length(const co_decoder *);
unsigned co_decoder_last_query_rows(const co_decoder *);
unsigned co_decoder_cache_switches(const co_decoder *); /* successful geometry switches since reset */
int co_decoder_cache_export(const co_decoder *,_Float16 *rotated_k,_Float16 *v,unsigned capacity_tokens);
/* X/Y[tokens,model_dim]. Optional mask[tokens,old_length+tokens]. Both residuals
 * and FFN run on NPU through precompiled graphs; attention is explicitly hybrid.
 * Cache and output commit after the ENTIRE block succeeds, including post-FFN.
 * Tokens=1 selects M=1 geometry; KV is copied only when switching geometries.
 * Matrices/gammas are copied at create, fd stays caller-owned. Synchronous and
 * not thread-safe. No CPU MatMul fallback or token sampling/model loader. */
int co_decoder_step(co_decoder *,unsigned tokens,const _Float16 *x,const _Float16 *mask,_Float16 *y);
#endif
