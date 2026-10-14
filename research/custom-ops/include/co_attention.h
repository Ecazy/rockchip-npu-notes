// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef CO_ATTENTION_H
#define CO_ATTENTION_H
#include "co_rope.h"
typedef struct {
    unsigned max_query,capacity,heads,kv_heads,head_dim,rotary_dim;
    float theta,scale; /* scale=0 chooses 1/sqrt(head_dim) */
    enum co_rope_style style;
    int causal;
    unsigned window; /* 0=full; causal only, includes current token */
    size_t device_budget; /* 0=default 256 MiB */
} co_attention_config;
typedef struct {
    size_t device_bytes,host_bytes;
    unsigned long long tasks_per_step,chain_submits_per_step,upload_calls,download_calls;
    size_t host_activation_reads,score_writes,statistic_downloads,activation_downloads;
    size_t host_upload_elements,dynamic_weight_elements;
} co_attention_plan;
int co_attention_plan_init(const co_attention_config *,co_attention_plan *);
typedef struct co_attention co_attention;
int co_attention_create(int fd,const co_attention_config *,enum co_mode,co_attention **);
/* Reset cache and absolute starting position; capacity cannot change. Reserves
 * position range for padded queries too, rejecting uint32 position overflow. */
int co_attention_reset(co_attention *,uint32_t position_base);
/* Discard a committed suffix (block-level rollback). Weight/scratch BOs are
 * rebuilt on the next step. Clone copies only committed KV and its position
 * base, validating the destination capacity and RoPE/KV encoding. */
int co_attention_truncate(co_attention *,unsigned length);
int co_attention_cache_clone(co_attention *destination,const co_attention *source);
unsigned co_attention_cache_length(const co_attention *);
/* Copies committed rotated K and raw V [length,kv_heads,head_dim]. Outputs must
 * not overlap; capacity_tokens>=length. No BO readback: cache is host-owned. */
int co_attention_cache_export(const co_attention *,_Float16 *keys,_Float16 *values,unsigned capacity_tokens);
/* Append tokens and compute self-attention for those tokens. Q[token,Hq,D],
 * K/V[token,Hkv,D]. Optional additive mask [tokens,old_length+tokens], shared by
 * heads; finite or -inf. NaN/+inf rejected before mutation. All masked -> zero.
 * Cache length/data and output commit only after all heads succeed. On failure
 * scratch/weights may change, but a retry fully rebuilds them from committed KV.
 * Fixed capacity/max_query padded computation, sequential heads: a functional
 * hybrid baseline, not an optimized decode or FlashAttention implementation.
 * QK/PV run on NPU; Softmax and dynamic K/V packing involve CPU processing.
 * No CPU MatMul fallback. Caller owns fd; synchronous and not thread-safe. */
int co_attention_step(co_attention *,unsigned tokens,const _Float16 *q,const _Float16 *k,
                      const _Float16 *v,const _Float16 *optional_mask,_Float16 *out);
void co_attention_destroy(co_attention *);
#endif
