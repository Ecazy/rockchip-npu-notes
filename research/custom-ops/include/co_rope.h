// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef CO_ROPE_H
#define CO_ROPE_H
#include "custom_ops.h"
enum co_rope_style { CO_ROPE_INTERLEAVED, CO_ROPE_SPLIT_HALF };
typedef struct {
    unsigned rows,heads,head_dim,rotary_dim;
    float theta;
    enum co_rope_style style;
} co_rope_config;
typedef struct {
    unsigned cols,tiles,column_classes;
    size_t device_bytes,host_bytes;
    unsigned long long tasks,chain_submits,upload_calls;
    size_t host_activation_reads,host_upload_elements;
} co_rope_plan;
int co_rope_plan_init(const co_rope_config *,co_rope_plan *);
typedef struct co_rope co_rope;
int co_rope_create(int fd,const co_rope_config *,enum co_mode,co_rope **);
/* Input/output [rows,heads*head_dim] in the canonical graph tile collection.
 * CPU reads/gathers paired elements and generates FP16 cos/sin; NPU computes
 * fp16(x*cos), fp16(rotate(x)*sin), then fp16 add. Unrotated dimensions preserved.
 * Absolute position of row r is position_base+r; checked for uint32 overflow.
 * Device errors may leave earlier output tiles completed. No in-place views. */
int co_rope_run(co_rope *,const co_view *inputs,unsigned ninputs,
                const co_view *outputs,unsigned noutputs,uint32_t position_base);
void co_rope_destroy(co_rope *);
#endif
