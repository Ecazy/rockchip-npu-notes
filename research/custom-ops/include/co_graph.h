// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef CO_GRAPH_H
#define CO_GRAPH_H
#include "custom_ops.h"
#include "co_attention.h"
#define CO_GRAPH_MAX_NODES 64u
#define CO_GRAPH_NO_NODE UINT32_MAX
/* Static FP16 rank-2 DAG. Linear weights are row-major [out,in], with optional
 * bias. Non-input shapes are inferred; Linear.cols specifies output channels.
 * Forward references are legal; reachable cycles/unsupported ops are rejected.
 * Unreachable nodes are eliminated. No implicit broadcast or CPU compute fallback. */
enum co_graph_op { CO_GRAPH_INPUT, CO_GRAPH_LINEAR, CO_GRAPH_ADD, CO_GRAPH_MUL,
                   CO_GRAPH_SILU, CO_GRAPH_GELU, CO_GRAPH_HSIG,
                   CO_GRAPH_RMSNORM, CO_GRAPH_SOFTMAX, CO_GRAPH_ROPE, CO_GRAPH_ATTENTION };
#define CO_GRAPH_ALLOW_NEG_INF 1u /* Input only; Softmax scores or Attention mask only */
typedef struct {
    enum co_graph_op op;
    unsigned a,b,rows,cols;
    unsigned c,mask_node,has_mask; /* Attention: a=Q,b=K,c=V, optional mask */
    const _Float16 *weights,*bias;
    const _Float16 *gamma; /* RMSNorm gamma[cols], copied at compile */
    float epsilon; /* RMSNorm: finite and positive */
    unsigned flags;
    unsigned heads,kv_heads,head_dim,rotary_dim;
    float theta,scale;
    enum co_rope_style rope_style;
    uint32_t position_base;
    int causal;
    unsigned window;
} co_graph_node;
typedef struct {
    const co_graph_node *nodes;
    unsigned count,output;
    size_t device_budget; /* 0 = default 256 MiB; BO bytes, not driver page rounding */
} co_graph_desc;
typedef struct {
    unsigned rows,cols,tiles,slot,first,last;
    size_t bytes;
} co_graph_tensor_plan;
typedef struct {
    unsigned steps,inputs,slots,order[CO_GRAPH_MAX_NODES];
    unsigned noutputs,outputs[CO_GRAPH_MAX_NODES];
    co_graph_tensor_plan tensors[CO_GRAPH_MAX_NODES];
    size_t slot_bytes[CO_GRAPH_MAX_NODES];
    size_t tensor_bytes,linear_bytes,rowop_bytes,rope_bytes,attention_bytes,workspace_bytes,device_bytes;
    unsigned hybrid_nodes;
    unsigned long long host_upload_calls,statistic_download_calls,activation_download_calls;
    size_t host_activation_reads,statistic_downloads,activation_downloads,host_uploads;
    unsigned identity_mask;
    size_t gate_bytes;
    unsigned long long tasks,chain_submits;
} co_graph_plan;
/* Pure analysis: topology, shape/layout, 256-bound tiling, liveness/BO reuse,
 * and exact requested resident BO capacity. No model data reads or device calls. */
int co_graph_plan_init(const co_graph_desc *,co_graph_plan *);
/* Compile/analyze the union of multiple roots; shared nodes execute once and
 * every exported tensor remains live through the final download. */
int co_graph_plan_many(const co_graph_desc *,const unsigned *outputs,unsigned count,co_graph_plan *);
typedef struct co_graph co_graph;
int co_graph_compile(int fd,const co_graph_desc *,enum co_mode,co_graph **);
int co_graph_compile_many(int fd,const co_graph_desc *,const unsigned *outputs,unsigned count,
                          enum co_mode,co_graph **);
void co_graph_destroy(co_graph *);
typedef struct { unsigned node;const _Float16 *data; } co_graph_input;
/* Bind each reachable input exactly once, by node ID. Inputs are validated before
 * execution. Linear/elementwise intermediate tiles stay on device. Hybrid
 * RMSNorm downloads per-tile row statistics; Softmax additionally reads logits
 * on the host. RoPE reads/gathers paired elements. Attention stages Q/K/V (and
 * optional mask) on the host and uploads its output tiles; its products run on
 * NPU. All these mixed transfers are reported by the plan. RMSNorm/Softmax
 * materialize/upload coefficients as reported by the plan. Output is published after full success. Synchronous, not thread-safe.
 * Caller may release graph descriptors and weights after compile, but owns fd. */
int co_graph_run(co_graph *,const co_graph_input *,unsigned count,_Float16 *output);
typedef struct { unsigned node;_Float16 *data; } co_graph_output;
/* Bind every compiled output once. All results are staged and finite-checked
 * before publishing any; output arrays must not overlap each other. */
int co_graph_run_many(co_graph *,const co_graph_input *,unsigned ninputs,
                      const co_graph_output *,unsigned noutputs);
#endif
