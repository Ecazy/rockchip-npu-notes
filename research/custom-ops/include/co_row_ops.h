// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef CO_ROW_OPS_H
#define CO_ROW_OPS_H
#include "custom_ops.h"
enum co_row_kind { CO_ROW_RMSNORM, CO_ROW_SOFTMAX };
typedef struct {
    unsigned rows,cols,tiles,column_classes;
    size_t device_bytes;
    unsigned long long tasks,chain_submits;
    size_t host_activation_reads,statistic_downloads,host_uploads;
} co_row_plan;
int co_row_plan_init(co_row_plan *,enum co_row_kind,unsigned rows,unsigned cols);
typedef struct co_rowop co_rowop;
/* Explicit hybrid operators. RMSNorm: device FP16 square/FP32 per-tile sums,
 * host global row sums/rsqrt and gamma*inverse coefficients, device scaling.
 * Softmax: host scans logits for a global row max and exp bands, device exp,
 * coefficient multiplication/FP32 sums, host denominator, device normalization.
 * Softmax accepts -inf mask entries; all-masked rows produce zero. NaN/+inf error.
 * Exp bands avoid the LUT's nonzero floor becoming mass on long negative tails.
 * RMSNorm rejects any per-tile row square sum >=65504, conservatively catching
 * hardware saturation (including some finite high-energy rows). Gamma is copied at create; epsilon finite and positive. No fallback on errors. */
int co_rowop_create(int fd,enum co_row_kind,unsigned rows,unsigned cols,
                    const _Float16 *gamma,float epsilon,enum co_mode,co_rowop **);
int co_rowop_run(co_rowop *,const co_view *inputs,unsigned ninputs,
                 const co_view *outputs,unsigned noutputs);
void co_rowop_destroy(co_rowop *);
#endif
