// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef CO_GDN_CPU_H
#define CO_GDN_CPU_H
#include "custom_ops.h"
enum co_gdn_cpu_precision { CO_GDN_CPU_F16, CO_GDN_CPU_F32 };
typedef struct co_gdn_cpu co_gdn_cpu;
/* CPU-resident row-major state. FP16 rounds D, p, r, e, S_next, o separately;
 * rank-1 product+add uses FP32 FMA before S_next rounding, matching fused task 5.
 * FP32 uses float state/intermediates/FMA and never stores FP16 internally.
 * Input vectors/gates are FP16 for both arms. Output/export type follows mode.
 * No per-step allocation, no device calls. AArch64 uses FP32 NEON FMA.
 * Reduction tree differs from NPU; bit-identical arithmetic is not promised. */
int co_gdn_cpu_create(unsigned dv,unsigned dk,enum co_gdn_cpu_precision,co_gdn_cpu **);
int co_gdn_cpu_reset(co_gdn_cpu *,const _Float16 *initial);
int co_gdn_cpu_step(co_gdn_cpu *,const _Float16 *k,const _Float16 *q,const _Float16 *v,
                    _Float16 alpha,_Float16 beta,void *out);
int co_gdn_cpu_export(const co_gdn_cpu *,void *state);
void co_gdn_cpu_destroy(co_gdn_cpu *);
const char *co_gdn_cpu_backend(void);
/* Standalone resident scaling reference; output does not alias input. */
void co_gdn_cpu_scale_f16(const _Float16 *,_Float16 *,size_t n,_Float16);
#endif
