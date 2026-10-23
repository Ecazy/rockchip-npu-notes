// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef CO_GDN_H
#define CO_GDN_H
#include "custom_ops.h"
/* Bounded single-head FP16 recurrent execution, dv/dk in 1..256.
 * S[dv,dk], k/q[dk], v[dv]. Six tasks:
 * D=alpha*S; p=D*k; r=v-p; e=beta*r; S'=D+e*k^T; o=S'*q.
 * Scalars use packed diagonal weights, subtraction a negative identity; no
 * implicit broadcast. State stays in feature cubes, in two distinct BOs.
 * Host supplies and packs dynamic k/q/v/scalars each step; no state readback
 * during step. FP16 inter-task rounding. Not a full GDN layer/runtime.
 * On any device error context is poisoned until reset; current state is only
 * committed after all six tasks succeed. Not concurrent/thread safe.
 * Caller buffers must have the stated element counts; destroy before fd close. */
typedef struct co_gdn co_gdn;
int co_gdn_create(int fd,unsigned dv,unsigned dk,enum co_mode,co_gdn **out);
int co_gdn_reset(co_gdn *,const _Float16 *state);
int co_gdn_step(co_gdn *,const _Float16 *k,const _Float16 *q,
                const _Float16 *v,_Float16 alpha,_Float16 beta,_Float16 *out);
int co_gdn_export(co_gdn *,_Float16 *state);
void co_gdn_destroy(co_gdn *);
enum co_gdn_graph { CO_GDN_SIX, CO_GDN_BETA_FOLD, CO_GDN_MERGED };
enum co_gdn_scaling { CO_GDN_DIAGONAL, CO_GDN_EXPANDED, CO_GDN_NATIVE };
typedef struct { int reuse; enum co_gdn_graph graph; enum co_gdn_scaling alpha; } co_gdn_config;
typedef struct {
    double total_us,host_allocation_us,prepare_us;
    double pack_k_us,pack_q_us,pack_v_us,pack_coeff_us,bo_sync_copy_us;
    double emit_us,execute_api_us,guard_us,download_us;
    co_execute_profile executor;
    double task_submit_wait_us[12]; /* populated only by separate diagnostic */
    unsigned tasks,host_allocations;
} co_gdn_profile;
/* Experimental alternatives are opt-in. Default create retains six-task recipe.
 * reuse preallocates coefficient staging and command BO, but provider internals
 * may still allocate. Profile NULL removes per-stage timing (total API unchanged). */
int co_gdn_create_ex(int fd,unsigned dv,unsigned dk,enum co_mode,
                      const co_gdn_config *,co_gdn **);
int co_gdn_step_profiled(co_gdn *,const _Float16 *,const _Float16 *,const _Float16 *,
                         _Float16,_Float16,_Float16 *,co_gdn_profile *);
#endif
