// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef RK_PLAN_INTERNAL_H
#define RK_PLAN_INTERNAL_H
#include "rk_plan.h"
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>
#include "custom_ops.h"
#include "co_execute_internal.h"

typedef struct rk_plan {
    uint32_t schema_version;
    uint32_t flags;
    uint64_t profile_hash;
    uint64_t source_hash;
    uint32_t nvalues;
    rk_value_desc *values;
    char **value_names;
    uint32_t nbuffers;
    rk_buffer_desc *buffers;
    uint32_t nviews;
    rk_view_desc *views;
    uint32_t nactions;
    rk_action_desc *actions;
    uint32_t **action_inputs;
    uint32_t **action_outputs;
    uint32_t **action_deps;
    uint32_t nsubmits;
    rk_submit_desc *submits;
    uint32_t **submit_tasks;
    uint32_t **submit_deps;
    /* action id -> owning submit id, UINT32_MAX for host actions */
    uint32_t *task_submit;
    uint32_t npublish;
    rk_publish_desc *publish;
    uint64_t device_budget_bytes;
    uint64_t host_budget_bytes;
    uint64_t claimed_device_bytes;
    uint64_t claimed_host_bytes;
    uint32_t runtime_abi_version;
} rk_plan;

/* Total Host bytes this workspace keeps resident for ``capacity`` repeats:
 * staging arena + download tile + emission array + output bookkeeping.  The
 * group entry uses it to bound growth, and the host tests use it to derive
 * boundaries instead of hard-coding them. */
uint64_t rk_workspace_host_bytes(const rk_workspace *ws, uint32_t capacity);

void rk_diag_clear(rk_diag *diag);
void rk_diag_set(rk_diag *diag, int code, const char *stage,
                 const char *message, const char *source,
                 uint32_t action_id, int native_status);
int rk_plan_validate_desc(const rk_plan_desc *desc, rk_diag *diag);
int rk_plan_clone(const rk_plan_desc *desc, rk_plan **out, rk_diag *diag);
int rk_plan_clone_plan(const rk_plan *src, rk_plan **out, rk_diag *diag);
void rk_plan_free(rk_plan *plan);
const rk_value_desc *rk_plan_value(const rk_plan *plan, uint32_t id);
const rk_buffer_desc *rk_plan_buffer(const rk_plan *plan, uint32_t id);
const rk_view_desc *rk_plan_view(const rk_plan *plan, uint32_t id);
const rk_action_desc *rk_plan_action(const rk_plan *plan, uint32_t id);

/* Emit one validated NPU action into the existing custom-ops program seam. */
int rk_emit_action(const rk_plan *plan, const co_view *views,
                   uint32_t action_id, co_program *out, rk_diag *diag);

/* One cached submit: the creation-time programs plus the immutable command
 * plan.  The plan owns its command BO (built with workspace=NULL), so submits
 * never share or overwrite a command region. */
typedef struct {
    uint32_t ready;
    uint32_t submit_id;
    uint32_t n_tasks;
    enum co_mode mode;
    co_program *programs;   /* slice of ws->multi_submit_programs */
    co_command_plan plan;
} rk_submit_cache;

typedef struct rk_workspace {
    rk_plan *plan;
    int fd;
    co_buffer *slots;
    uint32_t nslots;
    co_view *views;
    co_execute_workspace execute_workspace;
    int execute_workspace_ready;
    atomic_int busy;
    atomic_int poisoned;
    int created;
    uint32_t max_submit_tasks;
    /* Snapshot at workspace creation; changing the environment afterwards does
     * not change this workspace's execution policy. */
    uint32_t upload_coalesce;
    uint32_t upload_fast_pack;
    /* Experimental prepared-binding validation.  Off by default.  When
     * enabled, the workspace caches the exact binding signatures after one
     * full validation; later calls with identical signatures scan input
     * finiteness but skip the repeated static shape/alias checks. */
    uint32_t prepared_validation;
    uint32_t prepared_validation_ready;
    rk_binding *prepared_inputs;
    rk_binding *prepared_outputs;
    uint32_t prepared_ninputs;
    uint32_t prepared_noutputs;
    /* Experimental creation-time co_program cache.  This candidate is limited
     * to plans whose complete NPU task set is one submit, so the existing
     * batch_scratch can hold the cached programs with no extra co_program
     * allocation.  Default on; opt-out with RK_PLAN_PROGRAM_CACHE=0.
     * Workspace-local only. */
    uint32_t program_cache;
    uint32_t program_cache_ready;
    uint32_t program_cache_submit;
    uint32_t program_cache_tasks;
    /* Bounded command-BO packing reuse.  Eligible only when the creation-time
     * co_program cache is ready for the same single submit, so the packed
     * commands, task descriptors, BO addresses and submit configuration are
     * all fixed by construction.  The plan borrows the existing
     * execute_workspace command BO; disabling this flag keeps the original
     * per-submit packing path.  Default on for eligible workspaces; opt out with
     * RK_PLAN_COMMAND_CACHE=0.  The separate RK_PLAN_COMMAND_CACHE_SYNC=1 arm
     * re-syncs the unchanged command BO every submit and exists for A/B. */
    uint32_t command_cache;
    uint32_t command_cache_sync;     /* re-sync unchanged command BO per submit */
    uint32_t command_cache_ready;
    co_command_plan *command_plan;   /* allocated only for an eligible workspace */
    /* Budgeted multi-submit static preparation.  Each ready submit owns its
     * creation-time programs and an independent command BO; default on, with
     * RK_PLAN_MULTI_SUBMIT_CACHE=0 as the control and
     * RK_PLAN_MULTI_SUBMIT_BUDGET bounding host+device cache bytes. */
    uint32_t multi_submit_cache;
    uint32_t multi_submit_sync;      /* reuse command-cache sync policy */
    uint32_t multi_submit_ready;
    uint32_t multi_submit_cached;
    uint64_t multi_submit_budget;
    uint32_t multi_submit_program_capacity;
    co_program *multi_submit_programs;
    rk_submit_cache *submit_cache;
    uint64_t multi_submit_device_bytes;
    /* Pre-allocated per-run Host scratch.  The single-run gate gives exclusive
     * ownership of these buffers to whichever entry point runs, so a plan call
     * no longer allocates or frees the emission array, the per-output tile
     * counters or the output staging block. */
    uint32_t ninputs;
    uint32_t noutputs;
    size_t staging_stride;      /* bytes per repeat: sum of output value nbytes */
    uint32_t staging_capacity;  /* repeats the staging arena can hold */
    void *staging_arena;
    size_t *staging_offsets;    /* [noutputs] offsets inside one repeat block */
    uint32_t *tile_count_scratch;                 /* [noutputs] */
    co_program *batch_scratch;                    /* [max_submit_tasks] */
    /* One download scratch for the largest downloaded tile.  The tile is copied
     * into the staging block before the next download runs, so a single buffer
     * is enough and the per-download malloc/free disappears. */
    void *tile_scratch;
    size_t tile_scratch_bytes;
    /* action id -> index inside its submit's task list (UINT32_MAX if none).
     * The emission array is indexed by this, so tasks do not have to be a
     * contiguous action range: a static-repeat plan interleaves each task with
     * its own uploads, which is what makes every read's nearest producer its
     * own upload. */
    uint32_t *emit_index;
} rk_workspace;

#endif
