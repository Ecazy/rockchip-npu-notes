// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef RK_PLAN_INTERNAL_H
#define RK_PLAN_INTERNAL_H
#include "rk_plan.h"
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>
#include "custom_ops.h"

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
