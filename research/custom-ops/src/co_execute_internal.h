// SPDX-License-Identifier: GPL-3.0-or-later
/* Internal command-plan seam shared by the production executor and the bounded
 * creation-time command-BO cache.  This header is deliberately NOT installed:
 * co_execute_workspace and co_execute_profile keep their existing public layout
 * and callers.  Only rk_plan_runtime.c and host tests include this file.
 *
 * The plan records exactly what the one existing packing path produced:
 * packed command-BO contents (in the caller-owned co_execute_workspace), the
 * task descriptors, the handle lists and the submit flags.  It owns no BO and
 * retains no cross-workspace state; ``owns_commands`` is set only for the
 * no-workspace compatibility path, which allocates and frees a private BO. */
#ifndef CO_EXECUTE_INTERNAL_H
#define CO_EXECUTE_INTERNAL_H

#include "custom_ops.h"

#define CO_COMMAND_MAX_HANDLES (CO_MAX_PROGRAMS * 3u + 1u)

typedef struct {
    /* Set to 1 only by the independent breakdown probe.  Zero keeps the
     * production path at the original two outer timestamps (allocation and the
     * continuous command interval) and avoids adding control-path probes. */
    int detail;
    /* Sub-intervals of one build, for the independent breakdown probe only.
     * total_us is the continuous outer interval and is the value the public
     * command_us attribution uses; the parts may not sum to it. */
    double allocation_us;
    double clear_us;
    double pack_us;
    double verify_us;
    double sync_us;
    double handles_us;
    double total_us;
} co_command_times;

typedef struct {
    int fd;
    const co_program *programs; /* identity of the array this plan was built from */
    rocket_bo commands;      /* borrowed from a workspace, or owned */
    int owns_commands;
    unsigned n_tasks;
    unsigned n_in;
    unsigned n_out;
    uint32_t flags;
    enum co_mode mode;
    size_t words;
    rocket_task_desc tasks[CO_MAX_PROGRAMS];
    uint32_t ins[CO_COMMAND_MAX_HANDLES];
    uint32_t outs[CO_COMMAND_MAX_HANDLES];
    /* Unique output BOs whose completion must be synchronised before the host
     * may read the submitted outputs.  This is the original dedup set, recorded
     * at build time; it is not an instruction to skip the sync. */
    co_buffer *sync_buffers[CO_MAX_PROGRAMS];
    unsigned n_sync;
    /* Per-program outputs, retained for the CO_OUTPUT_SYNC_REFERENCE control
     * build only; the production path uses sync_buffers. */
    co_buffer *program_outputs[CO_MAX_PROGRAMS];
} co_command_plan;

/* Pack one submit through the production packing path.  ``workspace`` supplies
 * the command BO; when it is NULL a private BO is allocated and released by
 * co_command_plan_release().  ``times`` is optional. */
int co_command_plan_build(int fd, const co_program *p, unsigned n,
                          enum co_mode mode, co_execute_workspace *workspace,
                          co_command_plan *plan, co_command_times *times);
void co_command_plan_release(co_command_plan *plan);

/* Submit a plan and perform the existing post-submit output sync.  When
 * ``sync_command`` is set, the command BO gets the historical PROPFINI pair
 * first; cached callers that can prove the device only reads immutable command
 * bytes pass 0, which is the only sync this experiment may omit. */
int co_command_plan_submit(co_command_plan *plan, int sync_command,
                           co_execute_profile *prof);

/* Cached entry used by rk_plan_runtime after the plan was built at workspace
 * creation.  The plan must belong to the same fd and command BO. */
int co_execute_cached_profiled(int fd, const co_program *p, unsigned n,
                               enum co_mode mode, co_execute_workspace *workspace,
                               co_command_plan *plan,
                               co_execute_profile *prof, int sync_command);

/* Cached entry for a workspace-owned command BO produced with workspace=NULL
 * (the multi-submit case).  The plan must remain paired with the exact program
 * array, mode and task count it was built from. */
int co_execute_plan_profiled(const co_program *p, unsigned n, enum co_mode mode,
                             co_command_plan *plan, co_execute_profile *prof,
                             int sync_command);

#endif /* CO_EXECUTE_INTERNAL_H */
