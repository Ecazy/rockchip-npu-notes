// SPDX-License-Identifier: GPL-3.0-or-later
/* Host-only proof for the bounded command-BO packing reuse.
 *
 * This file exercises the real packing path (BO clear, regcmd copy, chain
 * sealing/verification, task descriptors and handle lists) with the mock
 * provider.  The submit ioctl itself is stubbed so the test can try success and
 * failure without a device; everything below the submit is production code.
 */
#include "co_execute_internal.h"
#include "rocket_chain.h"
#include "mock_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks, failures;
#define EXPECT(e) do { checks++; if (!(e)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #e); failures++; } } while (0)

/* Capture one submit without entering the provider. */
static unsigned submit_calls, submit_fail_at;
static unsigned last_n_tasks, last_n_in, last_n_out;
static uint32_t last_flags;
static rocket_task_desc last_tasks[CO_MAX_PROGRAMS];
static uint32_t last_ins[CO_COMMAND_MAX_HANDLES];
static uint32_t last_outs[CO_COMMAND_MAX_HANDLES];

int __wrap_rocket_submit_tasks_flags(int fd, const rocket_task_desc *tasks,
                                     uint32_t n_tasks, const uint32_t *ins,
                                     uint32_t n_in, const uint32_t *outs,
                                     uint32_t n_out, uint32_t flags) {
    (void)fd;
    submit_calls++;
    last_n_tasks = n_tasks;
    last_n_in = n_in;
    last_n_out = n_out;
    last_flags = flags;
    if (tasks) memcpy(last_tasks, tasks, n_tasks * sizeof(*tasks));
    if (ins) memcpy(last_ins, ins, n_in * sizeof(*ins));
    if (outs) memcpy(last_outs, outs, n_out * sizeof(*outs));
    if (submit_fail_at && submit_calls == submit_fail_at) return ROCKET_E_DEVICE;
    return 0;
}

static int make_tensor_programs(co_program p[2], co_view *x, co_view *w,
                                co_view *y0, co_view *y1, co_buffer b[4]) {
    co_tensor xt, wt, yt;
    if (co_tensor_input(&xt, 32, 256) || co_tensor_weight(&wt, 32, 256) ||
        co_tensor_matrix(&yt, 32, 32, CO_F16))
        return -1;
    if (co_buffer_alloc(7, xt.bytes, &b[0]) ||
        co_buffer_alloc(7, wt.bytes, &b[1]) ||
        co_buffer_alloc(7, yt.bytes, &b[2]) ||
        co_buffer_alloc(7, yt.bytes, &b[3]))
        return -1;
    if (co_view_init(x, &b[0], 0, &xt) || co_view_init(w, &b[1], 0, &wt) ||
        co_view_init(y0, &b[2], 0, &yt) || co_view_init(y1, &b[3], 0, &yt))
        return -1;
    if (co_emit_mm(&p[0], *x, *w, *y0) || co_emit_mm(&p[1], *x, *w, *y1))
        return -1;
    return 0;
}

static int same_tasks(const co_command_plan *a, const co_command_plan *b,
                      uint64_t base_a, uint64_t base_b) {
    if (a->n_tasks != b->n_tasks) return 0;
    for (unsigned i = 0; i < a->n_tasks; i++) {
        if (a->tasks[i].regcmd_count != b->tasks[i].regcmd_count) return 0;
        if ((uint64_t)a->tasks[i].regcmd - base_a !=
            (uint64_t)b->tasks[i].regcmd - base_b) return 0;
    }
    return 1;
}

static void command_plan_case(void) {
    co_program p[2];
    co_view x, w, y0, y1;
    co_buffer data_bos[4] = {0};
    EXPECT(make_tensor_programs(p, &x, &w, &y0, &y1, data_bos) == 0);

    co_execute_workspace ws1 = {0}, ws2 = {0};
    EXPECT(co_execute_workspace_init(7, 2, &ws1) == 0);
    EXPECT(co_execute_workspace_init(7, 2, &ws2) == 0);

    co_command_plan a, b, c;
    co_command_times ta, tb;
    EXPECT(co_command_plan_build(7, p, 2, CO_CHAIN, &ws1, &a, &ta) == 0);
    EXPECT(a.n_tasks == 2u && a.n_in == 3u && a.n_out == 2u);
    EXPECT(a.flags == (p[0].flags | ROCKET_JOB_BATCHED));
    EXPECT(a.mode == CO_CHAIN && a.words > 0);
    EXPECT(ta.clear_us >= 0 && ta.pack_us >= 0 && ta.verify_us >= 0 &&
           ta.sync_us >= 0 && ta.handles_us >= 0);

    /* Same implementation, second build: identical command bytes, task order,
     * sealed chain and handle sets.  This is the cache-vs-original comparison
     * without any address relocation. */
    size_t bytes = a.words * 8u;
    unsigned char *saved = malloc(bytes ? bytes : 1u);
    EXPECT(saved != NULL);
    if (saved) memcpy(saved, ws1.commands.ptr, bytes);
    EXPECT(co_command_plan_build(7, p, 2, CO_CHAIN, &ws1, &b, &tb) == 0);
    EXPECT(saved && memcmp(saved, ws1.commands.ptr, bytes) == 0);
    EXPECT(same_tasks(&a, &b, ws1.commands.dma_address, ws1.commands.dma_address));
    EXPECT(a.n_in == b.n_in && a.n_out == b.n_out &&
           memcmp(a.ins, b.ins, a.n_in * sizeof(a.ins[0])) == 0 &&
           memcmp(a.outs, b.outs, a.n_out * sizeof(a.outs[0])) == 0);

    /* Independent workspace: same relative task layout and handle set, but the
     * descriptors point inside that workspace's own command BO. */
    EXPECT(co_command_plan_build(7, p, 2, CO_CHAIN, &ws2, &c, NULL) == 0);
    EXPECT(c.flags == a.flags && c.mode == a.mode && c.words == a.words);
    EXPECT(c.n_in == a.n_in && c.n_out == a.n_out);
    EXPECT(memcmp(c.outs, a.outs, a.n_out * sizeof(a.outs[0])) == 0);
    EXPECT(memcmp(c.ins, a.ins,
                  (a.n_in - 1u) * sizeof(a.ins[0])) == 0);
    EXPECT(c.ins[c.n_in - 1u] == ws2.commands.handle);
    EXPECT(same_tasks(&a, &c, ws1.commands.dma_address, ws2.commands.dma_address));
    EXPECT(ws1.commands.handle != ws2.commands.handle);
    EXPECT(rkt_chain_verify(&ws2.commands, c.tasks, (int)c.n_tasks) == 0);

    /* Cached submit: no CPU write to the command BO, no command sync.  The
     * packed bytes must still be identical after many executions, and the
     * recorded handle/task sets must not move. */
    rk_mock_reset();
    submit_calls = 0;
    uint32_t *ins_copy = malloc(a.n_in * sizeof(a.ins[0]));
    uint32_t *outs_copy = malloc(a.n_out * sizeof(a.outs[0]));
    rocket_task_desc *tasks_copy = malloc(a.n_tasks * sizeof(a.tasks[0]));
    EXPECT(ins_copy && outs_copy && tasks_copy);
    if (!ins_copy || !outs_copy || !tasks_copy) {
        free(ins_copy); free(outs_copy); free(tasks_copy); free(saved);
        co_command_plan_release(&a); co_command_plan_release(&b);
        co_command_plan_release(&c);
        co_execute_workspace_destroy(&ws1); co_execute_workspace_destroy(&ws2);
        return;
    }
    memcpy(ins_copy, a.ins, a.n_in * sizeof(a.ins[0]));
    memcpy(outs_copy, a.outs, a.n_out * sizeof(a.outs[0]));
    memcpy(tasks_copy, a.tasks, a.n_tasks * sizeof(a.tasks[0]));
    for (unsigned i = 0; i < 200; i++)
        EXPECT(co_command_plan_submit(&a, 0, NULL) == 0);
    EXPECT(submit_calls == 200u);
    EXPECT(memcmp(saved, ws1.commands.ptr, bytes) == 0);
    EXPECT(a.n_in == last_n_in && a.n_out == last_n_out &&
           a.n_tasks == last_n_tasks && last_flags == a.flags);
    EXPECT(memcmp(ins_copy, last_ins, a.n_in * sizeof(a.ins[0])) == 0);
    EXPECT(memcmp(outs_copy, last_outs, a.n_out * sizeof(a.outs[0])) == 0);
    EXPECT(memcmp(tasks_copy, last_tasks, a.n_tasks * sizeof(a.tasks[0])) == 0);

    /* Sync policy is observable and opt-in.  Output sync is unchanged in both
     * arms; only the per-submit command-BO pair differs. */
    unsigned p0 = rk_mock_prep_calls(), f0 = rk_mock_fini_calls();
    for (unsigned i = 0; i < 5; i++) EXPECT(co_command_plan_submit(&a, 0, NULL) == 0);
    unsigned no_cmd_prep = rk_mock_prep_calls() - p0;
    unsigned no_cmd_fini = rk_mock_fini_calls() - f0;
    EXPECT(no_cmd_prep == 10u && no_cmd_fini == 10u); /* 2 output BOs x 5 */
    p0 = rk_mock_prep_calls(); f0 = rk_mock_fini_calls();
    for (unsigned i = 0; i < 5; i++) EXPECT(co_command_plan_submit(&a, 1, NULL) == 0);
    EXPECT(rk_mock_prep_calls() - p0 == 15u);
    EXPECT(rk_mock_fini_calls() - f0 == 15u);

    /* Failed submit: the plan is unchanged, no output sync follows, and a later
     * submit through the same plan still works. */
    unsigned before_calls = submit_calls;
    p0 = rk_mock_prep_calls(); f0 = rk_mock_fini_calls();
    submit_fail_at = submit_calls + 1u;
    EXPECT(co_command_plan_submit(&a, 0, NULL) == ROCKET_E_DEVICE);
    EXPECT(submit_calls == before_calls + 1u);
    EXPECT(rk_mock_prep_calls() == p0 && rk_mock_fini_calls() == f0);
    EXPECT(memcmp(saved, ws1.commands.ptr, bytes) == 0);
    submit_fail_at = 0;
    EXPECT(co_command_plan_submit(&a, 0, NULL) == 0);

    /* CO_SEPARATE validates the whole program array before it fans out.  A
     * later invalid program must produce zero submissions, not one for the
     * earlier valid program.  The legal path still submits once per program. */
    submit_calls = 0;
    co_program invalid_second = p[1];
    invalid_second.count = 0;
    EXPECT(co_execute_profiled(7, p, 2, CO_SEPARATE, &ws1, NULL) == 0);
    EXPECT(submit_calls == 2u);
    EXPECT((last_flags & ROCKET_JOB_BATCHED) == 0u);
    submit_calls = 0;
    {
        co_program bad_pair[2];
        memcpy(bad_pair, p, sizeof(bad_pair));
        bad_pair[1] = invalid_second;
        EXPECT(co_execute_profiled(7, bad_pair, 2, CO_SEPARATE, &ws1,
                                   NULL) == ROCKET_E_SHAPE);
        EXPECT(submit_calls == 0u);
    }

    /* Identity guards: wrong fd, wrong workspace and a reallocated command BO
     * are refused rather than executing a stale plan. */
    EXPECT(co_execute_cached_profiled(8, p, 2, CO_CHAIN, &ws1, &a, NULL, 0) ==
           ROCKET_E_SHAPE);
    EXPECT(co_execute_cached_profiled(7, p, 2, CO_SEPARATE, &ws1, &a, NULL, 0) ==
           ROCKET_E_SHAPE);
    co_program programs_copy[2];
    memcpy(programs_copy, p, sizeof(programs_copy));
    EXPECT(co_execute_cached_profiled(7, programs_copy, 2, CO_CHAIN, &ws1, &a,
                                      NULL, 0) == ROCKET_E_SHAPE);
    co_execute_workspace other = {0};
    EXPECT(co_execute_workspace_init(7, 2, &other) == 0);
    EXPECT(co_execute_cached_profiled(7, p, 2, CO_CHAIN, &other, &a, NULL, 0) ==
           ROCKET_E_SHAPE);
    co_execute_workspace_destroy(&other);

    /* Build failure does not leak a private command BO. */
    int live = rk_mock_live_bos();
    co_program bad = p[0];
    bad.count = 0;
    co_command_plan bad_plan;
    EXPECT(co_command_plan_build(7, &bad, 1, CO_CHAIN, NULL, &bad_plan, NULL) ==
           ROCKET_E_SHAPE);
    EXPECT(rk_mock_live_bos() == live);

    free(ins_copy); free(outs_copy); free(tasks_copy); free(saved);
    co_command_plan_release(&a);
    co_command_plan_release(&b);
    co_command_plan_release(&c);
    co_execute_workspace_destroy(&ws1);
    co_execute_workspace_destroy(&ws2);
    for (unsigned i = 0; i < 4; i++) co_buffer_free(&data_bos[i]);
}

int main(void) {
    command_plan_case();
    printf("command_cache: %u checks, %u failures\n", checks, failures);
    return failures ? 1 : 0;
}
