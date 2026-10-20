// SPDX-License-Identifier: GPL-3.0-or-later
/* Host-only RKPlan ABI/runtime test.  BO allocation and the blocking execution
 * seam are test doubles; plan validation, packing, emitters and lifetime checks
 * are the production code.  This proves orchestration only, not hardware numerics. */
#include "rk_plan_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks, failures, live, allocs, fail_alloc, execs, fail_exec;
static size_t resident;
static uint32_t next_handle = 1;
#define EXPECT(e) do { checks++; if (!(e)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #e); failures++; } } while (0)

int __wrap_rocket_bo_alloc32(int fd, size_t size, rocket_bo *bo) {
    (void)fd;
    if (++allocs == fail_alloc) return ROCKET_E_NOMEM;
    memset(bo, 0, sizeof(*bo));
    bo->ptr = calloc(1, size);
    if (!bo->ptr) return ROCKET_E_NOMEM;
    bo->size = size;
    bo->handle = next_handle++;
    bo->dma_address = (uint64_t)bo->handle * 0x400000u;
    live++;
    resident += size;
    return 0;
}
void __wrap_rocket_bo_free(int fd, rocket_bo *bo) {
    (void)fd;
    if (!bo || !bo->handle) return;
    free(bo->ptr);
    live--;
    resident -= bo->size;
    memset(bo, 0, sizeof(*bo));
}
int __wrap_rocket_bo_prep(int fd, rocket_bo *bo, int dir, uint64_t timeout) {
    (void)fd; (void)bo; (void)dir; (void)timeout;
    return 0;
}
int __wrap_rocket_bo_fini(int fd, rocket_bo *bo) {
    (void)fd; (void)bo;
    return 0;
}
int __wrap_rocket_batched_submit_supported(void) { return 1; }

static _Float16 *view_data(co_view v) {
    return (_Float16 *)((char *)v.buffer->bo.ptr + v.offset);
}

int __wrap_co_execute_profiled(int fd, const co_program *p, unsigned count,
                               enum co_mode mode, co_execute_workspace *workspace,
                               co_execute_profile *profile) {
    (void)fd; (void)mode; (void)workspace;
    if (profile) memset(profile, 0, sizeof(*profile));
    if (++execs == fail_exec) return ROCKET_E_DEVICE;
    for (unsigned i = 0; i < count; i++) {
        co_view y = p[i].output;
        co_view x = p[i].reads[0];
        co_view w = p[i].reads[1];
        _Float16 *dst = view_data(y);
        _Float16 *src = view_data(x);
        _Float16 *weight = view_data(w);
        for (unsigned r = 0; r < y.tensor.physical_rows; r++) {
            for (unsigned c = 0; c < y.tensor.physical_cols; c++) {
                double value = 0;
                for (unsigned q = 0; q < x.tensor.physical_cols; q++)
                    value += (double)src[co_index(&x.tensor, r, q)] *
                             weight[co_index(&w.tensor, c, q)];
                if (p[i].nreads == 3) {
                    co_view prior = p[i].reads[2];
                    value += view_data(prior)[co_index(&prior.tensor, r, c)];
                }
                dst[co_index(&y.tensor, r, c)] = (_Float16)value;
            }
        }
    }
    return 0;
}

static _Float16 value(unsigned i, unsigned seed) {
    return (_Float16)(((int)((i * 17u + seed) % 31u) - 15) / 64.0);
}

static void reference(unsigned m, unsigned k, unsigned n,
                      const _Float16 *x, const _Float16 *w, _Float16 *y) {
    for (unsigned r = 0; r < m; r++) {
        for (unsigned c = 0; c < n; c++) {
            _Float16 acc = 0;
            for (unsigned k0 = 0; k0 < k; k0 += 256) {
                double sum = acc;
                unsigned end = k - k0 > 256 ? k0 + 256 : k;
                for (unsigned q = k0; q < end; q++)
                    sum += (double)x[(size_t)r * k + q] *
                           w[(size_t)q * n + c];
                acc = (_Float16)sum;
            }
            y[(size_t)r * n + c] = acc;
        }
    }
}

int main(void) {
    enum { M = 32, K = 512, N = 48 };
    const uint32_t slack_guard = CO_SLACK + CO_GUARD;
    rk_value_desc values[3] = {
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_value_desc),
         .id = 0, .role = RK_ROLE_INPUT, .dtype = RK_DTYPE_F16, .rank = 2,
         .shape = {M, K}, .nbytes = (uint64_t)M * K * 2,
         .name = "A", .name_len = 1},
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_value_desc),
         .id = 1, .role = RK_ROLE_CONSTANT, .dtype = RK_DTYPE_F16, .rank = 2,
         .shape = {K, N}, .nbytes = (uint64_t)K * N * 2,
         .name = "B", .name_len = 1},
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_value_desc),
         .id = 2, .role = RK_ROLE_OUTPUT, .dtype = RK_DTYPE_F16, .rank = 2,
         .shape = {M, N}, .nbytes = (uint64_t)M * N * 2,
         .name = "Y", .name_len = 1},
    };

    co_tensor x_t, w_t, y_t;
    EXPECT(co_tensor_input(&x_t, M, 256) == 0);
    EXPECT(co_tensor_weight(&w_t, N, 256) == 0);
    EXPECT(co_tensor_matrix(&y_t, M, N, CO_F16) == 0);
    const uint64_t x_bytes = x_t.bytes, w_bytes = w_t.bytes, y_bytes = y_t.bytes;

    rk_buffer_desc buffers[6];
    memset(buffers, 0, sizeof(buffers));
    const uint64_t caps[6] = {x_bytes, x_bytes, w_bytes, w_bytes,
                              y_bytes, y_bytes};
    for (uint32_t i = 0; i < 6; i++) {
        buffers[i].abi_version = RK_ABI_VERSION;
        buffers[i].struct_size = sizeof(rk_buffer_desc);
        buffers[i].id = i;
        buffers[i].slot_id = i;
        buffers[i].alignment = 16;
        buffers[i].layout_id = i == 2 || i == 3 ? RK_LAYOUT_WEIGHT_F16
                                                : RK_LAYOUT_CUBE_F16;
        buffers[i].capacity = caps[i];
        buffers[i].lifetime_first = 0;
        buffers[i].lifetime_last = 6;
        buffers[i].padding_state = 0;
        buffers[i].role = i < 2 ? RK_ROLE_INPUT_SLICE :
                          (i < 4 ? RK_ROLE_WEIGHT : RK_ROLE_ACCUMULATOR);
    }
    buffers[4].role = RK_ROLE_ACCUMULATOR;
    buffers[5].role = RK_ROLE_OUTPUT;

    rk_tensor_desc tx = {
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_tensor_desc),
        .dtype = RK_DTYPE_F16, .layout = RK_LAYOUT_CUBE_F16,
        .rows = M, .cols = 256, .physical_rows = x_t.physical_rows,
        .physical_cols = x_t.physical_cols, .atom = 8, .bytes = x_t.bytes};
    rk_tensor_desc tw = {
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_tensor_desc),
        .dtype = RK_DTYPE_F16, .layout = RK_LAYOUT_WEIGHT_F16,
        .rows = N, .cols = 256, .physical_rows = w_t.physical_rows,
        .physical_cols = w_t.physical_cols, .atom = 32, .bytes = w_t.bytes};
    rk_tensor_desc ty = {
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_tensor_desc),
        .dtype = RK_DTYPE_F16, .layout = RK_LAYOUT_CUBE_F16,
        .rows = M, .cols = N, .physical_rows = y_t.physical_rows,
        .physical_cols = y_t.physical_cols, .atom = 8, .bytes = y_t.bytes};
    rk_view_desc views[6];
    memset(views, 0, sizeof(views));
    for (uint32_t i = 0; i < 6; i++) {
        views[i].abi_version = RK_ABI_VERSION;
        views[i].struct_size = sizeof(rk_view_desc);
        views[i].id = i;
        views[i].buffer_id = i;
        views[i].tensor = (i == 0 || i == 1) ? tx : (i == 2 || i == 3) ? tw : ty;
        views[i].valid_region[0] = 0;
        views[i].valid_region[1] = views[i].tensor.rows;
        views[i].valid_region[2] = 0;
        views[i].valid_region[3] = views[i].tensor.cols;
    }

    uint32_t a_x0 = 0, a_x1 = 1, a_w0 = 2, a_w1 = 3, a_mm = 4;
    uint32_t view_p = 4, view_z = 5;
    uint32_t mm_in[] = {0, 2};
    uint32_t acc_in[] = {1, 3, 4};
    uint32_t acc_dep[] = {a_mm};
    uint32_t down_in[] = {5};
    rk_action_desc actions[7];
    memset(actions, 0, sizeof(actions));
    for (uint32_t i = 0; i < 7; i++) {
        actions[i].abi_version = RK_ABI_VERSION;
        actions[i].struct_size = sizeof(rk_action_desc);
        actions[i].id = i;
        snprintf(actions[i].origin, sizeof(actions[i].origin), "host-test:%u", i);
    }
    actions[0].kind = RK_ACTION_UPLOAD;
    actions[0].source_binding = 0;
    actions[0].source_row_stride = K;
    actions[0].source_col_offset = 0;
    actions[0].noutputs = 1;
    actions[0].outputs = &a_x0;
    actions[1].kind = RK_ACTION_UPLOAD;
    actions[1].source_binding = 0;
    actions[1].source_row_stride = K;
    actions[1].source_col_offset = 256;
    actions[1].noutputs = 1;
    actions[1].outputs = &a_x1;
    actions[2].kind = RK_ACTION_UPLOAD;
    actions[2].source_binding = 1;
    actions[2].source_row_stride = N;
    actions[2].source_col_offset = 0;
    actions[2].source_row_offset = 0;
    actions[2].flags = RK_ACTION_FLAG_TRANSPOSE_SOURCE | RK_ACTION_FLAG_CACHE_CONSTANT;
    actions[2].noutputs = 1;
    actions[2].outputs = &a_w0;
    actions[3].kind = RK_ACTION_UPLOAD;
    actions[3].source_binding = 1;
    actions[3].source_row_stride = N;
    actions[3].source_col_offset = 0;
    actions[3].source_row_offset = 256;
    actions[3].flags = RK_ACTION_FLAG_TRANSPOSE_SOURCE | RK_ACTION_FLAG_CACHE_CONSTANT;
    actions[3].noutputs = 1;
    actions[3].outputs = &a_w1;
    actions[4].kind = RK_ACTION_NPU_TASK;
    actions[4].template_id = RK_TEMPLATE_MM_F16;
    actions[4].ninputs = 2;
    actions[4].inputs = mm_in;
    actions[4].noutputs = 1;
    actions[4].outputs = &view_p;
    actions[5].kind = RK_ACTION_NPU_TASK;
    actions[5].template_id = RK_TEMPLATE_MM_ACC_F16;
    actions[5].ninputs = 3;
    actions[5].inputs = acc_in;
    actions[5].noutputs = 1;
    actions[5].outputs = &view_z;
    actions[5].ndeps = 1;
    actions[5].deps = acc_dep;
    actions[6].kind = RK_ACTION_DOWNLOAD;
    actions[6].source_binding = 2;
    actions[6].ninputs = 1;
    actions[6].inputs = down_in;

    uint32_t task_ids[] = {4, 5};
    rk_submit_desc submit = {
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_submit_desc),
        .id = 0, .mode = RK_SUBMIT_CHAIN, .core_policy = RK_CORE_SINGLE,
        .ntask_ids = 2, .task_ids = task_ids};
    rk_publish_desc publish = {
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_publish_desc),
        .binding_id = 2, .view_id = 5};
    uint64_t payload = x_bytes * 2 + w_bytes * 2 + y_bytes * 2;
    uint64_t claimed = payload + 6u * slack_guard + 2u * CO_MAX_WORDS * 8u;
    rk_plan_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.abi_version = RK_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    desc.schema_version = RK_SCHEMA_VERSION;
    desc.profile_hash = 0x1234;
    desc.values = values;
    desc.nvalues = 3;
    desc.buffers = buffers;
    desc.nbuffers = 6;
    desc.views = views;
    desc.nviews = 6;
    desc.actions = actions;
    desc.nactions = 7;
    desc.submits = &submit;
    desc.nsubmits = 1;
    desc.publish = &publish;
    desc.npublish = 1;
    desc.device_budget_bytes = claimed;
    desc.host_budget_bytes = 1u << 20;
    desc.claimed_device_bytes = claimed;
    desc.claimed_host_bytes = 1u << 16;

    rk_diag diag;
    rk_plan *plan = NULL;
    EXPECT(rk_plan_create(&desc, &plan, &diag) == 0);
    EXPECT(plan != NULL);
    if (!plan) return 1;
    /* Direct descriptor negatives exercise the C verifier independently of the
     * Python/Pydantic-like layers. */
    rk_plan_desc bad = desc;
    rk_plan *badplan = (rk_plan *)1;
    bad.values = NULL;
    EXPECT(rk_plan_create(&bad, &badplan, &diag) != 0 && badplan == NULL);
    rk_action_desc bad_actions[7];
    memcpy(bad_actions, actions, sizeof(bad_actions));
    bad_actions[4].template_id = 99;
    bad = desc;
    bad.actions = bad_actions;
    badplan = (rk_plan *)1;
    EXPECT(rk_plan_create(&bad, &badplan, &diag) != 0 && badplan == NULL);
    rk_view_desc bad_views[6];
    memcpy(bad_views, views, sizeof(bad_views));
    bad_views[0].offset_bytes = 1;
    bad = desc;
    bad.views = bad_views;
    badplan = (rk_plan *)1;
    EXPECT(rk_plan_create(&bad, &badplan, &diag) != 0 && badplan == NULL);
    bad = desc;
    bad.claimed_device_bytes = 1;
    badplan = (rk_plan *)1;
    EXPECT(rk_plan_create(&bad, &badplan, &diag) != 0 && badplan == NULL);
    /* Descriptor arrays may be released/overwritten after rk_plan_create. */
    memset(actions, 0xA5, sizeof(actions));
    EXPECT(plan->nactions == 7 && plan->actions[4].template_id == RK_TEMPLATE_MM_F16);

    _Float16 *x = malloc((size_t)M * K * 2);
    _Float16 *w = malloc((size_t)K * N * 2);
    _Float16 *y = malloc((size_t)M * N * 2);
    _Float16 *ref = malloc((size_t)M * N * 2);
    EXPECT(x && w && y && ref);
    if (!x || !w || !y || !ref) return 1;
    for (size_t i = 0; i < (size_t)M * K; i++) x[i] = value((unsigned)i, 3);
    for (size_t i = 0; i < (size_t)K * N; i++) w[i] = value((unsigned)i, 9);
    reference(M, K, N, x, w, ref);

    rk_binding constants[] = {{
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_binding),
        .id = 1, .dtype = RK_DTYPE_F16, .rank = 2, .shape = {K, N},
        .data = w, .nbytes = (uint64_t)K * N * 2}};
    rk_workspace *ws = NULL;
    EXPECT(rk_workspace_create(3, plan, constants, 1, &ws, &diag) == 0);
    EXPECT(ws != NULL);
    if (!ws) return 1;
    EXPECT(live == 7); /* six data slots plus one command workspace BO */
    EXPECT(resident >= payload);

    rk_binding inputs[] = {{
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_binding),
        .id = 0, .dtype = RK_DTYPE_F16, .rank = 2, .shape = {M, K},
        .data = x, .nbytes = (uint64_t)M * K * 2}};
    rk_binding outputs[] = {{
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_binding),
        .id = 2, .dtype = RK_DTYPE_F16, .rank = 2, .shape = {M, N},
        .data = y, .nbytes = (uint64_t)M * N * 2}};
    rk_run_report report;
    memset(y, 0x23, (size_t)M * N * 2);
    EXPECT(rk_workspace_run(ws, inputs, 1, outputs, 1, &report, &diag) == 0);
    EXPECT(report.task_count == 2 && report.submit_count == 1);
    EXPECT(!memcmp(y, ref, (size_t)M * N * 2));

    /* A second run must re-upload changed inputs, not stale constants. */
    for (size_t i = 0; i < (size_t)M * K; i++) x[i] = value((unsigned)i, 11);
    reference(M, K, N, x, w, ref);
    EXPECT(rk_workspace_run(ws, inputs, 1, outputs, 1, &report, &diag) == 0);
    EXPECT(!memcmp(y, ref, (size_t)M * N * 2));

    /* Host input/output alias is rejected before any submission. */
    unsigned before = execs;
    rk_binding bad_out = outputs[0];
    bad_out.data = x;
    bad_out.nbytes = (uint64_t)M * N * 2;
    EXPECT(rk_workspace_run(ws, inputs, 1, &bad_out, 1, &report, &diag) == RK_E_ALIAS);
    EXPECT(execs == before);

    EXPECT(rk_workspace_destroy(ws, &diag) == 0);
    EXPECT(!live);
    EXPECT(resident == 0);
    rk_plan_destroy(plan);

    /* Descriptor-level rejection still happens without Python. */
    bad = desc;
    bad.nactions = 0;
    bad.actions = NULL;
    bad.nsubmits = 0;
    bad.submits = NULL;
    bad.npublish = 0;
    bad.publish = NULL;
    plan = NULL;
    EXPECT(rk_plan_create(&bad, &plan, &diag) != 0);
    EXPECT(plan == NULL);
    free(x); free(w); free(y); free(ref);
    printf("RK_PLAN host checks=%u failures=%u\n", checks, failures);
    return failures ? 1 : 0;
}
