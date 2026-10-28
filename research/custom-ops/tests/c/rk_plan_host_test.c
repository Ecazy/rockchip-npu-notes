// SPDX-License-Identifier: GPL-3.0-or-later
/* Host-only RKPlan ABI/runtime test.  BO allocation and the blocking execution
 * seam are test doubles; plan validation, packing, emitters and lifetime checks
 * are the production code.  This proves orchestration only, not hardware numerics. */
#include "rk_plan_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The mock seam is shared with librknpu_plan_mock.so; the API lives in one
 * test-only header (tests/mock/mock_npu.h). */
#include "mock_npu.h"
#define execs (rk_mock_execs())
#define live (rk_mock_live_bos())
#define live_bos (rk_mock_live_bos())
#define resident (rk_mock_resident())
static unsigned checks, failures;
#define allocs (rk_mock_allocs())
/* live_bos/resident/execs come from the shared mock seam (see the macros below). */
#define EXPECT(e) do { checks++; if (!(e)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #e); failures++; } } while (0)
/* Rejection evidence must pin the exact ABI contract: return code, null handle
 * and the matching diagnostic code, not merely "something failed". */
#define EXPECT_REJECT(call, out_handle, expected_code, needle) do { \
    checks++; \
    (out_handle) = (rk_plan *)1; \
    int rc_ = (call); \
    if (rc_ != (expected_code) || (out_handle) != NULL || \
        diag.code != (expected_code) || \
        ((needle) && !strstr(diag.message, (needle)))) { \
        fprintf(stderr, "FAIL line %d: %s -> rc=%d diag.code=%d out=%p \"%s\"\n", \
                __LINE__, #call, rc_, diag.code, (void *)(out_handle), diag.message); \
        failures++; \
    } \
} while (0)

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


/* Multi-tile output: M=257 splits into a full 256-row tile and a 1-row tail.
 * Two device tiles must publish into one [257,32] output binding at the right
 * offsets, and only the valid tail rows may be read back. */
static void multitile_publish_case(void) {
    enum { MM = 257, KK = 256, NN = 32 };
    rk_value_desc values[3] = {
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_value_desc),
         .id = 0, .role = RK_ROLE_INPUT, .dtype = RK_DTYPE_F16, .rank = 2,
         .shape = {MM, KK}, .nbytes = (uint64_t)MM * KK * 2,
         .name = "A", .name_len = 1},
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_value_desc),
         .id = 1, .role = RK_ROLE_CONSTANT, .dtype = RK_DTYPE_F16, .rank = 2,
         .shape = {KK, NN}, .nbytes = (uint64_t)KK * NN * 2,
         .name = "B", .name_len = 1},
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_value_desc),
         .id = 2, .role = RK_ROLE_OUTPUT, .dtype = RK_DTYPE_F16, .rank = 2,
         .shape = {MM, NN}, .nbytes = (uint64_t)MM * NN * 2,
         .name = "Y", .name_len = 1},
    };
    co_tensor x0_t, x1_t, w_t, y0_t, y1_t;
    EXPECT(co_tensor_input(&x0_t, 256, KK) == 0);
    EXPECT(co_tensor_input(&x1_t, 1, KK) == 0);
    EXPECT(co_tensor_weight(&w_t, NN, KK) == 0);
    EXPECT(co_tensor_matrix(&y0_t, 256, NN, CO_F16) == 0);
    EXPECT(co_tensor_matrix(&y1_t, 1, NN, CO_F16) == 0);
    const co_tensor *tensors[5] = {&x0_t, &x1_t, &w_t, &y0_t, &y1_t};
    const uint32_t layouts[5] = {RK_LAYOUT_CUBE_F16, RK_LAYOUT_CUBE_F16,
                                 RK_LAYOUT_WEIGHT_F16, RK_LAYOUT_CUBE_F16,
                                 RK_LAYOUT_CUBE_F16};
    const uint32_t roles[5] = {RK_ROLE_INPUT_SLICE, RK_ROLE_INPUT_SLICE,
                               RK_ROLE_WEIGHT, RK_ROLE_OUTPUT, RK_ROLE_OUTPUT};

    rk_buffer_desc buffers[5];
    rk_view_desc views[5];
    memset(buffers, 0, sizeof(buffers));
    memset(views, 0, sizeof(views));
    uint64_t payload = 0;
    for (uint32_t i = 0; i < 5; i++) {
        const co_tensor *t = tensors[i];
        payload += t->bytes;
        buffers[i].abi_version = RK_ABI_VERSION;
        buffers[i].struct_size = sizeof(rk_buffer_desc);
        buffers[i].id = i;
        buffers[i].role = roles[i];
        buffers[i].slot_id = i;
        buffers[i].alignment = 16;
        buffers[i].layout_id = layouts[i];
        buffers[i].capacity = t->bytes;
        buffers[i].lifetime_first = 0;
        buffers[i].lifetime_last = 6;
        views[i].abi_version = RK_ABI_VERSION;
        views[i].struct_size = sizeof(rk_view_desc);
        views[i].id = i;
        views[i].buffer_id = i;
        views[i].tensor.abi_version = RK_ABI_VERSION;
        views[i].tensor.struct_size = sizeof(rk_tensor_desc);
        views[i].tensor.dtype = RK_DTYPE_F16;
        views[i].tensor.layout = layouts[i];
        views[i].tensor.rows = t->rows == 0 ? 1 : t->rows;
        views[i].tensor.cols = t->cols;
        views[i].tensor.physical_rows = t->physical_rows;
        views[i].tensor.physical_cols = t->physical_cols;
        views[i].tensor.atom = t->atom;
        views[i].tensor.bytes = t->bytes;
        views[i].valid_region[1] = views[i].tensor.rows;
        views[i].valid_region[3] = views[i].tensor.cols;
    }

    uint32_t v_x0 = 0, v_x1 = 1, v_w = 2, v_y0 = 3, v_y1 = 4;
    uint32_t mm0_in[] = {0, 2};
    uint32_t mm1_in[] = {1, 2};
    uint32_t mm0_deps[] = {0, 2};
    uint32_t mm1_deps[] = {1, 2};
    uint32_t down0_in[] = {3};
    uint32_t down1_in[] = {4};
    uint32_t down0_deps[] = {3};
    uint32_t down1_deps[] = {4};
    rk_action_desc actions[7];
    memset(actions, 0, sizeof(actions));
    for (uint32_t i = 0; i < 7; i++) {
        actions[i].abi_version = RK_ABI_VERSION;
        actions[i].struct_size = sizeof(rk_action_desc);
        actions[i].id = i;
        snprintf(actions[i].origin, sizeof(actions[i].origin), "multitile:%u", i);
    }
    actions[0].kind = RK_ACTION_UPLOAD;
    actions[0].source_binding = 0;
    actions[0].source_row_stride = KK;
    actions[0].source_row_offset = 0;
    actions[0].noutputs = 1;
    actions[0].outputs = &v_x0;
    actions[1].kind = RK_ACTION_UPLOAD;
    actions[1].source_binding = 0;
    actions[1].source_row_stride = KK;
    actions[1].source_row_offset = 256;
    actions[1].noutputs = 1;
    actions[1].outputs = &v_x1;
    actions[2].kind = RK_ACTION_UPLOAD;
    actions[2].source_binding = 1;
    actions[2].source_row_stride = NN;
    actions[2].flags = RK_ACTION_FLAG_TRANSPOSE_SOURCE | RK_ACTION_FLAG_CACHE_CONSTANT;
    actions[2].noutputs = 1;
    actions[2].outputs = &v_w;
    actions[3].kind = RK_ACTION_NPU_TASK;
    actions[3].template_id = RK_TEMPLATE_MM_F16;
    actions[3].ninputs = 2;
    actions[3].inputs = mm0_in;
    actions[3].noutputs = 1;
    actions[3].outputs = &v_y0;
    actions[3].ndeps = 2;
    actions[3].deps = mm0_deps;
    actions[4].kind = RK_ACTION_NPU_TASK;
    actions[4].template_id = RK_TEMPLATE_MM_F16;
    actions[4].ninputs = 2;
    actions[4].inputs = mm1_in;
    actions[4].noutputs = 1;
    actions[4].outputs = &v_y1;
    actions[4].ndeps = 2;
    actions[4].deps = mm1_deps;
    actions[5].kind = RK_ACTION_DOWNLOAD;
    actions[5].source_binding = 2;
    actions[5].ninputs = 1;
    actions[5].inputs = down0_in;
    actions[5].ndeps = 1;
    actions[5].deps = down0_deps;
    actions[6].kind = RK_ACTION_DOWNLOAD;
    actions[6].source_binding = 2;
    actions[6].ninputs = 1;
    actions[6].inputs = down1_in;
    actions[6].ndeps = 1;
    actions[6].deps = down1_deps;

    uint32_t task_ids_0[] = {3};
    uint32_t task_ids_1[] = {4};
    uint32_t submit_deps[] = {0};
    rk_submit_desc submits[2] = {
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_submit_desc),
         .id = 0, .mode = RK_SUBMIT_CHAIN, .core_policy = RK_CORE_SINGLE,
         .ntask_ids = 1, .task_ids = task_ids_0},
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_submit_desc),
         .id = 1, .mode = RK_SUBMIT_CHAIN, .core_policy = RK_CORE_SINGLE,
         .ntask_ids = 1, .ndeps = 1, .task_ids = task_ids_1,
         .deps = submit_deps},
    };
    rk_publish_desc publish[2] = {
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_publish_desc),
         .binding_id = 2, .view_id = 3, .dst_row_offset = 0, .dst_col_offset = 0,
         .rows = 256, .cols = NN},
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_publish_desc),
         .binding_id = 2, .view_id = 4, .dst_row_offset = 256, .dst_col_offset = 0,
         .rows = 1, .cols = NN},
    };
    uint64_t claimed = payload + 5u * (CO_SLACK + CO_GUARD) + 2u * CO_MAX_WORDS * 8u;
    rk_plan_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.abi_version = RK_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    desc.schema_version = RK_SCHEMA_VERSION;
    desc.values = values;
    desc.nvalues = 3;
    desc.buffers = buffers;
    desc.nbuffers = 5;
    desc.views = views;
    desc.nviews = 5;
    desc.actions = actions;
    desc.nactions = 7;
    desc.submits = submits;
    desc.nsubmits = 2;
    desc.publish = publish;
    desc.npublish = 2;
    desc.device_budget_bytes = claimed;
    desc.host_budget_bytes = 1u << 23;
    desc.claimed_device_bytes = claimed;
    desc.claimed_host_bytes = 1u << 22; /* bindings + staging, see the C check */
    desc.runtime_abi_version = RK_ABI_VERSION;

    rk_diag diag;
    rk_plan *plan = NULL;
    EXPECT(rk_plan_create(&desc, &plan, &diag) == 0);
    if (!plan) return;

    /* The same two tiles must not be allowed to cover each other. */
    rk_publish_desc overlap[2] = {publish[0], publish[1]};
    overlap[1].dst_row_offset = 255;
    rk_plan_desc bad = desc;
    bad.publish = overlap;
    rk_plan *badplan = (rk_plan *)1;
    EXPECT(rk_plan_create(&bad, &badplan, &diag) != 0 && badplan == NULL);

    /* Missing the tail tile leaves the binding uncovered. */
    bad = desc;
    bad.npublish = 1;
    EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan, RK_E_SHAPE,
                  "cover the output binding");

    /* Two independent output tiles with interleaved live_bos ranges must not share
     * a slot: tile 0 is downloaded after tile 1 has been written. */
    {
        rk_buffer_desc shared[5];
        memcpy(shared, buffers, sizeof(shared));
        shared[4].slot_id = shared[3].slot_id;
        rk_plan_desc clash = desc;
        clash.buffers = shared;
        EXPECT_REJECT(rk_plan_create(&clash, &badplan, &diag), badplan,
                      RK_E_ALIAS, "overlapping live ranges");
    }

    _Float16 *x = malloc((size_t)MM * KK * 2);
    _Float16 *w = malloc((size_t)KK * NN * 2);
    _Float16 *y = malloc((size_t)MM * NN * 2);
    _Float16 *ref = malloc((size_t)MM * NN * 2);
    EXPECT(x && w && y && ref);
    if (!x || !w || !y || !ref) return;
    for (size_t i = 0; i < (size_t)MM * KK; i++) x[i] = value((unsigned)i, 5);
    for (size_t i = 0; i < (size_t)KK * NN; i++) w[i] = value((unsigned)i, 13);
    for (unsigned r = 0; r < MM; r++)
        for (unsigned c = 0; c < NN; c++) {
            double sum = 0;
            for (unsigned q = 0; q < KK; q++)
                sum += (double)x[(size_t)r * KK + q] * w[(size_t)q * NN + c];
            ref[(size_t)r * NN + c] = (_Float16)sum;
        }

    rk_binding constants[] = {{
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_binding),
        .id = 1, .dtype = RK_DTYPE_F16, .rank = 2, .shape = {KK, NN},
        .data = w, .nbytes = (uint64_t)KK * NN * 2}};
    rk_workspace *ws = NULL;
    EXPECT(rk_workspace_create(3, plan, constants, 1, &ws, &diag) == 0);
    if (!ws) return;
    /* The two-submit plan exercises the budgeted multi-submit cache through
     * the production runtime. */
    EXPECT(ws->multi_submit_ready == 1u);
    EXPECT(ws->multi_submit_cached >= 1u);
    rk_binding inputs[] = {{
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_binding),
        .id = 0, .dtype = RK_DTYPE_F16, .rank = 2, .shape = {MM, KK},
        .data = x, .nbytes = (uint64_t)MM * KK * 2}};
    rk_binding outputs[] = {{
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_binding),
        .id = 2, .dtype = RK_DTYPE_F16, .rank = 2, .shape = {MM, NN},
        .data = y, .nbytes = (uint64_t)MM * NN * 2}};
    rk_run_report report;
    memset(y, 0x23, (size_t)MM * NN * 2);
    EXPECT(rk_workspace_run(ws, inputs, 1, outputs, 1, &report, &diag) == 0);
    EXPECT(report.task_count == 2 && report.download_calls == 2);
    EXPECT(!memcmp(y, ref, (size_t)MM * NN * 2));

    /* A tight multi-submit budget may cache only part of the plan; the
     * uncached submits must keep the original path and the output must stay
     * identical. */
    {
        setenv("RK_PLAN_MULTI_SUBMIT_BUDGET", "30000", 1);
        rk_workspace *partial = NULL;
        EXPECT(rk_workspace_create(3, plan, constants, 1, &partial, &diag) == 0);
        EXPECT(partial != NULL);
        if (partial) {
            EXPECT(partial->multi_submit_ready == 1u);
            EXPECT(partial->multi_submit_cached == 1u);
            memset(y, 0x23, (size_t)MM * NN * 2);
            EXPECT(rk_workspace_run(partial, inputs, 1, outputs, 1,
                                    &report, &diag) == 0);
            EXPECT(!memcmp(y, ref, (size_t)MM * NN * 2));
            EXPECT(rk_workspace_destroy(partial, &diag) == 0);
        }
        setenv("RK_PLAN_MULTI_SUBMIT_BUDGET", "4194304", 1);
    }

    /* Input finiteness is rechecked on every call, even when the pointer and
     * binding metadata are unchanged.  Reject before execution/publication. */
    {
        const unsigned execs_before = execs;
        const unsigned char marker = 0x23;
        const _Float16 nonfinite[] = {
            (_Float16)NAN, (_Float16)INFINITY, (_Float16)-INFINITY
        };
        for (size_t bad = 0; bad < sizeof(nonfinite) / sizeof(nonfinite[0]); bad++) {
            memset(y, marker, (size_t)MM * NN * 2);
            x[0] = nonfinite[bad];
            EXPECT(rk_workspace_run(ws, inputs, 1, outputs, 1, &report, &diag) ==
                   RK_E_NUMERIC_CONTRACT);
            EXPECT(execs == execs_before);
            {
                int untouched = 1;
                for (size_t i = 0; i < (size_t)MM * NN * 2; i++)
                    if (((unsigned char *)y)[i] != marker) untouched = 0;
                EXPECT(untouched);
            }
        }
        x[0] = value(0, 5);
        EXPECT(rk_workspace_run(ws, inputs, 1, outputs, 1, &report, &diag) == 0);
        EXPECT(!memcmp(y, ref, (size_t)MM * NN * 2));
    }

    /* A non-finite device result must never reach the published output. */
    rk_mock_inject_nan(1);
    memset(y, 0x23, (size_t)MM * NN * 2);
    EXPECT(rk_workspace_run(ws, inputs, 1, outputs, 1, &report, &diag) ==
           RK_E_NUMERIC_CONTRACT);
    {
        int untouched = 1;
        for (size_t i = 0; i < (size_t)MM * NN * 2; i++)
            if (((unsigned char *)y)[i] != 0x23) untouched = 0;
        EXPECT(untouched);
    }
    rk_mock_inject_nan(0);
    EXPECT(rk_workspace_run(ws, inputs, 1, outputs, 1, &report, &diag) == 0);
    EXPECT(!memcmp(y, ref, (size_t)MM * NN * 2));
    EXPECT(rk_workspace_destroy(ws, &diag) == 0);
    EXPECT(!live_bos && resident == 0);
    rk_plan_destroy(plan);
    free(x); free(w); free(y); free(ref);
}


/* A writer that touches a buffer is not the same as a writer that covers the
 * reader's region.  Uploading 12 rows into a 13-row cube must be rejected even
 * though the buffer has a producer. */
static void partial_coverage_case(void) {
    enum { M = 13, K = 64, N = 32 };
    rk_value_desc values[3] = {
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_value_desc),
         .id = 0, .role = RK_ROLE_INPUT, .dtype = RK_DTYPE_F16, .rank = 2,
         .shape = {M, K}, .nbytes = (uint64_t)M * K * 2, .name = "A", .name_len = 1},
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_value_desc),
         .id = 1, .role = RK_ROLE_CONSTANT, .dtype = RK_DTYPE_F16, .rank = 2,
         .shape = {K, N}, .nbytes = (uint64_t)K * N * 2, .name = "B", .name_len = 1},
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_value_desc),
         .id = 2, .role = RK_ROLE_OUTPUT, .dtype = RK_DTYPE_F16, .rank = 2,
         .shape = {M, N}, .nbytes = (uint64_t)M * N * 2, .name = "Y", .name_len = 1},
    };
    co_tensor x13, x12, w, y;
    EXPECT(co_tensor_input(&x13, M, K) == 0);
    EXPECT(co_tensor_input(&x12, 12, K) == 0);
    EXPECT(co_tensor_weight(&w, N, K) == 0);
    EXPECT(co_tensor_matrix(&y, M, N, CO_F16) == 0);

    rk_buffer_desc buffers[3];
    memset(buffers, 0, sizeof(buffers));
    const uint64_t caps[3] = {x13.bytes, w.bytes, y.bytes};
    const uint32_t lyt[3] = {RK_LAYOUT_CUBE_F16, RK_LAYOUT_WEIGHT_F16,
                             RK_LAYOUT_CUBE_F16};
    for (uint32_t i = 0; i < 3; i++) {
        buffers[i].abi_version = RK_ABI_VERSION;
        buffers[i].struct_size = sizeof(rk_buffer_desc);
        buffers[i].id = i;
        buffers[i].slot_id = i;
        buffers[i].alignment = 16;
        buffers[i].layout_id = lyt[i];
        buffers[i].capacity = caps[i];
        buffers[i].lifetime_first = 0;
        buffers[i].lifetime_last = 3;
    }
    rk_view_desc views[4];
    memset(views, 0, sizeof(views));
    const co_tensor *tv[4] = {&x12, &x13, &w, &y};
    const uint32_t tbuf[4] = {0, 0, 1, 2};
    for (uint32_t i = 0; i < 4; i++) {
        views[i].abi_version = RK_ABI_VERSION;
        views[i].struct_size = sizeof(rk_view_desc);
        views[i].id = i;
        views[i].buffer_id = tbuf[i];
        views[i].tensor.abi_version = RK_ABI_VERSION;
        views[i].tensor.struct_size = sizeof(rk_tensor_desc);
        views[i].tensor.dtype = RK_DTYPE_F16;
        views[i].tensor.layout = lyt[tbuf[i]];
        views[i].tensor.rows = tv[i]->rows;
        views[i].tensor.cols = tv[i]->cols;
        views[i].tensor.physical_rows = tv[i]->physical_rows;
        views[i].tensor.physical_cols = tv[i]->physical_cols;
        views[i].tensor.atom = tv[i]->atom;
        views[i].tensor.bytes = tv[i]->bytes;
        views[i].valid_region[1] = views[i].tensor.rows;
        views[i].valid_region[3] = views[i].tensor.cols;
    }
    uint32_t v_xw = 0, v_xr = 1, v_w = 2, v_y = 3;
    uint32_t mm_in[] = {1, 2};
    uint32_t up_in[] = {0, 1};
    uint32_t down_in[] = {3};
    uint32_t down_deps[] = {2};
    rk_action_desc actions[4];
    memset(actions, 0, sizeof(actions));
    for (uint32_t i = 0; i < 4; i++) {
        actions[i].abi_version = RK_ABI_VERSION;
        actions[i].struct_size = sizeof(rk_action_desc);
        actions[i].id = i;
    }
    actions[0].kind = RK_ACTION_UPLOAD;
    actions[0].source_binding = 0;
    actions[0].source_row_stride = K;
    actions[0].noutputs = 1;
    actions[0].outputs = &v_xw;
    actions[1].kind = RK_ACTION_UPLOAD;
    actions[1].source_binding = 1;
    actions[1].source_row_stride = N;
    actions[1].flags = RK_ACTION_FLAG_TRANSPOSE_SOURCE | RK_ACTION_FLAG_CACHE_CONSTANT;
    actions[1].noutputs = 1;
    actions[1].outputs = &v_w;
    actions[2].kind = RK_ACTION_NPU_TASK;
    actions[2].template_id = RK_TEMPLATE_MM_F16;
    actions[2].ninputs = 2;
    actions[2].inputs = mm_in;
    actions[2].noutputs = 1;
    actions[2].outputs = &v_y;
    actions[2].ndeps = 2;
    actions[2].deps = up_in;
    actions[3].kind = RK_ACTION_DOWNLOAD;
    actions[3].source_binding = 2;
    actions[3].ninputs = 1;
    actions[3].inputs = down_in;
    actions[3].ndeps = 1;
    actions[3].deps = down_deps;

    uint32_t task_ids[] = {2};
    rk_submit_desc submit = {
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_submit_desc),
        .id = 0, .mode = RK_SUBMIT_CHAIN, .core_policy = RK_CORE_SINGLE,
        .ntask_ids = 1, .task_ids = task_ids};
    rk_publish_desc publish = {
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_publish_desc),
        .binding_id = 2, .view_id = 3, .dst_row_offset = 0, .dst_col_offset = 0,
        .rows = M, .cols = N};
    uint64_t claimed = x13.bytes + w.bytes + y.bytes + 3u * (CO_SLACK + CO_GUARD) +
                       1u * CO_MAX_WORDS * 8u;
    rk_plan_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.abi_version = RK_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    desc.schema_version = RK_SCHEMA_VERSION;
    desc.values = values;
    desc.nvalues = 3;
    desc.buffers = buffers;
    desc.nbuffers = 3;
    desc.views = views;
    desc.nviews = 4;
    desc.actions = actions;
    desc.nactions = 4;
    desc.submits = &submit;
    desc.nsubmits = 1;
    desc.publish = &publish;
    desc.npublish = 1;
    desc.device_budget_bytes = claimed;
    desc.host_budget_bytes = 1u << 23;
    desc.claimed_device_bytes = claimed;
    desc.claimed_host_bytes = 1u << 22; /* bindings + staging, see the C check */
    desc.runtime_abi_version = RK_ABI_VERSION;

    rk_diag diag;
    rk_plan *plan = NULL;
    /* Either "no write before this read produces the consumed range" (nothing
     * addresses the reader) or "... do not cover the consumed range" (partial
     * coverage) is correct; both name the consumed range. */
    EXPECT_REJECT(rk_plan_create(&desc, &plan, &diag), plan, RK_E_SHAPE,
                  "consumed range");

    /* Widening the producer to the full 13-row cube makes the plan valid. */
    rk_action_desc fixed[4];
    memcpy(fixed, actions, sizeof(fixed));
    fixed[0].outputs = &v_xr;
    rk_plan_desc good = desc;
    good.actions = fixed;
    plan = NULL;
    EXPECT(rk_plan_create(&good, &plan, &diag) == 0);
    EXPECT(plan != NULL);
    rk_plan_destroy(plan);
}


/* LUT segmentation: one buffer written by several *disjoint* linear segments of
 * different sizes, then read back as the original cube by an EW multiply.
 *
 *   0 upload identity (constant)
 *   1 upload acc
 *   2 LUT(acc.lo)  -> gate.lo
 *   3 LUT(acc.hi)  -> gate.hi
 *   4 EW(acc, gate, identity) -> out
 *   5 download out
 *
 * The verifier must accept the unequal segments, and must reject the reordered
 * program where the EW multiply runs between the two segments: it would read the
 * upper half of the gate buffer before that half exists. */
static void lut_segment_case(void) {
    enum { ROWS = 16, COLS = 32, LO = 128, HI = 384 }; /* 512 elements total */
    rk_value_desc values[3] = {
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_value_desc),
         .id = 0, .role = RK_ROLE_INPUT, .dtype = RK_DTYPE_F16, .rank = 2,
         .shape = {ROWS, COLS}, .nbytes = (uint64_t)ROWS * COLS * 2,
         .name = "A", .name_len = 1},
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_value_desc),
         .id = 1, .role = RK_ROLE_CONSTANT, .dtype = RK_DTYPE_F16, .rank = 2,
         .shape = {COLS, COLS}, .nbytes = (uint64_t)COLS * COLS * 2,
         .name = "I", .name_len = 1},
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_value_desc),
         .id = 2, .role = RK_ROLE_OUTPUT, .dtype = RK_DTYPE_F16, .rank = 2,
         .shape = {ROWS, COLS}, .nbytes = (uint64_t)ROWS * COLS * 2,
         .name = "Y", .name_len = 1},
    };
    co_tensor cube_t, weight_t;
    EXPECT(co_tensor_matrix(&cube_t, ROWS, COLS, CO_F16) == 0);
    EXPECT(co_tensor_weight(&weight_t, COLS, COLS) == 0);

    rk_buffer_desc buffers[4];
    memset(buffers, 0, sizeof(buffers));
    const uint64_t caps[4] = {cube_t.bytes, cube_t.bytes, weight_t.bytes,
                              cube_t.bytes};
    const uint32_t lyt[4] = {RK_LAYOUT_CUBE_F16, RK_LAYOUT_CUBE_F16,
                             RK_LAYOUT_WEIGHT_F16, RK_LAYOUT_CUBE_F16};
    const uint32_t roles[4] = {RK_ROLE_INPUT_SLICE, RK_ROLE_GATE,
                               RK_ROLE_IDENTITY, RK_ROLE_OUTPUT};
    for (uint32_t i = 0; i < 4; i++) {
        buffers[i].abi_version = RK_ABI_VERSION;
        buffers[i].struct_size = sizeof(rk_buffer_desc);
        buffers[i].id = i;
        buffers[i].role = roles[i];
        buffers[i].slot_id = i;
        buffers[i].alignment = 16;
        buffers[i].layout_id = lyt[i];
        buffers[i].capacity = caps[i];
        buffers[i].lifetime_first = 0;
        buffers[i].lifetime_last = 5;
    }

    rk_view_desc views[8];
    memset(views, 0, sizeof(views));
    /* 0 acc cube, 1 acc lo, 2 acc hi, 3 gate lo, 4 gate hi, 5 gate cube,
     * 6 identity, 7 output */
    const uint32_t view_buffer[8] = {0, 0, 0, 1, 1, 1, 2, 3};
    const uint32_t view_layout[8] = {RK_LAYOUT_CUBE_F16, RK_LAYOUT_LINEAR_F16,
                                     RK_LAYOUT_LINEAR_F16, RK_LAYOUT_LINEAR_F16,
                                     RK_LAYOUT_LINEAR_F16, RK_LAYOUT_CUBE_F16,
                                     RK_LAYOUT_WEIGHT_F16, RK_LAYOUT_CUBE_F16};
    const uint64_t view_bytes[8] = {0, 0, (uint64_t)LO * 2, 0, (uint64_t)LO * 2,
                                    0, 0, 0};
    const uint32_t view_cols[8] = {COLS, LO, HI, LO, HI, COLS, COLS, COLS};
    const uint32_t view_rows[8] = {ROWS, 1, 1, 1, 1, ROWS, COLS, ROWS};
    for (uint32_t i = 0; i < 8; i++) {
        views[i].abi_version = RK_ABI_VERSION;
        views[i].struct_size = sizeof(rk_view_desc);
        views[i].id = i;
        views[i].buffer_id = view_buffer[i];
        views[i].offset_bytes = view_bytes[i];
        views[i].tensor.abi_version = RK_ABI_VERSION;
        views[i].tensor.struct_size = sizeof(rk_tensor_desc);
        views[i].tensor.dtype = RK_DTYPE_F16;
        views[i].tensor.layout = view_layout[i];
        views[i].tensor.rows = view_rows[i];
        views[i].tensor.cols = view_cols[i];
        if (view_layout[i] == RK_LAYOUT_LINEAR_F16) {
            views[i].tensor.physical_rows = 1;
            views[i].tensor.physical_cols = view_cols[i];
            views[i].tensor.atom = 1;
            views[i].tensor.bytes = (uint64_t)view_cols[i] * 2;
        } else if (view_layout[i] == RK_LAYOUT_WEIGHT_F16) {
            views[i].tensor.physical_rows = COLS;
            views[i].tensor.physical_cols = COLS;
            views[i].tensor.atom = 32;
            views[i].tensor.bytes = (uint64_t)COLS * COLS * 2;
        } else {
            views[i].tensor.physical_rows = ROWS;
            views[i].tensor.physical_cols = COLS;
            views[i].tensor.atom = 8;
            views[i].tensor.bytes = (uint64_t)ROWS * COLS * 2;
        }
        views[i].valid_region[1] = views[i].tensor.rows;
        views[i].valid_region[3] = views[i].tensor.cols;
    }

    uint32_t v_acc = 0, v_gate_lo = 3, v_gate_hi = 4, v_identity = 6, v_out = 7;
    uint32_t lut_lo_in[] = {1}, lut_hi_in[] = {2}, ew_in[] = {0, 5, 6};
    uint32_t down_in[] = {7};
    uint32_t dep_acc[] = {1};
    uint32_t dep_lo[] = {1};
    uint32_t dep_hi[] = {2};
    uint32_t dep_ew[] = {3, 0, 1};
    uint32_t dep_down[] = {4};

    rk_action_desc actions[6];
    memset(actions, 0, sizeof(actions));
    for (uint32_t i = 0; i < 6; i++) {
        actions[i].abi_version = RK_ABI_VERSION;
        actions[i].struct_size = sizeof(rk_action_desc);
        actions[i].id = i;
        snprintf(actions[i].origin, sizeof(actions[i].origin), "lut-seg:%u", i);
    }
    actions[0].kind = RK_ACTION_UPLOAD;
    actions[0].source_binding = 1;
    actions[0].source_row_stride = COLS;
    actions[0].flags = RK_ACTION_FLAG_TRANSPOSE_SOURCE |
                       RK_ACTION_FLAG_CACHE_CONSTANT;
    actions[0].noutputs = 1;
    actions[0].outputs = &v_identity;
    actions[1].kind = RK_ACTION_UPLOAD;
    actions[1].source_binding = 0;
    actions[1].source_row_stride = COLS;
    actions[1].noutputs = 1;
    actions[1].outputs = &v_acc;
    actions[2].kind = RK_ACTION_NPU_TASK;
    actions[2].template_id = RK_TEMPLATE_LUT_SIGMOID_F16;
    actions[2].ninputs = 1;
    actions[2].inputs = lut_lo_in;
    actions[2].noutputs = 1;
    actions[2].outputs = &v_gate_lo;
    actions[2].ndeps = 1;
    actions[2].deps = dep_lo;
    actions[3].kind = RK_ACTION_NPU_TASK;
    actions[3].template_id = RK_TEMPLATE_LUT_SIGMOID_F16;
    actions[3].ninputs = 1;
    actions[3].inputs = lut_hi_in;
    actions[3].noutputs = 1;
    actions[3].outputs = &v_gate_hi;
    actions[3].ndeps = 1;
    actions[3].deps = dep_hi;
    actions[4].kind = RK_ACTION_NPU_TASK;
    actions[4].template_id = RK_TEMPLATE_EW_MUL_F16;
    actions[4].ninputs = 3;
    actions[4].inputs = ew_in;
    actions[4].noutputs = 1;
    actions[4].outputs = &v_out;
    actions[4].ndeps = 3;
    actions[4].deps = dep_ew;
    actions[5].kind = RK_ACTION_DOWNLOAD;
    actions[5].source_binding = 2;
    actions[5].ninputs = 1;
    actions[5].inputs = down_in;
    actions[5].ndeps = 1;
    actions[5].deps = dep_down;
    (void)dep_acc;

    uint32_t task_ids[] = {2, 3, 4};
    rk_submit_desc submit = {
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_submit_desc),
        .id = 0, .mode = RK_SUBMIT_CHAIN, .core_policy = RK_CORE_SINGLE,
        .ntask_ids = 3, .task_ids = task_ids};
    rk_publish_desc publish = {
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_publish_desc),
        .binding_id = 2, .view_id = 7, .dst_row_offset = 0, .dst_col_offset = 0,
        .rows = ROWS, .cols = COLS};
    uint64_t claimed = 3u * cube_t.bytes + weight_t.bytes +
                       4u * (CO_SLACK + CO_GUARD) + 3u * CO_MAX_WORDS * 8u;
    rk_plan_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.abi_version = RK_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    desc.schema_version = RK_SCHEMA_VERSION;
    desc.values = values;
    desc.nvalues = 3;
    desc.buffers = buffers;
    desc.nbuffers = 4;
    desc.views = views;
    desc.nviews = 8;
    desc.actions = actions;
    desc.nactions = 6;
    desc.submits = &submit;
    desc.nsubmits = 1;
    desc.publish = &publish;
    desc.npublish = 1;
    desc.device_budget_bytes = claimed;
    desc.host_budget_bytes = 1u << 23;
    desc.claimed_device_bytes = claimed;
    desc.claimed_host_bytes = 1u << 22; /* bindings + staging, see the C check */
    desc.runtime_abi_version = RK_ABI_VERSION;

    rk_diag diag;
    rk_plan *plan = NULL;
    /* Unequal, disjoint segments must be accepted. */
    int good_rc = rk_plan_create(&desc, &plan, &diag);
    if (good_rc) fprintf(stderr, "LUT-SEG DIAG: %d %s\n", diag.code, diag.message);
    EXPECT(good_rc == 0);
    EXPECT(plan != NULL);
    if (plan) {
        int lo = -1, hi = -1;
        for (uint32_t i = 0; i < plan->nbuffers; i++)
            if (plan->buffers[i].role == RK_ROLE_GATE) { lo = (int)i; break; }
        EXPECT(lo >= 0);
        (void)hi;
        rk_plan_destroy(plan);
    }

    /* Reordered program: EW multiply between the two LUT segments.  The gate
     * cube reader then has only the lower half produced before it runs. */
    rk_action_desc reordered[6];
    memcpy(reordered, actions, sizeof(reordered));
    uint32_t dep_ew_early[] = {2, 0, 1};
    uint32_t dep_hi_late[] = {3};
    uint32_t dep_down_late[] = {4};
    reordered[3].template_id = RK_TEMPLATE_EW_MUL_F16;
    reordered[3].ninputs = 3;
    reordered[3].inputs = ew_in;
    reordered[3].noutputs = 1;
    reordered[3].outputs = &v_out;
    reordered[3].ndeps = 3;
    reordered[3].deps = dep_ew_early;
    reordered[4].template_id = RK_TEMPLATE_LUT_SIGMOID_F16;
    reordered[4].ninputs = 1;
    reordered[4].inputs = lut_hi_in;
    reordered[4].noutputs = 1;
    reordered[4].outputs = &v_gate_hi;
    reordered[4].ndeps = 1;
    reordered[4].deps = dep_hi_late;
    reordered[5].deps = dep_down_late;
    /* A LUT run that is interrupted by the EW multiply is not a registered
     * chain, so the submit whitelist rejects it before coverage is evaluated.
     * The coverage rule itself is exercised by ``partial_coverage_case``. */
    rk_plan_desc bad = desc;
    bad.actions = reordered;
    EXPECT_REJECT(rk_plan_create(&bad, &plan, &diag), plan, RK_E_SCHEMA,
                  "submit task sequence");

    /* --- Numeric regression on the rk_plan path -----------------------------
     * Runs the same descriptor through rk_plan_create -> rk_workspace_create ->
     * rk_workspace_run.  The reference is computed independently in logical
     * row-major order with the declared FP16 rounding points (LUT output and EW
     * output are both stored as FP16); it does not reuse the mock's helpers. */
    {
        const unsigned rows = ROWS, cols = COLS;
        _Float16 *acc = malloc((size_t)rows * cols * 2);
        _Float16 *out = malloc((size_t)rows * cols * 2);
        _Float16 *identity = malloc((size_t)cols * cols * 2);
        _Float16 *gate_ref = malloc((size_t)rows * cols * 2);
        _Float16 *ref = malloc((size_t)rows * cols * 2);
        EXPECT(acc && out && identity && gate_ref && ref);
        if (!acc || !out || !identity || !gate_ref || !ref) return;
        for (size_t i = 0; i < (size_t)cols * cols; i++) identity[i] = 0;
        for (unsigned i = 0; i < cols; i++)
            identity[(size_t)i * cols + i] = (_Float16)1;

        rk_plan *live_plan = NULL;
        EXPECT(rk_plan_create(&desc, &live_plan, &diag) == 0);
        EXPECT(live_plan != NULL);
        if (!live_plan) return;
        rk_binding constants[] = {{
            .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_binding),
            .id = 1, .dtype = RK_DTYPE_F16, .rank = 2, .shape = {cols, cols},
            .data = identity, .nbytes = (uint64_t)cols * cols * 2}};
        rk_workspace *ws = NULL;
        EXPECT(rk_workspace_create(0, live_plan, constants, 1, &ws, &diag) == 0);
        EXPECT(ws != NULL);
        if (!ws) { rk_plan_destroy(live_plan); return; }
        /* A single-submit workspace with a ready co_program cache must also
         * own the packed command plan; the control arm with that prerequisite
         * removed must not. */
        EXPECT(ws->program_cache_ready == 1u);
        EXPECT(ws->command_cache_ready == 1u);
        EXPECT(ws->command_plan != NULL);
        setenv("RK_PLAN_PROGRAM_CACHE", "0", 1);
        rk_workspace *no_program_cache = NULL;
        EXPECT(rk_workspace_create(0, live_plan, constants, 1,
                                   &no_program_cache, &diag) == 0);
        EXPECT(no_program_cache && no_program_cache->command_cache_ready == 0u &&
               no_program_cache->command_plan == NULL);
        if (no_program_cache) rk_workspace_destroy(no_program_cache, &diag);
        setenv("RK_PLAN_PROGRAM_CACHE", "1", 1);
        /* Every simulated BO owns an independent, aligned, non-overlapping
         * 32-bit DMA window, and the lookup agrees with the table. */
        {
            unsigned count = rk_mock_bo_count();
            EXPECT(count > 0);
            for (unsigned i = 0; i < count; i++) {
                uint32_t handle = 0, dma = 0;
                size_t size = 0;
                EXPECT(rk_mock_dma_range(i, &handle, &dma, &size) == 0);
                EXPECT(size > 0 && (size % 16u) == 0);
                EXPECT(dma >= 0x01000000u && dma % 16u == 0);
                EXPECT(dma + size <= 0xF0000000u);
                EXPECT(rk_mock_dma_of(handle) == dma);
                for (unsigned j = i + 1u; j < count; j++) {
                    uint32_t h2 = 0, d2 = 0;
                    size_t s2 = 0;
                    EXPECT(rk_mock_dma_range(j, &h2, &d2, &s2) == 0);
                    EXPECT(h2 != handle);                    /* unique handles */
                    EXPECT(dma + size <= d2 || d2 + s2 <= dma); /* no overlap */
                }
            }
        }
        rk_binding inputs[] = {{
            .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_binding),
            .id = 0, .dtype = RK_DTYPE_F16, .rank = 2, .shape = {rows, cols},
            .data = acc, .nbytes = (uint64_t)rows * cols * 2}};
        rk_binding outputs[] = {{
            .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_binding),
            .id = 2, .dtype = RK_DTYPE_F16, .rank = 2, .shape = {rows, cols},
            .data = out, .nbytes = (uint64_t)rows * cols * 2}};
        rk_run_report report;

        for (unsigned pass = 0; pass < 2; pass++) {
            for (unsigned i = 0; i < rows * cols; i++)
                acc[i] = pass == 0 ? (_Float16)(((int)(i % 23u) - 11) / 32.0)
                                   : (_Float16)(((int)(i % 11u) - 5) / 16.0);
            for (unsigned i = 0; i < rows * cols; i++) {
                double v = (double)acc[i];
                gate_ref[i] = (_Float16)(1.0 / (1.0 + exp(-v)));
                ref[i] = (_Float16)((double)acc[i] * (double)gate_ref[i]);
            }
            memset(out, 0x23, (size_t)rows * cols * 2);
            EXPECT(rk_workspace_run(ws, inputs, 1, outputs, 1, &report, &diag) == 0);
            unsigned bad = 0;
            for (unsigned i = 0; i < rows * cols; i++)
                if ((double)out[i] != (double)ref[i]) bad++;
            EXPECT(bad == 0);
        }
        rk_mock_inject_nan(1);
        memset(out, 0x23, (size_t)rows * cols * 2);
        EXPECT(rk_workspace_run(ws, inputs, 1, outputs, 1, &report, &diag) ==
               RK_E_NUMERIC_CONTRACT);
        {
            int untouched = 1;
            for (size_t i = 0; i < (size_t)rows * cols * 2; i++)
                if (((unsigned char *)out)[i] != 0x23) untouched = 0;
            EXPECT(untouched);
        }
        rk_mock_inject_nan(0);
        /* The identity weight is content-checked: corrupting it must change the
         * result, otherwise the mock would silently ignore the operand. */
        EXPECT(rk_workspace_destroy(ws, &diag) == 0);
        /* Perturb the HOST row-major identity: with TRANSPOSE_SOURCE the device
         * weight is W[c][q] = host[q][c], so host[0][1]=3 sets W[1][0]=3 and only
         * column 1 may change. */
        identity[0 * cols + 1] = (_Float16)3;
        rk_workspace *ws2 = NULL;
        EXPECT(rk_workspace_create(0, live_plan, constants, 1, &ws2, &diag) == 0);
        EXPECT(ws2 != NULL);
        if (ws2) {
            memset(out, 0, (size_t)rows * cols * 2);
            EXPECT(rk_workspace_run(ws2, inputs, 1, outputs, 1, &report, &diag) == 0);
            unsigned bad_r = UINT32_MAX, bad_c = UINT32_MAX;
            for (unsigned r = 0; r < rows; r++) {
                double pass1 = (double)acc[(size_t)r * cols + 1] +
                              3.0 * (double)acc[(size_t)r * cols + 0];
                /* the EW operand is the sigmoid LUT output, not x */
                double expect1 = pass1 * (double)gate_ref[(size_t)r * cols + 1];
                if ((double)out[(size_t)r * cols + 1] != (_Float16)expect1 &&
                    bad_r == UINT32_MAX) { bad_r = r; bad_c = 1; }
                EXPECT((double)out[(size_t)r * cols + 1] == (_Float16)expect1);
                for (unsigned c = 0; c < cols; c++) {
                    if (c == 1) continue;
                    if ((double)out[(size_t)r * cols + c] !=
                        (double)ref[(size_t)r * cols + c] && bad_r == UINT32_MAX)
                        { bad_r = r; bad_c = c; }
                    EXPECT((double)out[(size_t)r * cols + c] ==
                           (double)ref[(size_t)r * cols + c]);
                }
            }
            {
                unsigned where_bo = 0;
                size_t where_off = 0;
                uint16_t three = 0x4200; /* (_Float16)3.0 */
                /* The perturbation must land at device W[1][0]: flat 32 inside a
                 * 32-wide packed atom.  This is the direct upload-orientation
                 * proof, asserted rather than printed. */
                EXPECT(rk_mock_find_half(three, &where_bo, &where_off) == 0);
                EXPECT(where_off == (size_t)cols);
            }
            if (bad_r != UINT32_MAX)
                fprintf(stderr,
                        "EW-ORIENT first mismatch r=%u c=%u out=%.6f expected_perturbed=%.6f ref_unperturbed=%.6f acc01=%.6f\n",
                        bad_r, bad_c,
                        (double)out[(size_t)bad_r * cols + bad_c],
                        (double)((_Float16)(((double)acc[(size_t)bad_r * cols + 1] +
                                            3.0 * (double)acc[(size_t)bad_r * cols + 0]) *
                                           (double)gate_ref[(size_t)bad_r * cols + 1])),
                        (double)ref[(size_t)bad_r * cols + bad_c],
                        (double)acc[(size_t)bad_r * cols + 0]);
            EXPECT(rk_workspace_destroy(ws2, &diag) == 0);
        }

        rk_workspace *ws3 = NULL;
        unsigned legacy_tasks = 0, legacy_submits = 0;
        EXPECT(rk_workspace_create(0, live_plan, constants, 1, &ws3, &diag) == 0);
        EXPECT(ws3 != NULL);
        if (!ws3) return;

        /* --- ABI write boundary ---------------------------------------------
         * The pre-extension caller received an 88-byte rk_run_report and was
         * never required to initialise struct_size, so the legacy entry must
         * write exactly that struct and not one byte more.  A tail sentinel
         * proves the boundary for both the running and the rejected pre-flight
         * path. */
        {
            struct { rk_run_report report; unsigned char tail[72]; } legacy;
            unsigned size = (unsigned)sizeof(rk_run_report);
            EXPECT(size == 88u);
            EXPECT(sizeof(legacy) == size + 72u);

            memset(&legacy, 0x5a, sizeof(legacy));    /* struct_size garbage */
            EXPECT(rk_workspace_run(ws3, inputs, 1, outputs, 1, &legacy.report,
                                    &diag) == 0);
            {
                int tail_intact = 1;
                for (unsigned i = 0; i < 72; i++)
                    if (legacy.tail[i] != 0x5a) tail_intact = 0;
                EXPECT(tail_intact);
                EXPECT(legacy.report.struct_size == size);
                EXPECT(legacy.report.task_count >= 1u);
                legacy_tasks = legacy.report.task_count;
                legacy_submits = legacy.report.submit_count;
            }
            /* A zeroed legacy report (ctypes/userspace idiom) is just as safe. */
            memset(&legacy, 0x00, sizeof(legacy));
            EXPECT(rk_workspace_run(ws3, inputs, 1, outputs, 1, &legacy.report,
                                    &diag) == 0);
            {
                int tail_intact = 1;
                for (unsigned i = 0; i < 72; i++)
                    if (legacy.tail[i] != 0x00) tail_intact = 0;
                EXPECT(tail_intact);
            }
            /* The rejected pre-flight path (no workspace) must not overrun
             * either, even before any plan work happens. */
            memset(&legacy, 0xa5, sizeof(legacy));
            EXPECT(rk_workspace_run(NULL, inputs, 1, outputs, 1, &legacy.report,
                                    &diag) == RK_E_INVALID);
            {
                int tail_intact = 1;
                for (unsigned i = 0; i < 72; i++)
                    if (legacy.tail[i] != 0xa5) tail_intact = 0;
                EXPECT(tail_intact);
            }
        }

        /* --- explicit attribution entry -------------------------------------
         * The extended entry refuses an unusable declaration instead of
         * guessing a size, and fills the whole struct when the caller states it
         * correctly. */
        {
            rk_run_attribution env;
            const uint32_t size = (uint32_t)sizeof(env);
            EXPECT(size == RK_RUN_ATTRIBUTION_V3_SIZE);
            EXPECT(RK_RUN_ATTRIBUTION_V1_SIZE < size);
            EXPECT(RK_RUN_ATTRIBUTION_V2_SIZE < size);
            memset(&env, 0, sizeof(env));
            env.struct_size = 0;
            EXPECT(rk_workspace_run_ex(ws3, inputs, 1, outputs, 1, &env, &diag) ==
                   RK_E_SCHEMA);
            env.struct_size = (uint32_t)sizeof(rk_run_report); /* legacy size */
            EXPECT(rk_workspace_run_ex(ws3, inputs, 1, outputs, 1, &env, &diag) ==
                   RK_E_SCHEMA);
            env.struct_size = size + 8u;                        /* too large */
            EXPECT(rk_workspace_run_ex(ws3, inputs, 1, outputs, 1, &env, &diag) ==
                   RK_E_SCHEMA);
            env.struct_size = size;
            EXPECT(rk_workspace_run_ex(ws3, inputs, 1, outputs, 1, &env, &diag) ==
                   0);
            EXPECT(env.abi_version == RK_ABI_VERSION);
            EXPECT(env.struct_size == size);
            /* The attributed entry reports the same device work as the legacy
             * one; only the envelope detail differs. */
            EXPECT(env.task_count == legacy_tasks);
            EXPECT(env.submit_count == legacy_submits);
            EXPECT(env.submit_count >= 1u);
            EXPECT(env.total_us >= 0.0 && env.clock_us >= 0.0);
            /* The attribution intervals are disjoint parts of the whole call. */
            {
                double attributed = env.validate_us + env.pack_us + env.emit_us +
                                    env.command_us + env.submit_alloc_us +
                                    env.submit_wait_us + env.output_sync_us +
                                    env.download_us + env.guard_us +
                                    env.publish_us + env.alloc_us;
                EXPECT(attributed <= env.total_us + 1.0);
            }
            EXPECT(isfinite(env.total_us));
        }
        EXPECT(rk_workspace_destroy(ws3, &diag) == 0);

        /* --- a refused caller must not release another caller's lock ---------
         * The busy flag is the single-run gate.  A caller that is refused because
         * the workspace is running must leave that ownership untouched, otherwise
         * a third caller (or destroy) can enter a workspace that is still running.
         * The flag is set directly here to model a concurrent owner without
         * needing a second thread. */
        {
            rk_workspace *bws = NULL;
            EXPECT(rk_workspace_create(0, live_plan, constants, 1, &bws, &diag) == 0);
            EXPECT(bws != NULL);
            if (bws) {
                rk_run_report lreport;
                rk_run_attribution att;
                rk_group_report greport;
                memset(&att, 0, sizeof(att));
                att.struct_size = sizeof(att);
                memset(&greport, 0, sizeof(greport));
                greport.struct_size = sizeof(greport);
                atomic_store_explicit(&bws->busy, 1, memory_order_release);
                for (unsigned i = 0; i < 3; i++) {
                    EXPECT(rk_workspace_run(bws, inputs, 1, outputs, 1, &lreport,
                                            &diag) == RK_E_BUSY);
                    EXPECT(atomic_load_explicit(&bws->busy,
                                                memory_order_acquire) == 1);
                    EXPECT(rk_workspace_run_ex(bws, inputs, 1, outputs, 1, &att,
                                               &diag) == RK_E_BUSY);
                    EXPECT(atomic_load_explicit(&bws->busy,
                                                memory_order_acquire) == 1);
                    /* The group entry needs repeat-major bindings; two repeats
                     * of (input, output) are enough to reach the gate. */
                    {
                        rk_binding pair[4] = {inputs[0], outputs[0],
                                              inputs[0], outputs[0]};
                        EXPECT(rk_workspace_run_group(bws, pair, 2, &greport,
                                                      &diag) == RK_E_BUSY);
                    }
                    EXPECT(atomic_load_explicit(&bws->busy,
                                                memory_order_acquire) == 1);
                    EXPECT(rk_workspace_destroy(bws, &diag) == RK_E_BUSY);
                    EXPECT(atomic_load_explicit(&bws->busy,
                                                memory_order_acquire) == 1);
                }
                /* Releasing the owner restores normal service, and a completed
                 * run leaves the flag at 0 (it does not stay held). */
                atomic_store_explicit(&bws->busy, 0, memory_order_release);
                EXPECT(rk_workspace_run_ex(bws, inputs, 1, outputs, 1, &att,
                                           &diag) == 0);
                EXPECT(atomic_load_explicit(&bws->busy,
                                            memory_order_acquire) == 0);
                EXPECT(rk_workspace_destroy(bws, &diag) == 0);
            }
        }

        /* --- group staging must respect the plan Host budget -----------------
         * Growing the repeat arena is Host memory: it is capped by the plan's
         * declared Host budget (and by its declared requirement when no budget
         * is stated), rejected before any allocation, and a rejected group
         * publishes nothing and leaves the workspace usable. */
        {
            rk_plan_desc tight = desc;
            tight.host_budget_bytes = tight.claimed_host_bytes;
            rk_plan *tight_plan = NULL;
            EXPECT(rk_plan_create(&tight, &tight_plan, &diag) == 0);
            EXPECT(tight_plan != NULL);
            if (tight_plan) {
                rk_workspace *gws = NULL;
                EXPECT(rk_workspace_create(0, tight_plan, constants, 1, &gws,
                                           &diag) == 0);
                EXPECT(gws != NULL);
                if (gws) {
                    /* This plan has one input and one output, so one repeat is
                     * (input, output) and the group layout is repeat-major. */
                    const uint32_t stride = (uint32_t)(rows * cols * 2u);
                    const uint64_t budget = tight.claimed_host_bytes;
                    const uint64_t fixed =
                        rk_workspace_host_bytes(gws, 0);   /* everything but arena */
                    /* Boundary derived from the runtime's own accounting: the
                     * largest capacity whose *total* Host footprint fits, and one
                     * more.  The rejected case must be one where the staging block
                     * alone would still fit -- otherwise the test would pass under
                     * the old staging-only rule and prove nothing. */
                    const uint32_t fit = (uint32_t)((budget - fixed) / stride);
                    const uint32_t over = fit + 1u;
                    EXPECT(fit >= 1u);
                    EXPECT(fixed + (uint64_t)fit * stride <= budget);
                    EXPECT(fixed + (uint64_t)over * stride > budget);
                    EXPECT((uint64_t)over * stride <= budget);   /* staging fits */
                    rk_binding *group = calloc((size_t)over * 2u, sizeof(*group));
                    EXPECT(group != NULL);
                    if (group) {
                        for (uint32_t r = 0; r < over; r++) {
                            group[2u * r + 0u] = inputs[0];
                            group[2u * r + 1u] = outputs[0];
                        }
                        rk_group_report greport;
                        int rc;
                        memset(&greport, 0, sizeof(greport));
                        greport.struct_size = sizeof(greport);
                        memset(out, 0x7e, (size_t)rows * cols * 2);
                        rc = rk_workspace_run_group(gws, group, over, &greport,
                                                    &diag);
                        EXPECT(rc == RK_E_BUDGET);
                        EXPECT(greport.fail_repeat == UINT32_MAX);
                        {
                            int untouched = 1;
                            for (size_t i = 0; i < (size_t)rows * cols * 2; i++)
                                if (((unsigned char *)out)[i] != 0x7e)
                                    untouched = 0;
                            EXPECT(untouched);
                        }
                        /* An unusable declaration is refused before any run. */
                        greport.struct_size = 0;
                        EXPECT(rk_workspace_run_group(gws, group, fit, &greport,
                                                      &diag) == RK_E_SCHEMA);
                        greport.struct_size =
                            (uint32_t)sizeof(rk_run_report);   /* too small */
                        EXPECT(rk_workspace_run_group(gws, group, fit, &greport,
                                                      &diag) == RK_E_SCHEMA);
                        /* A repeat set inside the total footprint still works,
                         * and the workspace is not poisoned by the rejections. */
                        greport.struct_size = (uint32_t)sizeof(greport);
                        memset(out, 0, (size_t)rows * cols * 2);
                        EXPECT(rk_workspace_run_group(gws, group, fit, &greport,
                                                      &diag) == 0);
                        EXPECT(greport.repeat_count == fit);
                        EXPECT(greport.repeat_completed == fit);
                        EXPECT(greport.fail_repeat == UINT32_MAX);
                        EXPECT(greport.submit_count == fit);   /* unchanged */
                        {
                            int published = 0;
                            for (size_t i = 0; i < (size_t)rows * cols * 2; i++)
                                if (((unsigned char *)out)[i]) published = 1;
                            EXPECT(published);
                        }
                        free(group);
                    }
                    EXPECT(rk_workspace_destroy(gws, &diag) == 0);
                }
                rk_plan_destroy(tight_plan);
            }
        }
        rk_plan_destroy(live_plan);
        EXPECT(rk_mock_bo_count() == 0);   /* table drained with the workspace */
        free(acc); free(out); free(identity); free(gate_ref); free(ref);
    }
}




/* Genuine multi-output plan: two independent MM results published into two separate
 * output bindings.  Host staging must account for both outputs at once. */
static void multi_output_budget_case(void) {
    enum { M = 32, K = 256, N = 48 }; /* one primitive K tile */
    co_tensor x_t, w_t, y_t;
    EXPECT(co_tensor_matrix(&x_t, M, K, CO_F16) == 0);
    EXPECT(co_tensor_weight(&w_t, N, K) == 0);
    EXPECT(co_tensor_matrix(&y_t, M, N, CO_F16) == 0);
    const uint64_t x_logical = (uint64_t)M * K * 2u;
    const uint64_t w_logical = (uint64_t)N * K * 2u;
    const uint64_t y_logical = (uint64_t)M * N * 2u;

    rk_value_desc values[4];
    memset(values, 0, sizeof(values));
    const uint32_t vshape[4][2] = {{M, K}, {N, K}, {M, N}, {M, N}};
    const uint32_t vrole[4] = {RK_ROLE_INPUT, RK_ROLE_CONSTANT, RK_ROLE_OUTPUT,
                               RK_ROLE_OUTPUT};
    const uint64_t vbytes[4] = {x_logical, w_logical, y_logical, y_logical};
    const char *vname[4] = {"A", "W", "Y0", "Y1"};
    for (uint32_t i = 0; i < 4; i++) {
        values[i].abi_version = RK_ABI_VERSION;
        values[i].struct_size = sizeof(rk_value_desc);
        values[i].id = i;
        values[i].role = vrole[i];
        values[i].dtype = RK_DTYPE_F16;
        values[i].rank = 2;
        values[i].shape[0] = vshape[i][0];
        values[i].shape[1] = vshape[i][1];
        values[i].nbytes = vbytes[i];
        values[i].name = vname[i];
        values[i].name_len = (uint32_t)strlen(vname[i]);
    }

    const co_tensor *geometry[4] = {&x_t, &w_t, &y_t, &y_t};
    const uint32_t layouts[4] = {RK_LAYOUT_CUBE_F16, RK_LAYOUT_WEIGHT_F16,
                                 RK_LAYOUT_CUBE_F16, RK_LAYOUT_CUBE_F16};
    const uint32_t broles[4] = {RK_ROLE_INPUT_SLICE, RK_ROLE_WEIGHT,
                                RK_ROLE_OUTPUT, RK_ROLE_OUTPUT};
    rk_buffer_desc buffers[4];
    rk_view_desc views[4];
    memset(buffers, 0, sizeof(buffers));
    memset(views, 0, sizeof(views));
    for (uint32_t i = 0; i < 4; i++) {
        buffers[i].abi_version = RK_ABI_VERSION;
        buffers[i].struct_size = sizeof(rk_buffer_desc);
        buffers[i].id = i;
        buffers[i].role = broles[i];
        buffers[i].slot_id = i;
        buffers[i].alignment = 16;
        buffers[i].layout_id = layouts[i];
        buffers[i].capacity = geometry[i]->bytes;
        buffers[i].lifetime_last = 5;

        views[i].abi_version = RK_ABI_VERSION;
        views[i].struct_size = sizeof(rk_view_desc);
        views[i].id = i;
        views[i].buffer_id = i;
        views[i].tensor.abi_version = RK_ABI_VERSION;
        views[i].tensor.struct_size = sizeof(rk_tensor_desc);
        views[i].tensor.dtype = RK_DTYPE_F16;
        views[i].tensor.layout = layouts[i];
        views[i].tensor.rows = (uint32_t)geometry[i]->rows;
        views[i].tensor.cols = (uint32_t)geometry[i]->cols;
        views[i].tensor.physical_rows = (uint32_t)geometry[i]->physical_rows;
        views[i].tensor.physical_cols = (uint32_t)geometry[i]->physical_cols;
        views[i].tensor.atom = (uint32_t)geometry[i]->atom;
        views[i].tensor.bytes = geometry[i]->bytes;
        views[i].valid_region[1] = views[i].tensor.rows;
        views[i].valid_region[3] = views[i].tensor.cols;
    }

    uint32_t mm_in[] = {0, 1};
    uint32_t both_deps[] = {0, 1};
    uint32_t v_y0 = 2, v_y1 = 3;
    uint32_t down0_in[] = {2}, down1_in[] = {3};
    uint32_t d0_deps[] = {2}, d1_deps[] = {4};
    rk_action_desc actions[6];
    memset(actions, 0, sizeof(actions));
    for (uint32_t i = 0; i < 6; i++) {
        actions[i].abi_version = RK_ABI_VERSION;
        actions[i].struct_size = sizeof(rk_action_desc);
        actions[i].id = i;
        snprintf(actions[i].origin, sizeof(actions[i].origin), "multi-out:%u", i);
    }
    actions[0].kind = RK_ACTION_UPLOAD;
    actions[0].source_binding = 0;
    actions[0].source_row_stride = K;
    actions[0].noutputs = 1;
    actions[0].outputs = &actions[0].id; /* replaced below */
    uint32_t v_x = 0, v_w = 1;
    actions[0].outputs = &v_x;
    actions[1].kind = RK_ACTION_UPLOAD;
    actions[1].source_binding = 1;
    actions[1].source_row_stride = K; /* binding is already [N,K] */
    actions[1].noutputs = 1;
    actions[1].outputs = &v_w;
    actions[2].kind = RK_ACTION_NPU_TASK;
    actions[2].template_id = RK_TEMPLATE_MM_F16;
    actions[2].ninputs = 2;
    actions[2].inputs = mm_in;
    actions[2].noutputs = 1;
    actions[2].outputs = &v_y0;
    actions[2].ndeps = 2;
    actions[2].deps = both_deps;
    actions[3].kind = RK_ACTION_DOWNLOAD;
    actions[3].source_binding = 2;
    actions[3].ninputs = 1;
    actions[3].inputs = down0_in;
    actions[3].ndeps = 1;
    actions[3].deps = d0_deps;
    actions[4].kind = RK_ACTION_NPU_TASK;
    actions[4].template_id = RK_TEMPLATE_MM_F16;
    actions[4].ninputs = 2;
    actions[4].inputs = mm_in;
    actions[4].noutputs = 1;
    actions[4].outputs = &v_y1;
    actions[4].ndeps = 2;
    actions[4].deps = both_deps;
    actions[5].kind = RK_ACTION_DOWNLOAD;
    actions[5].source_binding = 3;
    actions[5].ninputs = 1;
    actions[5].inputs = down1_in;
    actions[5].ndeps = 1;
    actions[5].deps = d1_deps;

    uint32_t tasks0[] = {2}, tasks1[] = {4}, submit_dep[] = {0};
    rk_submit_desc submits[2] = {
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_submit_desc),
         .id = 0, .mode = RK_SUBMIT_CHAIN, .core_policy = RK_CORE_SINGLE,
         .ntask_ids = 1, .task_ids = tasks0},
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_submit_desc),
         .id = 1, .mode = RK_SUBMIT_CHAIN, .core_policy = RK_CORE_SINGLE,
         .ntask_ids = 1, .ndeps = 1, .deps = submit_dep, .task_ids = tasks1},
    };
    rk_publish_desc publish[2] = {
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_publish_desc),
         .binding_id = 2, .view_id = 2, .rows = M, .cols = N},
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_publish_desc),
         .binding_id = 3, .view_id = 3, .rows = M, .cols = N},
    };
    const uint64_t output_total = 2u * y_logical;
    const uint64_t staging = output_total > 65536u ? output_total : 65536u;
    /* Runtime-owned Host scratch is part of the declared requirement: one
     * co_program per submit task, the download tile (largest output) and the
     * per-output offset/count bookkeeping. */
    const uint64_t runtime_scratch =
        1u * (uint64_t)sizeof(co_program) + y_logical +
        2u * (uint64_t)(sizeof(size_t) + sizeof(uint32_t));
    const uint64_t host = x_logical + w_logical + output_total + staging +
                          runtime_scratch;
    /* Device request: one slot per buffer, plus the command area sized by the
     * *largest single submit* (each submit here carries one task). */
    const uint64_t device = x_t.bytes + w_t.bytes + 2u * y_t.bytes +
                            4u * (CO_SLACK + CO_GUARD) +
                            1u * CO_MAX_WORDS * 8u;

    rk_plan_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.abi_version = RK_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    desc.schema_version = RK_SCHEMA_VERSION;
    desc.values = values;
    desc.nvalues = 4;
    desc.buffers = buffers;
    desc.nbuffers = 4;
    desc.views = views;
    desc.nviews = 4;
    desc.actions = actions;
    desc.nactions = 6;
    desc.submits = submits;
    desc.nsubmits = 2;
    desc.publish = publish;
    desc.npublish = 2;
    desc.device_budget_bytes = device;
    desc.host_budget_bytes = host + (1u << 20);
    desc.claimed_device_bytes = device;
    desc.claimed_host_bytes = host;
    desc.runtime_abi_version = RK_ABI_VERSION;

    rk_diag diag;
    rk_plan *plan = NULL;
    int rc = rk_plan_create(&desc, &plan, &diag);
    if (rc) fprintf(stderr, "MULTI-OUT DIAG: %d %s\n", diag.code, diag.message);
    EXPECT(rc == 0);
    EXPECT(plan != NULL);
    rk_plan_destroy(plan);
    /* one byte short of staging both outputs at once */
    rk_plan_desc tight = desc;
    tight.claimed_host_bytes = host - 1u;
    plan = NULL;
    EXPECT_REJECT(rk_plan_create(&tight, &plan, &diag), plan, RK_E_BUDGET,
                  "host bytes");
    /* zero claim is not a bypass */
    tight.claimed_host_bytes = 0;
    plan = NULL;
    EXPECT_REJECT(rk_plan_create(&tight, &plan, &diag), plan, RK_E_BUDGET,
                  "must be stated");
    /* Device bytes are recomputed per slot: an obviously short claim is rejected.
     * (The exact boundary is covered by the host side; the device formula also
     * depends on the slot packing, so this case only proves the check is live_bos.) */
    rk_plan_desc tight_dev = desc;
    tight_dev.claimed_device_bytes = device - 1u;
    plan = NULL;
    EXPECT_REJECT(rk_plan_create(&tight_dev, &plan, &diag), plan, RK_E_BUDGET,
                  "device");
    tight_dev.claimed_device_bytes = 1;
    plan = NULL;
    EXPECT_REJECT(rk_plan_create(&tight_dev, &plan, &diag), plan, RK_E_BUDGET,
                  "device");
}



/* Direct BO allocation: the simulated address must land in the real struct field
 * the emitters patch into registers, not only in a side table. */
static void bo_dma_field_case(void) {
    rocket_bo bo;
    memset(&bo, 0xAB, sizeof(bo));
    EXPECT(rocket_bo_alloc32(3, 4096, &bo) == 0);
    EXPECT(bo.dma_address != 0);
    EXPECT(bo.dma_address % 16u == 0);
    EXPECT(bo.dma_address >= 0x01000000u && bo.dma_address < 0xF0000000u);
    EXPECT(bo.dma_address + bo.size <= 0xF0000000u);
    EXPECT((uint32_t)rk_mock_dma_of(bo.handle) == (uint32_t)bo.dma_address);
    EXPECT(rk_mock_bo_dma_address(bo.handle) == bo.dma_address);
    /* a second allocation gets a different, non-overlapping window */
    rocket_bo second;
    EXPECT(rocket_bo_alloc32(3, 1024, &second) == 0);
    EXPECT(second.handle != bo.handle);
    EXPECT(second.dma_address != bo.dma_address);
    EXPECT(bo.dma_address + bo.size <= second.dma_address ||
           second.dma_address + second.size <= bo.dma_address);
    rocket_bo_free(3, &second);
    rocket_bo_free(3, &bo);
    EXPECT(rk_mock_bo_count() == 0);
    EXPECT(rk_mock_live_bos() == 0);
}


/* Named mock scenario A: a zeroed program (no views at all) must be refused by
 * the call/structure validation, before anything is dereferenced. */
static void mock_zero_program_case(void) {
    rk_mock_reset();
    int live_before = rk_mock_live_bos();
    size_t res_before = rk_mock_resident();
    co_program fake;
    memset(&fake, 0, sizeof(fake));
    fake.count = 1;
    fake.nreads = 2;
    co_execute_profile profile;
    EXPECT(co_execute_profiled(0, &fake, 1, CO_CHAIN, NULL, &profile) ==
           ROCKET_E_SHAPE);
    EXPECT(rk_mock_untracked() == 1);
    EXPECT(rk_mock_live_bos() == live_before);
    EXPECT(rk_mock_resident() == res_before);
    /* malformed calls are refused too */
    EXPECT(co_execute_profiled(0, NULL, 1, CO_CHAIN, NULL, &profile) ==
           ROCKET_E_SHAPE);
    EXPECT(co_execute_profiled(0, &fake, 0, CO_CHAIN, NULL, &profile) ==
           ROCKET_E_SHAPE);
    EXPECT(co_execute_profiled(0, &fake, 1, (enum co_mode)99, NULL, &profile) ==
           ROCKET_E_SHAPE);
    rk_mock_reset();
    EXPECT(rk_mock_untracked() == 0);
}

/* Named mock scenario B: structurally valid but untracked (tags cleared) must
 * increment the untracked counter and must not disturb live BO accounting. */
static void mock_untracked_program_case(void) {
    rk_mock_reset();
    rocket_bo bo;
    EXPECT(rocket_bo_alloc32(3, 4096, &bo) == 0);
    int live_before = rk_mock_live_bos();
    size_t res_before = rk_mock_resident();
    co_program fake;
    memset(&fake, 0, sizeof(fake));
    fake.count = 1;
    fake.nreads = 2;
    /* a *valid* looking view pointing at a live BO, but never emitted */
    fake.output.buffer = NULL;
    fake.reads[0].buffer = NULL;
    rk_mock_reset();                 /* clears tags only */
    EXPECT(rk_mock_live_bos() == live_before);   /* BOs survive the reset */
    EXPECT(rk_mock_resident() == res_before);
    co_execute_profile profile;
    EXPECT(co_execute_profiled(0, &fake, 1, CO_CHAIN, NULL, &profile) ==
           ROCKET_E_SHAPE);
    EXPECT(rk_mock_untracked() == 1);
    EXPECT(rk_mock_live_bos() == live_before);
    EXPECT(rk_mock_resident() == res_before);
    rocket_bo_free(3, &bo);
    rk_mock_reset();
}

/* Named mock scenario C: a full tag table must make execution refuse instead of
 * dispatching an untracked program. */
static void mock_tag_capacity_case(void) {
    rk_mock_reset();
    EXPECT(rk_mock_tag_capacity() >= 64u);
    EXPECT(rk_mock_tag_overflow() == 0);
    rk_mock_force_tag_overflow(1);
    EXPECT(rk_mock_tag_overflow() == 1);
    co_program fake;
    memset(&fake, 0, sizeof(fake));
    fake.count = 1;
    fake.nreads = 2;
    co_execute_profile profile;
    EXPECT(co_execute_profiled(0, &fake, 1, CO_CHAIN, NULL, &profile) ==
           ROCKET_E_SHAPE);
    rk_mock_force_tag_overflow(0);
    rk_mock_reset();
    EXPECT(rk_mock_tag_overflow() == 0);
    EXPECT(rk_mock_untracked() == 0);
}


/* A legal plan with 130 runtime uploads in one consecutive run.  It is larger
 * than the coalescing scratch array; the runtime must fall back to the original
 * per-slice path rather than rejecting a plan the verifier already accepted. */
static void many_upload_fallback_case(void) {
    enum { NINPUT = 130, NACTION = NINPUT + 2, NBUFFER = NINPUT + 1,
           NVIEW = NINPUT + 1 };
    rk_value_desc *values = calloc(NINPUT + 1, sizeof(*values));
    rk_buffer_desc *buffers = calloc(NBUFFER, sizeof(*buffers));
    rk_view_desc *views = calloc(NVIEW, sizeof(*views));
    rk_action_desc *actions = calloc(NACTION, sizeof(*actions));
    uint32_t *task_ids = malloc(sizeof(uint32_t));
    co_tensor cube, weight, out_t;
    rk_publish_desc *publish = calloc(1, sizeof(*publish));
    char (*names)[8] = calloc(NINPUT + 1, sizeof(*names));
    _Float16 *x = calloc((size_t)NINPUT * 256u, sizeof(*x));
    _Float16 *y = calloc(256u, sizeof(*y));
    rk_binding *inputs = calloc(NINPUT, sizeof(*inputs));
    rk_binding *outputs = calloc(1, sizeof(*outputs));
    rk_diag diag;
    rk_plan *plan = NULL;
    rk_workspace *ws = NULL;
    int ok = values && buffers && views && actions && task_ids && publish &&
             names && x && y && inputs && outputs &&
             co_tensor_matrix(&cube, 16, 16, CO_F16) == 0 &&
             co_tensor_weight(&weight, 16, 16) == 0 &&
             co_tensor_matrix(&out_t, 16, 16, CO_F16) == 0;
    EXPECT(ok);
    if (!ok) goto done;
    memset(&diag, 0, sizeof(diag));
    for (uint32_t i = 0; i < NINPUT; i++) {
        values[i].abi_version = RK_ABI_VERSION;
        values[i].struct_size = sizeof(rk_value_desc);
        values[i].id = i;
        values[i].role = RK_ROLE_INPUT;
        values[i].dtype = RK_DTYPE_F16;
        values[i].rank = 2;
        values[i].shape[0] = 16; values[i].shape[1] = 16;
        values[i].nbytes = 512;
        snprintf(names[i], sizeof(names[i]), "A%u", i);
        values[i].name = names[i];
        values[i].name_len = (uint32_t)strlen(names[i]);
        const co_tensor *t = (i == 1) ? &weight : &cube;
        buffers[i].abi_version = RK_ABI_VERSION;
        buffers[i].struct_size = sizeof(rk_buffer_desc);
        buffers[i].id = i;
        buffers[i].role = (i == 1) ? RK_ROLE_WEIGHT : RK_ROLE_INPUT_SLICE;
        buffers[i].slot_id = i;
        buffers[i].alignment = 16;
        buffers[i].layout_id = (i == 1) ? RK_LAYOUT_WEIGHT_F16 : RK_LAYOUT_CUBE_F16;
        buffers[i].capacity = t->bytes;
        buffers[i].lifetime_first = 0;
        buffers[i].lifetime_last = NACTION - 1;
        buffers[i].padding_state = 0;
        views[i].abi_version = RK_ABI_VERSION;
        views[i].struct_size = sizeof(rk_view_desc);
        views[i].id = i;
        views[i].buffer_id = i;
        views[i].tensor.abi_version = RK_ABI_VERSION;
        views[i].tensor.struct_size = sizeof(rk_tensor_desc);
        views[i].tensor.dtype = RK_DTYPE_F16;
        views[i].tensor.layout = (i == 1) ? RK_LAYOUT_WEIGHT_F16 : RK_LAYOUT_CUBE_F16;
        views[i].tensor.rows = 16; views[i].tensor.cols = 16;
        views[i].tensor.physical_rows = t->physical_rows;
        views[i].tensor.physical_cols = t->physical_cols;
        views[i].tensor.atom = (i == 1) ? 32u : 8u;
        views[i].tensor.bytes = t->bytes;
        views[i].valid_region[1] = 16; views[i].valid_region[3] = 16;
        actions[i].abi_version = RK_ABI_VERSION;
        actions[i].struct_size = sizeof(rk_action_desc);
        actions[i].id = i;
        actions[i].kind = RK_ACTION_UPLOAD;
        actions[i].source_binding = i;
        actions[i].source_row_stride = 16;
        actions[i].flags = (i == 1) ? RK_ACTION_FLAG_TRANSPOSE_SOURCE : 0;
        actions[i].noutputs = 1;
        actions[i].outputs = &views[i].id;
        snprintf(actions[i].origin, sizeof(actions[i].origin), "many_upload:%u", i);
    }
    values[NINPUT].abi_version = RK_ABI_VERSION;
    values[NINPUT].struct_size = sizeof(rk_value_desc);
    values[NINPUT].id = NINPUT;
    values[NINPUT].role = RK_ROLE_OUTPUT;
    values[NINPUT].dtype = RK_DTYPE_F16;
    values[NINPUT].rank = 2;
    values[NINPUT].shape[0] = 16; values[NINPUT].shape[1] = 16;
    values[NINPUT].nbytes = 512;
    snprintf(names[NINPUT], sizeof(names[NINPUT]), "Y");
    values[NINPUT].name = names[NINPUT];
    values[NINPUT].name_len = (uint32_t)strlen(names[NINPUT]);

    buffers[NINPUT].abi_version = RK_ABI_VERSION;
    buffers[NINPUT].struct_size = sizeof(rk_buffer_desc);
    buffers[NINPUT].id = NINPUT;
    buffers[NINPUT].role = RK_ROLE_OUTPUT;
    buffers[NINPUT].slot_id = NINPUT;
    buffers[NINPUT].alignment = 16;
    buffers[NINPUT].layout_id = RK_LAYOUT_CUBE_F16;
    buffers[NINPUT].capacity = out_t.bytes;
    buffers[NINPUT].lifetime_first = 0;
    buffers[NINPUT].lifetime_last = NACTION - 1;
    buffers[NINPUT].padding_state = 0;
    views[NINPUT].abi_version = RK_ABI_VERSION;
    views[NINPUT].struct_size = sizeof(rk_view_desc);
    views[NINPUT].id = NINPUT;
    views[NINPUT].buffer_id = NINPUT;
    views[NINPUT].tensor.abi_version = RK_ABI_VERSION;
    views[NINPUT].tensor.struct_size = sizeof(rk_tensor_desc);
    views[NINPUT].tensor.dtype = RK_DTYPE_F16;
    views[NINPUT].tensor.layout = RK_LAYOUT_CUBE_F16;
    views[NINPUT].tensor.rows = 16; views[NINPUT].tensor.cols = 16;
    views[NINPUT].tensor.physical_rows = out_t.physical_rows;
    views[NINPUT].tensor.physical_cols = out_t.physical_cols;
    views[NINPUT].tensor.atom = 8;
    views[NINPUT].tensor.bytes = out_t.bytes;
    views[NINPUT].valid_region[1] = 16; views[NINPUT].valid_region[3] = 16;

    uint32_t mm_in[2] = {0, 1};
    uint32_t mm_dep[2] = {0, 1};
    uint32_t mm_out = NINPUT;
    rk_action_desc *mm = &actions[NINPUT];
    mm->abi_version = RK_ABI_VERSION;
    mm->struct_size = sizeof(rk_action_desc);
    mm->id = NINPUT;
    mm->kind = RK_ACTION_NPU_TASK;
    mm->template_id = RK_TEMPLATE_MM_F16;
    mm->ninputs = 2; mm->inputs = mm_in;
    mm->noutputs = 1; mm->outputs = &mm_out;
    mm->ndeps = 2; mm->deps = mm_dep;
    snprintf(mm->origin, sizeof(mm->origin), "many_upload:mm");
    rk_action_desc *down = &actions[NINPUT + 1];
    down->abi_version = RK_ABI_VERSION;
    down->struct_size = sizeof(rk_action_desc);
    down->id = NINPUT + 1;
    down->kind = RK_ACTION_DOWNLOAD;
    down->source_binding = NINPUT;
    down->ninputs = 1; down->inputs = &mm_out;
    down->ndeps = 1; down->deps = &mm->id;
    snprintf(down->origin, sizeof(down->origin), "many_upload:down");

    task_ids[0] = NINPUT;
    rk_submit_desc submit = {
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_submit_desc),
        .id = 0, .mode = RK_SUBMIT_CHAIN, .core_policy = RK_CORE_SINGLE,
        .ntask_ids = 1, .task_ids = task_ids};
    publish[0].abi_version = RK_ABI_VERSION;
    publish[0].struct_size = sizeof(rk_publish_desc);
    publish[0].binding_id = NINPUT;
    publish[0].view_id = NINPUT;
    publish[0].rows = 16; publish[0].cols = 16;

    rk_plan_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.abi_version = RK_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    desc.schema_version = RK_SCHEMA_VERSION;
    desc.values = values; desc.nvalues = NINPUT + 1;
    desc.buffers = buffers; desc.nbuffers = NBUFFER;
    desc.views = views; desc.nviews = NVIEW;
    desc.actions = actions; desc.nactions = NACTION;
    desc.submits = &submit; desc.nsubmits = 1;
    desc.publish = publish; desc.npublish = 1;
    desc.device_budget_bytes = 64u << 20;
    desc.host_budget_bytes = 64u << 20;
    desc.claimed_device_bytes = 64u << 20;
    desc.claimed_host_bytes = 64u << 20;
    desc.runtime_abi_version = RK_ABI_VERSION;
    int plan_rc = rk_plan_create(&desc, &plan, &diag);
    if (plan_rc) fprintf(stderr, "many_upload plan_create rc=%d: %s\n",
                         plan_rc, diag.message);
    EXPECT(plan_rc == 0);
    if (!plan) goto done;
    EXPECT(rk_workspace_create(3, plan, NULL, 0, &ws, &diag) == 0);
    if (!ws) goto done;
    for (uint32_t i = 0; i < NINPUT; i++) {
        for (uint32_t q = 0; q < 256; q++) x[(size_t)i * 256u + q] =
            value((unsigned)(i * 13u + q), 1);
        inputs[i].abi_version = RK_ABI_VERSION;
        inputs[i].struct_size = sizeof(rk_binding);
        inputs[i].id = i;
        inputs[i].dtype = RK_DTYPE_F16;
        inputs[i].rank = 2;
        inputs[i].shape[0] = 16; inputs[i].shape[1] = 16;
        inputs[i].data = x + (size_t)i * 256u;
        inputs[i].nbytes = 512;
    }
    outputs[0].abi_version = RK_ABI_VERSION;
    outputs[0].struct_size = sizeof(rk_binding);
    outputs[0].id = NINPUT;
    outputs[0].dtype = RK_DTYPE_F16;
    outputs[0].rank = 2;
    outputs[0].shape[0] = 16; outputs[0].shape[1] = 16;
    outputs[0].data = y;
    outputs[0].nbytes = 512;
    rk_run_report report;
    EXPECT(rk_workspace_run(ws, inputs, NINPUT, outputs, 1, &report, &diag) == 0);
    EXPECT(report.upload_calls == NINPUT);
    EXPECT(report.task_count == 1u);
done:
    if (ws) rk_workspace_destroy(ws, &diag);
    if (plan) rk_plan_destroy(plan);
    free(publish); free(inputs); free(outputs); free(x); free(y); free(names);
    free(actions); free(views); free(buffers); free(values); free(task_ids);
}


/* Budget closure: a small four-submit plan lets us set Host and device budgets
 * at exact boundaries without changing the fixture's other limits. */
static void multi_submit_budget_case(void) {
    enum { M = 32, K = 32, N = 32 };
    co_tensor x_t, w_t, y_t;
    EXPECT(co_tensor_input(&x_t, M, K) == 0);
    EXPECT(co_tensor_weight(&w_t, N, K) == 0);
    EXPECT(co_tensor_matrix(&y_t, M, N, CO_F16) == 0);

    rk_value_desc values[3];
    memset(values, 0, sizeof(values));
    const uint32_t shape[3][2] = {{M, K}, {N, K}, {M, N}};
    const uint32_t roles[3] = {RK_ROLE_INPUT, RK_ROLE_CONSTANT,
                               RK_ROLE_OUTPUT};
    const uint64_t nbytes[3] = {x_t.bytes, w_t.bytes, y_t.bytes};
    const char *names[3] = {"A", "B", "Y"};
    for (uint32_t i = 0; i < 3; i++) {
        values[i].abi_version = RK_ABI_VERSION;
        values[i].struct_size = sizeof(rk_value_desc);
        values[i].id = i;
        values[i].role = roles[i];
        values[i].dtype = RK_DTYPE_F16;
        values[i].rank = 2;
        values[i].shape[0] = shape[i][0];
        values[i].shape[1] = shape[i][1];
        values[i].nbytes = nbytes[i];
        values[i].name = names[i];
        values[i].name_len = (uint32_t)strlen(names[i]);
    }

    rk_buffer_desc buffers[3];
    rk_view_desc views[3];
    memset(buffers, 0, sizeof(buffers));
    memset(views, 0, sizeof(views));
    const uint32_t layouts[3] = {RK_LAYOUT_CUBE_F16, RK_LAYOUT_WEIGHT_F16,
                                 RK_LAYOUT_CUBE_F16};
    const uint32_t broles[3] = {RK_ROLE_INPUT_SLICE, RK_ROLE_WEIGHT,
                                RK_ROLE_OUTPUT};
    const co_tensor *geom[3] = {&x_t, &w_t, &y_t};
    for (uint32_t i = 0; i < 3; i++) {
        buffers[i].abi_version = RK_ABI_VERSION;
        buffers[i].struct_size = sizeof(rk_buffer_desc);
        buffers[i].id = i;
        buffers[i].role = broles[i];
        buffers[i].slot_id = i;
        buffers[i].alignment = 16;
        buffers[i].layout_id = layouts[i];
        buffers[i].capacity = geom[i]->bytes;
        buffers[i].lifetime_last = 6;
        views[i].abi_version = RK_ABI_VERSION;
        views[i].struct_size = sizeof(rk_view_desc);
        views[i].id = i;
        views[i].buffer_id = i;
        views[i].tensor.abi_version = RK_ABI_VERSION;
        views[i].tensor.struct_size = sizeof(rk_tensor_desc);
        views[i].tensor.dtype = RK_DTYPE_F16;
        views[i].tensor.layout = layouts[i];
        views[i].tensor.rows = geom[i]->rows;
        views[i].tensor.cols = geom[i]->cols;
        views[i].tensor.physical_rows = geom[i]->physical_rows;
        views[i].tensor.physical_cols = geom[i]->physical_cols;
        views[i].tensor.atom = geom[i]->atom;
        views[i].tensor.bytes = geom[i]->bytes;
        views[i].valid_region[1] = views[i].tensor.rows;
        views[i].valid_region[3] = views[i].tensor.cols;
    }

    uint32_t v_x = 0, v_w = 1, v_y = 2;
    uint32_t task_in[] = {v_x, v_w};
    uint32_t task_deps[][4] = {{0, 1}, {0, 1, 2}, {0, 1, 3}, {0, 1, 4}};
    uint32_t down_in[] = {v_y};
    uint32_t down_deps[] = {5};
    rk_action_desc actions[7];
    memset(actions, 0, sizeof(actions));
    for (uint32_t i = 0; i < 7; i++) {
        actions[i].abi_version = RK_ABI_VERSION;
        actions[i].struct_size = sizeof(rk_action_desc);
        actions[i].id = i;
        snprintf(actions[i].origin, sizeof(actions[i].origin), "budget:%u", i);
    }
    actions[0].kind = RK_ACTION_UPLOAD;
    actions[0].source_binding = 0;
    actions[0].source_row_stride = K;
    actions[0].noutputs = 1;
    actions[0].outputs = &v_x;
    actions[1].kind = RK_ACTION_UPLOAD;
    actions[1].source_binding = 1;
    actions[1].source_row_stride = K;
    actions[1].flags = RK_ACTION_FLAG_CACHE_CONSTANT;
    actions[1].noutputs = 1;
    actions[1].outputs = &v_w;
    for (uint32_t t = 0; t < 4; t++) {
        actions[2 + t].kind = RK_ACTION_NPU_TASK;
        actions[2 + t].template_id = RK_TEMPLATE_MM_F16;
        actions[2 + t].ninputs = 2;
        actions[2 + t].inputs = task_in;
        actions[2 + t].noutputs = 1;
        actions[2 + t].outputs = &v_y;
        actions[2 + t].ndeps = 2 + t;
        actions[2 + t].deps = task_deps[t];
    }
    actions[6].kind = RK_ACTION_DOWNLOAD;
    actions[6].source_binding = 2;
    actions[6].ninputs = 1;
    actions[6].inputs = down_in;
    actions[6].ndeps = 1;
    actions[6].deps = down_deps;

    uint32_t task0[] = {2}, task1[] = {3}, task2[] = {4}, task3[] = {5};
    uint32_t submit_dep0[] = {0}, submit_dep1[] = {1}, submit_dep2[] = {2};
    rk_submit_desc submits[4] = {
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_submit_desc),
         .id = 0, .mode = RK_SUBMIT_CHAIN, .core_policy = RK_CORE_SINGLE,
         .ntask_ids = 1, .task_ids = task0},
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_submit_desc),
         .id = 1, .mode = RK_SUBMIT_CHAIN, .core_policy = RK_CORE_SINGLE,
         .ntask_ids = 1, .ndeps = 1, .deps = submit_dep0, .task_ids = task1},
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_submit_desc),
         .id = 2, .mode = RK_SUBMIT_CHAIN, .core_policy = RK_CORE_SINGLE,
         .ntask_ids = 1, .ndeps = 1, .deps = submit_dep1, .task_ids = task2},
        {.abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_submit_desc),
         .id = 3, .mode = RK_SUBMIT_CHAIN, .core_policy = RK_CORE_SINGLE,
         .ntask_ids = 1, .ndeps = 1, .deps = submit_dep2, .task_ids = task3},
    };
    rk_publish_desc publish = {
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_publish_desc),
        .binding_id = 2, .view_id = 2, .rows = M, .cols = N};

    const uint64_t reserved = x_t.bytes + w_t.bytes + y_t.bytes +
                              3u * (CO_SLACK + CO_GUARD);
    const uint64_t base_device = reserved + CO_MAX_WORDS * 8u;
    const uint64_t host_req = x_t.bytes + w_t.bytes + y_t.bytes +
                              65536u + sizeof(co_program) + y_t.bytes +
                              (sizeof(size_t) + sizeof(uint32_t));

    rk_diag diag;
    memset(&diag, 0, sizeof(diag));
    rk_plan_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.abi_version = RK_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    desc.schema_version = RK_SCHEMA_VERSION;
    desc.values = values; desc.nvalues = 3;
    desc.buffers = buffers; desc.nbuffers = 3;
    desc.views = views; desc.nviews = 3;
    desc.actions = actions; desc.nactions = 7;
    desc.submits = submits; desc.nsubmits = 4;
    desc.publish = &publish; desc.npublish = 1;
    desc.runtime_abi_version = RK_ABI_VERSION;

    _Float16 *xbuf = calloc(x_t.bytes / 2u, 2);
    _Float16 *wbuf = calloc(w_t.bytes / 2u, 2);
    _Float16 *yctl = calloc(y_t.bytes / 2u, 2);
    _Float16 *ycache = calloc(y_t.bytes / 2u, 2);
    EXPECT(xbuf && wbuf && yctl && ycache);
    if (!xbuf || !wbuf || !yctl || !ycache) goto done;
    for (uint32_t i = 0; i < x_t.bytes / 2u; i++)
        xbuf[i] = value(i, 3);
    for (uint32_t i = 0; i < w_t.bytes / 2u; i++)
        wbuf[i] = value(i, 7);
    rk_binding inputs[] = {{.abi_version = RK_ABI_VERSION,
        .struct_size = sizeof(rk_binding), .id = 0, .dtype = RK_DTYPE_F16,
        .rank = 2, .shape = {M, K}, .data = xbuf, .nbytes = x_t.bytes}};
    rk_binding constants[] = {{.abi_version = RK_ABI_VERSION,
        .struct_size = sizeof(rk_binding), .id = 1, .dtype = RK_DTYPE_F16,
        .rank = 2, .shape = {N, K}, .data = wbuf, .nbytes = w_t.bytes}};
    rk_binding outputs[] = {{.abi_version = RK_ABI_VERSION,
        .struct_size = sizeof(rk_binding), .id = 2, .dtype = RK_DTYPE_F16,
        .rank = 2, .shape = {M, N}, .data = yctl, .nbytes = y_t.bytes}};

    /* Loose base plan used for the invalid-config and partial-budget checks. */
    desc.device_budget_bytes = base_device + (1u << 20);
    desc.claimed_device_bytes = base_device + (1u << 20);
    desc.host_budget_bytes = host_req + (1u << 20);
    desc.claimed_host_bytes = host_req;
    rk_plan *plan = NULL;
    int plan_rc = rk_plan_create(&desc, &plan, &diag);
    if (plan_rc)
        fprintf(stderr, "BUDGET-PLAN diag: %d %s\n", diag.code, diag.message);
    EXPECT(plan_rc == 0);
    if (!plan) goto done;
    rk_workspace *control = NULL;
    EXPECT(rk_workspace_create(3, plan, constants, 1, &control, &diag) == 0);
    rk_run_report report;
    if (control)
        EXPECT(rk_workspace_run(control, inputs, 1, outputs, 1, &report,
                                &diag) == 0);
    if (control) EXPECT(rk_workspace_destroy(control, &diag) == 0);
    rk_plan_destroy(plan);

    /* Exact device budget: no room for any cache command BO. */
    desc.device_budget_bytes = base_device;
    desc.claimed_device_bytes = base_device;
    plan = NULL;
    EXPECT(rk_plan_create(&desc, &plan, &diag) == 0);
    if (plan) {
        memcpy(ycache, yctl, y_t.bytes);
        unsigned allocs_before = rk_mock_allocs();
        rk_workspace *ws = NULL;
        EXPECT(rk_workspace_create(3, plan, constants, 1, &ws, &diag) == 0);
        EXPECT(ws && ws->multi_submit_cached == 0u);
        EXPECT(rk_mock_allocs() - allocs_before == 4u); /* 3 slots + fallback */
        if (ws) {
            EXPECT(rk_workspace_run(ws, inputs, 1, outputs, 1, &report, &diag) == 0);
            EXPECT(!memcmp(yctl, ycache, y_t.bytes));
            EXPECT(rk_workspace_destroy(ws, &diag) == 0);
        }
        rk_plan_destroy(plan);
    }

    /* Tight Host budget: plan is valid, but cache Host bytes do not fit. */
    desc.device_budget_bytes = base_device + (1u << 20);
    desc.claimed_device_bytes = base_device + (1u << 20);
    desc.host_budget_bytes = host_req;
    desc.claimed_host_bytes = host_req;
    plan = NULL;
    EXPECT(rk_plan_create(&desc, &plan, &diag) == 0);
    if (plan) {
        unsigned allocs_before = rk_mock_allocs();
        rk_workspace *ws = NULL;
        EXPECT(rk_workspace_create(3, plan, constants, 1, &ws, &diag) == 0);
        EXPECT(ws && ws->multi_submit_ready == 1u &&
               ws->multi_submit_cached == 3u);
        EXPECT(rk_mock_allocs() - allocs_before == 7u); /* base + 3 cache BOs */
        if (ws) {
            EXPECT(rk_workspace_host_bytes(ws, 1) <= desc.host_budget_bytes);
            EXPECT(rk_workspace_run(ws, inputs, 1, outputs, 1, &report, &diag) == 0);
            EXPECT(!memcmp(yctl, ycache, y_t.bytes));
            EXPECT(rk_workspace_destroy(ws, &diag) == 0);
        }
        rk_plan_destroy(plan);
    }

    /* Partial cache: only one of four submits fits the cache budget. */
    desc.device_budget_bytes = base_device + (1u << 20);
    desc.claimed_device_bytes = base_device + (1u << 20);
    desc.host_budget_bytes = host_req + (1u << 20);
    desc.claimed_host_bytes = host_req;
    plan = NULL;
    EXPECT(rk_plan_create(&desc, &plan, &diag) == 0);
    if (plan) {
        setenv("RK_PLAN_MULTI_SUBMIT_BUDGET", "40000", 1);
        unsigned allocs_before = rk_mock_allocs();
        rk_workspace *ws = NULL;
        EXPECT(rk_workspace_create(3, plan, constants, 1, &ws, &diag) == 0);
        EXPECT(ws && ws->multi_submit_ready == 1u &&
               ws->multi_submit_cached == 1u);
        EXPECT(rk_mock_allocs() - allocs_before == 5u); /* base + 1 cache BO */
        if (ws) {
            EXPECT(rk_workspace_run(ws, inputs, 1, outputs, 1, &report, &diag) == 0);
            EXPECT(!memcmp(yctl, ycache, y_t.bytes));
            EXPECT(rk_workspace_destroy(ws, &diag) == 0);
        }
        setenv("RK_PLAN_MULTI_SUBMIT_BUDGET", "4194304", 1);
        rk_plan_destroy(plan);
        plan = NULL;
    }

    /* Malformed budgets fail closed: no cache and unchanged output. */
    plan = NULL;
    EXPECT(rk_plan_create(&desc, &plan, &diag) == 0);
    if (!plan) goto done;
    const char *bad_budgets[] = {"-1", "not-a-number", "0",
                                 "999999999999999999999999999999"};
    for (uint32_t i = 0; i < 4; i++) {
        setenv("RK_PLAN_MULTI_SUBMIT_BUDGET", bad_budgets[i], 1);
        unsigned allocs_before = rk_mock_allocs();
        rk_workspace *ws = NULL;
        EXPECT(rk_workspace_create(3, plan, constants, 1, &ws, &diag) == 0);
        EXPECT(ws && ws->multi_submit_ready == 0u &&
               ws->submit_cache == NULL);
        EXPECT(rk_mock_allocs() - allocs_before == 4u);
        if (ws) {
            EXPECT(rk_workspace_run(ws, inputs, 1, outputs, 1, &report, &diag) == 0);
            EXPECT(!memcmp(yctl, ycache, y_t.bytes));
            EXPECT(rk_workspace_destroy(ws, &diag) == 0);
        }
    }
    setenv("RK_PLAN_MULTI_SUBMIT_BUDGET", "4194304", 1);
    rk_plan_destroy(plan);

done:
    free(xbuf); free(wbuf); free(yctl); free(ycache);
}

int main(void) {
    /* Exercise the bounded command-BO reuse through the production runtime in
     * this process.  The flag is still a creation-time snapshot; individual
     * tests below may turn it off to check the control path. */
    setenv("RK_PLAN_COMMAND_CACHE", "1", 1);
    setenv("RK_PLAN_MULTI_SUBMIT_CACHE", "1", 1);
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

    uint32_t a_x0 = 0, a_x1 = 1, a_w0 = 2, a_w1 = 3, a_mm = 4, a_acc = 5;
    uint32_t view_p = 4, view_z = 5;
    uint32_t mm_in[] = {0, 2};
    uint32_t acc_in[] = {1, 3, 4};
    uint32_t mm_deps[] = {0, 1, 2, 3}; /* every host action that fills this tile */
    uint32_t acc_dep[] = {a_mm};
    uint32_t down_in[] = {5};
    uint32_t down_deps[] = {a_acc};
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
    actions[4].ndeps = 4;
    actions[4].deps = mm_deps;
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
    actions[6].ndeps = 1;
    actions[6].deps = down_deps;

    uint32_t task_ids[] = {4, 5};
    (void)0;
    rk_submit_desc submit = {
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_submit_desc),
        .id = 0, .mode = RK_SUBMIT_CHAIN, .core_policy = RK_CORE_SINGLE,
        .ntask_ids = 2, .task_ids = task_ids};
    rk_publish_desc publish = {
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_publish_desc),
        .binding_id = 2, .view_id = 5,
        .dst_row_offset = 0, .dst_col_offset = 0, .rows = M, .cols = N};
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
    desc.host_budget_bytes = 1u << 23;
    desc.claimed_device_bytes = claimed;
    desc.claimed_host_bytes = 1u << 22; /* bindings + staging, see the C check */
    desc.runtime_abi_version = RK_ABI_VERSION;

    rk_diag diag;
    rk_plan *plan = NULL;
    EXPECT(rk_plan_create(&desc, &plan, &diag) == 0);
    EXPECT(plan != NULL);
    if (!plan) return 1;
    /* Direct descriptor negatives exercise the C verifier independently of the
     * Python/Pydantic-like layers.  Every case pins the exact status code, the
     * NULL handle and a diagnostic fragment. */
    rk_plan_desc bad = desc;
    rk_plan *badplan = NULL;
    rk_action_desc bad_actions[7];
    rk_view_desc bad_views[6];
    rk_buffer_desc bad_buffers[6];
    (void)badplan;

    bad = desc;
    bad.values = NULL;
    EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan, RK_E_SCHEMA,
                  "invalid values array");

    memcpy(bad_actions, actions, sizeof(bad_actions));
    bad_actions[4].template_id = 99;
    bad = desc;
    bad.actions = bad_actions;
    /* The arity table rejects an unknown template before the emit layer. */
    EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan, RK_E_SCHEMA,
                  "unknown NPU task template");

    memcpy(bad_views, views, sizeof(bad_views));
    bad_views[0].offset_bytes = 1;
    bad = desc;
    bad.views = bad_views;
    EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan, RK_E_SHAPE,
                  "view exceeds buffer capacity");

    bad = desc;
    bad.claimed_device_bytes = 1;
    EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan, RK_E_BUDGET,
                  "understate");

    /* Review blocker 1: a stride that would walk outside the borrowed binding. */
    memcpy(bad_actions, actions, sizeof(bad_actions));
    bad_actions[0].source_row_stride = 0xffffffffu;
    bad = desc;
    bad.actions = bad_actions;
    EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan, RK_E_SHAPE,
                  "source_row_stride");

    /* Review blocker 2: swapping the K chain makes the accumulator task read a
     * partial that has not been produced yet.  IDs stay topological, so only the
     * emission-order analysis can reject it. */
    uint32_t swapped_tasks[] = {5, 4};
    rk_submit_desc swapped_submit = submit;
    swapped_submit.task_ids = swapped_tasks;
    bad = desc;
    bad.submits = &swapped_submit;
    /* Rejected either by the emission-order rule or by the submit whitelist. */
    EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan, RK_E_SCHEMA,
                  "submit task sequence");

    /* Publish coverage: a tile larger than the output binding. */
    rk_publish_desc bad_publish = publish;
    bad_publish.rows = M + 1;
    bad = desc;
    bad.publish = &bad_publish;
    EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan, RK_E_SHAPE,
                  "exceeds the output binding");

    /* Publish coverage: a tile smaller than the binding is not a full tiling. */
    bad_publish = publish;
    bad_publish.rows = M - 1;
    bad = desc;
    bad.publish = &bad_publish;
    EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan, RK_E_SHAPE,
                  "publish tile must publish the whole view");

    /* Slot reuse: the output tile is written by the same mm_acc that reads w1,
     * so sharing w1's slot is a real read/write overlap inside one submit. */
    memcpy(bad_buffers, buffers, sizeof(bad_buffers));
    bad_buffers[5].slot_id = bad_buffers[3].slot_id;
    bad = desc;
    bad.buffers = bad_buffers;
    EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan, RK_E_ALIAS,
                  "aliases/shared BO");

    /* A later output must not reuse a constant weight's slot: the weight is
     * uploaded once at create and has to survive every run. */
    memcpy(bad_buffers, buffers, sizeof(bad_buffers));
    bad_buffers[5].slot_id = bad_buffers[2].slot_id; /* w0 slot */
    bad = desc;
    bad.buffers = bad_buffers;
    EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan, RK_E_ALIAS,
                  "constant workspace buffer");

    /* A fixed-size ABI string must be NUL terminated.  A full 64-byte origin
     * previously made the clone append its terminator one byte past the field
     * (ASan: heap-buffer-overflow WRITE of size 1). */
    {
        rk_action_desc raw_actions[7];
        memcpy(raw_actions, actions, sizeof(raw_actions));
        memset(raw_actions[6].origin, 'A', sizeof(raw_actions[6].origin));
        bad = desc;
        bad.actions = raw_actions;
        EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan, RK_E_SCHEMA,
                      "not NUL terminated");
    }
    /* Accepted boundary: 63 characters plus the NUL exactly fills the field and
     * must be copied verbatim. */
    {
        rk_action_desc long_actions[7];
        memcpy(long_actions, actions, sizeof(long_actions));
        memset(long_actions[6].origin, 'B', sizeof(long_actions[6].origin));
        long_actions[6].origin[63] = 0;
        rk_plan_desc boundary = desc;
        boundary.actions = long_actions;
        rk_plan *boundary_plan = NULL;
        int brc = rk_plan_create(&boundary, &boundary_plan, &diag);
        EXPECT(brc == 0);
        EXPECT(boundary_plan != NULL);
        if (boundary_plan) {
            EXPECT(strlen(boundary_plan->actions[6].origin) == 63);
            EXPECT(boundary_plan->actions[6].origin[63] == 0);
            EXPECT(boundary_plan->actions[6].origin[62] == 'B');
            rk_plan_destroy(boundary_plan);
        }
    }

    /* A read whose producer is not declared as a dependency must be rejected:
     * dependency order alone is not enough, the declared edges must cover the
     * actual data flow. */
    {
        rk_action_desc raw_actions[7];
        memcpy(raw_actions, actions, sizeof(raw_actions));
        raw_actions[6].ndeps = 0;
        raw_actions[6].deps = NULL; /* download no longer depends on the mm_acc */
        bad = desc;
        bad.actions = raw_actions;
        EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan,
                      RK_E_SCHEMA, "not a declared dependency");
    }
    /* Repeated device writes to one buffer are the accumulator ping-pong and are
     * allowed; an extra host upload of the same buffer is not. */
    {
        rk_action_desc raw_actions[7];
        memcpy(raw_actions, actions, sizeof(raw_actions));
        uint32_t p_view = 4;
        raw_actions[1].outputs = &p_view; /* host upload into the P accumulator */
        bad = desc;
        bad.actions = raw_actions;
        EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan,
                      RK_E_SCHEMA, "written by both the host and the device");
    }
    /* Host staging bytes are recomputed, not trusted. */
    bad = desc;
    bad.claimed_host_bytes = 1024;
    EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan, RK_E_BUDGET,
                  "host bytes");

    /* Forged argument counts must be rejected by the arity table before the input
     * or output array is walked (ASan would catch the out-of-bounds read). */
    {
        rk_action_desc raw_actions[7];
        uint32_t nin[] = {0, 2};
        memcpy(raw_actions, actions, sizeof(raw_actions));
        raw_actions[4].ninputs = UINT32_MAX;
        raw_actions[4].inputs = nin;
        bad = desc;
        bad.actions = raw_actions;
        EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan,
                      RK_E_SCHEMA, "argument count");
    }
    {
        rk_action_desc raw_actions[7];
        uint32_t nout[] = {4};
        memcpy(raw_actions, actions, sizeof(raw_actions));
        raw_actions[4].noutputs = UINT32_MAX;
        raw_actions[4].outputs = nout;
        bad = desc;
        bad.actions = raw_actions;
        EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan,
                      RK_E_SCHEMA, "argument count");
    }
    /* Task id validation must happen before any index into desc->actions, so an
     * out-of-range id is a diagnostic, never an out-of-bounds read.  Both the
     * first and a later position are covered, with nactions and UINT32_MAX. */
    {
        uint32_t first_bad[] = {(uint32_t)desc.nactions, 5u};
        rk_submit_desc bad_submit = submit;
        bad_submit.task_ids = first_bad;
        bad = desc;
        bad.submits = &bad_submit;
        EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan,
                      RK_E_SCHEMA, "out of range");
    }
    {
        uint32_t later_bad[] = {4u, UINT32_MAX};
        rk_submit_desc bad_submit = submit;
        bad_submit.task_ids = later_bad;
        bad = desc;
        bad.submits = &bad_submit;
        EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan,
                      RK_E_SCHEMA, "out of range");
    }
    /* A zero claim is not a budget bypass. */
    {
        bad = desc;
        bad.claimed_host_bytes = 0;
        EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan,
                      RK_E_BUDGET, "must be stated");
    }
    /* Stale runtime ABI. */
    bad = desc;
    bad.runtime_abi_version = RK_ABI_VERSION + 1u;
    EXPECT_REJECT(rk_plan_create(&bad, &badplan, &diag), badplan, RK_E_SCHEMA,
                  "runtime_abi_version");

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
    EXPECT(live_bos == 7); /* six data slots plus one command workspace BO */
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
    EXPECT(!live_bos);
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
    multitile_publish_case();
    partial_coverage_case();
    lut_segment_case();
    bo_dma_field_case();
    mock_zero_program_case();
    mock_untracked_program_case();
    mock_tag_capacity_case();
    multi_output_budget_case();
    many_upload_fallback_case();
    multi_submit_budget_case();
    printf("RK_PLAN host checks=%u failures=%u\n", checks, failures);
    return failures ? 1 : 0;
}
