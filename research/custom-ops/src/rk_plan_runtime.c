// SPDX-License-Identifier: GPL-3.0-or-later
#define _POSIX_C_SOURCE 200809L
#include "rk_plan_internal.h"
#include "rk_plan_download.h"
#include "rknpu_submit.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Wall-clock counter for the timing probes.  ``now_us_calls`` lets an entry
 * point report how much time its own instrumentation consumed, so a reader can
 * separate the measured work from the measurement.
 *
 * RK_PLAN_PROBE_OFF builds the identical runtime with the probes compiled out,
 * which is the end-to-end control for "does the instrumentation cost anything".
 * The counters then stay zero and every attribution field reads 0. */
#ifdef RK_PLAN_PROBE_OFF
static double now_us(void) { return 0.0; }
#else
static atomic_ulong now_us_calls;

static double now_us(void);

/* Cost of one probe, calibrated once with the same primitive under the same
 * clock.  A wildly implausible calibration is discarded rather than reported. */
static double probe_overhead_us(void) {
    static double overhead = -1.0;
    if (overhead >= 0.0) return overhead;
    double sink = 0.0;
    double t0 = now_us();
    for (int i = 0; i < 1000; i++) sink += now_us();
    double t1 = now_us();
    double value = (t1 - t0) / 1000.0;
    if (!(value > 0.0) || value > 50.0) value = 0.0;
    overhead = value;
    (void)sink;
    return overhead;
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    atomic_fetch_add_explicit(&now_us_calls, 1, memory_order_relaxed);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}
#endif /* RK_PLAN_PROBE_OFF */

static int map_native(int rc) {
    if (rc == 0) return 0;
    if (rc == ROCKET_E_NOMEM) return RK_E_NOMEM;
    if (rc == ROCKET_E_SHAPE) return RK_E_SHAPE;
    if (rc == ROCKET_E_UNSUPPORTED) return RK_E_UNSUPPORTED;
    return RK_E_DEVICE;
}

__attribute__((weak)) int rk_target_check(rk_diag *diag) {
    (void)diag;
    return 0;
}

static int fail_native(rk_diag *diag, const char *stage, const char *what,
                       uint32_t action_id, int native) {
    int code = map_native(native);
    rk_diag_set(diag, code, stage, what, "", action_id, native);
    return code;
}

static const rk_binding *find_binding(const rk_binding *arr, uint32_t count,
                                      uint32_t id) {
    const rk_binding *found = NULL;
    for (uint32_t i = 0; i < count; i++) {
        if (arr[i].id == id) {
            if (found) return NULL; /* duplicate */
            found = &arr[i];
        }
    }
    return found;
}

static int binding_matches(const rk_value_desc *v, const rk_binding *b,
                           rk_diag *diag, const char *stage) {
    if (!v || !b) {
        rk_diag_set(diag, RK_E_SHAPE, stage, "missing binding", "", UINT32_MAX, 0);
        return RK_E_SHAPE;
    }
    if (b->abi_version != RK_ABI_VERSION || b->struct_size < sizeof(*b) ||
        b->dtype != v->dtype || b->rank != v->rank ||
        b->shape[0] != v->shape[0] || b->shape[1] != v->shape[1] ||
        b->nbytes != v->nbytes || (v->nbytes && !b->data)) {
        rk_diag_set(diag, RK_E_SHAPE, stage,
                    "binding dtype/rank/shape/nbytes mismatch or null data",
                    "", UINT32_MAX, 0);
        return RK_E_SHAPE;
    }
    return 0;
}

static int check_finite_binding(const rk_binding *b, rk_diag *diag) {
    const _Float16 *data = (const _Float16 *)b->data;
    size_t count = (size_t)(b->nbytes / 2u);
    for (size_t i = 0; i < count; i++) {
        if (!isfinite((double)data[i])) {
            rk_diag_set(diag, RK_E_NUMERIC_CONTRACT, "run",
                        "non-finite FP16 input/constant is rejected before upload",
                        "", UINT32_MAX, 0);
            return RK_E_NUMERIC_CONTRACT;
        }
    }
    return 0;
}

static int ranges_overlap(const void *a, uint64_t an, const void *b, uint64_t bn) {
    uintptr_t ap = (uintptr_t)a, bp = (uintptr_t)b;
    if (!a || !b || !an || !bn) return 0;
    if (ap > UINTPTR_MAX - an || bp > UINTPTR_MAX - bn) return 1;
    return ap < bp + bn && bp < ap + an;
}

static co_tensor tensor_from_desc(const rk_tensor_desc *t) {
    co_tensor out;
    memset(&out, 0, sizeof(out));
    out.type = CO_F16;
    out.layout = t->layout == RK_LAYOUT_WEIGHT_F16 ? CO_WEIGHT :
                 (t->layout == RK_LAYOUT_LINEAR_F16 ? CO_LINEAR : CO_CUBE);
    out.rows = t->rows;
    out.cols = t->cols;
    out.physical_rows = t->physical_rows;
    out.physical_cols = t->physical_cols;
    out.atom = t->atom;
    out.bytes = (size_t)t->bytes;
    return out;
}

static int refresh_views(rk_workspace *ws, rk_diag *diag) {
    for (uint32_t i = 0; i < ws->plan->nviews; i++) {
        const rk_view_desc *v = &ws->plan->views[i];
        const rk_buffer_desc *b = rk_plan_buffer(ws->plan, v->buffer_id);
        if (!b || b->slot_id >= ws->nslots) {
            rk_diag_set(diag, RK_E_SHAPE, "workspace", "view buffer/slot missing",
                        "", UINT32_MAX, 0);
            return RK_E_SHAPE;
        }
        co_tensor t = tensor_from_desc(&v->tensor);
        int rc = co_view_init(&ws->views[i], &ws->slots[b->slot_id],
                              (size_t)(b->offset + v->offset_bytes), &t);
        if (rc) return fail_native(diag, "workspace", "co_view_init rejected a view",
                                   UINT32_MAX, rc);
    }
    return 0;
}

static int upload_action(rk_workspace *ws, const rk_action_desc *a,
                         const rk_binding *binding, uint64_t *bytes_out,
                         rk_diag *diag) {
    if (!a || a->kind != RK_ACTION_UPLOAD || a->noutputs != 1) return RK_E_INVALID;
    co_view *dst = &ws->views[a->outputs[0]];
    if (!dst->buffer || !dst->buffer->bo.handle) return RK_E_INVALID;
    int rc = rocket_bo_prep(ws->fd, &dst->buffer->bo, 1, 2000000000ull);
    if (rc) return fail_native(diag, "upload", "rocket_bo_prep(write) failed",
                               a->id, rc);
    memset((char *)dst->buffer->bo.ptr + dst->offset, 0, dst->tensor.bytes);
    const _Float16 *src = (const _Float16 *)binding->data;
    _Float16 *base = (_Float16 *)((char *)dst->buffer->bo.ptr + dst->offset);
    uint32_t transpose = (a->flags & RK_ACTION_FLAG_TRANSPOSE_SOURCE) != 0;
    uint32_t broadcast_scalar = (a->flags & RK_ACTION_FLAG_BROADCAST_SCALAR) != 0;
    uint32_t broadcast_channel = (a->flags & RK_ACTION_FLAG_BROADCAST_CHANNEL) != 0;
    const uint64_t elements = binding->nbytes / 2u;
    for (uint32_t r = 0; r < dst->tensor.rows; r++) {
        for (uint32_t c = 0; c < dst->tensor.cols; c++) {
            _Float16 value;
            if (broadcast_scalar) {
                if (elements != 1u) {
                    (void)rocket_bo_fini(ws->fd, &dst->buffer->bo);
                    rk_diag_set(diag, RK_E_SHAPE, "upload",
                                "scalar broadcast binding is not one element", "",
                                a->id, 0);
                    return RK_E_SHAPE;
                }
                value = src[0];
            } else if (broadcast_channel) {
                uint64_t source_col = (uint64_t)a->source_col_offset + c;
                if (source_col >= elements) {
                    (void)rocket_bo_fini(ws->fd, &dst->buffer->bo);
                    rk_diag_set(diag, RK_E_SHAPE, "upload",
                                "channel broadcast binding is shorter than the "
                                "destination row", "", a->id, 0);
                    return RK_E_SHAPE;
                }
                value = src[source_col];
            } else {
                uint32_t sr = transpose ? a->source_row_offset + c
                                        : a->source_row_offset + r;
                uint32_t sc = transpose ? a->source_col_offset + r
                                        : a->source_col_offset + c;
                /* Re-checked here as well: the verifier already proved this, but the
                 * packing loop must never index outside the borrowed binding even if
                 * a future caller bypasses rk_plan_create. */
                if ((uint64_t)sr * a->source_row_stride + sc >= elements) {
                    (void)rocket_bo_fini(ws->fd, &dst->buffer->bo);
                    rk_diag_set(diag, RK_E_SHAPE, "upload",
                                "packer source index exceeds the binding", "", a->id, 0);
                    return RK_E_SHAPE;
                }
                value = src[(size_t)sr * a->source_row_stride + sc];
            }
            base[co_index(&dst->tensor, r, c)] = value;
        }
    }
    rc = rocket_bo_fini(ws->fd, &dst->buffer->bo);
    if (rc) return fail_native(diag, "upload", "rocket_bo_fini failed", a->id, rc);
    /* Transfer accounting counts the packed bytes actually written for this
     * action, not the whole borrowing binding once per tile. */
    if (bytes_out) *bytes_out += (uint64_t)dst->tensor.bytes;
    return 0;
}

static int upload_role(rk_workspace *ws, uint32_t role,
                       const rk_binding *bindings, uint32_t nbindings,
                       uint64_t *calls, uint64_t *bytes, rk_diag *diag) {
    for (uint32_t i = 0; i < ws->plan->nactions; i++) {
        const rk_action_desc *a = &ws->plan->actions[i];
        if (a->kind != RK_ACTION_UPLOAD) continue;
        const rk_value_desc *v = rk_plan_value(ws->plan, a->source_binding);
        if (!v || v->role != role) continue;
        const rk_binding *b = find_binding(bindings, nbindings, a->source_binding);
        if (!b) {
            rk_diag_set(diag, RK_E_SHAPE, "upload",
                        "required input/constant binding is absent or duplicated",
                        "", a->id, 0);
            return RK_E_SHAPE;
        }
        int rc = upload_action(ws, a, b, bytes, diag);
        if (rc) return rc;
        if (calls) (*calls)++;
    }
    return 0;
}

int rk_workspace_create(int fd, const rk_plan *plan,
                        const rk_binding *constants, uint32_t nconstants,
                        rk_workspace **out, rk_diag *diag) {
    rk_diag_clear(diag);
    if (!out) return RK_E_INVALID;
    *out = NULL;
    if (!plan || fd < 0) {
        rk_diag_set(diag, RK_E_INVALID, "workspace_create",
                    "fd>=0 and a validated plan are required", "", UINT32_MAX, 0);
        return RK_E_INVALID;
    }
    int target_rc = rk_target_check(diag);
    if (target_rc) return target_rc;
    int context_rc = rocket_context_faulted(fd);
    if (context_rc != 0) {
        int code = context_rc > 0 ? RK_E_POISONED : RK_E_DEVICE;
        rk_diag_set(diag, code, "workspace_create",
                    context_rc > 0
                        ? "provider context is quarantined; no workspace may "
                          "reuse this fd until process-level recovery"
                        : "provider context state is unavailable",
                    "", UINT32_MAX, context_rc);
        return code;
    }
    if ((nconstants && !constants) || nconstants > plan->nvalues) {
        rk_diag_set(diag, RK_E_SCHEMA, "workspace_create",
                    "constant pointer/count mismatch", "", UINT32_MAX, 0);
        return RK_E_SCHEMA;
    }
    rk_workspace *ws = calloc(1, sizeof(*ws));
    if (!ws) return RK_E_NOMEM;
    ws->fd = fd;
    atomic_init(&ws->busy, 0);
    atomic_init(&ws->poisoned, 0);
    if (rk_plan_clone_plan(plan, &ws->plan, diag)) {
        free(ws);
        return diag && diag->code ? diag->code : RK_E_INVALID;
    }
    uint32_t constant_values = 0;
    for (uint32_t i = 0; i < ws->plan->nvalues; i++)
        if (ws->plan->values[i].role == RK_ROLE_CONSTANT) constant_values++;
    if (nconstants != constant_values) {
        rk_diag_set(diag, RK_E_SHAPE, "workspace_create",
                    "constant binding count mismatch", "", UINT32_MAX, 0);
        rk_plan_free(ws->plan);
        free(ws);
        return RK_E_SHAPE;
    }
    for (uint32_t i = 0; i < ws->plan->nvalues; i++) {
        const rk_value_desc *v = &ws->plan->values[i];
        if (v->role != RK_ROLE_CONSTANT) continue;
        const rk_binding *b = find_binding(constants, nconstants, i);
        int rc = binding_matches(v, b, diag, "workspace_create");
        if (rc) {
            rk_plan_free(ws->plan);
            free(ws);
            return rc;
        }
        rc = check_finite_binding(b, diag);
        if (rc) {
            rk_plan_free(ws->plan);
            free(ws);
            return rc;
        }
    }

    /* Constants must match every constant binding exactly once. */
    uint32_t max_slot = 0;
    for (uint32_t i = 0; i < ws->plan->nbuffers; i++)
        if (ws->plan->buffers[i].slot_id > max_slot)
            max_slot = ws->plan->buffers[i].slot_id;
    ws->nslots = ws->plan->nbuffers ? max_slot + 1u : 0u;
    ws->slots = calloc(ws->nslots ? ws->nslots : 1u, sizeof(*ws->slots));
    ws->views = calloc(ws->plan->nviews ? ws->plan->nviews : 1u, sizeof(*ws->views));
    if (!ws->slots || !ws->views) goto nomem;
    uint64_t *slot_cap = calloc(ws->nslots ? ws->nslots : 1u, sizeof(uint64_t));
    if (!slot_cap) goto nomem;
    for (uint32_t i = 0; i < ws->plan->nbuffers; i++) {
        const rk_buffer_desc *b = &ws->plan->buffers[i];
        uint64_t end = b->offset + b->capacity;
        if (end > slot_cap[b->slot_id]) slot_cap[b->slot_id] = end;
    }
    for (uint32_t i = 0; i < ws->nslots; i++) {
        if (!slot_cap[i]) continue;
        int rc = co_buffer_alloc(fd, (size_t)slot_cap[i], &ws->slots[i]);
        if (rc) {
            free(slot_cap);
            fail_native(diag, "workspace_create", "co_buffer_alloc failed",
                        UINT32_MAX, rc);
            goto fail;
        }
    }
    free(slot_cap);
    if (refresh_views(ws, diag)) goto fail;
    ws->max_submit_tasks = 1;
    for (uint32_t i = 0; i < ws->plan->nsubmits; i++)
        if (ws->plan->submits[i].ntask_ids > ws->max_submit_tasks)
            ws->max_submit_tasks = ws->plan->submits[i].ntask_ids;
    int rc = co_execute_workspace_init(fd, ws->max_submit_tasks,
                                       &ws->execute_workspace);
    if (rc) {
        fail_native(diag, "workspace_create", "command workspace allocation failed",
                    UINT32_MAX, rc);
        goto fail;
    }
    ws->execute_workspace_ready = 1;

    /* Per-run Host scratch is sized once from the immutable plan: input/output
     * arity, the staging stride (sum of output value bytes) and the deepest
     * submit group.  Nothing here depends on the caller's data. */
    for (uint32_t i = 0; i < ws->plan->nvalues; i++) {
        const rk_value_desc *v = &ws->plan->values[i];
        if (v->role == RK_ROLE_INPUT) ws->ninputs++;
        if (v->role == RK_ROLE_OUTPUT) {
            ws->noutputs++;
            ws->staging_stride += (size_t)v->nbytes;
        }
    }
    ws->staging_offsets = calloc(ws->noutputs ? ws->noutputs : 1u,
                                 sizeof(*ws->staging_offsets));
    ws->tile_count_scratch = calloc(ws->noutputs ? ws->noutputs : 1u,
                                    sizeof(*ws->tile_count_scratch));
    ws->batch_scratch = calloc(ws->max_submit_tasks,
                               sizeof(*ws->batch_scratch));
    if (!ws->staging_offsets || !ws->tile_count_scratch || !ws->batch_scratch)
        goto nomem;
    ws->staging_capacity = 1;
    ws->staging_arena = calloc(1u, ws->staging_stride ? ws->staging_stride : 1u);
    if (!ws->staging_arena) goto nomem;
    /* Largest downloaded tile: fixed by the immutable plan. */
    for (uint32_t i = 0; i < ws->plan->nactions; i++) {
        const rk_action_desc *a = &ws->plan->actions[i];
        if (a->kind != RK_ACTION_DOWNLOAD || a->ninputs != 1) continue;
        if (a->inputs[0] >= ws->plan->nviews) continue;
        size_t bytes = (size_t)ws->views[a->inputs[0]].tensor.bytes;
        if (bytes > ws->tile_scratch_bytes) ws->tile_scratch_bytes = bytes;
    }
    ws->tile_scratch = calloc(1u, ws->tile_scratch_bytes ? ws->tile_scratch_bytes
                                                         : 1u);
    if (!ws->tile_scratch) goto nomem;

    /* The plan's declared Host budget must cover the whole resident footprint,
     * not just the binding/staging estimate: the verifier now includes the same
     * runtime scratch terms, so a plan that under-declares is refused here
     * instead of being absorbed at run time. */
    {
        uint64_t budget = ws->plan->host_budget_bytes
                              ? ws->plan->host_budget_bytes
                              : ws->plan->claimed_host_bytes;
        uint64_t resident = rk_workspace_host_bytes(ws, 1);
        if (budget && resident > budget) {
            rk_diag_set(diag, RK_E_BUDGET, "workspace_create",
                        "workspace Host footprint exceeds the plan Host budget; "
                        "recompile the plan with the corrected accounting",
                        "", UINT32_MAX, 0);
            goto fail;
        }
    }

    uint64_t ignored_calls = 0, ignored_bytes = 0;
    rc = upload_role(ws, RK_ROLE_CONSTANT, constants, nconstants,
                     &ignored_calls, &ignored_bytes, diag);
    if (rc) goto fail;
    ws->created = 1;
    *out = ws;
    return 0;

nomem:
    rk_diag_set(diag, RK_E_NOMEM, "workspace_create", "out of memory",
                "", UINT32_MAX, 0);
    goto fail;
fail:
    if (ws->execute_workspace_ready)
        co_execute_workspace_destroy(&ws->execute_workspace);
    if (ws->slots) {
        for (uint32_t i = 0; i < ws->nslots; i++) co_buffer_free(&ws->slots[i]);
    }
    free(ws->slots);
    free(ws->views);
    free(ws->staging_offsets);
    free(ws->tile_count_scratch);
    free(ws->batch_scratch);
    free(ws->staging_arena);
    free(ws->tile_scratch);
    rk_plan_free(ws->plan);
    free(ws);
    return diag && diag->code ? diag->code : RK_E_INVALID;
}

/* ------------------------------------------------------------------------- */
/* Envelope attribution.                                                      */
/*                                                                            */
/* Every field below is a disjoint interval inside the enclosing entry-point  */
/* dwell, so their sum plus the caller-observed residual accounts for the     */
/* whole call.  They are Host envelopes, not device compute times.            */
/* ------------------------------------------------------------------------- */

/* The extended entries validate the caller's declaration before writing a single
 * byte.  An unusable size is a schema error, never a guess: guessing is exactly
 * what would let an old 88-byte caller be overrun by a newer library. */
static int attribution_size(const rk_run_attribution *envelope, const char *stage,
                            rk_diag *diag, uint32_t *out) {
    uint32_t size = envelope->struct_size;
    if (size < RK_RUN_ATTRIBUTION_V1_SIZE || size > (uint32_t)sizeof(*envelope)) {
        rk_diag_set(diag, RK_E_SCHEMA, stage,
                    "rk_run_attribution.struct_size is not a usable ABI "
                    "declaration", "", UINT32_MAX, 0);
        return RK_E_SCHEMA;
    }
    *out = size;
    return 0;
}

static int group_size(const rk_group_report *report, const char *stage,
                      rk_diag *diag, uint32_t *out) {
    uint32_t size = report->struct_size;
    if (size < RK_GROUP_REPORT_V1_SIZE || size > (uint32_t)sizeof(*report)) {
        rk_diag_set(diag, RK_E_SCHEMA, stage,
                    "rk_group_report.struct_size is not a usable ABI "
                    "declaration", "", UINT32_MAX, 0);
        return RK_E_SCHEMA;
    }
    *out = size;
    return 0;
}

#define ATTRIBUTION_HAS(r, field)                                            \
    ((r)->struct_size >=                                                     \
     (uint32_t)(offsetof(rk_run_attribution, field) + sizeof((r)->field)))
#define ATTRIBUTION_SET(r, field, value)                                     \
    do {                                                                     \
        if (ATTRIBUTION_HAS(r, field)) (r)->field = (value);                   \
    } while (0)
#define GROUP_REPORT_HAS(r, field)                                           \
    ((r)->struct_size >=                                                     \
     (uint32_t)(offsetof(rk_group_report, field) + sizeof((r)->field)))
#define GROUP_REPORT_SET(r, field, value)                                    \
    do {                                                                     \
        if (GROUP_REPORT_HAS(r, field)) (r)->field = (value);                  \
    } while (0)

/* The legacy report has a fixed size; the entry writes exactly that much and
 * never consults struct_size, because pre-extension callers were not required
 * to initialise it. */
static void run_report_clear(rk_run_report *report) {
    memset(report, 0, sizeof(*report));
    report->abi_version = RK_ABI_VERSION;
    report->struct_size = (uint32_t)sizeof(*report);
    report->fail_action_id = UINT32_MAX;
    report->numeric_mode = 1;
}

/* Callers of the extended structs have already validated struct_size; clear
 * exactly the declared extent so no byte outside it is touched. */
static void attribution_clear(rk_run_attribution *envelope) {
    uint32_t size = envelope->struct_size;
    memset(envelope, 0, size);
    envelope->struct_size = size;
    ATTRIBUTION_SET(envelope, abi_version, RK_ABI_VERSION);
    ATTRIBUTION_SET(envelope, fail_action_id, UINT32_MAX);
    ATTRIBUTION_SET(envelope, numeric_mode, 1);
}

static void group_report_clear(rk_group_report *report) {
    uint32_t size = report->struct_size;
    memset(report, 0, size);
    report->struct_size = size;
    GROUP_REPORT_SET(report, abi_version, RK_ABI_VERSION);
    GROUP_REPORT_SET(report, fail_action_id, UINT32_MAX);
    GROUP_REPORT_SET(report, fail_repeat, UINT32_MAX);
    GROUP_REPORT_SET(report, numeric_mode, 1);
}

typedef struct {
    /* Per-repeat Host scratch borrowed from the workspace (single-run gate). */
    void *staging;              /* this repeat's staging block, zeroed per repeat */
    size_t staging_stride;
    uint32_t *tile_count;       /* [noutputs] */
    co_program *batch;          /* [RK_MAX_PROGRAMS_PER_SUBMIT] */
    uint32_t noutputs;
    /* Counters and attribution accumulate across repeats. */
    uint64_t upload_calls, upload_bytes, download_calls, download_bytes;
    uint32_t submit_count;
    /* Copied out of the plan while the single-run lock is held: the public
     * entries fill their report after the lock is released, and must never
     * dereference the workspace again (another thread may destroy it). */
    uint32_t task_count;
    double validate_us, pack_us, emit_us, command_us, submit_alloc_us;
    double submit_wait_us, output_sync_us, download_us, guard_us, publish_us;
    double alloc_us;
    /* Self-consistency witnesses. */
    double total_us;
    double clock_us;
} rk_run_state;

/* Bracket one entry point: ``now_us_calls`` is read at entry and exit so the
 * probe cost can be reported instead of being folded into a named bucket.
 * The counter is process-global; concurrent workspaces in one process would
 * share it, which is why the reported value is only claimed per call for the
 * serial execution policy this runtime admits. */
typedef struct {
    unsigned long probes;
    double start;
} rk_clock_window;

static rk_clock_window clock_window_open(void) {
    rk_clock_window w;
#ifdef RK_PLAN_PROBE_OFF
    w.probes = 0;
#else
    w.probes = atomic_load_explicit(&now_us_calls, memory_order_relaxed);
#endif
    w.start = now_us();
    return w;
}

static void clock_window_close(const rk_clock_window *w, rk_run_state *st) {
#ifdef RK_PLAN_PROBE_OFF
    (void)w;
    st->total_us = 0.0;
    st->clock_us = 0.0;
#else
    double end = now_us();
    unsigned long used = atomic_load_explicit(&now_us_calls,
                                              memory_order_relaxed) - w->probes;
    st->total_us = end - w->start;
    st->clock_us = (double)used * probe_overhead_us();
#endif
}

/* The legacy entry writes exactly the original report and nothing else. */
static void run_report_fill(rk_run_report *report, const rk_run_state *st,
                            uint32_t task_count) {
    report->task_count = task_count;
    report->submit_count = st->submit_count;
    report->upload_calls = st->upload_calls;
    report->upload_bytes = st->upload_bytes;
    report->download_calls = st->download_calls;
    report->download_bytes = st->download_bytes;
    report->pack_us = st->pack_us;
    report->emit_us = st->emit_us;
    report->submit_wait_us = st->submit_wait_us;
    report->output_sync_us = st->output_sync_us;
}

static void attribution_fill(rk_run_attribution *report, const rk_run_state *st,
                             uint32_t task_count) {
    ATTRIBUTION_SET(report, task_count, task_count);
    ATTRIBUTION_SET(report, submit_count, st->submit_count);
    ATTRIBUTION_SET(report, upload_calls, st->upload_calls);
    ATTRIBUTION_SET(report, upload_bytes, st->upload_bytes);
    ATTRIBUTION_SET(report, download_calls, st->download_calls);
    ATTRIBUTION_SET(report, download_bytes, st->download_bytes);
    ATTRIBUTION_SET(report, validate_us, st->validate_us);
    ATTRIBUTION_SET(report, pack_us, st->pack_us);
    ATTRIBUTION_SET(report, emit_us, st->emit_us);
    ATTRIBUTION_SET(report, command_us, st->command_us);
    ATTRIBUTION_SET(report, submit_alloc_us, st->submit_alloc_us);
    ATTRIBUTION_SET(report, submit_wait_us, st->submit_wait_us);
    ATTRIBUTION_SET(report, output_sync_us, st->output_sync_us);
    ATTRIBUTION_SET(report, download_us, st->download_us);
    ATTRIBUTION_SET(report, guard_us, st->guard_us);
    ATTRIBUTION_SET(report, publish_us, st->publish_us);
    ATTRIBUTION_SET(report, alloc_us, st->alloc_us);
    ATTRIBUTION_SET(report, total_us, st->total_us);
    ATTRIBUTION_SET(report, clock_us, st->clock_us);
}

static void group_report_fill(rk_group_report *report, const rk_run_state *st,
                              uint32_t nrepeat, uint32_t completed,
                              uint32_t fail_repeat, uint32_t fail_action,
                              uint32_t task_count) {
    GROUP_REPORT_SET(report, repeat_count, nrepeat);
    GROUP_REPORT_SET(report, repeat_completed, completed);
    GROUP_REPORT_SET(report, fail_repeat, fail_repeat);
    GROUP_REPORT_SET(report, fail_action_id, fail_action);
    GROUP_REPORT_SET(report, task_count, task_count);
    GROUP_REPORT_SET(report, submit_count, st->submit_count);
    GROUP_REPORT_SET(report, upload_calls, st->upload_calls);
    GROUP_REPORT_SET(report, upload_bytes, st->upload_bytes);
    GROUP_REPORT_SET(report, download_calls, st->download_calls);
    GROUP_REPORT_SET(report, download_bytes, st->download_bytes);
    GROUP_REPORT_SET(report, validate_us, st->validate_us);
    GROUP_REPORT_SET(report, pack_us, st->pack_us);
    GROUP_REPORT_SET(report, emit_us, st->emit_us);
    GROUP_REPORT_SET(report, command_us, st->command_us);
    GROUP_REPORT_SET(report, submit_alloc_us, st->submit_alloc_us);
    GROUP_REPORT_SET(report, submit_wait_us, st->submit_wait_us);
    GROUP_REPORT_SET(report, output_sync_us, st->output_sync_us);
    GROUP_REPORT_SET(report, download_us, st->download_us);
    GROUP_REPORT_SET(report, guard_us, st->guard_us);
    GROUP_REPORT_SET(report, publish_us, st->publish_us);
    GROUP_REPORT_SET(report, alloc_us, st->alloc_us);
    GROUP_REPORT_SET(report, total_us, st->total_us);
    GROUP_REPORT_SET(report, clock_us, st->clock_us);
}

uint64_t rk_workspace_host_bytes(const rk_workspace *ws, uint32_t capacity) {
    if (!ws) return 0;
    uint64_t total = (uint64_t)ws->staging_stride * (uint64_t)capacity;
    total += (uint64_t)ws->tile_scratch_bytes;
    /* The emission array is per-submit-task and resident for the workspace's
     * lifetime; it is part of the Host footprint the plan budget must cover. */
    total += (uint64_t)ws->max_submit_tasks * (uint64_t)sizeof(co_program);
    total += (uint64_t)ws->noutputs *
             (uint64_t)(sizeof(size_t) + sizeof(uint32_t));
    return total;
}

/* Acquires the single-run gate, then observes poison/context.  Acquiring busy
 * first means a failing run can never race a concurrent poison write.  Both
 * entry points must call this before touching any device state. */
static int run_gate(rk_workspace *ws, rk_run_state *st, rk_diag *diag) {
    double t0 = now_us();
    if (atomic_exchange_explicit(&ws->busy, 1, memory_order_acq_rel)) {
        rk_diag_set(diag, RK_E_BUSY, "run", "workspace is already running",
                    "", UINT32_MAX, 0);
        return RK_E_BUSY;
    }
    if (st) st->validate_us += now_us() - t0;
    if (atomic_load_explicit(&ws->poisoned, memory_order_acquire)) {
        atomic_store_explicit(&ws->busy, 0, memory_order_release);
        rk_diag_set(diag, RK_E_POISONED, "run",
                    "workspace is poisoned after an unclassified device error",
                    "", UINT32_MAX, 0);
        return RK_E_POISONED;
    }
    int context_rc = rocket_context_faulted(ws->fd);
    if (context_rc != 0) {
        int code = context_rc > 0 ? RK_E_POISONED : RK_E_DEVICE;
        atomic_store_explicit(&ws->poisoned, 1, memory_order_release);
        atomic_store_explicit(&ws->busy, 0, memory_order_release);
        rk_diag_set(diag, code, "run",
                    context_rc > 0
                        ? "provider context is quarantined; refusing submit"
                        : "provider context state is unavailable",
                    "", UINT32_MAX, context_rc);
        return code;
    }
    return 0;
}

/* Validates caller bindings for one repeat: arity, roles, dtype/rank/shape/
 * nbytes, input finiteness and host alias rules.  Every check that
 * ``rk_workspace_run`` performed per call is still performed per repeat, so the
 * group entry does not weaken any input gate. */
static int validate_repeat_bindings(rk_workspace *ws, const rk_binding *inputs,
                                    uint32_t ninputs, const rk_binding *outputs,
                                    uint32_t noutputs, rk_diag *diag) {
    uint32_t input_values = 0, output_values = 0;
    for (uint32_t i = 0; i < ws->plan->nvalues; i++) {
        if (ws->plan->values[i].role == RK_ROLE_INPUT) input_values++;
        if (ws->plan->values[i].role == RK_ROLE_OUTPUT) output_values++;
    }
    if (ninputs != input_values || noutputs != output_values ||
        (ninputs && !inputs) || (noutputs && !outputs)) {
        rk_diag_set(diag, RK_E_SHAPE, "run", "input/output binding count mismatch",
                    "", UINT32_MAX, 0);
        return RK_E_SHAPE;
    }
    for (uint32_t i = 0; i < ws->plan->nvalues; i++) {
        const rk_value_desc *v = &ws->plan->values[i];
        if (v->role != RK_ROLE_INPUT && v->role != RK_ROLE_OUTPUT) continue;
        const rk_binding *b =
            find_binding(v->role == RK_ROLE_INPUT ? inputs : outputs,
                         v->role == RK_ROLE_INPUT ? ninputs : noutputs, i);
        int rc = binding_matches(v, b, diag, "run");
        if (rc) return rc;
        if (v->role == RK_ROLE_INPUT) {
            rc = check_finite_binding(b, diag);
            if (rc) return rc;
        }
    }
    for (uint32_t i = 0; i < ninputs; i++) {
        for (uint32_t j = 0; j < noutputs; j++) {
            if (ranges_overlap(inputs[i].data, inputs[i].nbytes,
                               outputs[j].data, outputs[j].nbytes)) {
                rk_diag_set(diag, RK_E_ALIAS, "run",
                            "input and output host ranges overlap", "", UINT32_MAX, 0);
                return RK_E_ALIAS;
            }
        }
    }
    for (uint32_t i = 0; i < noutputs; i++)
        for (uint32_t j = 0; j < i; j++)
            if (ranges_overlap(outputs[i].data, outputs[i].nbytes,
                               outputs[j].data, outputs[j].nbytes)) {
                rk_diag_set(diag, RK_E_ALIAS, "run", "output host ranges overlap",
                            "", UINT32_MAX, 0);
                return RK_E_ALIAS;
            }
    return 0;
}

/* Reserves this repeat's staging region inside the shared arena.  Offsets follow
 * the caller's output order and every binding has already been proven to match
 * the plan value's nbytes, so the block always fits. */
static void assign_staging_offsets(rk_workspace *ws, const rk_binding *outputs,
                                   uint32_t noutputs) {
    size_t offset = 0;
    for (uint32_t i = 0; i < noutputs; i++) {
        ws->staging_offsets[i] = offset;
        offset += (size_t)outputs[i].nbytes;
    }
}

/* ---- Linear schedule for one repeat: host preparation, blocking submits and
 * downloads execute in the verified action order.  A task group is flushed as
 * one blocking submit when its last task is reached.  Downloads are staged; the
 * caller entry point performs the guard gate and the publication. ---- */
static int run_repeat(rk_workspace *ws, const rk_binding *inputs,
                      uint32_t ninputs, const rk_binding *outputs,
                      uint32_t noutputs, rk_run_state *st, rk_diag *diag) {
    double t0 = now_us();
    int rc = validate_repeat_bindings(ws, inputs, ninputs, outputs, noutputs,
                                      diag);
    st->validate_us += now_us() - t0;
    if (rc) return rc;

    memset(st->tile_count, 0, sizeof(*st->tile_count) * st->noutputs);
    memset(st->staging, 0, st->staging_stride);
    assign_staging_offsets(ws, outputs, noutputs);
    char *staging_base = (char *)st->staging;

    for (uint32_t i = 0; i < ws->plan->nactions; i++) {
        const rk_action_desc *a = &ws->plan->actions[i];
        if (a->kind == RK_ACTION_UPLOAD) {
            const rk_value_desc *v = rk_plan_value(ws->plan, a->source_binding);
            if (!v) {
                rk_diag_set(diag, RK_E_SCHEMA, "run", "upload binding missing",
                            "", a->id, 0);
                return RK_E_SCHEMA;
            }
            if (v->role == RK_ROLE_CONSTANT) continue; /* uploaded once at create */
            const rk_binding *b = find_binding(inputs, ninputs, a->source_binding);
            if (!b) {
                rk_diag_set(diag, RK_E_SHAPE, "run",
                            "input binding is absent or duplicated", "", a->id, 0);
                return RK_E_SHAPE;
            }
            uint64_t bytes = 0;
            double u0 = now_us();
            int native = upload_action(ws, a, b, &bytes, diag);
            st->pack_us += now_us() - u0;
            if (native) return native;
            st->upload_calls++;
            st->upload_bytes += bytes;
            continue;
        }
        if (a->kind == RK_ACTION_NPU_TASK) {
            uint32_t sid = ws->plan->task_submit[a->id];
            if (sid == UINT32_MAX) {
                rk_diag_set(diag, RK_E_SCHEMA, "run",
                            "NPU task is not assigned to a submit", "", a->id, 0);
                return RK_E_SCHEMA;
            }
            const rk_submit_desc *submit = &ws->plan->submits[sid];
            uint32_t index = a->id - submit->task_ids[0];
            double e0 = now_us();
            int native = rk_emit_action(ws->plan, ws->views, a->id,
                                        &st->batch[index], diag);
            st->emit_us += now_us() - e0;
            if (native) return native;
            if (index + 1u != submit->ntask_ids) continue;
            co_execute_profile profile;
            memset(&profile, 0, sizeof(profile));
            native = co_execute_profiled(ws->fd, st->batch, submit->ntask_ids,
                    submit->mode == RK_SUBMIT_CHAIN ? CO_CHAIN : CO_SEPARATE,
                    &ws->execute_workspace, &profile);
            st->submit_wait_us += profile.submit_wait_us;
            st->output_sync_us += profile.output_sync_us;
            st->command_us += profile.command_us;
            st->submit_alloc_us += profile.allocation_us;
            if (native) {
                atomic_store_explicit(&ws->poisoned, 1, memory_order_release);
                rk_diag_set(diag, RK_E_DEVICE, "run",
                            "submission failed; workspace is poisoned", "",
                            submit->task_ids[0], native);
                return RK_E_DEVICE;
            }
            st->submit_count++;
            continue;
        }
        if (a->kind == RK_ACTION_DOWNLOAD) {
            if (a->ninputs != 1) {
                rk_diag_set(diag, RK_E_SCHEMA, "run", "download arity invalid",
                            "", a->id, 0);
                return RK_E_SCHEMA;
            }
            double d0 = now_us();
            const rk_publish_desc *pub = NULL;
            for (uint32_t q = 0; q < ws->plan->npublish; q++)
                if (ws->plan->publish[q].binding_id == a->source_binding &&
                    ws->plan->publish[q].view_id == a->inputs[0])
                    pub = &ws->plan->publish[q];
            const rk_value_desc *value = rk_plan_value(ws->plan, a->source_binding);
            co_view *view = &ws->views[a->inputs[0]];
            uint32_t slot = UINT32_MAX;
            for (uint32_t q = 0; q < noutputs; q++)
                if (outputs[q].id == a->source_binding) { slot = q; break; }
            if (!pub || !value || slot == UINT32_MAX) {
                rk_diag_set(diag, RK_E_SCHEMA, "run",
                            "download/publish mapping invalid", "", a->id, 0);
                return RK_E_SCHEMA;
            }
            /* The tile goes straight into the workspace scratch: it is copied
             * into the staging block before the next download can run, so the
             * per-download allocation is pure overhead (A2). */
            if ((size_t)view->tensor.bytes > ws->tile_scratch_bytes) {
                rk_diag_set(diag, RK_E_SCHEMA, "run",
                            "download tile exceeds the preallocated scratch",
                            "", a->id, 0);
                return RK_E_SCHEMA;
            }
            void *tile = ws->tile_scratch;
            int native = co_download(view, tile);
            if (native) {
                atomic_store_explicit(&ws->poisoned, 1, memory_order_release);
                rk_diag_set(diag, RK_E_DEVICE, "run",
                            "output download failed; workspace is poisoned",
                            "", a->id, native);
                st->download_us += now_us() - d0;
                return RK_E_DEVICE;
            }
            st->download_calls++;
            st->download_bytes += (uint64_t)view->tensor.bytes;
            const uint16_t *tile_data = (const uint16_t *)tile;
            /* co_download fills the logical rows*cols elements in row-major
             * order; the rest of the physical tile is padding that the Host
             * never wrote.  Scanning the physical extent would read
             * uninitialised memory and produce random false failures.  Padding
             * validity is a device-side property and is checked separately by
             * guard/sentinel campaigns, not here. */
            const uint64_t logical = (uint64_t)view->tensor.rows * view->tensor.cols;
            if (!rk_f16_all_finite(tile_data, logical)) {
                rk_diag_set(diag, RK_E_NUMERIC_CONTRACT, "run",
                            "downloaded output tile is not finite; refusing to "
                            "publish", "", a->id, 0);
                st->download_us += now_us() - d0;
                return RK_E_NUMERIC_CONTRACT;
            }
            rk_publish_rows(staging_base + ws->staging_offsets[slot], tile_data,
                            pub->rows, pub->cols, pub->dst_row_offset,
                            pub->dst_col_offset, value->shape[1]);
            st->tile_count[slot]++;
            st->download_us += now_us() - d0;
            continue;
        }
        rk_diag_set(diag, RK_E_UNSUPPORTED, "run",
                    "unsupported action in the verified schedule", "", a->id, 0);
        return RK_E_UNSUPPORTED;
    }
    for (uint32_t i = 0; i < noutputs; i++) {
        if (!st->tile_count[i]) {
            rk_diag_set(diag, RK_E_INVALID, "run", "missing staged output",
                        "", UINT32_MAX, 0);
            return RK_E_INVALID;
        }
    }
    return 0;
}

/* Single publication gate: every allocated data BO is checked before any staged
 * byte reaches a caller buffer.  The group entry runs it once for the whole
 * repeat set, which keeps the identical check intensity (all slots, all slack
 * bytes) while removing the per-repeat repetition of the same gate. */
static int publication_gate(rk_workspace *ws, rk_run_state *st, rk_diag *diag) {
    double g0 = now_us();
    for (uint32_t slot = 0; slot < ws->nslots; slot++) {
        int native = co_guard_check(&ws->slots[slot]);
        if (native) {
            st->guard_us += now_us() - g0;
            atomic_store_explicit(&ws->poisoned, 1, memory_order_release);
            rk_diag_set(diag, RK_E_DEVICE, "run",
                        "data BO slack/guard check failed; refusing to publish",
                        "", UINT32_MAX, native);
            return RK_E_DEVICE;
        }
    }
    st->guard_us += now_us() - g0;
    return 0;
}

/* Shared body: gate, schedule, publication gate, publish.  Neither entry point
 * writes a caller report from here, so the report boundary stays explicit. */
static uint32_t plan_task_count(const rk_plan *plan) {
    uint32_t tasks = 0;
    for (uint32_t i = 0; i < plan->nsubmits; i++)
        tasks += plan->submits[i].ntask_ids;
    return tasks;
}

/* Gate, schedule, publication gate and publish for one binding set.
 *
 * Lock ownership is scoped to this function: it releases the workspace busy flag
 * only on the path that actually acquired it.  A caller that is refused because
 * another run holds the lock must never clear that run's ownership, which is why
 * the release cannot live in the public entries. */
static int workspace_run_body(rk_workspace *ws,
                              const rk_binding *inputs, uint32_t ninputs,
                              const rk_binding *outputs, uint32_t noutputs,
                              rk_run_state *st, rk_diag *diag) {
    int result = run_gate(ws, st, diag);
    if (result) return result;   /* lock not held: nothing to release */
    st->task_count = plan_task_count(ws->plan);
    st->staging = ws->staging_arena;
    st->staging_stride = ws->staging_stride;
    st->tile_count = ws->tile_count_scratch;
    st->batch = ws->batch_scratch;
    st->noutputs = ws->noutputs;
    result = run_repeat(ws, inputs, ninputs, outputs, noutputs, st, diag);
    if (!result) {
        result = publication_gate(ws, st, diag);
        if (!result) {
            double p0 = now_us();
            for (uint32_t i = 0; i < noutputs; i++)
                memcpy((void *)outputs[i].data,
                       (char *)ws->staging_arena + ws->staging_offsets[i],
                       (size_t)outputs[i].nbytes);
            st->publish_us += now_us() - p0;
        }
    }
    atomic_store_explicit(&ws->busy, 0, memory_order_release);
    return result;
}

int rk_workspace_run(rk_workspace *ws,
                     const rk_binding *inputs, uint32_t ninputs,
                     const rk_binding *outputs, uint32_t noutputs,
                     rk_run_report *report, rk_diag *diag) {
    rk_diag_clear(diag);
    if (report) run_report_clear(report);
    if (!ws) return RK_E_INVALID;
    rk_run_state st;
    memset(&st, 0, sizeof(st));
    rk_clock_window window = clock_window_open();
    int result = workspace_run_body(ws, inputs, ninputs, outputs, noutputs,
                                    &st, diag);
    clock_window_close(&window, &st);
    if (report) {
        run_report_fill(report, &st, result ? 0 : st.task_count);
        if (result) report->fail_action_id = diag ? diag->action_id : UINT32_MAX;
    }
    return result;   /* workspace_run_body owns the busy release */
}

int rk_workspace_run_ex(rk_workspace *ws,
                        const rk_binding *inputs, uint32_t ninputs,
                        const rk_binding *outputs, uint32_t noutputs,
                        rk_run_attribution *envelope, rk_diag *diag) {
    rk_diag_clear(diag);
    if (envelope) {
        uint32_t size = 0;
        int rc = attribution_size(envelope, "run_ex", diag, &size);
        if (rc) return rc;
        attribution_clear(envelope);
    }
    if (!ws || !envelope) return RK_E_INVALID;
    rk_run_state st;
    memset(&st, 0, sizeof(st));
    rk_clock_window window = clock_window_open();
    int result = workspace_run_body(ws, inputs, ninputs, outputs, noutputs,
                                    &st, diag);
    clock_window_close(&window, &st);
    attribution_fill(envelope, &st, result ? 0 : st.task_count);
    if (result) ATTRIBUTION_SET(envelope, fail_action_id,
                                diag ? diag->action_id : UINT32_MAX);
    return result;   /* workspace_run_body owns the busy release */
}

int rk_workspace_run_group(rk_workspace *ws, const rk_binding *bindings,
                           uint32_t nrepeat, rk_group_report *report,
                           rk_diag *diag) {
    rk_diag_clear(diag);
    if (report) {
        uint32_t size = 0;
        int rc = group_size(report, "run_group", diag, &size);
        if (rc) return rc;
        group_report_clear(report);
    }
    if (!ws || !nrepeat || !report) return RK_E_INVALID;
    const uint32_t set = ws->ninputs + ws->noutputs;
    if (!bindings || !set) {
        rk_diag_set(diag, RK_E_SHAPE, "run_group",
                    "empty binding set for the group entry", "", UINT32_MAX, 0);
        return RK_E_SHAPE;
    }
    rk_run_state st;
    memset(&st, 0, sizeof(st));
    rk_clock_window window = clock_window_open();
    /* Lock ownership starts here: every path below either reaches ``out`` (which
     * releases) or returns before acquiring, exactly like workspace_run_body. */
    int result = run_gate(ws, &st, diag);
    if (result) {
        clock_window_close(&window, &st);
        if (report) group_report_fill(report, &st, nrepeat, 0, UINT32_MAX,
                                      UINT32_MAX, 0);
        return result;   /* lock not held: must not be released */
    }
    st.batch = ws->batch_scratch;
    st.tile_count = ws->tile_count_scratch;
    st.noutputs = ws->noutputs;
    st.staging_stride = ws->staging_stride;

    uint32_t completed = 0, fail_repeat = UINT32_MAX, fail_action = UINT32_MAX;
    uint32_t plan_tasks = 0;
    for (uint32_t i = 0; i < ws->plan->nsubmits; i++)
        plan_tasks += ws->plan->submits[i].ntask_ids;

    if (ws->staging_capacity < nrepeat) {
        double a0 = now_us();
        /* The bound is the *whole* Host footprint, not the staging block alone:
         * the emission array, the download tile and the per-output bookkeeping
         * stay resident next to the arena, so a group that fits the budget only
         * when those are ignored is still refused.  A zero plan budget is not a
         * bypass: the verifier-recomputed Host claim becomes the ceiling, and the
         * ceiling is never tightened below what the single-run configuration
         * already holds. */
        uint64_t budget = ws->plan->host_budget_bytes
                              ? ws->plan->host_budget_bytes
                              : ws->plan->claimed_host_bytes;
        /* No ceiling inflation: the plan's declared Host budget is the bound.  A
         * plan whose single-run footprint already exceeds it cannot grow a repeat
         * arena at all -- that is a plan-package accounting defect to fix by
         * recompiling the plan, not something the runtime may absorb. */
        uint64_t resident_now = rk_workspace_host_bytes(ws, ws->staging_capacity);
        uint64_t ceiling = budget;
        if (ws->staging_stride &&
            (uint64_t)nrepeat > UINT64_MAX / (uint64_t)ws->staging_stride) {
            st.alloc_us += now_us() - a0;
            rk_diag_set(diag, RK_E_BUDGET, "run_group",
                        "group staging size overflows", "", UINT32_MAX, 0);
            result = RK_E_BUDGET;
            goto out;
        }
        uint64_t want_total = rk_workspace_host_bytes(ws, nrepeat);
        if (want_total > ceiling || want_total > (uint64_t)SIZE_MAX) {
            st.alloc_us += now_us() - a0;
            rk_diag_set(diag, RK_E_BUDGET, "run_group",
                        resident_now > ceiling
                            ? "plan Host budget is below its own single-run Host "
                              "footprint; recompile the plan (accounting defect)"
                            : "group Host footprint exceeds the plan Host budget",
                        "", UINT32_MAX, 0);
            result = RK_E_BUDGET;
            goto out;
        }
        uint64_t want_bytes = (uint64_t)ws->staging_stride * (uint64_t)nrepeat;
        size_t want = (size_t)want_bytes;
        void *grown = realloc(ws->staging_arena, want ? want : 1u);
        st.alloc_us += now_us() - a0;
        if (!grown) {
            rk_diag_set(diag, RK_E_NOMEM, "run_group",
                        "out of memory for group staging", "", UINT32_MAX, 0);
            result = RK_E_NOMEM;
            goto out;
        }
        ws->staging_arena = grown;
        ws->staging_capacity = nrepeat;
    }

    for (uint32_t r = 0; r < nrepeat; r++) {
        const rk_binding *rin = bindings + (size_t)r * set;
        const rk_binding *rout = rin + ws->ninputs;
        st.staging = (char *)ws->staging_arena +
                     ws->staging_stride * (size_t)r;
        result = run_repeat(ws, rin, ws->ninputs, rout, ws->noutputs, &st, diag);
        if (result) {
            fail_repeat = r;
            fail_action = diag && diag->action_id != UINT32_MAX
                              ? diag->action_id : UINT32_MAX;
            break;
        }
        completed++;
    }
    if (result) goto out;
    result = publication_gate(ws, &st, diag);
    if (result) {
        fail_repeat = UINT32_MAX;
        goto out;
    }
    {
        double p0 = now_us();
        for (uint32_t r = 0; r < nrepeat; r++) {
            const rk_binding *rout = bindings + (size_t)r * set + ws->ninputs;
            const char *block = (const char *)ws->staging_arena +
                                ws->staging_stride * (size_t)r;
            for (uint32_t i = 0; i < ws->noutputs; i++)
                memcpy((void *)rout[i].data, block + ws->staging_offsets[i],
                       (size_t)rout[i].nbytes);
        }
        st.publish_us += now_us() - p0;
    }
    result = 0;
out:
    clock_window_close(&window, &st);
    if (report)
        /* ``task_count`` counts the plan tasks of every repeat that completed;
         * a partially completed repeat set therefore reports what actually ran
         * instead of pretending nothing happened. */
        group_report_fill(report, &st, nrepeat, completed, fail_repeat,
                          fail_action, plan_tasks * completed);
    atomic_store_explicit(&ws->busy, 0, memory_order_release);
    return result;
}

int rk_workspace_destroy(rk_workspace *ws, rk_diag *diag) {
    rk_diag_clear(diag);
    if (!ws) return 0;
    if (atomic_exchange_explicit(&ws->busy, 1, memory_order_acq_rel)) {
        rk_diag_set(diag, RK_E_BUSY, "workspace_destroy",
                    "destroy raced a running workspace", "", UINT32_MAX, 0);
        return RK_E_BUSY;
    }
    if (atomic_load_explicit(&ws->poisoned, memory_order_acquire)) {
        /* Without a provider-confirmed quiescence protocol we do not release BOs
         * that the device may still be touching.  The handle stays caller-owned. */
        atomic_store_explicit(&ws->busy, 0, memory_order_release);
        rk_diag_set(diag, RK_E_POISONED, "workspace_destroy",
                    "workspace is poisoned; process-level recovery required",
                    "", UINT32_MAX, 0);
        return RK_E_POISONED;
    }
    if (ws->execute_workspace_ready)
        co_execute_workspace_destroy(&ws->execute_workspace);
    for (uint32_t i = 0; i < ws->nslots; i++) co_buffer_free(&ws->slots[i]);
    free(ws->slots);
    free(ws->views);
    free(ws->staging_offsets);
    free(ws->tile_count_scratch);
    free(ws->batch_scratch);
    free(ws->staging_arena);
    free(ws->tile_scratch);
    rk_plan_free(ws->plan);
    free(ws);
    return 0;
}
