// SPDX-License-Identifier: GPL-3.0-or-later
#define _POSIX_C_SOURCE 200809L
#include "rk_plan_internal.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static int map_native(int rc) {
    if (rc == 0) return 0;
    if (rc == ROCKET_E_NOMEM) return RK_E_NOMEM;
    if (rc == ROCKET_E_SHAPE) return RK_E_SHAPE;
    if (rc == ROCKET_E_UNSUPPORTED) return RK_E_UNSUPPORTED;
    return RK_E_DEVICE;
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
    for (uint32_t r = 0; r < dst->tensor.rows; r++) {
        for (uint32_t c = 0; c < dst->tensor.cols; c++) {
            uint32_t sr = transpose ? a->source_row_offset + c
                                    : a->source_row_offset + r;
            uint32_t sc = transpose ? a->source_col_offset + r
                                    : a->source_col_offset + c;
            size_t source_index = (size_t)sr * a->source_row_stride + sc;
            base[co_index(&dst->tensor, r, c)] = src[source_index];
        }
    }
    rc = rocket_bo_fini(ws->fd, &dst->buffer->bo);
    if (rc) return fail_native(diag, "upload", "rocket_bo_fini failed", a->id, rc);
    if (bytes_out) *bytes_out += binding->nbytes;
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
    if ((nconstants && !constants) || nconstants > plan->nvalues) {
        rk_diag_set(diag, RK_E_SCHEMA, "workspace_create",
                    "constant pointer/count mismatch", "", UINT32_MAX, 0);
        return RK_E_SCHEMA;
    }
    rk_workspace *ws = calloc(1, sizeof(*ws));
    if (!ws) return RK_E_NOMEM;
    ws->fd = fd;
    ws->busy = 0;
    ws->poisoned = 0;
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
    rk_plan_free(ws->plan);
    free(ws);
    return diag && diag->code ? diag->code : RK_E_INVALID;
}

int rk_workspace_run(rk_workspace *ws,
                     const rk_binding *inputs, uint32_t ninputs,
                     const rk_binding *outputs, uint32_t noutputs,
                     rk_run_report *report, rk_diag *diag) {
    rk_diag_clear(diag);
    if (report) {
        memset(report, 0, sizeof(*report));
        report->abi_version = RK_ABI_VERSION;
        report->struct_size = (uint32_t)sizeof(*report);
        report->fail_action_id = UINT32_MAX;
        report->numeric_mode = 1;
    }
    if (!ws) return RK_E_INVALID;
    if (ws->poisoned) {
        rk_diag_set(diag, RK_E_POISONED, "run",
                    "workspace is poisoned after an unclassified device error",
                    "", UINT32_MAX, 0);
        return RK_E_POISONED;
    }
    if (atomic_exchange_explicit(&ws->busy, 1, memory_order_acq_rel)) {
        rk_diag_set(diag, RK_E_BUSY, "run", "workspace is already running",
                    "", UINT32_MAX, 0);
        return RK_E_BUSY;
    }
    int result = RK_E_INVALID;
    uint32_t input_values = 0, output_values = 0;
    for (uint32_t i = 0; i < ws->plan->nvalues; i++) {
        if (ws->plan->values[i].role == RK_ROLE_INPUT) input_values++;
        if (ws->plan->values[i].role == RK_ROLE_OUTPUT) output_values++;
    }
    if (ninputs != input_values || noutputs != output_values ||
        (ninputs && !inputs) || (noutputs && !outputs)) {
        rk_diag_set(diag, RK_E_SHAPE, "run", "input/output binding count mismatch",
                    "", UINT32_MAX, 0);
        result = RK_E_SHAPE;
        goto out;
    }
    for (uint32_t i = 0; i < ws->plan->nvalues; i++) {
        const rk_value_desc *v = &ws->plan->values[i];
        if (v->role != RK_ROLE_INPUT && v->role != RK_ROLE_OUTPUT) continue;
        const rk_binding *b = find_binding(v->role == RK_ROLE_INPUT ? inputs : outputs,
                                           v->role == RK_ROLE_INPUT ? ninputs : noutputs,
                                           i);
        int rc = binding_matches(v, b, diag, "run");
        if (rc) { result = rc; goto out; }
        if (v->role == RK_ROLE_INPUT) {
            rc = check_finite_binding(b, diag);
            if (rc) { result = rc; goto out; }
        }
    }
    for (uint32_t i = 0; i < ninputs; i++) {
        for (uint32_t j = 0; j < noutputs; j++) {
            if (ranges_overlap(inputs[i].data, inputs[i].nbytes,
                               outputs[j].data, outputs[j].nbytes)) {
                rk_diag_set(diag, RK_E_ALIAS, "run",
                            "input and output host ranges overlap", "", UINT32_MAX, 0);
                result = RK_E_ALIAS;
                goto out;
            }
        }
    }
    for (uint32_t i = 0; i < noutputs; i++)
        for (uint32_t j = 0; j < i; j++)
            if (ranges_overlap(outputs[i].data, outputs[i].nbytes,
                               outputs[j].data, outputs[j].nbytes)) {
                rk_diag_set(diag, RK_E_ALIAS, "run", "output host ranges overlap",
                            "", UINT32_MAX, 0);
                result = RK_E_ALIAS;
                goto out;
            }

    uint64_t upload_calls = 0, upload_bytes = 0;
    result = upload_role(ws, RK_ROLE_INPUT, inputs, ninputs,
                         &upload_calls, &upload_bytes, diag);
    if (result) goto out;

    double pack_us = 0.0, emit_us = 0.0, submit_us = 0.0, sync_us = 0.0;
    for (uint32_t s = 0; s < ws->plan->nsubmits; s++) {
        const rk_submit_desc *submit = &ws->plan->submits[s];
        co_program *programs = calloc(submit->ntask_ids, sizeof(*programs));
        if (!programs) {
            rk_diag_set(diag, RK_E_NOMEM, "run", "out of memory for programs",
                        "", UINT32_MAX, 0);
            result = RK_E_NOMEM;
            goto out;
        }
        double t0 = now_us();
        for (uint32_t t = 0; t < submit->ntask_ids; t++) {
            int rc = rk_emit_action(ws->plan, ws->views, submit->task_ids[t],
                                    &programs[t], diag);
            if (rc) {
                free(programs);
                result = rc;
                goto out;
            }
            if (programs[t].count > CO_MAX_WORDS) {
                free(programs);
                rk_diag_set(diag, RK_E_SHAPE, "run", "program word count overflow",
                            "", submit->task_ids[t], 0);
                result = RK_E_SHAPE;
                goto out;
            }
        }
        double t1 = now_us();
        co_execute_profile profile;
        memset(&profile, 0, sizeof(profile));
        int native = co_execute_profiled(ws->fd, programs, submit->ntask_ids,
                submit->mode == RK_SUBMIT_CHAIN ? CO_CHAIN : CO_SEPARATE,
                &ws->execute_workspace, &profile);
        emit_us += t1 - t0;
        submit_us += profile.submit_wait_us;
        sync_us += profile.output_sync_us;
        free(programs);
        if (native) {
            ws->poisoned = 1;
            rk_diag_set(diag, RK_E_DEVICE, "run",
                        "submission failed; workspace is poisoned", "",
                        submit->task_ids[0], native);
            if (report) report->fail_action_id = submit->task_ids[0];
            result = RK_E_DEVICE;
            goto out;
        }
    }

    /* Download into host staging and publish only after every output succeeds. */
    void **staging = calloc(noutputs ? noutputs : 1u, sizeof(void *));
    if (!staging) {
        rk_diag_set(diag, RK_E_NOMEM, "run", "out of memory for output staging",
                    "", UINT32_MAX, 0);
        result = RK_E_NOMEM;
        goto out;
    }
    uint64_t download_calls = 0, download_bytes = 0;
    for (uint32_t p = 0; p < ws->plan->npublish; p++) {
        const rk_publish_desc *pub = &ws->plan->publish[p];
        const rk_value_desc *value = rk_plan_value(ws->plan, pub->binding_id);
        co_view *view = &ws->views[pub->view_id];
        void *tmp = malloc(value->nbytes ? (size_t)value->nbytes : 1u);
        if (!tmp) {
            rk_diag_set(diag, RK_E_NOMEM, "run", "out of memory for output",
                        "", UINT32_MAX, 0);
            result = RK_E_NOMEM;
            goto downloads_done;
        }
        int native = co_download(view, tmp);
        if (native) {
            free(tmp);
            ws->poisoned = 1;
            rk_diag_set(diag, RK_E_DEVICE, "run",
                        "output download failed; workspace is poisoned",
                        "", UINT32_MAX, native);
            result = RK_E_DEVICE;
            goto downloads_done;
        }
        download_calls++;
        download_bytes += value->nbytes;
        for (uint32_t i = 0; i < noutputs; i++) {
            if (outputs[i].id == pub->binding_id) {
                staging[i] = tmp;
                tmp = NULL;
                break;
            }
        }
        free(tmp);
    }
    for (uint32_t i = 0; i < noutputs; i++) {
        if (!staging[i]) {
            rk_diag_set(diag, RK_E_INVALID, "run", "missing staged output",
                        "", UINT32_MAX, 0);
            result = RK_E_INVALID;
            goto downloads_done;
        }
        memcpy((void *)outputs[i].data, staging[i], (size_t)outputs[i].nbytes);
    }
    result = 0;
downloads_done:
    for (uint32_t i = 0; i < noutputs; i++) free(staging[i]);
    free(staging);
    if (result) goto out;

    if (report) {
        report->task_count = 0;
        report->submit_count = ws->plan->nsubmits;
        for (uint32_t i = 0; i < ws->plan->nsubmits; i++)
            report->task_count += ws->plan->submits[i].ntask_ids;
        report->upload_calls = upload_calls;
        report->upload_bytes = upload_bytes;
        report->download_calls = download_calls;
        report->download_bytes = download_bytes;
        report->pack_us = pack_us;
        report->emit_us = emit_us;
        report->submit_wait_us = submit_us;
        report->output_sync_us = sync_us;
    }
out:
    atomic_store_explicit(&ws->busy, 0, memory_order_release);
    return result;
}

int rk_workspace_destroy(rk_workspace *ws, rk_diag *diag) {
    rk_diag_clear(diag);
    if (!ws) return 0;
    if (atomic_load_explicit(&ws->busy, memory_order_acquire)) {
        rk_diag_set(diag, RK_E_BUSY, "workspace_destroy",
                    "destroy raced a running workspace", "", UINT32_MAX, 0);
        return RK_E_BUSY;
    }
    if (ws->poisoned) {
        /* Without a provider-confirmed quiescence protocol we do not release BOs
         * that the device may still be touching.  The handle stays caller-owned. */
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
    rk_plan_free(ws->plan);
    free(ws);
    return 0;
}
