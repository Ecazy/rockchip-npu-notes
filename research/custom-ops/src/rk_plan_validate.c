// SPDX-License-Identifier: GPL-3.0-or-later
#define _POSIX_C_SOURCE 200809L
#include "rk_plan_internal.h"
#include "custom_ops.h"
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_pow2(uint32_t v) { return v && !(v & (v - 1u)); }
static int valid_role(uint32_t role) { return role <= RK_ROLE_IDENTITY; }
static int valid_layout(uint32_t layout) {
    return layout == RK_LAYOUT_CUBE_F16 || layout == RK_LAYOUT_WEIGHT_F16 ||
           layout == RK_LAYOUT_LINEAR_F16;
}
static int valid_kind(uint32_t kind) { return kind >= RK_ACTION_HOST_PACK && kind <= RK_ACTION_COMMIT; }

void rk_diag_clear(rk_diag *diag) {
    if (!diag) return;
    memset(diag, 0, sizeof(*diag));
    diag->abi_version = RK_ABI_VERSION;
    diag->struct_size = (uint32_t)sizeof(*diag);
    diag->code = 0;
    diag->native_status = 0;
    diag->has_action = 0;
    diag->action_id = 0;
}

void rk_diag_set(rk_diag *diag, int code, const char *stage,
                 const char *message, const char *source,
                 uint32_t action_id, int native_status) {
    if (!diag) return;
    rk_diag_clear(diag);
    diag->code = code;
    diag->native_status = native_status;
    if (action_id != UINT32_MAX) {
        diag->action_id = action_id;
        diag->has_action = 1;
    }
    if (stage) snprintf(diag->stage, sizeof(diag->stage), "%s", stage);
    if (message) snprintf(diag->message, sizeof(diag->message), "%s", message);
    if (source) snprintf(diag->source, sizeof(diag->source), "%s", source);
}

static int fail(rk_diag *diag, int code, const char *stage, const char *message) {
    rk_diag_set(diag, code, stage, message, "", UINT32_MAX, 0);
    return code;
}

static int struct_ok(uint32_t abi, uint32_t size, size_t expected) {
    return abi == RK_ABI_VERSION && size >= expected;
}

static uint32_t round_up_u32(uint32_t value, uint32_t multiple) {
    return (value + multiple - 1u) / multiple * multiple;
}

static int tensor_valid(const rk_tensor_desc *t) {
    if (!t || !struct_ok(t->abi_version, t->struct_size, sizeof(*t))) return 0;
    if (t->dtype != RK_DTYPE_F16 || !valid_layout(t->layout)) return 0;
    if (!t->rows || !t->cols || !t->physical_rows || !t->physical_cols) return 0;
    if (t->rows > t->physical_rows || t->cols > t->physical_cols) return 0;
    if (t->physical_rows > 256 || t->physical_cols > 65520) return 0;
    /* Physical geometry is not a free tuning knob: it is exactly what
     * co_tensor_matrix / co_tensor_weight / co_tensor_linear produce.  Accepting
     * an arbitrary padding would make the addressing of two views of one buffer
     * disagree, which is what the coverage check below must rely on. */
    if (t->layout == RK_LAYOUT_LINEAR_F16) {
        if (t->atom != 1 || t->physical_rows != 1 || t->physical_cols != t->cols)
            return 0;
    } else if (t->layout == RK_LAYOUT_WEIGHT_F16) {
        if (t->atom != 32 || t->physical_rows != round_up_u32(t->rows, 32) ||
            t->physical_cols != round_up_u32(t->cols, 32))
            return 0;
    } else {
        uint32_t rows = t->rows < 12u ? 12u : t->rows;
        if (t->atom != 8 || t->physical_cols > CO_MAX_K ||
            t->physical_rows != round_up_u32(rows, 4) ||
            t->physical_cols != round_up_u32(t->cols, 32))
            return 0;
    }
    if (t->physical_cols % t->atom) return 0;
    if ((uint64_t)t->physical_rows * t->physical_cols > UINT64_MAX / 2u) return 0;
    if (t->bytes != (uint64_t)t->physical_rows * t->physical_cols * 2u) return 0;
    return 1;
}

/* Two views address a buffer identically only when the producer's packing
 * function (co_index) and the reader's use the same physical shape. */
static int same_addressing(const rk_tensor_desc *a, const rk_tensor_desc *b) {
    return a->layout == b->layout && a->atom == b->atom &&
           a->physical_rows == b->physical_rows &&
           a->physical_cols == b->physical_cols;
}

static const rk_value_desc *value_at(const rk_plan_desc *desc, uint32_t id) {
    if (!desc || id >= desc->nvalues || !desc->values) return NULL;
    return &desc->values[id];
}
static const rk_buffer_desc *buffer_at(const rk_plan_desc *desc, uint32_t id) {
    if (!desc || id >= desc->nbuffers || !desc->buffers) return NULL;
    return &desc->buffers[id];
}
static const rk_view_desc *view_at(const rk_plan_desc *desc, uint32_t id) {
    if (!desc || id >= desc->nviews || !desc->views) return NULL;
    return &desc->views[id];
}
static const rk_action_desc *action_at(const rk_plan_desc *desc, uint32_t id) {
    if (!desc || id >= desc->nactions || !desc->actions) return NULL;
    return &desc->actions[id];
}

static int validate_upload(const rk_plan_desc *desc, const rk_action_desc *a, rk_diag *diag) {
    if (a->ninputs != 0 || a->noutputs != 1 || !a->outputs)
        return fail(diag, RK_E_SHAPE, "validate", "upload requires zero inputs and one output view");
    const rk_view_desc *out = view_at(desc, a->outputs[0]);
    const rk_value_desc *src = value_at(desc, a->source_binding);
    if (!out || !src)
        return fail(diag, RK_E_SCHEMA, "validate", "upload references missing view or binding");
    if (src->role != RK_ROLE_INPUT && src->role != RK_ROLE_CONSTANT)
        return fail(diag, RK_E_SHAPE, "validate", "upload source binding must be input or constant");
    if (src->dtype != RK_DTYPE_F16 || src->rank > 2)
        return fail(diag, RK_E_NUMERIC_CONTRACT, "validate",
                    "upload source must be rank-0/1/2 FP16");
    if (!tensor_valid(&out->tensor))
        return fail(diag, RK_E_SHAPE, "validate", "upload destination tensor is invalid");
    uint32_t transpose = (a->flags & RK_ACTION_FLAG_TRANSPOSE_SOURCE) != 0;
    uint32_t broadcast_scalar = (a->flags & RK_ACTION_FLAG_BROADCAST_SCALAR) != 0;
    uint32_t broadcast_channel = (a->flags & RK_ACTION_FLAG_BROADCAST_CHANNEL) != 0;
    if (broadcast_scalar && broadcast_channel)
        return fail(diag, RK_E_SCHEMA, "validate", "conflicting broadcast flags");
    if (broadcast_scalar || broadcast_channel) {
        if (src->role != RK_ROLE_CONSTANT)
            return fail(diag, RK_E_UNSUPPORTED, "validate",
                        "only constants may be broadcast-expanded; runtime tensor "
                        "broadcast is not supported");
        if (transpose || a->source_row_offset || a->source_row_stride)
            return fail(diag, RK_E_SHAPE, "validate",
                        "broadcast upload cannot transpose or use a source row offset/stride");
        if (broadcast_scalar) {
            if (a->source_col_offset || src->rank != 0 || src->shape[0] ||
                src->shape[1] || src->nbytes != 2u)
                return fail(diag, RK_E_SHAPE, "validate",
                            "scalar broadcast source must be rank-0 with one FP16 element");
        } else {
            if (src->rank != 1 || src->shape[1] ||
                src->nbytes != (uint64_t)src->shape[0] * 2u)
                return fail(diag, RK_E_SHAPE, "validate",
                            "channel broadcast source must be a rank-1 FP16 vector");
            if ((uint64_t)a->source_col_offset + out->tensor.cols > src->shape[0])
                return fail(diag, RK_E_SHAPE, "validate",
                            "channel broadcast window exceeds source N");
        }
        return 0;
    }
    if (src->rank != 2)
        return fail(diag, RK_E_UNSUPPORTED, "validate",
                    "rank-0/1 uploads require an explicit broadcast flag");
    if (a->source_row_stride != src->shape[1])
        return fail(diag, RK_E_SHAPE, "validate",
                    "upload source_row_stride must equal the binding row length");
    uint64_t rows = out->tensor.rows, cols = out->tensor.cols;
    uint64_t sr = transpose ? cols : rows;
    uint64_t sc = transpose ? rows : cols;
    if ((uint64_t)a->source_row_offset + sr > src->shape[0] ||
        (uint64_t)a->source_col_offset + sc > src->shape[1])
        return fail(diag, RK_E_SHAPE, "validate", "upload source region exceeds binding shape");
    /* Recompute the greatest element index actually read by the packer and
     * prove it is inside the borrowed binding.  Never trust the planner. */
    uint64_t last_row = (uint64_t)a->source_row_offset + sr - 1u;
    uint64_t last_col = (uint64_t)a->source_col_offset + sc - 1u;
    if (last_row > (UINT64_MAX - last_col) / (uint64_t)a->source_row_stride)
        return fail(diag, RK_E_SHAPE, "validate", "upload source index overflows");
    uint64_t last_index = last_row * (uint64_t)a->source_row_stride + last_col;
    uint64_t elements = src->nbytes / 2u;
    if (src->nbytes % 2u || last_index >= elements)
        return fail(diag, RK_E_SHAPE, "validate",
                    "upload source element range exceeds the binding nbytes");
    return 0;
}

static int validate_download(const rk_plan_desc *desc, const rk_action_desc *a, rk_diag *diag) {
    if (a->ninputs != 1 || a->noutputs != 0 || !a->inputs)
        return fail(diag, RK_E_SHAPE, "validate", "download requires one input view");
    const rk_view_desc *src = view_at(desc, a->inputs[0]);
    const rk_value_desc *dst = value_at(desc, a->source_binding);
    if (!src || !dst)
        return fail(diag, RK_E_SCHEMA, "validate", "download references missing view or output binding");
    if (dst->role != RK_ROLE_OUTPUT)
        return fail(diag, RK_E_SHAPE, "validate", "download target binding must be output");
    if (!tensor_valid(&src->tensor))
        return fail(diag, RK_E_SHAPE, "validate", "download view tensor is invalid");
    if (src->valid_region[0] || src->valid_region[2] ||
        src->valid_region[1] != src->tensor.rows ||
        src->valid_region[3] != src->tensor.cols)
        return fail(diag, RK_E_SHAPE, "validate",
                    "download view must expose its full logical tile");
    /* Placement into the output binding is validated per publish tile. */
    return 0;
}

/* Declared arity of every template this backend emits.  Used to bound the input
 * and output arrays *before* they are indexed: a forged count would otherwise be
 * walked out of bounds. */
static int template_arity(uint32_t template_id, uint32_t *nin, uint32_t *nout) {
    switch (template_id) {
    case RK_TEMPLATE_MM_F16: *nin = 2; *nout = 1; return 0;
    case RK_TEMPLATE_MM_ACC_F16: *nin = 3; *nout = 1; return 0;
    case RK_TEMPLATE_EW_ADD_F16:
    case RK_TEMPLATE_EW_MUL_F16:
    case RK_TEMPLATE_EW_MAX_F16: *nin = 3; *nout = 1; return 0;
    case RK_TEMPLATE_LUT_SIGMOID_F16:
    case RK_TEMPLATE_LUT_HARDSIGMOID_F16:
    case RK_TEMPLATE_LUT_GELU_GATE_F16: *nin = 1; *nout = 1; return 0;
    default: return -1;
    }
}

static int validate_template(const rk_plan_desc *desc, const rk_action_desc *a, rk_diag *diag) {
    if (!a->inputs || !a->outputs)
        return fail(diag, RK_E_SCHEMA, "validate", "NPU task has null input/output array");
    /* Arity first: never walk a caller-supplied count before it is bounded. */
    {
        uint32_t nin = 0, nout = 0;
        if (template_arity(a->template_id, &nin, &nout) != 0)
            return fail(diag, RK_E_SCHEMA, "validate", "unknown NPU task template");
        if (a->ninputs != nin || a->noutputs != nout)
            return fail(diag, RK_E_SCHEMA, "validate",
                        "NPU task argument counts do not match its template");
    }
    for (uint32_t i = 0; i < a->ninputs; i++)
        if (!view_at(desc, a->inputs[i]))
            return fail(diag, RK_E_SCHEMA, "validate", "NPU task references missing input view");
    for (uint32_t i = 0; i < a->noutputs; i++)
        if (!view_at(desc, a->outputs[i]))
            return fail(diag, RK_E_SCHEMA, "validate", "NPU task references missing output view");
    if (a->template_id == RK_TEMPLATE_MM_F16) {
        if (a->ninputs != 2 || a->noutputs != 1)
            return fail(diag, RK_E_SHAPE, "validate", "mm requires two inputs and one output");
        const rk_view_desc *x = view_at(desc, a->inputs[0]);
        const rk_view_desc *w = view_at(desc, a->inputs[1]);
        const rk_view_desc *o = view_at(desc, a->outputs[0]);
        if (!tensor_valid(&x->tensor) || !tensor_valid(&w->tensor) || !tensor_valid(&o->tensor) ||
            x->tensor.layout != RK_LAYOUT_CUBE_F16 ||
            w->tensor.layout != RK_LAYOUT_WEIGHT_F16 ||
            o->tensor.layout != RK_LAYOUT_CUBE_F16)
            return fail(diag, RK_E_SHAPE, "validate", "mm layout mismatch");
        if (x->tensor.rows > 256 || x->tensor.cols > CO_K_TILE ||
            w->tensor.rows > 256 || w->tensor.cols > CO_K_TILE)
            return fail(diag, RK_E_SHAPE, "validate", "mm primitive exceeds 256");
        if (x->tensor.rows != o->tensor.rows || x->tensor.cols != w->tensor.cols ||
            w->tensor.rows != o->tensor.cols)
            return fail(diag, RK_E_SHAPE, "validate", "mm extents mismatch");
        if (view_at(desc, a->inputs[0])->buffer_id == o->buffer_id ||
            view_at(desc, a->inputs[1])->buffer_id == o->buffer_id)
            return fail(diag, RK_E_ALIAS, "validate", "mm output aliases an input");
    } else if (a->template_id == RK_TEMPLATE_MM_ACC_F16) {
        if (a->ninputs != 3 || a->noutputs != 1)
            return fail(diag, RK_E_SHAPE, "validate", "mm_acc requires three inputs and one output");
        const rk_view_desc *x = view_at(desc, a->inputs[0]);
        const rk_view_desc *w = view_at(desc, a->inputs[1]);
        const rk_view_desc *p = view_at(desc, a->inputs[2]);
        const rk_view_desc *o = view_at(desc, a->outputs[0]);
        if (!tensor_valid(&x->tensor) || !tensor_valid(&w->tensor) ||
            !tensor_valid(&p->tensor) || !tensor_valid(&o->tensor))
            return fail(diag, RK_E_SHAPE, "validate", "mm_acc has invalid tensor");
        if (x->tensor.layout != RK_LAYOUT_CUBE_F16 ||
            w->tensor.layout != RK_LAYOUT_WEIGHT_F16 ||
            p->tensor.layout != RK_LAYOUT_CUBE_F16 ||
            o->tensor.layout != RK_LAYOUT_CUBE_F16)
            return fail(diag, RK_E_SHAPE, "validate", "mm_acc layout mismatch");
        if (p->tensor.rows != o->tensor.rows || p->tensor.cols != o->tensor.cols ||
            p->tensor.physical_rows != o->tensor.physical_rows ||
            p->tensor.physical_cols != o->tensor.physical_cols)
            return fail(diag, RK_E_SHAPE, "validate", "mm_acc prior/output tensors differ");
        if (x->tensor.rows != o->tensor.rows || x->tensor.cols != w->tensor.cols ||
            w->tensor.rows != o->tensor.cols)
            return fail(diag, RK_E_SHAPE, "validate", "mm_acc extents mismatch");
        if (p->buffer_id == o->buffer_id)
            return fail(diag, RK_E_ALIAS, "validate", "mm_acc prior/output alias");
        const rk_buffer_desc *pb = buffer_at(desc, p->buffer_id);
        const rk_buffer_desc *ob = buffer_at(desc, o->buffer_id);
        if (!pb || !ob || pb->slot_id == ob->slot_id)
            return fail(diag, RK_E_ALIAS, "validate", "mm_acc prior/output share a BO");
        if (view_at(desc, a->inputs[0])->buffer_id == o->buffer_id ||
            view_at(desc, a->inputs[1])->buffer_id == o->buffer_id)
            return fail(diag, RK_E_ALIAS, "validate", "mm_acc output aliases x/weight");
    } else if (a->template_id == RK_TEMPLATE_EW_ADD_F16 ||
               a->template_id == RK_TEMPLATE_EW_MUL_F16 ||
               a->template_id == RK_TEMPLATE_EW_MAX_F16) {
        if (a->ninputs != 3 || a->noutputs != 1)
            return fail(diag, RK_E_SHAPE, "validate", "EW requires three inputs and one output");
        const rk_view_desc *x = view_at(desc, a->inputs[0]);
        const rk_view_desc *y = view_at(desc, a->inputs[1]);
        const rk_view_desc *id = view_at(desc, a->inputs[2]);
        const rk_view_desc *o = view_at(desc, a->outputs[0]);
        if (!tensor_valid(&x->tensor) || !tensor_valid(&y->tensor) ||
            !tensor_valid(&id->tensor) || !tensor_valid(&o->tensor))
            return fail(diag, RK_E_SHAPE, "validate", "EW has invalid tensor");
        if (x->tensor.rows != y->tensor.rows || x->tensor.cols != y->tensor.cols ||
            x->tensor.physical_rows != y->tensor.physical_rows ||
            x->tensor.physical_cols != y->tensor.physical_cols ||
            x->tensor.rows != o->tensor.rows || x->tensor.cols != o->tensor.cols ||
            x->tensor.physical_rows != o->tensor.physical_rows ||
            x->tensor.physical_cols != o->tensor.physical_cols)
            return fail(diag, RK_E_SHAPE, "validate", "EW requires exact shapes");
        if (id->tensor.layout != RK_LAYOUT_WEIGHT_F16 ||
            id->tensor.rows != x->tensor.physical_cols ||
            id->tensor.cols != x->tensor.physical_cols)
            return fail(diag, RK_E_SHAPE, "validate", "EW identity weight geometry mismatch");
        if (x->buffer_id == o->buffer_id || y->buffer_id == o->buffer_id)
            return fail(diag, RK_E_ALIAS, "validate", "EW output aliases an operand");
    } else if (a->template_id >= RK_TEMPLATE_LUT_SIGMOID_F16 &&
               a->template_id <= RK_TEMPLATE_LUT_GELU_GATE_F16) {
        if (a->ninputs != 1 || a->noutputs != 1)
            return fail(diag, RK_E_SHAPE, "validate", "LUT requires one input and one output");
        const rk_view_desc *x = view_at(desc, a->inputs[0]);
        const rk_view_desc *o = view_at(desc, a->outputs[0]);
        if (!tensor_valid(&x->tensor) || !tensor_valid(&o->tensor) ||
            x->tensor.bytes != o->tensor.bytes)
            return fail(diag, RK_E_SHAPE, "validate", "LUT physical sizes mismatch");
        if (x->buffer_id == o->buffer_id)
            return fail(diag, RK_E_ALIAS, "validate", "LUT input/output alias");
        if (x->tensor.layout == RK_LAYOUT_CUBE_F16) {
            uint64_t elements = (uint64_t)x->tensor.physical_rows * x->tensor.physical_cols;
            if (elements > 65528)
                return fail(diag, RK_E_SHAPE, "validate", "LUT view exceeds 65528 elements");
        } else if (x->tensor.layout != RK_LAYOUT_LINEAR_F16 ||
                   o->tensor.layout != RK_LAYOUT_LINEAR_F16) {
            return fail(diag, RK_E_SHAPE, "validate", "LUT layout mismatch");
        }
    } else {
        return fail(diag, RK_E_UNSUPPORTED, "validate", "unknown template id");
    }
    return 0;
}

/* Verify the schedule the runtime will actually emit, not just ID ordering.
 *
 * The runtime performs three phases: every host upload in ascending action id,
 * then every NPU task in (submit order, task_ids order), then every host
 * download in ascending action id.  That exact linear order must be a valid
 * execution order:
 *   - every dependency precedes its consumer,
 *   - every buffer has at least one writer and no read happens before the last
 *     writer of that buffer,
 *   - one buffer is never written by both the host and the device, and multiple
 *     device writers only touch disjoint byte ranges,
 *   - two logical buffers that share an allocation slot never have overlapping
 *     [first, last] access intervals (covers WAR/WAW introduced by reuse).
 */
/* Is ``ancestor`` reachable from ``consumer`` by following declared deps? */
static int is_ancestor(const rk_plan_desc *desc, uint32_t ancestor,
                       uint32_t consumer, unsigned char *visited) {
    if (ancestor == consumer) return 1;
    memset(visited, 0, desc->nactions);
    uint32_t *stack = malloc((size_t)desc->nactions * sizeof(uint32_t));
    if (!stack) return 0;
    uint32_t top = 0;
    stack[top++] = consumer;
    visited[consumer] = 1;
    int found = 0;
    while (top && !found) {
        uint32_t node = stack[--top];
        const rk_action_desc *a = &desc->actions[node];
        for (uint32_t d = 0; d < a->ndeps && !found; d++) {
            uint32_t dep = a->deps[d];
            if (dep == ancestor) { found = 1; break; }
            if (dep < desc->nactions && !visited[dep]) {
                visited[dep] = 1;
                stack[top++] = dep;
            }
        }
    }
    free(stack);
    return found;
}

static int validate_execution_order(const rk_plan_desc *desc, rk_diag *diag) {
    const uint32_t nactions = desc->nactions, nbuffers = desc->nbuffers;
    uint32_t *seq = malloc((size_t)nactions * sizeof(uint32_t));
    uint32_t *writer_action = malloc((size_t)nbuffers * sizeof(uint32_t));
    uint32_t *writer_class = malloc((size_t)nbuffers * sizeof(uint32_t));
    uint32_t *last_writer_seq = malloc((size_t)nbuffers * sizeof(uint32_t));
    uint32_t *first = malloc((size_t)nbuffers * sizeof(uint32_t));
    uint32_t *last = malloc((size_t)nbuffers * sizeof(uint32_t));
    uint32_t *writer_count = calloc((size_t)nbuffers, sizeof(uint32_t));
    uint32_t *persistent = calloc((size_t)nbuffers, sizeof(uint32_t));
    unsigned char *visited = malloc(nactions ? (size_t)nactions : 1u);
    uint32_t *writer_view_action = malloc((size_t)(nactions ? nactions : 1u) *
                                          sizeof(uint32_t));
    int rc = 0;
    if (!seq || !writer_action || !writer_class || !last_writer_seq ||
        !first || !last || !writer_count || !persistent || !visited ||
        !writer_view_action) {
        rc = fail(diag, RK_E_NOMEM, "validate", "out of memory");
        goto done;
    }
    for (uint32_t i = 0; i < nactions; i++) seq[i] = UINT32_MAX;
    for (uint32_t i = 0; i < nbuffers; i++) {
        writer_action[i] = UINT32_MAX;
        writer_class[i] = 0;
        last_writer_seq[i] = UINT32_MAX;
        first[i] = UINT32_MAX;
        last[i] = 0;
    }

    /* 1. The runtime walks actions in ascending id order and flushes one
     *    blocking submit per task group, so the emission order is the action
     *    order.  A submit therefore has to own a *strictly increasing* run of
     *    NPU task ids (its task list is in emission order), and no other submit's
     *    task may sit between them.  Contiguity of the ids is not required: a
     *    static-repeat group plan interleaves each task with its own uploads so
     *    that every read's nearest producer is a declared dependency, and the
     *    runtime indexes its emission array through ws->emit_index. */
    for (uint32_t i = 0; i < nactions; i++) seq[i] = i;
    for (uint32_t s = 0; s < desc->nsubmits; s++) {
        const rk_submit_desc *sub = &desc->submits[s];
        for (uint32_t t = 1; t < sub->ntask_ids; t++)
            if (sub->task_ids[t] <= sub->task_ids[t - 1]) {
                rc = fail(diag, RK_E_SCHEMA, "validate",
                          "submit tasks must be listed in ascending emission order");
                goto done;
            }
        for (uint32_t id = sub->task_ids[0]; id <= sub->task_ids[sub->ntask_ids - 1]; id++) {
            if (id >= nactions || desc->actions[id].kind != RK_ACTION_NPU_TASK) {
                rc = fail(diag, RK_E_SCHEMA, "validate",
                          "a host action is interleaved inside a task group");
                goto done;
            }
        }
    }

    /* 2. Declared dependencies must hold in that order. */
    for (uint32_t i = 0; i < nactions; i++) {
        const rk_action_desc *a = &desc->actions[i];
        for (uint32_t d = 0; d < a->ndeps; d++)
            if (seq[a->deps[d]] >= seq[i]) {
                rc = fail(diag, RK_E_SCHEMA, "validate",
                          "dependency is scheduled after its consumer");
                goto done;
            }
    }

    /* 3. Writer ownership per logical buffer. */
    for (uint32_t i = 0; i < nactions; i++) {
        const rk_action_desc *a = &desc->actions[i];
        uint32_t cls = a->kind == RK_ACTION_UPLOAD ? 1u : 2u;
        for (uint32_t o = 0; o < a->noutputs; o++) {
            const rk_view_desc *v = view_at(desc, a->outputs[o]);
            if (!v) { rc = fail(diag, RK_E_SCHEMA, "validate", "output view missing"); goto done; }
            uint32_t b = v->buffer_id;
            if (writer_class[b] && writer_class[b] != cls) {
                rc = fail(diag, RK_E_SCHEMA, "validate",
                          "buffer is written by both the host and the device");
                goto done;
            }
            writer_class[b] = cls;
            if (writer_action[b] == UINT32_MAX) writer_action[b] = i;
            /* A constant upload happens once at workspace creation and the
             * buffer must survive every later run, so its BO can never be
             * reused by a per-run buffer. */
            if (a->kind == RK_ACTION_UPLOAD) {
                const rk_value_desc *src = value_at(desc, a->source_binding);
                if ((a->flags & RK_ACTION_FLAG_CACHE_CONSTANT) ||
                    (src && src->role == RK_ROLE_CONSTANT))
                    persistent[b] = 1;
            }
            if (++writer_count[b] > RK_MAX_WRITERS_PER_BUFFER) {
                rc = fail(diag, RK_E_SCHEMA, "validate",
                          "too many writers for one logical buffer");
                goto done;
            }
            if (last_writer_seq[b] == UINT32_MAX || seq[i] > last_writer_seq[b])
                last_writer_seq[b] = seq[i];
        }
    }

    /* 4. Reads: there must be a producer before the read, and it must be a
     *    declared dependency of the consumer.
     *
     * A buffer may legitimately be written more than once (the K accumulation
     * ping-pong): what has to hold is that the *latest* write before a read is
     * the one the consumer depends on.  Checking "the last writer overall comes
     * first" would reject every chain with more than two accumulator passes. */
    {
        for (uint32_t i = 0; i < nactions; i++) {
            const rk_action_desc *a = &desc->actions[i];
            for (uint32_t n = 0; n < a->ninputs; n++) {
                const rk_view_desc *v = view_at(desc, a->inputs[n]);
                if (!v) {
                    rc = fail(diag, RK_E_SCHEMA, "validate", "input view missing");
                    goto done;
                }
                uint32_t b = v->buffer_id;
                uint32_t producer = UINT32_MAX, producer_seq = 0;
                for (uint32_t j = 0; j < i; j++) {
                    const rk_action_desc *w = &desc->actions[j];
                    if (!w->noutputs) continue;
                    int writes = 0;
                    for (uint32_t o = 0; o < w->noutputs; o++)
                        if (view_at(desc, w->outputs[o])->buffer_id == b) writes = 1;
                    if (writes && seq[j] >= producer_seq) {
                        producer = j;
                        producer_seq = seq[j];
                    }
                }
                if (producer == UINT32_MAX) {
                    rc = fail(diag, RK_E_SHAPE, "validate",
                              "read of a buffer that no action writes before it");
                    goto done;
                }
                if (!is_ancestor(desc, producer, i, visited)) {
                    rc = fail(diag, RK_E_SCHEMA, "validate",
                              "the producer of a read is not a declared "
                              "dependency of the consumer");
                    goto done;
                }
            }
        }
    }

    /* 5. Group writer views per buffer (bounded by RK_MAX_WRITERS_PER_BUFFER). */
    uint32_t *group_start = malloc(((size_t)nbuffers + 1u) * sizeof(uint32_t));
    uint32_t *writer_view = malloc((size_t)(nactions ? nactions : 1u) * sizeof(uint32_t));
    if (!group_start || !writer_view) {
        free(group_start); free(writer_view);
        rc = fail(diag, RK_E_NOMEM, "validate", "out of memory");
        goto done;
    }
    group_start[0] = 0;
    for (uint32_t b = 0; b < nbuffers; b++)
        group_start[b + 1u] = group_start[b] + writer_count[b];
    {
        uint32_t *cursor = malloc(((size_t)nbuffers + 1u) * sizeof(uint32_t));
        if (!cursor) {
            free(group_start); free(writer_view);
            rc = fail(diag, RK_E_NOMEM, "validate", "out of memory");
            goto done;
        }
        memcpy(cursor, group_start, ((size_t)nbuffers + 1u) * sizeof(uint32_t));
        for (uint32_t i = 0; i < nactions; i++) {
            const rk_action_desc *a = &desc->actions[i];
            for (uint32_t o = 0; o < a->noutputs; o++) {
                uint32_t b = view_at(desc, a->outputs[o])->buffer_id;
                writer_view[cursor[b]] = a->outputs[o];
                writer_view_action[cursor[b]] = i;
                cursor[b]++;
            }
        }
        free(cursor);
    }

    /* 6. Writer geometry rules per buffer.
     *
     * Two different write patterns share one buffer in this backend:
     *   - accumulator ping-pong: the same bytes are written several times,
     *     always with identical addressing;
     *   - LUT segmentation: disjoint byte ranges, possibly of different sizes,
     *     are written by separate linear views.
     * Therefore only *overlapping* writers must agree on addressing; disjoint
     * segments are allowed to differ.  A host upload may never mix with device
     * writes on one buffer, and one buffer may have at most one host upload. */
    for (uint32_t b = 0; b < nbuffers; b++) {
        const uint32_t first_w = group_start[b], count = group_start[b + 1u] - first_w;
        uint32_t host_writers = 0;
        for (uint32_t i = 0; i < nactions; i++) {
            if (desc->actions[i].kind != RK_ACTION_UPLOAD) continue;
            for (uint32_t o = 0; o < desc->actions[i].noutputs; o++)
                if (view_at(desc, desc->actions[i].outputs[o])->buffer_id == b)
                    host_writers++;
        }
        if (host_writers > 1) {
            rc = fail(diag, RK_E_ALIAS, "validate",
                      "a buffer may only be uploaded by one host action");
            goto grouped_done;
        }
        if (host_writers == 1 && count > 1) {
            rc = fail(diag, RK_E_ALIAS, "validate",
                      "a host upload and device writes must not share a buffer");
            goto grouped_done;
        }
        for (uint32_t x = 0; x < count; x++) {
            const rk_view_desc *vx = view_at(desc, writer_view[first_w + x]);
            for (uint32_t y = x + 1; y < count; y++) {
                const rk_view_desc *vy = view_at(desc, writer_view[first_w + y]);
                int overlap = vx->offset_bytes < vy->offset_bytes + vy->tensor.bytes &&
                              vy->offset_bytes < vx->offset_bytes + vx->tensor.bytes;
                if (overlap && !same_addressing(&vx->tensor, &vy->tensor)) {
                    rc = fail(diag, RK_E_ALIAS, "validate",
                              "overlapping writers of one buffer use different "
                              "physical addressing");
                    goto grouped_done;
                }
            }
        }
    }
    /* 7. Every read must be covered by writes that already happened and that the
     *    consumer declares as dependencies.  Coverage may not borrow from a
     *    writer scheduled later: that would let the reader consume bytes that do
     *    not exist yet (e.g. an EW multiply reading the second LUT segment
     *    before that segment runs). */
    for (uint32_t i = 0; i < nactions; i++) {
        const rk_action_desc *a = &desc->actions[i];
        for (uint32_t n = 0; n < a->ninputs; n++) {
            const rk_view_desc *rd = view_at(desc, a->inputs[n]);
            uint32_t b = rd->buffer_id;
            const uint32_t first_w = group_start[b];
            const uint32_t count = group_start[b + 1u] - first_w;
            uint64_t need_start = rd->offset_bytes;
            uint64_t need_end = rd->offset_bytes + rd->tensor.bytes;
            const int reader_is_linear = rd->tensor.layout == RK_LAYOUT_LINEAR_F16;
            uint32_t usable = 0;
            for (uint32_t w = 0; w < count; w++) {
                const rk_view_desc *wv = view_at(desc, writer_view[first_w + w]);
                uint64_t ws = wv->offset_bytes;
                uint64_t we = wv->offset_bytes + wv->tensor.bytes;
                if (we <= need_start || ws >= need_end) continue; /* disjoint */
                /* Which writers may contribute to covering this read?  A linear
                 * reader reinterprets packed bytes flatly; a cube/weight reader
                 * needs identical addressing, or a flat byte image produced by
                 * linear writers (the LUT segmentation contract). */
                int contributes = reader_is_linear ||
                    wv->tensor.layout == RK_LAYOUT_LINEAR_F16 ||
                    (ws == need_start && same_addressing(&wv->tensor, &rd->tensor));
                if (!contributes) continue;
                uint32_t writer_action = writer_view_action[first_w + w];
                if (seq[writer_action] >= seq[i]) continue; /* not yet executed */
                if (!is_ancestor(desc, writer_action, i, visited)) {
                    rc = fail(diag, RK_E_SCHEMA, "validate",
                              "a producer covering this read is not a declared "
                              "dependency of the consumer");
                    goto grouped_done;
                }
                usable++;
            }
            if (!usable) {
                rc = fail(diag, RK_E_SHAPE, "validate",
                          "no write before this read produces the consumed range");
                goto grouped_done;
            }
            uint64_t cursor = need_start;
            while (cursor < need_end) {
                uint64_t best = cursor;
                for (uint32_t w = 0; w < count; w++) {
                    const rk_view_desc *wv = view_at(desc, writer_view[first_w + w]);
                    uint64_t ws = wv->offset_bytes;
                    uint64_t we = wv->offset_bytes + wv->tensor.bytes;
                    if (we <= need_start || ws >= need_end) continue;
                    int contributes = reader_is_linear ||
                        wv->tensor.layout == RK_LAYOUT_LINEAR_F16 ||
                        (ws == need_start && same_addressing(&wv->tensor, &rd->tensor));
                    if (!contributes) continue;
                    uint32_t writer_action = writer_view_action[first_w + w];
                    if (seq[writer_action] >= seq[i]) continue;
                    if (!is_ancestor(desc, writer_action, i, visited)) continue;
                    if (ws <= cursor && we > best) best = we;
                }
                if (best <= cursor) {
                    rc = fail(diag, RK_E_SHAPE, "validate",
                              "the writes that precede this read do not cover the "
                              "consumed range");
                    goto grouped_done;
                }
                cursor = best;
            }
        }
    }
grouped_done:
    free(group_start);
    free(writer_view);
    if (rc) goto done;

    /* 6. Access intervals per logical buffer, then slot reuse hazards. */
    for (uint32_t i = 0; i < nactions; i++) {
        const rk_action_desc *a = &desc->actions[i];
        for (uint32_t n = 0; n < a->ninputs; n++) {
            uint32_t b = view_at(desc, a->inputs[n])->buffer_id;
            if (seq[i] < first[b]) first[b] = seq[i];
            if (seq[i] > last[b]) last[b] = seq[i];
        }
        for (uint32_t o = 0; o < a->noutputs; o++) {
            uint32_t b = view_at(desc, a->outputs[o])->buffer_id;
            if (seq[i] < first[b]) first[b] = seq[i];
            if (seq[i] > last[b]) last[b] = seq[i];
        }
    }
    for (uint32_t b = 0; b < nbuffers; b++)
        if (first[b] == UINT32_MAX) {
            rc = fail(diag, RK_E_SHAPE, "validate", "logical buffer is never accessed");
            goto done;
        }
    for (uint32_t b = 0; b < nbuffers; b++) {
        uint32_t peers = 0;
        for (uint32_t c = 0; c < nbuffers; c++)
            if (desc->buffers[c].slot_id == desc->buffers[b].slot_id && ++peers > RK_MAX_BUFFERS_PER_SLOT) {
                rc = fail(diag, RK_E_SCHEMA, "validate", "too many buffers share one slot");
                goto done;
            }
        if (persistent[b] && peers != 1u) {
            rc = fail(diag, RK_E_ALIAS, "validate",
                      "a constant workspace buffer must own its slot for the "
                      "whole workspace lifetime");
            goto done;
        }
        for (uint32_t c = b + 1; c < nbuffers; c++) {
            if (desc->buffers[c].slot_id != desc->buffers[b].slot_id) continue;
            if (!(last[b] < first[c] || last[c] < first[b])) {
                rc = fail(diag, RK_E_ALIAS, "validate",
                          "buffers sharing a slot have overlapping live ranges");
                goto done;
            }
        }
    }

done:
    free(seq); free(writer_action); free(writer_class); free(last_writer_seq);
    free(first); free(last); free(writer_count); free(persistent);
    free(visited); free(writer_view_action);
    return rc;
}

int rk_plan_validate_desc(const rk_plan_desc *desc, rk_diag *diag) {
    rk_diag_clear(diag);
    if (!desc) return fail(diag, RK_E_INVALID, "validate", "null plan descriptor");
    if (!struct_ok(desc->abi_version, desc->struct_size, sizeof(*desc)))
        return fail(diag, RK_E_SCHEMA, "validate", "rk_plan_desc ABI/size mismatch");
    if (desc->schema_version != RK_SCHEMA_VERSION)
        return fail(diag, RK_E_SCHEMA, "validate", "unsupported plan schema version");
    if (desc->runtime_abi_version != RK_ABI_VERSION)
        return fail(diag, RK_E_SCHEMA, "validate",
                    "descriptor runtime_abi_version does not match this library");
    if (!desc->values || !desc->nvalues || desc->nvalues > 1000000)
        return fail(diag, RK_E_SCHEMA, "validate", "invalid values array");
    if (!desc->nbuffers || desc->nbuffers > 1000000 || !desc->nviews ||
        desc->nviews > 1000000 || !desc->nactions || desc->nactions > 1000000 ||
        !desc->nsubmits || desc->nsubmits > 1000000 || !desc->npublish ||
        desc->npublish > 1000000)
        return fail(diag, RK_E_SCHEMA, "validate", "missing or excessive plan arrays");
    if ((!desc->buffers && desc->nbuffers) || (!desc->views && desc->nviews) ||
        (!desc->actions && desc->nactions) || (!desc->submits && desc->nsubmits) ||
        (!desc->publish && desc->npublish))
        return fail(diag, RK_E_SCHEMA, "validate", "null array with non-zero count");

    /* Values and binding names. */
    for (uint32_t i = 0; i < desc->nvalues; i++) {
        const rk_value_desc *v = &desc->values[i];
        if (!struct_ok(v->abi_version, v->struct_size, sizeof(*v)))
            return fail(diag, RK_E_SCHEMA, "validate", "value descriptor ABI/size mismatch");
        if (v->id != i || !valid_role(v->role) || v->dtype != RK_DTYPE_F16 ||
            v->rank > 2 || (v->rank != 2 && v->role != RK_ROLE_CONSTANT))
            return fail(diag, RK_E_SCHEMA, "validate", "value id/role/dtype/rank invalid");
        if (!v->name || !v->name_len || v->name_len > 128)
            return fail(diag, RK_E_SCHEMA, "validate", "value name invalid");
        if (v->rank == 0) {
            if (v->shape[0] || v->shape[1] || v->nbytes != 2u)
                return fail(diag, RK_E_SHAPE, "validate", "scalar value shape/bytes invalid");
        } else if (v->rank == 1) {
            if (!v->shape[0] || v->shape[0] > 4096 || v->shape[1] ||
                v->nbytes != (uint64_t)v->shape[0] * 2u)
                return fail(diag, RK_E_SHAPE, "validate", "channel value shape/bytes invalid");
        } else {
            if (!v->shape[0] || !v->shape[1] ||
                v->shape[0] > 4096 || v->shape[1] > 4096)
                return fail(diag, RK_E_SHAPE, "validate", "value shape out of range");
            if (v->nbytes != (uint64_t)v->shape[0] * v->shape[1] * 2u)
                return fail(diag, RK_E_SHAPE, "validate", "value byte size mismatch");
        }
        for (uint32_t j = 0; j < i; j++)
            if (desc->values[j].name_len == v->name_len &&
                !memcmp(desc->values[j].name, v->name, v->name_len))
                return fail(diag, RK_E_SCHEMA, "validate", "duplicate value name");
    }

    /* Buffers. */
    for (uint32_t i = 0; i < desc->nbuffers; i++) {
        const rk_buffer_desc *b = &desc->buffers[i];
        if (!struct_ok(b->abi_version, b->struct_size, sizeof(*b)))
            return fail(diag, RK_E_SCHEMA, "validate", "buffer descriptor ABI/size mismatch");
        if (b->id != i || !valid_role(b->role) || !valid_layout(b->layout_id))
            return fail(diag, RK_E_SCHEMA, "validate", "buffer id/role/layout invalid");
        if (b->alignment < 16 || !is_pow2(b->alignment) || b->offset % b->alignment ||
            !b->capacity)
            return fail(diag, RK_E_SHAPE, "validate", "buffer alignment/capacity invalid");
        if (b->lifetime_first > b->lifetime_last || b->lifetime_last >= desc->nactions)
            return fail(diag, RK_E_SHAPE, "validate", "buffer lifetime invalid");
        if (b->slot_id >= desc->nbuffers)
            return fail(diag, RK_E_SHAPE, "validate", "buffer slot id out of range");
        if (b->offset > UINT64_MAX - b->capacity)
            return fail(diag, RK_E_SHAPE, "validate", "buffer range overflows");
    }

    /* Views. */
    for (uint32_t i = 0; i < desc->nviews; i++) {
        const rk_view_desc *v = &desc->views[i];
        if (!struct_ok(v->abi_version, v->struct_size, sizeof(*v)))
            return fail(diag, RK_E_SCHEMA, "validate", "view descriptor ABI/size mismatch");
        if (v->id != i || !tensor_valid(&v->tensor))
            return fail(diag, RK_E_SHAPE, "validate", "view id/tensor invalid");
        const rk_buffer_desc *b = buffer_at(desc, v->buffer_id);
        if (!b) return fail(diag, RK_E_SCHEMA, "validate", "view references missing buffer");
        if (v->offset_bytes % 16 || v->offset_bytes > b->capacity ||
            v->tensor.bytes > b->capacity - v->offset_bytes)
            return fail(diag, RK_E_SHAPE, "validate", "view exceeds buffer capacity");
        if (v->valid_region[0] > v->tensor.rows || v->valid_region[1] > v->tensor.rows ||
            v->valid_region[2] > v->tensor.cols || v->valid_region[3] > v->tensor.cols ||
            v->valid_region[0] + v->valid_region[1] > v->tensor.rows ||
            v->valid_region[2] + v->valid_region[3] > v->tensor.cols)
            return fail(diag, RK_E_SHAPE, "validate", "view valid_region invalid");
    }

    /* Actions and dependencies (dep < id is a sufficient topological proof). */
    uint32_t *task_submit = calloc(desc->nactions ? desc->nactions : 1, sizeof(uint32_t));
    if (!task_submit) return fail(diag, RK_E_NOMEM, "validate", "out of memory");
    for (uint32_t i = 0; i < desc->nactions; i++) task_submit[i] = UINT32_MAX;
    for (uint32_t i = 0; i < desc->nactions; i++) {
        const rk_action_desc *a = &desc->actions[i];
        if (!struct_ok(a->abi_version, a->struct_size, sizeof(*a))) {
            free(task_submit);
            return fail(diag, RK_E_SCHEMA, "validate", "action descriptor ABI/size mismatch");
        }
        if (a->id != i || !valid_kind(a->kind)) {
            free(task_submit);
            return fail(diag, RK_E_SCHEMA, "validate", "action id/kind invalid");
        }
        /* Fixed-size ABI strings must be NUL terminated inside their own field:
         * a copy routine that appends a terminator would otherwise write one
         * byte past the array. */
        if (!memchr(a->origin, 0, sizeof(a->origin))) {
            free(task_submit);
            return fail(diag, RK_E_SCHEMA, "validate",
                        "action origin is not NUL terminated");
        }
        if (a->ninputs > RK_MAX_ACTION_ARGS || a->noutputs > RK_MAX_ACTION_ARGS ||
            a->ndeps > RK_MAX_ACTION_DEPS) {
            free(task_submit);
            return fail(diag, RK_E_SCHEMA, "validate",
                        "action argument count exceeds the ABI bound");
        }
        if (a->ninputs && !a->inputs) { free(task_submit); return fail(diag, RK_E_SCHEMA, "validate", "action inputs pointer/count mismatch"); }
        if (a->noutputs && !a->outputs) { free(task_submit); return fail(diag, RK_E_SCHEMA, "validate", "action outputs pointer/count mismatch"); }
        if (a->ndeps && !a->deps) { free(task_submit); return fail(diag, RK_E_SCHEMA, "validate", "action deps pointer/count mismatch"); }
        for (uint32_t j = 0; j < a->ndeps; j++)
            if (a->deps[j] >= i) { free(task_submit); return fail(diag, RK_E_SCHEMA, "validate", "action dependency not topological"); }
        int rc = 0;
        if (a->kind == RK_ACTION_UPLOAD) rc = validate_upload(desc, a, diag);
        else if (a->kind == RK_ACTION_DOWNLOAD) rc = validate_download(desc, a, diag);
        else if (a->kind == RK_ACTION_NPU_TASK) rc = validate_template(desc, a, diag);
        else if (a->kind == RK_ACTION_HOST_COMPUTE) rc = fail(diag, RK_E_UNSUPPORTED, "validate", "hidden host compute is forbidden");
        else rc = fail(diag, RK_E_UNSUPPORTED, "validate", "unsupported action kind");
        if (rc) { free(task_submit); return rc; }
    }

    /* Submits. */
    for (uint32_t i = 0; i < desc->nsubmits; i++) {
        const rk_submit_desc *s = &desc->submits[i];
        if (!struct_ok(s->abi_version, s->struct_size, sizeof(*s)))
            { free(task_submit); return fail(diag, RK_E_SCHEMA, "validate", "submit descriptor ABI/size mismatch"); }
        if (s->id != i || s->mode > RK_SUBMIT_SEPARATE ||
            s->core_policy > RK_CORE_AUTO || !s->ntask_ids ||
            s->ntask_ids > RK_MAX_SUBMIT_TASKS || !s->task_ids ||
            (s->ndeps && !s->deps))
            { free(task_submit); return fail(diag, RK_E_SHAPE, "validate", "submit metadata invalid"); }
        if (s->ndeps && !s->deps)
            { free(task_submit); return fail(diag, RK_E_SCHEMA, "validate", "submit deps pointer/count mismatch"); }
        for (uint32_t j = 0; j < s->ndeps; j++)
            if (s->deps[j] >= i) { free(task_submit); return fail(diag, RK_E_SCHEMA, "validate", "submit dependency not topological"); }
        /* Task IDs are validated *before* any index into desc->actions: a
         * descriptor may carry id == nactions or UINT32_MAX, and reading the
         * template first would be an out-of-bounds access. */
        for (uint32_t j = 0; j < s->ntask_ids; j++) {
            uint32_t task = s->task_ids[j];
            if (task == UINT32_MAX || task >= desc->nactions) {
                free(task_submit);
                return fail(diag, RK_E_SCHEMA, "validate",
                            "submit task id is out of range");
            }
            const rk_action_desc *a = action_at(desc, task);
            if (!a || a->kind != RK_ACTION_NPU_TASK) {
                free(task_submit);
                return fail(diag, RK_E_SCHEMA, "validate",
                            "submit contains non-NPU task");
            }
            if (task_submit[task] != UINT32_MAX) {
                free(task_submit);
                return fail(diag, RK_E_SCHEMA, "validate",
                            "task assigned to multiple submits");
            }
            task_submit[task] = i;
        }
        /* Submit whitelist: a group is an all-MM chain, an all-MM_ACC chain that
         * continues one, a LUT run (optionally finished by one EW multiply) or a
         * single element-wise task.  Anything else would be an unverified chain.
         * Every id is known to be in range at this point. */
        {
            uint32_t first_tpl = desc->actions[s->task_ids[0]].template_id;
            /* A chain that starts with a plain MM task is admitted in two shapes:
             *   (a) a K-accumulation chain, where later tasks accumulate
             *       (first MM, then MM_ACC), or
             *   (b) a static-repeat group of *independent* MM tasks: every task is
             *       a plain MM and the group is structurally independent, i.e. the
             *       tasks write pairwise distinct buffers and no task reads a
             *       buffer written earlier in the same submit.  The independence
             *       is proven from the descriptor below, not from a recipe name
             *       (the C ABI carries no recipe string). */
            int all_plain_mm = (first_tpl == RK_TEMPLATE_MM_F16);
            for (uint32_t j = 1; j < s->ntask_ids && all_plain_mm; j++)
                if (desc->actions[s->task_ids[j]].template_id != RK_TEMPLATE_MM_F16)
                    all_plain_mm = 0;
            if (all_plain_mm && s->ntask_ids > 1) {
                /* Each task's written *region* (buffer base + view offset, view
                 * extent) must be disjoint from every other task's written region
                 * and from every task's read region.  Shared BOs are legal only
                 * as disjoint slices, which is what a static-repeat-expanded plan
                 * produces (one buffer per role, one view per tile). */
                typedef struct { uint32_t buffer_id; uint64_t start, end; } region;
                region writes[RK_MAX_SUBMIT_TASKS];
                region reads[RK_MAX_SUBMIT_TASKS][RK_MAX_ACTION_ARGS];
                uint32_t nreads[RK_MAX_SUBMIT_TASKS];
                uint32_t nwritten = 0, j;
                for (j = 0; j < s->ntask_ids; j++) {
                    const rk_action_desc *a = &desc->actions[s->task_ids[j]];
                    if (a->noutputs != 1 || a->ninputs > RK_MAX_ACTION_ARGS) {
                        all_plain_mm = 0;
                        break;
                    }
                    const rk_view_desc *ov = view_at(desc, a->outputs[0]);
                    const rk_buffer_desc *ob = ov ? buffer_at(desc, ov->buffer_id) : NULL;
                    if (!ov || !ob) { all_plain_mm = 0; break; }
                    writes[nwritten].buffer_id = ov->buffer_id;
                    writes[nwritten].start = ob->offset + ov->offset_bytes;
                    writes[nwritten].end = writes[nwritten].start + ov->tensor.bytes;
                    nwritten++;
                    nreads[j] = 0;
                    for (uint32_t ii = 0; ii < a->ninputs; ii++) {
                        const rk_view_desc *iv = view_at(desc, a->inputs[ii]);
                        const rk_buffer_desc *ib = iv ? buffer_at(desc, iv->buffer_id) : NULL;
                        if (!iv || !ib) { all_plain_mm = 0; break; }
                        reads[j][nreads[j]].buffer_id = iv->buffer_id;
                        reads[j][nreads[j]].start = ib->offset + iv->offset_bytes;
                        reads[j][nreads[j]].end =
                            reads[j][nreads[j]].start + iv->tensor.bytes;
                        nreads[j]++;
                    }
                    if (!all_plain_mm) break;
                }
                for (j = 0; j < nwritten && all_plain_mm; j++)
                    for (uint32_t q = j + 1; q < nwritten; q++)
                        if (writes[j].buffer_id == writes[q].buffer_id &&
                            writes[j].start < writes[q].end &&
                            writes[q].start < writes[j].end) {
                            all_plain_mm = 0;
                            break;
                        }
                for (j = 0; j < nwritten && all_plain_mm; j++)
                    for (uint32_t r = 0; r < s->ntask_ids && all_plain_mm; r++)
                        for (uint32_t q = 0; q < nreads[r]; q++)
                            if (reads[r][q].buffer_id == writes[j].buffer_id &&
                                reads[r][q].start < writes[j].end &&
                                writes[j].start < reads[r][q].end) {
                                all_plain_mm = 0;
                                break;
                            }
                if (!all_plain_mm) {
                    free(task_submit);
                    return fail(diag, RK_E_SCHEMA, "validate",
                                "independent MM chain has overlapping read/write "
                                "regions; a shared BO is only legal as disjoint "
                                "slices");
                }
            }
            if (all_plain_mm && s->ntask_ids > 1) continue;  /* admitted group */
            for (uint32_t j = 1; j < s->ntask_ids; j++) {
                uint32_t tpl = desc->actions[s->task_ids[j]].template_id;
                int ok;
                if (first_tpl == RK_TEMPLATE_MM_ACC_F16)
                    ok = tpl == RK_TEMPLATE_MM_ACC_F16;
                else if (first_tpl == RK_TEMPLATE_MM_F16)
                    ok = tpl == RK_TEMPLATE_MM_ACC_F16;
                else if (first_tpl == RK_TEMPLATE_LUT_SIGMOID_F16 ||
                         first_tpl == RK_TEMPLATE_LUT_HARDSIGMOID_F16 ||
                         first_tpl == RK_TEMPLATE_LUT_GELU_GATE_F16)
                    ok = tpl == first_tpl ||
                         (tpl == RK_TEMPLATE_EW_MUL_F16 && j + 1u == s->ntask_ids);
                else
                    ok = 0;
                if (!ok) {
                    free(task_submit);
                    return fail(diag, RK_E_SCHEMA, "validate",
                                "submit task sequence is not a registered chain");
                }
            }
        }
    }
    for (uint32_t i = 0; i < desc->nactions; i++)
        if (desc->actions[i].kind == RK_ACTION_NPU_TASK && task_submit[i] == UINT32_MAX)
            { free(task_submit); return fail(diag, RK_E_SCHEMA, "validate", "unassigned NPU task"); }
    free(task_submit);

    /* Host boundary coverage and NPU alias checks below are repeated here from a
     * different source of truth than the Python planner. */
    for (uint32_t i = 0; i < desc->nactions; i++) {
        const rk_action_desc *a = &desc->actions[i];
        if (a->kind == RK_ACTION_NPU_TASK) {
            for (uint32_t oi = 0; oi < a->noutputs; oi++) {
                const rk_view_desc *ov = view_at(desc, a->outputs[oi]);
                const rk_buffer_desc *ob = buffer_at(desc, ov->buffer_id);
                for (uint32_t ii = 0; ii < a->ninputs; ii++) {
                    const rk_view_desc *iv = view_at(desc, a->inputs[ii]);
                    const rk_buffer_desc *ib = buffer_at(desc, iv->buffer_id);
                    if (!ov || !ob || !iv || !ib)
                        return fail(diag, RK_E_SCHEMA, "validate", "NPU view/buffer missing");
                    if (iv->buffer_id == ov->buffer_id || ib->slot_id == ob->slot_id)
                        return fail(diag, RK_E_ALIAS, "validate", "NPU output aliases/shared BO with an input");
                }
            }
        }
    }
    for (uint32_t i = 0; i < desc->nvalues; i++) {
        const rk_value_desc *v = &desc->values[i];
        if (v->role != RK_ROLE_INPUT && v->role != RK_ROLE_CONSTANT &&
            v->role != RK_ROLE_OUTPUT) continue;
        int found = 0;
        for (uint32_t j = 0; j < desc->nactions; j++) {
            const rk_action_desc *a = &desc->actions[j];
            if (v->role == RK_ROLE_OUTPUT) {
                if (a->kind == RK_ACTION_DOWNLOAD && a->source_binding == v->id) found = 1;
            } else if (a->kind == RK_ACTION_UPLOAD && a->source_binding == v->id) {
                found = 1;
            }
        }
        if (!found)
            return fail(diag, RK_E_SCHEMA, "validate", "binding has no matching host boundary action");
    }

    {
        int order_rc = validate_execution_order(desc, diag);
        if (order_rc) return order_rc;
    }

    /* ---- Publish tiles must exactly tile every output binding. ---- */
    for (uint32_t i = 0; i < desc->npublish; i++) {
        const rk_publish_desc *p = &desc->publish[i];
        if (!struct_ok(p->abi_version, p->struct_size, sizeof(*p)))
            return fail(diag, RK_E_SCHEMA, "validate", "publish descriptor ABI/size mismatch");
        const rk_value_desc *v = value_at(desc, p->binding_id);
        const rk_view_desc *view = view_at(desc, p->view_id);
        if (!v || !view || v->role != RK_ROLE_OUTPUT)
            return fail(diag, RK_E_SCHEMA, "validate", "publish target invalid");
        if (!p->rows || !p->cols ||
            (uint64_t)p->dst_row_offset + p->rows > v->shape[0] ||
            (uint64_t)p->dst_col_offset + p->cols > v->shape[1])
            return fail(diag, RK_E_SHAPE, "validate", "publish tile exceeds the output binding");
        if (p->rows != view->tensor.rows || p->cols != view->tensor.cols)
            return fail(diag, RK_E_SHAPE, "validate",
                        "publish tile must publish the whole view");
        for (uint32_t j = 0; j < i; j++) {
            const rk_publish_desc *q = &desc->publish[j];
            if (q->binding_id != p->binding_id) continue;
            if (q->view_id == p->view_id)
                return fail(diag, RK_E_SCHEMA, "validate",
                            "the same view cannot be published twice");
            if (q->dst_row_offset < p->dst_row_offset + p->rows &&
                p->dst_row_offset < q->dst_row_offset + q->rows &&
                q->dst_col_offset < p->dst_col_offset + p->cols &&
                p->dst_col_offset < q->dst_col_offset + q->cols)
                return fail(diag, RK_E_SHAPE, "validate", "publish tiles overlap");
        }
        /* The tile must be the destination of a declared download boundary. */
        int downloaded = 0;
        for (uint32_t j = 0; j < desc->nactions; j++) {
            const rk_action_desc *a = &desc->actions[j];
            if (a->kind == RK_ACTION_DOWNLOAD && a->source_binding == p->binding_id &&
                a->ninputs == 1 && a->inputs[0] == p->view_id) {
                downloaded = 1;
                break;
            }
        }
        if (!downloaded)
            return fail(diag, RK_E_SCHEMA, "validate",
                        "publish view has no matching download action");
    }
    for (uint32_t i = 0; i < desc->nvalues; i++) {
        const rk_value_desc *v = &desc->values[i];
        if (v->role != RK_ROLE_OUTPUT) continue;
        uint64_t area = 0;
        for (uint32_t j = 0; j < desc->npublish; j++)
            if (desc->publish[j].binding_id == i)
                area += (uint64_t)desc->publish[j].rows * desc->publish[j].cols;
        if (area != (uint64_t)v->shape[0] * v->shape[1])
            return fail(diag, RK_E_SHAPE, "validate",
                        "publish tiles do not cover the output binding exactly");
    }
    /* Every download action must be published exactly once. */
    for (uint32_t i = 0; i < desc->nactions; i++) {
        const rk_action_desc *a = &desc->actions[i];
        if (a->kind != RK_ACTION_DOWNLOAD) continue;
        uint32_t matches = 0;
        for (uint32_t j = 0; j < desc->npublish; j++)
            if (desc->publish[j].binding_id == a->source_binding &&
                a->ninputs == 1 && desc->publish[j].view_id == a->inputs[0])
                matches++;
        if (matches != 1)
            return fail(diag, RK_E_SCHEMA, "validate",
                        "every download must map to exactly one publish tile");
    }

    /* Recompute capacity from independent slots.  Never trust claimed sizes. */
    uint64_t *slot_cap = calloc(desc->nbuffers ? desc->nbuffers : 1, sizeof(uint64_t));
    if (!slot_cap) return fail(diag, RK_E_NOMEM, "validate", "out of memory");
    for (uint32_t i = 0; i < desc->nbuffers; i++) {
        const rk_buffer_desc *b = &desc->buffers[i];
        uint64_t end = b->offset + b->capacity;
        if (end > slot_cap[b->slot_id]) slot_cap[b->slot_id] = end;
    }
    uint64_t reserved = 0, slots = 0;
    for (uint32_t i = 0; i < desc->nbuffers; i++) {
        if (slot_cap[i]) {
            slots++;
            if (slot_cap[i] > UINT64_MAX - CO_SLACK - CO_GUARD) {
                free(slot_cap);
                return fail(diag, RK_E_BUDGET, "validate", "slot capacity overflow");
            }
            reserved += slot_cap[i] + CO_SLACK + CO_GUARD;
            if (reserved < slot_cap[i]) {
                free(slot_cap);
                return fail(diag, RK_E_BUDGET, "validate", "reserved capacity overflow");
            }
        }
    }
    free(slot_cap);
    uint32_t max_tasks = 0;
    for (uint32_t i = 0; i < desc->nsubmits; i++)
        if (desc->submits[i].ntask_ids > max_tasks) max_tasks = desc->submits[i].ntask_ids;
    uint64_t command = (uint64_t)max_tasks * CO_MAX_WORDS * 8u;
    if (reserved > UINT64_MAX - command)
        return fail(diag, RK_E_BUDGET, "validate", "device request overflow");
    uint64_t device = reserved + command;
    if (!desc->claimed_device_bytes)
        return fail(diag, RK_E_BUDGET, "validate",
                    "claimed_device_bytes must be stated (0 is not a bypass)");
    if (desc->claimed_device_bytes < device)
        return fail(diag, RK_E_BUDGET, "validate", "claimed device bytes understate slots/command workspace");
    if (desc->device_budget_bytes && device > desc->device_budget_bytes)
        return fail(diag, RK_E_BUDGET, "validate", "device budget exceeded");
    /* Recompute the host staging requirement instead of trusting the claim:
     * every binding nbytes plus a staging buffer for the largest output. */
    {
        uint64_t host = 0, largest_output = 0, output_total = 0;
        for (uint32_t i = 0; i < desc->nvalues; i++) {
            const rk_value_desc *v = &desc->values[i];
            if (v->role != RK_ROLE_INPUT && v->role != RK_ROLE_CONSTANT &&
                v->role != RK_ROLE_OUTPUT) continue;
            if (host > UINT64_MAX - v->nbytes)
                return fail(diag, RK_E_BUDGET, "validate", "host bytes overflow");
            host += v->nbytes;
            if (v->role == RK_ROLE_OUTPUT) {
                /* every published output is staged at the same time */
                if (output_total > UINT64_MAX - v->nbytes)
                    return fail(diag, RK_E_BUDGET, "validate",
                                "host output bytes overflow");
                output_total += v->nbytes;
                if (v->nbytes > largest_output) largest_output = v->nbytes;
            }
        }
        if (output_total > largest_output) largest_output = output_total;
        if (largest_output < 65536u) largest_output = 65536u;
        if (host > UINT64_MAX - largest_output)
            return fail(diag, RK_E_BUDGET, "validate", "host bytes overflow");
        host += largest_output;
        /* Runtime-owned Host scratch lives as long as the workspace: one
         * co_program per submit task (emission array), one download tile, and
         * the per-output offset/count bookkeeping.  Counting it here is what
         * keeps a plan's declared Host requirement honest; the workspace checks
         * the same total before it allocates. */
        {
            uint64_t runtime = (uint64_t)max_tasks * (uint64_t)sizeof(co_program);
            uint64_t tile = 0, nout = 0;
            for (uint32_t i = 0; i < desc->nvalues; i++) {
                const rk_value_desc *v = &desc->values[i];
                if (v->role != RK_ROLE_OUTPUT) continue;
                nout++;
                if (v->nbytes > tile) tile = v->nbytes;
            }
            if (runtime > UINT64_MAX - tile)
                return fail(diag, RK_E_BUDGET, "validate", "host bytes overflow");
            runtime += tile;
            runtime += nout * (uint64_t)(sizeof(size_t) + sizeof(uint32_t));
            if (host > UINT64_MAX - runtime)
                return fail(diag, RK_E_BUDGET, "validate", "host bytes overflow");
            host += runtime;
        }
        /* A zero claim is not "unconstrained": it would bypass the check. */
        if (!desc->claimed_host_bytes)
            return fail(diag, RK_E_BUDGET, "validate",
                        "claimed_host_bytes must be stated (0 is not a bypass)");
        if (desc->claimed_host_bytes < host)
            return fail(diag, RK_E_BUDGET, "validate",
                        "claimed host bytes understate bindings and staging");
    }
    if (desc->host_budget_bytes && desc->claimed_host_bytes > desc->host_budget_bytes)
        return fail(diag, RK_E_BUDGET, "validate", "host budget exceeded");
    return 0;
}

static char *copy_name(const char *src, uint32_t len) {
    if (!src || !len) return NULL;
    char *out = malloc((size_t)len + 1);
    if (!out) return NULL;
    memcpy(out, src, len);
    out[len] = 0;
    return out;
}

int rk_plan_clone(const rk_plan_desc *desc, rk_plan **out, rk_diag *diag) {
    if (!out) return fail(diag, RK_E_INVALID, "clone", "null out pointer");
    *out = NULL;
    int rc = rk_plan_validate_desc(desc, diag);
    if (rc) return rc;
    rk_plan *p = calloc(1, sizeof(*p));
    if (!p) return fail(diag, RK_E_NOMEM, "clone", "out of memory");
    p->schema_version = desc->schema_version;
    p->flags = desc->flags;
    p->profile_hash = desc->profile_hash;
    p->source_hash = desc->source_hash;
    p->device_budget_bytes = desc->device_budget_bytes;
    p->host_budget_bytes = desc->host_budget_bytes;
    p->claimed_device_bytes = desc->claimed_device_bytes;
    p->claimed_host_bytes = desc->claimed_host_bytes;
    p->runtime_abi_version = desc->runtime_abi_version;
    p->nvalues = desc->nvalues;
    p->values = calloc(desc->nvalues, sizeof(*p->values));
    p->value_names = calloc(desc->nvalues, sizeof(char *));
    p->nbuffers = desc->nbuffers;
    p->buffers = calloc(desc->nbuffers ? desc->nbuffers : 1, sizeof(*p->buffers));
    p->nviews = desc->nviews;
    p->views = calloc(desc->nviews ? desc->nviews : 1, sizeof(*p->views));
    p->nactions = desc->nactions;
    p->actions = calloc(desc->nactions, sizeof(*p->actions));
    p->action_inputs = calloc(desc->nactions, sizeof(uint32_t *));
    p->action_outputs = calloc(desc->nactions, sizeof(uint32_t *));
    p->action_deps = calloc(desc->nactions, sizeof(uint32_t *));
    p->nsubmits = desc->nsubmits;
    p->submits = calloc(desc->nsubmits, sizeof(*p->submits));
    p->submit_tasks = calloc(desc->nsubmits, sizeof(uint32_t *));
    p->submit_deps = calloc(desc->nsubmits, sizeof(uint32_t *));
    p->npublish = desc->npublish;
    p->publish = calloc(desc->npublish ? desc->npublish : 1, sizeof(*p->publish));
    if (!p->values || !p->value_names || !p->buffers || !p->views ||
        !p->actions || !p->action_inputs || !p->action_outputs ||
        !p->action_deps || !p->submits || !p->submit_tasks ||
        !p->submit_deps || !p->publish) {
        rk_plan_free(p);
        fail(diag, RK_E_NOMEM, "clone", "out of memory");
        return RK_E_NOMEM;
    }
    for (uint32_t i = 0; i < desc->nvalues; i++) {
        p->values[i] = desc->values[i];
        p->value_names[i] = copy_name(desc->values[i].name, desc->values[i].name_len);
        p->values[i].name = p->value_names[i];
        if (!p->value_names[i]) { rk_plan_free(p); fail(diag, RK_E_NOMEM, "clone", "out of memory"); return RK_E_NOMEM; }
    }
    memcpy(p->buffers, desc->buffers, desc->nbuffers * sizeof(*p->buffers));
    memcpy(p->views, desc->views, desc->nviews * sizeof(*p->views));
    for (uint32_t i = 0; i < desc->nactions; i++) {
        p->actions[i] = desc->actions[i];
        size_t origin_len = strnlen(desc->actions[i].origin,
                                    sizeof(desc->actions[i].origin) - 1u);
        memcpy(p->actions[i].origin, desc->actions[i].origin, origin_len);
        p->actions[i].origin[origin_len] = 0;
        p->actions[i].inputs = NULL;
        p->actions[i].outputs = NULL;
        p->actions[i].deps = NULL;
        if (desc->actions[i].ninputs) {
            p->action_inputs[i] = malloc(desc->actions[i].ninputs * sizeof(uint32_t));
            if (!p->action_inputs[i]) { rk_plan_free(p); fail(diag, RK_E_NOMEM, "clone", "out of memory"); return RK_E_NOMEM; }
            memcpy(p->action_inputs[i], desc->actions[i].inputs,
                   desc->actions[i].ninputs * sizeof(uint32_t));
            p->actions[i].inputs = p->action_inputs[i];
        }
        if (desc->actions[i].noutputs) {
            p->action_outputs[i] = malloc(desc->actions[i].noutputs * sizeof(uint32_t));
            if (!p->action_outputs[i]) { rk_plan_free(p); fail(diag, RK_E_NOMEM, "clone", "out of memory"); return RK_E_NOMEM; }
            memcpy(p->action_outputs[i], desc->actions[i].outputs,
                   desc->actions[i].noutputs * sizeof(uint32_t));
            p->actions[i].outputs = p->action_outputs[i];
        }
        if (desc->actions[i].ndeps) {
            p->action_deps[i] = malloc(desc->actions[i].ndeps * sizeof(uint32_t));
            if (!p->action_deps[i]) { rk_plan_free(p); fail(diag, RK_E_NOMEM, "clone", "out of memory"); return RK_E_NOMEM; }
            memcpy(p->action_deps[i], desc->actions[i].deps,
                   desc->actions[i].ndeps * sizeof(uint32_t));
            p->actions[i].deps = p->action_deps[i];
        }
    }
    for (uint32_t i = 0; i < desc->nsubmits; i++) {
        p->submits[i] = desc->submits[i];
        if (desc->submits[i].ntask_ids) {
            p->submit_tasks[i] = malloc(desc->submits[i].ntask_ids * sizeof(uint32_t));
            if (!p->submit_tasks[i]) { rk_plan_free(p); fail(diag, RK_E_NOMEM, "clone", "out of memory"); return RK_E_NOMEM; }
            memcpy(p->submit_tasks[i], desc->submits[i].task_ids,
                   desc->submits[i].ntask_ids * sizeof(uint32_t));
            p->submits[i].task_ids = p->submit_tasks[i];
        }
        if (desc->submits[i].ndeps) {
            p->submit_deps[i] = malloc(desc->submits[i].ndeps * sizeof(uint32_t));
            if (!p->submit_deps[i]) { rk_plan_free(p); fail(diag, RK_E_NOMEM, "clone", "out of memory"); return RK_E_NOMEM; }
            memcpy(p->submit_deps[i], desc->submits[i].deps,
                   desc->submits[i].ndeps * sizeof(uint32_t));
            p->submits[i].deps = p->submit_deps[i];
        }
    }
    memcpy(p->publish, desc->publish, desc->npublish * sizeof(*p->publish));
    p->task_submit = malloc(desc->nactions * sizeof(uint32_t));
    if (!p->task_submit) { rk_plan_free(p); fail(diag, RK_E_NOMEM, "clone", "out of memory"); return RK_E_NOMEM; }
    for (uint32_t i = 0; i < desc->nactions; i++) p->task_submit[i] = UINT32_MAX;
    for (uint32_t s = 0; s < desc->nsubmits; s++)
        for (uint32_t t = 0; t < desc->submits[s].ntask_ids; t++)
            p->task_submit[desc->submits[s].task_ids[t]] = s;
    *out = p;
    return 0;
}

void rk_plan_free(rk_plan *p) {
    if (!p) return;
    if (p->value_names) {
        for (uint32_t i = 0; i < p->nvalues; i++) free(p->value_names[i]);
    }
    if (p->action_inputs) for (uint32_t i = 0; i < p->nactions; i++) free(p->action_inputs[i]);
    if (p->action_outputs) for (uint32_t i = 0; i < p->nactions; i++) free(p->action_outputs[i]);
    if (p->action_deps) for (uint32_t i = 0; i < p->nactions; i++) free(p->action_deps[i]);
    if (p->submit_tasks) for (uint32_t i = 0; i < p->nsubmits; i++) free(p->submit_tasks[i]);
    if (p->submit_deps) for (uint32_t i = 0; i < p->nsubmits; i++) free(p->submit_deps[i]);
    free(p->value_names); free(p->values); free(p->buffers); free(p->views);
    free(p->actions); free(p->action_inputs); free(p->action_outputs);
    free(p->action_deps); free(p->submits); free(p->submit_tasks);
    free(p->submit_deps); free(p->publish); free(p->task_submit);
    free(p);
}

const rk_value_desc *rk_plan_value(const rk_plan *p, uint32_t id) {
    return p && id < p->nvalues ? &p->values[id] : NULL;
}
const rk_buffer_desc *rk_plan_buffer(const rk_plan *p, uint32_t id) {
    return p && id < p->nbuffers ? &p->buffers[id] : NULL;
}
const rk_view_desc *rk_plan_view(const rk_plan *p, uint32_t id) {
    return p && id < p->nviews ? &p->views[id] : NULL;
}
const rk_action_desc *rk_plan_action(const rk_plan *p, uint32_t id) {
    return p && id < p->nactions ? &p->actions[id] : NULL;
}

int rk_plan_clone_plan(const rk_plan *src, rk_plan **out, rk_diag *diag) {
    if (!src || !out) return fail(diag, RK_E_INVALID, "clone", "null internal plan argument");
    rk_plan_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.abi_version = RK_ABI_VERSION;
    desc.struct_size = (uint32_t)sizeof(desc);
    desc.schema_version = src->schema_version;
    desc.flags = src->flags;
    desc.profile_hash = src->profile_hash;
    desc.source_hash = src->source_hash;
    desc.values = src->values;
    desc.nvalues = src->nvalues;
    desc.buffers = src->buffers;
    desc.nbuffers = src->nbuffers;
    desc.views = src->views;
    desc.nviews = src->nviews;
    desc.actions = src->actions;
    desc.nactions = src->nactions;
    desc.submits = src->submits;
    desc.nsubmits = src->nsubmits;
    desc.publish = src->publish;
    desc.npublish = src->npublish;
    desc.device_budget_bytes = src->device_budget_bytes;
    desc.host_budget_bytes = src->host_budget_bytes;
    desc.claimed_device_bytes = src->claimed_device_bytes;
    desc.claimed_host_bytes = src->claimed_host_bytes;
    desc.runtime_abi_version = src->runtime_abi_version;
    return rk_plan_clone(&desc, out, diag);
}

int rk_plan_create(const rk_plan_desc *desc, rk_plan **out, rk_diag *diag) {
    if (!out) return fail(diag, RK_E_INVALID, "plan_create", "null out pointer");
    *out = NULL;
    return rk_plan_clone(desc, out, diag);
}

void rk_plan_destroy(rk_plan *plan) { rk_plan_free(plan); }
uint32_t rk_abi_version(void) { return RK_ABI_VERSION; }
