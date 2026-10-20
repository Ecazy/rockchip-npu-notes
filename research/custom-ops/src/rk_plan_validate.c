// SPDX-License-Identifier: GPL-3.0-or-later
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

static int tensor_valid(const rk_tensor_desc *t) {
    if (!t || !struct_ok(t->abi_version, t->struct_size, sizeof(*t))) return 0;
    if (t->dtype != RK_DTYPE_F16 || !valid_layout(t->layout)) return 0;
    if (!t->rows || !t->cols || !t->physical_rows || !t->physical_cols) return 0;
    if (t->rows > t->physical_rows || t->cols > t->physical_cols) return 0;
    if (t->physical_rows > 256 || t->physical_cols > 65520) return 0;
    if (t->layout == RK_LAYOUT_LINEAR_F16) {
        if (t->atom != 1 || t->physical_rows != 1) return 0;
    } else if (t->layout == RK_LAYOUT_WEIGHT_F16) {
        if (t->atom != 32 || t->physical_rows % 16 || t->physical_cols % 32) return 0;
    } else {
        if (t->atom != 8 || t->physical_cols > CO_MAX_K) return 0;
    }
    if (t->physical_cols % t->atom) return 0;
    if ((uint64_t)t->physical_rows * t->physical_cols > UINT64_MAX / 2u) return 0;
    if (t->bytes != (uint64_t)t->physical_rows * t->physical_cols * 2u) return 0;
    return 1;
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
    if (src->dtype != RK_DTYPE_F16 || src->rank != 2)
        return fail(diag, RK_E_NUMERIC_CONTRACT, "validate", "upload source must be rank-2 FP16");
    if (!tensor_valid(&out->tensor))
        return fail(diag, RK_E_SHAPE, "validate", "upload destination tensor is invalid");
    uint64_t rows = out->tensor.rows, cols = out->tensor.cols;
    uint32_t transpose = (a->flags & RK_ACTION_FLAG_TRANSPOSE_SOURCE) != 0;
    if (a->source_row_stride == 0)
        return fail(diag, RK_E_SHAPE, "validate", "upload source row stride is zero");
    uint64_t sr = transpose ? cols : rows;
    uint64_t sc = transpose ? rows : cols;
    if ((uint64_t)a->source_row_offset + sr > src->shape[0] ||
        (uint64_t)a->source_col_offset + sc > src->shape[1])
        return fail(diag, RK_E_SHAPE, "validate", "upload source region exceeds binding shape");
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
    if (src->tensor.rows < dst->shape[0] || src->tensor.cols < dst->shape[1])
        return fail(diag, RK_E_SHAPE, "validate", "download view is smaller than output binding");
    return 0;
}

static int validate_template(const rk_plan_desc *desc, const rk_action_desc *a, rk_diag *diag) {
    if (!a->inputs || !a->outputs)
        return fail(diag, RK_E_SCHEMA, "validate", "NPU task has null input/output array");
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
               a->template_id == RK_TEMPLATE_EW_MUL_F16) {
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

int rk_plan_validate_desc(const rk_plan_desc *desc, rk_diag *diag) {
    rk_diag_clear(diag);
    if (!desc) return fail(diag, RK_E_INVALID, "validate", "null plan descriptor");
    if (!struct_ok(desc->abi_version, desc->struct_size, sizeof(*desc)))
        return fail(diag, RK_E_SCHEMA, "validate", "rk_plan_desc ABI/size mismatch");
    if (desc->schema_version != RK_SCHEMA_VERSION)
        return fail(diag, RK_E_SCHEMA, "validate", "unsupported plan schema version");
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
        if (v->id != i || !valid_role(v->role) || v->dtype != RK_DTYPE_F16 || v->rank != 2)
            return fail(diag, RK_E_SCHEMA, "validate", "value id/role/dtype/rank invalid");
        if (!v->name || !v->name_len || v->name_len > 128)
            return fail(diag, RK_E_SCHEMA, "validate", "value name invalid");
        if (v->shape[0] == 0 || v->shape[1] == 0 ||
            v->shape[0] > 4096 || v->shape[1] > 4096)
            return fail(diag, RK_E_SHAPE, "validate", "value shape out of range");
        if (v->nbytes != (uint64_t)v->shape[0] * v->shape[1] * 2u)
            return fail(diag, RK_E_SHAPE, "validate", "value byte size mismatch");
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
        for (uint32_t j = 0; j < s->ntask_ids; j++) {
            uint32_t task = s->task_ids[j];
            const rk_action_desc *a = action_at(desc, task);
            if (!a || a->kind != RK_ACTION_NPU_TASK)
                { free(task_submit); return fail(diag, RK_E_SCHEMA, "validate", "submit contains non-NPU task"); }
            if (task_submit[task] != UINT32_MAX)
                { free(task_submit); return fail(diag, RK_E_SCHEMA, "validate", "task assigned to multiple submits"); }
            task_submit[task] = i;
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

    /* Publish set. */
    uint32_t output_count = 0;
    for (uint32_t i = 0; i < desc->nvalues; i++)
        if (desc->values[i].role == RK_ROLE_OUTPUT) output_count++;
    if (desc->npublish != output_count)
        return fail(diag, RK_E_SCHEMA, "validate", "publish count must match outputs");
    for (uint32_t i = 0; i < desc->npublish; i++) {
        const rk_publish_desc *p = &desc->publish[i];
        if (!struct_ok(p->abi_version, p->struct_size, sizeof(*p)))
            return fail(diag, RK_E_SCHEMA, "validate", "publish descriptor ABI/size mismatch");
        for (uint32_t j = 0; j < i; j++)
            if (desc->publish[j].binding_id == p->binding_id ||
                desc->publish[j].view_id == p->view_id)
                return fail(diag, RK_E_SCHEMA, "validate", "duplicate publish entry");
        const rk_value_desc *v = value_at(desc, p->binding_id);
        const rk_view_desc *view = view_at(desc, p->view_id);
        if (!v || !view || v->role != RK_ROLE_OUTPUT ||
            view->tensor.rows < v->shape[0] || view->tensor.cols < v->shape[1])
            return fail(diag, RK_E_SHAPE, "validate", "publish target invalid");
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
    if (desc->claimed_device_bytes && desc->claimed_device_bytes < device)
        return fail(diag, RK_E_BUDGET, "validate", "claimed device bytes understate slots/command workspace");
    if (desc->device_budget_bytes && device > desc->device_budget_bytes)
        return fail(diag, RK_E_BUDGET, "validate", "device budget exceeded");
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
                                    sizeof(desc->actions[i].origin));
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
    free(p->submit_deps); free(p->publish);
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
    return rk_plan_clone(&desc, out, diag);
}

int rk_plan_create(const rk_plan_desc *desc, rk_plan **out, rk_diag *diag) {
    if (!out) return fail(diag, RK_E_INVALID, "plan_create", "null out pointer");
    *out = NULL;
    return rk_plan_clone(desc, out, diag);
}

void rk_plan_destroy(rk_plan *plan) { rk_plan_free(plan); }
uint32_t rk_abi_version(void) { return RK_ABI_VERSION; }
