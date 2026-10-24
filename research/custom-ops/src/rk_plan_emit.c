// SPDX-License-Identifier: GPL-3.0-or-later
#include "rk_plan_internal.h"
#include <string.h>

int rk_emit_action(const rk_plan *plan, const co_view *views,
                   uint32_t action_id, co_program *out, rk_diag *diag) {
    if (!plan || !views || !out) return RK_E_INVALID;
    const rk_action_desc *a = rk_plan_action(plan, action_id);
    if (!a || a->kind != RK_ACTION_NPU_TASK) return RK_E_INVALID;
    memset(out, 0, sizeof(*out));
    int rc = 0;
    switch (a->template_id) {
    case RK_TEMPLATE_MM_F16:
        rc = co_emit_mm(out, views[a->inputs[0]], views[a->inputs[1]],
                        views[a->outputs[0]]);
        break;
    case RK_TEMPLATE_MM_ACC_F16:
        rc = co_emit_mm_acc(out, views[a->inputs[0]], views[a->inputs[1]],
                            views[a->inputs[2]], views[a->outputs[0]]);
        break;
    case RK_TEMPLATE_EW_ADD_F16:
        rc = co_emit_ew(out, views[a->inputs[0]], views[a->inputs[1]],
                        views[a->inputs[2]], views[a->outputs[0]], 0);
        break;
    case RK_TEMPLATE_EW_MUL_F16:
        rc = co_emit_ew(out, views[a->inputs[0]], views[a->inputs[1]],
                        views[a->inputs[2]], views[a->outputs[0]], 1);
        break;
    case RK_TEMPLATE_EW_MAX_F16:
        rc = co_emit_ew_max(out, views[a->inputs[0]], views[a->inputs[1]],
                            views[a->inputs[2]], views[a->outputs[0]]);
        break;
    case RK_TEMPLATE_LUT_SIGMOID_F16:
        rc = co_emit_lut(out, views[a->inputs[0]], views[a->outputs[0]], CO_SIGMOID);
        break;
    case RK_TEMPLATE_LUT_HARDSIGMOID_F16:
        rc = co_emit_lut(out, views[a->inputs[0]], views[a->outputs[0]], CO_HARDSIGMOID);
        break;
    case RK_TEMPLATE_LUT_GELU_GATE_F16:
        rc = co_emit_lut(out, views[a->inputs[0]], views[a->outputs[0]], CO_GELU_GATE);
        break;
    default:
        rc = RK_E_UNSUPPORTED;
        break;
    }
    if (rc) {
        rk_diag_set(diag, RK_E_DEVICE, "emit",
                    "custom-ops emitter rejected a validated action", "", action_id, rc);
        return RK_E_DEVICE;
    }
    if (!out->count || out->count > CO_MAX_WORDS || out->nreads > 3) {
        rk_diag_set(diag, RK_E_INVALID, "emit",
                    "emitter produced an invalid word/read count", "", action_id, 0);
        return RK_E_INVALID;
    }
    return 0;
}
