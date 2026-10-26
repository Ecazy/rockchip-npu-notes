// SPDX-License-Identifier: GPL-3.0-or-later
/* Pure layout/ABI checks.  This test never opens a device or allocates a BO. */
#include "rk_plan.h"
#include "custom_ops.h"
#include <stdio.h>
#include <string.h>

static unsigned checks, failures;
#define EXPECT(e) do { checks++; if (!(e)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #e); failures++; } } while (0)

/* Machine-readable field layout so the Python ctypes definitions can be compared
 * field by field, not just by total size (equal sizes can hide swapped fields). */
#define RK_DUMP_FIELD(struct_name, field) \
    printf("ABI_FIELD %s %s %zu %zu\n", #struct_name, #field, \
           offsetof(struct_name, field), sizeof(((struct_name *)0)->field))

static void dump_layout(FILE *out) {
    (void)out;
    /* Generated from include/rk_plan.h: every field, plus every struct size. */
    printf("ABI_SIZE %s %zu\n", "rk_diag", sizeof(rk_diag));
    RK_DUMP_FIELD(rk_diag, abi_version);
    RK_DUMP_FIELD(rk_diag, struct_size);
    RK_DUMP_FIELD(rk_diag, code);
    RK_DUMP_FIELD(rk_diag, native_status);
    RK_DUMP_FIELD(rk_diag, action_id);
    RK_DUMP_FIELD(rk_diag, has_action);
    RK_DUMP_FIELD(rk_diag, stage);
    RK_DUMP_FIELD(rk_diag, message);
    RK_DUMP_FIELD(rk_diag, source);
    RK_DUMP_FIELD(rk_diag, profile);
    printf("ABI_SIZE %s %zu\n", "rk_value_desc", sizeof(rk_value_desc));
    RK_DUMP_FIELD(rk_value_desc, abi_version);
    RK_DUMP_FIELD(rk_value_desc, struct_size);
    RK_DUMP_FIELD(rk_value_desc, id);
    RK_DUMP_FIELD(rk_value_desc, role);
    RK_DUMP_FIELD(rk_value_desc, dtype);
    RK_DUMP_FIELD(rk_value_desc, rank);
    RK_DUMP_FIELD(rk_value_desc, shape);
    RK_DUMP_FIELD(rk_value_desc, nbytes);
    RK_DUMP_FIELD(rk_value_desc, name);
    RK_DUMP_FIELD(rk_value_desc, name_len);
    RK_DUMP_FIELD(rk_value_desc, reserved);
    printf("ABI_SIZE %s %zu\n", "rk_tensor_desc", sizeof(rk_tensor_desc));
    RK_DUMP_FIELD(rk_tensor_desc, abi_version);
    RK_DUMP_FIELD(rk_tensor_desc, struct_size);
    RK_DUMP_FIELD(rk_tensor_desc, dtype);
    RK_DUMP_FIELD(rk_tensor_desc, layout);
    RK_DUMP_FIELD(rk_tensor_desc, rows);
    RK_DUMP_FIELD(rk_tensor_desc, cols);
    RK_DUMP_FIELD(rk_tensor_desc, physical_rows);
    RK_DUMP_FIELD(rk_tensor_desc, physical_cols);
    RK_DUMP_FIELD(rk_tensor_desc, atom);
    RK_DUMP_FIELD(rk_tensor_desc, reserved);
    RK_DUMP_FIELD(rk_tensor_desc, bytes);
    printf("ABI_SIZE %s %zu\n", "rk_buffer_desc", sizeof(rk_buffer_desc));
    RK_DUMP_FIELD(rk_buffer_desc, abi_version);
    RK_DUMP_FIELD(rk_buffer_desc, struct_size);
    RK_DUMP_FIELD(rk_buffer_desc, id);
    RK_DUMP_FIELD(rk_buffer_desc, role);
    RK_DUMP_FIELD(rk_buffer_desc, slot_id);
    RK_DUMP_FIELD(rk_buffer_desc, alignment);
    RK_DUMP_FIELD(rk_buffer_desc, layout_id);
    RK_DUMP_FIELD(rk_buffer_desc, padding_state);
    RK_DUMP_FIELD(rk_buffer_desc, lifetime_first);
    RK_DUMP_FIELD(rk_buffer_desc, lifetime_last);
    RK_DUMP_FIELD(rk_buffer_desc, offset);
    RK_DUMP_FIELD(rk_buffer_desc, capacity);
    printf("ABI_SIZE %s %zu\n", "rk_view_desc", sizeof(rk_view_desc));
    RK_DUMP_FIELD(rk_view_desc, abi_version);
    RK_DUMP_FIELD(rk_view_desc, struct_size);
    RK_DUMP_FIELD(rk_view_desc, id);
    RK_DUMP_FIELD(rk_view_desc, buffer_id);
    RK_DUMP_FIELD(rk_view_desc, offset_bytes);
    RK_DUMP_FIELD(rk_view_desc, valid_region);
    RK_DUMP_FIELD(rk_view_desc, tensor);
    printf("ABI_SIZE %s %zu\n", "rk_action_desc", sizeof(rk_action_desc));
    RK_DUMP_FIELD(rk_action_desc, abi_version);
    RK_DUMP_FIELD(rk_action_desc, struct_size);
    RK_DUMP_FIELD(rk_action_desc, id);
    RK_DUMP_FIELD(rk_action_desc, kind);
    RK_DUMP_FIELD(rk_action_desc, template_id);
    RK_DUMP_FIELD(rk_action_desc, ninputs);
    RK_DUMP_FIELD(rk_action_desc, noutputs);
    RK_DUMP_FIELD(rk_action_desc, ndeps);
    RK_DUMP_FIELD(rk_action_desc, source_binding);
    RK_DUMP_FIELD(rk_action_desc, source_row_offset);
    RK_DUMP_FIELD(rk_action_desc, source_col_offset);
    RK_DUMP_FIELD(rk_action_desc, source_row_stride);
    RK_DUMP_FIELD(rk_action_desc, flags);
    RK_DUMP_FIELD(rk_action_desc, act_kind);
    RK_DUMP_FIELD(rk_action_desc, inputs);
    RK_DUMP_FIELD(rk_action_desc, outputs);
    RK_DUMP_FIELD(rk_action_desc, deps);
    RK_DUMP_FIELD(rk_action_desc, origin);
    printf("ABI_SIZE %s %zu\n", "rk_submit_desc", sizeof(rk_submit_desc));
    RK_DUMP_FIELD(rk_submit_desc, abi_version);
    RK_DUMP_FIELD(rk_submit_desc, struct_size);
    RK_DUMP_FIELD(rk_submit_desc, id);
    RK_DUMP_FIELD(rk_submit_desc, mode);
    RK_DUMP_FIELD(rk_submit_desc, core_policy);
    RK_DUMP_FIELD(rk_submit_desc, ntask_ids);
    RK_DUMP_FIELD(rk_submit_desc, ndeps);
    RK_DUMP_FIELD(rk_submit_desc, task_ids);
    RK_DUMP_FIELD(rk_submit_desc, deps);
    printf("ABI_SIZE %s %zu\n", "rk_publish_desc", sizeof(rk_publish_desc));
    RK_DUMP_FIELD(rk_publish_desc, abi_version);
    RK_DUMP_FIELD(rk_publish_desc, struct_size);
    RK_DUMP_FIELD(rk_publish_desc, binding_id);
    RK_DUMP_FIELD(rk_publish_desc, view_id);
    RK_DUMP_FIELD(rk_publish_desc, dst_row_offset);
    RK_DUMP_FIELD(rk_publish_desc, dst_col_offset);
    RK_DUMP_FIELD(rk_publish_desc, rows);
    RK_DUMP_FIELD(rk_publish_desc, cols);
    printf("ABI_SIZE %s %zu\n", "rk_plan_desc", sizeof(rk_plan_desc));
    RK_DUMP_FIELD(rk_plan_desc, abi_version);
    RK_DUMP_FIELD(rk_plan_desc, struct_size);
    RK_DUMP_FIELD(rk_plan_desc, schema_version);
    RK_DUMP_FIELD(rk_plan_desc, flags);
    RK_DUMP_FIELD(rk_plan_desc, profile_hash);
    RK_DUMP_FIELD(rk_plan_desc, source_hash);
    RK_DUMP_FIELD(rk_plan_desc, values);
    RK_DUMP_FIELD(rk_plan_desc, nvalues);
    RK_DUMP_FIELD(rk_plan_desc, reserved0);
    RK_DUMP_FIELD(rk_plan_desc, buffers);
    RK_DUMP_FIELD(rk_plan_desc, nbuffers);
    RK_DUMP_FIELD(rk_plan_desc, reserved1);
    RK_DUMP_FIELD(rk_plan_desc, views);
    RK_DUMP_FIELD(rk_plan_desc, nviews);
    RK_DUMP_FIELD(rk_plan_desc, reserved2);
    RK_DUMP_FIELD(rk_plan_desc, actions);
    RK_DUMP_FIELD(rk_plan_desc, nactions);
    RK_DUMP_FIELD(rk_plan_desc, reserved3);
    RK_DUMP_FIELD(rk_plan_desc, submits);
    RK_DUMP_FIELD(rk_plan_desc, nsubmits);
    RK_DUMP_FIELD(rk_plan_desc, reserved4);
    RK_DUMP_FIELD(rk_plan_desc, publish);
    RK_DUMP_FIELD(rk_plan_desc, npublish);
    RK_DUMP_FIELD(rk_plan_desc, reserved5);
    RK_DUMP_FIELD(rk_plan_desc, device_budget_bytes);
    RK_DUMP_FIELD(rk_plan_desc, host_budget_bytes);
    RK_DUMP_FIELD(rk_plan_desc, claimed_device_bytes);
    RK_DUMP_FIELD(rk_plan_desc, claimed_host_bytes);
    RK_DUMP_FIELD(rk_plan_desc, runtime_abi_version);
    RK_DUMP_FIELD(rk_plan_desc, reserved6);
    printf("ABI_SIZE %s %zu\n", "rk_binding", sizeof(rk_binding));
    RK_DUMP_FIELD(rk_binding, abi_version);
    RK_DUMP_FIELD(rk_binding, struct_size);
    RK_DUMP_FIELD(rk_binding, id);
    RK_DUMP_FIELD(rk_binding, dtype);
    RK_DUMP_FIELD(rk_binding, rank);
    RK_DUMP_FIELD(rk_binding, shape);
    RK_DUMP_FIELD(rk_binding, data);
    RK_DUMP_FIELD(rk_binding, nbytes);
    printf("ABI_SIZE %s %zu\n", "rk_run_report", sizeof(rk_run_report));
    RK_DUMP_FIELD(rk_run_report, abi_version);
    RK_DUMP_FIELD(rk_run_report, struct_size);
    RK_DUMP_FIELD(rk_run_report, task_count);
    RK_DUMP_FIELD(rk_run_report, submit_count);
    RK_DUMP_FIELD(rk_run_report, fail_action_id);
    RK_DUMP_FIELD(rk_run_report, numeric_mode);
    RK_DUMP_FIELD(rk_run_report, upload_calls);
    RK_DUMP_FIELD(rk_run_report, upload_bytes);
    RK_DUMP_FIELD(rk_run_report, download_calls);
    RK_DUMP_FIELD(rk_run_report, download_bytes);
    RK_DUMP_FIELD(rk_run_report, pack_us);
    RK_DUMP_FIELD(rk_run_report, emit_us);
    RK_DUMP_FIELD(rk_run_report, submit_wait_us);
    RK_DUMP_FIELD(rk_run_report, output_sync_us);
    /* Extended attribution lives in its own struct behind its own entry point,
     * so rk_run_report stays exactly the original ABI surface. */
    printf("ABI_SIZE %s %zu\n", "rk_run_attribution", sizeof(rk_run_attribution));
    RK_DUMP_FIELD(rk_run_attribution, abi_version);
    RK_DUMP_FIELD(rk_run_attribution, struct_size);
    RK_DUMP_FIELD(rk_run_attribution, task_count);
    RK_DUMP_FIELD(rk_run_attribution, submit_count);
    RK_DUMP_FIELD(rk_run_attribution, fail_action_id);
    RK_DUMP_FIELD(rk_run_attribution, numeric_mode);
    RK_DUMP_FIELD(rk_run_attribution, upload_calls);
    RK_DUMP_FIELD(rk_run_attribution, upload_bytes);
    RK_DUMP_FIELD(rk_run_attribution, download_calls);
    RK_DUMP_FIELD(rk_run_attribution, download_bytes);
    RK_DUMP_FIELD(rk_run_attribution, validate_us);
    RK_DUMP_FIELD(rk_run_attribution, pack_us);
    RK_DUMP_FIELD(rk_run_attribution, emit_us);
    RK_DUMP_FIELD(rk_run_attribution, command_us);
    RK_DUMP_FIELD(rk_run_attribution, submit_alloc_us);
    RK_DUMP_FIELD(rk_run_attribution, submit_wait_us);
    RK_DUMP_FIELD(rk_run_attribution, output_sync_us);
    RK_DUMP_FIELD(rk_run_attribution, download_us);
    RK_DUMP_FIELD(rk_run_attribution, guard_us);
    RK_DUMP_FIELD(rk_run_attribution, publish_us);
    RK_DUMP_FIELD(rk_run_attribution, alloc_us);
    RK_DUMP_FIELD(rk_run_attribution, total_us);
    RK_DUMP_FIELD(rk_run_attribution, clock_us);
    /* Not a ctypes mirror: the emission array size that the Host requirement
     * formula must use.  Python pins its own constant against this value. */
    printf("ABI_CONST %s %zu\n", "co_program", sizeof(co_program));
    printf("ABI_SIZE %s %zu\n", "rk_group_report", sizeof(rk_group_report));
    RK_DUMP_FIELD(rk_group_report, abi_version);
    RK_DUMP_FIELD(rk_group_report, struct_size);
    RK_DUMP_FIELD(rk_group_report, repeat_count);
    RK_DUMP_FIELD(rk_group_report, repeat_completed);
    RK_DUMP_FIELD(rk_group_report, fail_repeat);
    RK_DUMP_FIELD(rk_group_report, fail_action_id);
    RK_DUMP_FIELD(rk_group_report, task_count);
    RK_DUMP_FIELD(rk_group_report, submit_count);
    RK_DUMP_FIELD(rk_group_report, numeric_mode);
    RK_DUMP_FIELD(rk_group_report, reserved0);
    RK_DUMP_FIELD(rk_group_report, upload_calls);
    RK_DUMP_FIELD(rk_group_report, upload_bytes);
    RK_DUMP_FIELD(rk_group_report, download_calls);
    RK_DUMP_FIELD(rk_group_report, download_bytes);
    RK_DUMP_FIELD(rk_group_report, validate_us);
    RK_DUMP_FIELD(rk_group_report, pack_us);
    RK_DUMP_FIELD(rk_group_report, emit_us);
    RK_DUMP_FIELD(rk_group_report, command_us);
    RK_DUMP_FIELD(rk_group_report, submit_alloc_us);
    RK_DUMP_FIELD(rk_group_report, submit_wait_us);
    RK_DUMP_FIELD(rk_group_report, output_sync_us);
    RK_DUMP_FIELD(rk_group_report, download_us);
    RK_DUMP_FIELD(rk_group_report, guard_us);
    RK_DUMP_FIELD(rk_group_report, publish_us);
    RK_DUMP_FIELD(rk_group_report, alloc_us);
    RK_DUMP_FIELD(rk_group_report, total_us);
    RK_DUMP_FIELD(rk_group_report, clock_us);
}

int main(void) {
    EXPECT(rk_abi_version() == RK_ABI_VERSION);
    /* rk_run_report is frozen at its original 88 bytes: the extended entry
     * point exists precisely so this struct never grows under old callers. */
    EXPECT(sizeof(rk_run_report) == 88u);
    EXPECT(offsetof(rk_run_report, pack_us) == 56u);
    EXPECT(offsetof(rk_run_report, output_sync_us) == 80u);
    /* The extended declarations must be self-describing and non-empty. */
    EXPECT(RK_RUN_ATTRIBUTION_V1_SIZE == sizeof(rk_run_attribution));
    EXPECT(RK_GROUP_REPORT_V1_SIZE == sizeof(rk_group_report));
    EXPECT(RK_RUN_ATTRIBUTION_V1_SIZE > sizeof(rk_run_report));
    EXPECT(RK_SCHEMA_VERSION == 2u);

    co_tensor cube13x33;
    EXPECT(co_tensor_matrix(&cube13x33, 13, 33, CO_F16) == 0);
    EXPECT(cube13x33.physical_rows == 16);
    EXPECT(cube13x33.physical_cols == 64);
    EXPECT(cube13x33.bytes == 16u * 64u * 2u);
    EXPECT(co_index(&cube13x33, 0, 0) == 0);
    EXPECT(co_index(&cube13x33, 1, 0) == 8);
    EXPECT(co_index(&cube13x33, 0, 8) == 16u * 8u);

    co_tensor full, slice0, slice1;
    size_t off0 = 1, off1 = 1;
    EXPECT(co_tensor_input(&full, 13, 257) == 0);
    EXPECT(full.physical_rows == 16);
    EXPECT(full.physical_cols == 288);
    EXPECT(co_tensor_input_slice(&slice0, &full, 0, 256, &off0) == 0);
    EXPECT(co_tensor_input_slice(&slice1, &full, 256, 1, &off1) == 0);
    EXPECT(off0 == 0);
    EXPECT(off1 == 16u * 256u * 2u);
    EXPECT(slice0.physical_cols == 256 && slice0.bytes == 16u * 256u * 2u);
    EXPECT(slice1.physical_cols == 32 && slice1.bytes == 16u * 32u * 2u);

    co_tensor weight;
    EXPECT(co_tensor_weight(&weight, 33, 256) == 0);
    EXPECT(weight.layout == CO_WEIGHT);
    EXPECT(weight.physical_rows == 64);
    EXPECT(weight.physical_cols == 256);
    EXPECT(weight.bytes == 64u * 256u * 2u);
    EXPECT(co_index(&weight, 0, 0) == 0);
    EXPECT(co_index(&weight, 16, 0) == 4096u);
    EXPECT(co_index(&weight, 0, 32) == 16u * 32u);

    /* ABI structure sizes are intentionally part of this test: ctypes must be
     * compared against these same values rather than hand-calculated offsets. */
    printf("ABI %u value=%zu tensor=%zu buffer=%zu view=%zu action=%zu "
           "submit=%zu publish=%zu plan=%zu binding=%zu report=%zu diag=%zu\n",
           rk_abi_version(), sizeof(rk_value_desc), sizeof(rk_tensor_desc),
           sizeof(rk_buffer_desc), sizeof(rk_view_desc), sizeof(rk_action_desc),
           sizeof(rk_submit_desc), sizeof(rk_publish_desc), sizeof(rk_plan_desc),
           sizeof(rk_binding), sizeof(rk_run_report), sizeof(rk_diag));
    dump_layout(stdout);
    printf("RK_PLAN contract checks=%u failures=%u\n", checks, failures);
    return failures ? 1 : 0;
}
