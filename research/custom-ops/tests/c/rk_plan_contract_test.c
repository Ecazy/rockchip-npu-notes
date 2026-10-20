// SPDX-License-Identifier: GPL-3.0-or-later
/* Pure layout/ABI checks.  This test never opens a device or allocates a BO. */
#include "rk_plan.h"
#include "custom_ops.h"
#include <stdio.h>
#include <string.h>

static unsigned checks, failures;
#define EXPECT(e) do { checks++; if (!(e)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #e); failures++; } } while (0)

int main(void) {
    EXPECT(rk_abi_version() == RK_ABI_VERSION);

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
    printf("RK_PLAN contract checks=%u failures=%u\n", checks, failures);
    return failures ? 1 : 0;
}
