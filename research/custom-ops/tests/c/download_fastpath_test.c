// SPDX-License-Identifier: GPL-3.0-or-later
/* Exact row-order equivalence for the CUBE-F16 download fast path. */
#include "custom_ops.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures;
#define CHECK(e) do { if (!(e)) { fprintf(stderr, "FAIL line %d: %s\n", \
    __LINE__, #e); failures++; } } while (0)

int main(void) {
    co_tensor cube;
    CHECK(co_tensor_matrix(&cube, 17, 37, CO_F16) == 0);
    co_buffer buf = {0};
    CHECK(co_buffer_alloc(0, cube.bytes, &buf) == 0);
    co_view view;
    CHECK(co_view_init(&view, &buf, 0, &cube) == 0);
    _Float16 *base = (_Float16 *)buf.bo.ptr;

    /* Give padding deliberately non-zero values so an accidental physical
     * rectangle copy cannot match the logical-row reference. */
    for (unsigned r = 0; r < cube.physical_rows; r++)
        for (unsigned c = 0; c < cube.physical_cols; c++)
            base[co_index(&cube, r, c)] = (_Float16)(0.25f + r * 0.01f + c);

    enum { OUT_WORDS = 2048 };
    _Float16 out[OUT_WORDS];
    for (unsigned i = 0; i < OUT_WORDS; i++) out[i] = (_Float16)-7.0f;
    CHECK(co_download(&view, out) == 0);
    unsigned logical = cube.rows * cube.cols;
    for (unsigned r = 0; r < cube.rows; r++)
        for (unsigned c = 0; c < cube.cols; c++) {
            _Float16 want = base[co_index(&cube, r, c)];
            _Float16 got = out[r * cube.cols + c];
            CHECK(memcmp(&want, &got, sizeof(want)) == 0);
        }
    for (unsigned i = logical; i < OUT_WORDS; i++)
        CHECK(out[i] == (_Float16)-7.0f);

    co_buffer_free(&buf);
    printf("DOWNLOAD_FASTPATH checks=%u failures=%d\n", logical, failures);
    return failures ? 1 : 0;
}
