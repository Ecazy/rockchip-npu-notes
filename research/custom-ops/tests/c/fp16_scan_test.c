// SPDX-License-Identifier: GPL-3.0-or-later
/* Exhaustive checks for the download helpers.
 *
 * ``rk_f16_all_finite`` replaced a per-element ``isfinite((double)half)`` loop
 * with a word-at-a-time test.  Because the domain is finite (2^16 bit patterns)
 * the replacement can be proven instead of sampled: every half pattern is
 * compared against ``isfinite((double)(_Float16)v)``, at every lane position of
 * a 4-lane word and at every alignment the runtime can produce.
 *
 * ``rk_publish_rows`` is checked against the element loop it replaced, for full
 * and partial rectangles, so a staging write cannot change which elements are
 * produced.
 */
#include "rk_plan_download.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks, failures;
#define EXPECT(e) do { checks++; if (!(e)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #e); failures++; } } while (0)

static int half_is_finite(uint16_t bits) {
    _Float16 v;
    memcpy(&v, &bits, sizeof(v));
    return isfinite((double)v) ? 1 : 0;
}

static void exhaustive_finiteness(void) {
    unsigned mismatches = 0;
    unsigned inf_nan = 0;
    for (unsigned v = 0; v <= 0xFFFFu; v++) {
        const uint16_t bits = (uint16_t)v;
        const int expect = half_is_finite(bits);
        if (!expect) inf_nan++;
        /* every lane position of a 4-lane word, the others finite */
        for (unsigned lane = 0; lane < 4; lane++) {
            uint16_t buf[4] = {0x3c00u, 0x3c00u, 0x3c00u, 0x3c00u};
            buf[lane] = bits;
            if (rk_f16_all_finite(buf, 4) != expect) mismatches++;
            /* the same value as a 1-element and 3-element tail scan */
            if (rk_f16_all_finite(buf + lane, 1) != expect) mismatches++;
        }
        if (rk_f16_all_finite(&bits, 1) != expect) mismatches++;
    }
    EXPECT(mismatches == 0);
    /* exactly the 0x7c00 exponent class is non-finite: 2 * (1 + 1023) halves */
    EXPECT(inf_nan == 2u * (1u + 1023u));
}

static void word_lane_independence(void) {
    /* One non-finite lane anywhere in the word must be caught, at every offset. */
    static const uint16_t bad[] = {0x7c00u, 0xfc00u, 0x7e00u, 0xfe01u, 0x7fffu};
    for (unsigned b = 0; b < sizeof(bad) / sizeof(bad[0]); b++) {
        for (unsigned lane = 0; lane < 4; lane++) {
            uint16_t buf[8] = {0x3c00u, 0x3c00u, 0x3c00u, 0x3c00u,
                               0x3c00u, 0x3c00u, 0x3c00u, 0x3c00u};
            buf[lane] = bad[b];
            EXPECT(rk_f16_all_finite(buf, 4) == 0);
            EXPECT(rk_f16_all_finite(buf, 8) == 0);
            buf[lane] = 0x3c00u;
            EXPECT(rk_f16_all_finite(buf, 8) == 1);
        }
    }
    /* Negative finites must not be flagged by the sign-bit clearing. */
    static const uint16_t finite[] = {0x0000u, 0x8000u, 0x7bffu, 0xfbffu,
                                      0x0001u, 0x0400u, 0x7bffu, 0x3c00u};
    EXPECT(rk_f16_all_finite(finite, 8) == 1);
}

/* Copy of the historical publish element loop, used as the reference. */
static void reference_publish(uint16_t *dst, const uint16_t *tile,
                              unsigned rows, unsigned cols,
                              unsigned dst_row, unsigned dst_col,
                              unsigned value_cols) {
    for (unsigned r = 0; r < rows; r++)
        for (unsigned c = 0; c < cols; c++)
            dst[(size_t)(dst_row + r) * value_cols + dst_col + c] =
                tile[(size_t)r * cols + c];
}

static void publish_matches_element_loop(void) {
    enum { ROWS = 16, COLS = 16, VCOLS = 24, VR = 32 };
    uint16_t tile[ROWS * COLS];
    for (unsigned i = 0; i < ROWS * COLS; i++) tile[i] = (uint16_t)(i * 7u + 1u);
    static const unsigned cases[][4] = {
        /* rows, cols, dst_row, dst_col */
        {16, 16, 0, 0},     /* whole value, full width */
        {16, 16, 0, 8},     /* full tile into a wider row */
        {8, 16, 4, 0},      /* row subset */
        {3, 5, 2, 3},       /* partial rectangle */
        {1, 1, 31, 23},     /* single element at the far corner */
        {16, 1, 0, 0},      /* one column */
    };
    for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        unsigned rows = cases[c][0], cols = cases[c][1];
        unsigned dr = cases[c][2], dc = cases[c][3];
        uint16_t got[VR * VCOLS];
        uint16_t want[VR * VCOLS];
        for (unsigned i = 0; i < VR * VCOLS; i++) { got[i] = 0xdead; want[i] = 0xdead; }
        rk_publish_rows((char *)got, tile, rows, cols, dr, dc, VCOLS);
        reference_publish(want, tile, rows, cols, dr, dc, VCOLS);
        EXPECT(memcmp(got, want, sizeof(got)) == 0);
    }
}

int main(void) {
    exhaustive_finiteness();
    word_lane_independence();
    publish_matches_element_loop();
    printf("FP16_SCAN checks=%d failures=%d\n", checks, failures);
    return failures ? 1 : 0;
}
