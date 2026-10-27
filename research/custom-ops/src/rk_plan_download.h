// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef RK_PLAN_DOWNLOAD_H
#define RK_PLAN_DOWNLOAD_H
/* Host-side helpers for the download action.
 *
 * Both are pure functions of their inputs and are kept in a header so the exact
 * behaviour (which values count as non-finite, which elements a publish writes)
 * can be unit tested exhaustively instead of only through a device.
 */
#include <stdint.h>
#include <string.h>

/* FP16 finiteness, four lanes at a time.
 *
 * A half is finite iff its 5-bit exponent is not all ones, i.e. iff
 * ((h & 0x7fff) + 0x0400) does not carry into the sign bit: the largest finite
 * exponent (0x7bff) becomes 0x7fff, while 0x7c00 (Inf) and any NaN become
 * >= 0x8000.  Clearing the sign bit first keeps negative values from tripping
 * the carry test, and every lane stays <= 0x7fff so a lane addition can never
 * carry into its neighbour. */
static inline int rk_f16_word_all_finite(uint64_t word) {
    uint64_t a = word & 0x7fff7fff7fff7fffull;
    uint64_t z = a + 0x0400040004000400ull;
    return (z & 0x8000800080008000ull) == 0;
}

static inline int rk_f16_all_finite(const void *data, uint64_t count) {
    const unsigned char *bytes = (const unsigned char *)data;
    uint64_t i = 0;
    for (; i + 4u <= count; i += 4u) {
        uint64_t word;
        memcpy(&word, bytes + (size_t)i * sizeof(uint16_t), sizeof(word));
        if (!rk_f16_word_all_finite(word)) {
            for (unsigned k = 0; k < 4; k++) {
                uint16_t lane;
                memcpy(&lane, bytes + ((size_t)i + k) * sizeof(lane), sizeof(lane));
                if ((lane & 0x7c00u) == 0x7c00u) return 0;
            }
        }
    }
    for (; i < count; i++) {
        uint16_t lane;
        memcpy(&lane, bytes + (size_t)i * sizeof(lane), sizeof(lane));
        if ((lane & 0x7c00u) == 0x7c00u) return 0;
    }
    return 1;
}

/* Copy one publish rectangle from a downloaded tile into the staged output.
 * The staging row is contiguous, so each source row is one run; the element loop
 * it replaces wrote exactly these elements. */
static inline void rk_publish_rows(char *staging, const uint16_t *tile,
                                   unsigned rows, unsigned cols,
                                   unsigned dst_row, unsigned dst_col,
                                   unsigned value_cols) {
    uint16_t *dst = (uint16_t *)staging;
    for (unsigned r = 0; r < rows; r++) {
        uint16_t *drow = dst + ((size_t)dst_row + r) * value_cols + dst_col;
        memcpy(drow, tile + (size_t)r * cols, (size_t)cols * sizeof(uint16_t));
    }
}

#endif
