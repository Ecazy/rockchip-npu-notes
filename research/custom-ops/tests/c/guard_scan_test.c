// SPDX-License-Identifier: GPL-3.0-or-later
/* Guard-scan coverage test.
 *
 * ``co_guard_check`` used to walk the 32 KiB slack region one byte at a time,
 * which made it the largest Host cost of a small plan call.  The scan is now
 * word-at-a-time.  Coverage must be identical, so this test corrupts a single
 * byte at many offsets -- including unaligned ones, both ends of each region and
 * every word position -- and requires the check to fail each time.  It also
 * proves an untouched buffer still passes, so the faster check cannot become a
 * no-op.
 */
#include "custom_ops.h"
#include <stdio.h>
#include <string.h>

static int checks, failures;
#define EXPECT(e) do { checks++; if (!(e)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #e); failures++; } } while (0)

int main(void) {
    co_tensor cube;
    if (co_tensor_matrix(&cube, 16, 16, CO_F16) != 0) return 2;
    co_buffer buf = {0};
    if (co_buffer_alloc(0, cube.bytes, &buf) != 0) return 2;
    co_view view;
    EXPECT(co_view_init(&view, &buf, 0, &cube) == 0);

    unsigned char *base = (unsigned char *)buf.bo.ptr;
    unsigned char *slack = base + buf.payload;
    unsigned char *guard = base + buf.payload + CO_SLACK;

    /* A freshly allocated buffer passes: the check is not vacuously failing. */
    EXPECT(co_guard_check(&buf) == 0);

    /* Every byte of both protected regions must be covered.  Offsets chosen to
     * hit the head, the tail, every intra-word position and the word boundary. */
    static const unsigned offsets[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17};
    for (unsigned i = 0; i < sizeof(offsets) / sizeof(offsets[0]); i++) {
        unsigned off = offsets[i];
        slack[off] ^= 0xffu;
        EXPECT(co_guard_check(&buf) == ROCKET_E_DEVICE);
        slack[off] ^= 0xffu;
        guard[off] ^= 0xffu;
        EXPECT(co_guard_check(&buf) == ROCKET_E_DEVICE);
        guard[off] ^= 0xffu;
    }
    /* Last byte of the slack region and last byte of the guard region. */
    slack[CO_SLACK - 1] ^= 0xffu;
    EXPECT(co_guard_check(&buf) == ROCKET_E_DEVICE);
    slack[CO_SLACK - 1] ^= 0xffu;
    guard[CO_GUARD - 1] ^= 0xffu;
    EXPECT(co_guard_check(&buf) == ROCKET_E_DEVICE);
    guard[CO_GUARD - 1] ^= 0xffu;

    /* Restoring every byte restores a clean verdict. */
    EXPECT(co_guard_check(&buf) == 0);
    /* A payload write is outside the slack region and must not trip the check. */
    base[0] ^= 0x5au;
    EXPECT(co_guard_check(&buf) == 0);
    base[0] ^= 0x5au;

    co_buffer_free(&buf);
    printf("GUARD_SCAN checks=%d failures=%d\n", checks, failures);
    return failures ? 1 : 0;
}
