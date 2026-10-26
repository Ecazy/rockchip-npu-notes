// SPDX-License-Identifier: GPL-3.0-or-later
/* Host-side guard for the per-task reference comparison.
 *
 * The chain probe compares N chained results against N *distinct* reference
 * buffers.  An earlier version downloaded every reference into one shared buffer
 * and then compared all results against it, which makes distinct tasks look
 * broken and identical tasks look fine.  This test pins the correct semantics
 * without a device:
 *
 *   - every task passes against its own slice;
 *   - swapping two tasks' results is detected;
 *   - comparing all results against the last task's slice (the old behaviour)
 *     is detected as wrong whenever the tasks genuinely differ.
 */
#include "rk_task_ref.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks, failures;
#define EXPECT(e) do { checks++; if (!(e)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #e); failures++; } } while (0)

static unsigned compare_all(unsigned char *const *results,
                            const unsigned char *ref_all,
                            size_t per_task_bytes, unsigned n) {
    unsigned bad = 0;
    for (unsigned i = 0; i < n; i++)
        if (!rk_task_ref_equal(results[i], ref_all, per_task_bytes, i)) bad++;
    return bad;
}

static void run_case(unsigned n) {
    const size_t bytes = 16 * 16 * 2;
    unsigned char *ref_all = malloc((size_t)n * bytes);
    unsigned char *results = malloc((size_t)n * bytes);
    unsigned char **ptrs = malloc(sizeof(*ptrs) * n);
    EXPECT(ref_all && results && ptrs);
    if (!ref_all || !results || !ptrs) {
        free(ref_all); free(results); free(ptrs);
        return;
    }
    /* Distinct, deterministic content per task, mirroring distinct inputs. */
    for (unsigned i = 0; i < n; i++) {
        for (size_t b = 0; b < bytes; b++)
            ref_all[(size_t)i * bytes + b] = (unsigned char)(i * 37u + b * 11u);
        memcpy(results + (size_t)i * bytes, ref_all + (size_t)i * bytes, bytes);
        ptrs[i] = results + (size_t)i * bytes;
    }
    EXPECT(compare_all(ptrs, ref_all, bytes, n) == 0);

    /* Swapping two neighbouring tasks must be detected (for n > 1). */
    if (n > 1) {
        memcpy(ptrs[0], ref_all + (size_t)1 * bytes, bytes);
        memcpy(ptrs[1], ref_all, bytes);
        EXPECT(compare_all(ptrs, ref_all, bytes, n) == 2);
        /* restore */
        memcpy(ptrs[0], ref_all, bytes);
        memcpy(ptrs[1], ref_all + (size_t)1 * bytes, bytes);
        EXPECT(compare_all(ptrs, ref_all, bytes, n) == 0);
    }

    /* The historical bug: one shared reference (the last task's slice) compared
     * against every result.  With distinct tasks this must report n-1 misses. */
    {
        unsigned bad = 0;
        for (unsigned i = 0; i < n; i++)
            if (memcmp(ptrs[i], ref_all + (size_t)(n - 1) * bytes, bytes) != 0)
                bad++;
        EXPECT(bad == n - 1);
    }

    /* A single corrupted byte in one task's own slice is detected. */
    if (n > 1) {
        ptrs[n / 2][3] ^= 0xffu;
        EXPECT(compare_all(ptrs, ref_all, bytes, n) == 1);
        ptrs[n / 2][3] ^= 0xffu;
        EXPECT(compare_all(ptrs, ref_all, bytes, n) == 0);
    }
    free(ref_all);
    free(results);
    free(ptrs);
}

int main(void) {
    run_case(2);
    run_case(4);
    run_case(32);
    /* Identical tasks: the old shared-reference form happens to pass, which is
     * exactly why it could hide a broken chain. */
    {
        const size_t bytes = 8;
        unsigned char ref_all[3 * 8];
        unsigned char got[8];
        memset(ref_all, 0x5a, sizeof(ref_all));
        memset(got, 0x5a, sizeof(got));
        EXPECT(rk_task_ref_equal(got, ref_all, bytes, 0) == 1);
        EXPECT(rk_task_ref_equal(got, ref_all, bytes, 2) == 1);
        got[0] = 0x00;
        EXPECT(rk_task_ref_equal(got, ref_all, bytes, 0) == 0);
    }
    printf("TASK_REF checks=%d failures=%d\n", checks, failures);
    return failures ? 1 : 0;
}
