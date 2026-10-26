// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef RK_TASK_REF_H
#define RK_TASK_REF_H
/* Per-task reference comparison for the static-repeat chain probe.
 *
 * N independent tasks each produce their own result, so the reference set is N
 * *distinct* buffers stored contiguously and every result must be compared with
 * its OWN slice.  Comparing every result against one shared reference buffer
 * (the last task installed) is a silent bug: distinct tasks then "fail" while
 * identical tasks "pass".  Keeping the slice arithmetic in one place makes that
 * testable without a device.
 */
#include <stddef.h>
#include <string.h>

static inline const unsigned char *rk_task_ref_slice(const unsigned char *ref_all,
                                                     size_t per_task_bytes,
                                                     unsigned index) {
    return ref_all + (size_t)index * per_task_bytes;
}

static inline int rk_task_ref_equal(const unsigned char *got,
                                    const unsigned char *ref_all,
                                    size_t per_task_bytes, unsigned index) {
    return memcmp(got, rk_task_ref_slice(ref_all, per_task_bytes, index),
                  per_task_bytes) == 0;
}

#endif
