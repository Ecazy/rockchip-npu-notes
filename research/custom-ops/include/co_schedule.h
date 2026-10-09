// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef CO_SCHEDULE_H
#define CO_SCHEDULE_H
#include "custom_ops.h"

/* Vendor DRM path only. Mask 0=AUTO (all three cores), 1/2/4=strict single core.
 * 3/5/6/7 are ganged submissions, not candidate sets, and are rejected here.
 * A dependent chain is indivisible and stays on one core; submit independent
 * tasks/jobs separately to give them different core assignments. No env caching. */
typedef struct {
    uint64_t request_id;
    uint32_t tid, requested_mask, actual_mask, task_count, int_status;
    uint64_t host_begin_ns, ioctl_begin_ns, ioctl_end_ns, host_done_ns;
    int64_t driver_service_ns;
    int result;
} co_schedule_profile;

int co_schedule_mask_valid(unsigned mask);
/* Userspace round-robin selection inside a nonempty allowed set. The returned
 * single bit is the actual kernel request, not an alleged kernel candidate mask. */
int co_schedule_pick(unsigned allowed_mask,unsigned ordinal,unsigned *single_bit);
int co_schedule_execute(int fd,const co_program *programs,unsigned count,
                        unsigned mask,int trace_marker_fd,uint64_t request_id,
                        co_schedule_profile *profile);
#endif
