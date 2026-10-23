/* Minimal, bounded interposer: timestamp every RKNPU submit ioctl only.
 *
 * Two clock_gettime calls per ioctl and one fprintf to stderr; no MMIO reads, no
 * regcmd capture.  Used to place submits on a common wall-clock axis so that
 * intra-submit gaps (sampled by the blockwatch hook) can be told apart from
 * inter-submit scheduling gaps.
 *
 * stderr lines:  SUBMIT <enter_ns> <leave_ns> <ret> <task_number>
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <time.h>

struct rknpu_subcore_task {
    uint32_t task_start;
    uint32_t task_number;
};

struct rknpu_submit {
    uint32_t flags;
    uint32_t timeout;
    uint32_t task_start;
    uint32_t task_number;
    uint32_t task_counter;
    int32_t priority;
    uint64_t task_obj_addr;
    uint32_t iommu_domain_id;
    uint32_t reserved;
    uint64_t task_base_addr;
    int64_t hw_elapse_time;
    uint32_t core_mask;
    int32_t fence_fd;
    struct rknpu_subcore_task subcore_task[5];
};

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int ioctl(int fd, unsigned long request, ...) {
    static int (*real_ioctl)(int, unsigned long, void*) = NULL;
    va_list ap;
    void* arg;
    if (real_ioctl == NULL) real_ioctl = (int (*)(int, unsigned long, void*))dlsym(RTLD_NEXT, "ioctl");
    va_start(ap, request);
    arg = va_arg(ap, void*);
    va_end(ap);
    const int is_submit = _IOC_SIZE(request) == sizeof(struct rknpu_submit);
    if (!is_submit) return real_ioctl(fd, request, arg);
    const uint64_t t0 = now_ns();
    const int ret = real_ioctl(fd, request, arg);
    const uint64_t t1 = now_ns();
    struct rknpu_submit* s = (struct rknpu_submit*)arg;
    fprintf(stderr, "SUBMIT %llu %llu %d %u\n", (unsigned long long)t0, (unsigned long long)t1, ret,
            s ? s->task_number : 0u);
    return ret;
}
