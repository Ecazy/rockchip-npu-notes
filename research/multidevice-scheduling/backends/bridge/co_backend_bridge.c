/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * co_backend_bridge.c -- a thin, stable C ABI over the existing RK3588 custom-ops
 * implementation, for the Python backend adapter to load with ctypes.
 *
 * It adds NO new device access and NO new execution path.  Every operation forwards
 * to the already-validated custom-ops entry points:
 *
 *   open/close/name     rocket_open / rocket_close / rocket_hw_current
 *   buffers             co_buffer_alloc / co_buffer_free
 *   transfer            the same rocket_bo_prep -> memcpy -> rocket_bo_fini cache
 *                       bracket that co_upload_f16 / co_download use
 *   execute             co_execute (single and batched programs)
 *   schedule            co_schedule_execute
 *
 * The bridge owns a live registry of opaque non-reused integer IDs. Invalid and
 * freed IDs are rejected without dereferencing caller-provided addresses.  No global
 * state beyond that; the device fd is supplied by the caller for every call.
 *
 * THIS FILE DOES NOT MODIFY custom-ops.  It only includes its public headers and links
 * its targets (see CMakeLists.txt).
 */

#include "custom_ops.h"
#include "co_schedule.h"
#include "rocket_npu.h"
#include "rocket_hw_profile.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define COBS_ABI_VERSION 2u
#define COBS_BUF_MAGIC 0x434f425342554630ULL /* "COBSBUF0" */
#define COBS_JOB_MAGIC 0x434f424a4f423030ULL /* "COBJOB00" */

typedef struct cobs_buffer {
    uint64_t magic, id;
    struct cobs_buffer *next;
    unsigned refs;
    co_buffer buf;
    int fd;
    int has_tensor;
    co_tensor tensor;
} cobs_buffer_t;

typedef struct cobs_job {
    uint64_t magic, id;
    struct cobs_job *next;
    cobs_buffer_t *buffers[4];
    unsigned nbuf;
    co_program prog;
} cobs_job_t;

/* Stable non-reused opaque IDs: no dereference of user-supplied addresses.
 * One adapter serializes this bridge; no concurrent calls on one process registry. */
static cobs_buffer_t *buffers;
static cobs_job_t *jobs;
static uint64_t next_id = 1;
static cobs_buffer_t *as_buffer(uint64_t handle) {
    for (cobs_buffer_t *b = buffers; b; b = b->next)
        if (b->id == handle) return b;
    return NULL;
}
static cobs_job_t *as_job(uint64_t handle) {
    for (cobs_job_t *j = jobs; j; j = j->next)
        if (j->id == handle) return j;
    return NULL;
}

uint32_t cobs_abi_version(void) { return COBS_ABI_VERSION; }

const char *cobs_device_name(void) {
    const struct rocket_hw_profile *p = rocket_hw_current();
    return p ? p->name : NULL;
}

/* -- lifecycle ------------------------------------------------------------ */

int cobs_open(void) { return rocket_open(); }

void cobs_close(int fd) {
    if (fd >= 0) rocket_close(fd);
}

/* -- buffers -------------------------------------------------------------- */

int cobs_buffer_alloc(int fd, size_t payload, uint64_t *out_handle) {
    if (fd < 0 || !out_handle || !payload) return ROCKET_E_SHAPE;
    cobs_buffer_t *b = (cobs_buffer_t *)calloc(1, sizeof(*b));
    if (!b) return ROCKET_E_NOMEM;
    int rc = co_buffer_alloc(fd, payload, &b->buf);
    if (rc) {
        free(b);
        return rc;
    }
    b->magic = COBS_BUF_MAGIC;
    b->fd = fd;
    b->has_tensor = 0;
    b->id = next_id++;
    b->next = buffers;
    buffers = b;
    *out_handle = b->id;
    return 0;
}

int cobs_buffer_free(uint64_t handle) {
    cobs_buffer_t *b = as_buffer(handle);
    if (!b || b->refs) return ROCKET_E_SHAPE;
    cobs_buffer_t **slot = &buffers;
    while (*slot != b) slot = &(*slot)->next;
    *slot = b->next;
    co_buffer_free(&b->buf);
    free(b);
    return 0;
}

size_t cobs_buffer_payload(uint64_t handle) {
    cobs_buffer_t *b = as_buffer(handle);
    return b ? b->buf.payload : 0;
}

size_t cobs_buffer_allocated(uint64_t handle) {
    cobs_buffer_t *b = as_buffer(handle);
    return b ? b->buf.bo.size : 0;
}

size_t cobs_tensor_bytes(uint64_t handle) {
    cobs_buffer_t *b = as_buffer(handle);
    return b && b->has_tensor ? b->tensor.bytes : 0;
}

int cobs_buffer_set_matrix(uint64_t handle, unsigned rows, unsigned cols, int type) {
    cobs_buffer_t *b = as_buffer(handle);
    if (!b || b->refs) return ROCKET_E_SHAPE;
    if (type != CO_F16 && type != CO_F32) return ROCKET_E_SHAPE;
    int rc = co_tensor_matrix(&b->tensor, rows, cols, (enum co_type)type);
    if (!rc) b->has_tensor = 1;
    return rc;
}

int cobs_buffer_set_input(uint64_t handle, unsigned rows, unsigned cols) {
    cobs_buffer_t *b = as_buffer(handle);
    if (!b || b->refs) return ROCKET_E_SHAPE;
    int rc = co_tensor_input(&b->tensor, rows, cols);
    if (!rc) b->has_tensor = 1;
    return rc;
}

int cobs_buffer_set_weight(uint64_t handle, unsigned outputs, unsigned inputs) {
    cobs_buffer_t *b = as_buffer(handle);
    if (!b || b->refs) return ROCKET_E_SHAPE;
    int rc = co_tensor_weight(&b->tensor, outputs, inputs);
    if (!rc) b->has_tensor = 1;
    return rc;
}

int cobs_buffer_set_linear(uint64_t handle, unsigned elements) {
    cobs_buffer_t *b = as_buffer(handle);
    if (!b || b->refs) return ROCKET_E_SHAPE;
    int rc = co_tensor_linear(&b->tensor, elements);
    if (!rc) b->has_tensor = 1;
    return rc;
}

/* -- transfer (same cache bracket as co_upload_f16 / co_download) ---------- */

int cobs_upload(uint64_t handle, size_t offset, const void *src, size_t size) {
    cobs_buffer_t *b = as_buffer(handle);
    if (!b || !src || !size) return ROCKET_E_SHAPE;
    if (offset > b->buf.payload || size > b->buf.payload - offset) return ROCKET_E_SHAPE;
    int rc = rocket_bo_prep(b->fd, &b->buf.bo, 1, 2000000000ull);
    if (rc) return rc;
    memcpy((char *)b->buf.bo.ptr + offset, src, size);
    return rocket_bo_fini(b->fd, &b->buf.bo);
}

int cobs_download(uint64_t handle, size_t offset, void *dst, size_t size) {
    cobs_buffer_t *b = as_buffer(handle);
    if (!b || !dst || !size) return ROCKET_E_SHAPE;
    if (offset > b->buf.payload || size > b->buf.payload - offset) return ROCKET_E_SHAPE;
    int rc = rocket_bo_prep(b->fd, &b->buf.bo, 0, 2000000000ull);
    if (rc) return rc;
    memcpy(dst, (char *)b->buf.bo.ptr + offset, size);
    return rocket_bo_fini(b->fd, &b->buf.bo);
}

/* -- jobs ----------------------------------------------------------------- */

int cobs_job_create(uint64_t out_handle, size_t out_offset, uint64_t *out_job) {
    cobs_buffer_t *ob = as_buffer(out_handle);
    if (!ob || !out_job || !ob->has_tensor) return ROCKET_E_SHAPE;
    cobs_job_t *j = (cobs_job_t *)calloc(1, sizeof(*j));
    if (!j) return ROCKET_E_NOMEM;
    co_view out;
    int rc = co_view_init(&out, &ob->buf, out_offset, &ob->tensor);
    if (rc) {
        free(j);
        return rc;
    }
    j->prog.output = out;
    j->prog.nreads = 0;
    j->prog.count = 0;
    j->prog.flags = 0;
    j->magic = COBS_JOB_MAGIC;
    j->id = next_id++;
    j->buffers[j->nbuf++] = ob;
    ob->refs++;
    j->next = jobs;
    jobs = j;
    *out_job = j->id;
    return 0;
}

int cobs_job_add_read(uint64_t job_handle, uint64_t buffer_handle, size_t offset) {
    cobs_job_t *j = as_job(job_handle);
    cobs_buffer_t *b = as_buffer(buffer_handle);
    if (!j || !b || !b->has_tensor || b->fd != j->buffers[0]->fd) return ROCKET_E_SHAPE;
    if (j->prog.nreads >= 3) return ROCKET_E_SHAPE;
    co_view v;
    int rc = co_view_init(&v, &b->buf, offset, &b->tensor);
    if (rc) return rc;
    j->prog.reads[j->prog.nreads++] = v;
    j->buffers[j->nbuf++] = b;
    b->refs++;
    return 0;
}

int cobs_job_set_words(uint64_t job_handle, const uint64_t *words, unsigned count,
                       unsigned flags) {
    cobs_job_t *j = as_job(job_handle);
    if (!j || !words || !count || count > CO_MAX_WORDS) return ROCKET_E_SHAPE;
    memcpy(j->prog.words, words, (size_t)count * sizeof(uint64_t));
    j->prog.count = count;
    j->prog.flags = flags;
    return 0;
}

int cobs_job_validate(uint64_t job_handle) {
    cobs_job_t *j = as_job(job_handle);
    if (!j) return ROCKET_E_SHAPE;
    if (!j->prog.count || j->prog.count > CO_MAX_WORDS) return ROCKET_E_SHAPE;
    if (j->prog.nreads > 3) return ROCKET_E_SHAPE;
    if (!j->prog.output.buffer) return ROCKET_E_SHAPE;
    return 0;
}

void cobs_job_free(uint64_t job_handle) {
    cobs_job_t *j = as_job(job_handle);
    if (!j) return;
    cobs_job_t **slot = &jobs;
    while (*slot != j) slot = &(*slot)->next;
    *slot = j->next;
    for (unsigned i = 0; i < j->nbuf; i++) j->buffers[i]->refs--;
    free(j);
}

/* -- execution ------------------------------------------------------------ */

int cobs_execute(int fd, uint64_t job_handle, unsigned mode) {
    cobs_job_t *j = as_job(job_handle);
    if (!j || fd < 0 || fd != j->buffers[0]->fd || mode > CO_CHAIN) return ROCKET_E_SHAPE;
    if (cobs_job_validate(job_handle)) return ROCKET_E_SHAPE;
    return co_execute(fd, &j->prog, 1, (enum co_mode)mode);
}

int cobs_execute_batch(int fd, const uint64_t *job_handles, unsigned n, unsigned mode) {
    if (!job_handles || !n || n > CO_MAX_PROGRAMS || fd < 0 || mode > CO_CHAIN) return ROCKET_E_SHAPE;
    co_program *progs = (co_program *)calloc(n, sizeof(*progs));
    if (!progs) return ROCKET_E_NOMEM;
    int rc = 0;
    for (unsigned i = 0; i < n; i++) {
        cobs_job_t *j = as_job(job_handles[i]);
        if (!j || j->buffers[0]->fd != fd || cobs_job_validate(job_handles[i])) {
            rc = ROCKET_E_SHAPE;
            break;
        }
        progs[i] = j->prog;
    }
    if (!rc) rc = co_execute(fd, progs, n, (enum co_mode)mode);
    free(progs);
    return rc;
}

int cobs_schedule_execute(int fd, uint64_t job_handle, unsigned mask, uint64_t request_id,
                          uint64_t *host_begin_ns, uint64_t *ioctl_begin_ns,
                          uint64_t *ioctl_end_ns, uint64_t *host_done_ns,
                          int64_t *driver_service_ns, unsigned *actual_mask,
                          unsigned *task_count, int *int_status, int *result) {
    cobs_job_t *j = as_job(job_handle);
    if (!j || fd < 0 || fd != j->buffers[0]->fd || !co_schedule_mask_valid(mask)) return ROCKET_E_SHAPE;
    if (cobs_job_validate(job_handle)) return ROCKET_E_SHAPE;
    co_schedule_profile profile = {0};
    int rc = co_schedule_execute(fd, &j->prog, 1, mask, -1, request_id, &profile);
    if (host_begin_ns) *host_begin_ns = profile.host_begin_ns;
    if (ioctl_begin_ns) *ioctl_begin_ns = profile.ioctl_begin_ns;
    if (ioctl_end_ns) *ioctl_end_ns = profile.ioctl_end_ns;
    if (host_done_ns) *host_done_ns = profile.host_done_ns;
    if (driver_service_ns) *driver_service_ns = profile.driver_service_ns;
    if (actual_mask) *actual_mask = profile.actual_mask;
    if (task_count) *task_count = profile.task_count;
    if (int_status) *int_status = profile.int_status;
    if (result) *result = profile.result;
    return rc;
}

/* -- status --------------------------------------------------------------- */

const char *cobs_status_name(int status) {
    switch (status) {
        case ROCKET_OK: return "OK";
        case ROCKET_E_SHAPE: return "E_SHAPE";
        case ROCKET_E_TILING: return "E_TILING";
        case ROCKET_E_NOMEM: return "E_NOMEM";
        case ROCKET_E_DEVICE: return "E_DEVICE";
        case ROCKET_E_UNSUPPORTED: return "E_UNSUPPORTED";
        default: return "E_UNKNOWN";
    }
}

/* Hardware smoke/portable emitter: consumes the EXISTING RK3588 LUT generator. */
int cobs_job_emit_lut(uint64_t job_handle, uint64_t input_handle, size_t offset, unsigned act) {
    cobs_job_t *j = as_job(job_handle);
    cobs_buffer_t *b = as_buffer(input_handle);
    if (!j || !b || !b->has_tensor || j->nbuf != 1 || b->fd != j->buffers[0]->fd || act > CO_RSQRT_BOUNDED)
        return ROCKET_E_SHAPE;
    co_view input;
    int rc = co_view_init(&input, &b->buf, offset, &b->tensor);
    if (rc) return rc;
    co_view output = j->prog.output;
    rc = co_emit_lut(&j->prog, input, output, (enum co_act)act);
    if (rc) return rc;
    j->buffers[j->nbuf++] = b;
    b->refs++;
    return 0;
}
int cobs_guard_check(uint64_t handle) {
    cobs_buffer_t *b = as_buffer(handle);
    return b ? co_guard_check(&b->buf) : ROCKET_E_SHAPE;
}
