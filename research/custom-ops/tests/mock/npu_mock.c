/* Test-only NPU seam shared by the C host tests and librknpu_plan_mock.so.
 *
 * It implements the host side of the vendor calls the planner emits (BO
 * allocation, submit, and the primitive semantics) so a plan can be executed
 * without any device.  The production library never links this file.
 */
#include <math.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "custom_ops.h"
#include "rocket_npu.h"

#include "mock_npu.h"   /* the single source of the test-only seam API */

enum { MOCK_TAG_LUT = 1, MOCK_TAG_EW = 2, MOCK_TAG_MM = 3,
       MOCK_TAG_MM_ACC = 4, MOCK_TAG_EW_MAX = 5 };
static struct {
    const co_program *prog;   /* emission-time address (the slot key) */
    co_program *snapshot;    /* full copy of what that emission produced */
    unsigned tag;
    unsigned arg;
} mock_tags[4096];
static unsigned mock_tag_count;
static unsigned mock_tag_overflow;
static unsigned mock_untracked;

static unsigned mock_allocs;     /* BO allocations so far */
static unsigned mock_fail_alloc; /* 1-based allocation index that must fail */
static unsigned mock_handle_seq = 0x40000u;
/* Independent, 16-byte aligned, non-overlapping 32-bit DMA windows per BO. */
#define MOCK_DMA_BASE 0x01000000u
#define MOCK_DMA_LIMIT 0xF0000000u
struct mock_bo_entry { uint32_t handle; uint32_t dma; size_t size; void *host; };
static struct mock_bo_entry mock_bos[512];
static unsigned mock_bo_count;
static unsigned mock_dma_next;   /* next free byte offset inside the window */ /* unique, 32-bit, never pointer-derived */
static unsigned mock_execs;      /* how many times the device path was entered */
static unsigned mock_fail_at;    /* 1-based execution index that must fail */
static unsigned mock_inject_nan;
static unsigned mock_corrupt_guard;
static unsigned mock_corrupt_guard_at;
static unsigned mock_delay_us;
static unsigned mock_prep_count;
static unsigned mock_fail_prep_at;
#define MOCK_CONTEXT_FDS 4096
static atomic_int mock_context_faulted[MOCK_CONTEXT_FDS];
static int mock_live_bos;
static size_t mock_resident;

unsigned rk_mock_execs(void) { return mock_execs; }
unsigned rk_mock_allocs(void) { return mock_allocs; }
void rk_mock_fail_alloc(unsigned index) { mock_fail_alloc = index; }
void rk_mock_fail_at(unsigned index) { mock_fail_at = index; }
void rk_mock_inject_nan(unsigned on) { mock_inject_nan = on; }
void rk_mock_corrupt_guard(unsigned mode) { mock_corrupt_guard = mode; }
void rk_mock_corrupt_guard_at(unsigned exec_index) {
    mock_corrupt_guard_at = exec_index;
}
void rk_mock_delay_us(unsigned us) { mock_delay_us = us; }
void rk_mock_fail_prep_at(unsigned index) { mock_fail_prep_at = index; }
unsigned rk_mock_prep_calls(void) { return mock_prep_count; }
void rk_mock_reset_tags(void) {
    for (unsigned i = 0; i < mock_tag_count; i++) {
        free(mock_tags[i].snapshot);
        mock_tags[i].snapshot = NULL;
    }
    mock_tag_count = 0;
    mock_tag_overflow = 0;
}
void rk_mock_context_fault(unsigned fd, unsigned on) {
    if (fd < MOCK_CONTEXT_FDS)
        atomic_store_explicit(&mock_context_faulted[fd], on ? 1 : 0,
                              memory_order_release);
}
int rk_mock_context_faulted(int fd) {
    if (fd < 0 || fd >= MOCK_CONTEXT_FDS) return 0;
    return atomic_load_explicit(&mock_context_faulted[fd],
                                memory_order_acquire) ? 1 : 0;
}
int __wrap_rocket_context_faulted(int fd) {
    return rk_mock_context_faulted(fd);
}
int rk_mock_live_bos(void) { return mock_live_bos; }
size_t rk_mock_resident(void) { return mock_resident; }
void rk_mock_reset(void) {
    mock_execs = mock_fail_at = mock_inject_nan = 0;
    mock_corrupt_guard = 0;
    mock_corrupt_guard_at = 0;
    mock_delay_us = 0;
    mock_prep_count = 0;
    mock_fail_prep_at = 0;
    for (unsigned i = 0; i < MOCK_CONTEXT_FDS; i++)
        atomic_store_explicit(&mock_context_faulted[i], 0,
                              memory_order_relaxed);
    for (unsigned i = 0; i < mock_tag_count; i++) {
        free(mock_tags[i].snapshot);
        mock_tags[i].snapshot = NULL;
    }
    mock_tag_count = 0;
    mock_tag_overflow = 0;
    mock_untracked = 0;
}
unsigned rk_mock_tag_count(void) { return mock_tag_count; }
unsigned rk_mock_bo_count(void) { return mock_bo_count; }
/* DMA window of a live simulated BO, or 0 when the handle is unknown. */
int rk_mock_dma_range(unsigned index, uint32_t *handle, uint32_t *dma,
                      size_t *size) {
    if (index >= mock_bo_count) return -1;
    if (handle) *handle = mock_bos[index].handle;
    if (dma) *dma = mock_bos[index].dma;
    if (size) *size = mock_bos[index].size;
    return 0;
}
/* Diagnostic: where did a given FP16 bit pattern land inside the live BOs? */
int rk_mock_find_half(uint16_t pattern, unsigned *bo_index, size_t *offset) {
    for (unsigned i = 0; i < mock_bo_count; i++) {
        const uint16_t *half = (const uint16_t *)mock_bos[i].host;
        if (!half) continue;
        for (size_t j = 0; j < mock_bos[i].size / 2u; j++)
            if (half[j] == pattern) {
                if (bo_index) *bo_index = i;
                if (offset) *offset = j;
                return 0;
            }
    }
    return -1;
}
uint64_t rk_mock_bo_dma_address(uint32_t handle) {
    /* Returns the dma_address the allocator actually wrote into the struct. */
    for (unsigned i = 0; i < mock_bo_count; i++)
        if (mock_bos[i].handle == handle) return (uint64_t)mock_bos[i].dma;
    return 0;
}
uint32_t rk_mock_dma_of(uint32_t handle) {
    for (unsigned i = 0; i < mock_bo_count; i++)
        if (mock_bos[i].handle == handle) return mock_bos[i].dma;
    return 0;
}
unsigned rk_mock_tag_overflow(void) { return mock_tag_overflow; }
/* Test hook: simulate a full tag table so the executor's guard can be exercised
 * without emitting 256 throw-away programs. */
void rk_mock_force_tag_overflow(unsigned on) { mock_tag_overflow = on ? 1u : 0u; }
unsigned rk_mock_tag_capacity(void) {
    return (unsigned)(sizeof(mock_tags) / sizeof(mock_tags[0]));
}
unsigned rk_mock_untracked(void) { return mock_untracked; }

int __wrap_rocket_bo_alloc32(int fd, size_t size, rocket_bo *bo) {
    (void)fd;
    if (!bo) return ROCKET_E_SHAPE;
    memset(bo, 0, sizeof(*bo));       /* never leave fields uninitialised */
    if (++mock_allocs == mock_fail_alloc) return ROCKET_E_NOMEM;
    if (mock_bo_count >= sizeof(mock_bos) / sizeof(mock_bos[0]))
        return ROCKET_E_NOMEM;        /* table exhausted: no host memory taken */
    if (size > SIZE_MAX - 15u) return ROCKET_E_NOMEM;  /* alignment overflow */
    size_t aligned = (size + 15u) & ~(size_t)15u;
    if (!aligned) aligned = 16u;
    if (mock_dma_next > MOCK_DMA_LIMIT - MOCK_DMA_BASE ||
        aligned > MOCK_DMA_LIMIT - MOCK_DMA_BASE - mock_dma_next)
        return ROCKET_E_NOMEM;        /* 32-bit DMA space exhausted */
    if (mock_handle_seq == UINT32_MAX)
        return ROCKET_E_NOMEM;        /* handle space exhausted */
    void *ptr = calloc(1, aligned);
    if (!ptr) return ROCKET_E_NOMEM;
    mock_bos[mock_bo_count].handle = mock_handle_seq;
    mock_bos[mock_bo_count].dma = MOCK_DMA_BASE + mock_dma_next;
    mock_bos[mock_bo_count].size = aligned;
    mock_bos[mock_bo_count].host = ptr;
    mock_bo_count++;
    mock_dma_next += aligned;
    bo->dma_address = (uint64_t)mock_bos[mock_bo_count - 1].dma;
    bo->handle = mock_handle_seq++;   /* unique simulated id, not a truncation */
    bo->size = size;
    bo->ptr = ptr;
    mock_live_bos++;
    mock_resident += size;
    return 0;
}
void __wrap_rocket_bo_free(int fd, rocket_bo *bo) {
    (void)fd;
    if (!bo || !bo->ptr) return;
    mock_live_bos--;
    mock_resident -= bo->size;
    for (unsigned i = 0; i < mock_bo_count; i++)
        if (mock_bos[i].handle == bo->handle) {
            mock_bos[i] = mock_bos[mock_bo_count - 1];
            mock_bo_count--;
            break;
        }
    free(bo->ptr);
    memset(bo, 0, sizeof(*bo));
}
int __wrap_rocket_bo_prep(int fd, rocket_bo *bo, int dir, uint64_t ns) {
    (void)fd; (void)bo; (void)dir; (void)ns;
    mock_prep_count++;
    if (mock_fail_prep_at && mock_prep_count == mock_fail_prep_at)
        return ROCKET_E_DEVICE;
    return 0;
}
int __wrap_rocket_bo_fini(int fd, rocket_bo *bo) { (void)fd; (void)bo; return 0; }
int __wrap_rocket_batched_submit_supported(void) { return 1; }

/* Private side table: the mock remembers which template produced each emitted
 * program.  ``co_program.flags`` carries production meaning (``co_schedule.c``
 * requires zero, ``custom_ops.c`` uses ROCKET_JOB_PPU_DONE), so the seam must
 * never write a tag into it. */
static void mock_tag(const co_program *p, unsigned tag, unsigned arg) {
    for (unsigned i = 0; i < mock_tag_count; i++)
        if (mock_tags[i].prog == p) {
            /* Re-emission into a slot we already know: the slot is refreshed with
             * THIS emission.  The address is the slot key; the new content is
             * allowed to be a different template (EW -> MM) and the entry count
             * does not grow. */
            memcpy(mock_tags[i].snapshot, p, sizeof(*p));
            mock_tags[i].tag = tag;
            mock_tags[i].arg = arg;
            return;
        }
    if (mock_tag_count >= sizeof(mock_tags) / sizeof(mock_tags[0])) {
        mock_tag_overflow = 1;   /* the executor turns this into a hard error */
        return;
    }
    co_program *snap = malloc(sizeof(*snap));
    if (!snap) { mock_tag_overflow = 1; return; }
    memcpy(snap, p, sizeof(*snap));
    mock_tags[mock_tag_count].prog = p;
    mock_tags[mock_tag_count].snapshot = snap;
    mock_tags[mock_tag_count].tag = tag;
    mock_tags[mock_tag_count].arg = arg;
    mock_tag_count++;
}

static int mock_lookup(const co_program *p, unsigned *tag, unsigned *arg) {
    for (unsigned i = 0; i < mock_tag_count; i++)
        if (mock_tags[i].prog == p) {
            /* An address hit only identifies the slot.  The program must still be
             * byte-identical to what the emitter produced for it, otherwise a
             * modified or stale buffer would be executed as the old template. */
            if (memcmp(mock_tags[i].snapshot, p, sizeof(*p)) != 0) return -1;
            *tag = mock_tags[i].tag;
            *arg = mock_tags[i].arg;
            return 0;
        }
    return -1;
}


/* Emitter wrappers: every successful emission (re)registers its slot. */
int __real_co_emit_mm(co_program *, co_view, co_view, co_view);
int __wrap_co_emit_mm(co_program *p, co_view x, co_view w, co_view y) {
    int rc = __real_co_emit_mm(p, x, w, y);
    if (!rc) mock_tag(p, MOCK_TAG_MM, 0);
    return rc;
}
int __real_co_emit_mm_acc(co_program *, co_view, co_view, co_view, co_view);
int __wrap_co_emit_mm_acc(co_program *p, co_view x, co_view w, co_view prior,
                          co_view y) {
    int rc = __real_co_emit_mm_acc(p, x, w, prior, y);
    if (!rc) mock_tag(p, MOCK_TAG_MM_ACC, 0);
    return rc;
}
int __real_co_emit_lut(co_program *, co_view, co_view, enum co_act);
int __wrap_co_emit_lut(co_program *p, co_view x, co_view y, enum co_act kind) {
    int rc = __real_co_emit_lut(p, x, y, kind);
    if (!rc) mock_tag(p, MOCK_TAG_LUT, (unsigned)kind);
    return rc;
}
int __real_co_emit_ew(co_program *, co_view, co_view, co_view, co_view, int);
int __wrap_co_emit_ew(co_program *p, co_view x, co_view a, co_view w, co_view y,
                      int mul) {
    int rc = __real_co_emit_ew(p, x, a, w, y, mul);
    if (!rc) mock_tag(p, MOCK_TAG_EW, (unsigned)mul);
    return rc;
}
int __real_co_emit_ew_max(co_program *, co_view, co_view, co_view, co_view);
int __wrap_co_emit_ew_max(co_program *p, co_view x, co_view a, co_view w,
                          co_view y) {
    int rc = __real_co_emit_ew_max(p, x, a, w, y);
    if (!rc) mock_tag(p, MOCK_TAG_EW_MAX, 0);
    return rc;
}

static _Float16 *mock_data(co_view v) {
    return (_Float16 *)((char *)v.buffer->bo.ptr + v.offset);
}

static void mock_maybe_corrupt_guard(const co_program *p) {
    if (!mock_corrupt_guard || !p || !p->output.buffer ||
        !p->output.buffer->bo.ptr)
        return;
    if (mock_corrupt_guard_at && mock_execs != mock_corrupt_guard_at)
        return;
    unsigned char *payload = (unsigned char *)p->output.buffer->bo.ptr;
    if (mock_corrupt_guard & 1u)
        payload[p->output.buffer->payload + CO_SLACK] ^= 0xffu;
    if (mock_corrupt_guard & 2u)
        payload[p->output.buffer->payload] ^= 0xffu;
}

/* Primitive semantics: matmul (with optional accumulate / element-wise mul),
 * LUT activation, and the element-wise add/mul that goes through the identity
 * weight.  No hardware LUT interpolation error is modelled. */
int __wrap_co_execute_profiled(int fd, const co_program *p, unsigned count,
                        enum co_mode mode, co_execute_workspace *workspace,
                        co_execute_profile *profile) {
    (void)fd; (void)workspace;
    /* The mock performs the work the provider would perform, so it reports that
     * work as the simulated blocking submit.  Without this the envelope
     * attribution would show an unexplained residual that only exists because
     * the provider seam is substituted, which would hide real logic errors. */
    struct timespec mock_start, mock_end;
    clock_gettime(CLOCK_MONOTONIC, &mock_start);
    if (mock_delay_us) usleep(mock_delay_us);
    if (!p || !count || count > 64u ||
        (mode != CO_CHAIN && mode != CO_SEPARATE))
        return ROCKET_E_SHAPE;   /* validate the call before any view access */
    if (profile) memset(profile, 0, sizeof(*profile));
    if (++mock_execs == mock_fail_at) {
        /* A submit-path failure quarantines the shared provider context before
         * returning.  The real provider does the same while holding submit_lock. */
        rk_mock_context_fault(fd, 1);
        return ROCKET_E_DEVICE;
    }
    if (mock_tag_overflow) return ROCKET_E_SHAPE; /* table exhausted: refuse */
    for (unsigned i = 0; i < count; i++) {
        /* tags first: an unknown program is refused without dereferencing it */
        unsigned tag = 0, arg = 0;
        if (mock_lookup(&p[i], &tag, &arg) != 0) {
            mock_untracked++;
            return ROCKET_E_SHAPE;
        }
        /* then the views the template actually needs */
        if (!p[i].output.buffer || !p[i].output.buffer->bo.ptr)
            return ROCKET_E_SHAPE;
        if (!p[i].reads[0].buffer || !p[i].reads[0].buffer->bo.ptr)
            return ROCKET_E_SHAPE;
        co_view y = p[i].output;
        co_view x = p[i].reads[0];
        _Float16 *dst = mock_data(y);
        if (tag == MOCK_TAG_LUT) {
            if (p[i].nreads != 1) return ROCKET_E_SHAPE;
            enum co_act kind = (enum co_act)arg;
            _Float16 *src = mock_data(x);
            if (x.tensor.rows != y.tensor.rows || x.tensor.cols != y.tensor.cols ||
                x.tensor.layout != y.tensor.layout)
                return ROCKET_E_SHAPE;
            for (unsigned r = 0; r < y.tensor.physical_rows; r++)
                for (unsigned c = 0; c < y.tensor.physical_cols; c++) {
                    double v = (double)src[co_index(&x.tensor, r, c)], out;
                    switch (kind) {
                    case CO_SIGMOID: out = 1.0 / (1.0 + exp(-v)); break;
                    case CO_HARDSIGMOID:
                        out = fmin(1.0, fmax(0.0, v / 6.0 + 0.5)); break;
                    case CO_GELU_GATE:
                        out = 0.5 * (1.0 + erf(v / sqrt(2.0))); break;
                    case CO_EXP_BOUNDED: out = exp(v) / 64.0; break;
                    case CO_EXP_NEG4: out = exp(fmin(0.0, v)); break;
                    case CO_RSQRT_BOUNDED:
                        out = 1.0 / (4.0 * sqrt(fmax(0.125, v))); break;
                    default: return ROCKET_E_SHAPE; /* not modelled */
                    }
                    dst[co_index(&y.tensor, r, c)] = (_Float16)out;
                }
            if (mock_inject_nan) dst[0] = (_Float16)NAN;
            mock_maybe_corrupt_guard(&p[i]);
            continue;
        }
        if (tag == MOCK_TAG_EW) {
            if (p[i].nreads != 3) return ROCKET_E_SHAPE;
            co_view id = p[i].reads[1];
            co_view op = p[i].reads[2];
            if (!id.buffer || !id.buffer->bo.ptr || !op.buffer || !op.buffer->bo.ptr)
                return ROCKET_E_SHAPE;
            if (id.tensor.rows != x.tensor.physical_cols ||
                id.tensor.cols != x.tensor.physical_cols)
                return ROCKET_E_SHAPE;
            if (op.tensor.rows != x.tensor.rows ||
                op.tensor.physical_rows != x.tensor.physical_rows ||
                op.tensor.physical_cols != x.tensor.physical_cols)
                return ROCKET_E_SHAPE;
            _Float16 *dst = mock_data(y);
            _Float16 *lhs = mock_data(x);
            _Float16 *rhs = mock_data(op);
            _Float16 *ident = mock_data(id);
            int mul = (int)(arg & 1u);
            for (unsigned r = 0; r < y.tensor.physical_rows; r++)
                for (unsigned c = 0; c < y.tensor.physical_cols; c++) {
                    double pass = 0;
                    for (unsigned q = 0; q < x.tensor.physical_cols; q++)
                        pass += (double)lhs[co_index(&x.tensor, r, q)] *
                                (double)ident[co_index(&id.tensor, c, q)];
                    double b = (double)rhs[co_index(&op.tensor, r, c)];
                    dst[co_index(&y.tensor, r, c)] =
                        (_Float16)(mul ? pass * b : pass + b);
                }
            if (mock_inject_nan) dst[0] = (_Float16)NAN;
            mock_maybe_corrupt_guard(&p[i]);
            continue;
        }
        if (tag == MOCK_TAG_EW_MAX) {
            if (p[i].nreads != 3) return ROCKET_E_SHAPE;
            co_view id = p[i].reads[1];
            co_view op = p[i].reads[2];
            if (!id.buffer || !id.buffer->bo.ptr || !op.buffer || !op.buffer->bo.ptr)
                return ROCKET_E_SHAPE;
            if (id.tensor.rows != x.tensor.physical_cols ||
                id.tensor.cols != x.tensor.physical_cols)
                return ROCKET_E_SHAPE;
            if (op.tensor.rows != x.tensor.rows ||
                op.tensor.physical_rows != x.tensor.physical_rows ||
                op.tensor.physical_cols != x.tensor.physical_cols)
                return ROCKET_E_SHAPE;
            _Float16 *lhs = mock_data(x);
            _Float16 *rhs = mock_data(op);
            _Float16 *ident = mock_data(id);
            for (unsigned r = 0; r < y.tensor.physical_rows; r++)
                for (unsigned c = 0; c < y.tensor.physical_cols; c++) {
                    double pass = 0;
                    for (unsigned q = 0; q < x.tensor.physical_cols; q++)
                        pass += (double)lhs[co_index(&x.tensor, r, q)] *
                                (double)ident[co_index(&id.tensor, c, q)];
                    double b = (double)rhs[co_index(&op.tensor, r, c)];
                    dst[co_index(&y.tensor, r, c)] =
                        (_Float16)fmax(pass, b);
                }
            if (mock_inject_nan) dst[0] = (_Float16)NAN;
            mock_maybe_corrupt_guard(&p[i]);
            continue;
        }
        if (tag != MOCK_TAG_MM && tag != MOCK_TAG_MM_ACC) return ROCKET_E_SHAPE;
        if (tag == MOCK_TAG_MM ? p[i].nreads != 2 : p[i].nreads != 3)
            return ROCKET_E_SHAPE;
        co_view w = p[i].reads[1];
        if (!w.buffer || !w.buffer->bo.ptr) return ROCKET_E_SHAPE;
        _Float16 *weight = mock_data(w);
        for (unsigned r = 0; r < y.tensor.physical_rows; r++)
            for (unsigned c = 0; c < y.tensor.physical_cols; c++) {
                double value = 0;
                for (unsigned q = 0; q < x.tensor.physical_cols; q++)
                    value += (double)mock_data(x)[co_index(&x.tensor, r, q)] *
                             (double)weight[co_index(&w.tensor, c, q)];
                if (tag == MOCK_TAG_MM_ACC) {
                    co_view op = p[i].reads[2];
                    double b = (double)mock_data(op)[co_index(&op.tensor, r, c)];
                    value = (tag == MOCK_TAG_EW && (arg & 1u)) ? value * b
                                                               : value + b;
                }
                dst[co_index(&y.tensor, r, c)] = (_Float16)value;
            }
        if (mock_inject_nan) dst[0] = (_Float16)NAN;
        mock_maybe_corrupt_guard(&p[i]);
    }
    if (profile) {
        clock_gettime(CLOCK_MONOTONIC, &mock_end);
        profile->submit_wait_us =
            (double)(mock_end.tv_sec - mock_start.tv_sec) * 1e6 +
            (double)(mock_end.tv_nsec - mock_start.tv_nsec) / 1e3;
    }
    return 0;
}
