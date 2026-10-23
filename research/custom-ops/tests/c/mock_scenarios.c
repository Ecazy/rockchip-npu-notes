/* Test-only mock scenarios driven through the PRODUCTION emitters.
 *
 * BOs are allocated with the real rocket_bo_alloc32 (wrapped to the mock seam by
 * the linker), views are built with the production co_tensor_* helpers, and the
 * programs come from co_emit_ew / co_emit_mm / co_emit_mm_acc / co_emit_lut.
 * Nothing here re-implements a layout or an emitter; the helpers below only
 * allocate/initialise and register cleanup.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "custom_ops.h"
#include "mock_npu.h"

static unsigned checks, failures;

#define EXPECT(e) do { checks++; if (!(e)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #e); failures++; } } while (0)

#define TEST_FD 3

static _Float16 *view_data(co_view v) {
    return (_Float16 *)((char *)v.buffer->bo.ptr + v.offset);
}

/* --- allocation helpers: production allocator + production view init -------- */
typedef struct { co_buffer buf; co_view view; int live; } scenario_tensor;

static int make_tensor(scenario_tensor *t, const co_tensor *tensor) {
    memset(t, 0, sizeof(*t));
    /* payload must cover the 32-byte placement offset plus the tensor itself */
    size_t payload = 32u + tensor->bytes;
    int rc = co_buffer_alloc(TEST_FD, payload, &t->buf);
    if (rc) return rc;                       /* nothing to clean up yet */
    t->live = 1;
    rc = co_view_init(&t->view, &t->buf, 32u, tensor);
    if (rc) { co_buffer_free(&t->buf); memset(t, 0, sizeof(*t)); return rc; }
    return 0;
}

static int make_matrix(scenario_tensor *t, unsigned rows, unsigned cols) {
    co_tensor tensor;
    int rc = co_tensor_matrix(&tensor, rows, cols, CO_F16);
    if (rc) return rc;
    return make_tensor(t, &tensor);
}

static int make_weight(scenario_tensor *t, unsigned rows, unsigned cols) {
    co_tensor tensor;
    int rc = co_tensor_weight(&tensor, rows, cols);
    if (rc) return rc;
    return make_tensor(t, &tensor);
}

static void release(scenario_tensor *t) {
    if (t->live) { co_buffer_free(&t->buf); t->live = 0; }
    memset(t, 0, sizeof(*t));
}

static void fill(_Float16 *dst, size_t count, unsigned seed) {
    for (size_t i = 0; i < count; i++)
        dst[i] = (_Float16)(((int)((i * 17u + seed) % 29u) - 14) / 16.0);
}

/* --- scenario: independent Mul and Add through co_emit_ew -------------------- */
static void add_mul_case(void) {
    enum { ROWS = 16, COLS = 32 };
    scenario_tensor x, op, id, y;
    memset(&x, 0, sizeof(x)); memset(&op, 0, sizeof(op));
    memset(&id, 0, sizeof(id)); memset(&y, 0, sizeof(y));
    int ok = make_matrix(&x, ROWS, COLS) == 0 && make_matrix(&op, ROWS, COLS) == 0 &&
             make_matrix(&y, ROWS, COLS) == 0 && make_weight(&id, COLS, COLS) == 0;
    EXPECT(ok);
    if (!ok) {                      /* clean up whatever succeeded, then leave */
        release(&x); release(&op); release(&y); release(&id);
        EXPECT(rk_mock_live_bos() == 0);
        return;
    }
    int baseline = rk_mock_live_bos();
    EXPECT(baseline >= 4);

    _Float16 *xd = view_data(x.view), *od = view_data(op.view);
    _Float16 *idd = view_data(id.view), *yd = view_data(y.view);
    fill(xd, (size_t)ROWS * COLS, 3);
    fill(od, (size_t)ROWS * COLS, 11);
    memset(idd, 0, id.view.tensor.bytes);
    for (unsigned i = 0; i < COLS; i++)
        idd[co_index(&id.view.tensor, i, i)] = (_Float16)1;

    for (int mul = 0; mul < 2; mul++) {
        co_program program;
        memset(&program, 0, sizeof(program));
        EXPECT(co_emit_ew(&program, x.view, op.view, id.view, y.view, mul) == 0);
        memset(yd, 0x5A, y.view.tensor.bytes);
        co_execute_profile profile;
        EXPECT(co_execute_profiled(TEST_FD, &program, 1, CO_CHAIN, NULL,
                                   &profile) == 0);
        unsigned bad = 0;
        for (unsigned r = 0; r < ROWS; r++)
            for (unsigned c = 0; c < COLS; c++) {
                double a = (double)xd[co_index(&x.view.tensor, r, c)];
                double b = (double)od[co_index(&op.view.tensor, r, c)];
                double want = mul ? a * b : a + b;
                if ((double)yd[co_index(&y.view.tensor, r, c)] != (_Float16)want)
                    bad++;
            }
        EXPECT(bad == 0);
        EXPECT(rk_mock_untracked() == 0);
    }
    release(&x); release(&op); release(&id); release(&y);
    EXPECT(rk_mock_live_bos() == 0);
}

/* --- scenario: truncate-owned shutdown of a real BO --------------------------- */
static void cleanup_case(void) {
    scenario_tensor only;
    memset(&only, 0, sizeof(only));
    int ok = make_matrix(&only, 16, 32) == 0;
    EXPECT(ok);
    if (!ok) { release(&only); EXPECT(rk_mock_live_bos() == 0); return; }
    EXPECT(rk_mock_live_bos() == 1);
    release(&only);
    EXPECT(rk_mock_live_bos() == 0);
    EXPECT(rk_mock_resident() == 0);
}


/* --- scenario: tag-table capacity, three phases (global overflow semantics) ---
 * Phase 1: registering exactly `capacity` distinct programs succeeds and an
 *          already-registered program still executes.
 * Phase 2: registering program #capacity+1 trips the overflow flag; afterwards
 *          both an old and the new program are refused, with output bytes and
 *          resource counters unchanged.
 * Phase 3: after rk_mock_reset() a freshly emitted program registers and runs;
 *          a program from before the reset is NOT treated as registered.
 * The overflow is detected inside the emitter wrapper (mock_tag), not by the
 * production emitter: co_emit_ew itself still returns 0. */
static void capacity_case(void) {
    enum { ROWS = 16, COLS = 32 };
    scenario_tensor x, op, id, y;
    memset(&x, 0, sizeof(x)); memset(&op, 0, sizeof(op));
    memset(&id, 0, sizeof(id)); memset(&y, 0, sizeof(y));
    int ok = make_matrix(&x, ROWS, COLS) == 0 && make_matrix(&op, ROWS, COLS) == 0 &&
             make_matrix(&y, ROWS, COLS) == 0 && make_weight(&id, COLS, COLS) == 0;
    EXPECT(ok);
    if (!ok) { release(&x); release(&op); release(&y); release(&id); return; }

    rk_mock_reset();
    unsigned capacity = rk_mock_tag_capacity();
    EXPECT(capacity >= 16u);
    co_program *progs = calloc(capacity + 1u, sizeof(*progs));
    EXPECT(progs != NULL);
    if (!progs) { release(&x); release(&op); release(&y); release(&id); return; }
    int baseline = rk_mock_live_bos();
    size_t res_baseline = rk_mock_resident();

    /* phase 1 */
    unsigned registered = 0;
    for (unsigned i = 0; i < capacity; i++) {
        if (co_emit_ew(&progs[i], x.view, op.view, id.view, y.view, 0) != 0) break;
        registered++;
    }
    EXPECT(registered == capacity);
    EXPECT(rk_mock_tag_count() == capacity);
    EXPECT(rk_mock_tag_overflow() == 0);
    memset(view_data(y.view), 0x5A, y.view.tensor.bytes);
    co_execute_profile profile;
    EXPECT(co_execute_profiled(TEST_FD, &progs[0], 1, CO_CHAIN, NULL, &profile) == 0);

    /* phase 2: the first program beyond capacity */
    EXPECT(co_emit_ew(&progs[capacity], x.view, op.view, id.view, y.view, 0) == 0);
    EXPECT(rk_mock_tag_overflow() == 1);
    memset(view_data(y.view), 0x33, y.view.tensor.bytes);
    EXPECT(co_execute_profiled(TEST_FD, &progs[0], 1, CO_CHAIN, NULL, &profile) ==
           ROCKET_E_SHAPE);
    EXPECT(co_execute_profiled(TEST_FD, &progs[capacity], 1, CO_CHAIN, NULL,
                               &profile) == ROCKET_E_SHAPE);
    {   /* nothing was written, nothing leaked */
        unsigned touched = 0;
        unsigned char *raw = (unsigned char *)view_data(y.view);
        for (size_t i = 0; i < y.view.tensor.bytes; i++)
            if (raw[i] != 0x33) touched++;
        EXPECT(touched == 0);
    }
    EXPECT(rk_mock_live_bos() == baseline);
    EXPECT(rk_mock_resident() == res_baseline);

    /* phase 3: reset clears the table; old programs are not registered anymore */
    rk_mock_reset();
    EXPECT(rk_mock_tag_count() == 0);
    EXPECT(rk_mock_tag_overflow() == 0);
    co_program fresh;
    memset(&fresh, 0, sizeof(fresh));
    EXPECT(co_emit_ew(&fresh, x.view, op.view, id.view, y.view, 1) == 0);
    EXPECT(rk_mock_tag_count() == 1);
    EXPECT(co_execute_profiled(TEST_FD, &fresh, 1, CO_CHAIN, NULL, &profile) == 0);
    unsigned before = rk_mock_untracked();
    EXPECT(co_execute_profiled(TEST_FD, &progs[0], 1, CO_CHAIN, NULL, &profile) ==
           ROCKET_E_SHAPE);
    EXPECT(rk_mock_untracked() == before + 1);

    free(progs);
    release(&x); release(&op); release(&y); release(&id);
    EXPECT(rk_mock_live_bos() == 0);
    EXPECT(rk_mock_resident() == 0);   /* final resource baseline */
    rk_mock_reset();
}


/* --- scenario: tag lifecycle (refresh / copy / snapshot integrity) ----------- */
static void tag_lifecycle_case(void) {
    enum { ROWS = 16, COLS = 32 };
    scenario_tensor x, w, op, id, y;
    memset(&x, 0, sizeof(x)); memset(&w, 0, sizeof(w));
    memset(&op, 0, sizeof(op)); memset(&id, 0, sizeof(id));
    memset(&y, 0, sizeof(y));
    int ok = make_matrix(&x, ROWS, COLS) == 0 && make_weight(&w, COLS, COLS) == 0 &&
             make_matrix(&op, ROWS, COLS) == 0 && make_matrix(&y, ROWS, COLS) == 0 &&
             make_weight(&id, COLS, COLS) == 0;
    EXPECT(ok);
    if (!ok) { release(&x); release(&w); release(&op); release(&id); release(&y); return; }

    _Float16 *xd = view_data(x.view), *wd = view_data(w.view);
    _Float16 *od = view_data(op.view), *idd = view_data(id.view);
    _Float16 *yd = view_data(y.view);
    fill(xd, (size_t)ROWS * COLS, 5);
    /* weight of a [COLS x COLS] matmul: W out x in, so W[c][q] */
    fill(wd, (size_t)COLS * COLS, 7);
    fill(od, (size_t)ROWS * COLS, 13);
    memset(idd, 0, id.view.tensor.bytes);
    for (unsigned i = 0; i < COLS; i++) idd[co_index(&id.view.tensor, i, i)] = (_Float16)1;

    rk_mock_reset();
    int baseline = rk_mock_live_bos();
    size_t res_baseline = rk_mock_resident();
    co_program slot;                       /* one object, re-emitted twice */
    memset(&slot, 0, sizeof(slot));
    co_execute_profile profile;

    /* (1) same address: EW then MM, entry count must not grow */
    EXPECT(co_emit_ew(&slot, x.view, op.view, id.view, y.view, 1) == 0);
    unsigned after_ew = rk_mock_tag_count();
    EXPECT(after_ew == 1);
    memset(yd, 0, y.view.tensor.bytes);
    EXPECT(co_execute_profiled(TEST_FD, &slot, 1, CO_CHAIN, NULL, &profile) == 0);
    {
        unsigned bad = 0;
        for (unsigned r = 0; r < ROWS; r++)
            for (unsigned c = 0; c < COLS; c++) {
                double want = (double)xd[co_index(&x.view.tensor, r, c)] *
                              (double)od[co_index(&op.view.tensor, r, c)];
                if ((double)yd[co_index(&y.view.tensor, r, c)] != (_Float16)want) bad++;
            }
        EXPECT(bad == 0);
    }
    EXPECT(co_emit_mm(&slot, x.view, w.view, y.view) == 0);   /* EW -> MM */
    EXPECT(rk_mock_tag_count() == after_ew);                 /* no new entry */
    memset(yd, 0, y.view.tensor.bytes);
    EXPECT(co_execute_profiled(TEST_FD, &slot, 1, CO_CHAIN, NULL, &profile) == 0);
    {
        unsigned bad = 0;
        for (unsigned r = 0; r < ROWS; r++)
            for (unsigned c = 0; c < COLS; c++) {
                double acc = 0;
                for (unsigned q = 0; q < COLS; q++)
                    acc += (double)xd[co_index(&x.view.tensor, r, q)] *
                           (double)wd[co_index(&w.view.tensor, c, q)];
                if ((double)yd[co_index(&y.view.tensor, r, c)] != (_Float16)acc) bad++;
            }
        EXPECT(bad == 0);
    }

    /* (2) a copy at a different address is NOT registered: it must be refused */
    co_program original, copy;
    memset(&original, 0, sizeof(original));
    EXPECT(co_emit_ew(&original, x.view, op.view, id.view, y.view, 0) == 0);
    unsigned entries = rk_mock_tag_count();
    memcpy(&copy, &original, sizeof(copy));
    memset(yd, 0x77, y.view.tensor.bytes);
    unsigned untracked_before = rk_mock_untracked();
    EXPECT(co_execute_profiled(TEST_FD, &copy, 1, CO_CHAIN, NULL, &profile) ==
           ROCKET_E_SHAPE);
    EXPECT(rk_mock_untracked() == untracked_before + 1);
    {
        unsigned touched = 0;
        unsigned char *raw = (unsigned char *)yd;
        for (size_t i = 0; i < y.view.tensor.bytes; i++)
            if (raw[i] != 0x77) touched++;
        EXPECT(touched == 0);
    }
    EXPECT(rk_mock_tag_count() == entries);
    EXPECT(rk_mock_live_bos() == baseline);
    EXPECT(rk_mock_resident() == res_baseline);
    EXPECT(co_execute_profiled(TEST_FD, &original, 1, CO_CHAIN, NULL, &profile) == 0);

    /* (3) a modified command word invalidates the registration until restored */
    uint64_t saved = original.words[0];   /* full width, so restore is exact */
    original.words[0] = saved ^ (uint64_t)1;
    EXPECT(co_execute_profiled(TEST_FD, &original, 1, CO_CHAIN, NULL, &profile) ==
           ROCKET_E_SHAPE);
    original.words[0] = saved;
    EXPECT(co_execute_profiled(TEST_FD, &original, 1, CO_CHAIN, NULL, &profile) == 0);

    rk_mock_reset();                       /* frees the snapshots it accumulated */
    EXPECT(rk_mock_tag_count() == 0);
    release(&x); release(&w); release(&op); release(&id); release(&y);
    EXPECT(rk_mock_live_bos() == 0);
    EXPECT(rk_mock_resident() == 0);
}

int main(void) {
    rk_mock_reset();
    add_mul_case();
    cleanup_case();
    capacity_case();
    tag_lifecycle_case();
    fprintf(stderr, "RK_MOCK scenarios checks=%u failures=%u\n", checks, failures);
    return failures ? 1 : 0;
}
