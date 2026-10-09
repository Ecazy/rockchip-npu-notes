// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Device K-tiling validation and fused/unfused single-submit performance control
 * for the multi-K co_linear_hsig operator (RK3588).
 *
 * What it proves, per acceptance contract:
 *   contract : co_linear_hsig_run and co_linear_hsig_run_cube over the full K and
 *              shape matrix, with external 16/64 B view offsets, input-unchanged and
 *              outside-slice guards, finite output, full FP64 and tile-aware FP16
 *              accumulation references, and exactly one submit for every CO_CHAIN
 *              invocation.
 *   raw      : an INDEPENDENT accumulation chain built with co_emit_mm for tile 0 and
 *              co_emit_mm_acc for the rest. It reads the final UNACTIVATED accumulator,
 *              so activation saturation cannot hide a lost or corrupt K partial. It
 *              also checks single nonzero channels across tile boundaries, signed
 *              cancellation, and (control only) each running prefix before reuse.
 *   perf     : a device-resident fused graph against an explicit unfused graph, both
 *              one submit, same shape/input/weights, prepacked weights/bias, warmups,
 *              alternating order and >=20 interleaved reps. Wall and rknpu hardware
 *              elapsed are both reported; neither arm gets a different guard policy.
 *
 * Host mode is structural only and never touches the NPU. Device modes exit 77 when
 * no NPU is present; 77 is SKIP and is never treated as a pass.
 */
#define _GNU_SOURCE
#include "custom_ops.h"
#include "rknpu_submit.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Declared up front, before any hardware result exists. */
#define TOL_ACT  0.004  /* activated output vs either reference (LUT quantization + fp16) */
#define TOL_RAW  0.004  /* unactivated accumulator vs either reference (per-tile fp16 narrowing) */

static unsigned checks, failures;
static FILE *records;
static const char *witness_dir;
static unsigned core_mask = 1;
static unsigned witness_seq;

#define EXPECT(expr) do { checks++; if(!(expr)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #expr); failures++; } } while(0)

#define TRY(call) do { int rc_ = (call); if(rc_) { \
    fprintf(stderr, "FAIL %s:%d %s = %d\n", __FILE__, __LINE__, #call, rc_); \
    failures++; checks++; rc = rc_; goto cleanup; } } while(0)

/* ---------------------------------------------------------------- primitives */

static uint32_t rng_state;
static uint32_t rng_next(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return rng_state = x;
}

static double hs(double v) {
    double r = v / 6.0 + 0.5;
    return r < 0 ? 0 : r > 1 ? 1 : r;
}

/* Small dyadic draw, exactly representable in FP16: keeps diagnostic sums exact. */
static _Float16 dyadic(void) {
    static const float v[] = {0.0f, 0.0625f, -0.0625f, 0.03125f, -0.03125f,
                              0.015625f, -0.015625f, 0.0078125f};
    return (_Float16)v[rng_next() % (sizeof(v) / sizeof(v[0]))];
}

/* Small non-dyadic draw in [-0.05, 0.05]: exercises rounding over a range. */
static _Float16 random_small(void) {
    return (_Float16)(((double)(rng_next() % 10001) - 5000.0) / 100000.0);
}

static void fill_operand(_Float16 *x, size_t n, int pattern) {
    for (size_t i = 0; i < n; i++) x[i] = pattern ? random_small() : dyadic();
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}
static uint64_t percentile(uint64_t *v, unsigned n, double p) {
    uint64_t *sorted=malloc(n*sizeof(*sorted));
    if(!sorted){fprintf(stderr,"percentile allocation failed\n");exit(1);}
    memcpy(sorted,v,n*sizeof(*sorted));
    qsort(sorted, n, sizeof(*sorted), cmp_u64);
    unsigned i = (unsigned)(p * (double)(n - 1) + 0.5);
    if (i >= n) i = n - 1;
    uint64_t value=sorted[i];free(sorted);return value;
}

/* --------------------------------------------------------------- references */

/* Full FP64 mathematical sum and tile-aware FP16-narrowed sum, pre-activation.
 * The device narrows through FP16 once per K tile, exactly as this emulates; this is
 * NOT an fp32-exact claim. */
static void reference_variant(unsigned m, unsigned n, unsigned k,
                            const _Float16 *x, const _Float16 *w,
                            double *raw64, double *raw16, int unfused) {
    unsigned nt = co_k_tiles(k);
    for (unsigned r = 0; r < m; r++) {
        for (unsigned j = 0; j < n; j++) {
            double sum64 = 0, sum16 = 0;
            for (unsigned t = 0; t < nt; t++) {
                unsigned k0 = t * CO_K_TILE, kt = co_k_tile_length(k, t);
                double ts = 0;
                for (unsigned c = k0; c < k0 + kt; c++)
                    ts += (double)x[(size_t)r * k + c] * (double)w[(size_t)j * k + c];
                sum64 += ts;
                if(unfused)ts=(double)(_Float16)ts;
                sum16 = (t == 0) ? (double)(_Float16)ts : (double)(_Float16)(sum16 + ts);
            }
            raw64[(size_t)r * n + j] = sum64;
            raw16[(size_t)r * n + j] = sum16;
        }
    }
}

static void reference_block(unsigned m,unsigned n,unsigned k,const _Float16 *x,
                            const _Float16 *w,double *raw64,double *raw16) {
    reference_variant(m,n,k,x,w,raw64,raw16,0);
}

static double prefix16(unsigned r, unsigned j, unsigned k,
                       const _Float16 *x, const _Float16 *w, unsigned count) {
    double acc = 0;
    for (unsigned t = 0; t < count; t++) {
        unsigned k0 = t * CO_K_TILE, kt = co_k_tile_length(k, t);
        double ts = 0;
        for (unsigned c = k0; c < k0 + kt; c++)
            ts += (double)x[(size_t)r * k + c] * (double)w[(size_t)j * k + c];
        acc = (t == 0) ? (double)(_Float16)ts : (double)(_Float16)(acc + ts);
    }
    return acc;
}

typedef struct { double peak64, peak16, unsat; size_t bad; int finite; } score_t;

/* tol < 0 means bitwise-exact against (the FP16 encoding of) ref16. */
static void score_block(const char *who, size_t n, const _Float16 *got,
                        const double *ref64, const double *ref16,
                        double tol, score_t *st) {
    size_t bad = 0, unsat = 0;
    st->peak64 = st->peak16 = 0; st->finite = 1;
    for (size_t i = 0; i < n; i++) {
        double g = (double)got[i];
        if (!isfinite(g)) { st->finite = 0; bad++; continue; }
        double d64 = fabs(g - ref64[i]);
        double d16 = fabs(g - ref16[i]);
        if (d64 > st->peak64) st->peak64 = d64;
        if (d16 > st->peak16) st->peak16 = d16;
        int ok = (tol < 0.0) ? (got[i] == (_Float16)ref16[i])
                             : (d64 <= tol && d16 <= tol);
        if (!ok) bad++;
        if (ref64[i] > 0.02 && ref64[i] < 0.98) unsat++;
    }
    st->bad = bad;
    st->unsat = n ? (double)unsat / (double)n : 0;
    if (bad) fprintf(stderr, "FAIL %s bad=%zu peak64=%.6g peak16=%.6g finite=%d\n",
                     who, bad, st->peak64, st->peak16, st->finite);
    checks++;
    if (bad || !st->finite) failures++;
}

/* --------------------------------------------------------------- accounting */

static unsigned lut_chunk_count(unsigned m, unsigned n) {
    co_tensor y;
    if (co_tensor_matrix(&y, m, n, CO_F16)) return 0;
    unsigned e = y.physical_rows * y.physical_cols;
    return (e + 32767u) / 32768u;
}
static unsigned contract_tasks(unsigned m, unsigned n, unsigned k) {
    return co_k_tiles(k) + 1u + lut_chunk_count(m, n);
}
static unsigned perf_fused_tasks(unsigned m, unsigned n, unsigned k) {
    return co_k_tiles(k) + 1u + lut_chunk_count(m, n);
}
static unsigned perf_unfused_tasks(unsigned m, unsigned n, unsigned k) {
    return 2u * co_k_tiles(k) + lut_chunk_count(m, n);
}

/* Counters are process-cumulative; reset immediately before the measured invocation. */
static int submit_counts(const char *who, unsigned esubs, unsigned etasks,
                         uint64_t *subs_out, uint64_t *tasks_out) {
    uint64_t s = rocket_submit_ioctl_count(), t = rocket_submit_task_count();
    if (subs_out) *subs_out = s;
    if (tasks_out) *tasks_out = t;
    checks++;
    if (s != esubs || t != etasks) {
        fprintf(stderr, "FAIL %s submits=%llu expected=%u tasks=%llu expected=%u\n",
                who, (unsigned long long)s, esubs, (unsigned long long)t, etasks);
        failures++;
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------ buffer guards */

static int snapshot(co_buffer *b, unsigned char **out) {
    int rc = rocket_bo_prep(b->fd, &b->bo, 0, 2000000000ull);
    if (rc) return rc;
    unsigned char *copy = malloc(b->payload);
    if (!copy) { rocket_bo_fini(b->fd, &b->bo); return ROCKET_E_NOMEM; }
    memcpy(copy, b->bo.ptr, b->payload);
    rc = rocket_bo_fini(b->fd, &b->bo);
    if (rc) { free(copy); return rc; }
    *out = copy;
    return 0;
}
static int unchanged(co_buffer *b, const unsigned char *snap) {
    int rc = rocket_bo_prep(b->fd, &b->bo, 0, 2000000000ull);
    if (rc) return rc;
    if (memcmp(b->bo.ptr, snap, b->payload)) rc = ROCKET_E_DEVICE;
    int end = rocket_bo_fini(b->fd, &b->bo);
    return rc ? rc : end;
}
static int stamp_outside(co_view v, unsigned char fill) {
    int rc = rocket_bo_prep(v.buffer->fd, &v.buffer->bo, 1, 2000000000ull);
    if (rc) return rc;
    unsigned char *p = v.buffer->bo.ptr;
    memset(p, fill, v.offset);
    memset(p + v.offset + v.tensor.bytes, fill, v.buffer->payload - v.offset - v.tensor.bytes);
    return rocket_bo_fini(v.buffer->fd, &v.buffer->bo);
}
static int outside_intact(co_view v, unsigned char fill) {
    int rc = rocket_bo_prep(v.buffer->fd, &v.buffer->bo, 0, 2000000000ull);
    if (rc) return rc;
    unsigned char *base = v.buffer->bo.ptr;
    for (size_t i = 0; i < v.offset; i++) if (base[i] != fill) rc = ROCKET_E_DEVICE;
    unsigned char *p = base + v.offset + v.tensor.bytes;
    for (size_t i = 0; i < v.buffer->payload - v.offset - v.tensor.bytes; i++)
        if (p[i] != fill) rc = ROCKET_E_DEVICE;
    int end = rocket_bo_fini(v.buffer->fd, &v.buffer->bo);
    return rc ? rc : end;
}

/* --------------------------------------------------------------- witnesses */

static void write_file(const char *path, const void *data, size_t bytes) {
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(data, 1, bytes, f) != bytes || fclose(f))
        fprintf(stderr, "WARN witness write failed: %s\n", path);
}
static void witness(const char *kind, unsigned m, unsigned n, unsigned k,
                    const _Float16 *x, const _Float16 *w, const _Float16 *b,
                    const _Float16 *y, double peak64, double peak16,
                    unsigned long submits, unsigned long tasks, int pass) {
    if (!witness_dir) return;
    unsigned id = witness_seq++;
    char path[1024];
    char xf[1100], wf[1100], bf[1100], yf[1100];
    snprintf(xf, sizeof(xf), "%s/case-%03u-x.bin", witness_dir, id);
    snprintf(wf, sizeof(wf), "%s/case-%03u-w.bin", witness_dir, id);
    snprintf(bf, sizeof(bf), "%s/case-%03u-b.bin", witness_dir, id);
    snprintf(yf, sizeof(yf), "%s/case-%03u-y.bin", witness_dir, id);
    snprintf(path, sizeof(path), "%s/case-%03u.json", witness_dir, id);
    /* row-major logical operands, little-endian fp16 */
    {
        _Float16 *tmp = malloc(((size_t)m * k + (size_t)n * k + n + (size_t)m * n) * 2);
        if (!tmp) return;
        memcpy(tmp, x, (size_t)m * k * 2);
        memcpy(tmp + (size_t)m * k, w, (size_t)n * k * 2);
        if (b) memcpy(tmp + (size_t)m * k + (size_t)n * k, b, (size_t)n * 2);
        else memset(tmp + (size_t)m * k + (size_t)n * k, 0, (size_t)n * 2);
        memcpy(tmp + (size_t)m * k + (size_t)n * k + n, y, (size_t)m * n * 2);
        FILE *j = fopen(path, "w");
        if (j) {
            fprintf(j, "{\"id\":%u,\"kind\":\"%s\",\"m\":%u,\"n\":%u,\"k\":%u,"
                       "\"peak_abs_fp64\":%.9g,\"peak_abs_fp16\":%.9g,"
                       "\"submits\":%lu,\"tasks\":%lu,\"pass\":%s,"
                       "\"x\":\"%s\",\"weights\":\"%s\",\"bias\":\"%s\",\"y\":\"%s\"}\n",
                    id, kind, m, n, k, peak64, peak16, submits, tasks,
                    pass ? "true" : "false",
                    strrchr(xf, '/') + 1, strrchr(wf, '/') + 1,
                    strrchr(bf, '/') + 1, strrchr(yf, '/') + 1);
            fclose(j);
        }
        write_file(xf, tmp, (size_t)m * k * 2);
        write_file(wf, tmp + (size_t)m * k, (size_t)n * k * 2);
        write_file(bf, tmp + (size_t)m * k + (size_t)n * k, (size_t)n * 2);
        write_file(yf, tmp + (size_t)m * k + (size_t)n * k + n, (size_t)m * n * 2);
        free(tmp);
    }
}

/* ================================================================ raw chain */

typedef struct {
    int fd;
    unsigned m, n, k, nt, offx;
    co_tensor xf, yf, wt;
    co_buffer xb, a0b, a1b, *wb;
    co_view xv, a0v, a1v, *wv;
    co_program *p;
} rawchain;

static void rawchain_free(rawchain *c) {
    if (!c) return;
    for (unsigned i = 0; i < c->nt; i++) co_buffer_free(&c->wb[i]);
    co_buffer_free(&c->xb);
    co_buffer_free(&c->a0b);
    co_buffer_free(&c->a1b);
    free(c->wb); free(c->wv); free(c->p);
    memset(c, 0, sizeof(*c));
}

static int tile_view(co_view full, const co_tensor *fullt,
                     unsigned k0, unsigned kt, co_view *out) {
    co_tensor slice; size_t off;
    int rc = co_tensor_input_slice(&slice, fullt, k0, kt, &off);
    if (!rc) rc = co_view_init(out, full.buffer, full.offset + off, &slice);
    return rc;
}

static int rawchain_build(rawchain *c, int fd, unsigned m, unsigned n, unsigned k,
                          const _Float16 *x, const _Float16 *w, unsigned offx) {
    memset(c, 0, sizeof(*c));
    c->fd = fd; c->m = m; c->n = n; c->k = k; c->offx = offx; c->nt = co_k_tiles(k);
    int rc = co_tensor_input(&c->xf, m, k);
    if (!rc) rc = co_tensor_matrix(&c->yf, m, n, CO_F16);
    if (!rc) rc = co_buffer_alloc(fd, c->xf.bytes + offx, &c->xb);
    if (!rc) rc = co_view_init(&c->xv, &c->xb, offx, &c->xf);
    if (!rc) rc = co_upload_f16(&c->xv, x);
    if (rc) return rc;
    c->wb = calloc(c->nt, sizeof(*c->wb));
    c->wv = calloc(c->nt, sizeof(*c->wv));
    c->p = calloc(c->nt, sizeof(*c->p));
    if (!c->wb || !c->wv || !c->p) return ROCKET_E_NOMEM;
    for (unsigned t = 0; t < c->nt; t++) {
        unsigned k0 = t * CO_K_TILE, kt = co_k_tile_length(k, t);
        rc = co_tensor_weight(&c->wt, n, kt);
        if (!rc) rc = co_buffer_alloc(fd, c->wt.bytes, &c->wb[t]);
        if (!rc) rc = co_view_init(&c->wv[t], &c->wb[t], 0, &c->wt);
        if (rc) return rc;
        _Float16 *tile = malloc((size_t)n * kt * 2);
        if (!tile) return ROCKET_E_NOMEM;
        for (unsigned r = 0; r < n; r++)
            for (unsigned q = 0; q < kt; q++)
                tile[(size_t)r * kt + q] = w[(size_t)r * k + k0 + q];
        rc = co_upload_f16(&c->wv[t], tile);
        free(tile);
        if (rc) return rc;
    }
    rc = co_buffer_alloc(fd, c->yf.bytes, &c->a0b);
    if (!rc) rc = co_view_init(&c->a0v, &c->a0b, 0, &c->yf);
    if (!rc) rc = co_buffer_alloc(fd, c->yf.bytes, &c->a1b);
    if (!rc) rc = co_view_init(&c->a1v, &c->a1b, 0, &c->yf);
    if (rc) return rc;
    co_view xs;
    rc = tile_view(c->xv, &c->xf, 0, co_k_tile_length(k, 0), &xs);
    if (!rc) rc = co_emit_mm(&c->p[0], xs, c->wv[0], c->a0v);
    if (rc) return rc;
    for (unsigned t = 1; t < c->nt; t++) {
        rc = tile_view(c->xv, &c->xf, t * CO_K_TILE, co_k_tile_length(k, t), &xs);
        if (rc) return rc;
        co_view prior = (t & 1) ? c->a0v : c->a1v;
        co_view dst = (t & 1) ? c->a1v : c->a0v;
        rc = co_emit_mm_acc(&c->p[t], xs, c->wv[t], prior, dst);
        if (rc) return rc;
    }
    return 0;
}

static co_view rawchain_output(const rawchain *c, unsigned count) {
    if (count == 0) count = c->nt;
    return ((count - 1) & 1) ? c->a1v : c->a0v;
}

static int rawchain_execute(rawchain *c, unsigned count, unsigned etasks, const char *who) {
    rocket_submit_counters_reset();
    int rc = co_execute(c->fd, c->p, count, CO_CHAIN);
    if (rc) { failures++; checks++; return rc; }
    uint64_t subs = 0, tasks = 0;
    if (submit_counts(who, 1, etasks, &subs, &tasks)) return ROCKET_E_DEVICE;
    return 0;
}

/* Generic raw-accumulation validation with optional per-prefix controls. */
static int raw_run(int fd, const char *who, unsigned m, unsigned n, unsigned k,
                   const _Float16 *x, const _Float16 *w, int pattern,
                   int verify_prefix, int write_witness, int exact) {
    unsigned nt = co_k_tiles(k);
    double *r64 = malloc((size_t)m * n * sizeof(*r64));
    double *r16 = malloc((size_t)m * n * sizeof(*r16));
    double *p64 = malloc((size_t)m * n * sizeof(*p64));
    double *p16 = malloc((size_t)m * n * sizeof(*p16));
    _Float16 *got = malloc((size_t)m * n * 2);
    rawchain c; memset(&c, 0, sizeof(c));
    unsigned char *snap = NULL;
    int rc = 0;
    unsigned f0 = failures;
    if (!r64 || !r16 || !p64 || !p16 || !got) { rc = ROCKET_E_NOMEM; goto cleanup; }
    (void)pattern;
    if ((rc = rawchain_build(&c, fd, m, n, k, x, w, (witness_seq & 1) ? 64u : 16u))) goto cleanup;
    if ((rc = snapshot(&c.xb, &snap))) goto cleanup;
    reference_block(m, n, k, x, w, r64, r16);

    if (verify_prefix) {
        for (unsigned i = 1; i <= nt; i++) {
            if ((rc = rawchain_execute(&c, i, i, who))) goto cleanup;
            co_view out = rawchain_output(&c, i);
            if ((rc = co_download(&out, got))) goto cleanup;
            for (unsigned r = 0; r < m; r++)
                for (unsigned j = 0; j < n; j++) {
                    double s = prefix16(r, j, k, x, w, i);
                    p64[(size_t)r * n + j] = s;
                    p16[(size_t)r * n + j] = s;
                }
            score_t st; score_block("prefix", (size_t)m * n, got, p64, p16, TOL_RAW, &st);
        }
    }

    if ((rc = rawchain_execute(&c, nt, nt, who))) goto cleanup;
    co_view out = rawchain_output(&c, nt);
    if ((rc = co_download(&out, got))) goto cleanup;
    score_t st; score_block(who, (size_t)m * n, got, r64, r16, exact ? -1.0 : TOL_RAW, &st);
    for (unsigned i = 0; i < nt; i++)
        if ((rc = co_guard_check(&c.wb[i]))) goto cleanup;
    if ((rc = co_guard_check(&c.xb))) goto cleanup;
    if ((rc = unchanged(&c.xb, snap))) goto cleanup;
    checks++;
    if (records) {
        fprintf(records,
                "{\"event\":\"invocation\",\"kind\":\"raw\",\"who\":\"%s\",\"m\":%u,\"n\":%u,"
                "\"k\":%u,\"tiles\":%u,\"offset_x\":%u,\"submits\":1,\"tasks\":%u,"
                "\"expected_tasks\":%u,\"max_abs_fp64\":%.9g,\"max_abs_fp16\":%.9g,"
                "\"unsaturated_frac\":%.6g,\"finite\":%s,\"input_unchanged\":true,"
                "\"guards\":true,\"prefix_controls\":%d,\"exact\":%d,\"pass\":%s}\n",
                who, m, n, k, nt, c.offx, nt, nt, st.peak64, st.peak16, st.unsat,
                st.finite ? "true" : "false", verify_prefix ? (int)nt : 0, exact ? 1 : 0,
                (st.bad == 0 && st.finite) ? "true" : "false");
        fflush(records);
    }
    if (write_witness)
        witness("raw", m, n, k, x, w, NULL, got, st.peak64, st.peak16, 1, nt,
                st.bad == 0 && st.finite);
cleanup:
    free(r64); free(r16); free(p64); free(p16); free(got); free(snap);
    rawchain_free(&c);
    if (!rc && failures > f0) rc = ROCKET_E_DEVICE;
    return rc;
}

/* ============================================================ contract path */

static int contract_case(int fd, unsigned m, unsigned n, unsigned k, int pattern,
                         int write_witness) {
    size_t xn = (size_t)m * k, wn = (size_t)n * k, yn = (size_t)m * n;
    _Float16 *x = malloc(xn * 2), *w = malloc(wn * 2), *b = malloc(n * 2);
    _Float16 *yhost = malloc(yn * 2), *ycube = malloc(yn * 2);
    double *r64 = malloc(yn * sizeof(*r64)), *r16 = malloc(yn * sizeof(*r16));
    double *a64 = malloc(yn * sizeof(*a64)), *a16 = malloc(yn * sizeof(*a16));
    co_linear_hsig *ctx = NULL;
    co_buffer bx = {0}, by = {0};
    co_view vx, vy;
    unsigned char *snap = NULL;
    int rc = 0;
    unsigned f0 = failures;
    if (!x || !w || !b || !yhost || !ycube || !r64 || !r16 || !a64 || !a16)
        { rc = ROCKET_E_NOMEM; goto cleanup; }
    fill_operand(x, xn, pattern);
    fill_operand(w, wn, pattern);
    for (unsigned j = 0; j < n; j++) b[j] = pattern ? random_small() : dyadic();
    reference_block(m, n, k, x, w, r64, r16);
    for (unsigned r = 0; r < m; r++)
        for (unsigned j = 0; j < n; j++) {
            size_t i = (size_t)r * n + j;
            a64[i] = hs(r64[i] + (double)b[j]);
            a16[i] = hs((double)(_Float16)(r16[i] + (double)b[j]));
        }

    unsigned etasks = contract_tasks(m, n, k);
    if ((rc = co_linear_hsig_create(fd, m, k, n, w, b, CO_CHAIN, &ctx))) {
        fprintf(stderr, "FAIL create m=%u k=%u n=%u rc=%d\n", m, k, n, rc);
        failures++; checks++; goto cleanup;
    }

    /* Host run: uploads internally, one chained submit, guarded. */
    rocket_submit_counters_reset();
    if ((rc = co_linear_hsig_run(ctx, x, yhost))) { failures++; checks++; goto cleanup; }
    if ((rc = submit_counts("run", 1, etasks, NULL, NULL))) goto cleanup;
    score_t sh; memset(&sh, 0, sizeof(sh));
    score_block("run", yn, yhost, a64, a16, TOL_ACT, &sh);

    /* Cube handoff: external 16/64 B offsets, no outside-slice writes, input intact. */
    co_tensor xf, yf;
    if ((rc = co_tensor_input(&xf, m, k))) goto cleanup;
    if ((rc = co_tensor_matrix(&yf, m, n, CO_F16))) goto cleanup;
    unsigned offx = (checks & 1) ? 64u : 16u;
    unsigned offy = (checks & 1) ? 16u : 64u;
    if ((rc = co_buffer_alloc(fd, xf.bytes + offx, &bx))) goto cleanup;
    if ((rc = co_view_init(&vx, &bx, offx, &xf))) goto cleanup;
    if ((rc = co_upload_f16(&vx, x))) goto cleanup;
    if ((rc = co_buffer_alloc(fd, yf.bytes + offy, &by))) goto cleanup;
    if ((rc = co_view_init(&vy, &by, offy, &yf))) goto cleanup;
    if ((rc = stamp_outside(vx, 0x5a))) goto cleanup;
    if ((rc = stamp_outside(vy, 0xc3))) goto cleanup;
    if ((rc = snapshot(&bx, &snap))) goto cleanup;

    rocket_submit_counters_reset();
    if ((rc = co_linear_hsig_run_cube(ctx, vx, vy))) { failures++; checks++; goto cleanup; }
    if ((rc = submit_counts("run_cube", 1, etasks, NULL, NULL))) goto cleanup;
    if ((rc = unchanged(&bx, snap))) { failures++; checks++; goto cleanup; }
    if ((rc = outside_intact(vy, 0xc3))) { failures++; checks++; goto cleanup; }
    if ((rc = co_download(&vy, ycube))) goto cleanup;
    score_t sc; memset(&sc, 0, sizeof(sc));
    score_block("run_cube", yn, ycube, a64, a16, TOL_ACT, &sc);

    if (records) {
        fprintf(records,
                "{\"event\":\"invocation\",\"kind\":\"contract\",\"m\":%u,\"n\":%u,\"k\":%u,"
                "\"pattern\":%d,\"tiles\":%u,\"lut_chunks\":%u,\"offset_x\":%u,\"offset_y\":%u,"
                "\"submits\":1,\"tasks\":%u,\"expected_tasks\":%u,"
                "\"run_max_abs_fp64\":%.9g,\"run_max_abs_fp16\":%.9g,"
                "\"cube_max_abs_fp64\":%.9g,\"cube_max_abs_fp16\":%.9g,"
                "\"unsaturated_frac\":%.6g,\"finite\":%s,\"input_unchanged\":true,"
                "\"outside_intact\":true,\"guards\":true,\"pass\":%s}\n",
                m, n, k, pattern, co_k_tiles(k), lut_chunk_count(m, n), offx, offy,
                etasks, etasks, sh.peak64, sh.peak16, sc.peak64, sc.peak16, sc.unsat,
                (sh.finite && sc.finite) ? "true" : "false",
                (sh.bad == 0 && sc.bad == 0 && sh.finite && sc.finite) ? "true" : "false");
        fflush(records);
    }
    if (write_witness)
        witness("contract", m, n, k, x, w, b, ycube, sc.peak64, sc.peak16, 1, etasks,
                sh.bad == 0 && sc.bad == 0);
cleanup:
    co_linear_hsig_destroy(ctx);
    co_buffer_free(&bx); co_buffer_free(&by);
    free(x); free(w); free(b); free(yhost); free(ycube);
    free(r64); free(r16); free(a64); free(a16); free(snap);
    if (!rc && failures > f0) rc = ROCKET_E_DEVICE;
    return rc;
}

/* Reuse one resident context with changing inputs and alternate host/Cube entries. */
static int stress_case(int fd,unsigned m,unsigned n,unsigned k) {
    size_t xn=(size_t)m*k,wn=(size_t)n*k,yn=(size_t)m*n;
    _Float16 *x=malloc(xn*2),*w=malloc(wn*2),*b=malloc(n*2),*y=malloc(yn*2);
    double *r64=malloc(yn*8),*r16=malloc(yn*8),*a64=malloc(yn*8),*a16=malloc(yn*8);
    co_linear_hsig *ctx=NULL;co_buffer bx={0},by={0};co_view vx,vy;
    unsigned char *snap=NULL;int rc=0;unsigned f0=failures;
    if(!x||!w||!b||!y||!r64||!r16||!a64||!a16){rc=ROCKET_E_NOMEM;goto cleanup;}
    fill_operand(w,wn,1);for(unsigned j=0;j<n;j++)b[j]=dyadic();
    TRY(co_linear_hsig_create(fd,m,k,n,w,b,CO_CHAIN,&ctx));
    co_tensor ti,to;TRY(co_tensor_input(&ti,m,k));TRY(co_tensor_matrix(&to,m,n,CO_F16));
    TRY(co_buffer_alloc(fd,ti.bytes+128,&bx));TRY(co_view_init(&vx,&bx,64,&ti));
    TRY(co_buffer_alloc(fd,to.bytes+128,&by));TRY(co_view_init(&vy,&by,16,&to));
    for(unsigned iteration=0;iteration<8;iteration++) {
        for(size_t i=0;i<xn;i++)x[i]=(_Float16)(random_small()+(iteration&1?0.015625:-0.015625));
        reference_block(m,n,k,x,w,r64,r16);
        for(size_t i=0;i<yn;i++){a64[i]=hs(r64[i]+b[i%n]);a16[i]=hs((double)(_Float16)(r16[i]+b[i%n]));}
        unsigned et=contract_tasks(m,n,k);rocket_submit_counters_reset();
        if(iteration&1) {
            TRY(co_upload_f16(&vx,x));TRY(snapshot(&bx,&snap));
            TRY(stamp_outside(vy,0xc3));TRY(co_linear_hsig_run_cube(ctx,vx,vy));
            TRY(unchanged(&bx,snap));free(snap);snap=NULL;
            TRY(outside_intact(vy,0xc3));TRY(co_download(&vy,y));
        }else TRY(co_linear_hsig_run(ctx,x,y));
        TRY(submit_counts("stress",1,et,NULL,NULL));
        score_t s;score_block("changing-input",yn,y,a64,a16,TOL_ACT,&s);
        if(s.bad||!s.finite){rc=ROCKET_E_DEVICE;goto cleanup;}
        fprintf(records,"{\"event\":\"invocation\",\"kind\":\"stress\",\"m\":%u,\"n\":%u,\"k\":%u,\"iteration\":%u,\"cube\":%s,\"submits\":1,\"tasks\":%u,\"expected_tasks\":%u,\"finite\":true,\"guards\":true,\"input_unchanged\":true,\"max_abs_fp64\":%.9g,\"max_abs_fp16\":%.9g,\"pass\":true}\n",m,n,k,iteration,iteration&1?"true":"false",et,et,s.peak64,s.peak16);
    }
cleanup:
    co_linear_hsig_destroy(ctx);co_buffer_free(&bx);co_buffer_free(&by);free(snap);
    free(x);free(w);free(b);free(y);free(r64);free(r16);free(a64);free(a16);
    return rc?rc:failures>f0?ROCKET_E_DEVICE:0;
}

/* =============================================================== perf graph */

typedef struct {
    int fd;
    unsigned m, n, k, nt, lut;
    co_tensor xf, yf, wt, bf, idf, af;
    co_buffer xb, bb, idb, ab, yb, a0b, a1b, tb, *wb;
    co_view xv, bv, idv, av, yv, a0v, a1v, tv, *wv;
    co_program *pf, *pu;
    unsigned nf, nu;
} perfgraph;

static void perfgraph_free(perfgraph *g) {
    if (!g) return;
    for (unsigned i = 0; i < g->nt; i++) co_buffer_free(&g->wb[i]);
    co_buffer_free(&g->xb); co_buffer_free(&g->bb); co_buffer_free(&g->idb);
    co_buffer_free(&g->ab); co_buffer_free(&g->yb); co_buffer_free(&g->a0b);
    co_buffer_free(&g->a1b); co_buffer_free(&g->tb);
    free(g->wb); free(g->wv); free(g->pf); free(g->pu);
    memset(g, 0, sizeof(*g));
}

static int emit_lut_chunks(co_program *p, unsigned *idx, co_view src, co_view dst,
                           unsigned elements) {
    int rc = 0;
    if (elements <= 32768) {
        rc = co_emit_lut(&p[(*idx)++], src, dst, CO_HARDSIGMOID);
        return rc;
    }
    for (unsigned base = 0; !rc && base < elements; base += 32768) {
        unsigned chunk = elements - base;
        if (chunk > 32768) chunk = 32768;
        co_tensor t;
        co_view s, d;
        rc = co_tensor_linear(&t, chunk);
        if (!rc) rc = co_view_init(&s, src.buffer, src.offset + (size_t)base * 2, &t);
        if (!rc) rc = co_view_init(&d, dst.buffer, dst.offset + (size_t)base * 2, &t);
        if (!rc) rc = co_emit_lut(&p[(*idx)++], s, d, CO_HARDSIGMOID);
    }
    return rc;
}

static int perfgraph_build(perfgraph *g, int fd, unsigned m, unsigned n, unsigned k,
                           const _Float16 *x, const _Float16 *w, const _Float16 *bias) {
    memset(g, 0, sizeof(*g));
    g->fd = fd; g->m = m; g->n = n; g->k = k; g->nt = co_k_tiles(k);
    int rc = co_tensor_input(&g->xf, m, k);
    if (!rc) rc = co_tensor_matrix(&g->yf, m, n, CO_F16);
    if (!rc) rc = co_tensor_matrix(&g->bf, m, n, CO_F16);
    if (!rc) rc = co_tensor_matrix(&g->af, m, n, CO_F16);
    if (!rc) rc = co_tensor_weight(&g->idf, g->yf.physical_cols, g->yf.physical_cols);
    if (rc) return rc;
    unsigned e = g->yf.physical_rows * g->yf.physical_cols;
    g->lut = (e + 32767u) / 32768u;

    if ((rc = co_buffer_alloc(fd, g->xf.bytes, &g->xb))) return rc;
    if ((rc = co_view_init(&g->xv, &g->xb, 0, &g->xf))) return rc;
    if ((rc = co_upload_f16(&g->xv, x))) return rc;
    if ((rc = co_buffer_alloc(fd, g->bf.bytes, &g->bb))) return rc;
    if ((rc = co_view_init(&g->bv, &g->bb, 0, &g->bf))) return rc;
    if ((rc = co_bias_upload(&g->bv, bias))) return rc;
    if ((rc = co_buffer_alloc(fd, g->idf.bytes, &g->idb))) return rc;
    if ((rc = co_view_init(&g->idv, &g->idb, 0, &g->idf))) return rc;
    if ((rc = co_identity_upload(&g->idv))) return rc;
    if ((rc = co_buffer_alloc(fd, g->af.bytes, &g->ab))) return rc;
    if ((rc = co_view_init(&g->av, &g->ab, 0, &g->af))) return rc;
    if ((rc = co_buffer_alloc(fd, g->yf.bytes, &g->yb))) return rc;
    if ((rc = co_view_init(&g->yv, &g->yb, 0, &g->yf))) return rc;
    if ((rc = co_buffer_alloc(fd, g->yf.bytes, &g->a0b))) return rc;
    if ((rc = co_view_init(&g->a0v, &g->a0b, 0, &g->yf))) return rc;
    if ((rc = co_buffer_alloc(fd, g->yf.bytes, &g->a1b))) return rc;
    if ((rc = co_view_init(&g->a1v, &g->a1b, 0, &g->yf))) return rc;
    if ((rc = co_buffer_alloc(fd, g->yf.bytes, &g->tb))) return rc;
    if ((rc = co_view_init(&g->tv, &g->tb, 0, &g->yf))) return rc;

    g->wb = calloc(g->nt, sizeof(*g->wb));
    g->wv = calloc(g->nt, sizeof(*g->wv));
    g->pf = calloc(g->nt + 1 + g->lut, sizeof(*g->pf));
    g->pu = calloc(2 * g->nt + g->lut, sizeof(*g->pu));
    if (!g->wb || !g->wv || !g->pf || !g->pu) return ROCKET_E_NOMEM;
    for (unsigned t = 0; t < g->nt; t++) {
        unsigned k0 = t * CO_K_TILE, kt = co_k_tile_length(k, t);
        if ((rc = co_tensor_weight(&g->wt, n, kt))) return rc;
        if ((rc = co_buffer_alloc(fd, g->wt.bytes, &g->wb[t]))) return rc;
        if ((rc = co_view_init(&g->wv[t], &g->wb[t], 0, &g->wt))) return rc;
        _Float16 *tile = malloc((size_t)n * kt * 2);
        if (!tile) return ROCKET_E_NOMEM;
        for (unsigned r = 0; r < n; r++)
            for (unsigned q = 0; q < kt; q++)
                tile[(size_t)r * kt + q] = w[(size_t)r * k + k0 + q];
        rc = co_upload_f16(&g->wv[t], tile);
        free(tile);
        if (rc) return rc;
    }

    /* Fused: exactly the production emission shape. */
    {
        unsigned idx = 0; co_view xs;
        rc = tile_view(g->xv, &g->xf, 0, co_k_tile_length(k, 0), &xs);
        if (!rc) rc = co_emit_mm(&g->pf[idx++], xs, g->wv[0], g->a0v);
        for (unsigned t = 1; !rc && t < g->nt; t++) {
            rc = tile_view(g->xv, &g->xf, t * CO_K_TILE, co_k_tile_length(k, t), &xs);
            if (rc) break;
            co_view prior = (t & 1) ? g->a0v : g->a1v;
            co_view dst = (t & 1) ? g->a1v : g->a0v;
            rc = co_emit_mm_acc(&g->pf[idx++], xs, g->wv[t], prior, dst);
        }
        co_view acc = ((g->nt - 1) & 1) ? g->a1v : g->a0v;
        if (!rc) rc = co_emit_ew(&g->pf[idx++], acc, g->bv, g->idv, g->av, 0);
        if (!rc) rc = emit_lut_chunks(g->pf, &idx, g->av, g->yv, e);
        if (rc) return rc;
        g->nf = idx;
    }
    /* Unfused: plain MM to a temporary, then an identity-conv EW add of prior+temp
     * into the ping-pong accumulator. Same bias and LUT epilogue. */
    {
        unsigned idx = 0; co_view xs;
        rc = tile_view(g->xv, &g->xf, 0, co_k_tile_length(k, 0), &xs);
        if (!rc) rc = co_emit_mm(&g->pu[idx++], xs, g->wv[0], g->a0v);
        for (unsigned t = 1; !rc && t < g->nt; t++) {
            rc = tile_view(g->xv, &g->xf, t * CO_K_TILE, co_k_tile_length(k, t), &xs);
            if (rc) break;
            rc = co_emit_mm(&g->pu[idx++], xs, g->wv[t], g->tv);
            if (rc) break;
            co_view prior = (t & 1) ? g->a0v : g->a1v;
            co_view dst = (t & 1) ? g->a1v : g->a0v;
            rc = co_emit_ew(&g->pu[idx++], g->tv, prior, g->idv, dst, 0);
        }
        co_view acc = ((g->nt - 1) & 1) ? g->a1v : g->a0v;
        if (!rc) rc = co_emit_ew(&g->pu[idx++], acc, g->bv, g->idv, g->av, 0);
        if (!rc) rc = emit_lut_chunks(g->pu, &idx, g->av, g->yv, e);
        if (rc) return rc;
        g->nu = idx;
    }
    return 0;
}

static int perfgraph_guards(perfgraph *g) {
    int rc = 0;
    for (unsigned i = 0; !rc && i < g->nt; i++) rc = co_guard_check(&g->wb[i]);
    if (!rc) rc = co_guard_check(&g->xb);
    if (!rc) rc = co_guard_check(&g->bb);
    if (!rc) rc = co_guard_check(&g->idb);
    if (!rc) rc = co_guard_check(&g->ab);
    if (!rc) rc = co_guard_check(&g->yb);
    if (!rc) rc = co_guard_check(&g->a0b);
    if (!rc) rc = co_guard_check(&g->a1b);
    if (!rc) rc = co_guard_check(&g->tb);
    return rc;
}

static int perfexec(perfgraph *g, co_program *p, unsigned count, unsigned etasks,
                    const char *who, uint64_t *wall, uint64_t *hw) {
    rocket_submit_counters_reset();
    uint64_t t0 = now_ns();
    int rc = co_execute(g->fd, p, count, CO_CHAIN);
    uint64_t t1 = now_ns();
    if (rc) { failures++; checks++; return rc; }
    uint64_t subs = 0, tasks = 0;
    if (submit_counts(who, 1, etasks, &subs, &tasks)) return ROCKET_E_DEVICE;
    uint64_t nhw = 0;
    uint64_t h = rknpu_hw_elapse_total_ns(&nhw);
    if (wall) *wall = t1 - t0;
    if (hw) *hw = h;
    return 0;
}

static int perf_case(int fd, unsigned m, unsigned n, unsigned k, unsigned reps,
                     int write_witness) {
    size_t xn = (size_t)m * k, wn = (size_t)n * k, yn = (size_t)m * n;
    _Float16 *x = malloc(xn * 2), *w = malloc(wn * 2), *b = malloc(n * 2);
    _Float16 *yf = malloc(yn * 2), *yu = malloc(yn * 2);
    double *r64 = malloc(yn * sizeof(*r64)), *r16 = malloc(yn * sizeof(*r16));
    double *a64 = malloc(yn * sizeof(*a64)), *a16 = malloc(yn * sizeof(*a16));
    uint64_t *wf = calloc(reps, sizeof(*wf)), *hf = calloc(reps, sizeof(*hf));
    uint64_t *wu = calloc(reps, sizeof(*wu)), *hu = calloc(reps, sizeof(*hu));
    perfgraph g; memset(&g, 0, sizeof(g));
    int rc = 0;
    unsigned f0 = failures;
    if (!x || !w || !b || !yf || !yu || !r64 || !r16 || !a64 || !a16 ||
        !wf || !hf || !wu || !hu) { rc = ROCKET_E_NOMEM; goto cleanup; }
    fill_operand(x, xn, 1);
    fill_operand(w, wn, 1);
    for (unsigned j = 0; j < n; j++) b[j] = random_small();
    reference_block(m, n, k, x, w, r64, r16);
    for (unsigned r = 0; r < m; r++)
        for (unsigned j = 0; j < n; j++) {
            size_t i = (size_t)r * n + j;
            a64[i] = hs(r64[i] + (double)b[j]);
            a16[i] = hs((double)(_Float16)(r16[i] + (double)b[j]));
        }
    if ((rc = perfgraph_build(&g, fd, m, n, k, x, w, b))) goto cleanup;
    unsigned ef = perf_fused_tasks(m, n, k), eu = perf_unfused_tasks(m, n, k);

    /* Validate each arm before timing it. */
    if ((rc = perfexec(&g, g.pf, g.nf, ef, "perf-fused", NULL, NULL))) goto cleanup;
    if ((rc = perfgraph_guards(&g))) goto cleanup;
    if ((rc = co_download(&g.yv, yf))) goto cleanup;
    score_t sf; memset(&sf, 0, sizeof(sf));
    score_block("perf-fused", yn, yf, a64, a16, TOL_ACT, &sf);

    if ((rc = perfexec(&g, g.pu, g.nu, eu, "perf-unfused", NULL, NULL))) goto cleanup;
    if ((rc = perfgraph_guards(&g))) goto cleanup;
    if ((rc = co_download(&g.yv, yu))) goto cleanup;
    reference_variant(m,n,k,x,w,r64,r16,1);
    for(size_t i=0;i<yn;i++)a16[i]=hs((double)(_Float16)(r16[i]+b[i%n]));
    score_t su; memset(&su, 0, sizeof(su));
    score_block("perf-unfused", yn, yu, a64, a16, TOL_ACT, &su);

    /* Warmups (4 each), then alternating-order interleaved reps. */
    for (unsigned i = 0; i < 4; i++) {
        if ((rc = perfexec(&g, g.pf, g.nf, ef, "warm-fused", NULL, NULL))) goto cleanup;
        if ((rc = perfexec(&g, g.pu, g.nu, eu, "warm-unfused", NULL, NULL))) goto cleanup;
    }
    for (unsigned i = 0; i < reps; i++) {
        if (i & 1u) {
            if ((rc = perfexec(&g, g.pu, g.nu, eu, "time-unfused", &wu[i], &hu[i]))) goto cleanup;
            if ((rc = perfexec(&g, g.pf, g.nf, ef, "time-fused", &wf[i], &hf[i]))) goto cleanup;
        } else {
            if ((rc = perfexec(&g, g.pf, g.nf, ef, "time-fused", &wf[i], &hf[i]))) goto cleanup;
            if ((rc = perfexec(&g, g.pu, g.nu, eu, "time-unfused", &wu[i], &hu[i]))) goto cleanup;
        }
    }

    if (records) {
        uint64_t fm = percentile(wf, reps, 0.5), f10 = percentile(wf, reps, 0.1),
                 f90 = percentile(wf, reps, 0.9);
        uint64_t um = percentile(wu, reps, 0.5), u10 = percentile(wu, reps, 0.1),
                 u90 = percentile(wu, reps, 0.9);
        uint64_t fhm = percentile(hf, reps, 0.5), fh10 = percentile(hf, reps, 0.1),
                 fh90 = percentile(hf, reps, 0.9);
        uint64_t uhm = percentile(hu, reps, 0.5), uh10 = percentile(hu, reps, 0.1),
                 uh90 = percentile(hu, reps, 0.9);
        fprintf(records,
                "{\"event\":\"perf\",\"arm\":\"fused\",\"m\":%u,\"n\":%u,\"k\":%u,"
                "\"tiles\":%u,\"lut_chunks\":%u,\"submits\":1,\"tasks\":%u,"
                "\"expected_tasks\":%u,\"max_abs_fp64\":%.9g,\"max_abs_fp16\":%.9g,"
                "\"unsaturated_frac\":%.6g,\"finite\":%s,\"guards\":true,\"reps\":%u,"
                "\"wall_median\":%llu,\"wall_p10\":%llu,\"wall_p90\":%llu,"
                "\"hw_median\":%llu,\"hw_p10\":%llu,\"hw_p90\":%llu,"
                "\"wall_ns\":[", m, n, k, g.nt, g.lut, g.nf, ef, sf.peak64, sf.peak16,
                sf.unsat, sf.finite ? "true" : "false", reps,
                (unsigned long long)fm, (unsigned long long)f10, (unsigned long long)f90,
                (unsigned long long)fhm, (unsigned long long)fh10, (unsigned long long)fh90);
        for (unsigned i = 0; i < reps; i++) fprintf(records, "%s%llu", i ? "," : "", (unsigned long long)wf[i]);
        fprintf(records, "],\"hw_ns\":[");
        for (unsigned i = 0; i < reps; i++) fprintf(records, "%s%llu", i ? "," : "", (unsigned long long)hf[i]);
        fprintf(records, "],\"pass\":%s}\n", (sf.bad == 0 && sf.finite) ? "true" : "false");
        fprintf(records,
                "{\"event\":\"perf\",\"arm\":\"unfused\",\"m\":%u,\"n\":%u,\"k\":%u,"
                "\"tiles\":%u,\"lut_chunks\":%u,\"submits\":1,\"tasks\":%u,"
                "\"expected_tasks\":%u,\"max_abs_fp64\":%.9g,\"max_abs_fp16\":%.9g,"
                "\"unsaturated_frac\":%.6g,\"finite\":%s,\"guards\":true,\"reps\":%u,"
                "\"wall_median\":%llu,\"wall_p10\":%llu,\"wall_p90\":%llu,"
                "\"hw_median\":%llu,\"hw_p10\":%llu,\"hw_p90\":%llu,"
                "\"wall_ns\":[", m, n, k, g.nt, g.lut, g.nu, eu, su.peak64, su.peak16,
                su.unsat, su.finite ? "true" : "false", reps,
                (unsigned long long)um, (unsigned long long)u10, (unsigned long long)u90,
                (unsigned long long)uhm, (unsigned long long)uh10, (unsigned long long)uh90);
        for (unsigned i = 0; i < reps; i++) fprintf(records, "%s%llu", i ? "," : "", (unsigned long long)wu[i]);
        fprintf(records, "],\"hw_ns\":[");
        for (unsigned i = 0; i < reps; i++) fprintf(records, "%s%llu", i ? "," : "", (unsigned long long)hu[i]);
        fprintf(records, "],\"pass\":%s}\n", (su.bad == 0 && su.finite) ? "true" : "false");
        fflush(records);
    }
    if (write_witness) {
        witness("perf-fused", m, n, k, x, w, b, yf, sf.peak64, sf.peak16, 1, g.nf,
                sf.bad == 0 && sf.finite);
        witness("perf-unfused", m, n, k, x, w, b, yu, su.peak64, su.peak16, 1, g.nu,
                su.bad == 0 && su.finite);
    }
    if ((rc = perfgraph_guards(&g))) goto cleanup;
cleanup:
    perfgraph_free(&g);
    free(x); free(w); free(b); free(yf); free(yu);
    free(r64); free(r16); free(a64); free(a16);
    free(wf); free(hf); free(wu); free(hu);
    if (!rc && failures > f0) rc = ROCKET_E_DEVICE;
    return rc;
}

/* =================================================================== driver */

/* Contract shape matrix: the exact K regression set and ragged M/N, plus one
 * 256x256 output (two LUT chunks) at K=4096. */
static const struct { unsigned m, n, k; int pattern; } CONTRACT[] = {
    {1, 1, 255, 0}, {16, 32, 256, 0}, {29, 45, 257, 1}, {64, 64, 511, 0},
    {1, 1, 512, 1}, {16, 32, 513, 0}, {29, 45, 769, 1}, {64, 64, 1024, 0},
    {16, 32, 2049, 1}, {29, 45, 4096, 0}, {256, 256, 4096, 1},
};

/* --------------------------------------------------------------- host mode */

static unsigned host_case(unsigned m, unsigned k) {
    co_tensor full;
    if (co_tensor_input(&full, m, k)) return 1;
    if (full.physical_rows != (m < 12 ? 12u : (m + 3u) / 4u * 4u)) return 1;
    if (full.physical_cols != (k + 31u) / 32u * 32u) return 1;
    size_t expected_off = 0;
    unsigned total = 0;
    for (unsigned i = 0; i < co_k_tiles(k); i++) {
        unsigned k0 = i * CO_K_TILE, kt = co_k_tile_length(k, i);
        co_tensor slice; size_t off = 0;
        if (co_tensor_input_slice(&slice, &full, k0, kt, &off)) return 1;
        if (off != (size_t)k0 * full.physical_rows * 2 || off != expected_off) return 1;
        if (co_index(&full, 0, k0) != off / 2) return 1;
        expected_off = off + slice.bytes;
        total += kt;
    }
    if (total != k || expected_off != full.bytes) return 1;
    if (contract_tasks(m, 1, k) != co_k_tiles(k) + 2u) return 1;
    return 0;
}

static int host_mode(void) {
    const unsigned ks[] = {255, 256, 257, 511, 512, 513, 769, 1024, 2049, 4096};
    const unsigned ms[] = {1, 16, 29, 64, 16, 32, 64, 16, 32, 256};
    for (unsigned i = 0; i < sizeof(ks) / sizeof(ks[0]); i++)
        EXPECT(host_case(ms[i], ks[i]) == 0);
    EXPECT(lut_chunk_count(256, 256) == 2);
    EXPECT(lut_chunk_count(64, 64) == 1);
    EXPECT(perf_fused_tasks(64, 64, 4096) == 18);
    EXPECT(perf_unfused_tasks(64, 64, 4096) == 33);
    EXPECT(co_k_tiles(255) == 1 && co_k_tiles(256) == 1 && co_k_tiles(257) == 2);
    printf("KTILING_HOST checks=%u failures=%u %s\n", checks, failures,
           failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "host")) return host_mode();
    const char *mode = argc > 1 ? argv[1] : "all";
    unsigned reps = argc > 2 ? (unsigned)strtoul(argv[2], NULL, 0) : 24;
    const char *mask_env = getenv("RKNPU_CORE_MASK");
    if (mask_env) core_mask = (unsigned)strtoul(mask_env, NULL, 0);
    const char *json = getenv("KT_PROBE_JSON");
    witness_dir = getenv("KT_WITNESS_DIR");
    if (strcmp(mode, "all") && strcmp(mode, "contract") && strcmp(mode, "raw") &&
        strcmp(mode, "perf")) {
        fprintf(stderr, "usage: RKNPU_CORE_MASK=1 KT_PROBE_JSON=f %s host|all|contract|raw|perf [reps]\n", argv[0]);
        return 1;
    }
    if (!json || reps < 20 || (core_mask != 1 && core_mask != 2 && core_mask != 4)) {
        fprintf(stderr, "device mode needs KT_PROBE_JSON, RKNPU_CORE_MASK in {1,2,4}, reps>=20\n");
        return 1;
    }
    records = fopen(json, "w");
    if (!records) { perror(json); return 1; }
    int fd = rocket_open();
    if (fd < 0) {
        fprintf(stderr, "SKIP no NPU: %d\n", fd);
        fprintf(records, "{\"event\":\"summary\",\"status\":\"SKIP\",\"reason\":\"no NPU\"}\n");
        fclose(records);
        return 77;
    }
    int rc = 0;
    if (!strcmp(mode, "all") || !strcmp(mode, "contract")) {
        for (unsigned i = 0; i < sizeof(CONTRACT) / sizeof(CONTRACT[0]); i++) {
            rng_state = 0x9e3779b9u * (i + 1u) + core_mask * 131u + 7u;
            int w = witness_dir && (i % 3u == 0u || CONTRACT[i].k <= 512u);
            if ((rc = contract_case(fd, CONTRACT[i].m, CONTRACT[i].n, CONTRACT[i].k,
                                    CONTRACT[i].pattern, w))) break;
        }
        if(!rc){rng_state=0x93451u+core_mask;rc=stress_case(fd,29,45,769);}
        if(!rc){rng_state=0x23451u+core_mask;rc=stress_case(fd,16,32,4096);}
    }
    if (!rc && (!strcmp(mode, "all") || !strcmp(mode, "raw"))) {
        const struct { unsigned m, n, k; unsigned kk; } CH[] = {
            {16, 32, 255, 254}, {16, 32, 256, 255}, {16, 32, 257, 256},
            {16, 32, 513, 257}, {29, 45, 511, 255}, {29, 45, 512, 256},
            {29, 45, 513, 512}, {29, 45, 1024, 511},
            {32, 16, 769, 768}, {32, 16, 1024, 1023}, {32, 16, 2049, 2048},
            {64, 64, 4096, 4095},
        };
        for (unsigned i = 0; !rc && i < sizeof(CH) / sizeof(CH[0]); i++) {
            unsigned m = CH[i].m, n = CH[i].n, k = CH[i].k, kk = CH[i].kk;
            _Float16 *x = calloc((size_t)m * k, 2), *w = calloc((size_t)n * k, 2);
            if (!x || !w) { free(x); free(w); rc = ROCKET_E_NOMEM; break; }
            for (unsigned r = 0; r < m; r++) x[(size_t)r * k + kk] = (_Float16)1.0;
            for (unsigned j = 0; j < n; j++)
                w[(size_t)j * k + kk] = (_Float16)((j & 1u) ? -0.0625f : 0.0625f);
            rng_state = 0x2545f491u * (i + 1u) + core_mask;
            char who[64]; snprintf(who, sizeof(who), "chan-k%u", kk);
            rc = raw_run(fd, who, m, n, k, x, w, 0, 0, witness_dir != NULL, 1);
            free(x); free(w);
        }
        if (!rc) {
            unsigned m = 16, n = 16, k = 513, a = 17, bb = 300;
            _Float16 *x = calloc((size_t)m * k, 2), *w = calloc((size_t)n * k, 2);
            if (!x || !w) { free(x); free(w); rc = ROCKET_E_NOMEM; }
            else {
                for (unsigned r = 0; r < m; r++) {
                    x[(size_t)r * k + a] = (_Float16)1.0;
                    x[(size_t)r * k + bb] = (_Float16)1.0;
                }
                for (unsigned j = 0; j < n; j++) {
                    w[(size_t)j * k + a] = (_Float16)0.0625f;
                    w[(size_t)j * k + bb] = (_Float16)-0.0625f;
                }
                rc = raw_run(fd, "cancel-k513", m, n, k, x, w, 0, 0, witness_dir != NULL, 1);
                free(x); free(w);
            }
        }
        const unsigned ks[] = {257, 769, 1024};
        for (unsigned i = 0; !rc && i < sizeof(ks) / sizeof(ks[0]); i++) {
            unsigned m = 16, n = 32, k = ks[i];
            _Float16 *x = malloc((size_t)m * k * 2), *w = malloc((size_t)n * k * 2);
            if (!x || !w) { free(x); free(w); rc = ROCKET_E_NOMEM; break; }
            rng_state = 0x27d4eb2fu * (i + 1u) + core_mask;
            fill_operand(x, (size_t)m * k, i & 1u);
            fill_operand(w, (size_t)n * k, i & 1u);
            rc = raw_run(fd, "prefix", m, n, k, x, w, i & 1u, 1, 0, 0);
            free(x); free(w);
        }
    }
    if (!rc && (!strcmp(mode, "all") || !strcmp(mode, "perf"))) {
        const struct { unsigned m, n, k; } P[] = {{64, 64, 4096}, {256, 256, 1024}};
        for (unsigned i = 0; !rc && i < sizeof(P) / sizeof(P[0]); i++) {
            rng_state = 0xb5297a4du * (i + 3u) + core_mask * 17u;
            rc = perf_case(fd, P[i].m, P[i].n, P[i].k, reps, witness_dir && i == 0);
        }
    }
    rocket_close(fd);
    unsigned total_failures = failures + (rc != 0);
    fprintf(records,
            "{\"event\":\"summary\",\"mode\":\"%s\",\"core_mask\":%u,\"reps\":%u,"
            "\"checks\":%u,\"failures\":%u,\"status\":\"%s\"}\n",
            mode, core_mask, reps, checks, total_failures, total_failures ? "FAIL" : "PASS");
    fclose(records);
    printf("KTILING_SUMMARY checks=%u failures=%u status=%s\n",
           checks, total_failures, total_failures ? "FAIL" : "PASS");
    return total_failures ? 1 : 0;
}
