// SPDX-License-Identifier: GPL-3.0-or-later
// Narrow hardware probe for the add -> ReLU primitive fusion.
//
// Control: EW ADD -> EW MAX(zero) as two NPU jobs.
// Fused:   one EW ADD with EW_RELU_BYPASS cleared, same identity-conv main feed.
//
// The probe deliberately packs real fp16 cubes with the upstream feature_data
// layout and keeps the upstream minimum/padding rules (M>=12, M%4, CBUF guard).
// It reports bit-level mismatches and the exact input bits so a signed-zero,
// subnormal, cancellation or overflow difference cannot hide in a tolerance.
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "rocket_op.h"

#define NN 32
#define M_TILE 1020
#define M_MIN 12
#define CBUF_BANK 4096u
#define PROBE_MAX 8192

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static _Float16 f16bits(uint16_t bits) {
    _Float16 v;
    memcpy(&v, &bits, 2);
    return v;
}

static uint16_t bits16(_Float16 v) {
    uint16_t bits;
    memcpy(&bits, &v, 2);
    return bits;
}

static void die(const char *s) {
    fprintf(stderr, "ew_add_relu_probe: %s\n", s);
    exit(2);
}

static int submit_one(int fd, rocket_bo *rc, uint64_t *regs, int count,
                      rocket_bo *const *in, int nin, rocket_bo *out) {
    return rocket_op_submit_one(fd, "ew_add_relu_probe", rc, regs,
                                (uint32_t)count, in, nin, out);
}

static int run_tile(int fd, const _Float16 *a, const _Float16 *b,
                    _Float16 *out, int elems, int fused,
                    rocket_bo *wt, rocket_bo *zero, rocket_bo *rc,
                    uint64_t *regs, size_t regs_cap) {
    int total_rows = (elems + NN - 1) / NN;
    int M = total_rows < M_TILE ? total_rows : M_TILE;
    if (M < M_MIN) M = M_MIN;
    M = (M + 3) & ~3;
    size_t cube_sz = (size_t)M * NN * sizeof(_Float16) + CBUF_BANK;
    rocket_bo in_bo = {0}, op_bo = {0}, tmp_bo = {0}, out_bo = {0};
    int rcval = ROCKET_E_SHAPE;
    if (rocket_bo_alloc32(fd, cube_sz, &in_bo) ||
        rocket_bo_alloc32(fd, cube_sz, &op_bo) ||
        rocket_bo_alloc32(fd, cube_sz, &tmp_bo) ||
        rocket_bo_alloc32(fd, cube_sz, &out_bo))
        goto out;
    {
        rocket_bo *const set[] = {&in_bo, &op_bo, &tmp_bo, &out_bo, wt, zero, rc};
        if (rocket_op_iova_overflow("ew_add_relu_probe", set, 7)) goto out;
    }
    /* Pack the A and B cubes exactly as the upstream EW primitive does. */
    rocket_bo_prep(fd, &in_bo, 1, 0);
    rocket_bo_prep(fd, &op_bo, 1, 0);
    memset(in_bo.ptr, 0, in_bo.size);
    memset(op_bo.ptr, 0, op_bo.size);
    for (int t = 0; t < elems; t++) {
        int pos = feature_data(NN, M, 1, 8, (t % NN) + 1, (t / NN) + 1, 1);
        ((_Float16 *)in_bo.ptr)[pos] = a[t];
        ((_Float16 *)op_bo.ptr)[pos] = b[t];
    }
    rocket_bo_fini(fd, &in_bo);
    rocket_bo_fini(fd, &op_bo);
    rocket_bo_prep(fd, &out_bo, 1, 0);
    memset(out_bo.ptr, 0, out_bo.size);
    rocket_bo_fini(fd, &out_bo);

    matmul_params_t p;
    memset(&p, 0, sizeof(p));
    p.m = (uint16_t)M;
    p.k = NN;
    p.n = NN;
    p.input_dma = (uint32_t)in_bo.dma_address;
    p.weights_dma = (uint32_t)wt->dma_address;
    p.output_dma = (uint32_t)(fused ? out_bo.dma_address : tmp_bo.dma_address);
    p.tasks = regs;
    p.fp32tofp16 = 1;
    p.accumulate = 1;
    p.ew_op = 0;
    p.ew_mul = 0;
    p.ew_relu = (uint8_t)(fused ? 1 : 0);
    p.add_dma = (uint32_t)op_bo.dma_address;
    int g = gen_matmul_fp16(&p);
    if (g != 0 || p.task_count == 0 || (size_t)p.task_count > regs_cap) {
        fprintf(stderr, "gen task1 failed g=%d count=%u fused=%d\n", g,
                p.task_count, fused);
        goto out;
    }
    {
        rocket_bo *const ins[] = {&in_bo, wt, &op_bo};
        rocket_bo *dst = fused ? &out_bo : &tmp_bo;
        if (submit_one(fd, rc, regs, (int)p.task_count, ins, 3, dst) != 0) {
            fprintf(stderr, "submit task1 failed\n");
            goto out;
        }
    }
    if (!fused) {
        memset(&p, 0, sizeof(p));
        p.m = (uint16_t)M;
        p.k = NN;
        p.n = NN;
        p.input_dma = (uint32_t)tmp_bo.dma_address;
        p.weights_dma = (uint32_t)wt->dma_address;
        p.output_dma = (uint32_t)out_bo.dma_address;
        p.tasks = regs;
        p.fp32tofp16 = 1;
        p.accumulate = 1;
        p.ew_op = 2;             /* EW MAX */
        p.ew_mul = 0;
        p.ew_relu = 0;
        p.add_dma = (uint32_t)zero->dma_address;
        g = gen_matmul_fp16(&p);
        if (g != 0 || p.task_count == 0 || (size_t)p.task_count > regs_cap) {
            fprintf(stderr, "gen task2 failed g=%d count=%u\n", g, p.task_count);
            goto out;
        }
        {
            rocket_bo *const ins[] = {&tmp_bo, wt, zero};
            if (submit_one(fd, rc, regs, (int)p.task_count, ins, 3, &out_bo) != 0) {
                fprintf(stderr, "submit task2 failed\n");
                goto out;
            }
        }
    }
    if (rocket_bo_prep(fd, &out_bo, 0, 2000000000ull) != 0) {
        fprintf(stderr, "output prep failed\n");
        goto out;
    }
    for (int t = 0; t < elems; t++) {
        int pos = feature_data(NN, M, 1, 8, (t % NN) + 1, (t / NN) + 1, 1);
        out[t] = ((_Float16 *)out_bo.ptr)[pos];
    }
    rocket_bo_fini(fd, &out_bo);
    rcval = 0;
out:
    rocket_bo_free(fd, &out_bo);
    rocket_bo_free(fd, &tmp_bo);
    rocket_bo_free(fd, &op_bo);
    rocket_bo_free(fd, &in_bo);
    return rcval;
}

static int run_all(int fd, const _Float16 *a, const _Float16 *b,
                   _Float16 *out, int n, int fused,
                   rocket_bo *wt, rocket_bo *zero, rocket_bo *rc,
                   uint64_t *regs, size_t regs_cap) {
    int done = 0;
    while (done < n) {
        int elems = n - done;
        if (elems > M_TILE * NN) elems = M_TILE * NN;
        int r = run_tile(fd, a + done, b + done, out + done, elems, fused,
                         wt, zero, rc, regs, regs_cap);
        if (r) return r;
        done += elems;
    }
    return 0;
}

static void fill_boundary(_Float16 *a, _Float16 *b, int n) {
    static const uint16_t finite_bits[] = {
        0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff,
        0x0400, 0x8400, 0x3800, 0xb800, 0x3c00, 0xbc00,
        0x7bff, 0xfbff, 0x3555, 0xb555, 0x0002, 0x8002
    };
    static const uint16_t special_bits[] = {
        0x7c00, 0xfc00, 0x7e00
    };
    int nf = (int)(sizeof(finite_bits) / sizeof(finite_bits[0]));
    int ns = (int)(sizeof(special_bits) / sizeof(special_bits[0]));
    int idx = 0;
    /* Exhaustive finite boundary pairs. */
    for (int i = 0; i < nf; i++) {
        for (int j = 0; j < nf && idx < n; j++, idx++) {
            a[idx] = f16bits(finite_bits[i]);
            b[idx] = f16bits(finite_bits[j]);
        }
    }
    /* Exact cancellation and near-cancellation around every finite value. */
    for (int i = 0; i < nf && idx < n; i++) {
        _Float16 x = f16bits(finite_bits[i]);
        a[idx] = x;
        b[idx] = f16bits((uint16_t)(finite_bits[i] ^ 0x8000));
        idx++;
        if (idx < n) {
            a[idx] = x;
            b[idx] = f16bits((uint16_t)(finite_bits[i] ^ 0x8001));
            idx++;
        }
    }
    /* Special values are kept after the finite domain so they can be reported
     * separately by the caller. */
    for (int i = 0; i < ns && idx < n; i++) {
        a[idx] = f16bits(special_bits[i]);
        b[idx] = f16bits(0x0000);
        idx++;
    }
    /* Deterministic finite pseudo-random fill. */
    uint32_t s = 0x12345678u;
    while (idx < n) {
        s = s * 1664525u + 1013904223u;
        a[idx] = f16bits((uint16_t)(s & 0x7bff));
        s = s * 1664525u + 1013904223u;
        b[idx] = f16bits((uint16_t)((s & 0x7bff) | ((s >> 31) << 15)));
        idx++;
    }
}

int main(int argc, char **argv) {
    int n = PROBE_MAX;
    int include_special = 0;
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2 && !strcmp(argv[2], "special")) include_special = 1;
    if (n <= 0 || n > 1000000) die("bad element count");
    _Float16 *a = calloc((size_t)n, sizeof(*a));
    _Float16 *b = calloc((size_t)n, sizeof(*b));
    _Float16 *ctl = calloc((size_t)n, sizeof(*ctl));
    _Float16 *fus = calloc((size_t)n, sizeof(*fus));
    if (!a || !b || !ctl || !fus) die("host alloc");
    fill_boundary(a, b, n);
    int fd = rocket_open();
    if (fd < 0) die("rocket_open");
    rocket_bo wt = {0}, zero = {0}, rc_bo = {0};
    uint64_t regs[2048];
    if (rocket_bo_alloc32(fd, (size_t)NN * NN * sizeof(_Float16) + CBUF_BANK, &wt) ||
        rocket_bo_alloc32(fd, (size_t)M_TILE * NN * sizeof(_Float16) + CBUF_BANK, &zero) ||
        rocket_bo_alloc32(fd, sizeof(regs), &rc_bo))
        die("BO alloc");
    rocket_bo_prep(fd, &wt, 1, 0);
    memset(wt.ptr, 0, wt.size);
    for (int k = 1; k <= NN; k++)
        for (int c = 1; c <= NN; c++)
            ((_Float16 *)wt.ptr)[weight_fp16(NN, k, c)] =
                (_Float16)(k == c ? 1.0f : 0.0f);
    rocket_bo_fini(fd, &wt);
    rocket_bo_prep(fd, &zero, 1, 0);
    memset(zero.ptr, 0, zero.size);
    rocket_bo_fini(fd, &zero);
    uint64_t t0 = now_ns();
    if (run_all(fd, a, b, ctl, n, 0, &wt, &zero, &rc_bo, regs,
                sizeof(regs) / sizeof(regs[0])))
        die("control run");
    uint64_t t1 = now_ns();
    if (run_all(fd, a, b, fus, n, 1, &wt, &zero, &rc_bo, regs,
                sizeof(regs) / sizeof(regs[0])))
        die("fused run");
    uint64_t t2 = now_ns();
    int mismatches = 0, finite_mismatches = 0, special_mismatches = 0;
    for (int i = 0; i < n; i++) {
        uint16_t xb = bits16(a[i]), bb = bits16(b[i]);
        uint16_t cb = bits16(ctl[i]), fb = bits16(fus[i]);
        if (cb != fb) {
            int special = ((xb & 0x7c00) == 0x7c00) ||
                          ((bb & 0x7c00) == 0x7c00) ||
                          ((cb & 0x7c00) == 0x7c00) ||
                          ((fb & 0x7c00) == 0x7c00);
            if (special) special_mismatches++; else finite_mismatches++;
            if (mismatches < 16)
                printf("MISMATCH i=%d x=0x%04x b=0x%04x control=0x%04x fused=0x%04x%s\n",
                       i, xb, bb, cb, fb, special ? " special" : " finite");
            mismatches++;
        }
    }
    printf("EW_ADD_RELU_PROBE n=%d mismatches=%d finite=%d special=%d control_us=%.1f fused_us=%.1f\n",
           n, mismatches, finite_mismatches, special_mismatches,
           (double)(t1 - t0) / 1000.0, (double)(t2 - t1) / 1000.0);
    /* Failure behavior: ew_relu is only defined for accumulate ADD. */
    matmul_params_t bad;
    memset(&bad, 0, sizeof(bad));
    bad.m = 16; bad.k = NN; bad.n = NN; bad.tasks = regs;
    bad.accumulate = 0; bad.ew_relu = 1;
    int reject_no_accum = gen_matmul_fp16(&bad);
    memset(&bad, 0, sizeof(bad));
    bad.m = 16; bad.k = NN; bad.n = NN; bad.tasks = regs;
    bad.accumulate = 1; bad.ew_op = 2; bad.ew_relu = 1;
    int reject_max = gen_matmul_fp16(&bad);
    memset(&bad, 0, sizeof(bad));
    bad.m = 16; bad.k = NN; bad.n = NN; bad.tasks = regs;
    bad.accumulate = 1; bad.ew_mul = 1; bad.ew_relu = 1;
    int reject_mul = gen_matmul_fp16(&bad);
    printf("EW_ADD_RELU_PROBE_REJECT no_accum=%d max=%d mul=%d\n",
           reject_no_accum, reject_max, reject_mul);
    if (reject_no_accum != ROCKET_E_UNSUPPORTED ||
        reject_max != ROCKET_E_UNSUPPORTED ||
        reject_mul != ROCKET_E_UNSUPPORTED) {
        fprintf(stderr, "ew_relu invalid combination was not rejected\n");
        return 3;
    }
    printf("EW_ADD_RELU_PROBE_ZERO_PAIRS");
    for (int i = 0; i < n; i++) {
        if ((bits16(a[i]) == 0x0000 || bits16(a[i]) == 0x8000) &&
            (bits16(b[i]) == 0x0000 || bits16(b[i]) == 0x8000)) {
            printf(" x=0x%04x b=0x%04x ctl=0x%04x fused=0x%04x",
                   bits16(a[i]), bits16(b[i]), bits16(ctl[i]), bits16(fus[i]));
        }
    }
    printf("\n");
    rocket_bo_free(fd, &rc_bo);
    rocket_bo_free(fd, &zero);
    rocket_bo_free(fd, &wt);
    rocket_close(fd);
    free(a); free(b); free(ctl); free(fus);
    return finite_mismatches ? 1 : 0;
}
