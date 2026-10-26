// SPDX-License-Identifier: GPL-3.0-or-later
/* Primitive cost floor of one RKPlan call.
 *
 * The plan envelope is dominated by a handful of repeated primitives: BO cache
 * sync ioctls, the guard scan, the blocking submit and the download copy.  This
 * probe measures each of them in isolation on the target so an attributed
 * envelope can be checked against the cost of the primitives it is made of.
 *
 * It is a measurement tool, not an execution path: nothing here is used by the
 * runtime.  On a host without a device every case is skipped and the probe
 * reports "unsupported" rather than fabricating numbers.
 */
#define _POSIX_C_SOURCE 200809L
#include "custom_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ITERS 200
#define WARMUP 20

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void report(const char *name, double *samples, unsigned n) {
    if (!n) {
        printf("{\"primitive\":\"%s\",\"status\":\"unsupported\"}\n", name);
        return;
    }
    qsort(samples, n, sizeof(double), cmp_double);
    double sum = 0.0;
    for (unsigned i = 0; i < n; i++) sum += samples[i];
    printf("{\"primitive\":\"%s\",\"status\":\"ok\",\"iters\":%u,"
           "\"median_us\":%.4f,\"mean_us\":%.4f,\"min_us\":%.4f,"
           "\"p90_us\":%.4f,\"max_us\":%.4f}\n",
           name, n, samples[n / 2], sum / n, samples[0],
           samples[(n * 9) / 10], samples[n - 1]);
}

int main(void) {
    int fd = rocket_open();
    if (fd < 0) {
        printf("{\"probe\":\"plan_envelope_primitives\",\"status\":"
               "\"unsupported\",\"reason\":\"rocket_open failed\"}\n");
        return 77;
    }
    double samples[ITERS];

    /* 1. The timing primitive itself. */
    for (unsigned i = 0; i < WARMUP; i++) (void)now_us();
    for (unsigned i = 0; i < ITERS; i++) {
        double a = now_us();
        double b = now_us();
        samples[i] = b - a;
    }
    report("clock_gettime", samples, ITERS);

    /* A 16x16 FP16 cube: the geometry of one Tiny-ViT head matmul. */
    co_tensor cube;
    if (co_tensor_matrix(&cube, 16, 16, CO_F16)) {
        printf("{\"probe\":\"plan_envelope_primitives\",\"status\":"
               "\"invalid\"}\n");
        rocket_close(fd);
        return 1;
    }
    co_buffer buf = {0};
    co_view view;
    int rc = co_buffer_alloc(fd, cube.bytes, &buf);
    if (rc || co_view_init(&view, &buf, 0, &cube)) {
        printf("{\"probe\":\"plan_envelope_primitives\",\"status\":"
               "\"alloc_failed\",\"native\":%d}\n", rc);
        rocket_close(fd);
        return 1;
    }

    /* 2. One cache-sync round trip, both directions: what every upload,
     *    download and guard pays before touching a BO. */
    for (unsigned i = 0; i < WARMUP; i++) {
        (void)rocket_bo_prep(fd, &buf.bo, 0, 2000000000ull);
        (void)rocket_bo_fini(fd, &buf.bo);
    }
    for (unsigned i = 0; i < ITERS; i++) {
        double a = now_us();
        (void)rocket_bo_prep(fd, &buf.bo, 0, 2000000000ull);
        (void)rocket_bo_fini(fd, &buf.bo);
        samples[i] = now_us() - a;
    }
    report("bo_sync_read_pair", samples, ITERS);

    for (unsigned i = 0; i < ITERS; i++) {
        double a = now_us();
        (void)rocket_bo_prep(fd, &buf.bo, 1, 2000000000ull);
        (void)rocket_bo_fini(fd, &buf.bo);
        samples[i] = now_us() - a;
    }
    report("bo_sync_write_pair", samples, ITERS);

    /* 3. The publication guard as the runtime runs it. */
    for (unsigned i = 0; i < ITERS; i++) {
        double a = now_us();
        (void)co_guard_check(&buf);
        samples[i] = now_us() - a;
    }
    report("guard_check", samples, ITERS);

    /* 4. Download and upload of one cube. */
    _Float16 host[16 * 16];
    for (unsigned i = 0; i < ITERS; i++) {
        double a = now_us();
        (void)co_download(&view, host);
        samples[i] = now_us() - a;
    }
    report("download_cube16", samples, ITERS);
    for (unsigned i = 0; i < ITERS; i++) {
        double a = now_us();
        (void)co_upload_f16(&view, host);
        samples[i] = now_us() - a;
    }
    report("upload_cube16", samples, ITERS);

    /* 5. The submit floor: one minimal EW task through the same seam the plan
     *    runtime uses.  This is the cost a per-head submit cannot go below. */
    co_buffer dst_buf = {0}, id_buf = {0};
    co_view dst, identity;
    co_tensor idt;
    rc = co_buffer_alloc(fd, cube.bytes, &dst_buf);
    if (co_tensor_weight(&idt, cube.physical_cols, cube.physical_cols) == 0)
        rc = rc ? rc : co_buffer_alloc(fd, idt.bytes, &id_buf);
    if (rc == 0 && co_view_init(&dst, &dst_buf, 0, &cube) == 0 &&
        co_view_init(&identity, &id_buf, 0, &idt) == 0 &&
        co_identity_upload(&identity) == 0) {
        co_program prog;
        co_execute_workspace ws;
        if (co_emit_ew(&prog, view, view, identity, dst, 1) == 0 &&
            co_execute_workspace_init(fd, 1, &ws) == 0) {
            co_execute_profile profile;
            for (unsigned i = 0; i < WARMUP; i++)
                (void)co_execute_profiled(fd, &prog, 1, CO_SEPARATE, &ws,
                                          &profile);
            unsigned n = 0;
            for (unsigned i = 0; i < ITERS; i++) {
                double a = now_us();
                int r = co_execute_profiled(fd, &prog, 1, CO_SEPARATE, &ws,
                                            &profile);
                samples[n] = now_us() - a;
                if (r) break;
                n++;
            }
            report("submit_one_task", samples, n);
            co_execute_workspace_destroy(&ws);
        }
    }
    co_buffer_free(&dst_buf);
    co_buffer_free(&id_buf);

    co_buffer_free(&buf);
    rocket_close(fd);
    printf("{\"probe\":\"plan_envelope_primitives\",\"status\":\"ok\"}\n");
    return 0;
}
