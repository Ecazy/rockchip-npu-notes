// SPDX-License-Identifier: GPL-3.0-or-later
/* Bounded validity experiment for the B2 candidate: N *independent* matmuls in
 * one chain versus N separate submits.
 *
 * The optimisation plan allows a chained group only when a legal recipe and board
 * evidence exist, because "tasks are independent and the flags match" is not a
 * proof.  This probe supplies the device-side evidence, and is deliberately
 * written so that a chain doing *nothing* cannot look correct:
 *
 *   - every task owns its input, weight and output BO, exactly like the
 *     per-(batch,head) QK work;
 *   - the reference pass and the chain pass each start from the **same FP16
 *     sentinel** written into every output BO, so a skipped task, an unwritten
 *     output or a chain that only ran its first task is detected;
 *   - after the chain every output must (a) differ from the sentinel, (b) match
 *     its own separate-submit reference byte for byte, (c) be finite, and (d) pass
 *     the same slack/guard check the runtime applies before publishing;
 *   - a negative control corrupts one reference byte and requires the comparator
 *     to notice, so a broken comparison cannot pass;
 *   - timing uses its own loop with no sentinel traffic, and any failed stage
 *     makes the process exit non-zero (a campaign must not record a failure as
 *     evidence).  Device faults are reported, never retried.
 *
 * It is a measurement tool: it adds no recipe, no plan and no runtime path.
 */
#define _POSIX_C_SOURCE 200809L
#include "custom_ops.h"
#include "rk_task_ref.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_TASKS 64

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

typedef struct {
    co_buffer a, w, y;
    co_view va, vw, vy;
    co_program prog;
} task;

static int build(task *t, int fd, const co_tensor *at, const co_tensor *wt,
                 const co_tensor *yt, unsigned seed) {
    memset(t, 0, sizeof(*t));
    _Float16 *a_host = malloc((size_t)at->rows * at->cols * 2);
    _Float16 *w_host = malloc((size_t)wt->rows * wt->cols * 2);
    if (!a_host || !w_host) {
        free(a_host);
        free(w_host);
        return 1;
    }
    for (unsigned r = 0; r < at->rows; r++)
        for (unsigned c = 0; c < at->cols; c++)
            a_host[r * at->cols + c] =
                (_Float16)(((int)((r * 7u + c * 3u + seed) % 23u) - 11) / 16.0);
    for (unsigned r = 0; r < wt->rows; r++)
        for (unsigned c = 0; c < wt->cols; c++)
            w_host[r * wt->cols + c] =
                (_Float16)(((int)((r * 5u + c * 11u + seed) % 19u) - 9) / 16.0);
    int rc = co_buffer_alloc(fd, at->bytes, &t->a);
    if (!rc) rc = co_buffer_alloc(fd, wt->bytes, &t->w);
    if (!rc) rc = co_buffer_alloc(fd, yt->bytes, &t->y);
    if (!rc) rc = co_view_init(&t->va, &t->a, 0, at);
    if (!rc) rc = co_view_init(&t->vw, &t->w, 0, wt);
    if (!rc) rc = co_view_init(&t->vy, &t->y, 0, yt);
    if (!rc) rc = co_upload_f16(&t->va, a_host);
    if (!rc) rc = co_upload_f16(&t->vw, w_host);
    if (!rc) rc = co_emit_mm(&t->prog, t->va, t->vw, t->vy);
    free(a_host);
    free(w_host);
    return rc;
}

static void release(task *t) {
    co_buffer_free(&t->a);
    co_buffer_free(&t->w);
    co_buffer_free(&t->y);
}

static int all_finite_f16(const unsigned char *bytes, size_t n) {
    const uint16_t *h = (const uint16_t *)bytes;
    for (size_t i = 0; i < n / 2; i++)
        if ((h[i] & 0x7c00u) == 0x7c00u) return 0;
    return 1;
}

/* Second shape: all tasks share three BOs (A, B, Y) and address their own tile
 * by byte offset, which is what a static-repeat-expanded plan looks like.  It has
 * a single output BO, so one completion fence covers every tile -- the per-task-BO
 * shape above cannot rely on that, because a chained job's fence is per output BO. */
static int run_shared_shape(int fd, unsigned n, unsigned repeats,
                            int shared_inputs) {
    co_tensor at, wt, yt;
    if (co_tensor_matrix(&at, 16, 16, CO_F16) ||
        co_tensor_weight(&wt, 16, 16) ||
        co_tensor_matrix(&yt, 16, 16, CO_F16))
        return 1;
    const size_t a_tile = at.bytes, w_tile = wt.bytes, y_tile = yt.bytes;
    const size_t y_bytes = (size_t)yt.rows * yt.cols * 2;
    co_buffer ab = {0}, wb = {0}, yb = {0};
    int rc = co_buffer_alloc(fd, a_tile * n, &ab);
    if (!rc) rc = co_buffer_alloc(fd, w_tile * n, &wb);
    if (!rc) rc = co_buffer_alloc(fd, y_tile * n, &yb);
    if (rc) {
        co_buffer_free(&ab); co_buffer_free(&wb); co_buffer_free(&yb);
        return 1;
    }
    co_view *va = calloc(n, sizeof(*va)), *vw = calloc(n, sizeof(*vw)),
            *vy = calloc(n, sizeof(*vy));
    co_program *progs = calloc(n, sizeof(*progs));
    unsigned char *sent = malloc(y_bytes), *ref = malloc((size_t)n * y_bytes),
                  *got = malloc(y_bytes);
    if (!va || !vw || !vy || !progs || !sent || !ref || !got) {
        free(va); free(vw); free(vy); free(progs);
        free(sent); free(ref); free(got);
        co_buffer_free(&ab); co_buffer_free(&wb); co_buffer_free(&yb);
        return 1;
    }
    for (unsigned i = 0; i < n; i++) {
        if ((rc = co_view_init(&va[i], &ab, (size_t)i * a_tile, &at)) ||
            (rc = co_view_init(&vw[i], &wb, (size_t)i * w_tile, &wt)) ||
            (rc = co_view_init(&vy[i], &yb, (size_t)i * y_tile, &yt)))
            break;
    }
    _Float16 *a_host = malloc((size_t)at.rows * at.cols * 2);
    _Float16 *w_host = malloc((size_t)wt.rows * wt.cols * 2);
    if (!rc && (!a_host || !w_host)) rc = 1;
    for (unsigned i = 0; i < n && !rc; i++) {
        for (unsigned r = 0; r < at.rows; r++)
            for (unsigned c = 0; c < at.cols; c++)
                a_host[r * at.cols + c] =
                    (_Float16)(((int)((r * 7u + c * 3u + i * 13u) % 23u) - 11) / 16.0);
        for (unsigned r = 0; r < wt.rows; r++)
            for (unsigned c = 0; c < wt.cols; c++)
                w_host[r * wt.cols + c] =
                    (_Float16)(((int)((r * 5u + c * 11u + i * 13u) % 19u) - 9) / 16.0);
        if (shared_inputs && i > 0) {
            if ((rc = co_emit_mm(&progs[i], va[0], vw[0], vy[i]))) break;
            continue;
        }
        if ((rc = co_upload_f16(&va[i], a_host)) ||
            (rc = co_upload_f16(&vw[i], w_host)) ||
            (rc = co_emit_mm(&progs[i], va[i], vw[i], vy[i])))
            break;
    }
    free(a_host);
    free(w_host);
    const uint16_t sentinel = 0x7e5au;
    for (size_t q = 0; q < y_bytes / 2; q++) ((uint16_t *)sent)[q] = sentinel;

    unsigned mismatched = 0, untouched = 0, nonfinite = 0, mismatched_late = 0;
    int stage = 0;
    if (!rc) {
        for (unsigned i = 0; i < n && !stage; i++)
            if (co_upload_f16(&vy[i], (_Float16 *)sent)) stage = 1;
        for (unsigned i = 0; i < n && !stage; i++)
            if (co_execute(fd, progs + i, 1, CO_SEPARATE)) stage = 2;
        for (unsigned i = 0; i < n && !stage; i++) {
            unsigned char *refi = (unsigned char *)rk_task_ref_slice(ref, y_bytes, i);
            if (co_download(&vy[i], refi)) stage = 3;
            else if (memcmp(refi, sent, y_bytes) == 0) stage = 4;
        }
        for (unsigned i = 0; i < n && !stage; i++)
            if (co_upload_f16(&vy[i], (_Float16 *)sent)) stage = 5;
        if (!stage && co_execute(fd, progs, n, CO_CHAIN)) stage = 6;
        /* First read: immediately after the chained submit returns. */
        for (unsigned i = 0; i < n && !stage; i++) {
            if (co_download(&vy[i], got)) { stage = 7; break; }
            if (memcmp(got, sent, y_bytes) == 0) untouched++;
            if (!rk_task_ref_equal(got, ref, y_bytes, i)) mismatched++;
            if (!all_finite_f16(got, y_bytes)) nonfinite++;
        }
        /* Second read after a delay: if this matches, the failure is a missing
         * completion point (race), not a wrong computation. */
        if (!stage) {
            {
                struct timespec ts = {.tv_sec = 0, .tv_nsec = 20000000L};
                nanosleep(&ts, NULL);
            }
            for (unsigned i = 0; i < n && !stage; i++) {
                if (co_download(&vy[i], got)) { stage = 7; break; }
                if (!rk_task_ref_equal(got, ref, y_bytes, i)) mismatched_late++;
            }
        }
        if (!stage && co_guard_check(&yb)) stage = 8;
    }
    double t_sep = 0.0, t_chain = 0.0;
    if (!stage) {
        for (unsigned rep = 0; rep < repeats; rep++) {
            double a = now_us();
            for (unsigned i = 0; i < n; i++)
                if (co_execute(fd, progs + i, 1, CO_SEPARATE)) { stage = 9; break; }
            t_sep += now_us() - a;
            if (stage) break;
        }
        for (unsigned rep = 0; rep < repeats && !stage; rep++) {
            double a = now_us();
            if (co_execute(fd, progs, n, CO_CHAIN)) stage = 10;
            t_chain += now_us() - a;
        }
    }
    int control_ok = 1;
    if (!stage) {
        unsigned char *ref0 = (unsigned char *)rk_task_ref_slice(ref, y_bytes, 0);
        unsigned char saved = ref0[0];
        memcpy(got, ref0, y_bytes);
        ref0[0] = (unsigned char)(saved ^ 0xffu);
        control_ok = !rk_task_ref_equal(got, ref, y_bytes, 0);
        ref0[0] = saved;
    }
    const int ok = (!stage && mismatched == 0 && untouched == 0 && nonfinite == 0 &&
                    control_ok);
    printf("{\"probe\":\"independent_mm_chain\",\"shape\":\"%s\","
           "\"tasks\":%u,\"status\":\"%s\",\"stage\":%d,"
           "\"mismatched_tasks\":%u,\"untouched_tasks\":%u,"
           "\"nonfinite_tasks\":%u,\"mismatched_after_20ms\":%u,"
           "\"separate_us\":%.3f,\"chain_us\":%.3f,"
           "\"separate_per_task_us\":%.3f,\"chain_per_task_us\":%.3f,"
           "\"speedup\":%.4f}\n",
           shared_inputs ? "shared_bos_shared_inputs" : "shared_bos",
           n, ok ? "ok" : "failed", stage, mismatched, untouched, nonfinite,
           mismatched_late,
           t_sep / repeats, t_chain / repeats,
           t_sep / repeats / n, t_chain / repeats / n,
           t_chain > 0.0 ? t_sep / t_chain : 0.0);
    free(va); free(vw); free(vy); free(progs);
    free(sent); free(ref); free(got);
    co_buffer_free(&ab); co_buffer_free(&wb); co_buffer_free(&yb);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    unsigned counts[] = {2, 4, 8, 16, 32};
    unsigned ncounts = sizeof(counts) / sizeof(counts[0]);
    unsigned repeats = 20;
    if (argc > 1) {
        counts[0] = (unsigned)strtoul(argv[1], NULL, 10);
        ncounts = 1;
    }
    if (argc > 2) repeats = (unsigned)strtoul(argv[2], NULL, 10);
    if (repeats < 1) repeats = 1;
    const int shared_only = (argc > 3 &&
                             (strcmp(argv[3], "shared") == 0 ||
                              strcmp(argv[3], "sharedin") == 0));
    const int shared_inputs = (argc > 3 && strcmp(argv[3], "sharedin") == 0);

    int fd = rocket_open();
    if (fd < 0) {
        printf("{\"probe\":\"independent_mm_chain\",\"status\":\"unsupported\"}\n");
        return 77;
    }
    co_tensor at, wt, yt;
    if (co_tensor_matrix(&at, 16, 16, CO_F16) ||
        co_tensor_weight(&wt, 16, 16) ||
        co_tensor_matrix(&yt, 16, 16, CO_F16)) {
        printf("{\"probe\":\"independent_mm_chain\",\"status\":\"geometry_failed\"}\n");
        rocket_close(fd);
        return 1;
    }
    const size_t y_bytes = (size_t)yt.rows * yt.cols * 2;

    task tasks[MAX_TASKS];
    memset(tasks, 0, sizeof(tasks));   /* a failed build must not free garbage */
    unsigned built = 0;
    for (unsigned i = 0; i < MAX_TASKS; i++) {
        int rc = build(&tasks[i], fd, &at, &wt, &yt, i * 13u);
        if (rc) {
            printf("{\"probe\":\"independent_mm_chain\",\"status\":"
                   "\"setup_failed\",\"native\":%d,\"built\":%u}\n", rc, built);
            for (unsigned k = 0; k < built; k++) release(&tasks[k]);
            rocket_close(fd);
            return 1;
        }
        built++;
    }

    const uint16_t sentinel = 0x7e5au;   /* finite, implausible, non-zero FP16 */
    unsigned char *sent = malloc(y_bytes);
    /* One reference slice per task: a shared buffer would keep only the last
     * task's result and make every other task look broken. */
    unsigned char *ref = malloc((size_t)MAX_TASKS * y_bytes);
    unsigned char *got = malloc(y_bytes);
    co_program *progs = malloc(sizeof(co_program) * MAX_TASKS);
    if (!sent || !ref || !got || !progs) {
        printf("{\"probe\":\"independent_mm_chain\",\"status\":\"nomem\"}\n");
        free(sent); free(ref); free(got); free(progs);
        for (unsigned k = 0; k < built; k++) release(&tasks[k]);
        rocket_close(fd);
        return 1;
    }
    for (size_t q = 0; q < y_bytes / 2; q++) ((uint16_t *)sent)[q] = sentinel;

    printf("{\"probe\":\"independent_mm_chain\",\"geometry\":"
           "{\"m\":16,\"k\":16,\"n\":16},\"repeats\":%u,\"sentinel\":%u}\n",
           repeats, (unsigned)sentinel);

    int ok_all = 1;
    if (shared_only) {
        for (unsigned ci = 0; ci < ncounts && ok_all; ci++) {
            const unsigned n = counts[ci];
            if (n < 2 || n > MAX_TASKS) continue;
            if (run_shared_shape(fd, n, repeats, shared_inputs)) ok_all = 0;
        }
        free(sent); free(ref); free(got); free(progs);
        for (unsigned k = 0; k < built; k++) release(&tasks[k]);
        rocket_close(fd);
        printf("{\"probe\":\"independent_mm_chain\",\"status\":\"%s\"}\n",
               ok_all ? "done" : "incomplete");
        return ok_all ? 0 : 1;
    }
    for (unsigned ci = 0; ci < ncounts && ok_all; ci++) {
        const unsigned n = counts[ci];
        if (n < 2 || n > MAX_TASKS) continue;
        for (unsigned i = 0; i < n; i++) progs[i] = tasks[i].prog;

        /* ---- correctness: reference pass, N separate submits ------------- */
        int stage = 0;
        for (unsigned i = 0; i < n && !stage; i++)
            if (co_upload_f16(&tasks[i].vy, (_Float16 *)sent)) stage = 1;
        for (unsigned i = 0; i < n && !stage; i++)
            if (co_execute(fd, progs + i, 1, CO_SEPARATE)) stage = 2;
        for (unsigned i = 0; i < n && !stage; i++) {
            unsigned char *refi = (unsigned char *)rk_task_ref_slice(ref, y_bytes, i);
            if (co_download(&tasks[i].vy, refi)) stage = 3;
            else if (memcmp(refi, sent, y_bytes) == 0) stage = 4;  /* not written */
            else if (!all_finite_f16(refi, y_bytes)) stage = 5;
        }
        if (stage) {
            printf("{\"probe\":\"independent_mm_chain\",\"tasks\":%u,"
                   "\"status\":\"reference_failed\",\"stage\":%d}\n", n, stage);
            ok_all = 0;
            break;
        }

        /* ---- correctness: one chained submit ---------------------------- */
        for (unsigned i = 0; i < n && !stage; i++)
            if (co_upload_f16(&tasks[i].vy, (_Float16 *)sent)) stage = 6;
        if (!stage && co_execute(fd, progs, n, CO_CHAIN)) stage = 7;
        unsigned mismatched = 0, untouched = 0, nonfinite = 0, guard_bad = 0;
        for (unsigned i = 0; i < n && !stage; i++) {
            if (co_download(&tasks[i].vy, got)) { stage = 8; break; }
            if (memcmp(got, sent, y_bytes) == 0) untouched++;
            if (!rk_task_ref_equal(got, ref, y_bytes, i)) mismatched++;
            if (!all_finite_f16(got, y_bytes)) nonfinite++;
            if (co_guard_check(&tasks[i].y)) guard_bad++;
        }
        if (stage) {
            printf("{\"probe\":\"independent_mm_chain\",\"tasks\":%u,"
                   "\"status\":\"chain_failed\",\"stage\":%d}\n", n, stage);
            ok_all = 0;
            break;
        }
        /* Negative control: the comparator must reject one flipped byte. */
        unsigned char *ref0 = (unsigned char *)rk_task_ref_slice(ref, y_bytes, 0);
        unsigned char saved = ref0[0];
        memcpy(got, ref0, y_bytes);          /* a copy of task 0's correct result */
        ref0[0] = (unsigned char)(saved ^ 0xffu);
        int comparator_works = !rk_task_ref_equal(got, ref, y_bytes, 0);
        ref0[0] = saved;
        if (!comparator_works) {
            printf("{\"probe\":\"independent_mm_chain\",\"tasks\":%u,"
                   "\"status\":\"self_check_failed\"}\n", n);
            ok_all = 0;
            break;
        }

        /* ---- timing: separate vs chain, no sentinel traffic ------------- */
        double t_sep = 0.0, t_chain = 0.0;
        for (unsigned rep = 0; rep < repeats; rep++) {
            double a = now_us();
            for (unsigned i = 0; i < n; i++)
                if (co_execute(fd, progs + i, 1, CO_SEPARATE)) { stage = 9; break; }
            t_sep += now_us() - a;
            if (stage) break;
        }
        for (unsigned rep = 0; rep < repeats && !stage; rep++) {
            double a = now_us();
            if (co_execute(fd, progs, n, CO_CHAIN)) stage = 10;
            t_chain += now_us() - a;
        }
        if (stage) {
            printf("{\"probe\":\"independent_mm_chain\",\"tasks\":%u,"
                   "\"status\":\"timing_failed\",\"stage\":%d}\n", n, stage);
            ok_all = 0;
            break;
        }

        const int ok = (mismatched == 0 && untouched == 0 && nonfinite == 0 &&
                        guard_bad == 0);
        printf("{\"probe\":\"independent_mm_chain\",\"tasks\":%u,"
               "\"status\":\"%s\",\"bitwise_equal\":%s,"
               "\"mismatched_tasks\":%u,\"untouched_tasks\":%u,"
               "\"nonfinite_tasks\":%u,\"guard_failed_tasks\":%u,"
               "\"separate_us\":%.3f,\"chain_us\":%.3f,"
               "\"separate_per_task_us\":%.3f,\"chain_per_task_us\":%.3f,"
               "\"speedup\":%.4f}\n",
               n, ok ? "ok" : "mismatch", ok ? "true" : "false",
               mismatched, untouched, nonfinite, guard_bad,
               t_sep / repeats, t_chain / repeats,
               t_sep / repeats / n, t_chain / repeats / n,
               t_chain > 0.0 ? t_sep / t_chain : 0.0);
        if (!ok) ok_all = 0;
    }

    free(sent);
    free(ref);
    free(got);
    free(progs);
    for (unsigned k = 0; k < built; k++) release(&tasks[k]);
    rocket_close(fd);
    printf("{\"probe\":\"independent_mm_chain\",\"status\":\"%s\"}\n",
           ok_all ? "done" : "incomplete");
    return ok_all ? 0 : 1;
}
