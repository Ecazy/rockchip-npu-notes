/* gdn_feed.c — run the converted Qwen3.5 GDN layer once and dump outputs.
 *
 * The probe model has one [1,196,4096] activation input and two outputs
 * (y [1,196,4096], final_state [32,128,128]).  Inputs are fed as FLOAT32 with
 * pass_through=0 so the runtime applies the model's own quantisation, matching
 * how attn_feed.c feeds the SDPA probe.
 *
 * build (board):
 *   gcc -O2 gdn_feed.c -I<rknn include> -L<rknn lib> -lrknnrt -o gdn_feed
 * usage:
 *   ./gdn_feed model.rknn out_prefix in0_f32.bin [in1_f32.bin ...]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rknn_api.h"

static void *read_file(const char *path, size_t *sz) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END); long l = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc((size_t)l);
    if (!b) { fclose(f); return NULL; }
    if (fread(b, 1, (size_t)l, f) != (size_t)l) { fclose(f); free(b); return NULL; }
    fclose(f); *sz = (size_t)l; return b;
}

static int write_file(const char *path, const void *buf, size_t sz) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    size_t n = fwrite(buf, 1, sz, f);
    fclose(f);
    return n == sz ? 0 : -1;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static const char *type_str(rknn_tensor_type t) {
    switch (t) {
    case RKNN_TENSOR_FLOAT32: return "F32";
    case RKNN_TENSOR_FLOAT16: return "F16";
    case RKNN_TENSOR_INT8: return "I8";
    case RKNN_TENSOR_UINT8: return "U8";
    case RKNN_TENSOR_INT16: return "I16";
    default: return "?";
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.rknn out_prefix in0_f32.bin [in1_f32.bin ...]\n", argv[0]);
        return 1;
    }
    const char *model = argv[1], *prefix = argv[2];

    size_t msz = 0;
    void *mbuf = read_file(model, &msz);
    if (!mbuf) return 2;

    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, mbuf, msz, 0, NULL);
    printf("rknn_init ret=%d\n", ret);
    if (ret < 0) return 3;

    rknn_sdk_version ver;
    if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == 0)
        printf("sdk=%s api=%s\n", ver.api_version, ver.drv_version);

    rknn_input_output_num io;
    if ((ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io))) < 0) {
        fprintf(stderr, "query in_out_num ret=%d\n", ret);
        return 4;
    }
    printf("io: in=%u out=%u\n", io.n_input, io.n_output);

    rknn_tensor_attr ia[8];
    for (uint32_t i = 0; i < io.n_input; i++) {
        memset(&ia[i], 0, sizeof(ia[i]));
        ia[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &ia[i], sizeof(ia[i]));
        printf("in[%u]: name=%s n_dims=%u dims=[", i, ia[i].name, ia[i].n_dims);
        for (uint32_t d = 0; d < ia[i].n_dims; d++)
            printf("%u%s", ia[i].dims[d], d + 1 < ia[i].n_dims ? "," : "");
        printf("] n_elems=%u type=%s fmt=%d scale=%g zp=%d\n",
               ia[i].n_elems, type_str(ia[i].type), ia[i].fmt, ia[i].scale, ia[i].zp);
    }
    if (io.n_input != (uint32_t)(argc - 3)) {
        fprintf(stderr, "model has %u inputs, %d files given\n", io.n_input, argc - 3);
        return 4;
    }
    void *ibuf[8];
    rknn_input inputs[8];
    memset(inputs, 0, sizeof(inputs));
    memset(ibuf, 0, sizeof(ibuf));
    for (uint32_t i = 0; i < io.n_input; i++) {
        size_t isz = 0;
        ibuf[i] = read_file(argv[3 + i], &isz);
        if (!ibuf[i]) return 2;
        size_t need = (size_t)ia[i].n_elems * sizeof(float);
        if (isz != need) {
            fprintf(stderr, "input %u size %zu != expected %zu\n", i, isz, need);
            return 5;
        }
        inputs[i].index = i;
        inputs[i].type = RKNN_TENSOR_FLOAT32;
        inputs[i].fmt = RKNN_TENSOR_NHWC;
        inputs[i].size = isz;
        inputs[i].buf = ibuf[i];
        inputs[i].pass_through = 0;
    }

    double t_set0 = now_ms();
    ret = rknn_inputs_set(ctx, io.n_input, inputs);
    printf("inputs_set ret=%d (%.3f ms)\n", ret, now_ms() - t_set0);
    if (ret < 0) return 6;

    const char *reps_env = getenv("GDN_REPEAT");
    int reps = (reps_env != NULL) ? atoi(reps_env) : 1;
    if (reps < 1) reps = 1;
    double best = 1e18, total = 0.0;
    for (int it = 0; it < reps; it++) {
        double t0 = now_ms();
        ret = rknn_run(ctx, NULL);
        double t1 = now_ms();
        if (ret < 0) {
            printf("run[%d] ret=%d\n", it, ret);
            return 7;
        }
        double dt = t1 - t0;
        total += dt;
        if (dt < best) best = dt;
        if (reps <= 5 || it == 0 || it == reps - 1)
            printf("run[%d] wall_ms=%.3f\n", it, dt);
    }
    printf("run_summary reps=%d min_ms=%.3f mean_ms=%.3f\n", reps, best, total / reps);

    if (io.n_output > 32) {
        fprintf(stderr, "too many outputs: %u\n", io.n_output);
        return 8;
    }
    rknn_tensor_attr oa[32];
    rknn_output outs[32];
    memset(outs, 0, sizeof(outs));
    for (uint32_t i = 0; i < io.n_output; i++) {
        memset(&oa[i], 0, sizeof(oa[i]));
        oa[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &oa[i], sizeof(oa[i]));
        outs[i].index = i;
        outs[i].want_float = 1;
    }
    double t_get0 = now_ms();
    ret = rknn_outputs_get(ctx, io.n_output, outs, NULL);
    printf("outputs_get ret=%d (%.3f ms)\n", ret, now_ms() - t_get0);
    if (ret < 0) return 8;

    for (uint32_t i = 0; i < io.n_output; i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s_out%02u.bin", prefix, i);
        size_t osz = (size_t)oa[i].n_elems * sizeof(float);
        if (write_file(path, outs[i].buf, osz) != 0) return 9;
        printf("out[%u]: name=%s n_elems=%u type=%s fmt=%d -> %s (%zu B)\n",
               i, oa[i].name, oa[i].n_elems, type_str(oa[i].type), oa[i].fmt, path, osz);
    }
    rknn_outputs_release(ctx, io.n_output, outs);

    rknn_destroy(ctx);
    free(mbuf);
    for (uint32_t i = 0; i < io.n_input; i++) free(ibuf[i]);
    return 0;
}
