/* attn_feed.c — §7.189 Task 3 Step 2: multi-input attention feeding.
 *
 * Root cause of "inputs_set failed: -5" (rknn_infer_dump.c): it derived the
 * element count from dims[n_dims-1] (last dim only) and only set input[0].
 * Fix: use attr.n_elems for each input, set ALL inputs (Q/K/V), FLOAT32
 * pass_through=0 (API quantizes to the model's int8/fp16 dtype).
 *
 * build (board):
 *   gcc -O2 attn_feed.c -I<rknn include> -L<rknn lib> -lrknnrt -o attn_feed
 * usage:
 *   ./attn_feed model.rknn <ramp|ones|zero|rand> [out.bin]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_api.h"

static void *read_file(const char *path, size_t *sz) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END); long l = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc((size_t)l);
    if (fread(b, 1, (size_t)l, f) != (size_t)l) { fclose(f); free(b); return NULL; }
    fclose(f); *sz = (size_t)l; return b;
}

static const char *fmt_str(rknn_tensor_format f) {
    switch (f) {
    case RKNN_TENSOR_NCHW: return "NCHW";
    case RKNN_TENSOR_NHWC: return "NHWC";
    default: return "?";
    }
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
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.rknn <ramp|ones|zero|rand> [out.bin]\n", argv[0]);
        return 1;
    }
    const char *model = argv[1], *mode = argv[2];
    const char *outpath = (argc >= 4) ? argv[3] : "attn_out.bin";

    size_t msz;
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
        fprintf(stderr, "query in_out_num ret=%d\n", ret); return 4;
    }
    printf("io: in=%u out=%u\n", io.n_input, io.n_output);
    if (io.n_input > 8 || io.n_output > 8) { fprintf(stderr, "too many io\n"); return 4; }

    rknn_tensor_attr ia[8], oa[8];
    for (uint32_t i = 0; i < io.n_input; i++) {
        memset(&ia[i], 0, sizeof(ia[i])); ia[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &ia[i], sizeof(ia[i]));
        printf("in[%u]: name=%s dims=[", i, ia[i].name);
        for (uint32_t d = 0; d < ia[i].n_dims; d++)
            printf("%u%s", ia[i].dims[d], d + 1 < ia[i].n_dims ? "," : "");
        printf("] n_elems=%u size=%u fmt=%s type=%s scale=%.9g zp=%d\n",
               ia[i].n_elems, ia[i].size, fmt_str(ia[i].fmt), type_str(ia[i].type),
               ia[i].scale, ia[i].zp);
    }
    for (uint32_t i = 0; i < io.n_output; i++) {
        memset(&oa[i], 0, sizeof(oa[i])); oa[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &oa[i], sizeof(oa[i]));
        printf("out[%u]: name=%s dims=[", i, oa[i].name);
        for (uint32_t d = 0; d < oa[i].n_dims; d++)
            printf("%u%s", oa[i].dims[d], d + 1 < oa[i].n_dims ? "," : "");
        printf("] n_elems=%u size=%u fmt=%s type=%s scale=%.9g zp=%d\n",
               oa[i].n_elems, oa[i].size, fmt_str(oa[i].fmt), type_str(oa[i].type),
               oa[i].scale, oa[i].zp);
    }

    rknn_input in[8];
    memset(in, 0, sizeof(in));
    for (uint32_t i = 0; i < io.n_input; i++) {
        uint32_t ne = ia[i].n_elems;
        float *buf = malloc((size_t)ne * sizeof(float));
        uint32_t rng = 0x12345678u + (uint32_t)(i + 1) * 0x9e3779b9u;
        for (uint32_t k = 0; k < ne; k++) {
            if (!strcmp(mode, "ones"))      buf[k] = 1.0f;
            else if (!strcmp(mode, "zero")) buf[k] = 0.0f;
            else if (!strcmp(mode, "rand")) {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                buf[k] = (float)(rng & 0xffffffu) / 8388608.0f - 1.0f;
            }
            else                            buf[k] = (float)((k % 16) + 1) * 0.25f;
        }
        in[i].index = i;
        in[i].buf = buf;
        in[i].size = ne * sizeof(float);
        in[i].pass_through = 0;
        in[i].type = RKNN_TENSOR_FLOAT32;
        in[i].fmt = ia[i].fmt;
    }
    ret = rknn_inputs_set(ctx, io.n_input, in);
    printf("inputs_set ret=%d (n=%u)\n", ret, io.n_input);
    if (ret < 0) return 5;

    ret = rknn_run(ctx, NULL);
    printf("run ret=%d\n", ret);
    if (ret < 0) return 6;

    rknn_output out[8];
    memset(out, 0, sizeof(out));
    for (uint32_t i = 0; i < io.n_output; i++) { out[i].want_float = 1; out[i].index = i; }
    ret = rknn_outputs_get(ctx, io.n_output, out, NULL);
    printf("outputs_get ret=%d\n", ret);

    FILE *fo = fopen(outpath, "wb");
    for (uint32_t i = 0; i < io.n_output; i++) {
        uint32_t ne = oa[i].n_elems;
        float *f = (float *)out[i].buf;
        if (fo) fwrite(out[i].buf, 1, (size_t)ne * sizeof(float), fo);
        double s = 0; float mn = 1e30f, mx = -1e30f; uint32_t nz = 0;
        for (uint32_t k = 0; k < ne; k++) {
            float v = f[k]; s += (double)v;
            if (v < mn) mn = v; if (v > mx) mx = v;
            if (v != 0.0f) nz++;
        }
        printf("out[%u] n=%u mean=%.6g min=%.6g max=%.6g nonzero=%u first8:",
               i, ne, ne ? s / ne : 0.0, mn, mx, nz);
        for (uint32_t k = 0; k < 8 && k < ne; k++) printf(" %.6g", f[k]);
        printf("\n");
    }
    if (fo) fclose(fo);
    printf("wrote %s\n", outpath);

    rknn_outputs_release(ctx, io.n_output, out);
    rknn_destroy(ctx);
    return 0;
}
