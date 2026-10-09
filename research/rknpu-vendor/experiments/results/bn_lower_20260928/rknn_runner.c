#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include "rknn_api.h"

static void *read_file(const char *path, size_t *sz) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END); long l = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc(l);
    if (fread(b, 1, l, f) != (size_t)l) { fclose(f); free(b); return NULL; }
    fclose(f); *sz = l; return b;
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
static const char *qnt_str(rknn_tensor_qnt_type q) {
    switch (q) {
    case RKNN_TENSOR_QNT_NONE: return "NONE";
    case RKNN_TENSOR_QNT_DFP: return "DFP";
    case RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC: return "AFF";
    default: return "?";
    }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.rknn input.bin out.bin\n", argv[0]); return 1; }
    size_t msz, isz;
    void *mbuf = read_file(argv[1], &msz);
    void *ibuf = read_file(argv[2], &isz);
    if (!mbuf || !ibuf) return 2;

    rknn_context ctx;
    int ret = rknn_init(&ctx, mbuf, msz, 0, NULL);
    printf("rknn_init ret=%d\n", ret);
    if (ret < 0) return 3;

    rknn_sdk_version ver;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver));
    printf("sdk=%s api=%s\n", ver.api_version, ver.drv_version);

    rknn_input_output_num io_num;
    io_num.n_input = 0; io_num.n_output = 0;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    printf("io: in=%u out=%u\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr ia[4], oa[4];
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        memset(&ia[i], 0, sizeof(ia[i])); ia[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &ia[i], sizeof(ia[i]));
        printf("in[%u]: name=%s dims=[", i, ia[i].name);
        for (uint32_t d = 0; d < ia[i].n_dims; d++) printf("%u%s", ia[i].dims[d], d + 1 < ia[i].n_dims ? "," : "");
        printf("] n_elems=%u size=%u fmt=%s type=%s qnt=%s scale=%.9g zp=%d fl=%d\n",
               ia[i].n_elems, ia[i].size, fmt_str(ia[i].fmt), type_str(ia[i].type),
               qnt_str(ia[i].qnt_type), ia[i].scale, ia[i].zp, ia[i].fl);
    }
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        memset(&oa[i], 0, sizeof(oa[i])); oa[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &oa[i], sizeof(oa[i]));
        printf("out[%u]: name=%s dims=[", i, oa[i].name);
        for (uint32_t d = 0; d < oa[i].n_dims; d++) printf("%u%s", oa[i].dims[d], d + 1 < oa[i].n_dims ? "," : "");
        printf("] n_elems=%u size=%u fmt=%s type=%s qnt=%s scale=%.9g zp=%d fl=%d\n",
               oa[i].n_elems, oa[i].size, fmt_str(oa[i].fmt), type_str(oa[i].type),
               qnt_str(oa[i].qnt_type), oa[i].scale, oa[i].zp, oa[i].fl);
    }

    if (isz < (size_t)ia[0].size) { fprintf(stderr, "input too small\n"); return 4; }

    rknn_input in[1];
    memset(in, 0, sizeof(in));
    in[0].index = 0;
    in[0].buf = ibuf;
    in[0].size = ia[0].n_elems * sizeof(float);
    in[0].pass_through = 0;
    in[0].type = RKNN_TENSOR_FLOAT32;
    in[0].fmt = ia[0].fmt;
    ret = rknn_inputs_set(ctx, 1, in);
    printf("inputs_set ret=%d\n", ret);

    ret = rknn_run(ctx, NULL);
    printf("run ret=%d\n", ret);

    rknn_output out[4];
    memset(out, 0, sizeof(out));
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        out[i].want_float = 1;
        out[i].index = i;
    }
    ret = rknn_outputs_get(ctx, io_num.n_output, out, NULL);
    printf("outputs_get ret=%d\n", ret);

    FILE *fo = fopen(argv[3], "wb");
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        fwrite(out[i].buf, 1, oa[i].n_elems * sizeof(float), fo);
        printf("out[%u] first8: ", i);
        float *f = (float *)out[i].buf;
        for (int k = 0; k < 8; k++) printf("%.6g ", f[k]);
        printf("\n");
    }
    fclose(fo);
    rknn_outputs_release(ctx, io_num.n_output, out);
    rknn_destroy(ctx);
    return 0;
}
