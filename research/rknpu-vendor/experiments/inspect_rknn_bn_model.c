/*
 * Inspect FlatBuffer tensor blobs for BatchNormalization inputs using the
 * same parser as openrknn. This does not call rknn_run or submit NPU work.
 * Build inside openrknn/ and link against its librknn_api.so.
 */
#define _GNU_SOURCE
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openrknn.h"

static const uint8_t *g_fb;
static const char *g_dump_dir;

static void inspect_tensor(struct orknn_model *m, uint32_t tidx)
{
    if (tidx >= m->tensor_count) {
        printf("    tensor[%u]: out of range\n", tidx);
        return;
    }

    uint32_t blob = m->tensor_weight_blob ? m->tensor_weight_blob[tidx] : UINT32_MAX;
    uint32_t offset = (blob != UINT32_MAX && blob < m->wt_blob_count &&
                       m->wt_blob_offsets) ? m->wt_blob_offsets[blob] : UINT32_MAX;
    uint32_t root = orknn_fb_u32(g_fb, 0);
    uint32_t sg_vec = orknn_fb_field(g_fb, root, 2);
    uint32_t sg = sg_vec ? orknn_fb_vec_at(g_fb, sg_vec, 0) : 0;
    uint32_t tensors = sg ? orknn_fb_field(g_fb, sg, 0) : 0;
    uint32_t table = tensors ? orknn_fb_vec_at(g_fb, tensors, tidx) : 0;
    char name[128] = {0}, dtype[32] = {0};
    uint32_t name_f = table ? orknn_fb_field(g_fb, table, 5) : 0;
    uint32_t dtype_f = table ? orknn_fb_field(g_fb, table, 7) : 0;
    uint32_t shape_f = table ? orknn_fb_field(g_fb, table, 4) : 0;
    uint32_t size_f = table ? orknn_fb_field(g_fb, table, 12) : 0;
    uint32_t blob_f = table ? orknn_fb_field(g_fb, table, 18) : 0;
    if (name_f) orknn_fb_string(g_fb, name_f, name, sizeof(name));
    if (dtype_f) orknn_fb_string(g_fb, dtype_f, dtype, sizeof(dtype));
    printf("    tensor[%u] name=%s dtype=%s size=%u shape=", tidx, name, dtype,
           size_f ? orknn_fb_u32(g_fb, size_f) : 0);
    if (shape_f) {
        uint32_t vec = orknn_fb_follow(g_fb, shape_f);
        uint32_t nd = orknn_fb_u32(g_fb, vec);
        for (uint32_t i = 0; i < nd && i < 8; i++)
            printf("%s%d", i ? "x" : "", orknn_fb_i32(g_fb, vec + 4 + i * 4));
    }
    printf(" weight_blob=%u parsed_blob=%u bo_offset=0x%x\n",
           blob_f ? orknn_fb_u32(g_fb, blob_f) : UINT32_MAX, blob, offset);

    if (blob == UINT32_MAX || blob >= m->wt_blob_count) return;
    const uint8_t *fb = g_fb;
    uint32_t wt_field = (m->version > 5) ? 20 : 4;
    uint32_t wt_fpos = orknn_fb_field(fb, root, wt_field);
    uint32_t entry = wt_fpos ? orknn_fb_vec_at(fb, wt_fpos, blob) : 0;
    uint32_t data_field = entry ? orknn_fb_field(fb, entry, 0) : 0;
    uint32_t data_len = 0;
    const uint8_t *data = data_field ? orknn_fb_bytes(fb, data_field, &data_len) : NULL;
    if (!data) {
        printf("      blob payload not found\n");
        return;
    }
    printf("      blob_len=%u head=", data_len);
    unsigned n = data_len < 64 ? data_len : 64;
    for (unsigned i = 0; i < n; i++) printf("%02x", data[i]);
    putchar('\n');
    if (g_dump_dir) {
        char path[512];
        snprintf(path, sizeof(path), "%s/tensor_%u_blob_%u.bin",
                 g_dump_dir, tidx, blob);
        FILE *out = fopen(path, "wb");
        if (out) {
            size_t written = fwrite(data, 1, data_len, out);
            fclose(out);
            printf("      dumped=%s bytes=%zu\n", path, written);
        } else {
            perror("fopen blob dump");
        }
    }
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s model.rknn [blob_dump_dir]\n", argv[0]);
        return 2;
    }
    if (argc == 3) g_dump_dir = argv[2];
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 1; }
    long file_len = ftell(f);
    if (file_len <= 0 || (unsigned long)file_len > UINT32_MAX) {
        fprintf(stderr, "invalid model length %ld\n", file_len);
        fclose(f);
        return 1;
    }
    rewind(f);
    uint8_t *bytes = malloc((size_t)file_len);
    if (!bytes || fread(bytes, 1, (size_t)file_len, f) != (size_t)file_len) {
        fprintf(stderr, "model read failed\n");
        free(bytes); fclose(f); return 1;
    }
    fclose(f);

    rknn_context handle = 0;
    int ret = rknn_init(&handle, bytes, (uint32_t)file_len, 0, NULL);
    free(bytes);
    printf("rknn_init=%d size=%ld\n", ret, file_len);
    if (ret != RKNN_SUCC) return 1;

    struct orknn_context *ctx = (struct orknn_context *)(uintptr_t)handle;
    struct orknn_model *m = &ctx->model;
    uint32_t fb_off = (m->version > 1) ? 0x40 : 0x18;
    g_fb = m->file_data + fb_off;
    printf("version=%" PRIu64 " model=%s ops=%u tensors=%u wt_blob_count=%u wt_size=%u\n",
           m->version, m->model_name ? m->model_name : "?", m->op_count,
           m->tensor_count, m->wt_blob_count, m->wt_size);
    printf("pointers ops=%p tensor_weight_blob=%p wt_blob_offsets=%p file_data=%p\n",
           (void *)m->ops, (void *)m->tensor_weight_blob,
           (void *)m->wt_blob_offsets, (void *)m->file_data);
    if (!m->ops || !m->tensor_weight_blob || !m->wt_blob_offsets || !m->file_data)
        return 1;

    for (uint32_t i = 0; i < m->op_count; i++) {
        struct orknn_op_info *op = &m->ops[i];
        if (strncmp(op->type, "BatchNormalization", 18) != 0 && i != 8) continue;
        printf("op[%u] type=%s inputs=%u outputs=%u\n", i, op->type,
               op->input_count, op->output_count);
        for (uint32_t j = 0; j < op->input_count; j++)
            inspect_tensor(m, op->input_tensors[j]);
    }

    return 0;
}
