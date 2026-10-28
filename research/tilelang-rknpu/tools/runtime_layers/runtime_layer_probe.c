// SPDX-License-Identifier: GPL-3.0-or-later
/* A/B/C/D runtime layering harness for one frozen RKPlan bundle.
 *
 * A = Python PlanCall (measured by benchmark_python_layers.py)
 * B = native C RKPlan workspace
 * C = direct custom-ops, same descriptor/emitters/executor, bypassing the
 *     generic RKPlan action interpreter and per-run shape/alias validation;
 *     it retains finite input checks, exact upload packing, cache sync,
 *     submit grouping, full guard coverage, finite output checks and staged
 *     publication.
 * D = resident execution: same emitted programs/submit groups; input updates
 *     and result validation are outside the timed interval.
 *
 * The bundle descriptor is parsed from bundle.bin.  C uses the same
 * rk_emit_action() and co_execute_profiled() seams as the runtime; it does not
 * generate registers or submit commands itself.
 */
#define _POSIX_C_SOURCE 200809L
#include "rk_plan_internal.h"
#include "custom_ops.h"
#include "rk_plan_download.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAGIC "RKPB1\0\0\0"
#define MAX_COALESCED_UPLOADS 256u

typedef struct {
    uint32_t id, rank, shape[2];
    uint64_t nbytes;
} meta_t;

typedef struct {
    uint32_t frames, ninputs, nconstants, noutputs;
    meta_t *inputs, *constants, *outputs;
} case_def;

typedef struct {
    rk_plan_desc desc;
    rk_value_desc *values;
    rk_buffer_desc *buffers;
    rk_view_desc *views;
    rk_action_desc *actions;
    rk_submit_desc *submits;
    rk_publish_desc *publish;
    char **names;
    uint32_t **ain, **aout, **adeps, **stasks, **sdeps;
} bundle;

static void die(const char *msg) {
    fprintf(stderr, "runtime_layer_probe: %s\n", msg);
    exit(2);
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void *read_file(const char *path, size_t *size_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); exit(2); }
    if (fseek(f, 0, SEEK_END)) die("fseek");
    long n = ftell(f);
    if (n < 0) die("ftell");
    rewind(f);
    void *p = malloc((size_t)n ? (size_t)n : 1);
    if (!p) die("malloc");
    if ((size_t)n != fread(p, 1, (size_t)n, f)) die("fread");
    fclose(f);
    *size_out = (size_t)n;
    return p;
}


static uint32_t ru32(FILE *f) {
    uint32_t v;
    if (fread(&v, 1, 4, f) != 4) die("short bundle read");
    return v;
}
static uint64_t ru64(FILE *f) {
    uint64_t v;
    if (fread(&v, 1, 8, f) != 8) die("short bundle read");
    return v;
}
static char *rstr(FILE *f, uint32_t *len_out) {
    uint32_t n = ru32(f);
    char *s = malloc((size_t)n + 1);
    if (!s) die("malloc");
    if (n && fread(s, 1, n, f) != n) die("short bundle string");
    s[n] = 0;
    if (len_out) *len_out = n;
    return s;
}

static void load_bundle(const char *path, bundle *b) {
    memset(b, 0, sizeof(*b));
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); exit(2); }
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "RKPB1", 5))
        die("bad bundle magic");
    uint32_t version = ru32(f);
    if (version != 1) die("bad bundle version");
    uint32_t nv = ru32(f), nb = ru32(f), nw = ru32(f), na = ru32(f);
    uint32_t ns = ru32(f), np = ru32(f), nf = ru32(f);
    (void)nf;
    uint64_t profile_hash = ru64(f), source_hash = ru64(f);
    uint64_t device_budget = ru64(f), host_budget = ru64(f);
    uint64_t claimed_device = ru64(f), claimed_host = ru64(f);
    b->values = calloc(nv ? nv : 1, sizeof(*b->values));
    b->names = calloc(nv ? nv : 1, sizeof(*b->names));
    b->buffers = calloc(nb ? nb : 1, sizeof(*b->buffers));
    b->views = calloc(nw ? nw : 1, sizeof(*b->views));
    b->actions = calloc(na ? na : 1, sizeof(*b->actions));
    b->submits = calloc(ns ? ns : 1, sizeof(*b->submits));
    b->publish = calloc(np ? np : 1, sizeof(*b->publish));
    b->ain = calloc(na ? na : 1, sizeof(*b->ain));
    b->aout = calloc(na ? na : 1, sizeof(*b->aout));
    b->adeps = calloc(na ? na : 1, sizeof(*b->adeps));
    b->stasks = calloc(ns ? ns : 1, sizeof(*b->stasks));
    b->sdeps = calloc(ns ? ns : 1, sizeof(*b->sdeps));
    if (!b->values || !b->names || !b->buffers || !b->views || !b->actions ||
        !b->submits || !b->publish || !b->ain || !b->aout || !b->adeps ||
        !b->stasks || !b->sdeps) die("bundle allocation");
    for (uint32_t i = 0; i < nv; i++) {
        rk_value_desc *v = &b->values[i];
        memset(v, 0, sizeof(*v));
        v->abi_version = RK_ABI_VERSION;
        v->struct_size = sizeof(*v);
        v->id = ru32(f); v->role = ru32(f); v->dtype = ru32(f);
        v->rank = ru32(f); v->shape[0] = ru32(f); v->shape[1] = ru32(f);
        v->nbytes = ru64(f);
        uint32_t nl; b->names[i] = rstr(f, &nl);
        v->name = b->names[i]; v->name_len = nl;
    }
    for (uint32_t i = 0; i < nb; i++) {
        rk_buffer_desc *x = &b->buffers[i];
        memset(x, 0, sizeof(*x));
        x->abi_version = RK_ABI_VERSION; x->struct_size = sizeof(*x);
        x->id = ru32(f); x->role = ru32(f); x->slot_id = ru32(f);
        x->alignment = ru32(f); x->layout_id = ru32(f);
        x->padding_state = ru32(f); x->lifetime_first = ru32(f);
        x->lifetime_last = ru32(f); x->offset = ru64(f);
        x->capacity = ru64(f);
    }
    for (uint32_t i = 0; i < nw; i++) {
        rk_view_desc *x = &b->views[i];
        memset(x, 0, sizeof(*x));
        x->abi_version = RK_ABI_VERSION; x->struct_size = sizeof(*x);
        x->id = ru32(f); x->buffer_id = ru32(f);
        x->offset_bytes = ru64(f);
        for (unsigned j = 0; j < 4; j++) x->valid_region[j] = ru32(f);
        rk_tensor_desc *t = &x->tensor;
        t->abi_version = RK_ABI_VERSION; t->struct_size = sizeof(*t);
        t->dtype = ru32(f); t->layout = ru32(f);
        t->rows = ru32(f); t->cols = ru32(f);
        t->physical_rows = ru32(f); t->physical_cols = ru32(f);
        t->atom = ru32(f); t->reserved = ru32(f); t->bytes = ru64(f);
    }
    for (uint32_t i = 0; i < na; i++) {
        rk_action_desc *x = &b->actions[i];
        memset(x, 0, sizeof(*x));
        x->abi_version = RK_ABI_VERSION; x->struct_size = sizeof(*x);
        x->id = ru32(f); x->kind = ru32(f); x->template_id = ru32(f);
        x->ninputs = ru32(f); x->noutputs = ru32(f); x->ndeps = ru32(f);
        x->source_binding = ru32(f); x->source_row_offset = ru32(f);
        x->source_col_offset = ru32(f); x->source_row_stride = ru32(f);
        x->flags = ru32(f); x->act_kind = ru32(f);
        if (x->ninputs) {
            b->ain[i] = calloc(x->ninputs, sizeof(uint32_t));
            for (uint32_t j = 0; j < x->ninputs; j++) b->ain[i][j] = ru32(f);
            x->inputs = b->ain[i];
        }
        if (x->noutputs) {
            b->aout[i] = calloc(x->noutputs, sizeof(uint32_t));
            for (uint32_t j = 0; j < x->noutputs; j++) b->aout[i][j] = ru32(f);
            x->outputs = b->aout[i];
        }
        if (x->ndeps) {
            b->adeps[i] = calloc(x->ndeps, sizeof(uint32_t));
            for (uint32_t j = 0; j < x->ndeps; j++) b->adeps[i][j] = ru32(f);
            x->deps = b->adeps[i];
        }
        uint32_t ol; char *origin = rstr(f, &ol);
        memcpy(x->origin, origin, ol < 63 ? ol : 63); free(origin);
    }
    for (uint32_t i = 0; i < ns; i++) {
        rk_submit_desc *x = &b->submits[i];
        memset(x, 0, sizeof(*x));
        x->abi_version = RK_ABI_VERSION; x->struct_size = sizeof(*x);
        x->id = ru32(f); x->mode = ru32(f); x->core_policy = ru32(f);
        x->ntask_ids = ru32(f); x->ndeps = ru32(f);
        if (x->ntask_ids) {
            b->stasks[i] = calloc(x->ntask_ids, sizeof(uint32_t));
            for (uint32_t j = 0; j < x->ntask_ids; j++) b->stasks[i][j] = ru32(f);
            x->task_ids = b->stasks[i];
        }
        if (x->ndeps) {
            b->sdeps[i] = calloc(x->ndeps, sizeof(uint32_t));
            for (uint32_t j = 0; j < x->ndeps; j++) b->sdeps[i][j] = ru32(f);
            x->deps = b->sdeps[i];
        }
    }
    for (uint32_t i = 0; i < np; i++) {
        rk_publish_desc *x = &b->publish[i];
        memset(x, 0, sizeof(*x));
        x->abi_version = RK_ABI_VERSION; x->struct_size = sizeof(*x);
        x->binding_id = ru32(f); x->view_id = ru32(f);
        x->dst_row_offset = ru32(f); x->dst_col_offset = ru32(f);
        x->rows = ru32(f); x->cols = ru32(f);
    }
    fclose(f);
    b->desc = (rk_plan_desc){
        .abi_version = RK_ABI_VERSION, .struct_size = sizeof(rk_plan_desc),
        .schema_version = RK_SCHEMA_VERSION, .flags = 0,
        .profile_hash = profile_hash, .source_hash = source_hash,
        .values = b->values, .nvalues = nv,
        .buffers = b->buffers, .nbuffers = nb,
        .views = b->views, .nviews = nw,
        .actions = b->actions, .nactions = na,
        .submits = b->submits, .nsubmits = ns,
        .publish = b->publish, .npublish = np,
        .device_budget_bytes = device_budget,
        .host_budget_bytes = host_budget,
        .claimed_device_bytes = claimed_device,
        .claimed_host_bytes = claimed_host,
        .runtime_abi_version = RK_ABI_VERSION,
    };
}

static void load_def(const char *path, case_def *c) {
    memset(c, 0, sizeof(*c));
    FILE *f = fopen(path, "r");
    if (!f) die("open case.def");
    char tok[32];
    if (fscanf(f, "%31s %u", tok, &c->frames) != 2 || strcmp(tok, "frames"))
        die("case.def frames");
    if (fscanf(f, "%31s %u", tok, &c->ninputs) != 2 || strcmp(tok, "inputs"))
        die("case.def inputs");
    c->inputs = calloc(c->ninputs ? c->ninputs : 1, sizeof(meta_t));
    for (uint32_t i = 0; i < c->ninputs; i++) {
        unsigned long long nb;
        if (fscanf(f, "%31s %u %u %u %u %llu", tok, &c->inputs[i].id,
                   &c->inputs[i].rank, &c->inputs[i].shape[0],
                   &c->inputs[i].shape[1], &nb) != 6 || strcmp(tok, "input"))
            die("case.def input");
        c->inputs[i].nbytes = nb;
    }
    if (fscanf(f, "%31s %u", tok, &c->nconstants) != 2 ||
        strcmp(tok, "constants")) die("case.def constants");
    c->constants = calloc(c->nconstants ? c->nconstants : 1, sizeof(meta_t));
    for (uint32_t i = 0; i < c->nconstants; i++) {
        unsigned long long nb;
        if (fscanf(f, "%31s %u %llu", tok, &c->constants[i].id, &nb) != 3 ||
            strcmp(tok, "constant")) die("case.def constant");
        c->constants[i].nbytes = nb;
        c->constants[i].rank = 2;
    }
    if (fscanf(f, "%31s %u", tok, &c->noutputs) != 2 || strcmp(tok, "outputs"))
        die("case.def outputs");
    c->outputs = calloc(c->noutputs ? c->noutputs : 1, sizeof(meta_t));
    for (uint32_t i = 0; i < c->noutputs; i++) {
        unsigned long long nb;
        if (fscanf(f, "%31s %u %u %u %u %llu", tok, &c->outputs[i].id,
                   &c->outputs[i].rank, &c->outputs[i].shape[0],
                   &c->outputs[i].shape[1], &nb) != 6 || strcmp(tok, "output"))
            die("case.def output");
        c->outputs[i].nbytes = nb;
    }
    fclose(f);
}

typedef struct {
    void **input_data, **output_data, **const_data;
    uint8_t *frames;
    size_t frame_bytes;
} host_data;

typedef struct {
    double finite_us, upload_us, emit_us, execute_us;
    double download_us, guard_us, publish_us, wall_us;
    uint32_t calls;
} phase_t;

typedef struct {
    rk_plan *plan;
    co_buffer *slots;
    uint32_t nslots;
    co_view *views;
    co_execute_workspace execws;
    int exec_ready;
    co_program *batch;
    uint32_t max_tasks;
    uint32_t *task_submit, *task_index, *submit_len, *submit_mode;
    uint8_t *tile;
    size_t tile_bytes;
    char **stage;
    uint8_t *seen;
    co_program **dsub_prog;
} direct_ctx;

static rk_binding binding_for(const rk_value_desc *v, void *data) {
    rk_binding b;
    memset(&b, 0, sizeof(b));
    b.abi_version = RK_ABI_VERSION;
    b.struct_size = sizeof(b);
    b.id = v->id;
    b.dtype = v->dtype;
    b.rank = v->rank;
    b.shape[0] = v->shape[0];
    b.shape[1] = v->shape[1];
    b.data = data;
    b.nbytes = v->nbytes;
    return b;
}

static void load_case_data(const char *dir, bundle *b, case_def *def,
                           host_data *hd) {
    char path[4096];
    size_t fsz;
    snprintf(path, sizeof(path), "%s/frames.bin", dir);
    hd->frames = read_file(path, &fsz);
    hd->frame_bytes = 0;
    for (uint32_t i = 0; i < def->ninputs; i++)
        hd->frame_bytes += def->inputs[i].nbytes;
    if (hd->frame_bytes * def->frames != fsz) die("frames size mismatch");
    hd->input_data = calloc(b->desc.nvalues ? b->desc.nvalues : 1,
                            sizeof(void *));
    hd->output_data = calloc(b->desc.nvalues ? b->desc.nvalues : 1,
                             sizeof(void *));
    hd->const_data = calloc(b->desc.nvalues ? b->desc.nvalues : 1,
                            sizeof(void *));
    if (!hd->input_data || !hd->output_data || !hd->const_data)
        die("host data allocation");
    for (uint32_t i = 0; i < def->ninputs; i++) {
        hd->input_data[def->inputs[i].id] = calloc(1, def->inputs[i].nbytes);
        if (!hd->input_data[def->inputs[i].id]) die("input allocation");
    }
    for (uint32_t i = 0; i < def->noutputs; i++) {
        hd->output_data[def->outputs[i].id] = calloc(1, def->outputs[i].nbytes);
        if (!hd->output_data[def->outputs[i].id]) die("output allocation");
    }
    for (uint32_t i = 0; i < def->nconstants; i++) {
        snprintf(path, sizeof(path), "%s/const_%u.bin", dir,
                 def->constants[i].id);
        size_t n;
        void *p = read_file(path, &n);
        if (n != def->constants[i].nbytes) die("constant size mismatch");
        hd->const_data[def->constants[i].id] = p;
    }
    if (!rk_f16_all_finite((const uint16_t *)hd->frames,
                           hd->frame_bytes * def->frames / 2u))
        die("generated frame is non-finite");
}

static void load_frame(case_def *def, host_data *hd, uint32_t frame) {
    uint8_t *p = hd->frames + (size_t)frame * hd->frame_bytes;
    for (uint32_t i = 0; i < def->ninputs; i++) {
        memcpy(hd->input_data[def->inputs[i].id], p, def->inputs[i].nbytes);
        p += def->inputs[i].nbytes;
    }
}

static co_tensor tensor_from_desc(const rk_tensor_desc *t) {
    co_tensor out;
    memset(&out, 0, sizeof(out));
    out.type = CO_F16;
    out.layout = t->layout == RK_LAYOUT_WEIGHT_F16 ? CO_WEIGHT :
                 (t->layout == RK_LAYOUT_LINEAR_F16 ? CO_LINEAR : CO_CUBE);
    out.rows = t->rows;
    out.cols = t->cols;
    out.physical_rows = t->physical_rows;
    out.physical_cols = t->physical_cols;
    out.atom = t->atom;
    out.bytes = (size_t)t->bytes;
    return out;
}

/* Exact fast path for the 16x16 CUBE/WEIGHT probe geometry.  This mirrors
 * the runtime's fixed_pack_16x16; it is data preparation, not a new emitter. */
static int direct_fast_pack_16x16(const co_tensor *dst, _Float16 *base,
                                  const _Float16 *src, uint32_t transpose,
                                  uint32_t stride, uint32_t col_offset,
                                  uint64_t elements) {
    if (stride < 16u || col_offset + 16u > stride ||
        dst->rows != 16u || dst->cols != 16u ||
        elements < (uint64_t)16u * stride) return 0;
    if (!transpose && dst->layout == CO_CUBE &&
        dst->physical_rows == 16u && dst->physical_cols == 32u &&
        dst->atom == 8u) {
        for (unsigned r = 0; r < 16; r++) {
            const _Float16 *row = src + (size_t)r * stride + col_offset;
            memcpy(base + (size_t)r * 8u, row, 8u * sizeof(_Float16));
            memcpy(base + 128u + (size_t)r * 8u,
                   row + 8u, 8u * sizeof(_Float16));
        }
        return 1;
    }
    if (dst->layout == CO_WEIGHT &&
        dst->physical_rows == 32u && dst->physical_cols == 32u &&
        dst->atom == 32u) {
        if (!transpose) {
            for (unsigned r = 0; r < 16; r++)
                memcpy(base + (size_t)r * 32u,
                       src + (size_t)r * stride + col_offset,
                       16u * sizeof(_Float16));
        } else {
            for (unsigned c = 0; c < 16; c++)
                for (unsigned r = 0; r < 16; r++)
                    base[(size_t)r * 32u + c] =
                        src[(size_t)c * stride + col_offset + r];
        }
        return 1;
    }
    return 0;
}

static int direct_pack(co_view *dst, const rk_action_desc *a,
                       const void *src, uint64_t nbytes) {
    if (!dst || !a || !src) return -1;
    const _Float16 *s = (const _Float16 *)src;
    const uint64_t elements = nbytes / 2u;
    const int scalar = (a->flags & RK_ACTION_FLAG_BROADCAST_SCALAR) != 0;
    const int channel = (a->flags & RK_ACTION_FLAG_BROADCAST_CHANNEL) != 0;
    const int transpose = (a->flags & RK_ACTION_FLAG_TRANSPOSE_SOURCE) != 0;
    _Float16 *base = (_Float16 *)((char *)dst->buffer->bo.ptr + dst->offset);
    memset(base, 0, dst->tensor.bytes);
    if (!scalar && !channel && a->source_row_offset == 0 &&
        direct_fast_pack_16x16(&dst->tensor, base, s, transpose,
                               a->source_row_stride, a->source_col_offset,
                               elements))
        return 0;
    for (unsigned r = 0; r < dst->tensor.rows; r++) {
        for (unsigned c = 0; c < dst->tensor.cols; c++) {
            _Float16 v;
            if (scalar) {
                if (elements != 1u) return -1;
                v = s[0];
            } else if (channel) {
                uint64_t col = (uint64_t)a->source_col_offset + c;
                if (col >= elements) return -1;
                v = s[col];
            } else {
                uint32_t sr = transpose ? a->source_row_offset + c
                                        : a->source_row_offset + r;
                uint32_t sc = transpose ? a->source_col_offset + r
                                        : a->source_col_offset + c;
                if ((uint64_t)sr * a->source_row_stride + sc >= elements)
                    return -1;
                v = s[(size_t)sr * a->source_row_stride + sc];
            }
            base[co_index(&dst->tensor, r, c)] = v;
        }
    }
    return 0;
}

static int direct_upload_one(int fd, direct_ctx *dc,
                             const rk_action_desc *a, const void *src,
                             uint64_t nbytes) {
    co_view *dst = &dc->views[a->outputs[0]];
    if (rocket_bo_prep(fd, &dst->buffer->bo, 1, 2000000000ull)) return -1;
    if (direct_pack(dst, a, src, nbytes)) return -1;
    return rocket_bo_fini(fd, &dst->buffer->bo);
}

static int plan_is_independent_mm_group(const rk_plan *p) {
    if (!p || p->nsubmits != 1u) return 0;
    const rk_submit_desc *s = &p->submits[0];
    if (s->ntask_ids <= 1u || s->ntask_ids > RK_MAX_INDEPENDENT_MM_TASKS)
        return 0;
    for (uint32_t i = 0; i < s->ntask_ids; i++) {
        const rk_action_desc *a = &p->actions[s->task_ids[i]];
        if (a->kind != RK_ACTION_NPU_TASK ||
            a->template_id != RK_TEMPLATE_MM_F16) return 0;
    }
    return 1;
}

static int direct_upload_run(direct_ctx *dc, const rk_plan *p, int fd,
                             uint32_t begin, uint32_t end, host_data *hd) {
    int coalesce = plan_is_independent_mm_group(p);
    if (!coalesce) {
        for (uint32_t i = begin; i < end; i++) {
            const rk_action_desc *a = &p->actions[i];
            if (a->kind != RK_ACTION_UPLOAD) continue;
            const rk_value_desc *v = &p->values[a->source_binding];
            if (v->role == RK_ROLE_CONSTANT) continue;
            if (direct_upload_one(fd, dc, a, hd->input_data[a->source_binding],
                                  v->nbytes)) return -1;
        }
        return 0;
    }
    co_buffer *prepped[MAX_COALESCED_UPLOADS];
    uint32_t nprepped = 0;
    for (uint32_t i = begin; i < end; i++) {
        const rk_action_desc *a = &p->actions[i];
        if (a->kind != RK_ACTION_UPLOAD) continue;
        const rk_value_desc *v = &p->values[a->source_binding];
        if (v->role == RK_ROLE_CONSTANT) continue;
        co_buffer *buf = dc->views[a->outputs[0]].buffer;
        int seen = 0;
        for (uint32_t q = 0; q < nprepped; q++)
            if (prepped[q] == buf) { seen = 1; break; }
        if (!seen) {
            if (rocket_bo_prep(fd, &buf->bo, 1, 2000000000ull)) return -1;
            prepped[nprepped++] = buf;
        }
        if (direct_pack(&dc->views[a->outputs[0]], a,
                        hd->input_data[a->source_binding], v->nbytes))
            return -1;
    }
    for (uint32_t i = 0; i < nprepped; i++)
        if (rocket_bo_fini(fd, &prepped[i]->bo)) return -1;
    return 0;
}

static uint32_t output_index(case_def *def, uint32_t value_id) {
    for (uint32_t i = 0; i < def->noutputs; i++)
        if (def->outputs[i].id == value_id) return i;
    return UINT32_MAX;
}

static const rk_publish_desc *find_publish(const rk_plan *p,
                                           uint32_t binding, uint32_t view) {
    for (uint32_t i = 0; i < p->npublish; i++)
        if (p->publish[i].binding_id == binding &&
            p->publish[i].view_id == view) return &p->publish[i];
    return NULL;
}

static int direct_download_publish(direct_ctx *dc, const rk_plan *p,
                                   int fd, case_def *def, host_data *hd,
                                   phase_t *prof) {
    (void)fd;
    memset(dc->seen, 0, def->noutputs * sizeof(uint8_t));
    for (uint32_t i = 0; i < def->noutputs; i++)
        memset(dc->stage[i], 0, def->outputs[i].nbytes);
    for (uint32_t i = 0; i < p->nactions; i++) {
        const rk_action_desc *a = &p->actions[i];
        if (a->kind != RK_ACTION_DOWNLOAD) continue;
        const rk_publish_desc *pub = find_publish(p, a->source_binding,
                                                  a->inputs[0]);
        uint32_t oi = output_index(def, a->source_binding);
        if (!pub || oi == UINT32_MAX) return -1;
        co_view *view = &dc->views[a->inputs[0]];
        if (view->tensor.bytes > dc->tile_bytes) return -1;
        double t0 = prof ? now_us() : 0;
        if (co_download(view, dc->tile)) return -1;
        if (prof) prof->download_us += now_us() - t0;
        const uint16_t *tile = (const uint16_t *)dc->tile;
        if (!rk_f16_all_finite(tile, (uint64_t)view->tensor.rows *
                                    view->tensor.cols))
            return -1;
        const rk_value_desc *value = &p->values[pub->binding_id];
        rk_publish_rows(dc->stage[oi], tile, pub->rows, pub->cols,
                        pub->dst_row_offset, pub->dst_col_offset,
                        value->shape[1]);
        dc->seen[oi] = 1;
    }
    for (uint32_t i = 0; i < def->noutputs; i++)
        if (!dc->seen[i]) return -1;
    double t0 = prof ? now_us() : 0;
    for (uint32_t s = 0; s < dc->nslots; s++)
        if (co_guard_check(&dc->slots[s])) return -1;
    if (prof) prof->guard_us += now_us() - t0;
    t0 = prof ? now_us() : 0;
    for (uint32_t i = 0; i < def->noutputs; i++)
        memcpy(hd->output_data[def->outputs[i].id], dc->stage[i],
               def->outputs[i].nbytes);
    if (prof) prof->publish_us += now_us() - t0;
    return 0;
}

static int direct_setup(bundle *b, case_def *def, host_data *hd, int fd,
                        direct_ctx *dc) {
    memset(dc, 0, sizeof(*dc));
    rk_diag diag;
    memset(&diag, 0, sizeof(diag));
    if (rk_plan_clone(&b->desc, &dc->plan, &diag)) return -1;
    uint32_t max_slot = 0;
    for (uint32_t i = 0; i < dc->plan->nbuffers; i++)
        if (dc->plan->buffers[i].slot_id > max_slot)
            max_slot = dc->plan->buffers[i].slot_id;
    dc->nslots = dc->plan->nbuffers ? max_slot + 1u : 0u;
    dc->slots = calloc(dc->nslots ? dc->nslots : 1, sizeof(*dc->slots));
    dc->views = calloc(dc->plan->nviews ? dc->plan->nviews : 1,
                       sizeof(*dc->views));
    if (!dc->slots || !dc->views) return -1;
    uint64_t *cap = calloc(dc->nslots ? dc->nslots : 1, sizeof(uint64_t));
    for (uint32_t i = 0; i < dc->plan->nbuffers; i++) {
        const rk_buffer_desc *x = &dc->plan->buffers[i];
        uint64_t end = x->offset + x->capacity;
        if (end > cap[x->slot_id]) cap[x->slot_id] = end;
    }
    for (uint32_t s = 0; s < dc->nslots; s++)
        if (co_buffer_alloc(fd, (size_t)(cap[s] ? cap[s] : 1),
                            &dc->slots[s])) return -1;
    free(cap);
    for (uint32_t i = 0; i < dc->plan->nviews; i++) {
        const rk_view_desc *v = &dc->plan->views[i];
        const rk_buffer_desc *bb = rk_plan_buffer(dc->plan, v->buffer_id);
        co_tensor t = tensor_from_desc(&v->tensor);
        if (!bb || co_view_init(&dc->views[i], &dc->slots[bb->slot_id],
                                (size_t)v->offset_bytes, &t)) return -1;
    }
    dc->max_tasks = 1;
    for (uint32_t i = 0; i < dc->plan->nsubmits; i++)
        if (dc->plan->submits[i].ntask_ids > dc->max_tasks)
            dc->max_tasks = dc->plan->submits[i].ntask_ids;
    if (co_execute_workspace_init(fd, dc->max_tasks, &dc->execws)) return -1;
    dc->exec_ready = 1;
    dc->batch = calloc(dc->max_tasks, sizeof(*dc->batch));
    if (!dc->batch) return -1;
    dc->task_submit = malloc(dc->plan->nactions * sizeof(uint32_t));
    dc->task_index = malloc(dc->plan->nactions * sizeof(uint32_t));
    dc->submit_len = calloc(dc->plan->nsubmits ? dc->plan->nsubmits : 1,
                            sizeof(uint32_t));
    dc->submit_mode = calloc(dc->plan->nsubmits ? dc->plan->nsubmits : 1,
                             sizeof(uint32_t));
    if (!dc->task_submit || !dc->task_index || !dc->submit_len ||
        !dc->submit_mode) return -1;
    for (uint32_t i = 0; i < dc->plan->nactions; i++) {
        dc->task_submit[i] = UINT32_MAX; dc->task_index[i] = UINT32_MAX;
    }
    for (uint32_t s = 0; s < dc->plan->nsubmits; s++) {
        dc->submit_len[s] = dc->plan->submits[s].ntask_ids;
        dc->submit_mode[s] = dc->plan->submits[s].mode;
        for (uint32_t j = 0; j < dc->submit_len[s]; j++) {
            uint32_t id = dc->plan->submits[s].task_ids[j];
            dc->task_submit[id] = s; dc->task_index[id] = j;
        }
    }
    dc->tile_bytes = 1;
    for (uint32_t i = 0; i < dc->plan->nactions; i++) {
        const rk_action_desc *a = &dc->plan->actions[i];
        if (a->kind == RK_ACTION_DOWNLOAD) {
            uint32_t v = a->inputs[0];
            if (dc->views[v].tensor.bytes > dc->tile_bytes)
                dc->tile_bytes = dc->views[v].tensor.bytes;
        }
    }
    dc->tile = malloc(dc->tile_bytes);
    dc->stage = calloc(def->noutputs ? def->noutputs : 1, sizeof(char *));
    dc->seen = calloc(def->noutputs ? def->noutputs : 1, sizeof(uint8_t));
    if (!dc->tile || !dc->stage || !dc->seen) return -1;
    for (uint32_t i = 0; i < def->noutputs; i++) {
        dc->stage[i] = calloc(1, def->outputs[i].nbytes);
        if (!dc->stage[i]) return -1;
    }
    /* Constant uploads are setup work, exactly once. */
    for (uint32_t i = 0; i < dc->plan->nactions; i++) {
        const rk_action_desc *a = &dc->plan->actions[i];
        if (a->kind != RK_ACTION_UPLOAD) continue;
        const rk_value_desc *v = &dc->plan->values[a->source_binding];
        if (v->role != RK_ROLE_CONSTANT) continue;
        if (!rk_f16_all_finite((const uint16_t *)
                               hd->const_data[a->source_binding],
                               v->nbytes / 2u)) return -1;
        if (direct_upload_one(fd, dc, a,
                              hd->const_data[a->source_binding], v->nbytes))
            return -1;
    }
    return 0;
}

static int direct_run(direct_ctx *dc, case_def *def, host_data *hd, int fd,
                      phase_t *prof) {
    double all0 = prof ? now_us() : 0;
    double t0 = all0;
    /* Same finite-input gate as the C runtime, once per distinct host array. */
    for (uint32_t i = 0; i < def->ninputs; i++) {
        uint32_t id = def->inputs[i].id;
        if (!rk_f16_all_finite((const uint16_t *)hd->input_data[id],
                               def->inputs[i].nbytes / 2u)) return -1;
    }
    if (prof) prof->finite_us += now_us() - t0;
    memset(dc->seen, 0, def->noutputs * sizeof(uint8_t));
    for (uint32_t i = 0; i < def->noutputs; i++)
        memset(dc->stage[i], 0, def->outputs[i].nbytes);
    memset(dc->batch, 0, dc->max_tasks * sizeof(*dc->batch));
    for (uint32_t i = 0; i < dc->plan->nactions; i++) {
        const rk_action_desc *a = &dc->plan->actions[i];
        if (a->kind == RK_ACTION_UPLOAD) {
            uint32_t end = i + 1;
            while (end < dc->plan->nactions &&
                   dc->plan->actions[end].kind == RK_ACTION_UPLOAD) end++;
            t0 = prof ? now_us() : 0;
            if (direct_upload_run(dc, dc->plan, fd, i, end, hd)) return -1;
            if (prof) prof->upload_us += now_us() - t0;
            i = end - 1;
            continue;
        }
        if (a->kind == RK_ACTION_NPU_TASK) {
            uint32_t s = dc->task_submit[a->id];
            uint32_t j = dc->task_index[a->id];
            if (s == UINT32_MAX || j == UINT32_MAX) return -1;
            rk_diag diag;
            memset(&diag, 0, sizeof(diag));
            t0 = prof ? now_us() : 0;
            if (rk_emit_action(dc->plan, dc->views, a->id,
                               &dc->batch[j], &diag)) return -1;
            if (prof) prof->emit_us += now_us() - t0;
            if (j + 1u == dc->submit_len[s]) {
                co_program *prog = dc->batch;
                enum co_mode mode = dc->submit_mode[s] == RK_SUBMIT_CHAIN
                                    ? CO_CHAIN : CO_SEPARATE;
                t0 = prof ? now_us() : 0;
                if (co_execute_profiled(fd, prog, dc->submit_len[s],
                                        mode, &dc->execws, NULL)) return -1;
                if (prof) prof->execute_us += now_us() - t0;
            }
            continue;
        }
        if (a->kind == RK_ACTION_DOWNLOAD) {
            /* handled in the single publication pass below */
            continue;
        }
    }
    t0 = prof ? now_us() : 0;
    if (direct_download_publish(dc, dc->plan, fd, def, hd, prof)) return -1;
    if (prof) {
        prof->wall_us += now_us() - all0;
        prof->calls++;
    }
    return 0;
}

typedef struct {
    rk_plan *cplan;
    rk_workspace *ws;
    rk_binding *inputs, *outputs, *constants;
    uint32_t ninputs, noutputs, nconstants;
} bctx;

static int b_setup(bundle *b, case_def *def, host_data *hd, int fd, bctx *bc) {
    memset(bc, 0, sizeof(*bc));
    rk_diag diag;
    memset(&diag, 0, sizeof(diag));
    if (rk_plan_create(&b->desc, &bc->cplan, &diag)) return -1;
    bc->ninputs = def->ninputs;
    bc->noutputs = def->noutputs;
    bc->nconstants = def->nconstants;
    bc->inputs = calloc(bc->ninputs ? bc->ninputs : 1, sizeof(rk_binding));
    bc->outputs = calloc(bc->noutputs ? bc->noutputs : 1, sizeof(rk_binding));
    bc->constants = calloc(bc->nconstants ? bc->nconstants : 1,
                           sizeof(rk_binding));
    if (!bc->inputs || !bc->outputs || !bc->constants) return -1;
    for (uint32_t i = 0; i < def->ninputs; i++)
        bc->inputs[i] = binding_for(&b->desc.values[def->inputs[i].id],
                                    hd->input_data[def->inputs[i].id]);
    for (uint32_t i = 0; i < def->noutputs; i++)
        bc->outputs[i] = binding_for(&b->desc.values[def->outputs[i].id],
                                     hd->output_data[def->outputs[i].id]);
    for (uint32_t i = 0; i < def->nconstants; i++)
        bc->constants[i] = binding_for(&b->desc.values[def->constants[i].id],
                                       hd->const_data[def->constants[i].id]);
    if (rk_workspace_create(fd, bc->cplan, bc->constants, bc->nconstants,
                            &bc->ws, &diag)) return -1;
    return 0;
}

static int b_run(bctx *bc) {
    rk_diag diag;
    rk_run_report report;
    memset(&diag, 0, sizeof(diag));
    memset(&report, 0, sizeof(report));
    int rc = rk_workspace_run(bc->ws, bc->inputs, bc->ninputs,
                              bc->outputs, bc->noutputs, &report, &diag);
    if (getenv("RL_DEBUG"))
        fprintf(stderr, "RL_DEBUG static=%u ready=%u n=%u/%u rc=%d\n",
                bc->ws->prepared_validation,
                bc->ws->prepared_validation_ready,
                bc->ws->prepared_ninputs, bc->ws->prepared_noutputs, rc);
    return rc;
}

static int b_attr_run(bctx *bc, rk_run_attribution *attr) {
    rk_diag diag;
    memset(&diag, 0, sizeof(diag));
    memset(attr, 0, sizeof(*attr));
    attr->abi_version = RK_ABI_VERSION;
    attr->struct_size = sizeof(*attr);
    return rk_workspace_run_ex(bc->ws, bc->inputs, bc->ninputs,
                               bc->outputs, bc->noutputs, attr, &diag);
}

static int direct_upload_inputs(direct_ctx *dc, case_def *def, host_data *hd,
                                int fd) {
    (void)def;
    for (uint32_t i = 0; i < dc->plan->nactions; i++) {
        const rk_action_desc *a = &dc->plan->actions[i];
        if (a->kind != RK_ACTION_UPLOAD) continue;
        uint32_t end = i + 1;
        while (end < dc->plan->nactions &&
               dc->plan->actions[end].kind == RK_ACTION_UPLOAD) end++;
        if (direct_upload_run(dc, dc->plan, fd, i, end, hd)) return -1;
        i = end - 1;
    }
    return 0;
}

typedef struct {
    direct_ctx dc;
    co_program **dprog;
    uint32_t *dlen, *dmode;
    uint32_t ndsub;
} dctx;

static int d_setup(bundle *b, case_def *def, host_data *hd, int fd, dctx *dc) {
    memset(dc, 0, sizeof(*dc));
    if (direct_setup(b, def, hd, fd, &dc->dc)) return -1;
    dc->ndsub = dc->dc.plan->nsubmits;
    dc->dprog = calloc(dc->ndsub ? dc->ndsub : 1, sizeof(co_program *));
    dc->dlen = calloc(dc->ndsub ? dc->ndsub : 1, sizeof(uint32_t));
    dc->dmode = calloc(dc->ndsub ? dc->ndsub : 1, sizeof(uint32_t));
    co_program *all = calloc(dc->dc.plan->nactions ? dc->dc.plan->nactions : 1,
                             sizeof(co_program));
    if (!dc->dprog || !dc->dlen || !dc->dmode || !all) return -1;
    rk_diag diag;
    memset(&diag, 0, sizeof(diag));
    for (uint32_t i = 0; i < dc->dc.plan->nactions; i++) {
        const rk_action_desc *a = &dc->dc.plan->actions[i];
        if (a->kind != RK_ACTION_NPU_TASK) continue;
        if (rk_emit_action(dc->dc.plan, dc->dc.views, a->id, &all[a->id],
                           &diag)) return -1;
    }
    for (uint32_t s = 0; s < dc->ndsub; s++) {
        const rk_submit_desc *sub = &dc->dc.plan->submits[s];
        dc->dlen[s] = sub->ntask_ids;
        dc->dmode[s] = sub->mode;
        dc->dprog[s] = calloc(sub->ntask_ids ? sub->ntask_ids : 1,
                              sizeof(co_program));
        if (!dc->dprog[s]) return -1;
        for (uint32_t j = 0; j < sub->ntask_ids; j++)
            dc->dprog[s][j] = all[sub->task_ids[j]];
    }
    free(all);
    return 0;
}

static int d_execute(dctx *dc, int fd) {
    for (uint32_t s = 0; s < dc->ndsub; s++) {
        enum co_mode mode = dc->dmode[s] == RK_SUBMIT_CHAIN
                            ? CO_CHAIN : CO_SEPARATE;
        if (co_execute_profiled(fd, dc->dprog[s], dc->dlen[s], mode,
                                &dc->dc.execws, NULL)) return -1;
    }
    return 0;
}

static void write_outputs(case_def *def, host_data *hd, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) die("open output");
    for (uint32_t i = 0; i < def->noutputs; i++)
        if (fwrite(hd->output_data[def->outputs[i].id], 1,
                   def->outputs[i].nbytes, f) != def->outputs[i].nbytes)
            die("write output");
    fclose(f);
}

static void print_timing(const char *setup_key, double setup_us,
                         const char *first_key, double first_us,
                         const double *warm, uint32_t nwarm) {
    printf("%s %.6f\n", setup_key, setup_us);
    printf("%s %.6f\n", first_key, first_us);
    printf("WARM_US");
    for (uint32_t i = 0; i < nwarm; i++) printf(" %.6f", warm[i]);
    printf("\n");
}

static int run_mode(const char *case_dir, const char *mode, int rounds,
                    const char *out_path) {
    char path[4096];
    bundle b;
    case_def def;
    host_data hd;
    snprintf(path, sizeof(path), "%s/bundle.bin", case_dir);
    load_bundle(path, &b);
    snprintf(path, sizeof(path), "%s/case.def", case_dir);
    load_def(path, &def);
    load_case_data(case_dir, &b, &def, &hd);
    int fd = rocket_open();
    if (fd < 0) die("rocket_open");
    double *warm = calloc((size_t)def.frames * rounds, sizeof(double));
    uint32_t nwarm = 0;
    int rc = 0;
    if (!warm) rc = -1;
    if (!rc && (!strcmp(mode, "b") || !strcmp(mode, "c") ||
                !strcmp(mode, "d") || !strcmp(mode, "battr") ||
                !strcmp(mode, "cprof") || !strcmp(mode, "bcmd") ||
                !strcmp(mode, "bcmdverify"))) {
        if (!strcmp(mode, "b")) {
            bctx bc;
            double t0 = now_us();
            if (b_setup(&b, &def, &hd, fd, &bc)) rc = -1;
            double setup = now_us() - t0;
            double first = 0;
            if (!rc) {
                load_frame(&def, &hd, 0);
                t0 = now_us();
                if (b_run(&bc)) rc = -1;
                first = now_us() - t0;
            }
            for (int r = 0; !rc && r < rounds; r++) {
                for (uint32_t f = 1; !rc && f < def.frames; f++) {
                    load_frame(&def, &hd, f);
                    t0 = now_us();
                    if (b_run(&bc)) rc = -1;
                    warm[nwarm++] = now_us() - t0;
                }
            }
            if (!rc) write_outputs(&def, &hd, out_path);
            print_timing("SETUP_US", setup, "FIRST_US", first, warm, nwarm);
        } else if (!strcmp(mode, "c")) {
            direct_ctx dc;
            double t0 = now_us();
            if (direct_setup(&b, &def, &hd, fd, &dc)) rc = -1;
            double setup = now_us() - t0;
            double first = 0;
            if (!rc) {
                load_frame(&def, &hd, 0);
                t0 = now_us();
                if (direct_run(&dc, &def, &hd, fd, NULL)) rc = -1;
                first = now_us() - t0;
            }
            for (int r = 0; !rc && r < rounds; r++) {
                for (uint32_t f = 1; !rc && f < def.frames; f++) {
                    load_frame(&def, &hd, f);
                    t0 = now_us();
                    if (direct_run(&dc, &def, &hd, fd, NULL)) rc = -1;
                    warm[nwarm++] = now_us() - t0;
                }
            }
            if (!rc) write_outputs(&def, &hd, out_path);
            print_timing("SETUP_US", setup, "FIRST_US", first, warm, nwarm);
        } else if (!strcmp(mode, "battr")) {
            bctx bc;
            double t0 = now_us();
            if (b_setup(&b, &def, &hd, fd, &bc)) rc = -1;
            double setup = now_us() - t0;
            if (!rc) {
                for (uint32_t f = 0; f < 8 && !rc; f++) {
                    load_frame(&def, &hd, f % def.frames);
                    if (b_run(&bc)) rc = -1;
                }
                uint32_t na = def.frames < 32 ? def.frames : 32;
                for (uint32_t f = 0; !rc && f < na; f++) {
                    rk_run_attribution attr;
                    load_frame(&def, &hd, f % def.frames);
                    if (b_attr_run(&bc, &attr)) { rc = -1; break; }
                    printf("ATTR_US %.6f %.6f %.6f %.6f %.6f %.6f "
                           "%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                           attr.validate_us, attr.pack_us, attr.emit_us,
                           attr.command_us, attr.submit_alloc_us,
                           attr.submit_wait_us, attr.output_sync_us,
                           attr.download_us, attr.guard_us, attr.publish_us,
                           attr.alloc_us, attr.total_us, attr.pack_layout_us);
                    printf("ATTR_COUNTS %llu %llu %llu\n",
                           (unsigned long long)attr.upload_calls,
                           (unsigned long long)attr.download_calls,
                           (unsigned long long)attr.task_count);
                }
            }
            if (!rc) write_outputs(&def, &hd, out_path);
            print_timing("SETUP_US", setup, "FIRST_US", 0, warm, 0);
        } else if (!strcmp(mode, "bcmdverify")) {
            /* Device-read-only command BO evidence: run the cached path many
             * times, then explicitly sync the command BO from the device and
             * compare it with the snapshot taken after creation.  The production
             * fast path skips this sync; the probe performs it only to read. */
            bctx bc;
            double t0 = now_us();
            if (b_setup(&b, &def, &hd, fd, &bc)) rc = -1;
            double setup = now_us() - t0;
            if (!rc && (!bc.ws->command_cache_ready || !bc.ws->command_plan)) {
                printf("CMDBYTES_NA program_ready=%u command_ready=%u\n",
                       bc.ws->program_cache_ready, bc.ws->command_cache_ready);
                rc = -1;
            }
            unsigned char *before = NULL;
            size_t nbytes = 0;
            if (!rc) {
                nbytes = bc.ws->command_plan->words * 8u;
                before = malloc(nbytes ? nbytes : 1u);
                if (!before) rc = -1;
                else memcpy(before, bc.ws->execute_workspace.commands.ptr,
                            nbytes);
            }
            for (int r = 0; !rc && r < 16; r++) {
                for (uint32_t f = 0; !rc && f < def.frames; f++) {
                    load_frame(&def, &hd, f);
                    if (b_run(&bc)) rc = -1;
                }
            }
            int same = -1;
            if (!rc) {
                int prep = rocket_bo_prep(fd, &bc.ws->execute_workspace.commands,
                                          0, 2000000000ull);
                if (prep) rc = -1;
                else same = memcmp(before,
                                   bc.ws->execute_workspace.commands.ptr,
                                   nbytes) == 0;
                (void)rocket_bo_fini(fd, &bc.ws->execute_workspace.commands);
            }
            if (!rc)
                printf("CMDBYTES_OK %d bytes=%zu runs=%d\n", same, nbytes,
                       16 * (int)def.frames);
            free(before);
            print_timing("SETUP_US", setup, "FIRST_US", 0, warm, 0);
        } else if (!strcmp(mode, "bcmd")) {
            /* Command-BO packing breakdown for an eligible single-submit
             * workspace.  The plan is built into a separate command workspace
             * on the same fd using the workspace's creation-time co_programs. */
            bctx bc;
            double t0 = now_us();
            if (b_setup(&b, &def, &hd, fd, &bc)) rc = -1;
            double setup = now_us() - t0;
            if (!rc) {
                load_frame(&def, &hd, 0);
                if (b_run(&bc)) rc = -1;
            }
            if (!rc && (!bc.ws->program_cache_ready ||
                        !bc.ws->command_cache_ready ||
                        bc.ws->plan->nsubmits != 1u)) {
                printf("CMDBUILD_NA program_ready=%u command_ready=%u submits=%u\n",
                       bc.ws->program_cache_ready, bc.ws->command_cache_ready,
                       bc.ws->plan->nsubmits);
                rc = -1;
            }
            co_execute_workspace measure_ws;
            memset(&measure_ws, 0, sizeof(measure_ws));
            co_command_plan plan;
            memset(&plan, 0, sizeof(plan));
            if (!rc) {
                const rk_submit_desc *submit = &bc.ws->plan->submits[0];
                enum co_mode cmode = submit->mode == RK_SUBMIT_CHAIN
                                     ? CO_CHAIN : CO_SEPARATE;
                uint32_t ntasks = submit->ntask_ids;
                if (co_execute_workspace_init(fd, bc.ws->max_submit_tasks,
                                              &measure_ws)) rc = -1;
                /* Host accounting: only the optional plan descriptor block is
                 * new; the command BO already belongs to execute_workspace. */
                uint64_t host_on = rk_workspace_host_bytes(bc.ws, 1);
                uint64_t host_off = 0;
                const char *old_cache = getenv("RK_PLAN_COMMAND_CACHE");
                char cache_copy[16] = {0};
                if (old_cache) snprintf(cache_copy, sizeof(cache_copy), "%s",
                                        old_cache);
                setenv("RK_PLAN_COMMAND_CACHE", "0", 1);
                rk_workspace *off_ws = NULL;
                rk_diag off_diag;
                memset(&off_diag, 0, sizeof(off_diag));
                if (!rk_workspace_create(fd, bc.cplan, bc.constants,
                                         bc.nconstants, &off_ws, &off_diag)) {
                    host_off = rk_workspace_host_bytes(off_ws, 1);
                    rk_workspace_destroy(off_ws, &off_diag);
                }
                if (cache_copy[0]) setenv("RK_PLAN_COMMAND_CACHE", cache_copy, 1);
                else unsetenv("RK_PLAN_COMMAND_CACHE");
                printf("CMDMEM host_on=%llu host_off=%llu delta=%lld plan_bytes=%zu\n",
                       (unsigned long long)host_on, (unsigned long long)host_off,
                       (long long)host_on - (long long)host_off,
                       sizeof(co_command_plan));
                for (unsigned i = 0; !rc && i < 64u; i++) {
                    co_command_times times;
                    if (co_command_plan_build(fd, bc.ws->batch_scratch, ntasks,
                                              cmode, &measure_ws, &plan,
                                              &times)) { rc = -1; break; }
                    printf("CMDBUILD_US %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                           times.allocation_us, times.clear_us, times.pack_us,
                           times.verify_us, times.sync_us, times.handles_us,
                           times.total_us);
                }
                for (int arm = 0; !rc && arm < 2; arm++) {
                    for (unsigned i = 0; !rc && i < 64u; i++) {
                        co_execute_profile prof;
                        memset(&prof, 0, sizeof(prof));
                        if (co_command_plan_submit(&plan, arm, &prof)) {
                            rc = -1; break;
                        }
                        printf("CMDRUN_US %d %.6f %.6f %.6f\n", arm,
                               prof.command_us, prof.submit_wait_us,
                               prof.output_sync_us);
                    }
                }
                co_command_plan_release(&plan);
                co_execute_workspace_destroy(&measure_ws);
            }
            print_timing("SETUP_US", setup, "FIRST_US", 0, warm, 0);
        } else if (!strcmp(mode, "cprof")) {
            direct_ctx dc;
            double t0 = now_us();
            if (direct_setup(&b, &def, &hd, fd, &dc)) rc = -1;
            double setup = now_us() - t0;
            phase_t prof;
            memset(&prof, 0, sizeof(prof));
            uint32_t nprof = def.frames < 32 ? def.frames : 32;
            for (uint32_t f = 0; !rc && f < nprof; f++) {
                load_frame(&def, &hd, f % def.frames);
                if (direct_run(&dc, &def, &hd, fd, &prof)) rc = -1;
            }
            if (!rc) {
                printf("CPROFILE %u %.6f %.6f %.6f %.6f %.6f %.6f "
                       "%.6f %.6f\n", prof.calls, prof.finite_us,
                       prof.upload_us, prof.emit_us, prof.execute_us,
                       prof.download_us, prof.guard_us, prof.publish_us,
                       prof.wall_us);
                write_outputs(&def, &hd, out_path);
            }
            print_timing("SETUP_US", setup, "FIRST_US", 0, warm, 0);
        } else {
            dctx dc;
            double t0 = now_us();
            if (d_setup(&b, &def, &hd, fd, &dc)) rc = -1;
            double setup = now_us() - t0;
            double first = 0;
            if (!rc) {
                load_frame(&def, &hd, 0);
                if (direct_upload_inputs(&dc.dc, &def, &hd, fd)) rc = -1;
                t0 = now_us();
                if (!rc && d_execute(&dc, fd)) rc = -1;
                first = now_us() - t0;
            }
            for (int r = 0; !rc && r < rounds; r++) {
                for (uint32_t f = 1; !rc && f < def.frames; f++) {
                    load_frame(&def, &hd, f);
                    if (direct_upload_inputs(&dc.dc, &def, &hd, fd)) {
                        rc = -1; break;
                    }
                    t0 = now_us();
                    if (d_execute(&dc, fd)) rc = -1;
                    warm[nwarm++] = now_us() - t0;
                }
            }
            if (!rc) {
                double v0 = now_us();
                if (direct_download_publish(&dc.dc, dc.dc.plan, fd, &def, &hd,
                                           NULL))
                    rc = -1;
                else write_outputs(&def, &hd, out_path);
                printf("VERIFY_US %.6f\n", now_us() - v0);
            }
            print_timing("SETUP_US", setup, "FIRST_US", first, warm, nwarm);
        }
    } else {
        rc = -1;
    }
    rocket_close(fd);
    free(warm);
    return rc ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s CASE_DIR b|c|d [rounds] [out.bin]\n", argv[0]);
        return 2;
    }
    int rounds = argc > 3 ? atoi(argv[3]) : 3;
    char out[4096];
    snprintf(out, sizeof(out), "%s/out_%s.bin", argv[1], argv[2]);
    if (argc > 4) snprintf(out, sizeof(out), "%s", argv[4]);
    return run_mode(argv[1], argv[2], rounds, out);
}

#ifdef RL_DIRECT_SHARED
static void load_case_adapter_data(const char *dir, bundle *b, case_def *def,
                                     host_data *hd) {
    /* The full-model adapter receives Host pointers on every run; do not
     * allocate or read the standalone frame file.  This keeps its resident
     * footprint limited to the BOs/views it actually executes. */
    memset(hd, 0, sizeof(*hd));
    hd->input_data = calloc(b->desc.nvalues ? b->desc.nvalues : 1,
                            sizeof(void *));
    hd->output_data = calloc(b->desc.nvalues ? b->desc.nvalues : 1,
                             sizeof(void *));
    hd->const_data = calloc(b->desc.nvalues ? b->desc.nvalues : 1,
                            sizeof(void *));
    if (!hd->input_data || !hd->output_data || !hd->const_data)
        die("adapter host data allocation");
    char path[4096];
    for (uint32_t i = 0; i < def->nconstants; i++) {
        snprintf(path, sizeof(path), "%s/const_%u.bin", dir,
                 def->constants[i].id);
        size_t n;
        void *p = read_file(path, &n);
        if (n != def->constants[i].nbytes) die("constant size mismatch");
        hd->const_data[def->constants[i].id] = p;
    }
}
#endif

#ifdef RL_DIRECT_SHARED
/* Experimental full-model adapter: expose one direct group context to Python.
 * It is not part of the default runtime and does not alter any product path. */
typedef struct {
    bundle b;
    case_def def;
    host_data hd;
    direct_ctx dc;
    int fd;
    int owns_fd;
} rl_group_ctx;

void *rl_group_open(const char *case_dir, int shared_fd) {
    rl_group_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    char path[4096];
    snprintf(path, sizeof(path), "%s/bundle.bin", case_dir);
    load_bundle(path, &ctx->b);
    snprintf(path, sizeof(path), "%s/case.def", case_dir);
    load_def(path, &ctx->def);
    load_case_adapter_data(case_dir, &ctx->b, &ctx->def, &ctx->hd);
    if (shared_fd >= 0) {
        ctx->fd = shared_fd;
    } else {
        ctx->fd = rocket_open();
        ctx->owns_fd = 1;
    }
    if (ctx->fd < 0) { free(ctx); return NULL; }
    if (direct_setup(&ctx->b, &ctx->def, &ctx->hd, ctx->fd, &ctx->dc)) {
        if (ctx->owns_fd) rocket_close(ctx->fd);
        free(ctx); return NULL;
    }
    return ctx;
}

int rl_group_run(void *opaque, const void *const *inputs,
                 void *const *outputs, uint32_t ninputs, uint32_t noutputs,
                 double *wall_us) {
    rl_group_ctx *ctx = (rl_group_ctx *)opaque;
    if (!ctx || ninputs != ctx->def.ninputs || noutputs != ctx->def.noutputs)
        return -1;
    for (uint32_t i = 0; i < ninputs; i++)
        ctx->hd.input_data[ctx->def.inputs[i].id] = (void *)inputs[i];
    for (uint32_t i = 0; i < noutputs; i++)
        ctx->hd.output_data[ctx->def.outputs[i].id] = outputs[i];
    double t0 = now_us();
    if (direct_run(&ctx->dc, &ctx->def, &ctx->hd, ctx->fd, NULL)) return -1;
    if (wall_us) *wall_us = now_us() - t0;
    return 0;
}

void rl_group_close(void *opaque) {
    rl_group_ctx *ctx = (rl_group_ctx *)opaque;
    if (!ctx) return;
    if (ctx->fd >= 0 && ctx->owns_fd) rocket_close(ctx->fd);
    free(ctx);
}
#endif
