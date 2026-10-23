/* gdn_cpu_layer.c — fp32 single-thread CPU reference and timing harness for the
 * Qwen3.5-9B GatedDeltaNet layer-0 prefill (S=196).
 *
 * It is the CPU half of the hybrid split measured in
 * results/gdn_timeline_20261022: the RK3588 NPU runs the front-end
 * (RMSNorm -> projections -> causal conv/SiLU -> q/k L2 norm -> beta/g gates),
 * this program runs whichever tail is kept on the CPU and provides the
 * all-CPU baseline for the same geometry.
 *
 * Modes
 *   full <x.bin> <out.bin>       whole layer on the CPU (baseline); prints
 *                                front-end / recurrence / norm / proj timings
 *   rec <fe_dir> <core.bin>      only the chunked delta-rule recurrence, reading
 *                                the NPU front-end tensors q,k,v,beta,g
 *   normproj <core.bin> <z.bin> <y.bin>
 *                                gated RMSNorm(core, z) + out_proj on the CPU
 *
 * Files (fp32 little-endian, row-major) in <fe_dir>:
 *   q.bin [196,32,128]  k.bin [196,32,128]  v.bin [196,32,128]
 *   beta.bin [196,32]   g.bin [196,32]      z.bin [196,4096]
 *
 * build (board):
 *   gcc -O3 -mcpu=native -ffast-math -o gdn_cpu_layer gdn_cpu_layer.c -lm
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define L 196
#define LP 256          /* padded length: 4 chunks x 64 */
#define HIDDEN 4096
#define KEY_DIM 2048
#define VALUE_DIM 4096
#define CONV_DIM 8192
#define HEADS 32        /* value heads (after GQA expansion) */
#define KHEADS 16
#define DK 128
#define DV 128
#define CHUNK 64
#define CHUNKS 4
#define GROUP 2
#define CONV_K 4
#define RMS_EPS 1e-6f
#define L2_EPS 1e-6f

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "oom (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

static float *alloc_f(size_t n) { return (float *)xmalloc(n * sizeof(float)); }

static void read_exact(const char *path, void *dst, size_t bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(2);
    }
    if (fread(dst, 1, bytes, f) != bytes) {
        fprintf(stderr, "%s: short read\n", path);
        exit(2);
    }
    fclose(f);
}

static void write_exact(const char *path, const void *src, size_t bytes) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        perror(path);
        exit(2);
    }
    fwrite(src, 1, bytes, f);
    fclose(f);
}

static void load_f(const char *path, float *dst, size_t n) { read_exact(path, dst, n * sizeof(float)); }

/* C[m,n] = sum_k A[m,k] * W[n,k]   (W is the [N,K] torch weight, untransposed) */
static void matmul_nt(float *C, const float *A, const float *W, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        const float *a = A + (size_t)m * K;
        float *c = C + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            const float *w = W + (size_t)n * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += a[k] * w[k];
            c[n] = acc;
        }
    }
}

/* C[m,n] = sum_k A[m,k] * B[k,n] */
static void matmul_nn(float *C, const float *A, const float *B, int M, int K, int N) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int m = 0; m < M; m++) {
        const float *a = A + (size_t)m * K;
        float *c = C + (size_t)m * N;
        for (int k = 0; k < K; k++) {
            const float av = a[k];
            const float *b = B + (size_t)k * N;
            for (int n = 0; n < N; n++) c[n] += av * b[n];
        }
    }
}

/* C[m,n] = sum_k A[k,m] * B[k,n]  (A transposed) */
static void matmul_tn(float *C, const float *A, const float *B, int M, int K, int N) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int k = 0; k < K; k++) {
        const float *a = A + (size_t)k * M;
        const float *b = B + (size_t)k * N;
        for (int m = 0; m < M; m++) {
            const float av = a[m];
            float *c = C + (size_t)m * N;
            for (int n = 0; n < N; n++) c[n] += av * b[n];
        }
    }
}

/* ---------------------------------------------------------------- weights */
typedef struct {
    float *ln;        /* [HIDDEN] */
    float *wqkv;      /* [CONV_DIM, HIDDEN] */
    float *wz;        /* [VALUE_DIM, HIDDEN] */
    float *wb;        /* [HEADS, HIDDEN] */
    float *wa;        /* [HEADS, HIDDEN] */
    float *conv;      /* [CONV_DIM, CONV_K] */
    float *dtbias;    /* [HEADS] */
    float *alog;      /* [HEADS] */
    float *normw;     /* [DV] */
    float *wout;      /* [HIDDEN, VALUE_DIM] */
} weights;

static void load_weights(weights *w, const char *dir) {
    char path[1024];
#define LOAD(field, n, name)                     \
    snprintf(path, sizeof(path), "%s/%s", dir, name); \
    w->field = alloc_f(n);                       \
    load_f(path, w->field, n)
    LOAD(ln, HIDDEN, "ln.bin");
    LOAD(wqkv, (size_t)CONV_DIM * HIDDEN, "wqkv.bin");
    LOAD(wz, (size_t)VALUE_DIM * HIDDEN, "wz.bin");
    LOAD(wb, (size_t)HEADS * HIDDEN, "wb.bin");
    LOAD(wa, (size_t)HEADS * HIDDEN, "wa.bin");
    LOAD(conv, (size_t)CONV_DIM * CONV_K, "conv.bin");
    LOAD(dtbias, HEADS, "dtb.bin");
    LOAD(alog, HEADS, "alog.bin");
    LOAD(normw, DV, "norm.bin");
    LOAD(wout, (size_t)HIDDEN * VALUE_DIM, "wout.bin");
#undef LOAD
}

/* ------------------------------------------------------------- front-end */
/* x: [L, HIDDEN]; outputs q,k,v [L,HEADS,DK/DV] (q already scaled, q/k L2
 * normalised, GQA expanded), beta,g [L,HEADS] and z [L,VALUE_DIM]. */
static void front_end(const weights *w, const float *x, float *q, float *k, float *v, float *beta, float *g,
                      float *z) {
    float *h = alloc_f((size_t)L * HIDDEN);
    float *sq = alloc_f((size_t)L * HIDDEN);
    float *mixed = alloc_f((size_t)L * CONV_DIM);
    float *padded = alloc_f((size_t)(L + CONV_K - 1) * CONV_DIM);
    float *acc = alloc_f((size_t)L * CONV_DIM);
    float *braw = alloc_f((size_t)L * HEADS);
    float *araw = alloc_f((size_t)L * HEADS);

    /* RMSNorm */
    for (int i = 0; i < L; i++) {
        const float *row = x + (size_t)i * HIDDEN;
        float s = 0.0f;
        for (int j = 0; j < HIDDEN; j++) s += row[j] * row[j];
        float inv = 1.0f / sqrtf(s / HIDDEN + RMS_EPS);
        for (int j = 0; j < HIDDEN; j++) h[(size_t)i * HIDDEN + j] = row[j] * inv * w->ln[j];
    }
    (void)sq;

    /* projections */
    matmul_nt(mixed, h, w->wqkv, L, HIDDEN, CONV_DIM);
    matmul_nt(z, h, w->wz, L, HIDDEN, VALUE_DIM);
    matmul_nt(braw, h, w->wb, L, HIDDEN, HEADS);
    matmul_nt(araw, h, w->wa, L, HIDDEN, HEADS);

    /* causal depthwise conv1d (left pad K-1) + SiLU */
    memset(padded, 0, (size_t)(CONV_K - 1) * CONV_DIM * sizeof(float));
    memcpy(padded + (size_t)(CONV_K - 1) * CONV_DIM, mixed, (size_t)L * CONV_DIM * sizeof(float));
    for (int t = 0; t < L; t++) {
        float *out = acc + (size_t)t * CONV_DIM;
        memset(out, 0, CONV_DIM * sizeof(float));
        for (int j = 0; j < CONV_K; j++) {
            const float *src = padded + (size_t)(t + j) * CONV_DIM;
            for (int c = 0; c < CONV_DIM; c++) out[c] += src[c] * w->conv[(size_t)c * CONV_K + j];
        }
    }
    for (size_t i = 0; i < (size_t)L * CONV_DIM; i++) {
        float a = acc[i];
        acc[i] = a / (1.0f + expf(-a));
    }

    /* split + L2 norm + GQA expansion + gates */
    for (int t = 0; t < L; t++) {
        const float *row = acc + (size_t)t * CONV_DIM;
        for (int hd = 0; hd < HEADS; hd++) {
            int kh = hd / GROUP;
            /* q */
            const float *qs = row + (size_t)kh * DK;
            float *qd = q + ((size_t)t * HEADS + hd) * DK;
            float ss = 0.0f;
            for (int d = 0; d < DK; d++) ss += qs[d] * qs[d];
            float inv = 1.0f / sqrtf(ss + L2_EPS);
            for (int d = 0; d < DK; d++) qd[d] = qs[d] * inv * (1.0f / sqrtf((float)DK));
            /* k */
            const float *ks = row + KEY_DIM + (size_t)kh * DK;
            float *kd = k + ((size_t)t * HEADS + hd) * DK;
            ss = 0.0f;
            for (int d = 0; d < DK; d++) ss += ks[d] * ks[d];
            inv = 1.0f / sqrtf(ss + L2_EPS);
            for (int d = 0; d < DK; d++) kd[d] = ks[d] * inv;
            /* v */
            memcpy(v + ((size_t)t * HEADS + hd) * DV, row + 2 * KEY_DIM + (size_t)hd * DV,
                   DV * sizeof(float));
            /* beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias) */
            float bv = braw[(size_t)t * HEADS + hd];
            beta[(size_t)t * HEADS + hd] = 1.0f / (1.0f + expf(-bv));
            float a = araw[(size_t)t * HEADS + hd] + w->dtbias[hd];
            float sp = log1pf(expf(-fabsf(a))) + (a > 0.0f ? a : 0.0f);
            g[(size_t)t * HEADS + hd] = -expf(w->alog[hd]) * sp;
        }
    }
    free(h);
    free(sq);
    free(mixed);
    free(padded);
    free(acc);
    free(braw);
    free(araw);
}

/* ------------------------------------------------------------- recurrence */
/* Chunked gated delta rule, no initial state.  q,k,v are [LP,HEADS,{DK,DV}]
 * zero padded; beta,g [LP,HEADS].  out [LP,HEADS,DV] -> trimmed to [L,...]. */
static void recurrence(const float *q, const float *k, const float *v, const float *beta, const float *g,
                       float *core) {
    const int T = CHUNK;
    float *qb = alloc_f((size_t)LP * HEADS * DK);
    float *kb = alloc_f((size_t)LP * HEADS * DK);
    float *vb = alloc_f((size_t)LP * HEADS * DV);
    float *bb = alloc_f((size_t)LP * HEADS);
    float *gb = alloc_f((size_t)LP * HEADS);
    memset(qb, 0, (size_t)LP * HEADS * DK * sizeof(float));
    memset(kb, 0, (size_t)LP * HEADS * DK * sizeof(float));
    memset(vb, 0, (size_t)LP * HEADS * DV * sizeof(float));
    memset(bb, 0, (size_t)LP * HEADS * sizeof(float));
    memset(gb, 0, (size_t)LP * HEADS * sizeof(float));
    memcpy(qb, q, (size_t)L * HEADS * DK * sizeof(float));
    memcpy(kb, k, (size_t)L * HEADS * DK * sizeof(float));
    memcpy(vb, v, (size_t)L * HEADS * DV * sizeof(float));
    memcpy(bb, beta, (size_t)L * HEADS * sizeof(float));
    memcpy(gb, g, (size_t)L * HEADS * sizeof(float));

    float *pwd = alloc_f((size_t)T * T);
    float *ut = alloc_f((size_t)T * T);
    float *kmul = alloc_f((size_t)T * DK);
    float *qmul = alloc_f((size_t)T * DK);
    float *vb_t = alloc_f((size_t)T * DV);
    float *kcd = alloc_f((size_t)T * DK);
    float *intra = alloc_f((size_t)T * T);
    float *state = alloc_f((size_t)DK * DV);
    float *newv = alloc_f((size_t)T * DV);
    float *vnew = alloc_f((size_t)T * DV);
    float *kdec = alloc_f((size_t)T * DK);
    float *qdec = alloc_f((size_t)T * DK);
    float *cum = alloc_f(T);
    float *cum_exp = alloc_f(T);
    float *decayed_kb = alloc_f((size_t)T * DK);

    memset(core, 0, (size_t)LP * HEADS * DV * sizeof(float));

    for (int h = 0; h < HEADS; h++) {
        memset(state, 0, (size_t)DK * DV * sizeof(float));
        for (int c = 0; c < CHUNKS; c++) {
            const int t0 = c * T;
            /* cum decay inside the chunk */
            float acc_cum = 0.0f;
            for (int t = 0; t < T; t++) {
                acc_cum += gb[(size_t)(t0 + t) * HEADS + h];
                cum[t] = acc_cum;
                cum_exp[t] = expf(acc_cum);
            }
            /* pairwise decay, strictly-upper = 0 */
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    pwd[(size_t)i * T + j] = (i >= j) ? expf(cum[i] - cum[j]) : 0.0f;
                }
            }
            /* k_beta and q copies for this chunk/head */
            for (int t = 0; t < T; t++) {
                const float bt = bb[(size_t)(t0 + t) * HEADS + h];
                const float *ks = kb + ((size_t)(t0 + t) * HEADS + h) * DK;
                const float *qs = qb + ((size_t)(t0 + t) * HEADS + h) * DK;
                for (int d = 0; d < DK; d++) {
                    kmul[(size_t)t * DK + d] = ks[d] * bt;
                    qmul[(size_t)t * DK + d] = qs[d];
                    kdec[(size_t)t * DK + d] = ks[d] * expf(cum[T - 1] - cum[t]);
                    qdec[(size_t)t * DK + d] = qs[d] * cum_exp[t];
                    decayed_kb[(size_t)t * DK + d] = ks[d] * bt * cum_exp[t];
                }
                memcpy(vb_t + (size_t)t * DV, vb + ((size_t)(t0 + t) * HEADS + h) * DV, DV * sizeof(float));
                for (int d = 0; d < DV; d++) vb_t[(size_t)t * DV + d] *= bt;
            }
            /* ut = (k_beta @ k^T) * pwd ; intra = (q @ k^T) * pwd */
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    const float *ki = kmul + (size_t)i * DK;
                    const float *kj = kb + ((size_t)(t0 + j) * HEADS + h) * DK;
                    float s1 = 0.0f, s2 = 0.0f;
                    for (int d = 0; d < DK; d++) {
                        s1 += ki[d] * kj[d];
                        s2 += qmul[(size_t)i * DK + d] * kj[d];
                    }
                    ut[(size_t)i * T + j] = s1 * pwd[(size_t)i * T + j];
                    intra[(size_t)i * T + j] = s2 * pwd[(size_t)i * T + j];
                }
            }
            /* (I + L) X = B by forward substitution: the reference unitriangular
             * solve (torch.linalg.solve_triangular(..., unitriangular=True)).
             * Stable in fp32 where the nilpotent-doubling inverse used by the
             * exported ONNX graph is not. */
            if (getenv("GDN_DUMP") && h == 0 && c == 0) {
                const char *d = getenv("GDN_DUMP");
                char p[1024];
                snprintf(p, sizeof(p), "%s/pwd.bin", d); write_exact(p, pwd, (size_t)T * T * sizeof(float));
                snprintf(p, sizeof(p), "%s/intra.bin", d); write_exact(p, intra, (size_t)T * T * sizeof(float));
                snprintf(p, sizeof(p), "%s/ut.bin", d); write_exact(p, ut, (size_t)T * T * sizeof(float));
                snprintf(p, sizeof(p), "%s/qdec.bin", d); write_exact(p, qdec, (size_t)T * DK * sizeof(float));
                snprintf(p, sizeof(p), "%s/kdec.bin", d); write_exact(p, kdec, (size_t)T * DK * sizeof(float));
                snprintf(p, sizeof(p), "%s/cum.bin", d); write_exact(p, cum, (size_t)T * sizeof(float));
            }
            memcpy(newv, vb_t, (size_t)T * DV * sizeof(float));
            for (int i = 1; i < T; i++) {
                float *dst_nv = newv + (size_t)i * DV;
                for (int j = 0; j < i; j++) {
                    const float l = ut[(size_t)i * T + j];
                    if (l == 0.0f) continue;
                    const float *src = newv + (size_t)j * DV;
                    for (int e = 0; e < DV; e++) dst_nv[e] -= l * src[e];
                }
            }
            memcpy(kcd, decayed_kb, (size_t)T * DK * sizeof(float));
            for (int i = 1; i < T; i++) {
                float *dst_kc = kcd + (size_t)i * DK;
                for (int j = 0; j < i; j++) {
                    const float l = ut[(size_t)i * T + j];
                    if (l == 0.0f) continue;
                    const float *src = kcd + (size_t)j * DK;
                    for (int d = 0; d < DK; d++) dst_kc[d] -= l * src[d];
                }
            }
            /* scan step for this chunk */
            for (int t = 0; t < T; t++) {
                /* v_new = new_values - k_cumdecay @ state */
                float *vn = vnew + (size_t)t * DV;
                memcpy(vn, newv + (size_t)t * DV, DV * sizeof(float));
                for (int d = 0; d < DK; d++) {
                    const float kv = kcd[(size_t)t * DK + d];
                    if (kv == 0.0f) continue;
                    const float *sr = state + (size_t)d * DV;
                    for (int e = 0; e < DV; e++) vn[e] -= kv * sr[e];
                }
            }
            for (int t = 0; t < T; t++) {
                float *out = core + ((size_t)(t0 + t) * HEADS + h) * DV;
                /* inter-chunk read: q_dec @ state */
                for (int e = 0; e < DV; e++) out[e] = 0.0f;
                for (int d = 0; d < DK; d++) {
                    const float qv = qdec[(size_t)t * DK + d];
                    const float *sr = state + (size_t)d * DV;
                    for (int e = 0; e < DV; e++) out[e] += qv * sr[e];
                }
                /* intra-chunk attention */
                for (int j = 0; j < T; j++) {
                    const float w = intra[(size_t)t * T + j];
                    if (w == 0.0f) continue;
                    const float *vn = vnew + (size_t)j * DV;
                    for (int e = 0; e < DV; e++) out[e] += w * vn[e];
                }
            }
            if (getenv("GDN_DUMP") && h == 0 && c == 0) {
                const char *d = getenv("GDN_DUMP");
                char p[1024];
                snprintf(p, sizeof(p), "%s/newv.bin", d); write_exact(p, newv, (size_t)T * DV * sizeof(float));
                snprintf(p, sizeof(p), "%s/kcd.bin", d); write_exact(p, kcd, (size_t)T * DK * sizeof(float));
                snprintf(p, sizeof(p), "%s/vnew.bin", d); write_exact(p, vnew, (size_t)T * DV * sizeof(float));
            }
            /* state = state * chunk_decay + k_dec^T @ v_new */
            const float cd = cum_exp[T - 1];
            for (size_t i = 0; i < (size_t)DK * DV; i++) state[i] *= cd;
            for (int t = 0; t < T; t++) {
                const float *kd = kdec + (size_t)t * DK;
                const float *vn = vnew + (size_t)t * DV;
                for (int d = 0; d < DK; d++) {
                    const float kv = kd[d];
                    if (kv == 0.0f) continue;
                    float *sr = state + (size_t)d * DV;
                    for (int e = 0; e < DV; e++) sr[e] += kv * vn[e];
                }
            }
            if (getenv("GDN_DUMP") && h == 0 && c == 0) {
                char p[1024];
                snprintf(p, sizeof(p), "%s/state0.bin", getenv("GDN_DUMP"));
                write_exact(p, state, (size_t)DK * DV * sizeof(float));
            }
        }
    }
    free(qb); free(kb); free(vb); free(bb); free(gb);
    free(pwd); free(ut); free(kmul); free(qmul); free(vb_t);
    free(kcd); free(intra); free(state); free(newv); free(vnew); free(kdec); free(qdec);
    free(cum); free(cum_exp); free(decayed_kb);
}

/* ------------------------------------------------ gated RMSNorm + out_proj */
static void norm_proj(const weights *w, const float *core, const float *z, float *y, double *t_norm,
                      double *t_proj) {
    double t0 = now_ms();
    float *rows = alloc_f((size_t)L * VALUE_DIM);
    for (int r = 0; r < L * HEADS; r++) {
        const float *src = core + (size_t)r * DV;
        const float *zz = z + (size_t)r * DV;
        float *dst = rows + (size_t)r * DV;
        float s = 0.0f;
        for (int d = 0; d < DV; d++) s += src[d] * src[d];
        float inv = 1.0f / sqrtf(s / DV + RMS_EPS);
        for (int d = 0; d < DV; d++) {
            float g = 1.0f / (1.0f + expf(-zz[d]));
            dst[d] = src[d] * inv * w->normw[d] * zz[d] * g;
        }
    }
    double t1 = now_ms();
    matmul_nt(y, rows, w->wout, L, VALUE_DIM, HIDDEN);
    double t2 = now_ms();
    *t_norm = t1 - t0;
    *t_proj = t2 - t1;
    free(rows);
}

static int mode_full(const weights *w, const char *xpath, const char *outpath) {
    float *x = alloc_f((size_t)L * HIDDEN);
    float *q = alloc_f((size_t)L * HEADS * DK);
    float *k = alloc_f((size_t)L * HEADS * DK);
    float *v = alloc_f((size_t)L * HEADS * DV);
    float *beta = alloc_f((size_t)L * HEADS);
    float *g = alloc_f((size_t)L * HEADS);
    float *z = alloc_f((size_t)L * VALUE_DIM);
    /* recurrence writes the padded sequence (LP rows) and only the first L are
     * consumed afterwards */
    float *core = alloc_f((size_t)LP * HEADS * DV);
    float *y = alloc_f((size_t)L * HIDDEN);
    load_f(xpath, x, (size_t)L * HIDDEN);

    double t0 = now_ms();
    front_end(w, x, q, k, v, beta, g, z);
    double t1 = now_ms();
    recurrence(q, k, v, beta, g, core);
    double t2 = now_ms();
    double t_norm = 0.0, t_proj = 0.0;
    norm_proj(w, core, z, y, &t_norm, &t_proj);
    printf("full_cpu front_end_ms=%.2f recurrence_ms=%.2f gated_norm_ms=%.2f out_proj_ms=%.2f total_ms=%.2f\n",
           t1 - t0, t2 - t1, t_norm, t_proj, (t2 - t0) + t_norm + t_proj);
    write_exact(outpath, y, (size_t)L * HIDDEN * sizeof(float));
    return 0;
}

static int mode_rec(const char *fedir, const char *outpath) {
    char path[1024];
    float *q = alloc_f((size_t)L * HEADS * DK);
    float *k = alloc_f((size_t)L * HEADS * DK);
    float *v = alloc_f((size_t)L * HEADS * DV);
    float *beta = alloc_f((size_t)L * HEADS);
    float *g = alloc_f((size_t)L * HEADS);
    float *core = alloc_f((size_t)LP * HEADS * DV);
#define LOADFE(field, n, name)                          \
    snprintf(path, sizeof(path), "%s/%s", fedir, name); \
    load_f(path, field, n)
    LOADFE(q, (size_t)L * HEADS * DK, "q.bin");
    LOADFE(k, (size_t)L * HEADS * DK, "k.bin");
    LOADFE(v, (size_t)L * HEADS * DV, "v.bin");
    LOADFE(beta, (size_t)L * HEADS, "beta.bin");
    LOADFE(g, (size_t)L * HEADS, "g.bin");
#undef LOADFE
    double t0 = now_ms();
    recurrence(q, k, v, beta, g, core);
    double t1 = now_ms();
    printf("cpu_recurrence_ms=%.2f\n", t1 - t0);
    write_exact(outpath, core, (size_t)L * HEADS * DV * sizeof(float));
    return 0;
}

static int mode_normproj(const weights *w, const char *corepath, const char *zpath, const char *outpath) {
    float *core = alloc_f((size_t)L * HEADS * DV);
    float *z = alloc_f((size_t)L * VALUE_DIM);
    float *y = alloc_f((size_t)L * HIDDEN);
    load_f(corepath, core, (size_t)L * HEADS * DV);
    load_f(zpath, z, (size_t)L * VALUE_DIM);
    double t_norm = 0.0, t_proj = 0.0;
    norm_proj(w, core, z, y, &t_norm, &t_proj);
    printf("cpu_gated_norm_ms=%.2f cpu_out_proj_ms=%.2f\n", t_norm, t_proj);
    write_exact(outpath, y, (size_t)L * HIDDEN * sizeof(float));
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage:\n"
                "  %s full <weights_dir> <x.bin> <y.bin>\n"
                "  %s rec <fe_dir> <core.bin>\n"
                "  %s normproj <weights_dir> <core.bin> <z.bin> <y.bin>\n",
                argv[0], argv[0], argv[0]);
        return 1;
    }
    weights w;
    if (strcmp(argv[1], "full") == 0 && argc == 5) {
        load_weights(&w, argv[2]);
        return mode_full(&w, argv[3], argv[4]);
    }
    if (strcmp(argv[1], "rec") == 0 && argc == 4) return mode_rec(argv[2], argv[3]);
    if (strcmp(argv[1], "normproj") == 0 && argc == 6) {
        load_weights(&w, argv[2]);
        return mode_normproj(&w, argv[3], argv[4], argv[5]);
    }
    fprintf(stderr, "bad arguments\n");
    return 1;
}
