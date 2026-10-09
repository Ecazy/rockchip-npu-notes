// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef CUSTOM_OPS_H
#define CUSTOM_OPS_H
#include "rocket_npu.h"

/* Deliberately bounded RK3588 single-tile API. No hidden CPU compute/fallback.
 * Matrices are row-major at the host boundary, C2 cubes on the device.
 * Matmul convention: output[M,N] = input[M,K] * weights[N,K]^T.
 * Logical dimensions may be ragged; physical rows >=12, %4; channels %32.
 * Unsupported shapes return a negative status before a generator is called. */
enum co_type { CO_F16, CO_F32 };
enum co_layout { CO_CUBE, CO_WEIGHT, CO_LINEAR };
enum co_act { CO_HARDSIGMOID, CO_SIGMOID, CO_GELU_GATE, CO_EXP_BOUNDED, CO_RSQRT_BOUNDED };
enum co_mode { CO_SEPARATE, CO_CHAIN };
#define CO_MAX_WORDS 2048
#define CO_MAX_PROGRAMS 64
/* Logical K for the production linear operator and the primitive tile width.
 * A full input cube may carry up to CO_MAX_K channels; every primitive emitter
 * still contracts at most one CO_K_TILE-wide slice (M/Kt/N <= 256). */
#define CO_MAX_K 4096u
#define CO_K_TILE 256u
#define CO_K_TILES ((CO_MAX_K + CO_K_TILE - 1u) / CO_K_TILE)
#define CO_SLACK 32768u
#define CO_GUARD 64u

typedef struct {
    enum co_type type;
    enum co_layout layout;
    unsigned rows, cols;                 /* logical rows / columns */
    unsigned physical_rows, physical_cols;
    unsigned atom;                       /* 8 fp16 or 4 fp32 elements */
    size_t bytes;
} co_tensor;

typedef struct { rocket_bo bo; int fd; size_t payload; } co_buffer;
typedef struct { co_buffer *buffer; size_t offset; co_tensor tensor; } co_view;
typedef struct {
    uint64_t words[CO_MAX_WORDS];
    uint32_t count, flags;
    co_view reads[3], output;
    unsigned nreads;
} co_program;

int co_tensor_matrix(co_tensor *, unsigned rows, unsigned cols, enum co_type);
/* Full-input cube: same [Kphysical/8, Mphysical, 8] FP16 layout as co_tensor_matrix,
 * but cols may reach CO_MAX_K (co_tensor_matrix stays capped at 256). Used for the
 * packed X[M,K] of the multi-K linear operator. */
int co_tensor_input(co_tensor *, unsigned rows, unsigned cols);
int co_tensor_weight(co_tensor *, unsigned outputs, unsigned inputs);
/* Canonical 256-wide K split of a full input cube. co_k_tiles(k) is the number of
 * tiles; co_k_tile_length(k,i) is tile i's logical length (<= CO_K_TILE). */
unsigned co_k_tiles(unsigned k);
unsigned co_k_tile_length(unsigned k, unsigned index);
/* Byte offset of channel k0 (a multiple of CO_K_TILE) inside a full input cube. */
size_t co_k_tile_offset(const co_tensor *full, unsigned k0);
/* Descriptor for logical channels [k0,k0+kt) of a full input cube. The slice is
 * contiguous at *offset bytes from the buffer base, keeps rows/atom, and rounds its
 * physical columns up to 32; the caller supplies the alignment-checked co_view. */
int co_tensor_input_slice(co_tensor *slice, const co_tensor *full,
                          unsigned k0, unsigned kt, size_t *offset);
int co_tensor_spatial(co_tensor *, unsigned channels, unsigned h, unsigned w);
int co_tensor_linear(co_tensor *, unsigned elements);
size_t co_index(const co_tensor *, unsigned row, unsigned col);
int co_buffer_alloc(int fd, size_t payload, co_buffer *);
void co_buffer_free(co_buffer *);
int co_view_init(co_view *, co_buffer *, size_t offset, const co_tensor *);
int co_upload_f16(co_view *, const _Float16 *row_major);
int co_download(co_view *, void *row_major); /* f16 or f32 according to tensor.type */
int co_sentinel(co_buffer *);                /* payload=NaN bits, tail guard=0xA5 */
int co_guard_check(co_buffer *);             /* checks tail beyond DMA-read slack */
int co_copy(co_view *dst, co_view *src);      /* explicit host round trip baseline */
int co_bias_upload(co_view *, const _Float16 *per_column_bias);
int co_identity_upload(co_view *weights);
int co_ones_upload(co_view *weights, unsigned logical_k, int mean);

int co_emit_mm(co_program *, co_view x, co_view weights, co_view out);
/* K-tile accumulation: out[M,N] = conv(x[M,Kt], weights[N,Kt]) + prior[M,N], fused
 * into one DPU EW add (gen_matmul_fp16 accumulate=1). x/weights are one <=256-wide
 * tile; prior and out must be the identical [M,N] cube in NON-overlapping BOs (the
 * ping-pong accumulator pair), never aliases of x/weights. */
int co_emit_mm_acc(co_program *, co_view x, co_view weights, co_view prior, co_view out);
int co_emit_ew(co_program *, co_view x, co_view operand, co_view identity,
               co_view out, int multiply);  /* exact-shape only; no implicit broadcast */
int co_emit_lut(co_program *, co_view x, co_view out, enum co_act);
/* CO_EXP_BOUNDED: caller guarantees x in [-4,4]; CO_RSQRT_BOUNDED: x in
 * [0.25,4]. These are bounded LUT approximations, not full-range exp/rsqrt.
 * The generator cannot inspect a device cube's values to enforce the domain. */
/* Fixed 512-element bit permutation from the p0312 vendor program; raw linear
 * buffers, NOT an arbitrary ONNX tensor-layout converter. See README gate. */
int co_emit_reformat_8x8x8(co_program *, co_view x, co_view out);
/* HxW is flattened into tensor rows; channels are tensor columns. No padding. */
int co_emit_pool(co_program *, co_view x, co_view out, unsigned h, unsigned w,
                 unsigned kh, unsigned kw, int average);
int co_execute(int fd, const co_program *, unsigned nprograms, enum co_mode);
/* SiLU (gelu=0) or erf-GELU (gelu=1): LUT gate then identity-conv EW multiply.
 * No CPU activation math or tensor round trip during execution. */
int co_gated_activation(int fd, co_view x, co_view out, int gelu, enum co_mode);
/* Explicit hybrid normalization: square/reductions/affine on the NPU; row
 * variance/rsqrt and coefficient materialization on the CPU. Input BO stays
 * cube-resident. FP16 square can overflow; nonfinite statistics are rejected.
 * gamma[cols], optional beta[cols]; layernorm=0 RMSNorm, 1 LayerNorm. */
int co_norm_hybrid(int fd, co_view x, co_view out, const _Float16 *gamma,
                   const _Float16 *beta, float eps, int layernorm, enum co_mode);

/* Reusable custom operator: Y[M,N] = HardSigmoid(X[M,K] * W[N,K]^T + b[N]).
 * W and b are copied/packed once at creation, one <=256-wide K tile per BO. The
 * logical K may reach CO_MAX_K (4096): X is split into contiguous 256-wide K slices
 * of the one packed full-input cube, each slice contracts against its weight tile,
 * and the partials accumulate on the NPU through the DPU EW add (ping-pong A/B
 * accumulators, never in-place). Bias and HardSigmoid run only after the last K
 * tile. Every accumulation pass narrows through fp16, so this is NOT fp32-exact
 * accumulation; it is the measured, shipping fp16 EW K-accumulation.
 * The caller owns fd and must destroy the context BEFORE closing it. Contexts
 * are synchronous and not thread-safe. No RKNN model or CPU compute fallback.
 * Supported logical M/N: 1..256; K: 1..4096. CO_CHAIN is exactly one submit/run.
 * Host pointer arrays must contain the documented element counts. */
typedef struct co_linear_hsig co_linear_hsig;
int co_linear_hsig_create(int fd, unsigned m, unsigned k, unsigned n,
                          const _Float16 *weights, const _Float16 *bias,
                          enum co_mode mode, co_linear_hsig **out);
int co_linear_hsig_run(co_linear_hsig *, const _Float16 *x, _Float16 *y);
/* Cube entry: consumes/emits caller BOs directly; no tensor host round trip.
 * x must use co_tensor_input(M,K)'s exact physical contract; y uses
 * co_tensor_matrix(M,N,CO_F16)'s. K slices are derived as offset views into the
 * caller's x BO without reading or copying its payload. Views belong to the
 * context's fd. In-place / aliases of internal scratch forbidden. Buffers remain
 * caller-owned and must live until this synchronous call returns. */
int co_linear_hsig_run_cube(co_linear_hsig *, co_view x, co_view y);
void co_linear_hsig_destroy(co_linear_hsig *);
#endif
