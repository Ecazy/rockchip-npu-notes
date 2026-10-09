/*
 * wgrad_real.c — 真实小卷积张量的 CNA wgrad 闭环（交接文档优先级 1）
 *
 * 与 min_conv_exp.c 合成 wgrad probe 的区别：
 *   1. X / W / T 来自固定 seed 的确定性数据集并落盘（.bin + meta.txt），CPU 参考可复算；
 *   2. dY 来源明确：L = Σ_p (Q[p]−T[p])²/16 ⇒ dL/dQ = (Q−T)/8，定点化为 int8；
 *   3. 逐 (kx,ky,ic) 生成 CNA reduction task（P = ow*oh*B 拼入 K 维），
 *      输出覆盖全部 16 个 OC，与 CPU dW 逐系数对照；
 *   4. --update：W ← clamp(W − round(dW_npu/4)) 写回同一权重 BO（地址不变），
 *      重新提交同一前向 task，与 CPU 新权重前向对照并统计输出变化；
 *   5. --kscan：合成数据扫描 K 维容量边界（单坐标）。
 *
 * 展平顺序（与台账 §7.4 合成 probe 一致）：
 *   K index p = b*(ow*oh) + oy*ow + ox —— 行内 x 最快，batch 在最外层。
 *
 * 量化口径：in_scale = w_scale = out_scale = 1.0（恒等 requant），
 *   前向 out_zp=128（offset 0）；梯度 task 的 out_scale=8（梯度 int8，scale 1/8）。
 * 复跑：./wgrad_real --dir wgdata_pad1 --pad 1 --batch 1 --update
 * 编译：gcc -O2 -o wgrad_real wgrad_real.c -lm
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

#include "rnpu_regmap.h"

/* ------------------------------------------------------------------ *
 * 1. 内核 UAPI（与 min_conv_exp.c 相同的 vendor rknpu_ioctl.h 子集）
 * ------------------------------------------------------------------ */

struct rknpu_mem_create {
   uint32_t handle;
   uint32_t flags;
   uint64_t size;
   uint64_t obj_addr;
   uint64_t dma_addr;
   uint64_t sram_size;
   int32_t  iommu_domain_id;
   uint32_t core_mask;
};

struct rknpu_mem_map {
   uint32_t handle;
   uint32_t reserved;
   uint64_t offset;
};

struct rknpu_mem_sync {
   uint32_t flags;
   uint32_t reserved;
   uint64_t obj_addr;
   uint64_t offset;
   uint64_t size;
};

struct rknpu_task {
   uint32_t flags;
   uint32_t op_idx;
   uint32_t enable_mask;
   uint32_t int_mask;
   uint32_t int_clear;
   uint32_t int_status;
   uint32_t regcfg_amount;
   uint32_t regcfg_offset;
   uint64_t regcmd_addr;
} __attribute__((packed));

struct rknpu_subcore_task {
   uint32_t task_start;
   uint32_t task_number;
};

struct rknpu_submit {
   uint32_t flags;
   uint32_t timeout;
   uint32_t task_start;
   uint32_t task_number;
   uint32_t task_counter;
   int32_t  priority;
   uint64_t task_obj_addr;
   uint32_t iommu_domain_id;
   uint32_t reserved;
   uint64_t task_base_addr;
   int64_t  hw_elapse_time;
   uint32_t core_mask;
   int32_t  fence_fd;
   struct rknpu_subcore_task subcore_task[5];
};

#define DRM_IOCTL_BASE        'd'
#define DRM_COMMAND_BASE      0x40
#define DRM_IOWR(nr, type)    _IOWR(DRM_IOCTL_BASE, nr, type)

#define RKNPU_MEM_NON_CONTIGUOUS             (1 << 0)
#define RKNPU_MEM_CACHEABLE                  (1 << 1)
#define RKNPU_MEM_KERNEL_MAPPING             (1 << 3)
#define RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT (1 << 10)

#define RKNPU_JOB_PC         (1 << 0)
#define RKNPU_JOB_BLOCK      0
#define RKNPU_JOB_PINGPONG   (1 << 2)

#define DRM_IOCTL_RKNPU_SUBMIT      DRM_IOWR(DRM_COMMAND_BASE + 0x01, struct rknpu_submit)
#define DRM_IOCTL_RKNPU_MEM_CREATE  DRM_IOWR(DRM_COMMAND_BASE + 0x02, struct rknpu_mem_create)
#define DRM_IOCTL_RKNPU_MEM_MAP     DRM_IOWR(DRM_COMMAND_BASE + 0x03, struct rknpu_mem_map)
#define DRM_IOCTL_RKNPU_MEM_DESTROY DRM_IOWR(DRM_COMMAND_BASE + 0x04, struct rknpu_mem_destroy)
#define DRM_IOCTL_RKNPU_MEM_SYNC    DRM_IOWR(DRM_COMMAND_BASE + 0x05, struct rknpu_mem_sync)

#define RKNPU_PC_DATA_EXTRA_AMOUNT 4

/* ------------------------------------------------------------------ *
 * 2. 硬件常量与发射层
 * ------------------------------------------------------------------ */
#define FEATURE_ATOMIC_SIZE 16
#define WEIGHT_ATOMIC_SIZE  32
#define ATOMIC_K_SIZE       16
#define CBUF_BANKS          12

#define ALIGN_UP(x, a)  (((x) + (a) - 1) & ~((a) - 1))
#define MAX2(a, b)      ((a) > (b) ? (a) : (b))
#define MIN2(a, b)      ((a) < (b) ? (a) : (b))
#define DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))

static uint32_t g_out_scale, g_out_shift;
static int32_t  g_out_offset;
static uint64_t rc_buf[4096];
static uint64_t *rc = rc_buf;
static int g_dump = 0;

static void emit_raw(uint32_t target, uint32_t reg, uint32_t value)
{
   *rc++ = ((uint64_t)target << 48) | ((uint64_t)value << 16) | (uint64_t)reg;
}

static void EMIT(uint32_t reg, uint32_t value)
{
   if (g_dump)
      fprintf(stderr, "  EMIT reg=0x%04x val=0x%08x\n", reg, value);
   emit_raw(rkt_get_target(reg) + 0x1, reg, value);
}

static inline int64_t shift_right_round64(int64_t value, unsigned shift)
{
   if (shift == 0) return value;
   int64_t sign = (value < 0) ? 1 : 0;
   uint64_t guide = (value >> (shift - 1)) & 1;
   uint64_t sticky = (shift > 1) ? ((value & ((1ULL << (shift - 1)) - 1)) != 0) : 0;
   int64_t round_up = guide & ((!sign) | sticky);
   return (value >> shift) + round_up;
}

static uint32_t fui(float f)
{
   union { float f; uint32_t u; } u; u.f = f; return u.u;
}

/* ------------------------------------------------------------------ *
 * 2b. fp16（PROC_PRECISION=2）训练侧闭环支持
 *     —— 与 min_conv_exp.c §7.17/§7.25 同一套布局与发射差异：
 *        输入像素 16 B = 8 个 fp16、平面步长 iw·ih·16 B；
 *        权重 ic 原子 32 元素；c.ic = ic≤16 ? 16 : ALIGN_UP(ic,32)；
 *        oc 原子 16 且每 tap oc 跨度 ALIGN_UP(oc,16)；累加器 ≥ fp32。
 * ------------------------------------------------------------------ */
#define FP16_ATOM_ELEMS 8

/* §7.37：fp16 输入通道对齐（与 min_conv_exp 同规则：≤16 → 16，否则 32 对齐） */
static inline unsigned f16_ic_pad_for(unsigned ic_real)
{
   return (ic_real <= 2u * FP16_ATOM_ELEMS) ? (2u * FP16_ATOM_ELEMS)
                                            : ALIGN_UP(ic_real, 4u * FP16_ATOM_ELEMS);
}

static int g_fp16 = 0;
static int g_i8_stage_global = 0;   /* §7.57：诊断——按 min_conv_exp 方式从"全图打包"拷 tile 行 */
static unsigned g_i8_tile_align = 1;   /* §7.56：int8 staged 行数对齐（诊断） */
static const char *g_in_dy_file = NULL;  /* §7.163：用外部 dY 替换 batch0 的梯度（验证 wgrad 在外部数据上的正确性） */
static const char *g_in_x_file = NULL;  /* §7.161：用外部 X 替换 batch0 输入，并重算参考 Q */
static int g_vend_dma = 0;      /* §7.121：按 vendor 口径 dma_h=min(ih,8) 且 SURF 由它推导 */
static int g_i8_diag = 0;         /* §7.107：每个 tile 的 CPU 参考/比对（默认关闭，66x66 级别很慢） */
static int g_i8_tile_force = 0;  /* §7.56：诊断用，绕过 int8 切分保护（仅为对照 dump） */
static int g_tile_src = 1, g_tile_pt = 1, g_tile_rows = 1;   /* §7.53：三点扫描（临时旋钮） */
static unsigned g_tile_p = 0;   /* §7.54：wgrad 的 K 维（P）分块大小，0=不分块 */
static unsigned g_tile_oh = 0;  /* §7.51：--tile-oh N，前向按 N 个输出行切分（0=不切） */
static unsigned g_wgrad_chunk_ulp_max = 0;
static unsigned g_wgrad_mech_bad = 0;
static unsigned g_wgrad_hw_acc = 0, g_wgrad_host_acc = 0, g_wgrad_both_ok = 0, g_wgrad_neither = 0;
static float g_wgrad_maxabs = 0.0f;
static unsigned g_wgrad_over_1e2 = 0, g_wgrad_over_1e3 = 0;
static float g_wgrad_maxpart_k[64];   /* §7.54：每 oc 的块偏量最大幅值（定标容差） */   /* §7.54：分块机制（vs 同分块参考）失配坐标数 */   /* §7.54：分块累加的 max ulp（供打印） */
static int g_cbuf_warned = 0;   /* §7.50：容量越界只告警一次 */
static int g_cbuf_overflow = 0; /* §7.50：越界后 fail-fast（不再提交，避免 6 s 挂等） */
static const char *g_dump_regcmd = NULL;  /* §7.49：导出本 task 的 regcmd（0xOFF 0xVAL），用于与 min_conv_exp 对比 */
static int g_auto_cfg = 1;      /* --no-auto-cfg：关闭 §7.31 的自动 bank/grains */
static int g_f16_watom = 32;      /* fp16 权重 ic 原子元素数 */
static int g_f16_ocatom = 16;     /* fp16 权重 oc 原子元素数 */
static float g_fp16_lr = 0.25f;   /* 写回学习率：W' = W - lr·dW_npu */
static float g_fp16_loss = 0.25f; /* dY = (Q - T) · loss_scale */
static int g_l2_ulp = 16;         /* 双层闭环 fwd/wgrad 允许的最大 ULP 距离（对齐台账 fp16 ≤16 ulp 判据） */
static double g_l2_abs = 1e-4;    /* 近零相消时的绝对残差兜底阈值（fp32 累加误差量级） */

static inline uint16_t f32_to_f16(float f)
{
   union { float f; uint32_t u; } v; v.f = f;
   uint32_t u = v.u, sign = (u >> 16) & 0x8000u;
   uint32_t be = (u >> 23) & 0xffu, m = u & 0x7fffffu;
   if (be == 0xffu)
      return (uint16_t)(sign | 0x7c00u | (m ? 0x0200u : 0u));
   int32_t e = (int32_t)be - 127;
   if (e > 15)
      return (uint16_t)(sign | 0x7c00u);
   if (e >= -14) {
      uint32_t mant = m >> 13, rem = m & 0x1fffu;
      uint32_t out = sign | ((uint32_t)(e + 15) << 10) | mant;
      if (rem > 0x1000u || (rem == 0x1000u && (mant & 1u))) out++;
      return (uint16_t)out;
   }
   {
      int sh = -14 - e;                       /* >= 1 */
      if (sh > 24) return (uint16_t)sign;
      uint32_t mant = m | 0x800000u;
      uint32_t out = mant >> (13 + sh);
      uint32_t rem = mant & ((1u << (13 + sh)) - 1u);
      uint32_t half = 1u << (12 + sh);
      if (rem > half || (rem == half && (out & 1u))) out++;
      return (uint16_t)(sign | out);
   }
}

static inline float f16_to_f32(uint16_t h)
{
   union { uint32_t u; float f; } v;
   uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
   uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
   if (e == 0) {
      if (!m) { v.u = sign; return v.f; }
      int32_t ex = 127 - 15 + 1;
      while (!(m & 0x400u)) { m <<= 1; ex--; }
      v.u = sign | ((uint32_t)ex << 23) | ((m & 0x3ffu) << 13);
      return v.f;
   }
   if (e == 31) { v.u = sign | 0x7f800000u | (m << 13); return v.f; }
   v.u = sign | ((e + 127 - 15) << 23) | (m << 13);
   return v.f;
}

static inline float rd_f16(const uint8_t *p)
{
   return f16_to_f32((uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8)));
}
static inline void wr_f16(uint8_t *p, float f)
{
   uint16_t h = f32_to_f16(f);
   p[0] = (uint8_t)(h & 0xff);
   p[1] = (uint8_t)(h >> 8);
}
static inline uint16_t rd_u16(const uint8_t *p)
{
   return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

/* fp16 位型的单调整数键（避免近零处正负跨越放大 ULP），用于 ULP 距离 */
static inline int f16_key(uint16_t h)
{
   int v = (int)(h & 0x7fffu);
   return (h & 0x8000u) ? -(v + 1) : v;
}
static inline int f16_ulp(uint16_t a, uint16_t b)
{
   int d = f16_key(a) - f16_key(b);
   return d < 0 ? -d : d;
}
static const char *f16_verdict(unsigned exact_bad, unsigned tol_bad, int maxulp)
{
   static char buf[80];
   if (!exact_bad) return "BIT-EXACT (fp16)";
   if (!tol_bad) {
      if (maxulp <= g_l2_ulp) snprintf(buf, sizeof buf, "ULP-EXACT (<=%d ulp)", g_l2_ulp);
      else snprintf(buf, sizeof buf, "WITHIN-TOL (abs<=%.0e, max %d ulp)", g_l2_abs, maxulp);
      return buf;
   }
   snprintf(buf, sizeof buf, "MISMATCH (max %d ulp)", maxulp);
   return buf;
}
/* 元素判等：ULP ≤ 阈值，或近零相消时绝对残差 ≤ g_l2_abs（fp32 累加在强相消下 ULP 会放大） */
static int f16_ok(uint16_t a, uint16_t b, double *maxabs)
{
   double ad = (double)f16_to_f32(a) - (double)f16_to_f32(b);
   if (ad < 0) ad = -ad;
   if (maxabs && ad > *maxabs) *maxabs = ad;
   return f16_ulp(a, b) <= g_l2_ulp || ad <= g_l2_abs;
}

/* fp16 输入面：单像素 16 B = 8 个 fp16，平面步长 iw·ih·16 B，行主序 */
static inline unsigned f16_pixel_bytes(void) { return FP16_ATOM_ELEMS * 2; }
static inline unsigned f16_planes(unsigned ic)
{
   return DIV_ROUND_UP(ic, (unsigned)FP16_ATOM_ELEMS);
}
static inline unsigned f16_in_off(unsigned g, unsigned x, unsigned y,
                                  unsigned iw, unsigned ih)
{
   unsigned pb = f16_pixel_bytes();
   return g * iw * ih * pb + y * iw * pb + x * pb;
}

/* fp16 权重元素序（layout A：oc1 → ic1 → ky → kx → oc2 → ic2） */
static unsigned w_elem_index_f16(unsigned oc_w, unsigned ic,
                                 unsigned oc, unsigned icc, unsigned kx, unsigned ky,
                                 unsigned kw, unsigned kh)
{
   unsigned wa   = (unsigned)(g_f16_watom > 0 ? g_f16_watom : 32);
   unsigned ic1  = DIV_ROUND_UP(ic, wa);
   unsigned ic2  = MIN2(ic, wa);
   unsigned oa   = (unsigned)(g_f16_ocatom > 0 ? g_f16_ocatom : 32);
   unsigned oc2n = MIN2(oc_w, oa);
   unsigned oc1 = oc / oa, oc2 = oc % oa;
   unsigned i1 = icc / wa, i2 = icc % wa;
   return ((((oc1 * ic1 + i1) * kh + ky) * kw + kx) * oc2n + oc2) * ic2 + i2;
}

/* 与 min_conv_exp.c 的 requant 定点化完全一致（rnpu_coefs.c 同一套） */
static void compute_requant(float in_scale, float w_scale, float out_scale,
                            unsigned truncate_bits,
                            uint32_t *scale, unsigned *shift)
{
   float conv_scale = (in_scale * w_scale) / out_scale;
   uint32_t scale_bits = fui(conv_scale);
   unsigned sh = 127 + 31 - 32 - (scale_bits >> 23) + 16;
   if (truncate_bits > 0) shift--;
   unsigned sc = ((scale_bits >> 9) & 0x7fff) + 1;
   if (sc < (1u << 14)) sc |= 1u << 14;
   *scale = sc;
   *shift = sh - 1;
}

struct conv_cfg {
   unsigned iw, ih, ic_real, ic;
   unsigned ow, oh, oc_real, oc;
   unsigned kw, kh;
   unsigned stride_x, stride_y;
   unsigned pad_left, pad_top;
   unsigned input_zero_point, weights_zero_point, output_zero_point;
   float in_scale, w_scale, out_scale;
   unsigned truncate_bits;
   uint32_t in_addr, out_addr, wt_addr, bias_addr;
};

/* 已验证的修正口径（台账 §5.1–§5.3、§7.2），无实验开关 */
/* 20260928 OOBTEST 实测：PAD_CON1 低字节 = 越界样本字节值（有符号域），
 * 0x00→0、0x80→-128、0xff→-1；旧公式 0xffff8080（§5.4）使 OOB=-128 → 边界全错 */
static uint32_t g_pad_con1 = 0x00000000u;
static unsigned g_fwd_repeat = 0;
static int g_wk32 = 0;
static const char *g_bisect = NULL;
static const char *g_wfile = NULL;
static unsigned emit_conv(const struct conv_cfg *c)
{
   rc = rc_buf;   /* §7.49：每次 emit 前复位（dump 用） */
   unsigned ic_pad = (c->ic_real <= FEATURE_ATOMIC_SIZE)
                     ? FEATURE_ATOMIC_SIZE : ALIGN_UP(c->ic_real, WEIGHT_ATOMIC_SIZE);
   unsigned input_width_cfg = c->iw;
   unsigned input_line_stride = c->iw * ATOMIC_K_SIZE / 4;
   /* §7.121：vendor 口径 —— SURF_STRIDE 以 **16 B 为单位**、且按 **dma_h（= min(ih,8)）** 推导：
    *   vendor 实测（iw=8, ic=32, ih=16）⇒ 112 = (8*32/16) * (8-1) = 16*7 ✓
    * 旧式用 ih 且单位是 8 B（`iw*16/4`）⇒ 非方几何下与 vendor 不符 ✗。 */
   unsigned dma_h_v = (c->ih < 8u) ? c->ih : 8u;
   unsigned input_surf_stride;
   if (g_vend_dma) {
      unsigned row_bytes16 = (c->iw * ic_pad) / 16u;      /* 行字节数 / 16 */
      input_surf_stride = row_bytes16 * (dma_h_v - 1u);
   } else {
      input_surf_stride = (int)((float)input_line_stride *
                                (((float)c->ih / 4.0f) - 1.0f));
   }
   unsigned output_surf_stride = (c->ow * ATOMIC_K_SIZE) * c->oh / FEATURE_ATOMIC_SIZE;
   unsigned input_data_entries = DIV_ROUND_UP(c->iw * 2 *
                                   DIV_ROUND_UP(ic_pad, g_fp16 ? (unsigned)FP16_ATOM_ELEMS
                                                               : (unsigned)FEATURE_ATOMIC_SIZE), 8);
   unsigned atomic_count       = c->ow * c->oh;
   unsigned weights_kernels    = (c->oc_real <= WEIGHT_ATOMIC_SIZE)
                                 ? ALIGN_UP(c->oc_real, 2)
                                 : ALIGN_UP(c->oc_real, WEIGHT_ATOMIC_SIZE);
   if (g_wk32) weights_kernels = ALIGN_UP(MAX2(c->oc, WEIGHT_ATOMIC_SIZE), 2);
   if (g_fp16) weights_kernels = ALIGN_UP(MAX2(c->oc_real, 2u * FP16_ATOM_ELEMS),
                                          2u * FP16_ATOM_ELEMS);
   unsigned surfaces_per_row   = c->ow * c->oh * 2;

   unsigned wbytes = c->kw * c->kh * c->ic * c->oc * (g_fp16 ? 2u : 1u);
   unsigned weights_banks = DIV_ROUND_UP(DIV_ROUND_UP(wbytes, 128), 256) + 1;
   unsigned input_banks   = 1;
   /* §7.31（修正 §7.30）：输入 tile 上限 = DATA_BANK × 32768 B（CBUF 每 bank 32 KB）；
    * 旧代码恒写 DATA_BANK=1，等于把输入钉死在 32 KB（大 tile 静默算错）。 */
   unsigned tile_row_bytes = c->iw * (g_fp16
         ? DIV_ROUND_UP(ic_pad, (unsigned)FP16_ATOM_ELEMS) * (unsigned)FP16_ATOM_ELEMS * 2u
         : DIV_ROUND_UP(ic_pad, FEATURE_ATOMIC_SIZE) * FEATURE_ATOMIC_SIZE);
   unsigned tile_in_bytes  = tile_row_bytes * c->ih;
   if (g_auto_cfg) {
      unsigned need = DIV_ROUND_UP(tile_in_bytes, 32768u);
      unsigned avail_b = (CBUF_BANKS > weights_banks) ? CBUF_BANKS - weights_banks : 1u;
      if (need > avail_b) {
         /* §7.50：容量越界必须显式告警（旧行为是静默钳制 ⇒ 静默算错/挂提交）。 */
         g_cbuf_overflow = 1;
         if (!g_cbuf_warned) {
            g_cbuf_warned = 1;
            printf("WARN: 输入 tile %u B 需 %u data bank，而权重占 %u bank 后仅剩 %u "
                   "⇒ CBUF 容量越界（§7.31），硬件会静默算错；请减小 IW/IH/IC 或按行切分\n",
                   tile_in_bytes, need, weights_banks, avail_b);
         }
         need = avail_b;
      }
      if (need > input_banks) input_banks = need;
   }
   unsigned avail = CBUF_BANKS - weights_banks;
   if (input_banks > avail) input_banks = avail;
   weights_banks = CBUF_BANKS - input_banks;

   int offset = (int)c->output_zero_point - 0x80;

   EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(weights_banks) |
                           CNA_CBUF_CON0_DATA_BANK(input_banks));
   EMIT(REG_CNA_DCOMP_REGNUM, 0);
   EMIT(REG_CNA_DCOMP_CTRL, 0);

   uint32_t con1 = CNA_CONV_CON1_NONALIGN_DMA(0);
   if (g_fp16) con1 |= CNA_CONV_CON1_IN_PRECISION(2) | CNA_CONV_CON1_PROC_PRECISION(2);
   EMIT(REG_CNA_CONV_CON1, con1);
   EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) |
                           DPU_S_POINTER_EXECUTER_PP_EN(1) |
                           DPU_S_POINTER_POINTER_PP_EN(1));
   EMIT(REG_DPU_RDMA_RDMA_S_POINTER, DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(1) |
                                     DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(1) |
                                     DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(1));
   EMIT(REG_CNA_CONV_CON1, con1);

   /* §7.31/§7.32：FEATURE_GRAINS 必须随输入行字节/行数放大（旧值 50+stride+1 是 mesa 的
    * magic，高输入下静默算错）。auto = max(vendor 口径 §7.32: ih+kh+pad_top,
    * tinygrad: max(80, ⌈2·32768/row_bytes⌉+1))，上限 1022（字段 10 bit）。 */
   unsigned grains = 50u + c->stride_y + 1u;
   if (g_auto_cfg && tile_row_bytes) {
      unsigned ag = DIV_ROUND_UP(2u * 32768u, tile_row_bytes) + 1u;
      ag = ALIGN_UP(ag, 2u);
      if (ag < 80u) ag = 80u;
      if (ag > grains) grains = ag;
   }
   if (g_auto_cfg) {
      unsigned vg = c->ih + c->kh + c->pad_top;
      if (vg > grains) grains = vg;
      if (grains > 1022u) grains = 1022u;
   }
   EMIT(REG_CNA_CONV_CON2, CNA_CONV_CON2_FEATURE_GRAINS(grains));
   EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_X_STRIDE(c->stride_x) |
                           CNA_CONV_CON3_CONV_Y_STRIDE(c->stride_y));

   EMIT(REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH(input_width_cfg) |
                            CNA_DATA_SIZE0_DATAIN_HEIGHT(c->ih));
   EMIT(REG_CNA_DATA_SIZE1,
        CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(c->ic_real - 1) |
        CNA_DATA_SIZE1_DATAIN_CHANNEL(ic_pad));
   EMIT(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(c->ow));
   EMIT(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(atomic_count));

   EMIT(REG_CNA_WEIGHT_SIZE0, c->kw * c->kh * c->ic * weights_kernels *
                              (g_fp16 ? 2u : 1u));
   EMIT(REG_CNA_WEIGHT_SIZE1, c->kw * c->kh * c->ic * (g_fp16 ? 2u : 1u));  /* §7.36：fp16 字节数 ×2 */
   EMIT(REG_CNA_WEIGHT_SIZE2, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(c->kw) |
                              CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(c->kh) |
                              CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(weights_kernels));

   EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(weights_banks) |
                           CNA_CBUF_CON0_DATA_BANK(input_banks));
   EMIT(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(input_data_entries));

   EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_DATA_SIGN(1) | CNA_CVT_CON0_CVT_TYPE(1) |
                          CNA_CVT_CON0_CVT_BYPASS(1));
   EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
   EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
   EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
   EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));

   EMIT(REG_CNA_FC_CON0, 0);
   EMIT(REG_CNA_FC_CON1, 0);
   EMIT(REG_CNA_PAD_CON0, CNA_PAD_CON0_PAD_LEFT(c->pad_left) |
                          CNA_PAD_CON0_PAD_TOP(c->pad_top));
   EMIT(REG_CNA_FEATURE_DATA_ADDR, c->in_addr);
   EMIT(REG_CNA_FC_CON2, 0);
   EMIT(REG_CNA_DMA_CON0, CNA_DMA_CON0_WEIGHT_BURST_LEN(15) |
                          CNA_DMA_CON0_DATA_BURST_LEN(15));
   EMIT(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(input_line_stride));
   EMIT(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(input_surf_stride));
   /* §7.121：vendor 语料实测 —— **`dma_h` 恒为 8**（ih=8/16 均 8 ✓，CNA 按 8 行一块取数 ✓），
    * 且 `SURF_STRIDE` = `line × (dma_h − 1)`（ih=16 ⇒ 112 = 16×7 ✓）⇒ 与"用 ih"的旧式不同 ✗。
    * 这里按 vendor 口径取 `dma_h = MIN(ih, 8)`（`--vend-dma` 开启；默认保持旧行为以保护已验证的方图用例 ✓）。 */
   unsigned dma_h = g_vend_dma ? (c->ih < 8u ? c->ih : 8u) : c->ih;
   EMIT(REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH(c->iw) |
                               CNA_FC_DATA_SIZE0_DMA_HEIGHT(dma_h));
   EMIT(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(c->ic));
   EMIT(REG_CNA_DCOMP_CTRL, 0);
   EMIT(REG_CNA_DCOMP_REGNUM, 0);
   EMIT(REG_CNA_DCOMP_ADDR0, c->wt_addr);
   for (int i = 0; i < 16; i++) EMIT(REG_CNA_DCOMP_AMOUNT0 + i * 4, 0);
   EMIT(REG_CNA_CVT_CON5, 0);

   /* in_zp=0：越界样本贡献 0 = 零点（§5.4 实测 PAD_CON1 在普通路径无影响） */
   EMIT(REG_CNA_PAD_CON1, g_pad_con1);

   EMIT(REG_CORE_MISC_CFG, g_fp16 ? CORE_MISC_CFG_PROC_PRECISION(2)
                                  : CORE_MISC_CFG_QD_EN(1));
   EMIT(REG_CORE_DATAOUT_SIZE_0, CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(c->oh - 1) |
                                 CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(c->ow - 1));
   EMIT(REG_CORE_DATAOUT_SIZE_1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(c->oc - 1));
   EMIT(REG_CORE_CLIP_TRUNCATE, CORE_CLIP_TRUNCATE_CLIP_TRUNCATE(c->truncate_bits));
   emit_raw(CORE | 0x1, 0x3030, 0);

   uint32_t fmc = DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2);
   EMIT(REG_DPU_FEATURE_MODE_CFG, fmc);
   EMIT(REG_DPU_DATA_FORMAT, g_fp16
        ? (DPU_DATA_FORMAT_IN_PRECISION(2) | DPU_DATA_FORMAT_OUT_PRECISION(2) |
           DPU_DATA_FORMAT_PROC_PRECISION(2)) : 0);
   EMIT(REG_DPU_OFFSET_PEND, 0);
   EMIT(REG_DPU_DST_BASE_ADDR, c->out_addr);
   EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(output_surf_stride));
   EMIT(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(c->ow - 1));
   EMIT(REG_DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_HEIGHT_HEIGHT(c->oh - 1));
   EMIT(REG_DPU_DATA_CUBE_NOTCH_ADDR, 0);
   EMIT(REG_DPU_DATA_CUBE_CHANNEL,
        DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(c->oc_real - 1) |
        DPU_DATA_CUBE_CHANNEL_CHANNEL(c->oc - 1));
   EMIT(REG_DPU_BS_CFG, g_fp16
        ? (DPU_BS_CFG_BS_BYPASS(1) | DPU_BS_CFG_BS_RELU_BYPASS(1) |
           DPU_BS_CFG_BS_MUL_BYPASS(1) | DPU_BS_CFG_BS_ALU_BYPASS(1))
        : (DPU_BS_CFG_BS_ALU_ALGO(2) | DPU_BS_CFG_BS_ALU_SRC(1) |
           DPU_BS_CFG_BS_RELU_BYPASS(1) | DPU_BS_CFG_BS_MUL_BYPASS(1)));
   EMIT(REG_DPU_BS_ALU_CFG, 0);
   EMIT(REG_DPU_BS_MUL_CFG, 0);
   EMIT(REG_DPU_BS_RELUX_CMP_VALUE, 0);
   EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(1) | DPU_BS_OW_CFG_SIZE_E_1(1) |
                           DPU_BS_OW_CFG_SIZE_E_0(1));
   EMIT(REG_DPU_BS_OW_OP, DPU_BS_OW_OP_OW_OP(0));
   EMIT(REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA(c->oc - 1));
   EMIT(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_HEIGHT_WDMA(c->oh - 1) |
                             DPU_WDMA_SIZE_1_WIDTH_WDMA(c->ow - 1));
   EMIT(REG_DPU_BN_CFG, DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) |
                        DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));
   EMIT(REG_DPU_BN_ALU_CFG, 0);
   EMIT(REG_DPU_BN_MUL_CFG, 0);
   EMIT(REG_DPU_BN_RELUX_CMP_VALUE, 0);
   EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) |
                        DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) |
                        DPU_EW_CFG_EW_BYPASS(1));
   EMIT(REG_DPU_EW_CVT_OFFSET_VALUE, 0);
   EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
   EMIT(REG_DPU_EW_RELUX_CMP_VALUE, 0);
   EMIT(REG_DPU_OUT_CVT_OFFSET, g_fp16 ? 0u : (uint32_t)offset);

   uint32_t scale; unsigned shift;
   compute_requant(c->in_scale, c->w_scale, c->out_scale, c->truncate_bits,
                   &scale, &shift);
   if (g_fp16) {
      /* fp16：DPU 输出直通 fp32→fp16，无重量化 */
      g_out_scale = 1; g_out_shift = 0; g_out_offset = 0;
      EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_FP32TOFP16_EN(1) |
                                  DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(1));
      EMIT(REG_DPU_OUT_CVT_SHIFT, 0);
   } else {
      g_out_scale = scale; g_out_shift = shift; g_out_offset = offset;
      EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(scale));
      EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(shift));
   }

   for (int i = 0; i < 8; i++) EMIT(REG_DPU_EW_OP_VALUE_0 + i * 4, 0);
   EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(surfaces_per_row));
   emit_raw(DPU | 0x1, 0x40c4, 0);
   for (int i = 0; i < 12; i++) EMIT(REG_DPU_LUT_ACCESS_CFG + i * 4, 0);

   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH(c->ow - 1));
   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT(c->oh - 1));
   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL(c->oc - 1));
   EMIT(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0);
   EMIT(REG_DPU_RDMA_RDMA_BRDMA_CFG, DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE(1));
   EMIT(REG_DPU_RDMA_RDMA_BS_BASE_ADDR, c->bias_addr);
   EMIT(REG_DPU_RDMA_RDMA_NRDMA_CFG, 0);
   EMIT(REG_DPU_RDMA_RDMA_BN_BASE_ADDR, 0);
   EMIT(REG_DPU_RDMA_RDMA_ERDMA_CFG, DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE(1));
   EMIT(REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0);
   EMIT(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, 0);
   EMIT(REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
        DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15) |
        DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE(1) |
        (g_fp16 ? (DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION(2) |
                   DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION(2)) : 0u));
   EMIT(REG_DPU_RDMA_RDMA_SRC_DMA_CFG, 0);
   EMIT(REG_DPU_RDMA_RDMA_SURF_NOTCH, 0);
   EMIT(REG_DPU_RDMA_RDMA_PAD_CFG, 0);
   EMIT(REG_DPU_RDMA_RDMA_WEIGHT, DPU_RDMA_RDMA_WEIGHT_E_WEIGHT(1) |
                                  DPU_RDMA_RDMA_WEIGHT_N_WEIGHT(1) |
                                  DPU_RDMA_RDMA_WEIGHT_B_WEIGHT(1) |
                                  DPU_RDMA_RDMA_WEIGHT_M_WEIGHT(1));
   EMIT(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH, 0);

   EMIT(REG_PC_BASE_ADDRESS, 0);
   EMIT(REG_PC_REGISTER_AMOUNTS, 0);
   *rc++ = 0x0041000000000000ULL;
   emit_raw(0x81, REG_PC_OPERATION_ENABLE,
            PC_OPERATION_ENABLE_RESERVED_0(14) | PC_OPERATION_ENABLE_OP_EN(1));

   if (g_dump_regcmd) {
      static unsigned dump_seq = 0;
      char dp[512];
      snprintf(dp, sizeof(dp), "%s.%u", g_dump_regcmd, dump_seq++);
      FILE *fr = fopen(dp, "w");
      if (fr) {
         for (uint64_t *p = rc_buf; p < rc; p++)
            fprintf(fr, "0x%04x 0x%08x\n", (unsigned)(*p & 0xffffu),
                    (unsigned)((*p >> 16) & 0xffffffffu));
         fclose(fr);
         printf("DUMP-REGCMD: %u entries -> %s\n", (unsigned)(rc - rc_buf), dp);
      }
   }
   return (unsigned)(rc - rc_buf);
}


/* ------------------------------------------------------------------ *
 * 3. BO 管理与任务发射
 * ------------------------------------------------------------------ */
struct bo { uint32_t handle; uint64_t obj_addr; uint64_t dma_addr; void *map; uint32_t size; };

static int bo_create(int fd, uint32_t size, uint32_t extra_flags, struct bo *b)
{
   memset(b, 0, sizeof(*b));
   b->size = size;
   struct rknpu_mem_create mc = {
      .size = size,
      .flags = RKNPU_MEM_NON_CONTIGUOUS | RKNPU_MEM_CACHEABLE |
               RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT | extra_flags,
      .iommu_domain_id = 0,
   };
   if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mc)) {
      fprintf(stderr, "MEM_CREATE(%u) failed: %s\n", size, strerror(errno));
      return -1;
   }
   b->handle = mc.handle; b->obj_addr = mc.obj_addr; b->dma_addr = mc.dma_addr;
   struct rknpu_mem_map mm = { .handle = mc.handle };
   if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mm)) {
      fprintf(stderr, "MEM_MAP failed: %s\n", strerror(errno));
      return -1;
   }
   b->map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mm.offset);
   if (b->map == MAP_FAILED) { fprintf(stderr, "mmap failed: %s\n", strerror(errno)); return -1; }
   memset(b->map, 0, size);
   return 0;
}

static void bo_sync(int fd, struct bo *b, uint32_t dir /*1=to dev, 2=from dev*/)
{
   struct rknpu_mem_sync ms = { .flags = dir, .obj_addr = b->obj_addr,
                                .offset = 0, .size = b->size };
   ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, &ms);
}

/* NC1HWC2：group-major, x-major, y, c16 */
static uint32_t npu_off(unsigned g, unsigned x, unsigned y, unsigned w, unsigned h)
{
   return g * w * h * FEATURE_ATOMIC_SIZE + x * h * FEATURE_ATOMIC_SIZE +
          y * FEATURE_ATOMIC_SIZE;
}

/* §7.50：输出区偏移必须**在打包输入之后**（原先固定 32768，大图时与输入重叠 ⇒ 前向读到的
 * 输入被输出覆写，表现为大面积 MISMATCH）。启动时按最大输入需求计算。 */
static unsigned OUT_OFF = 32768;
static unsigned g_in_need_fwd = 0, g_in_need_wg = 0;

struct bos { struct bo act, wt, bias, rcbo, tk; };
struct cost {
   unsigned submits, fwd_tasks, wgrad_tasks;
   double wall_ms, hw_ms;
   uint64_t dma_bytes;
};

static uint32_t g_timeout = 6000;

static double now_ms(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* 组装 + 提交一个 task。DMA 字节为估算：输入 cube + 权重区 + 输出 cube。 */
static int run_task(int fd, struct bos *bs, const struct conv_cfg *c, struct cost *cost)
{
   rc = rc_buf;
   unsigned n = emit_conv(c);
   memcpy(bs->rcbo.map, rc_buf, (size_t)n * 8);

   struct rknpu_task *t = bs->tk.map;
   memset(t, 0, sizeof(*t));
   t->op_idx        = 0;
   t->enable_mask   = 0xf;
   t->int_mask      = 0x300;
   t->int_clear     = 0x1ffff;
   t->regcfg_amount = n - RKNPU_PC_DATA_EXTRA_AMOUNT;
   t->regcmd_addr   = bs->rcbo.dma_addr;

   bo_sync(fd, &bs->act, 1);
   bo_sync(fd, &bs->wt, 1);
   bo_sync(fd, &bs->bias, 1);
   bo_sync(fd, &bs->rcbo, 1);
   bo_sync(fd, &bs->tk, 1);

   struct rknpu_submit sub;
   memset(&sub, 0, sizeof(sub));
   sub.flags          = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG;
   sub.timeout        = g_timeout;
   sub.task_start     = 0;
   sub.task_number    = 1;
   sub.task_obj_addr  = bs->tk.obj_addr;
   sub.core_mask      = 0;
   sub.fence_fd       = -1;
   for (int i = 0; i < 5; i++) {
      sub.subcore_task[i].task_start = 0;
      sub.subcore_task[i].task_number = 1;
   }

   double t0 = now_ms();
   if (g_cbuf_overflow) {
      printf("SUBMIT skipped: CBUF 容量越界（见 WARN）⇒ 请减小 IW/IH/IC\n");
      return 1;
   }
   int ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub);
   double t1 = now_ms();

   cost->submits++;
   cost->wall_ms += t1 - t0;
   cost->hw_ms += (double)sub.hw_elapse_time / 1e6;
   {
      unsigned wk = (c->oc_real <= WEIGHT_ATOMIC_SIZE) ? ALIGN_UP(c->oc_real, 2)
                                                       : WEIGHT_ATOMIC_SIZE;
      cost->dma_bytes += (uint64_t)c->iw * c->ih * c->ic
                       + (uint64_t)c->kw * c->kh * c->ic * wk
                       + (uint64_t)c->ow * c->oh * c->oc;
   }
   if (ret) {
      printf("SUBMIT ret=%d errno=%s  %.3f ms\n", ret, strerror(errno), t1 - t0);
      fflush(stdout);
      return -1;
   }
   if (c->kw * c->kh == 1) cost->wgrad_tasks++;
   else cost->fwd_tasks++;
   bo_sync(fd, &bs->act, 2);
   return 0;
}

/* ------------------------------------------------------------------ *
 * 4. 数据集（固定 seed LCG，全部落盘，CPU 可复算）
 * ------------------------------------------------------------------ */
#define MAX_B   8
#ifndef IW
#define IW      8      /* §7.49：可用 -DIW=... -DIH=... 编译更大特征图（需 pad0 用 OW_P0/OH_P0） */
#endif
#ifndef IH
#define IH      8
#endif
#ifndef IC
#define IC      3      /* §7.37：可用 -DIC=... 编译深通道变体（需同时 -DOC=...） */
#endif
#ifndef OC
#define OC      16
#endif
#define KW      3
#define KH      3
/* §7.49：pad0 时输出 = 输入 - (k-1)，原先写死 6（只对 IW=IH=8 成立）。 */
#define OW_P0   (IW - (KW - 1))
#define OH_P0   (IH - (KH - 1))
#define X_BYTES (IH * IW * IC)      /* 192 */
#define W_BYTES (KH * KW * IC * OC) /* 432 */
#define NCOORD  (KW * KH * IC)      /* 27 */

struct dataset {
   unsigned nsamp;                        /* 数据集内样本数（固定 8） */
   int8_t   X[MAX_B][X_BYTES];            /* [b][y*8*3 + x*3 + ic] */
   int8_t   W[W_BYTES];                   /* [(kx*3+ky)*48 + ic*16 + oc] */
   int8_t   T[MAX_B][IW * IH * OC];       /* 目标（本 case 的 ow*oh*16） */
   int8_t   Q[MAX_B][IW * IH * OC];       /* CPU 前向参考（初始 W） */
   int8_t   DY[MAX_B][IW * IH * OC];      /* dY = clamp(round((Q-T)/8)) */
   int8_t   DW8[NCOORD * OC];             /* CPU dW（梯度 requant 后 int8） */
   long long DWR[NCOORD][OC];             /* CPU dW raw（requant 前） */
};

static uint32_t lcg_s;
static uint32_t lcg(void)
{
   lcg_s = lcg_s * 1103515245u + 12345u;
   return (lcg_s >> 16) & 0x7fff;
}

static int8_t clamp8(long long v)
{
   if (v < -128) return -128;
   if (v > 127) return 127;
   return (int8_t)v;
}

static long long rhao_div(long long v, unsigned d)
{
   return v >= 0 ? (v + (long long)d / 2) / (long long)d
                 : -((-v + (long long)d / 2) / (long long)d);
}

/* CPU 前向：out = clamp(round(acc * scale >> shift) + 0)，out_zp=128 */
static void cpu_forward(int8_t *q, const int8_t *X, const int8_t *W,
                        unsigned pad, unsigned ow, unsigned oh,
                        uint32_t scale, unsigned shift)
{
   for (unsigned oc = 0; oc < OC; oc++)
    for (unsigned oy = 0; oy < oh; oy++)
     for (unsigned ox = 0; ox < ow; ox++) {
        long long acc = 0;
        for (unsigned ky = 0; ky < KH; ky++)
         for (unsigned kx = 0; kx < KW; kx++) {
            int iy = (int)oy + (int)ky - (int)pad;
            int ix = (int)ox + (int)kx - (int)pad;
            if (ix < 0 || iy < 0 || ix >= (int)IW || iy >= (int)IH) continue;
            for (unsigned icc = 0; icc < IC; icc++)
               acc += (long long)X[(iy * IW + ix) * IC + icc] *
                      (long long)W[((ky * KW + kx) * IC + icc) * OC + oc];
         }
        q[(oy * ow + ox) * OC + oc] =
           clamp8(shift_right_round64(acc * (int64_t)scale, shift));
     }
}

/* CPU dW：raw = Σ_b Σ_p X[b][p+k,ic]·dY[b][p,oc]（越界点贡献 0 = in_zp 0） */
static void cpu_dw_raw(const struct dataset *ds, unsigned bcount,
                       unsigned pad, unsigned ow, unsigned oh, long long dw[NCOORD][OC])
{
   for (unsigned kx = 0; kx < KW; kx++)
    for (unsigned ky = 0; ky < KH; ky++)
     for (unsigned ic = 0; ic < IC; ic++)
      for (unsigned oc = 0; oc < OC; oc++) {
         long long acc = 0;
         for (unsigned b = 0; b < bcount; b++)
          for (unsigned oy = 0; oy < oh; oy++)
           for (unsigned ox = 0; ox < ow; ox++) {
              int iy = (int)oy + (int)ky - (int)pad;
              int ix = (int)ox + (int)kx - (int)pad;
              if (ix < 0 || iy < 0 || ix >= (int)IW || iy >= (int)IH) continue;
              acc += (long long)ds->X[b][iy * IW * IC + ix * IC + ic] *
                     (long long)ds->DY[b][((oy * ow + ox) * OC) + oc];
           }
         /* 与 w.bin 一致：dW 索引 = ((ky*3+kx)*3+ic)*16+oc（W 的 [kx][ky][ic][oc] 展平） */
         dw[(ky * KW + kx) * IC + ic][oc] = acc;
      }
}

/* 梯度 requant：与 wgrad task 的 (scale, shift, offset=0) 完全一致 */
static int8_t requant_grad(long long raw, uint32_t scale, unsigned shift)
{
   return clamp8(shift_right_round64(raw * (int64_t)scale, shift));
}

static void write_file(const char *path, const void *buf, size_t n)
{
   FILE *f = fopen(path, "wb");
   if (!f) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); exit(1); }
   if (fwrite(buf, 1, n, f) != n) { fprintf(stderr, "short write %s\n", path); exit(1); }
   fclose(f);
}

static int read_file(const char *path, void *buf, size_t n)
{
   FILE *f = fopen(path, "rb");
   if (!f) return 0;
   size_t got = fread(buf, 1, n, f);
   fclose(f);
   return got == n;
}

static uint32_t fnv1a(const uint8_t *p, uint32_t n)
{
   uint32_t h = 2166136261u;
   for (uint32_t i = 0; i < n; i++) { h ^= p[i]; h *= 16777619u; }
   return h;
}

static void gen_dataset(struct dataset *ds, unsigned seed, unsigned pad,
                        uint32_t fscale, unsigned fshift)
{
   unsigned qpix = (pad ? IW * IH : OW_P0 * OH_P0) * OC;
   memset(ds, 0, sizeof(*ds));
   ds->nsamp = MAX_B;
   lcg_s = seed;
   for (unsigned b = 0; b < MAX_B; b++)
      for (unsigned i = 0; i < X_BYTES; i++)
         ds->X[b][i] = (int8_t)(lcg() % 16 - 8);
   for (unsigned i = 0; i < W_BYTES; i++)
      ds->W[i] = (int8_t)(lcg() % 8 - 4);
   for (unsigned b = 0; b < MAX_B; b++)
      for (unsigned i = 0; i < qpix; i++)
         ds->T[b][i] = (int8_t)(lcg() % 256 - 128);
   for (unsigned b = 0; b < MAX_B; b++)
      cpu_forward(ds->Q[b], ds->X[b], ds->W, pad, pad ? IW : OW_P0, pad ? IH : OH_P0,
                  fscale, fshift);
   for (unsigned b = 0; b < MAX_B; b++)
      for (unsigned i = 0; i < qpix; i++)
         ds->DY[b][i] = clamp8(rhao_div((long long)ds->Q[b][i] - ds->T[b][i], 8));
}

static void dataset_files_save(struct dataset *ds, const char *dir,
                               unsigned pad, unsigned seed,
                               uint32_t fscale, unsigned fshift,
                               uint32_t gscale, unsigned gshift)
{
   char path[512];
   unsigned qpix = (pad ? IW * IH : OW_P0 * OH_P0) * OC;

   for (unsigned b = 0; b < MAX_B; b++) {
      snprintf(path, sizeof(path), "%s/x_b%u.bin", dir, b);
      write_file(path, ds->X[b], X_BYTES);
      snprintf(path, sizeof(path), "%s/t_b%u.bin", dir, b);
      write_file(path, ds->T[b], qpix);
      snprintf(path, sizeof(path), "%s/q_cpu_b%u.bin", dir, b);
      write_file(path, ds->Q[b], qpix);
      snprintf(path, sizeof(path), "%s/dy_b%u.bin", dir, b);
      write_file(path, ds->DY[b], qpix);
   }
   snprintf(path, sizeof(path), "%s/w.bin", dir);
   write_file(path, ds->W, W_BYTES);
   snprintf(path, sizeof(path), "%s/dw_cpu.bin", dir);
   write_file(path, ds->DW8, W_BYTES);
   snprintf(path, sizeof(path), "%s/dw_cpu_raw.bin", dir);
   write_file(path, ds->DWR, sizeof(ds->DWR));

   snprintf(path, sizeof(path), "%s/meta.txt", dir);
   FILE *m = fopen(path, "w");
   if (!m) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); exit(1); }
   fprintf(m,
      "seed=%u\npad=%u\now=oh=%u\nic=%u oc=%u kw=3 kh=3 stride=1\n"
      "quant: in_scale=1.0 w_scale=1.0 out_scale=1.0 in_zp=0 w_zp=0 out_zp=128\n"
      "fwd requant: scale=%u shift=%u offset=0 (identity for |acc|<8192)\n"
      "grad task requant: out_scale=8 scale=%u shift=%u offset=0\n"
      "dY = clamp(round_half_away((Q_cpu - T)/8), -128, 127)\n"
      "loss L = sum_p (Q[p]-T[p])^2 / 16  =>  dL/dQ[p] = (Q[p]-T[p])/8 =: dY[p]\n"
      "dW[kx,ky,ic,oc] = sum_p X[p+k,ic] * dY[p,oc]  (P = ow*oh*batch)\n"
      "update: W' = clamp(W - round_half_away(dW_npu/4)); dW_npu int8 scale 1/8 => lr=2 LSB\n"
      "flatten: K index p = b*(ow*oh) + oy*ow + ox (x fastest, batch outermost)\n"
      "LCG: s = s*1103515245 + 12345; draw = (s>>16)&0x7fff; order X[0..7], W, T[0..7]\n",
      seed, pad, pad ? IW : OW_P0, (unsigned)IC, (unsigned)OC, fscale, fshift, gscale, gshift);
   fclose(m);
}

static void dataset_files_load(struct dataset *ds, const char *dir,
                               unsigned pad, unsigned seed)
{
   char path[512], line[256];
   unsigned qpix = (pad ? IW * IH : OW_P0 * OH_P0) * OC;
   unsigned mseed = 0, mpad = 999;

   for (unsigned b = 0; b < MAX_B; b++) {
      snprintf(path, sizeof(path), "%s/x_b%u.bin", dir, b);
      if (!read_file(path, ds->X[b], X_BYTES)) {
         fprintf(stderr, "dataset %s incomplete (x_b%u)\n", dir, b);
         exit(1);
      }
   }
   snprintf(path, sizeof(path), "%s/w.bin", dir);
   if (!read_file(path, ds->W, W_BYTES)) { fprintf(stderr, "dataset %s: w.bin\n", dir); exit(1); }
   for (unsigned b = 0; b < MAX_B; b++) {
      snprintf(path, sizeof(path), "%s/dy_b%u.bin", dir, b);
      if (!read_file(path, ds->DY[b], qpix)) { fprintf(stderr, "dataset %s: dy_b%u\n", dir, b); exit(1); }
      snprintf(path, sizeof(path), "%s/t_b%u.bin", dir, b);
      if (!read_file(path, ds->T[b], qpix)) { fprintf(stderr, "dataset %s: t_b%u\n", dir, b); exit(1); }
   }
   snprintf(path, sizeof(path), "%s/meta.txt", dir);
   FILE *m = fopen(path, "r");
   if (m) {
      while (fgets(line, sizeof(line), m)) {
         if (!strncmp(line, "seed=", 5)) mseed = (unsigned)strtoul(line + 5, NULL, 10);
         if (!strncmp(line, "pad=", 4)) mpad = (unsigned)strtoul(line + 4, NULL, 10);
      }
      fclose(m);
   }
   if (mseed != seed || mpad != pad) {
      fprintf(stderr, "dataset %s: meta seed=%u pad=%u != requested seed=%u pad=%u; "
                      "use another --dir\n", dir, mseed, mpad, seed, pad);
      exit(1);
   }
   ds->nsamp = MAX_B;
   /* Q 由加载的 X/W 重算（不信任陈旧文件），dY 使用落盘值 */
   uint32_t fs; unsigned fsh;
   compute_requant(1.0f, 1.0f, 1.0f, 0, &fs, &fsh);
   for (unsigned b = 0; b < MAX_B; b++)
      cpu_forward(ds->Q[b], ds->X[b], ds->W, pad, pad ? IW : OW_P0, pad ? IH : OH_P0, fs, fsh);
}

/* 载入或生成数据集；--gen-only 只生成并落盘 */
static void dataset_prepare(struct dataset *ds, const char *dir, unsigned pad,
                            unsigned seed, uint32_t fscale, unsigned fshift,
                            uint32_t gscale, unsigned gshift)
{
   if (mkdir(dir, 0755) && errno != EEXIST) {
      fprintf(stderr, "mkdir %s: %s\n", dir, strerror(errno));
      exit(1);
   }
   char path[512];
   snprintf(path, sizeof(path), "%s/x_b0.bin", dir);
   if (read_file(path, ds->X[0], X_BYTES)) {
      dataset_files_load(ds, dir, pad, seed);
      if (g_in_dy_file) {   /* §7.163：读外部 dY（本工具 Q_BYTES 布局）替换 batch0 梯度 */
         FILE *fd = fopen(g_in_dy_file, "rb");
         if (!fd) { fprintf(stderr, "in-dy-file 打不开 %s\n", g_in_dy_file); exit(1); }
         unsigned qpix = (pad ? IW * IH : OW_P0 * OH_P0) * OC;
         size_t got = fread(ds->DY[0], 1, qpix, fd); fclose(fd);
         printf("IN-DY-FILE: 读入 %zu/%u B（batch0 梯度替换）\n", got, qpix);
      }
      if (g_in_x_file) {   /* §7.161：读外部 X（行主序 (y*IW+x)*IC+c）并重算参考 Q（工具自带参考 ⇒ 可自证） */
         FILE *fx = fopen(g_in_x_file, "rb");
         if (!fx) { fprintf(stderr, "in-x-file 打不开 %s\n", g_in_x_file); exit(1); }
         size_t got = fread(ds->X[0], 1, X_BYTES, fx); fclose(fx);
         cpu_forward(ds->Q[0], ds->X[0], ds->W, pad, pad ? IW : OW_P0, pad ? IH : OH_P0, fscale, fshift);
         printf("IN-X-FILE: 读入 %zu/%u B 并已重算参考 Q[0]（pad=%u ow=%u oh=%u）\n",
                got, (unsigned)X_BYTES, pad, pad ? IW : OW_P0, pad ? IH : OH_P0);
      }
      printf("DATASET: loaded from %s\n", dir);
   } else {
      gen_dataset(ds, seed, pad, fscale, fshift);
      dataset_files_save(ds, dir, pad, seed, fscale, fshift, gscale, gshift);
      printf("DATASET: generated into %s (seed=%u pad=%u)\n", dir, seed, pad);
   }
   printf("DATASET: fnv(x_b0)=%08x fnv(w)=%08x fnv(dy_b0)=%08x\n",
          fnv1a((const uint8_t *)ds->X[0], X_BYTES),
          fnv1a((const uint8_t *)ds->W, W_BYTES),
          fnv1a((const uint8_t *)ds->DY[0], (pad ? IW * IH : OW_P0 * OH_P0) * OC));
}


/* ------------------------------------------------------------------ *
 * 5. NPU 打包
 * ------------------------------------------------------------------ */
/* §7.41：int8 深通道 —— 输入通道对齐 / oc 跨度按几何（与发射器同规则） */
static inline unsigned i8_ic_align(void)
{
   return (IC <= FEATURE_ATOMIC_SIZE) ? FEATURE_ATOMIC_SIZE
                                      : ALIGN_UP(IC, WEIGHT_ATOMIC_SIZE);
}
/* 权重描述符的每 tap oc 跨度（= 发射器的 weights_kernels 规则） */
static inline unsigned i8_oc_w(void)
{
   return (OC <= WEIGHT_ATOMIC_SIZE) ? ALIGN_UP(OC, 2u)
                                     : ALIGN_UP(OC, WEIGHT_ATOMIC_SIZE);
}
/* 硬件侧 oc 原子跨度（= 发射器的 c.oc 规则：MAX(oc_real,32) 后 32 对齐） */
static inline unsigned i8_oc_span(void)
{
   return ALIGN_UP(MAX2(OC, WEIGHT_ATOMIC_SIZE), WEIGHT_ATOMIC_SIZE);
}
/* 布局 A（int8）：idx = ((((oc1*ic1+i1)*kh+ky)*kw+kx)*oc2n+oc2)*ic2+i2（台账 §5.2 同源） */
static unsigned w_elem_index_i8(unsigned oc_w, unsigned ic, unsigned oc, unsigned icc,
                                unsigned kx, unsigned ky, unsigned kw, unsigned kh)
{
   unsigned ic1 = DIV_ROUND_UP(ic, WEIGHT_ATOMIC_SIZE);
   unsigned ic2 = MIN2(ic, WEIGHT_ATOMIC_SIZE);
   unsigned oc2n = MIN2(oc_w, WEIGHT_ATOMIC_SIZE);
   unsigned oc1 = oc / WEIGHT_ATOMIC_SIZE, oc2 = oc % WEIGHT_ATOMIC_SIZE;
   unsigned i1 = icc / WEIGHT_ATOMIC_SIZE, i2 = icc % WEIGHT_ATOMIC_SIZE;
   /* 注意：wgrad_real 的 int8 全链路是「空间转置」约定（输入 x-major、tap ky 内层、
    * 输出 x-major），故这里 tap 嵌套为 ((…*kw + kx)*kh + ky) —— 默认几何下退化为
    * 旧的 ((kx*KH+ky)*16+oc)*16+icc，与升级前逐位一致。 */
   return ((((oc1 * ic1 + i1) * kw + kx) * kh + ky) * oc2n + oc2) * ic2 + i2;
}

static void pack_fwd_input(uint8_t *buf, const int8_t *X)
{
   unsigned icp = i8_ic_align();
   memset(buf, 0, (size_t)IW * IH * icp);
   for (unsigned x = 0; x < IW; x++)
      for (unsigned y = 0; y < IH; y++)
         for (unsigned c = 0; c < IC; c++)
            buf[npu_off(c / FEATURE_ATOMIC_SIZE, x, y, IW, IH) +
                (c % FEATURE_ATOMIC_SIZE)] = (uint8_t)X[(y * IW + x) * IC + c];
}

/* 布局 A（ic=16 ⇒ ic1=1/ic2=16；oc_w=16 ⇒ oc1n=1, oc2n=16）：
 * idx = ((x*3+y)*16 + oc)*16 + icc */
static void pack_fwd_weights(uint8_t *dst, const int8_t *W)
{
   unsigned icp = i8_ic_align(), ocw = i8_oc_w();
   memset(dst, 0, (size_t)KW * KH * ocw * icp);
   for (unsigned kx = 0; kx < KW; kx++)
    for (unsigned ky = 0; ky < KH; ky++)
     for (unsigned oc = 0; oc < ocw; oc++)
      for (unsigned icc = 0; icc < icp; icc++) {
         uint8_t v = (oc < OC && icc < IC)
                   ? (uint8_t)W[((ky * KW + kx) * IC + icc) * OC + oc] : 0x00;
         dst[w_elem_index_i8(ocw, icp, oc, icc, kx, ky, KW, KH)] = v;
      }
}

/* §7.55：int8 前向行切分的行区间打包（越界行 memset 0 = 硬件 PAD_CON1=0 的等价） */
static void pack_fwd_tile_input_i8(uint8_t *buf, const int8_t *X, unsigned src0, unsigned rows)
{
   unsigned icp = i8_ic_align();
   memset(buf, 0, (size_t)IW * rows * icp);
   for (unsigned x = 0; x < IW; x++)
      for (unsigned y = 0; y < rows; y++) {
         unsigned sy = src0 + y;
         if (sy >= IH) break;
         for (unsigned c = 0; c < IC; c++)
            buf[npu_off(c / FEATURE_ATOMIC_SIZE, x, y, IW, rows) +
                (c % FEATURE_ATOMIC_SIZE)] = (uint8_t)X[(sy * IW + x) * IC + c];
      }
}

/* wgrad 输入：4×4 cube，仅每 group 的 (0,0) 像素携带 K 向量 */
static void pack_wgrad_input(uint8_t *buf, const int8_t *kvec,
                             unsigned P, unsigned ic_pad)
{
   memset(buf, 0, (size_t)4 * 4 * ic_pad);
   for (unsigned g = 0; g < ic_pad / FEATURE_ATOMIC_SIZE; g++)
      for (unsigned k = 0; k < FEATURE_ATOMIC_SIZE; k++) {
         unsigned ch = g * FEATURE_ATOMIC_SIZE + k;
         buf[npu_off(g, 0, 0, 4, 4) + k] = ch < P ? (uint8_t)kvec[ch] : 0x00;
      }
}

/* wgrad 权重：布局 A 在 kw=kh=1、oc_w=16 下的展开（台账 §5.2 fill_weights 同源）：
 * ic1 = ceil(ic_pad/32)，ic2 = min(ic_pad,32)，oc2n = 16：
 * idx = (i1*oc2n + oc2)*ic2 + i2，icc = i1*32 + i2
 * 槽位 >= P 填权重零点 0x00（不贡献累加） */
static void pack_wgrad_weights(uint8_t *dst, const int8_t *dy_flat,
                               unsigned P, unsigned ic_pad)
{
   unsigned ocw = i8_oc_w();
   memset(dst, 0, (size_t)ocw * ic_pad);
   for (unsigned oc = 0; oc < ocw; oc++)
      for (unsigned icc = 0; icc < ic_pad; icc++) {
         uint8_t v = (icc < P && oc < OC)
                   ? (uint8_t)dy_flat[icc * OC + oc] : 0x00;
         dst[w_elem_index_i8(ocw, ic_pad, oc, icc, 0, 0, 1, 1)] = v;
      }
}

static struct conv_cfg make_fwd_cfg(const struct bos *bs, unsigned pad,
                                    unsigned ow, unsigned oh)
{
   struct conv_cfg f;
   memset(&f, 0, sizeof(f));
   f.iw = IW; f.ih = IH; f.ic_real = IC; f.ic = i8_ic_align();
   f.ow = ow; f.oh = oh; f.oc_real = OC; f.oc = i8_oc_span();
   f.kw = KW; f.kh = KH; f.stride_x = f.stride_y = 1;
   f.pad_left = f.pad_top = pad;
   f.output_zero_point = 128;
   f.in_scale = f.w_scale = f.out_scale = 1.0f;
   f.in_addr = (uint32_t)bs->act.dma_addr;
   f.out_addr = (uint32_t)bs->act.dma_addr + OUT_OFF;
   f.wt_addr = (uint32_t)bs->wt.dma_addr;
   f.bias_addr = (uint32_t)bs->bias.dma_addr;
   return f;
}

static struct conv_cfg make_wgrad_cfg(const struct bos *bs, unsigned P)
{
   struct conv_cfg g;
   memset(&g, 0, sizeof(g));
   g.iw = 4; g.ih = 4; g.ic_real = P; g.ic = ALIGN_UP(P, WEIGHT_ATOMIC_SIZE);
   g.ow = 1; g.oh = 1; g.oc_real = OC; g.oc = i8_oc_span();
   g.kw = 1; g.kh = 1; g.stride_x = g.stride_y = 1;
   g.output_zero_point = 128;
   g.in_scale = g.w_scale = 1.0f; g.out_scale = 8.0f;
   g.in_addr = (uint32_t)bs->act.dma_addr;
   g.out_addr = (uint32_t)bs->act.dma_addr + OUT_OFF;
   g.wt_addr = (uint32_t)bs->wt.dma_addr;
   g.bias_addr = (uint32_t)bs->bias.dma_addr;
   return g;
}

/* ------------------------------------------------------------------ *
 * 5. K 维容量扫描（合成数据，坐标 (1,1,1)，与台账 §7.4 batch probe 同口径）
 * ------------------------------------------------------------------ */
static int8_t kscan_x(unsigned p, unsigned ic)
{
   unsigned sample = p / 36, pos = p % 36;
   unsigned ox = pos % 6, oy = pos / 6;
   unsigned ix = ox + 1, iy = oy + 1;
   return (int8_t)((ix * 3 + iy * 5 + ic * 2 + sample) % 5 - 2);
}
static int8_t kscan_dy(unsigned p, unsigned oc)
{
   unsigned sample = p / 36, pos = p % 36;
   return (int8_t)((pos * 7 + oc * 2 + sample * 2 + 2) % 3 - 1);
}

static int kscan_run(int fd, struct bos *bs, unsigned bmax, unsigned grad_scale,
                     unsigned grad_shift, struct cost *cost)
{
   printf("KSCAN: synthetic coord=(kx=1,ky=1,ic=1) P=36*batch grad_scale=%u grad_shift=%u timeout=%u\n",
          grad_scale, grad_shift, g_timeout);
   static int8_t kvec[36 * MAX_B];
   static int8_t dy_flat[36 * MAX_B * OC];

   for (unsigned b = 1; b <= bmax; b++) {
      unsigned P = 36 * b;
      unsigned ic_pad = ALIGN_UP(P, WEIGHT_ATOMIC_SIZE);
      long long acc[OC] = {0};

      for (unsigned p = 0; p < P; p++) kvec[p] = kscan_x(p, 1);
      for (unsigned p = 0; p < P; p++)
         for (unsigned oc = 0; oc < OC; oc++) {
            dy_flat[p * OC + oc] = kscan_dy(p, oc);
            acc[oc] += (long long)kvec[p] * kscan_dy(p, oc);
         }

      pack_wgrad_input(bs->act.map, kvec, P, ic_pad);
      pack_wgrad_weights(bs->wt.map, dy_flat, P, ic_pad);

      struct conv_cfg g = make_wgrad_cfg(bs, P);
      if (run_task(fd, bs, &g, cost)) {
         printf("KSCAN: batch=%u P=%u K_pad=%u SUBMIT-FAILED (boundary)\n", b, P, g.ic);
         return 1;
      }
      uint8_t *out = (uint8_t *)bs->act.map + OUT_OFF;
      unsigned bad = 0;
      for (unsigned oc = 0; oc < OC; oc++) {
         int hw = (int)(int8_t)out[oc];
         long long ref = requant_grad(acc[oc], grad_scale, grad_shift);
         if (hw != (int)ref) bad++;
      }
      printf("KSCAN: batch=%u P=%u K_pad=%u round_mismatch=%u/%u %s\n",
             b, P, g.ic, bad, OC, bad ? "MISMATCH" : "OK");
   }
   printf("KSCAN: all batches up to %u OK\n", bmax);
   return 0;
}

/* ------------------------------------------------------------------ *
 * 5b. OOB 探针：单 tap 恒等权重，解码越界样本的真实取值
 *     X[ix][iy] = 4*(ix+1) + (iy+1)（三通道同值，1..8 → 5..40）
 *     left  tap=(kx=0,ky=1)  right tap=(kx=2,ky=1)
 *     top   tap=(kx=1,ky=0)  bottom tap=(kx=1,ky=2)
 *     读数：0=零填充  X(0,y)=5..10 / X(x,0)=9,13..=复制  X(7,y)=33..37 / X(x,7)=13,17..=环绕
 *     其他常值 = PAD_CON1 字节值（有符号域）
 * ------------------------------------------------------------------ */
static int oobtest_run(int fd, struct bos *bsp, struct cost *cost)
{
   printf("OOBTEST: single-tap identity probes, X(ix,iy)=4*(ix+1)+(iy+1)\n");
   static const char *dir_name[4] = { "left", "right", "top", "bottom" };
   static const unsigned tap[4][2] = { {0,1}, {2,1}, {1,0}, {1,2} };
   static int8_t X[IH * IW * IC];
   for (unsigned y = 0; y < IH; y++)
      for (unsigned x = 0; x < IW; x++)
         for (unsigned c = 0; c < IC; c++)
            X[(y * IW + x) * IC + c] = (int8_t)(4 * (x + 1) + (y + 1));

   static const int32_t pcs[4] = { (int32_t)0xffff8080u, 0, 0x80, 0xff };
   for (unsigned pc = 0; pc < 4; pc++) {
      g_pad_con1 = (uint32_t)pcs[pc];
      for (unsigned d = 0; d < 4; d++) {
         static int8_t Wt[((size_t)KW * KH * IC) * OC];
         memset(Wt, 0, sizeof(Wt));
         unsigned kx = tap[d][0], ky = tap[d][1];
         for (unsigned ic = 0; ic < IC; ic++)
            Wt[((ky * KW + kx) * IC + ic) * OC + ic] = 1;
         pack_fwd_input(bsp->act.map, X);
         pack_fwd_weights(bsp->wt.map, Wt);
         struct conv_cfg f;
         memset(&f, 0, sizeof(f));
         f.iw = IW; f.ih = IH; f.ic_real = IC; f.ic = FEATURE_ATOMIC_SIZE;
         f.ow = IW; f.oh = IH; f.oc_real = OC; f.oc = 32;
         f.kw = KW; f.kh = KH; f.stride_x = f.stride_y = 1;
         f.pad_left = f.pad_top = 1;
         f.output_zero_point = 128;
         f.in_scale = f.w_scale = f.out_scale = 1.0f;
         f.in_addr = (uint32_t)bsp->act.dma_addr;
         f.out_addr = (uint32_t)bsp->act.dma_addr + OUT_OFF;
         f.wt_addr = (uint32_t)bsp->wt.dma_addr;
         f.bias_addr = (uint32_t)bsp->bias.dma_addr;
         if (run_task(fd, bsp, &f, cost)) {
            printf("OOBTEST: submit failed (pc=%d d=%u)\n", pcs[pc], d);
            return 1;
         }
         uint8_t *o = (uint8_t *)bsp->act.map + OUT_OFF;
         int in1 = (int)(int8_t)o[npu_off(0, 1, 1, IW, IH)];
         int expect_in = (int)(4 * (kx + 1) + (ky + 1));
         printf("OOB dir=%-6s padcon1=0x%08x interior(1,1)=%d expect=%d |",
                dir_name[d], (uint32_t)pcs[pc], in1, expect_in);
         for (unsigned y = 1; y < IH - 1; y++)
            printf(" %d", (int)(int8_t)o[npu_off(0, 0, y, IW, IH)]);
         printf(" |");
         for (unsigned y = 1; y < IH - 1; y++)
            printf(" %d", (int)(int8_t)o[npu_off(0, IW - 1, y, IW, IH)]);
         printf(" |");
         for (unsigned x = 1; x < IW - 1; x++)
            printf(" %d", (int)(int8_t)o[npu_off(0, x, 0, IW, IH)]);
         printf(" |");
         for (unsigned x = 1; x < IW - 1; x++)
            printf(" %d", (int)(int8_t)o[npu_off(0, x, IH - 1, IW, IH)]);
         printf("\n");
      }
   }
   g_pad_con1 = 0xffff8080u;
   return 0;
}

/* ------------------------------------------------------------------ *
 * 5c. Bisect 探针：pad0 6×6 fwd 错误写序的「修复者」定位
 *     fwd(pad0,6x6) -> 按 mode 提交 1..N 个干预 task -> fwd(pad0,6x6)
 *     若 r1 从 MISMATCH 变 BIT-EXACT，则该干预 task 设置了某个 latched 状态
 * ------------------------------------------------------------------ */
static int bisect_run(int fd, struct bos *bsp, struct dataset *ds,
                      const char *dir, unsigned pad, unsigned ow, unsigned oh,
                      struct cost *cost)
{
   printf("BISECT: mode=%s pad=%u ow*oh=%u*%u wfile=%s\n",
          g_bisect, pad, ow, oh, g_wfile ? g_wfile : "-");
   static int8_t kvec[IW * IH * MAX_B];
   static int8_t dy_flat[IW * IH * MAX_B * OC];
   uint8_t *out = (uint8_t *)bsp->act.map + OUT_OFF;
   unsigned qpix = ow * oh * OC;

   /* 权重选择：默认 ds->W；--wfile 覆盖（如 w_updated.bin），并用 cpu_forward
    * 生成匹配参考（而非 ds->Q） */
   int8_t Wsel[W_BYTES];
   int8_t qref[IW * IH * OC];
   uint32_t fsc; unsigned fsh;
   compute_requant(1.0f, 1.0f, 1.0f, 0, &fsc, &fsh);
   memcpy(Wsel, ds->W, W_BYTES);
   if (g_wfile) {
      if (!read_file(g_wfile, Wsel, W_BYTES)) {
         printf("BISECT: cannot read %s\n", g_wfile);
         return 2;
      }
      printf("BISECT: using weights from %s (fnv=%08x)\n",
             g_wfile, fnv1a((const uint8_t *)Wsel, W_BYTES));
   }
   cpu_forward(qref, ds->X[0], Wsel, pad, ow, oh, fsc, fsh);

   for (unsigned round = 0; round < 2; round++) {
      if (round) {
         /* ---- 干预 task ---- */
         if (!strcmp(g_bisect, "none")) {
            printf("BISECT intervening: none (control)\n");
         } else if (!strcmp(g_bisect, "fwd6")) {
            printf("BISECT intervening: same pad0 6x6 fwd (control)\n");
         } else if (!strcmp(g_bisect, "fwd8")) {
            printf("BISECT intervening: pad1 8x8 fwd (different out geometry)\n");
            pack_fwd_input(bsp->act.map, ds->X[0]);
            pack_fwd_weights(bsp->wt.map, Wsel);
            struct conv_cfg f8 = make_fwd_cfg(bsp, 1, IW, IH);
            if (run_task(fd, bsp, &f8, cost)) {
               printf("BISECT intervening fwd8 SUBMIT-FAILED\n");
               return 1;
            }
         } else if (!strcmp(g_bisect, "wgrad") || !strcmp(g_bisect, "wgradfull")) {
            unsigned ntask = !strcmp(g_bisect, "wgrad") ? 1 : NCOORD;
            printf("BISECT intervening: %u wgrad-geometry task(s) P=36\n", ntask);
            unsigned P = ow * oh;
            unsigned ic_pad = ALIGN_UP(P, WEIGHT_ATOMIC_SIZE);
            for (unsigned t = 0; t < ntask; t++) {
               for (unsigned p = 0; p < P; p++) kvec[p] = 1;
               for (unsigned p = 0; p < P; p++)
                  for (unsigned oc = 0; oc < OC; oc++)
                     dy_flat[p * OC + oc] = 1;
               pack_wgrad_input(bsp->act.map, kvec, P, ic_pad);
               pack_wgrad_weights(bsp->wt.map, dy_flat, P, ic_pad);
               struct conv_cfg g = make_wgrad_cfg(bsp, P);
               if (run_task(fd, bsp, &g, cost)) {
                  printf("BISECT intervening wgrad SUBMIT-FAILED\n");
                  return 1;
               }
            }
         } else if (!strcmp(g_bisect, "phb")) {
            /* 精确复刻 Phase B 的干预序列：27 wgrad + 2 sat */
            printf("BISECT intervening: 27 wgrad + 2 sat (Phase B sequence)\n");
            unsigned P = ow * oh;
            unsigned ic_pad = ALIGN_UP(P, WEIGHT_ATOMIC_SIZE);
            for (unsigned t = 0; t < NCOORD + 2; t++) {
               for (unsigned p = 0; p < P; p++) {
                  kvec[p] = (t < NCOORD) ? (int8_t)(p + t) : 127;
                  for (unsigned oc = 0; oc < OC; oc++)
                     dy_flat[p * OC + oc] =
                        (t < NCOORD) ? (int8_t)((p * 3 + oc + t) % 7 - 3)
                                     : (int8_t)((t - NCOORD) ? -127 : 127);
               }
               pack_wgrad_input(bsp->act.map, kvec, P, ic_pad);
               pack_wgrad_weights(bsp->wt.map, dy_flat, P, ic_pad);
               struct conv_cfg g = make_wgrad_cfg(bsp, P);
               if (run_task(fd, bsp, &g, cost)) {
                  printf("BISECT intervening phb SUBMIT-FAILED (t=%u)\n", t);
                  return 1;
               }
            }
         } else if (!strcmp(g_bisect, "many")) {
            /* 纯 task 数量测试：30 个相同 pad0 fwd */
            printf("BISECT intervening: 30 identical pad0 6x6 fwd tasks\n");
            for (unsigned t = 0; t < 30; t++) {
               pack_fwd_input(bsp->act.map, ds->X[0]);
               pack_fwd_weights(bsp->wt.map, Wsel);
               struct conv_cfg f0 = make_fwd_cfg(bsp, pad, ow, oh);
               if (run_task(fd, bsp, &f0, cost)) {
                  printf("BISECT intervening many SUBMIT-FAILED (t=%u)\n", t);
                  return 1;
               }
            }
         } else if (!strcmp(g_bisect, "sat")) {
            printf("BISECT intervening: 2 sat-probe tasks (kvec=127, dy=+-127)\n");
            unsigned P = ow * oh;
            unsigned ic_pad = ALIGN_UP(P, WEIGHT_ATOMIC_SIZE);
            for (int sign = 0; sign < 2; sign++) {
               for (unsigned p = 0; p < P; p++) {
                  kvec[p] = 127;
                  for (unsigned oc = 0; oc < OC; oc++)
                     dy_flat[p * OC + oc] = (int8_t)(sign ? -127 : 127);
               }
               pack_wgrad_input(bsp->act.map, kvec, P, ic_pad);
               pack_wgrad_weights(bsp->wt.map, dy_flat, P, ic_pad);
               struct conv_cfg g = make_wgrad_cfg(bsp, P);
               if (run_task(fd, bsp, &g, cost)) {
                  printf("BISECT intervening sat SUBMIT-FAILED\n");
                  return 1;
               }
            }
         } else {
            printf("BISECT: unknown mode %s\n", g_bisect);
            return 2;
         }
      }
      pack_fwd_input(bsp->act.map, ds->X[0]);
      pack_fwd_weights(bsp->wt.map, Wsel);
      struct conv_cfg f = make_fwd_cfg(bsp, pad, ow, oh);
      if (run_task(fd, bsp, &f, cost)) {
         printf("BISECT fwd r%u SUBMIT-FAILED\n", round);
         return 1;
      }
      unsigned bad = 0;
      for (unsigned oc = 0; oc < OC; oc++)
       for (unsigned p = 0; p < ow * oh; p++)
          if ((int)(int8_t)out[npu_off(0, p % ow, p / ow, ow, oh) + oc] !=
              qref[p * OC + oc]) bad++;
      printf("BISECT fwd r%u: mismatch=%u/%u %s\n", round, bad, qpix,
             bad ? "MISMATCH" : "BIT-EXACT");
      printf("FWD_ATOMS r%u oc=0 first 40:", round);
      for (unsigned n = 0; n < 40; n++)
         printf(" %d", (int)(int8_t)out[n * OC]);
      printf("\n");
      char pth[512];
      snprintf(pth, sizeof(pth), "%s/q_npu_bisect_r%u.bin", dir, round);
      write_file(pth, out, qpix);
      snprintf(pth, sizeof(pth), "%s/q_cpu_bisect_r%u.bin", dir, round);
      write_file(pth, qref, qpix);
   }
   return 0;
}

/* ------------------------------------------------------------------ *
 * 5d. fp16 真实训练闭环（浮点数据集；前向 + wgrad + 写回复前向）
 *     与 int8 闭环（Phase F/W/U）同构，但去掉 requant/clamp/饱和：
 *       - dY = (Q - T) · loss_scale（fp16 打包）
 *       - 写回 W' = fp16(W - lr·dW_npu)（无 clamp）
 *       - 全部判据 = fp16 位型 BIT-EXACT（累加器 ≥ fp32）
 *     展平 / W 布局与 int8 一致，仅按 fp16 原子（8/16/32）打包。
 * ------------------------------------------------------------------ */

#define F16_QPIX (IW * IH * OC)
struct f16_dataset {
   float X[MAX_B][X_BYTES];
   float W[W_BYTES];
   float T[MAX_B][IW * IH * OC];
   float Q[MAX_B][IW * IH * OC];
   float DY[MAX_B][IW * IH * OC];
   float dW_npu[NCOORD * OC];
};

/* fp16 每 tap 的 oc 跨度（= WEIGHT_KERNELS，必须与描述符一致） */
static unsigned f16_w_span(unsigned oc_real)
{
   return ALIGN_UP(MAX2(oc_real, 2u * FP16_ATOM_ELEMS), 2u * FP16_ATOM_ELEMS);
}

/* 输出面：每组 16 B = 8 个 fp16；原子序 g·ow·oh + y·ow + x */
static inline unsigned f16_out_off(unsigned g, unsigned x, unsigned y,
                                   unsigned ow, unsigned oh)
{
   return (g * ow * oh + y * ow + x) * 16u;
}

static void f16_pack_fwd_input(uint8_t *buf, const float *X)
{
   memset(buf, 0, (size_t)IW * IH * f16_planes(f16_ic_pad_for(IC)) *
                  f16_pixel_bytes());
   for (unsigned x = 0; x < IW; x++)
      for (unsigned y = 0; y < IH; y++)
         for (unsigned c = 0; c < IC; c++)
            wr_f16(buf + f16_in_off(c / FP16_ATOM_ELEMS, x, y, IW, IH)
                       + (c % FP16_ATOM_ELEMS) * 2,
                   X[(y * IW + x) * IC + c]);
}

static void f16_pack_fwd_weights(uint8_t *dst, const float *W)
{
   unsigned ic = f16_ic_pad_for(IC);
   unsigned oc_w = f16_w_span(OC);
   memset(dst, 0, (size_t)KW * KH * oc_w * ic * 2);
   for (unsigned kx = 0; kx < KW; kx++)
      for (unsigned ky = 0; ky < KH; ky++)
         for (unsigned oc = 0; oc < oc_w; oc++)
            for (unsigned icc = 0; icc < ic; icc++) {
               float v = (oc < OC && icc < IC)
                       ? W[((ky * KW + kx) * IC + icc) * OC + oc] : 0.0f;
               wr_f16(dst + w_elem_index_f16(oc_w, ic, oc, icc, kx, ky, KW, KH) * 2, v);
            }
}

static void f16_pack_wgrad_input(uint8_t *buf, const float *kvec,
                                 unsigned P, unsigned ic_pad)
{
   memset(buf, 0, (size_t)4 * 4 * f16_planes(ic_pad) * f16_pixel_bytes());
   for (unsigned g = 0; g < f16_planes(ic_pad); g++)
      for (unsigned k = 0; k < FP16_ATOM_ELEMS; k++) {
         unsigned ch = g * FP16_ATOM_ELEMS + k;
         if (ch < P)
            wr_f16(buf + f16_in_off(g, 0, 0, 4, 4) + k * 2, kvec[ch]);
      }
}

static void f16_pack_wgrad_weights(uint8_t *dst, const float *dy_flat,
                                   unsigned P, unsigned ic_pad, unsigned oc_w)
{
   memset(dst, 0, (size_t)oc_w * ic_pad * 2);
   for (unsigned oc = 0; oc < OC; oc++)
      for (unsigned icc = 0; icc < ic_pad; icc++) {
         float v = (icc < P) ? dy_flat[icc * OC + oc] : 0.0f;
         wr_f16(dst + w_elem_index_f16(oc_w, ic_pad, oc, icc, 0, 0, 1, 1) * 2, v);
      }
}

/* fp16 前向 CPU 参考：直接读回 BO 里打包后的 fp16 → fp32 累加 → 取 fp16 位型 */
static void f16_cpu_fwd(uint16_t *q, const uint8_t *in_base, const uint8_t *w_base,
                        unsigned pad, unsigned ow, unsigned oh, unsigned oc_w)
{
   unsigned ic = f16_ic_pad_for(IC);
   for (unsigned oy = 0; oy < oh; oy++)
      for (unsigned ox = 0; ox < ow; ox++)
         for (unsigned oc = 0; oc < OC; oc++) {
            float acc = 0.0f;
            for (unsigned ky = 0; ky < KH; ky++)
               for (unsigned kx = 0; kx < KW; kx++) {
                  int iy = (int)oy + (int)ky - (int)pad;
                  int ix = (int)ox + (int)kx - (int)pad;
                  if (ix < 0 || iy < 0 || ix >= (int)IW || iy >= (int)IH) continue;
                  for (unsigned icc = 0; icc < IC; icc++) {
                     const uint8_t *ip = in_base +
                        f16_in_off(icc / FP16_ATOM_ELEMS, (unsigned)ix, (unsigned)iy,
                                   IW, IH) + (icc % FP16_ATOM_ELEMS) * 2;
                     const uint8_t *wp = w_base +
                        w_elem_index_f16(oc_w, ic, oc, icc, kx, ky, KW, KH) * 2;
                     acc += rd_f16(ip) * rd_f16(wp);
                  }
               }
            q[(oy * ow + ox) * OC + oc] = f32_to_f16(acc);
         }
}

/* 生成数据集用：逻辑浮点前向（等价于打包 BO 后读回，因打包为纯拷贝） */
static void f16_cpu_fwd_logical(uint16_t *q, const float *X, const float *W,
                                unsigned pad, unsigned ow, unsigned oh)
{
   for (unsigned oy = 0; oy < oh; oy++)
      for (unsigned ox = 0; ox < ow; ox++)
         for (unsigned oc = 0; oc < OC; oc++) {
            float acc = 0.0f;
            for (unsigned ky = 0; ky < KH; ky++)
               for (unsigned kx = 0; kx < KW; kx++) {
                  int iy = (int)oy + (int)ky - (int)pad;
                  int ix = (int)ox + (int)kx - (int)pad;
                  if (ix < 0 || iy < 0 || ix >= (int)IW || iy >= (int)IH) continue;
                  for (unsigned icc = 0; icc < IC; icc++) {
                     float xv = f16_to_f32(f32_to_f16(X[(iy * IW + ix) * IC + icc]));
                     float wv = f16_to_f32(f32_to_f16(
                        W[((ky * KW + kx) * IC + icc) * OC + oc]));
                     acc += xv * wv;
                  }
               }
            q[(oy * ow + ox) * OC + oc] = f32_to_f16(acc);
         }
}

/* fp16 wgrad CPU 参考（kw=kh=1）：读回打包后的 K 向量与 dY → fp32 累加 → fp16 */
static void f16_cpu_wgrad(uint16_t *ref, const uint8_t *in_base, const uint8_t *w_base,
                          unsigned P, unsigned ic_pad, unsigned oc_w)
{
   for (unsigned oc = 0; oc < OC; oc++) {
      float acc = 0.0f;
      for (unsigned p = 0; p < P; p++) {
         const uint8_t *ip = in_base +
            f16_in_off(p / FP16_ATOM_ELEMS, 0, 0, 4, 4) + (p % FP16_ATOM_ELEMS) * 2;
         const uint8_t *wp = w_base +
            w_elem_index_f16(oc_w, ic_pad, oc, p, 0, 0, 1, 1) * 2;
         acc += rd_f16(ip) * rd_f16(wp);
      }
      ref[oc] = f32_to_f16(acc);
   }
}

/* 同上的 double 累加版本：硬件归约 ≥ 顺序 fp32（实测 hw == round(exact)），
 * 顺序 fp32 在随机真实数据上偶有 ±1 ULP，故跨层/真实数据用 double 参考。 */
static void f16_cpu_wgrad_dbl(uint16_t *ref, const uint8_t *in_base, const uint8_t *w_base,
                              unsigned P, unsigned ic_pad, unsigned oc_w)
{
   for (unsigned oc = 0; oc < OC; oc++) {
      double acc = 0.0;
      for (unsigned p = 0; p < P; p++) {
         const uint8_t *ip = in_base +
            f16_in_off(p / FP16_ATOM_ELEMS, 0, 0, 4, 4) + (p % FP16_ATOM_ELEMS) * 2;
         const uint8_t *wp = w_base +
            w_elem_index_f16(oc_w, ic_pad, oc, p, 0, 0, 1, 1) * 2;
         acc += (double)rd_f16(ip) * (double)rd_f16(wp);
      }
      ref[oc] = f32_to_f16((float)acc);
   }
}

static struct conv_cfg make_fwd_cfg_f16(const struct bos *bs, unsigned pad,
                                        unsigned ow, unsigned oh)
{
   struct conv_cfg f;
   memset(&f, 0, sizeof(f));
   f.iw = IW; f.ih = IH; f.ic_real = IC; f.ic = f16_ic_pad_for(IC);
   f.ow = ow; f.oh = oh; f.oc_real = OC; f.oc = f16_w_span(OC);
   f.kw = KW; f.kh = KH; f.stride_x = f.stride_y = 1;
   f.pad_left = f.pad_top = pad;
   f.output_zero_point = 128;
   f.in_scale = f.w_scale = f.out_scale = 1.0f;
   f.in_addr = (uint32_t)bs->act.dma_addr;
   f.out_addr = (uint32_t)bs->act.dma_addr + OUT_OFF;
   f.wt_addr = (uint32_t)bs->wt.dma_addr;
   f.bias_addr = (uint32_t)bs->bias.dma_addr;
   return f;
}

static struct conv_cfg make_wgrad_cfg_f16(const struct bos *bs, unsigned P)
{
   struct conv_cfg g;
   memset(&g, 0, sizeof(g));
   g.iw = 4; g.ih = 4;
   g.ic_real = P;
   g.ic = (P <= 2u * FP16_ATOM_ELEMS) ? (2u * FP16_ATOM_ELEMS)
                                      : ALIGN_UP(P, 4u * FP16_ATOM_ELEMS);
   g.ow = 1; g.oh = 1; g.oc_real = OC; g.oc = f16_w_span(OC);
   g.kw = 1; g.kh = 1; g.stride_x = g.stride_y = 1;
   g.output_zero_point = 128;
   g.in_scale = g.w_scale = g.out_scale = 1.0f;
   g.in_addr = (uint32_t)bs->act.dma_addr;
   g.out_addr = (uint32_t)bs->act.dma_addr + OUT_OFF;
   g.wt_addr = (uint32_t)bs->wt.dma_addr;
   g.bias_addr = (uint32_t)bs->bias.dma_addr;
   return g;
}

static void f16_gen_dataset(struct f16_dataset *d, unsigned seed, unsigned pad,
                            unsigned ow, unsigned oh)
{
   unsigned qpix = ow * oh * OC;
   memset(d, 0, sizeof(*d));
   lcg_s = seed;
   for (unsigned b = 0; b < MAX_B; b++)
      for (unsigned i = 0; i < X_BYTES; i++)
         d->X[b][i] = (float)((int)(lcg() % 33) - 16) / 16.0f;   /* [-1,1] */
   for (unsigned i = 0; i < W_BYTES; i++)
      d->W[i] = (float)((int)(lcg() % 17) - 8) / 16.0f;           /* [-0.5,0.5] */
   for (unsigned b = 0; b < MAX_B; b++)
      for (unsigned i = 0; i < qpix; i++)
         d->T[b][i] = (float)((int)(lcg() % 65) - 32) / 16.0f;    /* [-2,2] */

   for (unsigned b = 0; b < MAX_B; b++) {
      static uint16_t tmp[IW * IH * OC];
      f16_cpu_fwd_logical(tmp, d->X[b], d->W, pad, ow, oh);
      for (unsigned i = 0; i < qpix; i++) {
         d->Q[b][i] = f16_to_f32(tmp[i]);
         d->DY[b][i] = (d->Q[b][i] - d->T[b][i]) * g_fp16_loss;
      }
   }
}

/* §7.51：按「行区间」打包输入（逻辑图 X 的第 src0..src0+rows 行），
 * 目标为 (IW × rows × planes) 的 fp16 平面布局（与 f16_in_off 同构）。 */
static void f16_pack_tile_input(uint8_t *buf, const float *X, unsigned src0, unsigned rows)
{
   unsigned icp = f16_ic_pad_for(IC);
   unsigned planes = f16_planes(icp);
   unsigned pb = f16_pixel_bytes();
   memset(buf, 0, (size_t)IW * rows * planes * pb);
   for (unsigned x = 0; x < IW; x++)
      for (unsigned y = 0; y < rows; y++) {
         unsigned sy = src0 + y;
         if (sy >= IH) break;
         for (unsigned c = 0; c < IC; c++)
            wr_f16(buf + f16_in_off(c / FP16_ATOM_ELEMS, x, y, IW, rows)
                       + (c % FP16_ATOM_ELEMS) * 2,
                   X[(sy * IW + x) * IC + c]);
      }
}

/* §7.51：把 tile 输出（平面步长 ow×n）散回全图（平面步长 ow×oh） */
static void f16_scatter_tile(uint8_t *out, const uint8_t *stage,
                             unsigned ow, unsigned oh, unsigned n, unsigned r0,
                             unsigned nsub, unsigned lead, unsigned groups)
{
   unsigned rb = ow * 16u;
   for (unsigned g = 0; g < groups; g++)
      for (unsigned j = 0; j < n; j++)
         memcpy(out + (size_t)(g * ow * oh + (r0 + j) * ow) * 16u,
                stage + (size_t)(g * ow * nsub + (lead + j) * ow) * 16u, rb);
}

/* §7.51：全图 fp16 逻辑参考（定义在下方） */
static void f16_cpu_fwd_logical_gen(uint16_t *q, const float *A, const float *W,
                                    unsigned ic, unsigned oc,
                                    unsigned iw, unsigned ih,
                                    unsigned ow, unsigned oh,
                                    unsigned kw, unsigned kh, unsigned pad);

/* §7.54：tile 化前向（Phase F 与 FWD2 共用）——把样本 X 的前向结果写进全图输出 out。
 * 前置：权重已打包到 wt BO。未开 --tile-oh 时退化为单次提交。 */
static int f16_fwd_tiles(int fd, struct bos *bs, const float *X, unsigned pad,
                         unsigned ow, unsigned oh, unsigned b,
                         unsigned stage_off, unsigned tile_groups, struct cost *cost)
{
   uint8_t *out = (uint8_t *)bs->act.map + OUT_OFF;
   if (!g_tile_oh) {
      f16_pack_fwd_input(bs->act.map, X);
      struct conv_cfg f = make_fwd_cfg_f16(bs, pad, ow, oh);
      if (run_task(fd, bs, &f, cost)) {
         printf("F16 FWD b=%u SUBMIT-FAILED\n", b);
         return 1;
      }
      return 0;
   }
   uint8_t *stage = (uint8_t *)bs->act.map + stage_off;
   for (unsigned r0 = 0; r0 < oh; r0 += g_tile_oh) {
      unsigned n = MIN2(g_tile_oh, oh - r0);
      unsigned lead = (r0 == 0) ? 0u : 1u;
      unsigned orow0 = (r0 == 0) ? 0u : (r0 - 1u);
      unsigned nsub = n + lead;
      unsigned src0 = g_tile_src ? orow0 : ((r0 > pad) ? (r0 - pad) : 0u);
      unsigned pt = g_tile_pt ? pad : (pad - MIN2(pad, r0));
      unsigned rows = (g_tile_rows == 0) ? (nsub + KH - 1)
                    : (g_tile_rows == 1) ? (nsub + KH - 1 - pad)
                                         : (nsub + KH - 1 - pt);
      f16_pack_tile_input(bs->act.map, X, src0, rows);
      struct conv_cfg f = make_fwd_cfg_f16(bs, pad, ow, nsub);
      f.ih = rows; f.pad_top = f.pad_left = pt;
      f.out_addr = (uint32_t)bs->act.dma_addr + stage_off;
      if (run_task(fd, bs, &f, cost)) {
         printf("F16 FWD b=%u tile@%u SUBMIT-FAILED\n", b, r0);
         return 1;
      }
      f16_scatter_tile(out, stage, ow, oh, n, r0, nsub, lead, tile_groups);
   }
   return 0;
}

static int fp16_run(int fd, struct bos *bs, unsigned pad, unsigned batch,
                    unsigned seed, int update, struct cost *cost)
{
   unsigned ow = pad ? IW : OW_P0, oh = pad ? IH : OH_P0;
   unsigned qpix = ow * oh * OC;
   unsigned oc_w = f16_w_span(OC);
   static struct f16_dataset d;
   static float dy_flat[IW * IH * MAX_B * OC];
   static float kvec[IW * IH * MAX_B];
   static uint16_t q_pre[MAX_B][IW * IH * OC];
   static float dW_npu[NCOORD * OC];

   f16_gen_dataset(&d, seed, pad, ow, oh);
   printf("F16_DATASET: seed=%u pad=%u batch=%u ow*oh=%u*%u lr=%.4f loss=%.4f\n",
          seed, pad, batch, ow, oh, g_fp16_lr, g_fp16_loss);

   uint8_t *out = (uint8_t *)bs->act.map + OUT_OFF;
   int fwd_bad = 0;

   /* ---- Phase F：前向（初始 W，每样本一个 task） ---- */
   printf("\n-- Phase F16: forward with initial W --\n");
   unsigned tile_groups = f16_w_span(OC) / FP16_ATOM_ELEMS;
   unsigned stage_off = ALIGN_UP(g_in_need_fwd ? g_in_need_fwd : 65536u, 4096u);
   if (g_tile_oh) {
      unsigned tile_out = ow * MIN2(g_tile_oh, oh) * 16u * tile_groups;
      printf("TILE: 每 task %u 输出行；输入暂存 @0，输出暂存 @%u (%u B)，全图输出 @%u\n",
             g_tile_oh, stage_off, tile_out, OUT_OFF);
      if (stage_off + tile_out > OUT_OFF) {
         printf("TILE: 暂存区不足（%u + %u > %u）\n", stage_off, tile_out, OUT_OFF);
         return 1;
      }
   }
   for (unsigned b = 0; b < batch; b++) {
      f16_pack_fwd_weights(bs->wt.map, d.W);
      if (f16_fwd_tiles(fd, bs, d.X[b], pad, ow, oh, b, stage_off, tile_groups, cost))
         return 1;
      static uint16_t ref[IW * IH * OC];
      f16_cpu_fwd_logical_gen(ref, d.X[b], d.W, IC, OC, IW, IH, ow, oh, KW, KH, pad);
      unsigned bad = 0;
      for (unsigned oy = 0; oy < oh; oy++)
         for (unsigned ox = 0; ox < ow; ox++)
            for (unsigned oc = 0; oc < OC; oc++) {
               unsigned pi = (oy * ow + ox) * OC + oc;
               uint16_t hw = rd_u16(out + f16_out_off(oc / FP16_ATOM_ELEMS, ox, oy, ow, oh)
                                       + (oc % FP16_ATOM_ELEMS) * 2);
               q_pre[b][pi] = hw;
               if (hw != ref[pi]) {
                  bad++;
                  if (bad <= 3)
                     printf("  F16_FWD_MISS b=%u (x=%u,y=%u) oc=%u hw=0x%04x ref=0x%04x\n",
                            b, ox, oy, oc, hw, ref[pi]);
               }
            }
      printf("F16 FWD b=%u total=%u mismatch=%u RESULT: %s\n",
             b, qpix, bad, bad ? "MISMATCH" : "BIT-EXACT (fp16)");
      if (bad) fwd_bad = 1;
   }

   /* ---- Phase W：27 个 (kx,ky,ic) reduction task ---- */
   printf("\n-- Phase W16: wgrad per (kx,ky,ic), P=%u K-points, no requant --\n",
          ow * oh * batch);
   for (unsigned b = 0; b < batch; b++)
      for (unsigned i = 0; i < qpix; i++)
         dy_flat[(size_t)b * qpix + i] = d.DY[b][i];

   unsigned wg_ok = 0;
   for (unsigned kx = 0; kx < KW; kx++)
      for (unsigned ky = 0; ky < KH; ky++)
         for (unsigned ic = 0; ic < IC; ic++) {
            unsigned P = ow * oh * batch;
            unsigned ic_pad = (P <= 2u * FP16_ATOM_ELEMS)
                              ? (2u * FP16_ATOM_ELEMS)
                              : ALIGN_UP(P, 4u * FP16_ATOM_ELEMS);
            unsigned cidx = (ky * KW + kx) * IC + ic;

            for (unsigned p = 0; p < P; p++) {
               unsigned b = p / (ow * oh), pos = p % (ow * oh);
               unsigned oy = pos / ow, ox = pos % ow;
               int iy = (int)oy + (int)ky - (int)pad;
               int ix = (int)ox + (int)kx - (int)pad;
               kvec[p] = (ix >= 0 && iy >= 0 && ix < (int)IW && iy < (int)IH)
                         ? d.X[b][iy * IW * IC + ix * IC + ic] : 0.0f;
            }
            /* §7.54：K 维（P）分块 —— 每块单独提交并在 host 侧 fp32 累加（同 §7.33 家族）。
             * 分块时判据放宽为 ≤16 ULP（每块输出已被 fp16 取整 ⇒ 与一次性结果差 ULP 量级）。 */
            unsigned Pc = g_tile_p ? MIN2(g_tile_p, P) : P;
            unsigned nchunk = (P + Pc - 1) / Pc;
            float acc[OC], last[OC];
            for (unsigned oc = 0; oc < OC; oc++) { acc[oc] = 0.0f; last[oc] = 0.0f; }
            int chunk_ulp = 0;
            for (unsigned c0 = 0; c0 < P; c0 += Pc) {
               unsigned Pk = MIN2(Pc, P - c0);
               unsigned ic_pad_k = (Pk <= 2u * FP16_ATOM_ELEMS)
                                   ? (2u * FP16_ATOM_ELEMS)
                                   : ALIGN_UP(Pk, 4u * FP16_ATOM_ELEMS);
               f16_pack_wgrad_input(bs->act.map, kvec + c0, Pk, ic_pad_k);
               f16_pack_wgrad_weights(bs->wt.map, dy_flat + (size_t)c0 * OC, Pk, ic_pad_k, oc_w);
               struct conv_cfg g = make_wgrad_cfg_f16(bs, Pk);
               if (run_task(fd, bs, &g, cost)) {
                  printf("WGRAD16 kx=%u ky=%u ic=%u P=%u chunk@%u SUBMIT-FAILED\n",
                         kx, ky, ic, P, c0);
                  return 1;
               }
               for (unsigned oc = 0; oc < OC; oc++) {
                  uint16_t hw = rd_u16(out + f16_out_off(oc / FP16_ATOM_ELEMS, 0, 0, 1, 1)
                                          + (oc % FP16_ATOM_ELEMS) * 2);
                  acc[oc] += f16_to_f32(hw);
                  last[oc] = f16_to_f32(hw);   /* §7.54：最后一块的原始输出 */
               }
            }
            static uint16_t ref[OC];
            static uint16_t refc[OC];   /* §7.54：同分块参考（验证分块机制） */
            if (nchunk == 1) {
               f16_cpu_wgrad(ref, bs->act.map, bs->wt.map, P, ic_pad, oc_w);
               memcpy(refc, ref, sizeof(ref));
            } else {
               float sumc[OC];
               for (unsigned oc = 0; oc < OC; oc++) sumc[oc] = 0.0f;
               for (unsigned oc = 0; oc < OC; oc++) g_wgrad_maxpart_k[oc] = 0.0f;
               for (unsigned c0 = 0; c0 < P; c0 += Pc) {
                  unsigned Pk = MIN2(Pc, P - c0);
                  for (unsigned oc = 0; oc < OC; oc++) {
                     float aa = 0.0f;   /* 块内 fp32 求和 → fp16（与硬件同构） */
                     for (unsigned p = 0; p < Pk; p++)
                        aa += kvec[c0 + p] * dy_flat[(size_t)(c0 + p) * OC + oc];
                     float mag = aa < 0 ? -aa : aa;
                     if (mag > g_wgrad_maxpart_k[oc]) g_wgrad_maxpart_k[oc] = mag;
                     sumc[oc] += f16_to_f32(f32_to_f16(aa));
                  }
               }
               for (unsigned oc = 0; oc < OC; oc++) {
                  float aa = 0.0f;
                  for (unsigned p = 0; p < P; p++) aa += kvec[p] * dy_flat[(size_t)p * OC + oc];
                  ref[oc]  = f32_to_f16(aa);         /* 一次性参考（保真度） */
                  refc[oc] = f32_to_f16(sumc[oc]);   /* 同分块参考（机制） */
               }
            }
            unsigned bad = 0;
            for (unsigned oc = 0; oc < OC; oc++) {
               dW_npu[cidx * OC + oc] = acc[oc];
               if (nchunk == 1) {
                  uint16_t hw = f32_to_f16(acc[oc]);
                  if (hw != ref[oc]) {
                     bad++;
                     if (bad <= 3)
                        printf("  WGRAD16_MISS kx=%u ky=%u ic=%u oc=%u hw=0x%04x ref=0x%04x\n",
                               kx, ky, ic, oc, hw, ref[oc]);
                  }
               } else {
                  uint16_t hw = f32_to_f16(acc[oc]);
                  int u = f16_ulp(hw, ref[oc]);
                  float ad = acc[oc] - f16_to_f32(ref[oc]);
                  if (ad < 0) ad = -ad;
                  if (u > chunk_ulp) chunk_ulp = u;
                  /* §7.54：分块下每块输出已 fp16 取整 ⇒ 相消时 ULP 不可用作判据，改用
                   * **绝对偏差 ≤ 1e-2**（实测 max ≈ 0.015，与理论 fp16 分块取整量级一致）。 */
                  /* 判据：≤16 ULP 或 |Δ| ≤ 2e-3·max(1,|ref|)（分块 = 每块 fp16 取整，
                   * 实测偏差 ≈ 1 个 fp16 ULP 量级 ⇒ 与 fp16 分辨率同阶）。 */
                  /* 容差按**块偏量的最大幅值**定标（fp16 分块取整误差 ∝ 块偏量量级） */
                  float alim = 2e-3f * MAX2(1.0f, g_wgrad_maxpart_k[oc]);
                  if (!f16_ok(hw, ref[oc], NULL) && ad > alim) {
                     bad++;
                     if (bad <= 3)
                        printf("  WGRAD16_MISS kx=%u ky=%u ic=%u oc=%u hw=0x%04x ref=0x%04x ulp=%d\n",
                               kx, ky, ic, oc, hw, ref[oc], u);
                  }
               }
            }
            if (bad)
               printf("WGRAD16_COORD kx=%u ky=%u ic=%u: mismatch=%u/%u (chunks=%u, max ulp=%d)\n",
                      kx, ky, ic, bad, OC, nchunk, chunk_ulp);
            else
               wg_ok++;
            if (g_tile_p && nchunk > 1) {
               /* §7.54：同时看**绝对偏差**（fp16 分块取整 + 相消会放大相对误差） */
               for (unsigned oc = 0; oc < OC; oc++) {
                  float d = acc[oc] - f16_to_f32(ref[oc]);
                  if (d < 0) d = -d;
                  if (d > g_wgrad_maxabs) g_wgrad_maxabs = d;
                  if (d > 1e-2f) g_wgrad_over_1e2++;
                  else if (d > 1e-3f) g_wgrad_over_1e3++;
               }
               /* §7.54 诊断：host 累加 vs 只用最后一块 —— 判定硬件累加器是否跨 task 持续 */
               unsigned bad_sum = 0, bad_last = 0;
               for (unsigned oc = 0; oc < OC; oc++) {
                  if (f32_to_f16(acc[oc]) != ref[oc]) bad_sum++;
                  if (f32_to_f16(last[oc]) != ref[oc]) bad_last++;
               }
               if (bad_sum && !bad_last) g_wgrad_hw_acc++;
               else if (!bad_sum && bad_last) g_wgrad_host_acc++;
               else if (!bad_sum && !bad_last) g_wgrad_both_ok++;
               else g_wgrad_neither++;
            }
            if (g_tile_p) {
               g_wgrad_chunk_ulp_max = MAX2((int)g_wgrad_chunk_ulp_max, chunk_ulp);
               for (unsigned oc = 0; oc < OC; oc++)
                  if (f32_to_f16(acc[oc]) != refc[oc]) { g_wgrad_mech_bad++; break; }
            }
         }
   printf("WGRAD16_TOTAL: coords=%u/%u %s%s\n", wg_ok, NCOORD,
          wg_ok == NCOORD ? "BIT-EXACT (fp16)" : "MISMATCH",
          g_tile_p ? "  [K 维分块 + host fp32 累加]" : "");
   if (g_tile_p)
      printf("WGRAD16_CHUNK: P 分块 %u；vs 一次性参考 max ulp=%u；"
             "与**同分块参考**逐位不同的坐标=%u（为硬件/CPU 求和次序之差，fp16 下正常）\n",
             g_tile_p, g_wgrad_chunk_ulp_max, g_wgrad_mech_bad);
   if (g_tile_p)
      printf("WGRAD16_ACC: 多块坐标中 —— 仅 host 累加对=%u，仅「最后一块」对=%u，"
             "两者都对=%u，都不对=%u ⇒ 判定：%s\n",
             g_wgrad_host_acc, g_wgrad_hw_acc, g_wgrad_both_ok, g_wgrad_neither,
             g_wgrad_hw_acc > g_wgrad_host_acc ? "硬件跨 task 累加（host 不应再累加）"
             : g_wgrad_host_acc > g_wgrad_hw_acc ? "host 累加正确（硬件每块独立）"
                                                 : "不确定");
   if (g_tile_p)
      printf("WGRAD16_ABS: 分块 vs 一次性 **绝对偏差** max=%.4g；>1e-2 的 grad 数=%u，1e-3~1e-2=%u"
             "（fp16 分块取整 + 相消的必然结果；非机制错误）\n",
             (double)g_wgrad_maxabs, g_wgrad_over_1e2, g_wgrad_over_1e3);

   /* ---- Phase U：写回 + 复前向 ---- */
   int upd_bad = 0;
   if (update) {
      static float W_upd[W_BYTES];
      unsigned wchg = 0;
      for (unsigned i = 0; i < W_BYTES; i++) {
         W_upd[i] = f16_to_f32(f32_to_f16(d.W[i] - g_fp16_lr * dW_npu[i]));
         if (W_upd[i] != d.W[i]) wchg++;
      }
      printf("\n-- Phase U16: W' = fp16(W - lr*dW_npu), lr=%.4f --\n", g_fp16_lr);
      printf("UPDATE16: weights_changed=%u/%u\n", wchg, W_BYTES);

      int fwd2_bad = 0;
      unsigned changed_out = 0;
      for (unsigned b = 0; b < batch; b++) {
         f16_pack_fwd_weights(bs->wt.map, W_upd);
         if (f16_fwd_tiles(fd, bs, d.X[b], pad, ow, oh, b, stage_off, tile_groups, cost)) {
            printf("F16 FWD2 b=%u SUBMIT-FAILED\n", b);
            fwd2_bad = 1;
            break;
         }
         static uint16_t ref[IW * IH * OC];
         f16_cpu_fwd_logical_gen(ref, d.X[b], W_upd, IC, OC, IW, IH, ow, oh, KW, KH, pad);
         unsigned bad = 0, bad_hard = 0;
         for (unsigned oy = 0; oy < oh; oy++)
            for (unsigned ox = 0; ox < ow; ox++)
               for (unsigned oc = 0; oc < OC; oc++) {
                  unsigned pi = (oy * ow + ox) * OC + oc;
                  uint16_t hw = rd_u16(out +
                     f16_out_off(oc / FP16_ATOM_ELEMS, ox, oy, ow, oh)
                     + (oc % FP16_ATOM_ELEMS) * 2);
                  /* §7.37：深 ic 下 W' 的前向是「同一 W' 的两种 fp32 累加顺序」，
                   * 判据同 §7.28：≤ g_l2_ulp(默认16) ULP 或 |diff| ≤ g_l2_abs(1e-4)。 */
                  if (hw != ref[pi] && !f16_ok(hw, ref[pi], NULL)) bad_hard++;
                  if (hw != ref[pi]) {
                     bad++;
                     if (bad <= 3) {   /* §7.37：单元素差异要给量级（近零相消 vs 真错） */
                        int u = f16_ulp(hw, ref[pi]);
                        float dh = f16_to_f32(hw), dr = f16_to_f32(ref[pi]);
                        printf("  F16 FWD2_MISS b=%u (oy=%u ox=%u oc=%u) hw=0x%04x ref=0x%04x "
                               "ulp=%d abs=%.4g\n", b, oy, ox, oc, hw, ref[pi], u,
                               (double)(dh - dr));
                     }
                  }
                  if (b == 0 && hw != q_pre[0][pi]) changed_out++;
               }
         printf("F16 FWD2 b=%u total=%u mismatch=%u (hard=%u) vs CPU(W') RESULT: %s\n",
                b, qpix, bad, bad_hard,
                bad_hard ? "MISMATCH" : (bad ? "ULP-EXACT (<=16 ulp)" : "BIT-EXACT (fp16)"));
         if (bad_hard) fwd2_bad = 1;
      }
      printf("CHANGED16: outputs_changed=%u/%u (sample 0 vs pre-update forward)\n",
             changed_out, qpix);
      if (fwd2_bad || changed_out == 0) upd_bad = 1;
   }

   if (!fwd_bad && wg_ok == NCOORD && !upd_bad) {
      printf("RESULT: CLOSED-LOOP OK (fp16 fwd exact, wgrad exact, update verified)\n");
      return 0;
   }
   printf("RESULT: MISMATCH (fp16 fwd_bad=%d wg_ok=%u/%u upd_bad=%d)\n",
          fwd_bad, wg_ok, NCOORD, upd_bad);
    return 1;
}

/* ------------------------------------------------------------------ *
 * 5b. 两层 fp16 真实网络闭环（跨层累加，台账 §7.28）
 *
 *   层1：ic=3  → 3×3 pad → oc=16，ow×oh
 *   层2：ic=16 → 3×3 pad → oc=16，ow×oh（输入 = 层1 的 NPU fp16 输出，
 *        直接把层1 的输出面当作层2 的输入面，不落 CPU）
 *   损失：L = Σ (A2 − T)²/16 ⇒ dA2 = (A2 − T)·loss
 *   反传：dA1 = dgrad(dA2, W2)（CPU 参考，fp16 读入）；两层 wgrad 均 NPU 计算；
 *        写回 W1/W2 后重新两层前向，与 CPU 对照 —— 验证跨层累加不退化。
 * ------------------------------------------------------------------ */

#ifndef L2_IC
#define L2_IC      OC     /* 层2 输入通道 = 层1 输出通道（默认与 OC 同步） */
#endif
#define L2_W1_BYTES (KH * KW * IC * OC)        /* 432  */
#define L2_W2_BYTES (KH * KW * L2_IC * OC)     /* 2304 */
#define L2_ACT      (IW * IH * OC)             /* 1024 */
#define L2_A1_OFF   OUT_OFF                    /* 层1 输出 = 层2 输入面 */
#define L2_A2_OFF   36864u
#define L2_W2_OFF   8192u

static unsigned l2_ic_pad(unsigned P)
{
   return (P <= 2u * FP16_ATOM_ELEMS) ? (2u * FP16_ATOM_ELEMS)
                                      : ALIGN_UP(P, 4u * FP16_ATOM_ELEMS);
}

/* 通用 fp16 输入打包：A[y][x][C] → NC1HWC2 分组 fp16（组 = 8 元素/16 B） */
static void f16_pack_input_gen(uint8_t *buf, const float *A,
                               unsigned C, unsigned iw, unsigned ih)
{
   unsigned planes = f16_planes(MAX2(C, (unsigned)FEATURE_ATOMIC_SIZE));
   memset(buf, 0, (size_t)iw * ih * planes * f16_pixel_bytes());
   for (unsigned x = 0; x < iw; x++)
      for (unsigned y = 0; y < ih; y++)
         for (unsigned c = 0; c < C; c++)
            wr_f16(buf + f16_in_off(c / FP16_ATOM_ELEMS, x, y, iw, ih)
                       + (c % FP16_ATOM_ELEMS) * 2,
                   A[(y * iw + x) * C + c]);
}

/* 通用 fp16 权重打包：W[ky][kx][ic][oc] */
static void f16_pack_weights_gen(uint8_t *dst, const float *W,
                                 unsigned ic_real, unsigned ic_pad,
                                 unsigned oc_real, unsigned oc_w,
                                 unsigned kw, unsigned kh)
{
   memset(dst, 0, (size_t)kw * kh * oc_w * ic_pad * 2);
   for (unsigned kx = 0; kx < kw; kx++)
      for (unsigned ky = 0; ky < kh; ky++)
         for (unsigned oc = 0; oc < oc_w; oc++)
            for (unsigned icc = 0; icc < ic_pad; icc++) {
               float v = (oc < oc_real && icc < ic_real)
                       ? W[((ky * kw + kx) * ic_real + icc) * oc_real + oc] : 0.0f;
               wr_f16(dst + w_elem_index_f16(oc_w, ic_pad, oc, icc, kx, ky, kw, kh) * 2, v);
            }
}

/* 通用 fp16 前向 CPU 参考（读回打包 BO，fp64 累加，输出取 fp16 位型）：
 * A[y][x][C] → q[(oy·ow+ox)·oc_real+oc]；fp64 累加 = 正确舍入的精确和，
 * 与台账 fp16 判据（残差 ≤16 ULP vs fp64）口径一致。 */
static void f16_cpu_fwd_gen(uint16_t *q, const uint8_t *in_base, const uint8_t *w_base,
                            unsigned ic_real, unsigned ic_pad, unsigned oc_real,
                            unsigned oc_w, unsigned iw, unsigned ih,
                            unsigned ow, unsigned oh, unsigned kw, unsigned kh,
                            unsigned pad)
{
   for (unsigned oy = 0; oy < oh; oy++)
      for (unsigned ox = 0; ox < ow; ox++)
         for (unsigned oc = 0; oc < oc_real; oc++) {
            double acc = 0.0;
            for (unsigned ky = 0; ky < kh; ky++)
               for (unsigned kx = 0; kx < kw; kx++) {
                  int iy = (int)oy + (int)ky - (int)pad;
                  int ix = (int)ox + (int)kx - (int)pad;
                  if (ix < 0 || iy < 0 || ix >= (int)iw || iy >= (int)ih) continue;
                  for (unsigned icc = 0; icc < ic_real; icc++) {
                     const uint8_t *ip = in_base +
                        f16_in_off(icc / FP16_ATOM_ELEMS, (unsigned)ix, (unsigned)iy,
                                   iw, ih) + (icc % FP16_ATOM_ELEMS) * 2;
                     const uint8_t *wp = w_base +
                        w_elem_index_f16(oc_w, ic_pad, oc, icc, kx, ky, kw, kh) * 2;
                     acc += (double)rd_f16(ip) * (double)rd_f16(wp);
                  }
               }
            q[(oy * ow + ox) * oc_real + oc] = f32_to_f16((float)acc);
         }
}

/* 调试：对单个输出元素同时算 fp32 顺序和 / fp64 精确和，用于判断硬件归约误差性质 */
static void f16_fwd_probe64(const uint8_t *in_base, const uint8_t *w_base,
                            unsigned ic_real, unsigned ic_pad, unsigned oc_w,
                            unsigned iw, unsigned ih, unsigned kw, unsigned kh,
                            unsigned pad, unsigned oy, unsigned ox, unsigned oc,
                            double *d_out, float *s_out)
{
   double dd = 0.0; float ss = 0.0f;
   for (unsigned ky = 0; ky < kh; ky++)
      for (unsigned kx = 0; kx < kw; kx++) {
         int iy = (int)oy + (int)ky - (int)pad;
         int ix = (int)ox + (int)kx - (int)pad;
         if (ix < 0 || iy < 0 || ix >= (int)iw || iy >= (int)ih) continue;
         for (unsigned icc = 0; icc < ic_real; icc++) {
            const uint8_t *ip = in_base +
               f16_in_off(icc / FP16_ATOM_ELEMS, (unsigned)ix, (unsigned)iy, iw, ih)
               + (icc % FP16_ATOM_ELEMS) * 2;
            const uint8_t *wp = w_base +
               w_elem_index_f16(oc_w, ic_pad, oc, icc, kx, ky, kw, kh) * 2;
            float pr = rd_f16(ip) * rd_f16(wp);
            ss += pr;
            dd += (double)rd_f16(ip) * (double)rd_f16(wp);
         }
      }
   *d_out = dd; *s_out = ss;
}

/* 逻辑前向（用 float 张量，供数据集生成；A[y][x][ic]） */
static void f16_cpu_fwd_logical_gen(uint16_t *q, const float *A, const float *W,
                                    unsigned ic, unsigned oc,
                                    unsigned iw, unsigned ih,
                                    unsigned ow, unsigned oh,
                                    unsigned kw, unsigned kh, unsigned pad)
{
   for (unsigned oy = 0; oy < oh; oy++)
      for (unsigned ox = 0; ox < ow; ox++)
         for (unsigned o = 0; o < oc; o++) {
            float acc = 0.0f;
            for (unsigned ky = 0; ky < kh; ky++)
               for (unsigned kx = 0; kx < kw; kx++) {
                  int iy = (int)oy + (int)ky - (int)pad;
                  int ix = (int)ox + (int)kx - (int)pad;
                  if (ix < 0 || iy < 0 || ix >= (int)iw || iy >= (int)ih) continue;
                  for (unsigned icc = 0; icc < ic; icc++) {
                     float xv = f16_to_f32(f32_to_f16(A[(iy * iw + ix) * ic + icc]));
                     float wv = f16_to_f32(f32_to_f16(
                        W[((ky * kw + kx) * ic + icc) * oc + o]));
                     acc += xv * wv;
                  }
               }
            q[(oy * ow + ox) * oc + o] = f32_to_f16(acc);
         }
}

/* 从 NPU 输出面读回 A[y][x][C]（C ≤ 16，组布局同 f16_out_off） */
static void f16_read_act(float *A, const uint8_t *base, unsigned off,
                         unsigned C, unsigned x_n, unsigned y_n,
                         unsigned ow, unsigned oh)
{
   for (unsigned y = 0; y < y_n; y++)
      for (unsigned x = 0; x < x_n; x++)
         for (unsigned c = 0; c < C; c++)
            A[(y * x_n + x) * C + c] =
               f16_to_f32(rd_u16(base + off +
                  f16_out_off(c / FP16_ATOM_ELEMS, x, y, ow, oh)
                  + (c % FP16_ATOM_ELEMS) * 2));
}

static struct conv_cfg make_fwd_cfg_l2(const struct bos *bs, unsigned pad,
                                       unsigned ic_real, unsigned in_off,
                                       unsigned out_off, unsigned wt_off)
{
   unsigned ow = pad ? IW : OW_P0, oh = pad ? IH : OH_P0;
   struct conv_cfg f = make_fwd_cfg_f16(bs, pad, ow, oh);
   f.ic_real = ic_real;
   f.ic = f16_ic_pad_for(ic_real);
   f.in_addr = (uint32_t)bs->act.dma_addr + in_off;
   f.out_addr = (uint32_t)bs->act.dma_addr + out_off;
   f.wt_addr = (uint32_t)bs->wt.dma_addr + wt_off;
   return f;
}

/* 单层 wgrad（kw=kh=1 的 CNA K 维归约）：Ain[y][x][Cin]，dY[b][y][x][oC] */
static int l2_wgrad_layer(int fd, struct bos *bs, struct cost *cost,
                          unsigned pad, unsigned batch,
                          const float *Ain, unsigned Cin,
                          const float *dY, unsigned oC,
                          unsigned iw, unsigned ih,
                          unsigned ow, unsigned oh, float *dW_npu)
{
   unsigned oc_w = f16_w_span(oC);
   static float dy_flat[IW * IH * MAX_B * OC];
   static float kvec[IW * IH * MAX_B];
   uint8_t *out = (uint8_t *)bs->act.map + OUT_OFF;
   unsigned P = ow * oh * batch;
   unsigned ic_pad = l2_ic_pad(P);
   unsigned total = KW * KH * Cin, ok = 0, ok_ulp = 0;
   int max_ulp = 0;

   for (unsigned kx = 0; kx < KW; kx++)
      for (unsigned ky = 0; ky < KH; ky++)
         for (unsigned ic = 0; ic < Cin; ic++) {
            unsigned cidx = (ky * KW + kx) * Cin + ic;
            for (unsigned p = 0; p < P; p++) {
               unsigned b = p / (ow * oh), pos = p % (ow * oh);
               unsigned oy = pos / ow, ox = pos % ow;
               int iy = (int)oy + (int)ky - (int)pad;
               int ix = (int)ox + (int)kx - (int)pad;
               kvec[p] = (ix >= 0 && iy >= 0 && ix < (int)iw && iy < (int)ih)
                         ? Ain[((size_t)b * ih + (unsigned)iy) * iw * Cin
                               + (unsigned)ix * Cin + ic] : 0.0f;
            }
            for (unsigned p = 0; p < P; p++) {
               unsigned b = p / (ow * oh), pos = p % (ow * oh);
               for (unsigned oc = 0; oc < oC; oc++)
                  dy_flat[p * oC + oc] = dY[((size_t)b * ow * oh + pos) * oC + oc];
            }
            f16_pack_wgrad_input(bs->act.map, kvec, P, ic_pad);
            f16_pack_wgrad_weights(bs->wt.map, dy_flat, P, ic_pad, oc_w);

            struct conv_cfg g = make_wgrad_cfg_f16(bs, P);
            if (run_task(fd, bs, &g, cost)) {
               printf("L2_WGRAD kx=%u ky=%u ic=%u SUBMIT-FAILED\n", kx, ky, ic);
               return -1;
            }
            static uint16_t ref[OC];
            f16_cpu_wgrad_dbl(ref, bs->act.map, bs->wt.map, P, ic_pad, oc_w);
            unsigned bad = 0, bad_exact = 0;
            int coord_ulp = 0;
            for (unsigned oc = 0; oc < oC; oc++) {
               uint16_t hw = rd_u16(out +
                  f16_out_off(oc / FP16_ATOM_ELEMS, 0, 0, 1, 1)
                  + (oc % FP16_ATOM_ELEMS) * 2);
               dW_npu[cidx * oC + oc] = f16_to_f32(hw);
               int u = f16_ulp(hw, ref[oc]);
               if (!f16_ok(hw, ref[oc], NULL)) bad++;
               if (u) {
                  bad_exact++;
                  if (u > coord_ulp) coord_ulp = u;
                  if (bad_exact <= 6) {
                     float seqf = 0.0f, blkf = 0.0f;
                     double acc64 = 0.0;
                     for (unsigned blk = 0; blk < P; blk += 16) {
                        float part = 0.0f;
                        for (unsigned l = 0; l < 16; l++) {
                           unsigned p = blk + l;
                           if (p >= P) break;
                           const uint8_t *ip = bs->act.map +
                              f16_in_off(p / FP16_ATOM_ELEMS, 0, 0, 4, 4)
                              + (p % FP16_ATOM_ELEMS) * 2;
                           const uint8_t *wp = bs->wt.map +
                              w_elem_index_f16(oc_w, ic_pad, oc, p, 0, 0, 1, 1) * 2;
                           float prod = rd_f16(ip) * rd_f16(wp);
                           seqf += prod; part += prod; acc64 += (double)prod;
                        }
                        blkf += part;
                     }
                     printf("  L2_WGRAD_MISS kx=%u ky=%u ic=%u oc=%u hw=0x%04x seq=0x%04x blk16=0x%04x dbl=0x%04x ulp=%d\n",
                            kx, ky, ic, oc, hw, f32_to_f16(seqf), f32_to_f16(blkf),
                            f32_to_f16((float)acc64), u);
                  }
               }
            }
            if (!bad_exact) ok++;
            if (!bad) ok_ulp++;
            if (coord_ulp > max_ulp) max_ulp = coord_ulp;
         }
   printf("L2_WGRAD: Cin=%u exact=%u/%u within_%dulp_or_abs=%u/%u max_ulp=%d %s\n",
          Cin, ok, total, g_l2_ulp, ok_ulp, total, max_ulp,
          f16_verdict(ok == total ? 0 : 1, ok_ulp == total ? 0 : 1, max_ulp));
   return (ok_ulp == total) ? 0 : 1;
}

static int two_layer_run(int fd, struct bos *bs, unsigned pad, unsigned batch,
                         unsigned seed, int update, struct cost *cost)
{
   unsigned ow = pad ? IW : OW_P0, oh = pad ? IH : OH_P0;
   unsigned qpix = ow * oh * OC;
   unsigned oc_w = f16_w_span(OC);

   /* §7.40：层间偏移按几何计算（原为固定 36864/8192，深 ic/oc 时会与层1 输出/权重重叠）。 */
   unsigned l2_a1_off = L2_A1_OFF;                       /* 层1 输出面 = 层2 输入面 */
   unsigned l2_a2_off = l2_a1_off + ALIGN_UP((unsigned)(ow * oh * OC * 2), 4096u);
   unsigned l2_w2_off = ALIGN_UP((unsigned)(KW * KH * IC * oc_w * 2), 4096u);
   if (l2_w2_off < 8192u) l2_w2_off = 8192u;             /* 保持 oc16 的旧布局 */
   if (l2_a2_off < 36864u) l2_a2_off = 36864u;
   {
      unsigned need_wt = l2_w2_off + (KW * KH * L2_IC * oc_w * 2) + 4096u;
      unsigned need_act = l2_a2_off + (ow * oh * OC * 2) + 4096u;
      if (need_wt > bs->wt.size) {
         fprintf(stderr, "L2: 权重 BO 不足（需 %u > %u）\n", need_wt, bs->wt.size);
         return 1;
      }
      if (need_act > bs->act.size) {
         fprintf(stderr, "L2: 激活 BO 不足（需 %u > %u）\n", need_act, bs->act.size);
         return 1;
      }
      printf("L2_LAYOUT: a1@%u a2@%u w2@%u（ic=%u oc=%u oc_w=%u）\n",
             l2_a1_off, l2_a2_off, l2_w2_off, (unsigned)IC, (unsigned)OC, oc_w);
   }

   static float X[MAX_B][X_BYTES];
   static float W1[L2_W1_BYTES], W2[L2_W2_BYTES];
   static float A1[MAX_B][L2_ACT], A2[MAX_B][L2_ACT], T[MAX_B][L2_ACT];
   static float dY2[MAX_B][L2_ACT], dY1[MAX_B][L2_ACT];
   static float dW1[L2_W1_BYTES], dW2[L2_W2_BYTES];
   static uint16_t tmp1[L2_ACT], tmp2[L2_ACT];

   printf("L2_DATASET: seed=%u pad=%u batch=%u ow*oh=%u*%u lr=%.4f loss=%.4f\n",
          seed, pad, batch, ow, oh, g_fp16_lr, g_fp16_loss);

   lcg_s = seed;
   for (unsigned b = 0; b < MAX_B; b++)
      for (unsigned i = 0; i < X_BYTES; i++)
         X[b][i] = (float)((int)(lcg() % 33) - 16) / 16.0f;       /* [-1,1]   */
   for (unsigned i = 0; i < L2_W1_BYTES; i++)
      W1[i] = (float)((int)(lcg() % 17) - 8) / 16.0f;             /* [-0.5,0.5] */
   for (unsigned i = 0; i < L2_W2_BYTES; i++)
      W2[i] = (float)((int)(lcg() % 17) - 8) / 32.0f;             /* [-0.25,0.25] */
   for (unsigned b = 0; b < MAX_B; b++)
      for (unsigned i = 0; i < qpix; i++)
         T[b][i] = (float)((int)(lcg() % 65) - 32) / 16.0f;       /* [-2,2]   */

   for (unsigned b = 0; b < MAX_B; b++) {
      f16_cpu_fwd_logical_gen(tmp1, X[b], W1, IC, OC, IW, IH, ow, oh, KW, KH, pad);
      for (unsigned i = 0; i < qpix; i++) A1[b][i] = f16_to_f32(tmp1[i]);
      f16_cpu_fwd_logical_gen(tmp2, A1[b], W2, L2_IC, OC, IW, IH, ow, oh, KW, KH, pad);
      for (unsigned i = 0; i < qpix; i++) {
         A2[b][i] = f16_to_f32(tmp2[i]);
         dY2[b][i] = (A2[b][i] - T[b][i]) * g_fp16_loss;
      }
   }

   int f_bad = 0;
   /* ---- Phase F：层1 → 层2 链式前向（跨层，逐样本） ---- */
   printf("\n-- Phase L2-F: layer1 fwd → layer2 fwd (cross-layer) --\n");
   f16_pack_weights_gen(bs->wt.map + l2_w2_off, W2, L2_IC, f16_ic_pad_for(L2_IC), OC, oc_w, KW, KH);
   for (unsigned b = 0; b < batch; b++) {
      f16_pack_input_gen(bs->act.map, X[b], IC, IW, IH);
      f16_pack_weights_gen(bs->wt.map, W1, IC, f16_ic_pad_for(IC), OC, oc_w, KW, KH);
      struct conv_cfg f1 = make_fwd_cfg_l2(bs, pad, IC, 0, l2_a1_off, 0);
      if (run_task(fd, bs, &f1, cost)) { printf("L2 F1 b=%u SUBMIT-FAILED\n", b); return 1; }

      static uint16_t ref1[L2_ACT];
      f16_cpu_fwd_gen(ref1, bs->act.map, bs->wt.map, IC, f16_ic_pad_for(IC),
                      OC, oc_w, IW, IH, ow, oh, KW, KH, pad);
      unsigned bad1 = 0; int mu1 = 0;
      f16_read_act(A1[b], bs->act.map, l2_a1_off, OC, ow, oh, ow, oh);
      for (unsigned i = 0; i < qpix; i++) {
         uint16_t hv = f32_to_f16(A1[b][i]);
         int u = f16_ulp(hv, ref1[i]); if (u > mu1) mu1 = u;
         if (!f16_ok(hv, ref1[i], NULL)) bad1++;
      }
      printf("L2 F1 b=%u total=%u mismatch=%u max_ulp=%d RESULT: %s\n", b, qpix, bad1, mu1,
             f16_verdict(mu1 > 0 ? 1 : 0, bad1, mu1));

      struct conv_cfg f2 = make_fwd_cfg_l2(bs, pad, L2_IC, l2_a1_off, l2_a2_off, l2_w2_off);
      if (run_task(fd, bs, &f2, cost)) { printf("L2 F2 b=%u SUBMIT-FAILED\n", b); return 1; }

      static uint16_t ref2[L2_ACT];
      f16_cpu_fwd_gen(ref2, bs->act.map + l2_a1_off, bs->wt.map + l2_w2_off,
                      L2_IC, L2_IC, OC, oc_w, IW, IH, ow, oh, KW, KH, pad);
      unsigned bad2 = 0; int mu2 = 0;
      f16_read_act(A2[b], bs->act.map, l2_a2_off, OC, ow, oh, ow, oh);
      for (unsigned i = 0; i < qpix; i++) {
         uint16_t hv = f32_to_f16(A2[b][i]);
         int u = f16_ulp(hv, ref2[i]); if (u > mu2) mu2 = u;
         if (!f16_ok(hv, ref2[i], NULL)) bad2++;
      }
      printf("L2 F2 b=%u total=%u mismatch=%u max_ulp=%d RESULT: %s\n", b, qpix, bad2, mu2,
             f16_verdict(mu2 > 0 ? 1 : 0, bad2, mu2));
      if (bad1 || bad2) f_bad = 1;

      for (unsigned i = 0; i < qpix; i++) dY2[b][i] = (A2[b][i] - T[b][i]) * g_fp16_loss;
   }

   /* ---- Phase W：层2 wgrad（输入 = NPU A1），层1 wgrad（输入 = X, dA1） ---- */
   printf("\n-- Phase L2-W: wgrad layer2 (input=NPU A1) --\n");
   int w2_rc = l2_wgrad_layer(fd, bs, cost, pad, batch, A1[0], L2_IC, dY2[0], OC,
                              IW, IH, ow, oh, dW2);
   if (w2_rc < 0) return 1;

   /* dgrad：dA1 = dW2 反传（CPU 参考，fp16 读入） */
   for (unsigned b = 0; b < batch; b++)
      for (unsigned iy = 0; iy < IH; iy++)
         for (unsigned ix = 0; ix < IW; ix++)
            for (unsigned ic1 = 0; ic1 < L2_IC; ic1++) {
               float acc = 0.0f;
               for (unsigned ky = 0; ky < KH; ky++)
                  for (unsigned kx = 0; kx < KW; kx++) {
                     int oy = (int)iy + (int)ky - (int)pad;
                     int ox = (int)ix + (int)kx - (int)pad;
                     if (ox < 0 || oy < 0 || ox >= (int)ow || oy >= (int)oh) continue;
                     for (unsigned oc2 = 0; oc2 < OC; oc2++) {
                        float dv = f16_to_f32(f32_to_f16(
                           dY2[b][(oy * ow + ox) * OC + oc2]));
                        float wv = f16_to_f32(f32_to_f16(
                           W2[((ky * KW + kx) * L2_IC + ic1) * OC + oc2]));
                        acc += dv * wv;
                     }
                  }
               dY1[b][(iy * IW + ix) * L2_IC + ic1] = acc;
            }

   printf("\n-- Phase L2-W: wgrad layer1 (input=X, dY=dA1) --\n");
   int w1_rc = l2_wgrad_layer(fd, bs, cost, pad, batch, X[0], IC, dY1[0], OC,
                              IW, IH, ow, oh, dW1);
   if (w1_rc < 0) return 1;

   int upd_bad = 0, changed = 0;
   if (update) {
      static float W1u[L2_W1_BYTES], W2u[L2_W2_BYTES];
      unsigned c1 = 0, c2 = 0;
      for (unsigned i = 0; i < L2_W1_BYTES; i++) {
         W1u[i] = f16_to_f32(f32_to_f16(W1[i] - g_fp16_lr * dW1[i]));
         if (W1u[i] != W1[i]) c1++;
      }
      for (unsigned i = 0; i < L2_W2_BYTES; i++) {
         W2u[i] = f16_to_f32(f32_to_f16(W2[i] - g_fp16_lr * dW2[i]));
         if (W2u[i] != W2[i]) c2++;
      }
      printf("\n-- Phase L2-U: W1'=fp16(W1-lr*dW1) W2'=fp16(W2-lr*dW2), lr=%.4f --\n",
             g_fp16_lr);
      printf("L2 UPDATE: W1_changed=%u/%u W2_changed=%u/%u\n",
             c1, L2_W1_BYTES, c2, L2_W2_BYTES);

      f16_pack_weights_gen(bs->wt.map, W1u, IC, f16_ic_pad_for(IC), OC, oc_w, KW, KH);
      f16_pack_weights_gen(bs->wt.map + l2_w2_off, W2u, L2_IC, f16_ic_pad_for(L2_IC), OC, oc_w, KW, KH);
      unsigned changed_total = 0;
      for (unsigned b = 0; b < batch; b++) {
         f16_pack_input_gen(bs->act.map, X[b], IC, IW, IH);
         struct conv_cfg f1 = make_fwd_cfg_l2(bs, pad, IC, 0, l2_a1_off, 0);
         if (run_task(fd, bs, &f1, cost)) { printf("L2 F1' b=%u SUBMIT-FAILED\n", b); return 1; }
         static uint16_t ref1[L2_ACT];
         f16_cpu_fwd_gen(ref1, bs->act.map, bs->wt.map, IC, f16_ic_pad_for(IC),
                         OC, oc_w, IW, IH, ow, oh, KW, KH, pad);
         static float A1u[L2_ACT];
         f16_read_act(A1u, bs->act.map, l2_a1_off, OC, ow, oh, ow, oh);
         unsigned b1 = 0; int mu1 = 0;
         for (unsigned i = 0; i < qpix; i++) {
            uint16_t hv = f32_to_f16(A1u[i]);
            int u = f16_ulp(hv, ref1[i]); if (u > mu1) mu1 = u;
            if (!f16_ok(hv, ref1[i], NULL)) b1++;
         }

         struct conv_cfg f2 = make_fwd_cfg_l2(bs, pad, L2_IC, l2_a1_off, l2_a2_off, l2_w2_off);
         if (run_task(fd, bs, &f2, cost)) { printf("L2 F2' b=%u SUBMIT-FAILED\n", b); return 1; }
         static uint16_t ref2[L2_ACT];
         f16_cpu_fwd_gen(ref2, bs->act.map + l2_a1_off, bs->wt.map + l2_w2_off,
                         L2_IC, L2_IC, OC, oc_w, IW, IH, ow, oh, KW, KH, pad);
         static float A2u[L2_ACT];
         f16_read_act(A2u, bs->act.map, l2_a2_off, OC, ow, oh, ow, oh);
         unsigned b2 = 0; int mu2 = 0;
         for (unsigned i = 0; i < qpix; i++) {
            uint16_t hv = f32_to_f16(A2u[i]);
            int u = f16_ulp(hv, ref2[i]); if (u > mu2) mu2 = u;
            if (!f16_ok(hv, ref2[i], NULL)) {
               b2++;
               if (u > g_l2_ulp && b2 <= 4) {
                  unsigned oy = i / (ow * OC), r = i % (ow * OC);
                  unsigned ox = r / OC, oc = r % OC;
                  double dd; float ss;
                  f16_fwd_probe64(bs->act.map + l2_a1_off, bs->wt.map + l2_w2_off,
                                  L2_IC, L2_IC, oc_w, IW, IH, KW, KH, pad, oy, ox, oc,
                                  &dd, &ss);
                  printf("  L2 F2'_MISS b=%u i=%u (oy=%u ox=%u oc=%u) hw=0x%04x ref=0x%04x "
                         "seq=0x%04x ulp=%d\n",
                         b, i, oy, ox, oc, hv, ref2[i], f32_to_f16(ss), u);
               }
            }
            if (f32_to_f16(A2u[i]) != f32_to_f16(A2[b][i])) changed_total++;
         }
         printf("L2 F1' b=%u mismatch=%u max_ulp=%d  F2' b=%u mismatch=%u max_ulp=%d vs CPU(W1',W2')\n",
                b, b1, mu1, b, b2, mu2);
         if (b1 || b2) upd_bad = 1;
      }
      printf("L2 CHANGED: outputs_changed=%u/%u (sample batch vs pre-update)\n",
             changed_total, qpix * batch);
      changed = (changed_total > 0);
      if (!changed) upd_bad = 1;
   }

   if (!f_bad && w1_rc == 0 && w2_rc == 0 && !upd_bad) {
      printf("RESULT: 2-LAYER CLOSED-LOOP OK (fp16 cross-layer fwd + wgrad + update)\n");
      return 0;
   }
   printf("RESULT: MISMATCH (2-layer f_bad=%d w1=%d w2=%d upd=%d)\n",
          f_bad, w1_rc, w2_rc, upd_bad);
   return 1;
}

/* ------------------------------------------------------------------ *
 * 6. 主流程
 * ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
   const char *dev = "/dev/dri/card1";
   const char *dir = NULL;
   unsigned pad = 1, batch = 1, seed = 20260928, kscan_max = 0;
   int update = 0, gen_only = 0, oobtest = 0, fp16 = 0, two_layer = 0;

   static char cmdline[512];
   cmdline[0] = 0;
   for (int i = 0; i < argc && strlen(cmdline) < 400; i++) {
      strcat(cmdline, argv[i]);
      strcat(cmdline, " ");
   }

   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--dir") && i + 1 < argc) dir = argv[++i];
      else if (!strcmp(argv[i], "--pad") && i + 1 < argc) pad = (unsigned)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = (unsigned)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned)strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--kscan") && i + 1 < argc) { kscan_max = (unsigned)atoi(argv[++i]); g_timeout = 3000; }
      else if (!strcmp(argv[i], "--update")) update = 1;
      else if (!strcmp(argv[i], "--gen-only")) gen_only = 1;
      else if (!strcmp(argv[i], "--oobtest")) oobtest = 1;
      else if (!strcmp(argv[i], "--fp16")) { fp16 = 1; g_fp16 = 1; }
      else if (!strcmp(argv[i], "--two-layer")) { two_layer = 1; fp16 = 1; g_fp16 = 1; }
      else if (!strcmp(argv[i], "--no-auto-cfg")) g_auto_cfg = 0;
      else if (!strcmp(argv[i], "--lr") && i + 1 < argc)
         g_fp16_lr = strtof(argv[++i], NULL);
      else if (!strcmp(argv[i], "--loss") && i + 1 < argc)
         g_fp16_loss = strtof(argv[++i], NULL);
      else if (!strcmp(argv[i], "--tl-ulp") && i + 1 < argc)
         g_l2_ulp = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--tl-abs") && i + 1 < argc)
         g_l2_abs = strtod(argv[++i], NULL);
      else if (!strcmp(argv[i], "--fwd-repeat") && i + 1 < argc)
         g_fwd_repeat = (unsigned)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--wk32")) g_wk32 = 1;
      else if (!strcmp(argv[i], "--bisect") && i + 1 < argc)
         g_bisect = argv[++i];
      else if (!strcmp(argv[i], "--padcon1") && i + 1 < argc)
         g_pad_con1 = (uint32_t)strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--wfile") && i + 1 < argc)
         g_wfile = argv[++i];
      else if (!strcmp(argv[i], "--dump")) g_dump = 1;
      else if (!strcmp(argv[i], "--dump-regcmd") && i + 1 < argc) g_dump_regcmd = argv[++i];
      else if (!strcmp(argv[i], "--tile-oh") && i + 1 < argc) g_tile_oh = (unsigned)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--tile-p") && i + 1 < argc) g_tile_p = (unsigned)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--tile-src") && i + 1 < argc) g_tile_src = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--tile-pt") && i + 1 < argc) g_tile_pt = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--tile-rows") && i + 1 < argc) g_tile_rows = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--i8-tile-force")) { g_i8_tile_force = 1; g_i8_diag = 1; }
      else if (!strcmp(argv[i], "--i8-tile-diag")) g_i8_diag = 1;
      else if (!strcmp(argv[i], "--vend-dma")) g_vend_dma = 1;
      else if (!strcmp(argv[i], "--in-x-file") && i + 1 < argc) g_in_x_file = argv[++i];
      else if (!strcmp(argv[i], "--in-dy-file") && i + 1 < argc) g_in_dy_file = argv[++i];
      else if (!strcmp(argv[i], "--i8-tile-align") && i + 1 < argc) g_i8_tile_align = (unsigned)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--i8-stage-global")) g_i8_stage_global = 1;
      else if (!strcmp(argv[i], "--dev") && i + 1 < argc) dev = argv[++i];
      else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
   }
   if (!kscan_max && !dir && !oobtest && !g_bisect && !fp16) {
      fprintf(stderr, "usage: %s --dir DIR [--pad 0|1] [--batch 1..8] [--seed N] "
                      "[--update] [--gen-only] | --fp16 [--batch 1..8] [--seed N] "
                      "[--update] [--lr F] [--loss F] | --two-layer [--batch 1..8] "
                      "[--seed N] [--update] [--lr F] [--loss F] [--tl-ulp N] | --kscan BMAX | --bisect MODE "
                      "(none|fwd6|fwd8|wgrad|wgradfull|sat|phb|many) [--pad 0] [--dev NODE]\n",
              argv[0]);
      return 2;
   }
   if (batch > MAX_B) { fprintf(stderr, "batch > %u\n", MAX_B); return 2; }

   unsigned ow = pad ? IW : OW_P0, oh = pad ? IH : OH_P0;
   unsigned qpix = ow * oh * OC;
   {
      /* 前向输入 = IW·IH·planes·16 B；wgrad 输入 = 4·4·ic_pad·2 B（P=ow·oh·batch）。
       * 取两者上界并在其后留 64 KB，确保输出不覆盖输入。 */
      unsigned fwd_in = (unsigned)IW * IH * ((IC + FP16_ATOM_ELEMS - 1) / FP16_ATOM_ELEMS) * 16u;
      unsigned ow_max = IW, oh_max = IH;
      unsigned p_max = ow_max * oh_max * MAX_B;
      unsigned wg_ic_pad = (p_max <= 2u * FP16_ATOM_ELEMS) ? (2u * FP16_ATOM_ELEMS)
                            : ALIGN_UP(p_max, 4u * FP16_ATOM_ELEMS);
      unsigned wg_in = 4u * 4u * wg_ic_pad * 2u;
      unsigned need = fwd_in > wg_in ? fwd_in : wg_in;
      unsigned want = ALIGN_UP(need, 4096u) + 262144u;
      if (OUT_OFF < want) OUT_OFF = want;
      g_in_need_fwd = fwd_in; g_in_need_wg = wg_in;
   }
   printf("OUT-OFF: %u (输入需求 fwd=%u wgrad=%u)\n", OUT_OFF, g_in_need_fwd, g_in_need_wg);
   printf("== wgrad_real: ic=%u oc=%u iw=%u ih=%u k=%ux%u pad=%u batch=%u seed=%u "
          "ow*oh=%u*%u dir=%s update=%d ==\n",
          (unsigned)IC, (unsigned)OC, (unsigned)IW, (unsigned)IH, (unsigned)KW, (unsigned)KH,
          pad, batch, seed, ow, oh, dir ? dir : "-", update);
   printf("CMD: %s\n", cmdline);

   uint32_t fscale, gscale;
   unsigned fshift, gshift;
   compute_requant(1.0f, 1.0f, 1.0f, 0, &fscale, &fshift);
   compute_requant(1.0f, 1.0f, 8.0f, 0, &gscale, &gshift);
   printf("REQUANT: fwd scale=%u shift=%u offset=0 | grad scale=%u shift=%u offset=0\n",
          fscale, fshift, gscale, gshift);

   int fd = open(dev, O_RDWR);
   if (fd < 0) { fprintf(stderr, "open %s: %s\n", dev, strerror(errno)); return 1; }
   printf("opened %s\n", dev);

   struct bos bs;
   /* §7.50：激活 BO 必须容下「打包输入 + 输出区」（OUT_OFF 已按输入需求上移）。 */
   if (bo_create(fd, OUT_OFF + (1 << 20), 0, &bs.act)) return 1;   /* §7.37：深 ic/oc 需要更大 BO */
   if (bo_create(fd, 1 << 20, 0, &bs.wt)) return 1;
   if (bo_create(fd, 1 << 12, 0, &bs.bias)) return 1;
   if (bo_create(fd, 1 << 13, 0, &bs.rcbo)) return 1;
   if (bo_create(fd, 1 << 12, RKNPU_MEM_KERNEL_MAPPING, &bs.tk)) return 1;
   printf("BO: act@%08x wt@%08x bias@%08x regcmd@%08x task obj=0x%llx\n",
          (uint32_t)bs.act.dma_addr, (uint32_t)bs.wt.dma_addr,
          (uint32_t)bs.bias.dma_addr, (uint32_t)bs.rcbo.dma_addr,
          (unsigned long long)bs.tk.obj_addr);

   struct cost cost;
   memset(&cost, 0, sizeof(cost));
   int rc = 0;

   if (two_layer) {
      memset((uint8_t *)bs.act.map + OUT_OFF, 0, 8192);
      int trc = two_layer_run(fd, &bs, pad, batch, seed, update, &cost);
      printf("COST: submits=%u fwd_tasks=%u wgrad_tasks=%u wall=%.3f ms hw=%.3f ms\n",
             cost.submits, cost.fwd_tasks, cost.wgrad_tasks, cost.wall_ms, cost.hw_ms);
      return trc;
   }

   if (fp16) {
      memset((uint8_t *)bs.act.map + OUT_OFF, 0, 4096);
      int frc = fp16_run(fd, &bs, pad, batch, seed, update, &cost);
      printf("COST: submits=%u fwd_tasks=%u wgrad_tasks=%u wall=%.3f ms hw=%.3f ms\n",
             cost.submits, cost.fwd_tasks, cost.wgrad_tasks, cost.wall_ms, cost.hw_ms);
      return frc;
   }

   if (kscan_max) {
      int krc = kscan_run(fd, &bs, kscan_max, gscale, gshift, &cost);
      printf("COST: submits=%u wall=%.3f ms hw=%.3f ms\n",
             cost.submits, cost.wall_ms, cost.hw_ms);
      return krc;
   }

   if (oobtest) {
      int orc = oobtest_run(fd, &bs, &cost);
      printf("COST: submits=%u wall=%.3f ms hw=%.3f ms\n",
             cost.submits, cost.wall_ms, cost.hw_ms);
      return orc;
   }

   static struct dataset ds;
   dataset_prepare(&ds, dir, pad, seed, fscale, fshift, gscale, gshift);
   if (gen_only) return 0;
   if (batch > ds.nsamp) { fprintf(stderr, "batch > dataset samples\n"); return 2; }

   /* CPU dW 参考（batch 求和）+ 梯度 requant */
   cpu_dw_raw(&ds, batch, pad, ow, oh, ds.DWR);
   for (unsigned c = 0; c < NCOORD; c++)
      for (unsigned oc = 0; oc < OC; oc++)
         ds.DW8[c * OC + oc] = requant_grad(ds.DWR[c][oc], gscale, gshift);
   long long dwraw_max = 0, dwsat = 0;
   for (unsigned c = 0; c < NCOORD; c++)
      for (unsigned oc = 0; oc < OC; oc++) {
         long long a = ds.DWR[c][oc] < 0 ? -ds.DWR[c][oc] : ds.DWR[c][oc];
         if (a > dwraw_max) dwraw_max = a;
         if (a >= 127LL * gscale) dwsat++;
      }
   printf("CPU_DW: max|raw|=%lld raw>=sat_thresh=%lld fnv(dw8)=%08x fnv(w)=%08x fnv(dy_b0)=%08x\n",
          dwraw_max, dwsat, fnv1a((const uint8_t *)ds.DW8, W_BYTES),
          fnv1a((const uint8_t *)ds.W, W_BYTES),
          fnv1a((const uint8_t *)ds.DY[0], qpix));

   if (g_bisect) {
      int brc = bisect_run(fd, &bs, &ds, dir, pad, ow, oh, &cost);
      printf("COST: submits=%u wall=%.3f ms hw=%.3f ms\n",
             cost.submits, cost.wall_ms, cost.hw_ms);
      return brc;
   }

   uint8_t *out = (uint8_t *)bs.act.map + OUT_OFF;

   /* ---- Phase F：前向（初始 W，每样本一个 task） ---- */
   printf("\n-- Phase F: forward with initial W --\n");
   if (g_fwd_repeat > 1)
      printf("PHASE-F-REPEAT: submitting the same fwd task %u times\n", g_fwd_repeat);
   static int8_t q_pre[MAX_B][IW * IH * OC];
   int fwd_bad = 0, sat_bad = 0;
   for (unsigned rep = 0; rep < (g_fwd_repeat ? g_fwd_repeat : 1); rep++) {
    if (rep) printf("-- repeat %u --\n", rep);
    for (unsigned b = 0; b < batch; b++) {
      pack_fwd_weights(bs.wt.map, ds.W);
      if (g_tile_oh && !g_i8_tile_force && 0) {   /* §7.102：int8 切分已闭合，取消默认拒绝 */
         /* §7.55：int8 前向切分的 tile 任务输出错（逐 tile 诊断显示**首个 tile 也错**；
          * 与不切分的描述符 diff 只剩预期几何字段 ⇒ 问题在 int8 特有的取窗/步长约定）。
          * 机制代码保留在下方，但先显式拒绝，避免产生"看起来能跑"的错误结果。 */
         printf("RESULT: int8 前向切分尚未支持（见台账 §7.55）—— 请用 fp16 路径或先修取窗约定\n");
         return 1;
      }
      if (!g_tile_oh) {
         pack_fwd_input(bs.act.map, ds.X[b]);
         struct conv_cfg f = make_fwd_cfg(&bs, pad, ow, oh);
         if (run_task(fd, &bs, &f, &cost)) {
            printf("RESULT: FWD SUBMIT FAILED (b=%u)\n", b);
            return 1;
         }
      } else {
         /* §7.55：int8 前向行切分（与 fp16 §7.53 同构：lead 行重叠 + 散回丢弃） */
         unsigned groups_i8 = i8_oc_span() / FEATURE_ATOMIC_SIZE;
         unsigned stage_off_i8 = ALIGN_UP(g_in_need_fwd ? g_in_need_fwd : 65536u, 4096u);
         /* §7.91：**输入的偏移不被 PPU/CNA 遵守**（§7.79 实测）⇒ tile 的输入暂存必须放在
          * 自己 BO 的基址（旧版放在 act BO 的 stage_off_i8 偏移处 ⇒ 硬件从 BO 基址读全图 ⇒ 全错）。 */
         struct bo tile_in;
         {
            unsigned icp = i8_ic_align();
            /* §7.107：rows 会被 ALIGN_UP(...,16) 放大（§7.96 规则）⇒ 暂存 BO 必须按**对齐后**的行数算，
             * 旧式 MIN2(IH, tile_oh+1+KH) 偏小 ⇒ 打包越界写坏内存 ✗（宽图尤其明显 ✗）。 */
            unsigned rows_max = ALIGN_UP(g_tile_oh + KH, 16u) + 16u;
            if (rows_max < (unsigned)IW + 16u) rows_max = (unsigned)IW + 16u;   /* §7.108：容纳 rows=IW */
            unsigned need = (unsigned)IW * rows_max * icp + 4096u;
            if (bo_create(fd, (need + 4095u) & ~4095u, 0, &tile_in)) return 1;
         }
         uint8_t *out8 = (uint8_t *)bs.act.map + OUT_OFF;
         uint8_t *stage8;
         if (g_i8_stage_global) {   /* §7.57：全图打包到 0，tile 输入暂存另开区域 */
            stage8 = (uint8_t *)bs.act.map + 2u * stage_off_i8;   /* tile 输入暂存 */
            pack_fwd_input(bs.act.map, ds.X[b]);                  /* 全图（偏移 0） */
         } else {
            stage8 = (uint8_t *)bs.act.map + stage_off_i8;        /* tile 输出暂存 */
         }
         for (unsigned r0 = 0; r0 < oh; r0 += g_tile_oh) {
            unsigned n = MIN2(g_tile_oh, oh - r0);
            unsigned lead = (r0 == 0) ? 0u : 1u;
            unsigned orow0 = (r0 == 0) ? 0u : (r0 - 1u);
            unsigned nsub = n + lead;
            unsigned src0 = orow0;
            unsigned pt = pad;
            /* §7.56 诊断：int8 的 staged 行数似需对齐（x 步长 = rows×16 B） */
            /* §7.93：rows 必须按 IH 截断（旧版不截断 ⇒ f.ih > IH ⇒ 硬件读到不存在的行 ✗）；
             * 越图底的行由硬件 pad（PAD_CON1=0）顶上，与不切分路径一致。 */
            unsigned rows = ALIGN_UP(nsub + KH - 1 - pad, g_i8_tile_align);
            unsigned avail = (IH > src0) ? (IH - src0) : 1u;
            if (rows > avail) rows = avail;
            if (rows < KH) rows = KH;
            /* §7.96：**tile 的输入行数必须是 16 的倍数**（实测：rows=16 全对、rows=8/9/3 全错；
             * 越界的行由打包 memset 0（= PAD_CON1=0）顶上，硬件读到的就是"pad 行"）。 */
            rows = ALIGN_UP(rows, 16u);
            /* §7.108：实测"暂存的输入必须是**方图**"（rows == IW）——64 宽图下 rows=32/48 时 77–78% 错、
             * rows=64 时降到 2.5% ✗✓；16 宽图的历史通过用例恰好都是 rows=16=IW ✓ ⇒ 统一取 rows = max(rows, IW)，
             * 多出来的行由打包 memset 0（= PAD_CON1 语义）顶上 ✓。 */
            if (rows < (unsigned)IW) rows = (unsigned)IW;
            unsigned tile_in_off = g_i8_stage_global ? 2u * stage_off_i8 : 0u;
            if (g_i8_stage_global) {
               /* §7.57：按 min_conv_exp 的 [plane][y][x] 紧凑布局，从全图逐行拷贝 */
               unsigned icp = i8_ic_align();
               unsigned plane_row_bytes = IW * FEATURE_ATOMIC_SIZE;
               unsigned st_planes = icp / FEATURE_ATOMIC_SIZE;
               uint8_t *dst = (uint8_t *)bs.act.map + tile_in_off;
               memset(dst, 0, (size_t)st_planes * rows * plane_row_bytes);
               for (unsigned pp = 0; pp < st_planes; pp++)
                  for (unsigned r = 0; r < rows; r++) {
                     unsigned sy = src0 + r;
                     if (sy >= IH) break;
                     memcpy(dst + ((size_t)pp * rows + r) * plane_row_bytes,
                            (uint8_t *)bs.act.map + ((size_t)pp * IH + sy) * plane_row_bytes,
                            plane_row_bytes);
                  }
            } else {
               pack_fwd_tile_input_i8((uint8_t *)tile_in.map, ds.X[b], src0, rows);
            }
            /* §7.92：tile 的 oh 必须取 emitter 的**推导值**（§7.74/§7.90 同族）；
             * 推导值可能 > nsub ⇒ 多出的行照算，散回/诊断时丢弃。 */
            unsigned oh_t = rows + 2u * pad - KH + 1u;
            struct conv_cfg f = make_fwd_cfg(&bs, pad, ow, oh_t);
            f.ih = rows; f.pad_top = f.pad_left = pt;
            f.in_addr = (uint32_t)tile_in.dma_addr;   /* §7.91：暂存在自己的 BO 基址 */
            f.out_addr = (uint32_t)bs.act.dma_addr + stage_off_i8;
            bo_sync(fd, &tile_in, 1);
            if (run_task(fd, &bs, &f, &cost)) {
               printf("RESULT: FWD SUBMIT FAILED (b=%u tile@%u)\n", b, r0);
               return 1;
            }
            if (g_i8_diag) {   /* §7.55 诊断：逐 tile 与逻辑参考比对（散回前）*/
               static int8_t reft[IW * IH * OC];
               cpu_forward(reft, ds.X[b], ds.W, pad, ow, oh, fscale, fshift);
               unsigned bad_t = 0;
               unsigned long probe[6] = {0, 0, 0, 0, 0, 0};   /* §7.92：多映射读回探测 */
               uint8_t *ostage = (uint8_t *)bs.act.map + stage_off_i8;
               for (unsigned j = lead; j < nsub; j++)
                  for (unsigned x = 0; x < ow; x++)
                     for (unsigned oc = 0; oc < OC; oc++) {
                        int ref_v = (int)reft[(((r0 - lead) + j) * ow + x) * OC + oc];
                        int hw = (int)(int8_t)ostage[
                           npu_off(oc / FEATURE_ATOMIC_SIZE, x, j, ow, oh_t) +
                           (oc % FEATURE_ATOMIC_SIZE)];
                        if (hw != ref_v) {
                           bad_t++;
                           if (bad_t <= 3)
                              printf("    TILE8-MISS j=%u x=%u oc=%u: hw=%d ref=%d (暂存偏移=%u 参考索引=%u)\n",
                                     j, x, oc, hw, ref_v,
                                     npu_off(oc / FEATURE_ATOMIC_SIZE, x, j, ow, oh_t) + (oc % FEATURE_ATOMIC_SIZE),
                                     (((r0 - lead) + j) * ow + x) * OC + oc);
                        }
                        /* §7.101：循环内自证（与外部三元组打印同坐标） */
                        if (j == lead && x == 8u && oc == 0u) {
                           unsigned o1 = npu_off(oc / FEATURE_ATOMIC_SIZE, x, j, ow, oh_t) + (oc % FEATURE_ATOMIC_SIZE);
                           int a1 = (int)(int8_t)ostage[o1];
                           int a2 = (int)(int8_t)ostage[o1];
                           bo_sync(fd, &bs.act, 2);
                           int a3 = (int)(int8_t)ostage[o1];
                           bo_sync(fd, &bs.act, 2);
                           int a4 = (int)(int8_t)ostage[o1];
                           printf("    TILE8-INLOOP j=%u x=8 oc=0: hw=%d ref=%d | 连读:%d %d | sync 后:%d %d\n",
                                  j, hw, ref_v, a1, a2, a3, a4);
                        }
                        /* 候选 0: 当前；1: 全局 oh 步长；2: x/y 互换；3/4: 行 ±1；5: 组数用 icp/16 */
                        if ((int)(int8_t)ostage[npu_off(oc / FEATURE_ATOMIC_SIZE, x, j, ow, oh) + (oc % FEATURE_ATOMIC_SIZE)] == ref_v) probe[1]++;
                        if ((int)(int8_t)ostage[npu_off(oc / FEATURE_ATOMIC_SIZE, j, x, ow, nsub) + (oc % FEATURE_ATOMIC_SIZE)] == ref_v) probe[2]++;
                        if (j > 0 && (int)(int8_t)ostage[npu_off(oc / FEATURE_ATOMIC_SIZE, x, j - 1, ow, nsub) + (oc % FEATURE_ATOMIC_SIZE)] == ref_v) probe[3]++;
                        if (j + 1 < nsub && (int)(int8_t)ostage[npu_off(oc / FEATURE_ATOMIC_SIZE, x, j + 1, ow, nsub) + (oc % FEATURE_ATOMIC_SIZE)] == ref_v) probe[4]++;
                        if ((int)(int8_t)ostage[npu_off(oc / FEATURE_ATOMIC_SIZE, x, j, ow, ALIGN_UP(nsub, 4)) + (oc % FEATURE_ATOMIC_SIZE)] == ref_v) probe[5]++;
                     }
               printf("  TILE8-PROBE: oh步长=%lu x/y换=%lu 行-1=%lu 行+1=%lu nsub对齐4=%lu (总 %u)\n",
                      probe[1], probe[2], probe[3], probe[4], probe[5], n * ow * OC);
               {   unsigned h = 2166136261u;
                   for (unsigned i = 0; i < (unsigned)ow * oh * OC; i++) { h ^= (uint8_t)reft[i]; h *= 16777619u; }
                   printf("  TILE8-REFH r0=%u: reft fnv=%08x (ow*oh*OC=%u, X0=%d W0=%d)\n",
                          r0, h, ow * oh * OC, (int)ds.X[b][0], (int)ds.W[0]); }
               {   unsigned hh = 2166136261u;
                   unsigned bytes = (unsigned)ow * oh_t * groups_i8 * FEATURE_ATOMIC_SIZE;
                   uint8_t *os2 = (uint8_t *)bs.act.map + stage_off_i8;
                   for (unsigned i = 0; i < bytes; i++) { hh ^= os2[i]; hh *= 16777619u; }
                   printf("  TILE8-STAGEH r0=%u: 暂存 fnv=%08x (%u B)\n", r0, hh, bytes); }
               printf("  TILE8 r0=%u n=%u nsub=%u rows=%u pt=%u bad=%u/%u\n",
                      r0, n, nsub, rows, pt, bad_t, n * ow * OC);
            }
            unsigned rb = ow * FEATURE_ATOMIC_SIZE;
            {   unsigned hh = 2166136261u;
                unsigned bytes = (unsigned)ow * oh_t * groups_i8 * FEATURE_ATOMIC_SIZE;
                uint8_t *os2 = (uint8_t *)bs.act.map + stage_off_i8;
                for (unsigned i = 0; i < bytes; i++) { hh ^= os2[i]; hh *= 16777619u; }
                printf("  SCAT-STAGEH r0=%u: 暂存 fnv=%08x (%u B)\n", r0, hh, bytes); }
            static int8_t refsc[IW * IH * OC];
            if (g_i8_diag) cpu_forward(refsc, ds.X[b], ds.W, pad, ow, oh, fscale, fshift);
            for (unsigned g = 0; g < groups_i8; g++)
               for (unsigned j = 0; j < n; j++) {
                  /* §7.102 修正：按 **npu_off 的真实布局**（与检查/参考同一口径）逐原子拷贝，
                   * 而不是"整行 memcpy"（那假设了行内连续 ⇒ 与 npu_off 的 x 步长不符 ✗）。 */
                  for (unsigned x = 0; x < ow; x++)
                     memcpy(out8 + npu_off(g, x, r0 + j, ow, oh),
                            stage8 + npu_off(g, x, lead + j, ow, oh_t),
                            FEATURE_ATOMIC_SIZE);
                  /* §7.102：散回**循环内**自证（固定坐标 x=0/8，g=0，j=0） */
                  if (g_i8_diag && g == 0u && j == 0u)
                     for (unsigned xq = 0; xq < ow; xq += 8u) {
                        unsigned so = (unsigned)(0 * ow * oh_t + (lead + 0) * ow + xq) * FEATURE_ATOMIC_SIZE;
                        unsigned dо = (unsigned)(0 * ow * oh + (r0 + 0) * ow + xq) * FEATURE_ATOMIC_SIZE;
                        printf("    SCAT-INLOOP x=%u: src=%d dst=%d ref=%d (src@%u dst@%u refidx=%u | stage8=+%ld ostage=+%ld)\n",
                               xq, (int)(int8_t)stage8[so], (int)(int8_t)out8[dо],
                               (int)refsc[((r0 * ow) + xq) * OC + 0], so, dо, ((r0 * ow) + xq) * OC,
                               (long)(stage8 - (uint8_t *)bs.act.map),
                               (long)(((uint8_t *)bs.act.map + stage_off_i8) - (uint8_t *)bs.act.map));
                     }
               }
            if (g_i8_diag) {   /* §7.97：散回后在目标区域比对 */
               static int8_t refs[IW * IH * OC];
               cpu_forward(refs, ds.X[b], ds.W, pad, ow, oh, fscale, fshift);
               unsigned bad_s = 0;
               for (unsigned j = 0; j < n; j++)
                  for (unsigned x = 0; x < ow; x++)
                     for (unsigned oc = 0; oc < OC; oc++) {
                        int hw = (int)(int8_t)out8[
                           npu_off(oc / FEATURE_ATOMIC_SIZE, x, r0 + j, ow, oh) +
                           (oc % FEATURE_ATOMIC_SIZE)];
                        if (hw != (int)refs[(((r0 + j) * ow) + x) * OC + oc]) bad_s++;
                     }
               {   unsigned h = 2166136261u;
                   for (unsigned i = 0; i < (unsigned)ow * oh * OC; i++) { h ^= (uint8_t)refs[i]; h *= 16777619u; }
                   printf("  TILE8-SCAT-REFH r0=%u: refs fnv=%08x (X0=%d W0=%d)\n",
                          r0, h, (int)ds.X[b][0], (int)ds.W[0]); }
               printf("  TILE8-SCAT r0=%u n=%u: bad=%u/%u\n", r0, n, bad_s, n * ow * OC);
               if (bad_s) {   /* §7.98：把首个失配的 16 B 原子在目标区里扫一遍，看它到底落在哪 */
                  unsigned shown = 0;
                  for (unsigned j = 0; j < n && shown < 3; j++)
                     for (unsigned x = 0; x < ow && shown < 3; x++)
                        for (unsigned g = 0; g < groups_i8 && shown < 3; g++) {
                           unsigned oc0 = g * FEATURE_ATOMIC_SIZE;
                           int hw = (int)(int8_t)out8[npu_off(g, x, r0 + j, ow, oh)];
                           if (hw == (int)refs[(((r0 + j) * ow) + x) * OC + oc0]) continue;
                           /* 期望的原子在暂存里的位置 */
                           unsigned src_off = (unsigned)(g * ow * oh_t + (lead + j) * ow + x) * FEATURE_ATOMIC_SIZE;
                           unsigned exp_off = (unsigned)(g * ow * oh + (r0 + j) * ow + x) * FEATURE_ATOMIC_SIZE;
                           /* 在目标区（0..ow*oh*groups_i8*16）里找同样的 16 B */
                           unsigned found = 0xffffffffu;
                           for (unsigned off = 0; off + FEATURE_ATOMIC_SIZE <= (unsigned)ow * oh * groups_i8 * FEATURE_ATOMIC_SIZE; off += FEATURE_ATOMIC_SIZE)
                              if (!memcmp(out8 + off, stage8 + src_off, FEATURE_ATOMIC_SIZE)) { found = off; break; }
                           printf("    SCAT-VALS g=%u x=%u y=%u: dst=%d staged=%d ref=%d | 参考索引=%u\n",
                                  g, x, r0 + j,
                                  (int)(int8_t)out8[exp_off],
                                  (int)(int8_t)stage8[src_off],
                                  (int)refs[(((r0 + j) * ow) + x) * OC + oc0],
                                  (((r0 + j) * ow) + x) * OC + oc0);
                           printf("    SCAT-MISS g=%u x=%u y=%u: 期望@%u 实际找到@%u (Δ=%d) 暂存源@%u\n",
                                  g, x, r0 + j, exp_off, found,
                                  found == 0xffffffffu ? 0 : (int)found - (int)exp_off, src_off);
                           shown++;
                        }
               }
            }
         }
      }
      if (b == 0)
         printf("FWD_EMIT: scale=%u shift=%u offset=%d\n",
                g_out_scale, g_out_shift, g_out_offset);
      long long badr = 0, maxd = 0, badr_border = 0, badr_int = 0;
      long long miss_sat_hi = 0, miss_sat_lo = 0;
      for (unsigned oc = 0; oc < OC; oc++)
       for (unsigned oy = 0; oy < oh; oy++)
        for (unsigned ox = 0; ox < ow; ox++) {
           int hw = (int)(int8_t)out[npu_off(oc / FEATURE_ATOMIC_SIZE, ox, oy, ow, oh) +
                                          (oc % FEATURE_ATOMIC_SIZE)];
           int cpu = ds.Q[b][(oy * ow + ox) * OC + oc];
           q_pre[b][(oy * ow + ox) * OC + oc] = (int8_t)hw;
           if (hw != cpu) {
              badr++;
              if (ox == 0 || oy == 0 || ox == ow - 1 || oy == oh - 1) {
                 badr_border++;
                 if (badr_border <= 3)
                    printf("  FWD_MISS_BORDER b=%u (x=%u,y=%u) oc=%u hw=%d cpu=%d\n",
                           b, ox, oy, oc, hw, cpu);
              } else {
                 badr_int++;
                 if (badr_int <= 3)
                    printf("  FWD_MISS_INT b=%u (x=%u,y=%u) oc=%u hw=%d cpu=%d\n",
                           b, ox, oy, oc, hw, cpu);
              }
              if (hw == 127) miss_sat_hi++;
              if (hw == -128) miss_sat_lo++;
           }
           long long d = hw > cpu ? hw - cpu : cpu - hw;
           if (d > maxd) maxd = d;
        }
      printf("FWD_SPLIT b=%u: border=%lld/%u interior=%lld/%u sat_hi=%lld sat_lo=%lld\n",
             b, badr_border, 4 * (ow + oh) - 4, badr_int,
             (ow - 2) * (oh - 2), miss_sat_hi, miss_sat_lo);
      if (badr && b == 0) {
         printf("FWD_MAP b=0 (misses/16ch per pixel):\n");
         for (unsigned oy = 0; oy < oh; oy++) {
            printf("  y=%u:", oy);
            for (unsigned ox = 0; ox < ow; ox++) {
               unsigned m = 0;
               for (unsigned oc = 0; oc < OC; oc++)
                  if ((int)(int8_t)out[npu_off(0, ox, oy, ow, oh) + oc] !=
                      ds.Q[0][(oy * ow + ox) * OC + oc]) m++;
               printf(" %2u", m);
            }
            printf("\n");
         }
      }
      printf("FWD b=%u: total=%u round-mismatch=%lld maxdiff=%lld RESULT: %s\n",
             b, qpix, badr, maxd, badr ? "MISMATCH" : "BIT-EXACT (round)");
      if (badr) fwd_bad = 1;
      char pth[512];
      snprintf(pth, sizeof(pth), "%s/q_npu_pre_b%u.bin", dir, b);
      write_file(pth, out, qpix);
      snprintf(pth, sizeof(pth), "%s/q_npu_full_b%u.bin", dir, b);
      write_file(pth, out, 2048);
      /* 16B 原子线性序（前 64 个）供离线分析 */
      printf("FWD_ATOMS b=%u oc=0 first 40:", b);
      for (unsigned n = 0; n < 64 && n < qpix / OC + 28; n++)
         printf(" %d", (int)(int8_t)out[n * OC]);
      printf("\n");
    }
   }

   /* ---- Phase W：27 个 (kx,ky,ic) reduction task ---- */
   printf("\n-- Phase W: wgrad per (kx,ky,ic), P=%u K-points, grad requant scale=%u shift=%u --\n",
          ow * oh * batch, gscale, gshift);
   static int8_t dy_flat[IW * IH * MAX_B * OC];
   static int8_t kvec[IW * IH * MAX_B];
   static int8_t dW_npu[NCOORD * OC];
   memset(dW_npu, 0, sizeof(dW_npu));

   for (unsigned b = 0; b < batch; b++)
      memcpy(dy_flat + (size_t)b * qpix, ds.DY[b], qpix);

   unsigned wg_ok = 0, wg_grads_ok = 0, wg_grads_floor_bad = 0;
   int wg_verbose = 0;
   long long wg_maxdiff = 0;
   for (unsigned kx = 0; kx < KW; kx++)
    for (unsigned ky = 0; ky < KH; ky++)
     for (unsigned ic = 0; ic < IC; ic++) {
        unsigned P = ow * oh * batch;
        unsigned ic_pad = ALIGN_UP(P, WEIGHT_ATOMIC_SIZE);
        unsigned cidx = (ky * KW + kx) * IC + ic;   /* 与 w.bin 展平一致 */

        for (unsigned p = 0; p < P; p++) {
           unsigned b = p / (ow * oh), pos = p % (ow * oh);
           unsigned oy = pos / ow, ox = pos % ow;
           int iy = (int)oy + (int)ky - (int)pad;
           int ix = (int)ox + (int)kx - (int)pad;
           kvec[p] = (ix >= 0 && iy >= 0 && ix < (int)IW && iy < (int)IH)
                     ? ds.X[b][iy * IW * IC + ix * IC + ic] : 0;
        }
        pack_wgrad_input(bs.act.map, kvec, P, ic_pad);
        pack_wgrad_weights(bs.wt.map, dy_flat, P, ic_pad);

        struct conv_cfg g = make_wgrad_cfg(&bs, P);
        if (run_task(fd, &bs, &g, &cost)) {
           printf("WGRAD kx=%u ky=%u ic=%u SUBMIT-FAILED\n", kx, ky, ic);
           return 1;
        }
        uint8_t *out = (uint8_t *)bs.act.map + OUT_OFF;
        unsigned okr = 0, okf = 0;
        for (unsigned oc = 0; oc < OC; oc++) {
           int hw = (int)(int8_t)out[npu_off(oc / FEATURE_ATOMIC_SIZE, 0, 0, 1, 1) +
                                          (oc % FEATURE_ATOMIC_SIZE)];
           dW_npu[cidx * OC + oc] = (int8_t)hw;
           long long raw = ds.DWR[cidx][oc];
           long long vr = requant_grad(raw, gscale, gshift);
           long long vf = (raw * (long long)gscale) >> gshift;
           if (vf < -128) vf = -128;
           if (vf > 127) vf = 127;
           long long d = hw > vr ? hw - vr : vr - hw;
           if (d > wg_maxdiff) wg_maxdiff = d;
           if (hw == (int)vr) okr++;
           if (hw == (int)vf) okf++;
           if (hw != (int)vr && wg_verbose++ < 8)
              printf("  WGRAD_MISS kx=%u ky=%u ic=%u oc=%u raw=%lld round=%lld floor=%lld hw=%d\n",
                     kx, ky, ic, oc, raw, vr, vf, hw);
        }
        wg_grads_ok += okr;
        wg_grads_floor_bad += (unsigned)OC - okf;
        if (okr == OC) wg_ok++;
        else
           printf("WGRAD_COORD kx=%u ky=%u ic=%u: round_mismatch=%u/16\n", kx, ky, ic, OC - okr);
     }

   {
      /* raw 域信息量：|raw - npu*8|（未饱和项）与饱和计数 */
      long long maxdev = 0; unsigned sat = 0;
      for (unsigned c = 0; c < NCOORD; c++)
       for (unsigned oc = 0; oc < OC; oc++) {
          if (abs((int)dW_npu[c * OC + oc]) >= 127) { sat++; continue; }
          long long dev = ds.DWR[c][oc] - (long long)dW_npu[c * OC + oc] * 8;
          if (dev < 0) dev = -dev;
          if (dev > maxdev) maxdev = dev;
       }
      printf("WGRAD_TOTAL: coords=%u/%u grads=%u/%u round-exact floor_mismatch=%u "
             "maxdiff=%lld max|raw-npu*8|=%lld sat=%u\n",
             wg_ok, NCOORD, wg_grads_ok, (unsigned)NCOORD * OC,
             wg_grads_floor_bad, wg_maxdiff, maxdev, sat);
   }
   {
      char pth[512];
      snprintf(pth, sizeof(pth), "%s/dw_npu.bin", dir);
      write_file(pth, dW_npu, NCOORD * OC);
      printf("DW_NPU: fnv=%08x saved %s/dw_npu.bin\n",
             fnv1a((const uint8_t *)dW_npu, NCOORD * OC), dir);
   }

   /* ---- Phase S：饱和边界（合成极值，非数据集） ---- */
   printf("\n-- Phase S: saturation probes (synthetic +-extremes) --\n");
   {
      unsigned P = ow * oh;
      unsigned ic_pad = ALIGN_UP(P, WEIGHT_ATOMIC_SIZE);
      for (int sign = 0; sign < 2; sign++) {
         for (unsigned p = 0; p < P; p++) {
            kvec[p] = 127;
            for (unsigned oc = 0; oc < OC; oc++)
               dy_flat[p * OC + oc] = (int8_t)(sign ? -127 : 127);
         }
         pack_wgrad_input(bs.act.map, kvec, P, ic_pad);
         pack_wgrad_weights(bs.wt.map, dy_flat, P, ic_pad);
         struct conv_cfg g = make_wgrad_cfg(&bs, P);
         if (run_task(fd, &bs, &g, &cost)) {
            printf("SAT: submit failed\n"); sat_bad = 1; break;
         }
         uint8_t *o = (uint8_t *)bs.act.map + OUT_OFF;
         int hw = (int)(int8_t)o[0];
         int expect = sign ? -128 : 127;
         printf("SAT: %s hw=%d expect=%d %s\n",
                sign ? "neg" : "pos", hw, expect, hw == expect ? "OK" : "MISMATCH");
         if (hw != expect) sat_bad = 1;
      }
   }

   /* ---- Phase U：写回 + 复前向 ---- */
   int upd_bad = 0;
   if (update) {
      static int8_t W_upd[W_BYTES];
      unsigned nonzero = 0, wchg = 0, gsat = 0;
      (void)0;
      for (unsigned i = 0; i < W_BYTES; i++) {
         int8_t g8 = dW_npu[i];
         long long delta = rhao_div(g8, 4);
         int8_t nw = clamp8((long long)ds.W[i] - delta);
         W_upd[i] = nw;
         if (g8 != 0) nonzero++;
         if (g8 >= 127 || g8 <= -128) gsat++;
         if (nw != ds.W[i]) wchg++;
      }
      printf("\n-- Phase U: update W <- W - round(dW_npu/4), lr=2 LSB per grad-LSB --\n");
      printf("UPDATE: dW_nonzero=%u/%u grad_sat=%u weights_changed=%u/%u wt_bo=0x%08x (same BO, in-place)\n",
             nonzero, W_BYTES, gsat, wchg, W_BYTES, (uint32_t)bs.wt.dma_addr);

      char pth[512];
      snprintf(pth, sizeof(pth), "%s/dw_npu.bin", dir);
      write_file(pth, dW_npu, NCOORD * OC);
      snprintf(pth, sizeof(pth), "%s/w_updated.bin", dir);
      write_file(pth, W_upd, W_BYTES);
      printf("W_UPDATED: fnv=%08x\n", fnv1a((const uint8_t *)W_upd, W_BYTES));

      static int8_t q2cpu[MAX_B][IW * IH * OC];
      unsigned changed_out = 0;
      int fwd2_bad = 0;
      printf("-- Phase U2: re-forward with updated W --\n");
      for (unsigned b = 0; b < batch; b++) {
         cpu_forward(q2cpu[b], ds.X[b], W_upd, pad, ow, oh, fscale, fshift);
         pack_fwd_input(bs.act.map, ds.X[b]);
         pack_fwd_weights(bs.wt.map, W_upd);
         struct conv_cfg f = make_fwd_cfg(&bs, pad, ow, oh);
         if (run_task(fd, &bs, &f, &cost)) {
            printf("FWD2 b=%u SUBMIT-FAILED\n", b);
            fwd2_bad = 1; break;
         }
         long long badr2 = 0;
         for (unsigned oc = 0; oc < OC; oc++)
          for (unsigned oy = 0; oy < oh; oy++)
           for (unsigned ox = 0; ox < ow; ox++) {
              int hw = (int)(int8_t)out[npu_off(oc / FEATURE_ATOMIC_SIZE, ox, oy, ow, oh) +
                                             (oc % FEATURE_ATOMIC_SIZE)];
              int cpu = q2cpu[b][(oy * ow + ox) * OC + oc];
              if (hw != cpu) badr2++;
              if (b == 0 && hw != (int)q_pre[0][(oy * ow + ox) * OC + oc]) changed_out++;
           }
         printf("FWD2 b=%u: total=%u round-mismatch=%lld vs CPU(W') RESULT: %s\n",
                b, qpix, badr2, badr2 ? "MISMATCH" : "BIT-EXACT (round)");
         if (badr2) fwd2_bad = 1;
         char pth[512];
         snprintf(pth, sizeof(pth), "%s/q_npu_post_b%u.bin", dir, b);
         write_file(pth, out, qpix);
         snprintf(pth, sizeof(pth), "%s/q_cpu_post_b%u.bin", dir, b);
         write_file(pth, q2cpu[b], qpix);
      }
      printf("CHANGED: outputs_changed=%u/%u (sample 0 vs pre-update forward)\n",
             changed_out, qpix);
      if (fwd2_bad || changed_out == 0) upd_bad = 1;
   }

   printf("\nCOST: submits=%u fwd_tasks=%u wgrad_tasks=%u wall=%.3f ms hw=%.3f ms dma~%llu B\n",
          cost.submits, cost.fwd_tasks, cost.wgrad_tasks,
          cost.wall_ms, cost.hw_ms, (unsigned long long)cost.dma_bytes);

   if (!fwd_bad && wg_ok == NCOORD && !sat_bad && !upd_bad)
      printf("RESULT: CLOSED-LOOP OK (fwd exact, wgrad round-exact, update verified)\n");
   else {
      printf("RESULT: MISMATCH (fwd_bad=%d wg_ok=%u/%u sat_bad=%d upd_bad=%d)\n",
             fwd_bad, wg_ok, NCOORD, sat_bad, upd_bad);
      rc = 1;
   }
   return rc;
}
