/*
 * min_conv.c — 手写最小 CONV / DECONV 寄存器程序（不依赖任何 NPU 运行时库）
 *
 * 目的（对应文档《RKNPU 逆向-算子构建与训练可行性》§8 阶段 1/2）：
 *   1. 只用「寄存器定义 + 内核 UAPI」两样东西，手工发射一个 3x3 / stride=1 /
 *      pad=1 的卷积 regcmd，提交给 vendor RKNPU 驱动，与 CPU 参考对照。
 *   2. 在同一路径上置位 CNA_CONV_CON1_DECONV 与 CONV_CON3_DECONV_X/Y_STRIDE，
 *      观察硬件是否真的走转置卷积。
 *
 * 依赖：
 *   - rnpu_regmap.h   Mesa rules-ng-ng 生成的寄存器映射（唯一复用物，属硬件规格）
 *   - 本文件内的 rknpu UAPI 结构体副本（照抄 rockchip-linux/rknpu 的 rknpu_ioctl.h）
 *
 * 编译： gcc -O2 -o min_conv min_conv.c -lm
 * 运行： ./min_conv [--mode conv|deconv] [--iw N --ih N --ow N --oh N]
 *                 [--ic N --oc N] [--in-byte B] [--w-byte B] [--pad P]
 *                 [--deconv-stride N] [--dump]
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
#include <time.h>

#include "rnpu_regmap.h"

/* ------------------------------------------------------------------ *
 * 1. 内核 UAPI（照抄 vendor rknpu_ioctl.h，仅保留本实验用到的部分）
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

struct rknpu_mem_destroy {
   uint32_t handle;
   uint32_t reserved;
   uint64_t obj_addr;
};

struct rknpu_mem_sync {
   uint32_t flags;
   uint32_t reserved;
   uint64_t obj_addr;
   uint64_t offset;
   uint64_t size;
};

struct rknpu_task {              /* 40 bytes, packed */
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

/* PC_DATA_EXTRA_AMOUNT: regcmd 末尾 4 条由驱动自己编程，不计入 regcfg_amount */
#define RKNPU_PC_DATA_EXTRA_AMOUNT 4

/* ------------------------------------------------------------------ *
 * 2. 硬件常量
 * ------------------------------------------------------------------ */
#define FEATURE_ATOMIC_SIZE 16
#define WEIGHT_ATOMIC_SIZE  32
#define ATOMIC_K_SIZE       16
#define CBUF_BANKS          12

#define ALIGN_UP(x, a)  (((x) + (a) - 1) & ~((a) - 1))
#define MAX2(a, b)      ((a) > (b) ? (a) : (b))
#define MIN2(a, b)      ((a) < (b) ? (a) : (b))
#define DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))

/* ------------------------------------------------------------------ *
 * 3. regcmd 发射层
 * ------------------------------------------------------------------ */
static uint32_t g_out_scale, g_out_shift;
static int32_t  g_out_offset;
static uint64_t rc_buf[4096];
static uint64_t *rc = rc_buf;
static int g_dump = 0;

/* --- 实验开关（min_conv_exp 新增） ---
 * g_w_kernels_override : CNA_WEIGHT_SIZE2.WEIGHT_KERNELS，0 = 默认 ALIGN_UP(MAX2(oc,32),2)
 * g_w_oc2_override     : 每个 tap 的 oc 跨度（CLI --w-oc2），0 = 默认 ALIGN_UP(oc_real,2)
 * g_w_layout           : 权重区打包嵌套顺序
 *                        0 = A: [oc1][i1][kx][ky][oc2][i2]（librocketnpu 原生写法）
 *                        1 = C: [kx][ky][i1][oc1][oc2][i2]
 *                        2 = D: [kx][ky][oc1][oc2][i1][i2]
 */
static int g_w_kernels_override = 0;
static int g_w_layout = 0;
static int g_ic_pad_override = 0;   /* 强制 CNA 的 16/32 对齐通道数 */
static int g_hi_zero = 0;           /* pattern 模式下把 ch>=ic_real 的输入字节清零 */
static int g_icreal_reg = -1;       /* 覆盖 DATAIN_CHANNEL_REAL 寄存器（真实值-1） */
static int g_entries_override = -1; /* 覆盖 CNA_CBUF_CON1.DATA_ENTRIES */
static int g_autofix = 0;           /* 1 = 采用本次实测得到的修正口径 */
static int g_relu = 0;              /* 0=bypass, 1=ReLU, 2=ReLU-X(leaky) */
static uint32_t g_relux_cmp = 0;    /* BS_RELUX_CMP_DAT */

static void emit_raw(uint32_t target, uint32_t reg, uint32_t value)
{
   *rc++ = ((uint64_t)target << 48) | ((uint64_t)value << 16) | (uint64_t)reg;
}

/* vendor 编码 = Mesa 单元号 | 0x1 */
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

/* 一个 task 的全部参数 */
struct conv_cfg {
   unsigned iw, ih, ic_real, ic;          /* 输入 W/H/真实通道/16 对齐通道 */
   unsigned ow, oh, oc_real, oc;          /* 输出 W/H/真实通道/32 对齐通道 */
   unsigned kw, kh;
   unsigned stride_x, stride_y;
   unsigned pad_left, pad_top;
   unsigned input_zero_point, weights_zero_point, output_zero_point;
   float in_scale, w_scale, out_scale;
   unsigned truncate_bits;
   int deconv;
   unsigned deconv_stride;
   int depthwise;
   int pad_con1_override;                 /* 显式覆盖值（配合 pad_con1_given） */
   int pad_con1_given;                    /* 1 = 使用 pad_con1_override，0 = 用库公式 */
   uint32_t in_addr, out_addr, wt_addr, bias_addr;
};

/* 完全按文档 §3.2 的寄存器表手工组装，写成一段线性序列 */
static unsigned emit_conv(const struct conv_cfg *c)
{
   unsigned input_line_stride  = c->iw * ATOMIC_K_SIZE / 4;
   unsigned input_surf_stride  = (int)((float)input_line_stride *
                                       (((float)c->ih / 4.0f) - 1.0f));
   unsigned output_surf_stride = (c->ow * ATOMIC_K_SIZE) * c->oh / FEATURE_ATOMIC_SIZE;
   /* DATA_ENTRIES：本次实测确认要按 16 对齐后的通道数算，而不是 ic_real */
   unsigned input_data_entries = DIV_ROUND_UP(
         c->iw * 2 * DIV_ROUND_UP(g_autofix ? c->ic : c->ic_real,
                                  FEATURE_ATOMIC_SIZE), 8);
   unsigned atomic_count       = c->ow * c->oh;
   unsigned weights_kernels    = g_w_kernels_override > 0
                                 ? (unsigned)g_w_kernels_override
                                 : ALIGN_UP(MAX2(c->oc, WEIGHT_ATOMIC_SIZE), 2);
   unsigned surfaces_per_row   = c->ow * c->oh * 2;

   /* CBUF bank 规划：输入 bank 数（此处输入很小，退化为 1）+ 权重 bank 数 */
   unsigned wbytes = c->kw * c->kh * c->ic * (c->depthwise ? 1 : c->oc);
   unsigned weights_banks = DIV_ROUND_UP(DIV_ROUND_UP(wbytes, 128), 256) + 1;
   unsigned input_banks   = 1;
   if (c->depthwise) input_banks = 7;
   unsigned avail = CBUF_BANKS - weights_banks;
   if (input_banks > avail) input_banks = avail;
   weights_banks = CBUF_BANKS - input_banks;

   int offset = (int)c->output_zero_point - 0x80;

   /* ---- CNA ---- */
   EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(weights_banks) |
                           CNA_CBUF_CON0_DATA_BANK(input_banks));
   EMIT(REG_CNA_DCOMP_REGNUM, 0);
   EMIT(REG_CNA_DCOMP_CTRL, 0);

   uint32_t con1 = 0;
   if (c->deconv)             con1 |= CNA_CONV_CON1_DECONV(1);
   if (c->ic_real == 1)       con1 |= CNA_CONV_CON1_NONALIGN_DMA(1) |
                                     CNA_CONV_CON1_GROUP_LINE_OFF(1) |
                                     CNA_CONV_CON1_ARGB_IN(8);
   if (c->depthwise)          con1 |= CNA_CONV_CON1_CONV_MODE(3);
   EMIT(REG_CNA_CONV_CON1, con1);
   EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) |
                           DPU_S_POINTER_EXECUTER_PP_EN(1) |
                           DPU_S_POINTER_POINTER_PP_EN(1));
   EMIT(REG_DPU_RDMA_RDMA_S_POINTER, DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(1) |
                                     DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(1) |
                                     DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(1));
   EMIT(REG_CNA_CONV_CON1, con1);

   EMIT(REG_CNA_CONV_CON2, CNA_CONV_CON2_FEATURE_GRAINS(50 + c->stride_y + 1));
   uint32_t con3 = CNA_CONV_CON3_CONV_X_STRIDE(c->stride_x) |
                   CNA_CONV_CON3_CONV_Y_STRIDE(c->stride_y);
   if (c->deconv && c->deconv_stride)
      con3 |= CNA_CONV_CON3_DECONV_X_STRIDE(c->deconv_stride) |
              CNA_CONV_CON3_DECONV_Y_STRIDE(c->deconv_stride);
   EMIT(REG_CNA_CONV_CON3, con3);

   EMIT(REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH(c->iw) |
                            CNA_DATA_SIZE0_DATAIN_HEIGHT(c->ih));
   EMIT(REG_CNA_DATA_SIZE1,
        CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(g_icreal_reg >= 0 ? (unsigned)g_icreal_reg
                                                             : c->ic_real - 1) |
        CNA_DATA_SIZE1_DATAIN_CHANNEL(c->ic));
   EMIT(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(c->ow));
   EMIT(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(atomic_count));

   EMIT(REG_CNA_WEIGHT_SIZE0, c->kw * c->kh * c->ic * weights_kernels);
   EMIT(REG_CNA_WEIGHT_SIZE1, c->kw * c->kh * c->ic);
   EMIT(REG_CNA_WEIGHT_SIZE2, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(c->kw) |
                              CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(c->kh) |
                              CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(weights_kernels));

   EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(weights_banks) |
                           CNA_CBUF_CON0_DATA_BANK(input_banks));
   EMIT(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(
           g_entries_override >= 0 ? (unsigned)g_entries_override : input_data_entries));

   if (c->ic_real == 1) {
      unsigned truncate = 14, scale = 16384, cvt_off = 65408;
      EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_CVT_TRUNCATE_3(truncate) |
                             CNA_CVT_CON0_CVT_TRUNCATE_2(truncate) |
                             CNA_CVT_CON0_CVT_TRUNCATE_1(truncate) |
                             CNA_CVT_CON0_CVT_TRUNCATE_0(truncate));
      EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(scale) | CNA_CVT_CON1_CVT_OFFSET0(cvt_off));
      EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(scale) | CNA_CVT_CON2_CVT_OFFSET1(cvt_off));
      EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(scale) | CNA_CVT_CON3_CVT_OFFSET2(cvt_off));
      EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(scale) | CNA_CVT_CON4_CVT_OFFSET3(cvt_off));
   } else {
      EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_DATA_SIGN(1) | CNA_CVT_CON0_CVT_TYPE(1) |
                             CNA_CVT_CON0_CVT_BYPASS(1));
      EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
      EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
      EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
      EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
   }

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
   EMIT(REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH(c->iw) |
                               CNA_FC_DATA_SIZE0_DMA_HEIGHT(c->ih));
   EMIT(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(c->ic));
   EMIT(REG_CNA_DCOMP_CTRL, 0);
   EMIT(REG_CNA_DCOMP_REGNUM, 0);
   EMIT(REG_CNA_DCOMP_ADDR0, c->wt_addr);
   for (int i = 0; i < 16; i++) EMIT(REG_CNA_DCOMP_AMOUNT0 + i * 4, 0);
   EMIT(REG_CNA_CVT_CON5, c->ic_real == 1 ? 65535 : 0);

   int32_t pad_con1;
   if (c->pad_con1_given) pad_con1 = c->pad_con1_override;
   else {
      if (c->kw >= 3 && c->input_zero_point == 0x0) pad_con1 = 0xffff8080;
      else pad_con1 = (int32_t)c->input_zero_point - 0x80;
      if (c->depthwise && c->input_zero_point == 0x8b) pad_con1 = 0x0b0b;
   }
   EMIT(REG_CNA_PAD_CON1, (uint32_t)pad_con1);

   /* ---- CORE ---- */
   uint32_t misc_cfg = CORE_MISC_CFG_QD_EN(1);
   if (c->depthwise) misc_cfg |= CORE_MISC_CFG_DW_EN(1);
   EMIT(REG_CORE_MISC_CFG, misc_cfg);
   EMIT(REG_CORE_DATAOUT_SIZE_0, CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(c->oh - 1) |
                                 CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(c->ow - 1));
   EMIT(REG_CORE_DATAOUT_SIZE_1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(c->oc - 1));
   EMIT(REG_CORE_CLIP_TRUNCATE, CORE_CLIP_TRUNCATE_CLIP_TRUNCATE(c->truncate_bits));
   emit_raw(CORE | 0x1, 0x3030, 0);

   /* ---- DPU ---- */
   uint32_t fmc = DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2);
   if (c->depthwise) fmc |= DPU_FEATURE_MODE_CFG_CONV_MODE(3);
   EMIT(REG_DPU_FEATURE_MODE_CFG, fmc);
   EMIT(REG_DPU_DATA_FORMAT, 0);
   EMIT(REG_DPU_OFFSET_PEND, 0);
   EMIT(REG_DPU_DST_BASE_ADDR, c->out_addr);
   EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(output_surf_stride));
   EMIT(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(c->ow - 1));
   EMIT(REG_DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_HEIGHT_HEIGHT(c->oh - 1));
   EMIT(REG_DPU_DATA_CUBE_NOTCH_ADDR, 0);
   EMIT(REG_DPU_DATA_CUBE_CHANNEL,
        DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(c->oc_real - 1) |
        DPU_DATA_CUBE_CHANNEL_CHANNEL(c->oc - 1));
   /* 厂商 ReLU 实例（attn_relu 探针）：0x00020010 = ALU_ALGO(2) | MUL_BYPASS(1) | RELU 开启 */
   EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_ALU_ALGO(2) |
                        DPU_BS_CFG_BS_ALU_SRC(g_relu ? 0 : 1) |
                        (g_relu ? 0 : DPU_BS_CFG_BS_RELU_BYPASS(1)) |
                        (g_relu == 2 ? DPU_BS_CFG_BS_RELUX_EN(1) : 0) |
                        DPU_BS_CFG_BS_MUL_BYPASS(1));
   EMIT(REG_DPU_BS_ALU_CFG, 0);
   EMIT(REG_DPU_BS_MUL_CFG, 0);
   EMIT(REG_DPU_BS_RELUX_CMP_VALUE, g_relux_cmp);
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
   EMIT(REG_DPU_OUT_CVT_OFFSET, (uint32_t)offset);

   /* requant: conv_scale -> (scale, shift)，与 rnpu_coefs.c 同一套定点化 */
   float conv_scale = (c->in_scale * c->w_scale) / c->out_scale;
   uint32_t scale_bits = fui(conv_scale);
   unsigned shift = 127 + 31 - 32 - (scale_bits >> 23) + 16;
   if (c->truncate_bits > 0) shift--;
   unsigned scale = ((scale_bits >> 9) & 0x7fff) + 1;
   if (scale < (1u << 14)) scale |= 1u << 14;
   g_out_scale = scale; g_out_shift = shift - 1; g_out_offset = offset;
   EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(scale));
   EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(shift - 1));

   for (int i = 0; i < 8; i++) EMIT(REG_DPU_EW_OP_VALUE_0 + i * 4, 0);
   EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(surfaces_per_row));
   emit_raw(DPU | 0x1, 0x40c4, 0);
   for (int i = 0; i < 12; i++) EMIT(REG_DPU_LUT_ACCESS_CFG + i * 4, 0);

   /* ---- DPU_RDMA ---- */
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
        DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE(1));
   EMIT(REG_DPU_RDMA_RDMA_SRC_DMA_CFG, 0);
   EMIT(REG_DPU_RDMA_RDMA_SURF_NOTCH, 0);
   EMIT(REG_DPU_RDMA_RDMA_PAD_CFG, 0);
   EMIT(REG_DPU_RDMA_RDMA_WEIGHT, DPU_RDMA_RDMA_WEIGHT_E_WEIGHT(1) |
                                  DPU_RDMA_RDMA_WEIGHT_N_WEIGHT(1) |
                                  DPU_RDMA_RDMA_WEIGHT_B_WEIGHT(1) |
                                  DPU_RDMA_RDMA_WEIGHT_M_WEIGHT(1));
   EMIT(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH, 0);

   /* ---- 收尾：链指针占位 + 触发标记 ---- */
   EMIT(REG_PC_BASE_ADDRESS, 0);
   EMIT(REG_PC_REGISTER_AMOUNTS, 0);
   *rc++ = 0x0041000000000000ULL;
   emit_raw(0x81, REG_PC_OPERATION_ENABLE,
            PC_OPERATION_ENABLE_RESERVED_0(14) | PC_OPERATION_ENABLE_OP_EN(1));

   return (unsigned)(rc - rc_buf);
}

/* ------------------------------------------------------------------ *
 * 4. BO 管理
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

/* ------------------------------------------------------------------ *
 * 5. 数据布局：NC1HWC2（group-major, x-major, y, c16）
 * ------------------------------------------------------------------ */
static uint32_t npu_off(unsigned g, unsigned x, unsigned y, unsigned w, unsigned h)
{
   return g * w * h * FEATURE_ATOMIC_SIZE + x * h * FEATURE_ATOMIC_SIZE +
          y * FEATURE_ATOMIC_SIZE;
}

static int g_pattern = 0;
static int g_deconv_flip = 0;

static int8_t pat_in(unsigned x, unsigned y, unsigned ch)
{
   return (int8_t)(((int)(x * 7 + y * 5 + ch * 3) % 5) - 2);
}
static int8_t pat_w(unsigned x, unsigned y, unsigned oc, unsigned icc)
{
   return (int8_t)(((int)(x + y + oc + icc) % 3) - 1);
}

static void fill_input(uint8_t *buf, const struct conv_cfg *c,
                       uint8_t byte, uint8_t pad_byte)
{
   unsigned groups = DIV_ROUND_UP(c->ic, FEATURE_ATOMIC_SIZE);
   for (unsigned g = 0; g < groups; g++)
      for (unsigned x = 0; x < c->iw; x++)
         for (unsigned y = 0; y < c->ih; y++) {
            uint8_t *p = buf + npu_off(g, x, y, c->iw, c->ih);
            for (unsigned k = 0; k < FEATURE_ATOMIC_SIZE; k++) {
               unsigned ch = g * FEATURE_ATOMIC_SIZE + k;
               if (g_pattern && g_hi_zero && ch >= c->ic_real)
                  p[k] = 0;
               else if (g_pattern)
                  p[k] = (uint8_t)pat_in(x, y, ch);
               else
                  p[k] = (ch < c->ic_real) ? byte : pad_byte;
            }
         }
}

/* 权重打包：按 16 通道组与 32 输出通道组展开，布局由 g_w_layout 选择。
 *   oc_w  = 每个 tap 的 oc 跨度（= WEIGHT_KERNELS，必须与描述符一致）
 *   ic1/ic2 = 输入通道的组数 / 组内宽度
 *   oc1n/oc2n = oc 的组数 / 组内宽度
 * ic >= ic_real 的槽位填权重零点（本实验 wzp=0 -> 0x00，即不贡献）*/
static unsigned fill_weights(uint8_t *dst, const struct conv_cfg *c, uint8_t w_byte,
                             int oc_w_override)
{
   unsigned oc_w = oc_w_override > 0 ? (unsigned)oc_w_override
                                     : ALIGN_UP(c->oc_real, 2);
   unsigned ic1  = DIV_ROUND_UP(c->ic, WEIGHT_ATOMIC_SIZE);
   unsigned ic2  = MIN2(c->ic, WEIGHT_ATOMIC_SIZE);
   unsigned oc1n = DIV_ROUND_UP(oc_w, WEIGHT_ATOMIC_SIZE);
   unsigned oc2n = MIN2(oc_w, WEIGHT_ATOMIC_SIZE);
   unsigned n = 0;

   for (unsigned oc1 = 0; oc1 < oc1n; oc1++)
    for (unsigned i1 = 0; i1 < ic1; i1++)
     for (unsigned x = 0; x < c->kw; x++)
      for (unsigned y = 0; y < c->kh; y++)
       for (unsigned oc2 = 0; oc2 < oc2n; oc2++)
        for (unsigned i2 = 0; i2 < ic2; i2++) {
           unsigned oc  = oc1 * WEIGHT_ATOMIC_SIZE + oc2;
           unsigned icc = i1 * WEIGHT_ATOMIC_SIZE + i2;
           unsigned idx;

           if (oc >= oc_w) continue;

           switch (g_w_layout) {
           case 1:  /* C: [kx][ky][i1][oc1][oc2][i2] */
              idx = ((((x * c->kh + y) * ic1 + i1) * oc1n + oc1) * oc2n + oc2) * ic2 + i2;
              break;
           case 2:  /* D: [kx][ky][oc1][oc2][i1][i2] */
              idx = ((((x * c->kh + y) * oc1n + oc1) * oc2n + oc2) * ic1 + i1) * ic2 + i2;
              break;
           default: /* A: [oc1][i1][kx][ky][oc2][i2]，librocketnpu 原生写法 */
              idx = ((((oc1 * ic1 + i1) * c->kw + x) * c->kh + y) * oc2n + oc2) * ic2 + i2;
              break;
           }

           if (oc >= c->oc_real) dst[idx] = 0x00;
           else if (icc >= c->ic_real) dst[idx] = 0x00;  /* 权重零点 */
           else dst[idx] = g_pattern ? (uint8_t)pat_w(x, y, oc, icc) : w_byte;
           n++;
        }
   return n;
}


int main(int argc, char **argv)
{
   struct conv_cfg c = {
      .iw = 8, .ih = 8, .ic_real = 3, .ic = 16,
      .ow = 8, .oh = 8, .oc_real = 16, .oc = 32,
      .kw = 3, .kh = 3, .stride_x = 1, .stride_y = 1,
      .pad_left = 1, .pad_top = 1,
      .input_zero_point = 0, .weights_zero_point = 0, .output_zero_point = 0,
      .in_scale = 1.0f, .w_scale = 1.0f, .out_scale = 1.0f,
      .truncate_bits = 0, .deconv = 0, .deconv_stride = 0, .depthwise = 0,
      .pad_con1_override = -1,
   };
   const char *dev = "/dev/dri/card1";
   uint8_t in_byte = 0x01, w_byte = 0x01, pad_byte = 0xff;
   int pad_byte_given = 0;
   uint32_t enable_mask = 0xf;
   int iw_given = 0, ow_given = 0;
   int mode_deconv = 0, w_uniform = 0, w_oc2 = 0;

   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--dump")) g_dump = 1;
      else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
         i++; if (!strcmp(argv[i], "deconv")) mode_deconv = 1;
      }
      else if (!strcmp(argv[i], "--iw") && i + 1 < argc) { c.iw = atoi(argv[++i]); iw_given = 1; }
      else if (!strcmp(argv[i], "--ih") && i + 1 < argc) c.ih = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--ow") && i + 1 < argc) { c.ow = atoi(argv[++i]); ow_given = 1; }
      else if (!strcmp(argv[i], "--oh") && i + 1 < argc) c.oh = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--ic") && i + 1 < argc) c.ic_real = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--oc") && i + 1 < argc) c.oc_real = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--pad") && i + 1 < argc) { c.pad_left = c.pad_top = atoi(argv[++i]); }
      else if (!strcmp(argv[i], "--stride") && i + 1 < argc) { c.stride_x = c.stride_y = atoi(argv[++i]); }
      else if (!strcmp(argv[i], "--in-byte") && i + 1 < argc) in_byte = (uint8_t)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--w-byte") && i + 1 < argc) w_byte = (uint8_t)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--pad-byte") && i + 1 < argc) {
         pad_byte = (uint8_t)strtol(argv[++i], NULL, 0); pad_byte_given = 1;
      }
      else if (!strcmp(argv[i], "--pad-con1") && i + 1 < argc) {
         c.pad_con1_override = (int)(uint32_t)strtoul(argv[++i], NULL, 0);
         c.pad_con1_given = 1;
      }
      else if (!strcmp(argv[i], "--w-uniform")) w_uniform = 1;
      else if (!strcmp(argv[i], "--pattern")) g_pattern = 1;
      else if (!strcmp(argv[i], "--deconv-flip")) g_deconv_flip = 1;
      else if (!strcmp(argv[i], "--enable-mask") && i + 1 < argc) enable_mask = (uint32_t)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--w-oc2") && i + 1 < argc) w_oc2 = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--w-kernels") && i + 1 < argc) g_w_kernels_override = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--ic-pad") && i + 1 < argc) g_ic_pad_override = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--hi-zero")) g_hi_zero = 1;
      else if (!strcmp(argv[i], "--icreal-reg") && i + 1 < argc) g_icreal_reg = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--entries") && i + 1 < argc) g_entries_override = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--auto-fix")) g_autofix = 1;
      else if (!strcmp(argv[i], "--out-zp") && i + 1 < argc) c.output_zero_point = (unsigned)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--relu")) g_relu = 1;
      else if (!strcmp(argv[i], "--relux")) { g_relu = 2; if (i + 1 < argc) g_relux_cmp = (uint32_t)strtol(argv[++i], NULL, 0); }
      else if (!strcmp(argv[i], "--w-layout") && i + 1 < argc) {
         const char *m = argv[++i];
         if (!strcmp(m, "a") || !strcmp(m, "A")) g_w_layout = 0;
         else if (!strcmp(m, "c") || !strcmp(m, "C")) g_w_layout = 1;
         else if (!strcmp(m, "d") || !strcmp(m, "D")) g_w_layout = 2;
         else { fprintf(stderr, "unknown w-layout %s (a|c|d)\n", m); return 2; }
      }
      else if (!strcmp(argv[i], "--deconv-stride") && i + 1 < argc) {
         c.deconv_stride = atoi(argv[++i]); c.deconv = 1;
      }
      else if (!strcmp(argv[i], "--dev") && i + 1 < argc) dev = argv[++i];
      else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
   }

   c.ic = ALIGN_UP(MAX2(c.ic_real, FEATURE_ATOMIC_SIZE), FEATURE_ATOMIC_SIZE);
   if (g_autofix) {
      /* 修正口径：ic>16 时权重区按 32 对齐；每 tap 的 oc 跨度 = WEIGHT_KERNELS */
      c.ic = (c.ic_real <= FEATURE_ATOMIC_SIZE)
             ? FEATURE_ATOMIC_SIZE : ALIGN_UP(c.ic_real, WEIGHT_ATOMIC_SIZE);
      if (w_oc2 == 0)
         w_oc2 = (c.oc_real <= WEIGHT_ATOMIC_SIZE)
                 ? (int)ALIGN_UP(c.oc_real, 2)
                 : (int)ALIGN_UP(c.oc_real, WEIGHT_ATOMIC_SIZE);
      if (g_w_kernels_override == 0) g_w_kernels_override = w_oc2;
      g_w_layout = 0;
   }
   if (g_ic_pad_override > 0) c.ic = (unsigned)g_ic_pad_override;
   c.oc = ALIGN_UP(MAX2(c.oc_real, 32), 32);
   if (mode_deconv && !ow_given) { c.ow = c.iw; }
   if (mode_deconv && !iw_given) { /* keep */ }
   if (!ow_given && !mode_deconv) c.ow = c.iw;
   if (!ow_given && mode_deconv && c.deconv_stride) c.ow = c.iw * c.deconv_stride;
   if (!iw_given) { /* keep */ }

   printf("== min_conv: iw=%u ih=%u ic=%u(real %u) -> ow=%u oh=%u oc=%u(real %u) "
          "k=%ux%u s=%u pad=%u deconv=%d/%u in=0x%02x w=0x%02x ==\n",
          c.iw, c.ih, c.ic, c.ic_real, c.ow, c.oh, c.oc, c.oc_real,
          c.kw, c.kh, c.stride_x, c.pad_left, c.deconv, c.deconv_stride,
          in_byte, w_byte);

   int fd = open(dev, O_RDWR);
   if (fd < 0) { fprintf(stderr, "open %s: %s\n", dev, strerror(errno)); return 1; }
   printf("opened %s\n", dev);

   struct bo act, wt, bias, rcbo, tk;
   if (bo_create(fd, 1 << 16, 0, &act)) return 1;
   if (bo_create(fd, 1 << 16, 0, &wt)) return 1;   /* 实验：支持 ic_real=64 的 18432 B 权重区 */
   if (bo_create(fd, 1 << 12, 0, &bias)) return 1;
   if (bo_create(fd, 1 << 13, 0, &rcbo)) return 1;
   if (bo_create(fd, 1 << 12, RKNPU_MEM_KERNEL_MAPPING, &tk)) return 1;

   const uint32_t OUT_OFF = 32768;

   /* 输入 */
   uint32_t in_size = c.iw * c.ih * DIV_ROUND_UP(c.ic, FEATURE_ATOMIC_SIZE)
                      * FEATURE_ATOMIC_SIZE;
   if (!pad_byte_given) pad_byte = 0x80;
   fill_input((uint8_t *)act.map, &c, in_byte, pad_byte);

   /* 权重：默认按 NPU 打包布局填写；--w-uniform 则整段同字节 */
   unsigned wskernels = g_w_kernels_override > 0
                        ? (unsigned)g_w_kernels_override
                        : ALIGN_UP(MAX2(c.oc, WEIGHT_ATOMIC_SIZE), 2);
   uint32_t wt_size = c.kw * c.kh * c.ic * wskernels;
   if (w_uniform) {
      memset(wt.map, w_byte, wt_size);
   } else {
      memset(wt.map, 0, wt_size);
      unsigned wn = fill_weights((uint8_t *)wt.map, &c, w_byte, w_oc2);
      unsigned oc_w_eff = w_oc2 > 0 ? (unsigned)w_oc2 : ALIGN_UP(c.oc_real, 2);
      {
         const uint8_t *wp = (const uint8_t *)wt.map;
         uint32_t h = 2166136261u;
         for (uint32_t k = 0; k < wt_size; k++) { h ^= wp[k]; h *= 16777619u; }
         printf("weights packed: %u B of descriptor %u B  [w_kernels=%u w_oc2=%u layout=%c] fnv1a=%08x\n",
                wn, wt_size, wskernels, oc_w_eff, "ACD"[g_w_layout], h);
      }
   }

   /* 偏置：全 0 */
   memset(bias.map, 0, bias.size);

   c.in_addr   = (uint32_t)act.dma_addr;
   c.out_addr  = (uint32_t)act.dma_addr + OUT_OFF;
   c.wt_addr   = (uint32_t)wt.dma_addr;
   c.bias_addr = (uint32_t)bias.dma_addr;

   printf("BO: act@%08x(0x%zx) wt@%08x bias@%08x regcmd@%08x task obj=0x%llx\n",
          (uint32_t)act.dma_addr, (size_t)act.size, (uint32_t)wt.dma_addr,
          (uint32_t)bias.dma_addr, (uint32_t)rcbo.dma_addr,
          (unsigned long long)tk.obj_addr);
   printf("layout: input=%u B (groups=%u) output@+%u  weight_desc=%u B\n",
          in_size, DIV_ROUND_UP(c.ic, FEATURE_ATOMIC_SIZE), OUT_OFF, wt_size);

   unsigned n = emit_conv(&c);
   printf("regcmd: %u entries (%u uint64 = %u bytes), regcfg_amount=%u\n",
          n, n, n * 8, n - RKNPU_PC_DATA_EXTRA_AMOUNT);

   memcpy(rcbo.map, rc_buf, n * 8);

   struct rknpu_task *t = (struct rknpu_task *)tk.map;
   memset(t, 0, sizeof(*t));
   t->op_idx        = 0;
   t->enable_mask   = enable_mask;
   t->int_mask      = 0x300;
   t->int_clear     = 0x1ffff;
   t->regcfg_amount = n - RKNPU_PC_DATA_EXTRA_AMOUNT;
   t->regcmd_addr   = rcbo.dma_addr;

   bo_sync(fd, &act, 1);
   bo_sync(fd, &wt, 1);
   bo_sync(fd, &bias, 1);
   bo_sync(fd, &rcbo, 1);
   bo_sync(fd, &tk, 1);

   struct rknpu_submit sub;
   memset(&sub, 0, sizeof(sub));
   sub.flags          = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG;
   sub.timeout        = 6000;
   sub.task_start     = 0;
   sub.task_number    = 1;
   sub.task_obj_addr  = tk.obj_addr;
   sub.core_mask      = 0;
   sub.fence_fd       = -1;
   for (int i = 0; i < 5; i++) { sub.subcore_task[i].task_start = 0; sub.subcore_task[i].task_number = 1; }

   struct timespec t0, t1;
   clock_gettime(CLOCK_MONOTONIC, &t0);
   int ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub);
   clock_gettime(CLOCK_MONOTONIC, &t1);
   double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
   printf("SUBMIT ret=%d errno=%s  %.3f ms  hw_elapse=%lld ns\n",
          ret, ret ? strerror(errno) : "-", ms, (long long)sub.hw_elapse_time);
   if (ret) return 1;

   bo_sync(fd, &act, 2);

   /* 打印输出（group 0，通道 0..min(oc_real,16)-1），NHWC 语义 */
   uint8_t *outnpu = (uint8_t *)act.map + OUT_OFF;
   unsigned shown = MIN2(c.oc_real, FEATURE_ATOMIC_SIZE);
   printf("\n-- output group0 (raw bytes, shown as int8 / uint8) --\n");
   for (unsigned ch = 0; ch < shown; ch++) {
      printf("ch%02u: ", ch);
      for (unsigned y = 0; y < c.oh; y++) {
         printf("[");
         for (unsigned x = 0; x < c.ow; x++) {
            int8_t v = (int8_t)outnpu[npu_off(0, x, y, c.ow, c.oh) + ch];
            printf("%5d", v);
            if (x + 1 < c.ow) printf(",");
         }
         printf("]");
      }
      printf("\n");
      if (ch >= 3) { printf("  ... (rest omitted)\n"); break; }
   }

   /* ---- CPU 参考对照 ---- */
   if (g_pattern) {
      printf("\n-- CPU reference check (scale=%u shift=%u offset=%d) --\n",
             g_out_scale, g_out_shift, g_out_offset);
      long total = 0, bad_floor = 0, bad_round = 0, maxd = 0;
      long bad_rpre = 0, bad_rpost = 0;   /* ReLU 在 requant 之前 / 之后 */
      static long per_oc[256];
      memset(per_oc, 0, sizeof(per_oc));
      for (unsigned oc = 0; oc < c.oc_real; oc++)
        for (unsigned oy = 0; oy < c.oh; oy++)
          for (unsigned ox = 0; ox < c.ow; ox++) {
             int32_t acc = 0;
             if (c.deconv && c.deconv_stride) {
                /* 转置卷积（scatter 形式）：out[iy*S-P+ky][ix*S-P+kx] += in[iy][ix]*w[ky][kx] */
                int S = (int)c.deconv_stride + 1;
                for (unsigned ky = 0; ky < c.kh; ky++)
                  for (unsigned kx = 0; kx < c.kw; kx++) {
                     int ty = (int)oy + (int)c.pad_top - (int)ky;
                     int tx = (int)ox + (int)c.pad_left - (int)kx;
                     if (ty % S || tx % S) continue;
                     int iy = ty / S, ix = tx / S;
                     if (ix < 0 || iy < 0 || ix >= (int)c.iw || iy >= (int)c.ih) continue;
                     unsigned wx = g_deconv_flip ? c.kw - 1 - kx : kx;
                     unsigned wy = g_deconv_flip ? c.kh - 1 - ky : ky;
                     for (unsigned icc = 0; icc < c.ic_real; icc++)
                        acc += (int)pat_in(ix, iy, icc) * (int)pat_w(wx, wy, oc, icc);
                  }
             } else {
                for (unsigned ky = 0; ky < c.kh; ky++)
                  for (unsigned kx = 0; kx < c.kw; kx++) {
                     int ix = (int)ox * (int)c.stride_x + (int)kx - (int)c.pad_left;
                     int iy = (int)oy * (int)c.stride_y + (int)ky - (int)c.pad_top;
                     if (ix < 0 || iy < 0 || ix >= (int)c.iw || iy >= (int)c.ih) continue;
                     for (unsigned icc = 0; icc < c.ic_real; icc++)
                        acc += (int)pat_in(ix, iy, icc) * (int)pat_w(kx, ky, oc, icc);
                  }
             }
             int64_t vf = (((int64_t)acc * g_out_scale) >> g_out_shift) + g_out_offset;
             if (vf < -128) vf = -128; if (vf > 127) vf = 127;
             int64_t vr = shift_right_round64((int64_t)acc * g_out_scale, g_out_shift) + g_out_offset;
             if (vr < -128) vr = -128; if (vr > 127) vr = 127;
             /* 输出是 NC1HWC2，通道原子 16：oc>=16 落在下一个 group */
             unsigned og = oc / FEATURE_ATOMIC_SIZE;
             unsigned occ = oc % FEATURE_ATOMIC_SIZE;
             int hw = (int)(int8_t)outnpu[npu_off(og, ox, oy, c.ow, c.oh) + occ];
             total++;
             /* 候选 A：ReLU 作用在 requant 之前的累加器上 */
             int32_t accp = acc > 0 ? acc : 0;
             int64_t vpre = (((int64_t)accp * g_out_scale) >> g_out_shift) + g_out_offset;
             if (vpre < -128) vpre = -128; if (vpre > 127) vpre = 127;
             /* 候选 B：ReLU 作用在 requant 之后的 int8 上（量化域） */
             int64_t vpost = vf < 0 ? 0 : vf;
             if (hw != (int)vf) { bad_floor++; if (oc < 256) per_oc[oc]++; }
             if (hw != (int)vr) bad_round++;
             if (hw != (int)vpre)  bad_rpre++;
             if (hw != (int)vpost) bad_rpost++;
             long d = labs((long)hw - (long)vf);
             if (d > maxd) maxd = d;
             if (hw != (int)vf && bad_floor <= 5)
                printf("  mismatch oc=%u (x=%u,y=%u): acc=%d floor=%lld round=%lld hw=%d\n",
                       oc, ox, oy, acc, (long long)vf, (long long)vr, hw);
          }
      printf("  total=%ld  floor-mismatch=%ld (max abs diff %ld)  round-mismatch=%ld\n",
             total, bad_floor, maxd, bad_round);
      if (bad_floor) {
         printf("  mismatches per oc:");
         for (unsigned k = 0; k < c.oc_real && k < 256; k++)
            if (per_oc[k]) printf(" oc%u=%ld", k, per_oc[k]);
         printf("\n");
      }
      printf("  CFG: ic_real=%u ic=%u oc_real=%u oc=%u w_kernels=%u w_oc2=%u layout=%c "
             "pad_con1=%d\n",
             c.ic_real, c.ic, c.oc_real, c.oc, wskernels,
             w_oc2 > 0 ? (unsigned)w_oc2 : ALIGN_UP(c.oc_real, 2), "ACD"[g_w_layout],
             c.pad_con1_given ? c.pad_con1_override : -999);
      printf("  AUX: hi_zero=%d icreal_reg=%d ic_pad=%u\n", g_hi_zero, g_icreal_reg, c.ic);
      printf("  RELU refs: vs-max=%ld  vs-relu(pre-requant)=%ld  vs-relu(post-requant)=%ld\n",
             bad_floor, bad_rpre, bad_rpost);
      printf("  RESULT: %s\n", bad_floor == 0 ? "BIT-EXACT (floor)" :
             (bad_round == 0 ? "BIT-EXACT (round)" : "MISMATCH"));
   }

   /* 也打印十六进制前 64 字节，便于对齐 */
   printf("\n-- raw hex (first 128 bytes) --\n");
   for (int i = 0; i < 128; i++) {
      printf("%02x", outnpu[i]);
      if (i % 16 == 15) printf("\n");
   }

   return 0;
}
