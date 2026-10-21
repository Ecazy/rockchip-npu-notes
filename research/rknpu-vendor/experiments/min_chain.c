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
static int g_chain_mode = 1;
static int g_independent = 0;   /* 两个自洽 task：同输入、同权重，各写各的输出 */
static int g_chain_tail = 0;
static uint32_t g_chain_next = 0;   /* regcmd trailer 里写的下一条 regcmd 地址 */

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
   {
      /* 厂商在 trailer 里写的是真实条数：PC_REGISTER_AMOUNTS = (amount+4+1)/2 - 1 */
      unsigned amount = (unsigned)(rc - rc_buf);
      EMIT(REG_PC_BASE_ADDRESS, g_chain_next);   /* 链指针：指向下一条 regcmd */
      EMIT(REG_PC_REGISTER_AMOUNTS, (amount + 5) / 2 - 1);
   }
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



/* ================================================================== *
 * min_chain —— 两个 conv 串成一条 task 链，验证调度层
 *   task0: A(-->B)   task1: B(-->C)
 * 两次 emit_conv 写进两个 regcmd BO，task BO 里放两条 40 字节 task。
 * ================================================================== */

static int bo_create2(int fd, uint32_t size, uint32_t extra, struct bo *b);

int main(int argc, char **argv)
{
   struct conv_cfg c = {
      .iw = 8, .ih = 8, .ic_real = 16, .ic = 16,
      .ow = 8, .oh = 8, .oc_real = 16, .oc = 32,
      .kw = 3, .kh = 3, .stride_x = 1, .stride_y = 1,
      .pad_left = 1, .pad_top = 1,
      .input_zero_point = 0, .weights_zero_point = 0, .output_zero_point = 0,
      .in_scale = 1.0f, .w_scale = 1.0f, .out_scale = 1.0f,
      .truncate_bits = 0, .pad_con1_given = 1, .pad_con1_override = 0,
   };
   const char *dev = "/dev/dri/card1";
   int w_oc2 = 0, w_kernels = 0;
   uint32_t int_mask = 0x300, int_clear = 0x1ffff, timeout_ms = 500;
   uint32_t task_base_given = 0;
   uint32_t core_mask = 0;
   int subcore_zero = 0;
   uint32_t task_number_arg = 2;
   int alias_mode = 0, contig_mode = 0, subcore_first_only = 0, vendor_shape = 0;
   const uint32_t RC_STRIDE = 4096;
   int use_autofix = 0;

   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--ic") && i + 1 < argc) c.ic_real = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--oc") && i + 1 < argc) c.oc_real = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--iw") && i + 1 < argc) c.iw = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--ih") && i + 1 < argc) c.ih = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--w-oc2") && i + 1 < argc) w_oc2 = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--w-kernels") && i + 1 < argc) w_kernels = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--auto-fix")) use_autofix = 1;
      else if (!strcmp(argv[i], "--int-mask") && i + 1 < argc) int_mask = (uint32_t)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--timeout") && i + 1 < argc) timeout_ms = (uint32_t)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--task-base")) task_base_given = 1;
      else if (!strcmp(argv[i], "--independent")) g_independent = 1;
      else if (!strcmp(argv[i], "--chain-next")) g_chain_mode = 1;
      else if (!strcmp(argv[i], "--no-chain-next")) g_chain_mode = 0;
      else if (!strcmp(argv[i], "--tail-self")) g_chain_tail = 1;
      else if (!strcmp(argv[i], "--core-mask") && i + 1 < argc) core_mask = (uint32_t)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--subcore-zero")) subcore_zero = 1;
      else if (!strcmp(argv[i], "--tasks") && i + 1 < argc) task_number_arg = (uint32_t)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--alias")) alias_mode = 1;
      else if (!strcmp(argv[i], "--contig")) contig_mode = 1;
      else if (!strcmp(argv[i], "--subcore-first-only")) subcore_first_only = 1;
      else if (!strcmp(argv[i], "--vendor-shape")) vendor_shape = 1;
      else if (!strcmp(argv[i], "--dump")) g_dump = 1;
      else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
   }
   g_pattern = 1;
   if (use_autofix) {
      c.ic = (c.ic_real <= FEATURE_ATOMIC_SIZE) ? FEATURE_ATOMIC_SIZE
                                                : ALIGN_UP(c.ic_real, WEIGHT_ATOMIC_SIZE);
      if (w_oc2 == 0) w_oc2 = (c.oc_real <= WEIGHT_ATOMIC_SIZE)
                              ? (int)ALIGN_UP(c.oc_real, 2)
                              : (int)ALIGN_UP(c.oc_real, WEIGHT_ATOMIC_SIZE);
      if (w_kernels == 0) w_kernels = w_oc2;
      g_w_layout = 0;
   }
   if (w_kernels == 0) { g_w_kernels_override = 0; } else { g_w_kernels_override = w_kernels; }
   { unsigned ic1 = DIV_ROUND_UP(c.ic, WEIGHT_ATOMIC_SIZE);
     (void)ic1; }
   c.ow = c.iw; c.oh = c.ih;
   c.oc = ALIGN_UP(MAX2(c.oc_real, 32), 32);

   /* This emitter builds exactly two task descriptors and two command streams
    * (t[0] and t[1] below).  Any other count would submit descriptors that were
    * never written: that is a probe-construction bug, not a hardware limit. */
   if (task_number_arg < 1 || task_number_arg > 2) {
      fprintf(stderr, "--tasks must be 1 or 2: this emitter only builds two "
                      "descriptors and two command streams\n");
      return 2;
   }

   printf("== min_chain: 两个 %ux%ux%u conv 串联 -> %ux%ux%u (ic_real=%u oc_real=%u) ==\n",
          c.iw, c.ih, c.ic, c.ow, c.oh, c.oc, c.ic_real, c.oc_real);

   int fd = open(dev, O_RDWR);
   if (fd < 0) { fprintf(stderr, "open %s: %s\n", dev, strerror(errno)); return 1; }

   struct bo act, mid, wt, bias, rc0, rc1, tk;
   if (bo_create(fd, 1 << 16, 0, &act)) return 1;
   if (bo_create(fd, 1 << 15, 0, &mid)) return 1;
   if (bo_create(fd, 1 << 16, 0, &wt))  return 1;
   if (bo_create(fd, 1 << 12, 0, &bias)) return 1;
   if (bo_create(fd, 1 << 13, 0, &rc0)) return 1;
   if (bo_create(fd, 1 << 13, 0, &rc1)) return 1;
   if (bo_create(fd, 1 << 12, RKNPU_MEM_KERNEL_MAPPING, &tk)) return 1;

   const uint32_t OUT_OFF = 32768;
   const uint32_t W1_OFF  = 16384;

   /* 输入 A */
   uint8_t *inp = (uint8_t *)act.map;
   for (unsigned x = 0; x < c.iw; x++)
      for (unsigned y = 0; y < c.ih; y++)
         for (unsigned ch = 0; ch < c.ic; ch++)
            inp[npu_off(0, x, y, c.iw, c.ih) + ch] = (uint8_t)pat_in(x, y, ch);

   /* 两组权重：W0 在 wt 偏移 0，W1 在 W1_OFF */
   unsigned wsk = g_w_kernels_override > 0 ? (unsigned)g_w_kernels_override
                                           : ALIGN_UP(MAX2(c.oc, WEIGHT_ATOMIC_SIZE), 2);
   uint32_t wt_size = c.kw * c.kh * c.ic * wsk;
   memset(wt.map, 0, wt.size);
   unsigned wn = fill_weights((uint8_t *)wt.map, &c, 0x01, w_oc2);
   unsigned wn1 = fill_weights((uint8_t *)wt.map + W1_OFF, &c, 0x01, w_oc2);
   printf("weights: W0 %u B, W1 %u B, 描述符各 %u B (w_kernels=%u oc_w=%d)\n",
          wn, wn1, wt_size, wsk, w_oc2 > 0 ? w_oc2 : (int)ALIGN_UP(c.oc_real, 2));

   memset(bias.map, 0, bias.size);

   struct conv_cfg c0 = c, c1 = c;
   c0.in_addr = (uint32_t)act.dma_addr;
   c0.out_addr = (uint32_t)mid.dma_addr;
   c0.wt_addr = (uint32_t)wt.dma_addr;
   c0.bias_addr = (uint32_t)bias.dma_addr;
   c1 = c0;
   if (g_independent) {
      /* 每个 task 自洽：同样的输入、同样的权重，只是输出缓冲不同。
       * 这样链不依赖"task0 的输出布局 == task1 的输入布局"这个不成立的假设。 */
      c1.in_addr = c0.in_addr;
      c1.wt_addr = c0.wt_addr;
   } else {
      c1.in_addr = (uint32_t)mid.dma_addr;
      c1.wt_addr = (uint32_t)wt.dma_addr + W1_OFF;
   }
   c1.out_addr = (uint32_t)act.dma_addr + OUT_OFF;

   /* 链指针必须在发射 task0 之前就知道 task1 的地址 */
   g_chain_next = g_chain_mode ? (uint32_t)rc1.dma_addr : 0;
   uint32_t addr0 = (uint32_t)rc0.dma_addr;
   uint32_t addr1 = contig_mode ? (uint32_t)rc0.dma_addr + RC_STRIDE
                                : (uint32_t)rc1.dma_addr;
   if (contig_mode) memset(rc0.map, 0, rc0.size);

   g_chain_next = g_chain_mode ? addr1 : 0;
   rc = rc_buf;
   unsigned n0 = emit_conv(&c0);
   memcpy((uint8_t *)rc0.map, rc_buf, n0 * 8);
   g_chain_next = g_chain_tail ? addr1 : 0;
   rc = rc_buf;
   unsigned n1 = emit_conv(&c1);
   if (contig_mode) memcpy((uint8_t *)rc0.map + RC_STRIDE, rc_buf, n1 * 8);
   else             memcpy((uint8_t *)rc1.map, rc_buf, n1 * 8);
   printf("rc: task0@%08x task1@%08x contig=%d\n", addr0, addr1, contig_mode);
   printf("chain ptr: task0 -> %08x, task1 -> %08x\n",
          g_chain_next == (uint32_t)rc1.dma_addr ? (uint32_t)rc1.dma_addr : 0,
          g_chain_tail ? (uint32_t)rc1.dma_addr : 0);
   printf("regcmd: task0 %u 条 (amount=%u), task1 %u 条 (amount=%u)\n",
          n0, n0 - RKNPU_PC_DATA_EXTRA_AMOUNT, n1, n1 - RKNPU_PC_DATA_EXTRA_AMOUNT);
   printf("BO: act@%08x mid@%08x wt@%08x(+%u) rc0@%08x rc1@%08x task_obj=0x%llx\n",
          (uint32_t)act.dma_addr, (uint32_t)mid.dma_addr, (uint32_t)wt.dma_addr, W1_OFF,
          (uint32_t)rc0.dma_addr, (uint32_t)rc1.dma_addr, (unsigned long long)tk.obj_addr);

   struct rknpu_task *t = (struct rknpu_task *)tk.map;
   memset(t, 0, 2 * sizeof(*t));
   t[0].op_idx = 1; t[0].enable_mask = 0xf; t[0].int_mask = int_mask;
   t[0].int_clear = int_clear; t[0].regcfg_amount = n0 - RKNPU_PC_DATA_EXTRA_AMOUNT;
   t[0].regcmd_addr = addr0;
   t[1] = t[0];
   t[1].regcfg_amount = n1 - RKNPU_PC_DATA_EXTRA_AMOUNT;
   t[1].regcmd_addr = alias_mode ? addr0 : addr1;

   bo_sync(fd, &act, 1); bo_sync(fd, &mid, 1); bo_sync(fd, &wt, 1);
   bo_sync(fd, &bias, 1); bo_sync(fd, &rc0, 1); bo_sync(fd, &rc1, 1); bo_sync(fd, &tk, 1);
   printf("task0.addr=%08x task1.addr=%08x alias=%d\n",
          (uint32_t)t[0].regcmd_addr, (uint32_t)t[1].regcmd_addr, alias_mode);

   struct rknpu_submit sub;
   memset(&sub, 0, sizeof(sub));
   sub.flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG;
   sub.timeout = timeout_ms;
   sub.task_start = 0;
   sub.task_number = task_number_arg;
   sub.task_obj_addr = tk.obj_addr;
   sub.core_mask = core_mask;
   sub.fence_fd = -1;
   if (task_base_given) sub.task_base_addr = tk.dma_addr;
   for (int i = 0; i < 5; i++) {
      sub.subcore_task[i].task_start = 0;
      sub.subcore_task[i].task_number = subcore_zero ? 0 : task_number_arg;
   }
   if (subcore_first_only) {
      /* 只填第 0 槽，其余留 0（厂商 maxpool 的形状） */
      for (int i = 0; i < 5; i++) sub.subcore_task[i].task_number = 0;
      sub.subcore_task[0].task_number = task_number_arg;
   }
   if (vendor_shape) {
      /* 厂商 maxpool：3 个槽各 1 个 task，task_number=3 */
      for (int i = 0; i < 5; i++) sub.subcore_task[i].task_number = 0;
      for (int i = 0; i < 3; i++) sub.subcore_task[i].task_number = 1;
      sub.task_number = 3;
   }

   struct timespec t0, t1;
   clock_gettime(CLOCK_MONOTONIC, &t0);
   int ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub);
   clock_gettime(CLOCK_MONOTONIC, &t1);
   double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
   printf("SUBMIT(task_number=%u core_mask=0x%x subcore0={0,%u}) ret=%d errno=%s  %.3f ms  hw_elapse=%lld ns\n",
          task_number_arg, core_mask, sub.subcore_task[0].task_number,
          ret, ret ? strerror(errno) : "-", ms, (long long)sub.hw_elapse_time);
   if (ret) return 1;
   bo_sync(fd, &act, 2);
   bo_sync(fd, &mid, 2);

   /* ---- CPU 参考：先算中间 int8，再算第二层 ---- */
   uint8_t *midp = (uint8_t *)mid.map;
   uint8_t *outp = (uint8_t *)act.map + OUT_OFF;

   if (g_independent) {
      /* 两个 task 应该算出完全相同的结果；参考只算一次，分别比对两个输出缓冲。 */
      static uint8_t ref[1 << 16];
      memset(ref, 0, sizeof(ref));
      for (unsigned oc = 0; oc < c.oc_real; oc++)
        for (unsigned oy = 0; oy < c.oh; oy++)
          for (unsigned ox = 0; ox < c.ow; ox++) {
             int32_t acc = 0;
             for (unsigned ky = 0; ky < c.kh; ky++)
               for (unsigned kx = 0; kx < c.kw; kx++) {
                  int ix = (int)ox + (int)kx - (int)c.pad_left;
                  int iy = (int)oy + (int)ky - (int)c.pad_top;
                  if (ix < 0 || iy < 0 || ix >= (int)c.iw || iy >= (int)c.ih) continue;
                  for (unsigned icc = 0; icc < c.ic_real; icc++)
                     acc += (int)pat_in(ix, iy, icc) * (int)pat_w(kx, ky, oc, icc);
               }
             int64_t v = shift_right_round64((int64_t)acc * g_out_scale, g_out_shift) + g_out_offset;
             if (v < -128) { v = -128; }
             if (v > 127) { v = 127; }
             unsigned og = oc / FEATURE_ATOMIC_SIZE, occ = oc % FEATURE_ATOMIC_SIZE;
             ref[npu_off(og, ox, oy, c.ow, c.oh) + occ] = (uint8_t)(int8_t)v;
          }
      long total = 0, bad0 = 0, bad1 = 0, maxd0 = 0, maxd1 = 0;
      for (unsigned oc = 0; oc < c.oc_real; oc++)
        for (unsigned oy = 0; oy < c.oh; oy++)
          for (unsigned ox = 0; ox < c.ow; ox++) {
             unsigned og = oc / FEATURE_ATOMIC_SIZE, occ = oc % FEATURE_ATOMIC_SIZE;
             size_t idx = npu_off(og, ox, oy, c.ow, c.oh) + occ;
             int want = (int)(int8_t)ref[idx];
             int h0 = (int)(int8_t)midp[idx];
             int h1 = (int)(int8_t)outp[idx];
             total++;
             if (h0 != want) { bad0++; long d = labs((long)h0 - want); if (d > maxd0) maxd0 = d; }
             if (h1 != want) { bad1++; long d = labs((long)h1 - want); if (d > maxd1) maxd1 = d; }
          }
      /* 与 CPU 模型无关的判据：两个 task 的输出必须逐字节相同。
       * 这一条不依赖模型精度——它只要求"第二次执行和第一次执行一致"。 */
      const size_t out_bytes = (size_t)c.oc * c.ow * c.oh;
      long diff01 = 0, first01 = -1;
      for (size_t i = 0; i < out_bytes; i++)
         if (midp[i] != outp[i]) { diff01++; if (first01 < 0) first01 = (long)i; }
      printf("  task0(mid)      total=%ld mismatch=%ld (max abs diff %ld)\n", total, bad0, maxd0);
      printf("  task1(act+%u)   total=%ld mismatch=%ld (max abs diff %ld)\n", OUT_OFF, total, bad1, maxd1);
      printf("  PAIR            %zu bytes: %s%s\n", out_bytes,
             diff01 == 0 ? "IDENTICAL" : "DIFFER",
             diff01 == 0 ? "" : " (first at byte 0x... see count)");
      if (diff01) printf("  PAIR first-diff-byte=%ld differ=%ld\n", first01, diff01);
      const int pair_ok = (diff01 == 0) && (bad0 == bad1);
      printf("  RESULT: %s\n", pair_ok ? "PAIR BIT-EXACT (both tasks agree; model margin "
                                         "same as single task)" : "MISMATCH");
      return pair_ok ? 0 : 3;
   }

   long total = 0, bad = 0, maxd = 0;
   for (unsigned oc = 0; oc < c.oc_real; oc++)
     for (unsigned oy = 0; oy < c.oh; oy++)
       for (unsigned ox = 0; ox < c.ow; ox++) {
          int32_t acc = 0;
          for (unsigned ky = 0; ky < c.kh; ky++)
            for (unsigned kx = 0; kx < c.kw; kx++) {
               int ix = (int)ox + (int)kx - (int)c.pad_left;
               int iy = (int)oy + (int)ky - (int)c.pad_top;
               if (ix < 0 || iy < 0 || ix >= (int)c.iw || iy >= (int)c.ih) continue;
               for (unsigned icc = 0; icc < c.ic_real; icc++)
                  acc += (int)pat_in(ix, iy, icc) * (int)pat_w(kx, ky, oc, icc);
            }
          int64_t v = shift_right_round64((int64_t)acc * g_out_scale, g_out_shift) + g_out_offset;
          if (v < -128) v = -128; if (v > 127) v = 127;
          midp[npu_off(0, ox, oy, c.ow, c.oh) + (oc % FEATURE_ATOMIC_SIZE)
               + (oc / FEATURE_ATOMIC_SIZE) * c.ow * c.oh * FEATURE_ATOMIC_SIZE] = (uint8_t)(int8_t)v;
       }
   /* 第二层：输入用刚算出的中间结果 */
   for (unsigned oc = 0; oc < c.oc_real; oc++)
     for (unsigned oy = 0; oy < c.oh; oy++)
       for (unsigned ox = 0; ox < c.ow; ox++) {
          int32_t acc = 0;
          for (unsigned ky = 0; ky < c.kh; ky++)
            for (unsigned kx = 0; kx < c.kw; kx++) {
               int ix = (int)ox + (int)kx - (int)c.pad_left;
               int iy = (int)oy + (int)ky - (int)c.pad_top;
               if (ix < 0 || iy < 0 || ix >= (int)c.iw || iy >= (int)c.ih) continue;
               for (unsigned icc = 0; icc < c.ic_real; icc++) {
                  unsigned g = icc / FEATURE_ATOMIC_SIZE, cc = icc % FEATURE_ATOMIC_SIZE;
                  int8_t mv = (int8_t)midp[npu_off(g, ix, iy, c.ow, c.oh) + cc];
                  acc += (int)mv * (int)pat_w(kx, ky, oc, icc);
               }
            }
          int64_t v = shift_right_round64((int64_t)acc * g_out_scale, g_out_shift) + g_out_offset;
          if (v < -128) v = -128; if (v > 127) v = 127;
          unsigned og = oc / FEATURE_ATOMIC_SIZE, occ = oc % FEATURE_ATOMIC_SIZE;
          int hw = (int)(int8_t)outp[npu_off(og, ox, oy, c.ow, c.oh) + occ];
          total++;
          if (hw != (int)v) { bad++; long d = labs((long)hw - (long)v); if (d > maxd) maxd = d; }
       }
   printf("  total=%ld  mismatch=%ld (max abs diff %ld)\n", total, bad, maxd);
   printf("  RESULT: %s\n", bad == 0 ? "CHAIN BIT-EXACT" : "MISMATCH");
   return bad ? 3 : 0;
}
