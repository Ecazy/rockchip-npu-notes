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
 *                 [--ic N --oc N] [--in-byte B] [--w-byte B] [--out-zp N]
 *                 [--pad P --kw N --kh N]
 *                 [--bn-cfg V --bn-alu V --bn-mul V --bn-relux V]
 *                 [--dpu-data-format V]
 *                 [--bn-nrdma N --bn-table-offset N --bn-table-bias N --bn-table-step N]
 *                 [--bs-cfg V --bs-alu V --bs-mul V --brdma-cfg V]
 *                 [--brdma-bias N --brdma-bias-step N --brdma-mul N --brdma-mul-step N]
 *                 [--wgrad-dot|--wgrad-conv3x3 --wgrad-ic N --wgrad-kx N --wgrad-ky N --wgrad-batch N]
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
static int g_dump_bin = 0;

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
static int g_fill_xiw = 0;          /* 实验开关：输入 atom x-平面步长=iw（非硬件布局） */
static int g_ic1_layout = 1;        /* IC=1 输入 byte 排布 1=行主序 line=MAX2(iw,16)（实测）0=x*ph+y 2=y*iw+x 3=x*ih+y */
static int g_icreal_reg = -1;       /* 覆盖 DATAIN_CHANNEL_REAL 寄存器（真实值-1） */
static int g_din_ch = -1;           /* --din-ch N：覆盖 CNA_DATA_SIZE1.DATAIN_CHANNEL */
static int g_dma_ch = -1;           /* --dma-ch N：覆盖 CNA_FC_DATA_SIZE1.DMA_CHANNEL */
static int g_ws1 = -1;              /* --ws1 N：覆盖 CNA_WEIGHT_SIZE1 */
static int g_kernel_group = -1;    /* --kernel-group N：覆盖 CNA_CONV_CON2.KERNEL_GROUP（深 ic 分组迭代） */
static int g_weight_bank = -1;      /* --weight-bank N：覆盖 CBUF_CON0.WEIGHT_BANK，DATA_BANK=12-N */
static int g_grains = -1;           /* --grains N：覆盖 CONV_CON2.FEATURE_GRAINS */
static int g_entries_override = -1; /* 覆盖 CNA_CBUF_CON1.DATA_ENTRIES */
static int g_in_pad_rows = 0;       /* 实验：输入整体下移 N 行（前置清零行），base+N*iw*16 */
static int g_autofix = 0;           /* 1 = 采用本次实测得到的修正口径 */
static int32_t g_effective_pad_con1;
static int g_bn_cfg_override = -1;
static uint32_t g_bn_alu_override;
static uint32_t g_bn_mul_override;
static uint32_t g_bn_relux_override;
static uint32_t g_dpu_data_format;
static int g_dpu_data_format_given = 0;
static int g_bn_nrdma_override = -1;
static int32_t g_bn_table_offset_override;
static int32_t g_bn_table_bias;
static int32_t g_bn_table_step;
static int g_bn_table_fp32 = 0;
static float g_bn_table_f32_bias;
static float g_bn_table_f32_step;
static uint32_t g_bn_table_offset;
static int32_t g_bn_alu_dma[256];
static unsigned g_bn_alu_dma_count;
static int g_bs_cfg_override = -1;
static uint32_t g_bs_alu_override;
static uint32_t g_bs_mul_override;
static int g_brdma_cfg_override = -1;

static int g_subcore_mode = 0;
static int g_subcore_range = 0;      /* §7.149：每 subcore 一段**连续区间**（vendor 真实形态 [[0,36],[36,36],[72,36]]） */
static int g_subcore_single0 = 0;   /* §7.127：把 N 个 task 全放 subcore0（复测旧"不支持"结论） */
static int g_task_delta4 = 0;       /* §7.132：后续 task 只写 4 条增量块（真实负载 regcfg_amount=9 的形态） */
static int g_task_blocks = 0;       /* §7.114：每个 task 用**独立的 regcmd 块**（vendor 形态） */
static int g_subcore_hi = 0;        /* §7.170：3 核提交（use_core_num==3）时驱动读 subcore_task[core+2] ⇒ 填槽位 2..4 */
static int g_task_out_step = 0;     /* §7.170：--task-blocks 下每 task 独立输出区（patch 0x4020）+ 逐区校验 */
static const char *g_out_file_c = NULL;  /* §7.158：把卷积输出逐字节写文件（定量校验用） */
static const char *g_in_file = NULL;   /* §7.156：从文件读输入（与 --out-file 管线对接）；--in-xiw 表示 x 主序（池化输出格式） */
static int g_in_xiw = 0;
static int g_sp_prefix = 0;         /* §7.113：块首补写 S_POINTER=0xe（vendor 每个块都这么开头） */
static int g_task_count = 1;        /* §7.39：一次 submit 里放 N 个相同 task（测中断奇偶是否按 submit） */
static uint32_t g_ov_reg[16], g_ov_val[16];
static int g_ov_n = 0;
static int g_line_override = -1;    /* --line: 覆盖 CNA_DMA_CON1.LINE_STRIDE */
static int g_surf_override = -1;    /* --surf: 覆盖 CNA_DMA_CON2.SURF_STRIDE */
/* --- §7.31 输出行切分（row tiling）---
 * 单次 submit 的 CNA 输入 feature tile 字节必须 ≤ 32768（§7.30），超限硬件静默算错。
 * 修复口径：按输出行把一次 conv 拆成多个 submit；每个 tile 只算 T 行输出，其所需的
 * 输入行 [iy0, iy0+need)（need = (T-1)·sy + kh）被"staged"成紧凑块，使
 * staged 字节 = rows·row_bytes ≤ 32768。tile 输出面互不相交 ⇒ **无需跨 submit 累加**。 */
static int g_tile_oh = 0;            /* --tile-oh N：每 tile 的输出行数（0 = 关闭） */
static int g_tile_ow = 0;            /* --tile-ow N：每 tile 的输出列数（§7.35；0 = 自动/不切列） */
static int g_tile_auto = 0;          /* --tile-auto：按 32 KB 包线自动选 tile 高度 */
static int g_tile_rows_max = 0;      /* --tile-rows-max N：单次 submit 的 staged 输入行上限（0 = 仅用字节包线） */
static int g_tile_on = 0;            /* 运行期：是否真的多 submit（want_tile） */
static unsigned g_tile_max_in = 0;   /* 运行期：最大的单 tile staged 字节数（包线判别用） */

struct tile_plan {
   unsigned oy0;      /* 该 tile 的首个输出行（全局坐标） */
   unsigned T;        /* 该 tile 的输出行数 */
   unsigned g0;       /* staged 输入块对应的首个全局输入行 */
   unsigned rows;     /* staged 输入行数 */
   unsigned pad_top;  /* tile 局部 pad_top（中间 tile 为 0） */
   unsigned ox0;      /* §7.35：该 tile 的首个输出列 */
   unsigned Tc;       /* 该 tile 的输出列数 */
   unsigned x0;       /* staged 输入块对应的首个全局输入列 */
   unsigned cols;     /* staged 输入列数 */
   unsigned pad_left; /* tile 局部 pad_left */
};

/* --- §7.31 CBUF bank / FEATURE_GRAINS 自动口径（修正 §7.30 的「32 KB 硬包线」）---
 * 实测（RK3588 / driver v0.9.8）：
 *   ① 单次 submit 的 CNA 输入 feature tile 上限 = **DATA_BANK × 32768 B**（CBUF 每 bank
 *      32 KB；DATA_BANK = CNA_CBUF_CON0.DATA_BANK，WEIGHT_BANK = 12 − DATA_BANK）。
 *      旧发射器硬编码 DATA_BANK=1 ⇒ 表现为「32 KB 硬包线」（§7.30），其实是配置。
 *   ② FEATURE_GRAINS（CNA_CONV_CON2[13:4]，10 bit）必须随输入行字节放大；旧值
 *      `50+stride+1`（mesa 的 magic）在「高输入」下静默算错（fp16 iw16 ic16：53 行净、
 *      54 行错）—— 这正是 §7.30 的「次级约束」。采用 tinygrad rockchip 后端的口径：
 *      grains = max(80, ceil(2·32768 / row_bytes) + 1)（向上取偶，上限 1022）。
 * 两个旋钮都可由 `--no-auto-cfg` 关闭以复现旧行为。 */
static int g_auto_cfg = 1;           /* --no-auto-cfg 关闭自动 bank/grains */
/* --- §7.33 ic 维切分（跨 submit 累加）---
 * `--ic-groups G`：把输入通道切成 G 组（边界按 32 对齐），每组单独 emit+submit 到独立
 * partial 输出区，最后在 **host 用 fp32** 累加、fp32→fp16 写回全局输出。用于回答
 * 「真 CBUF 容量越界时 ic 切分能否替代行切分 / 是否 BIT-EXACT」；仅 fp16。 */
static int g_ic_groups = 0;
static int g_ic_offset = 0;          /* 打包某组权重时的全局 ic 通道偏移 */
/* --- §7.36 oc 维切分：`--oc-groups G` 把输出通道按 16 对齐切成 G 组；各组写**互不相交的
 * 输出平面** ⇒ **无需累加**（与行/列切分同性质），权重描述符按组打包（大幅缩小权重 bank）。 --- */
static int g_oc_groups = 0;
static int g_oc_offset = 0;          /* 打包某组权重时的全局 oc 通道偏移 */
static unsigned g_eff_input_banks, g_eff_weights_banks, g_eff_grains, g_eff_in_bytes;
static int g_auto_cfg_reported = 0;

static int g_chain_n = 2;          /* §7.44：链/多 task 个数（仿 vendor 最小形状用 3） */
static const char *g_dump_regcmd = NULL;  /* §7.52：--dump-tile K 只导出第 K 个 tile 的 regcmd */
static int g_dump_tile = 0;  /* §7.48：把本 task 的 regcmd 条目导出为 "0xOFF 0xVAL" 文本 */
static int g_chain_pc = 0;         /* §7.46：每个 task 尾部写 PC_BASE_ADDRESS/PC_REGISTER_AMOUNTS/PC_OP_EN（链表推进） */
static int g_no_trailer = 0;       /* §7.176：链段内不追加 emit_conv 的标准 trailer —— vendor 形态里
                                    * fetch 的「+4 EXTRA」就是尾写本身；trailer（PC_BASE=0/AMOUNTS=0）
                                    * 追加在 --ov 尾写之后会把链指针清零 ⇒ PC 停摆（超时根因 ②） */
static int g_chain_stub = 0;       /* §7.181：conv→stub 二分——段 1..N-1 用 attn 增量块形状（9 配置+尾写，
                                    * 真实写 our 输出区），隔离「conv 块能否交出链」vs「conv→conv 状态」 */
static uint32_t g_chain_op_en = 0x18; /* §7.181：链尾 PC_OPERATION_ENABLE（vendor attn/mul=0x18；
                                       * 我方单任务 trailer=0x1d 已证可触发；LN 流 0x18/0xd 并存） */
static int g_chain_align8 = 0;      /* §7.45：每个 task 的 regcmd 块按 8 条(64B)对齐（仿 vendor） */
static unsigned g_chain_stride = 0; /* 各 task 输出区步长（0 = ALIGN_UP(out_total,4096)） */
static int g_multi_core = 0;      /* §7.43：core_mask=0x3 + 每核 1 个不同 task（真多核派发） */
static int g_chain_id = 0;        /* §7.42：真 2-task 链（task0=主 conv → OUT_OFF，task1=同一 conv → CHAIN_OFF） */
static int g_chain_settle_ms = 0; /* §7.188：提交后 sleep N ms + 重新 bo_sync(from dev) 再对拍，
                                     * 判别 task1 数据损坏是「WDMA 排空/读早」还是「计算/CBUF 竞态」 */
static unsigned g_chain_off = 0;  /* 0 = 默认 OUT_OFF + 256 KB */
static int g_chain_armed = 0;
static unsigned g_act_size = 1u << 20;   /* --act-size N: 激活 BO 字节数（默认 1 MB）
                                         * 注：旧默认 64 KB/OUT_OFF=32768 时 in_size>32768
                                         * 的输入会与输出区重叠，污染深 ic 用例结果。 */
static unsigned g_out_off = 1u << 19;    /* --out-off N: 输出区在激活 BO 内的偏移（默认 512 KB） */
static int g_poke_atom[256];        /* --poke K:V：向输入原子 K 的通道 0 注入字节 V */
static int g_poke_val[256];
static unsigned g_poke_n;
static int32_t g_brdma_bias_base;
static int32_t g_brdma_bias_step;
static int16_t g_brdma_mul_value;
static int16_t g_brdma_mul_step;
static int32_t g_brdma_bias_table[256];
static int16_t g_brdma_mul_table[256];
static unsigned g_brdma_table_channels;

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

static inline int64_t shift_right_rne64(int64_t value, unsigned shift)
{
   if (shift == 0) return value;
   int64_t q = value >> shift;
   uint64_t rem = (uint64_t)value & ((1ULL << shift) - 1);
   uint64_t half = 1ULL << (shift - 1);
   if (rem > half || (rem == half && (q & 1))) q++;
   return q;
}

/* CPU oracle for the hardware-tested INT8 immediate BN subset. Positive and
 * negative MUL shifts come from different registers; BN MUL uses ties-to-even.
 * This is only used when --bn-cfg is supplied. */
static int32_t apply_bn_immediate(int32_t value, unsigned channel)
{
   if (g_bn_cfg_override < 0) return value;
   uint32_t cfg = (uint32_t)g_bn_cfg_override;
   if (cfg & 0x01) return value; /* BN_BYPASS */

   int32_t alu_operand = (int32_t)g_bn_alu_override;
   if ((cfg & 0x100) && g_bn_nrdma_override >= 0 &&
       channel < g_bn_alu_dma_count) /* BN_ALU_SRC=NRDMA */
      alu_operand = g_bn_alu_dma[channel];

   if (!(cfg & 0x02)) { /* !BN_ALU_BYPASS */
      switch ((cfg >> 16) & 0xf) {
      case 0: if (value < alu_operand) value = alu_operand; break;
      case 1: if (value > alu_operand) value = alu_operand; break;
      case 2: value += alu_operand; break;
      default: break;
      }
   }
   if (!(cfg & 0x10)) { /* !BN_MUL_BYPASS */
      int16_t mul = (int16_t)((g_bn_mul_override >> 16) & 0xffff);
      unsigned shift = value < 0 ? ((g_dpu_data_format >> 10) & 0x3f)
                                 : ((g_bn_mul_override >> 8) & 0x3f);
      if (!(cfg & 0x20) || value < 0) /* BN_MUL_PRELU */
         value = (int32_t)shift_right_rne64((int64_t)value * mul, shift);
   }
   if (!(cfg & 0x40)) { /* !BN_RELU_BYPASS */
      if (value < 0) value = 0;
      if ((cfg & 0x80) && value > (int32_t)g_bn_relux_override)
         value = (int32_t)g_bn_relux_override;
   }
   return value;
}

static int32_t apply_bs_brdma(int32_t value, unsigned channel)
{
   if (g_bs_cfg_override < 0) return value;
   uint32_t cfg = (uint32_t)g_bs_cfg_override;
   if (cfg & 0x01) return value; /* BS_BYPASS */

   int32_t alu_operand = (int32_t)g_bs_alu_override;
   if ((cfg & 0x100) && channel < g_brdma_table_channels) /* BS_ALU_SRC=BRDMA */
      alu_operand = g_brdma_bias_table[channel];
   if (!(cfg & 0x02)) { /* !BS_ALU_BYPASS */
      switch ((cfg >> 16) & 0xf) {
      case 0: if (value < alu_operand) value = alu_operand; break;
      case 1: if (value > alu_operand) value = alu_operand; break;
      case 2: value += alu_operand; break;
      default: break;
      }
   }
   if (!(cfg & 0x10)) { /* !BS_MUL_BYPASS */
      uint32_t mul_cfg = g_bs_mul_override;
      int16_t mul = (int16_t)((mul_cfg >> 16) & 0xffff);
      if ((mul_cfg & 1) && channel < g_brdma_table_channels)
         mul = g_brdma_mul_table[channel];
      unsigned shift = value < 0 ? ((g_dpu_data_format >> 4) & 0x3f)
                                 : ((mul_cfg >> 8) & 0x3f);
      if (!(cfg & 0x20) || value < 0) /* BS_MUL_PRELU */
         value = (int32_t)shift_right_rne64((int64_t)value * mul, shift);
   }
   if (!(cfg & 0x40)) { /* !BS_RELU_BYPASS */
      if (value < 0) value = 0;
      if ((cfg & 0x80) && value > (int32_t)g_bn_relux_override)
         value = (int32_t)g_bn_relux_override;
   }
   return value;
}

static uint32_t fui(float f)
{
   union { float f; uint32_t u; } u; u.f = f; return u.u;
}

/* ------------------------------------------------------------------ *
 * FP16（PROC_PRECISION=2）支持
 *   - 16 B 原子 = 8 个 fp16（int8 时是 16 个）
 *   - 输入/权重的“字节摆法”与 int8 完全一致，只有元素宽度与原子内元素数变化
 * ------------------------------------------------------------------ */
#define FP16_ATOM_ELEMS 8
static int g_fp16 = 0;                     /* --fp16：PROC_PRECISION=2 路径 */
static int g_f16_self_test = 0;            /* --f16-selftest */
static int g_f16_pix_elems = 0;            /* --f16-pix-elems N：输入像素原子元素数，0=用 c.ic */
static int g_f16_watom = 32;               /* --f16-watom N：权重 ic 原子元素数 */
static int g_f16_ocatom = 16;              /* --f16-ocatom N：权重 oc 原子数（实测 16） */
static int g_wk_given = 0;                 /* --w-kernels 显式给出 */
static int g_pp_val = -1;                  /* --pp-val V：覆盖 DPU/RDMA S_POINTER（默认 0xe） */
static long g_bs_tbl_wt_off = -1;          /* --bs-table-in-wt OFF：BRDMA 表放进 weight BO（0=自动） */
static int g_job_flags = -1;               /* --job-flags N：覆盖 submit flags（vendor=5，我们旧默认=7） */
static int g_warmup = -1;                  /* --warmup-submits N：热身上限；-1 = BS 激活时自动 1 次 */
static int g_f16_nest = 0;                 /* --f16-nest N：0=[oc1][i1][tap][oc2][i2] 1=[tap][oc1][i1][oc2][i2] 2=[tap][oc1][oc2][i1][i2] */
static int g_w_ic_tap = -1;                /* --w-ic-tap J：权重在 ic 维单点 one-hot */
static int g_in_ch_id = 0;                 /* --in-ch-id：输入 = ch+1（每通道可辨识） */
static int g_in_one_hot = -1;              /* --in-one-hot CH：输入仅通道 CH 为 1 */
static int g_in_ones = 0;                  /* --in-ones：输入全 1 */
static int g_w_ones = 0;                   /* --w-ones：权重全 1（隔离输入寻址） */
static int g_in_ruler = 0;                 /* --in-ruler：输入 = 元素字节偏移/2（标尺） */
static int g_in_cancel = 0;                /* --in-cancel：ch0=2048, ch1=1, ch2=-2048（判定累加器精度） */

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

/* §7.24 fp16 BS/BRDMA 语义（真机标定）：
 *   - ALU addend = fp32（immediate = BS_ALU_CFG 整个 32 位按 IEEE-754 读；
 *     BRDMA src = bias 表槽内 fp32）。algo 0/1/2 = max/min/add；先施加。
 *   - MUL factor = fp16（immediate = BS_MUL_CFG[31:16]；BRDMA src = mul 表槽内
 *     16 位按 fp16 读）。后施加，无 shift（与 int8 定点的 s2.14+shift 不同）。
 *   - RELU（bit6 bypass；bit7 上界 = BN_RELUX）。 */
static float apply_bs_fp16(float value, unsigned channel)
{
   if (g_bs_cfg_override < 0) return value;
   uint32_t cfg = (uint32_t)g_bs_cfg_override;
   if (cfg & 0x01u) return value; /* BS_BYPASS */
   if (!(cfg & 0x02u)) {          /* !BS_ALU_BYPASS */
      float operand;
      if ((cfg & 0x100u) && channel < g_brdma_table_channels) {
         memcpy(&operand, &g_brdma_bias_table[channel], sizeof(operand));
      } else {
         uint32_t u = g_bs_alu_override;
         memcpy(&operand, &u, sizeof(operand));
      }
      if (operand == operand) {
         switch ((cfg >> 16) & 0xfu) {
         case 0: if (value < operand) value = operand; break;
         case 1: if (value > operand) value = operand; break;
         case 2: value += operand; break;
         default: break;
         }
      }
   }
   if (!(cfg & 0x10u)) {          /* !BS_MUL_BYPASS */
      uint16_t mulbits;
      if ((g_bs_mul_override & 1u) && channel < g_brdma_table_channels)
         mulbits = (uint16_t)g_brdma_mul_table[channel];
      else
         mulbits = (uint16_t)((g_bs_mul_override >> 16) & 0xffffu);
      value *= f16_to_f32(mulbits);
   }
   if (!(cfg & 0x40u)) {          /* !BS_RELU_BYPASS */
      if (value < 0.0f) value = 0.0f;
      if ((cfg & 0x80u) && value > (float)g_bn_relux_override)
         value = (float)g_bn_relux_override;
   }
   return value;
}

static inline int g_is_fp16(void) { return g_fp16; }
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

/* ---- §7.31 CBUF bank / grains 助手（定义在 fp16 布局助手之后） ---- */
static unsigned cfg_row_bytes(const struct conv_cfg *c);
static unsigned cfg_in_bytes(const struct conv_cfg *c);
static unsigned cfg_wbytes(const struct conv_cfg *c);
static unsigned cfg_wbanks_need(const struct conv_cfg *c);
static unsigned cfg_data_banks_need(unsigned bytes);
static unsigned cfg_data_banks_avail(const struct conv_cfg *c);

/* 完全按文档 §3.2 的寄存器表手工组装，写成一段线性序列 */
static unsigned emit_conv(const struct conv_cfg *c){

   int ic1_path = (c->ic_real == 1 && !g_fp16);
   unsigned input_width_cfg = ic1_path
                              ? MAX2(c->iw, FEATURE_ATOMIC_SIZE) : c->iw;
   unsigned input_line_stride;
   unsigned input_surf_stride;
   if (ic1_path) {
      /* Mesa/librocketnpu special path: one-channel input packs Y bytes
       * contiguously per X and pads Y to 16; strides are in 32-bit units. */
      input_line_stride = MAX2(c->iw, FEATURE_ATOMIC_SIZE);
      input_surf_stride = input_line_stride * (c->ih - 1);
   } else {
      input_line_stride = c->iw * ATOMIC_K_SIZE / 4;
      input_surf_stride  = (int)((float)input_line_stride *
                                 (((float)c->ih / 4.0f) - 1.0f));
   }
   unsigned output_surf_stride = (c->ow * ATOMIC_K_SIZE) * c->oh / FEATURE_ATOMIC_SIZE;
     /* DATA_ENTRIES：按宽度驱动（iw=16→4, iw=8→2, iw=12→3），与 mesa/librocketnpu
      * rnpu_task.c 公式 DIV_ROUND_UP(iw*2*DIV_ROUND_UP(ic,16),8) 一致。 */
     unsigned input_data_entries = ic1_path
           ? input_width_cfg * c->ih
           : DIV_ROUND_UP(c->iw * 2 *
                          DIV_ROUND_UP(c->ic, g_fp16 ? (unsigned)FP16_ATOM_ELEMS
                                                     : (unsigned)FEATURE_ATOMIC_SIZE), 8);
   unsigned atomic_count       = c->ow * c->oh;
   unsigned weights_kernels    = g_w_kernels_override > 0
                                 ? (unsigned)g_w_kernels_override
                                 : ALIGN_UP(MAX2(c->oc, WEIGHT_ATOMIC_SIZE), 2);
   unsigned surfaces_per_row   = c->ow * c->oh * 2;

   /* CBUF bank 规划：输入 bank 数（§7.31：按 tile 字节自动取 ceil(in/32768)）+ 权重 bank 数 */
   unsigned wbytes = c->kw * c->kh * c->ic * (c->depthwise ? 1 : c->oc) *
                     (g_fp16 ? 2u : 1u);   /* fp16 权重每元素 2 B */
   unsigned weights_banks = DIV_ROUND_UP(DIV_ROUND_UP(wbytes, 128), 256) + 1;
   unsigned input_banks   = 1;
   if (c->depthwise) input_banks = 7;
   if (g_auto_cfg && !ic1_path && !c->depthwise) {
      /* §7.31：输入 feature tile 上限 = DATA_BANK × 32768 B；旧代码硬编码 1 bank
       * ⇒ 表现为 §7.30 的「32 KB 硬包线」。按实际 in_size 取足。 */
      unsigned need = cfg_data_banks_need(cfg_in_bytes(c));
      unsigned avail_b = (CBUF_BANKS > weights_banks) ? CBUF_BANKS - weights_banks : 1u;
      if (need > avail_b) need = avail_b;              /* 放不下则如实截断（RESULT 判越界） */
      if (need > input_banks) input_banks = need;
   }
   unsigned avail = CBUF_BANKS - weights_banks;
   if (input_banks > avail) input_banks = avail;
   weights_banks = CBUF_BANKS - input_banks;
   if (g_weight_bank >= 0) {
      weights_banks = (unsigned)g_weight_bank;
      input_banks   = CBUF_BANKS - weights_banks;
   }
   g_eff_input_banks = input_banks;
   g_eff_weights_banks = weights_banks;
   g_eff_in_bytes = cfg_in_bytes(c);

   /* §7.31/§7.32：FEATURE_GRAINS 必须随输入行字节/行数放大（旧值 50+stride+1 是 mesa 的
    * magic，高输入下静默算错）。auto 取「可用规则的上包络」，两者均经实测验证：
    *   ① vendor 口径（§7.32，32/32 真实 trace 精确吻合）：grains = ih + kh + pad_top
    *   ② tinygrad rockchip 后端口径：grains = max(80, ⌈2·32768/row_bytes⌉+1) 取偶
    * 上限 1022（字段 10 bit；≥1024 会被截断，如 --grains 2048 → 0 → 错）。 */
   unsigned grains = g_grains >= 0 ? (unsigned)g_grains : (50u + c->stride_y + 1u);
   if (g_grains < 0 && g_auto_cfg && !ic1_path) {
      unsigned rb = cfg_row_bytes(c);
      if (rb) {
         unsigned ag = DIV_ROUND_UP(2u * 32768u, rb) + 1u;
         ag = ALIGN_UP(ag, 2u);
         if (ag < 80u) ag = 80u;
         if (ag > grains) grains = ag;
      }
      {
         unsigned vg = c->ih + c->kh + c->pad_top;   /* vendor 口径（§7.32） */
         if (vg > grains) grains = vg;
      }
      if (grains > 1022u) grains = 1022u;
   }
   g_eff_grains = grains;

   int offset = (int)c->output_zero_point - 0x80;

   /* ---- CNA ---- */
   EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(weights_banks) |
                           CNA_CBUF_CON0_DATA_BANK(input_banks));
   EMIT(REG_CNA_DCOMP_REGNUM, 0);
   EMIT(REG_CNA_DCOMP_CTRL, 0);

   uint32_t con1 = 0;
   if (c->deconv)             con1 |= CNA_CONV_CON1_DECONV(1);
   if (ic1_path)              con1 |= CNA_CONV_CON1_NONALIGN_DMA(1) |
                                     CNA_CONV_CON1_GROUP_LINE_OFF(1) |
                                     CNA_CONV_CON1_ARGB_IN(8);
   if (c->depthwise)          con1 |= CNA_CONV_CON1_CONV_MODE(3);
   if (g_fp16)                con1 |= CNA_CONV_CON1_IN_PRECISION(2) |
                                      CNA_CONV_CON1_PROC_PRECISION(2);
   EMIT(REG_CNA_CONV_CON1, con1);
   uint32_t ppv = g_pp_val >= 0
                 ? (uint32_t)g_pp_val
                 : (DPU_S_POINTER_POINTER_PP_MODE(1) |
                    DPU_S_POINTER_EXECUTER_PP_EN(1) | DPU_S_POINTER_POINTER_PP_EN(1));
   EMIT(REG_DPU_S_POINTER, ppv);
   EMIT(REG_DPU_RDMA_RDMA_S_POINTER, ppv);
   EMIT(REG_CNA_CONV_CON1, con1);

   EMIT(REG_CNA_CONV_CON2, CNA_CONV_CON2_FEATURE_GRAINS(grains) |
        (g_kernel_group >= 0 ? CNA_CONV_CON2_KERNEL_GROUP((uint32_t)g_kernel_group) : 0));
   if (!g_auto_cfg_reported && (g_dump || g_auto_cfg)) {
      g_auto_cfg_reported = 1;
      printf("CBUF-CFG: in_size=%u B  row_bytes=%u B  DATA_BANK=%u  WEIGHT_BANK=%u  "
             "GRAINS=%u (auto_cfg=%d, grains_%s)\n",
             g_eff_in_bytes, cfg_row_bytes(c), input_banks, weights_banks, grains,
             g_auto_cfg, g_grains >= 0 ? "override" : "auto");
   }
   uint32_t con3 = CNA_CONV_CON3_CONV_X_STRIDE(c->stride_x) |
                   CNA_CONV_CON3_CONV_Y_STRIDE(c->stride_y);
   if (c->deconv && c->deconv_stride)
      con3 |= CNA_CONV_CON3_DECONV_X_STRIDE(c->deconv_stride) |
              CNA_CONV_CON3_DECONV_Y_STRIDE(c->deconv_stride);
   EMIT(REG_CNA_CONV_CON3, con3);

   EMIT(REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH(input_width_cfg) |
                            CNA_DATA_SIZE0_DATAIN_HEIGHT(c->ih));
   EMIT(REG_CNA_DATA_SIZE1,
        CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(g_icreal_reg >= 0 ? (unsigned)g_icreal_reg
                                                             : c->ic_real - 1) |
        CNA_DATA_SIZE1_DATAIN_CHANNEL(g_din_ch > 0 ? (unsigned)g_din_ch : c->ic));
   EMIT(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(c->ow));
   EMIT(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(atomic_count));

   EMIT(REG_CNA_WEIGHT_SIZE0, c->kw * c->kh * c->ic * weights_kernels *
                              (g_fp16 ? 2u : 1u));
   /* §7.36：WEIGHT_SIZE1 与 WEIGHT_SIZE0 一样是**字节数**，fp16 必须 ×2
    * （旧代码漏乘 ⇒ 小输入 + 多 ic/oc 块时权重取指不足，后半个 oc 块静默取错值；
    * 判别：`--ws1 kw*kh*ic*2` 即修复，见 results/fp16_ws1_20260930/）。 */
   EMIT(REG_CNA_WEIGHT_SIZE1, g_ws1 > 0 ? (unsigned)g_ws1
                                        : c->kw * c->kh * c->ic * (g_fp16 ? 2u : 1u));
   EMIT(REG_CNA_WEIGHT_SIZE2, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(c->kw) |
                              CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(c->kh) |
                              CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(weights_kernels));

   EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(weights_banks) |
                           CNA_CBUF_CON0_DATA_BANK(input_banks));
   EMIT(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(
           g_entries_override >= 0 ? (unsigned)g_entries_override : input_data_entries));

   if (ic1_path) {
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
    EMIT(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(
            g_line_override >= 0 ? (unsigned)g_line_override : input_line_stride));
    EMIT(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(g_surf_override >= 0
                              ? (unsigned)g_surf_override : input_surf_stride));
   EMIT(REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH(c->iw) |
                               CNA_FC_DATA_SIZE0_DMA_HEIGHT(c->ih));
   EMIT(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(
           g_dma_ch > 0 ? (unsigned)g_dma_ch : c->ic));
   EMIT(REG_CNA_DCOMP_CTRL, 0);
   EMIT(REG_CNA_DCOMP_REGNUM, 0);
   EMIT(REG_CNA_DCOMP_ADDR0, c->wt_addr);
   for (int i = 0; i < 16; i++) EMIT(REG_CNA_DCOMP_AMOUNT0 + i * 4, 0);
   EMIT(REG_CNA_CVT_CON5, ic1_path ? 65535 : 0);

   int32_t pad_con1;
   if (c->pad_con1_given) pad_con1 = c->pad_con1_override;
   else if (ic1_path)
      /* The IC=1 input CVT centers raw U8 to the signed activation domain.
       * The measured zero-point-0 case needs PAD_CON1=0 (post-CVT domain). */
      pad_con1 = (int32_t)c->input_zero_point;
   else {
      if (c->kw >= 3 && c->input_zero_point == 0x0) pad_con1 = 0xffff8080;
      else pad_con1 = (int32_t)c->input_zero_point - 0x80;
      if (c->depthwise && c->input_zero_point == 0x8b) pad_con1 = 0x0b0b;
   }
   g_effective_pad_con1 = pad_con1;
   EMIT(REG_CNA_PAD_CON1, (uint32_t)pad_con1);

   /* ---- CORE ---- */
   uint32_t misc_cfg = g_fp16 ? CORE_MISC_CFG_PROC_PRECISION(2)
                              : CORE_MISC_CFG_QD_EN(1);
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
   if (g_fp16 && !g_dpu_data_format_given)
      g_dpu_data_format = DPU_DATA_FORMAT_IN_PRECISION(2) |
                          DPU_DATA_FORMAT_OUT_PRECISION(2) |
                          DPU_DATA_FORMAT_PROC_PRECISION(2);
   EMIT(REG_DPU_DATA_FORMAT, g_dpu_data_format);
   EMIT(REG_DPU_OFFSET_PEND, 0);
   EMIT(REG_DPU_DST_BASE_ADDR, c->out_addr);
   EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(output_surf_stride));
   EMIT(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(c->ow - 1));
   EMIT(REG_DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_HEIGHT_HEIGHT(c->oh - 1));
   EMIT(REG_DPU_DATA_CUBE_NOTCH_ADDR, 0);
   EMIT(REG_DPU_DATA_CUBE_CHANNEL,
        DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(c->oc_real - 1) |
        DPU_DATA_CUBE_CHANNEL_CHANNEL(c->oc - 1));
   EMIT(REG_DPU_BS_CFG, g_bs_cfg_override >= 0
                        ? (uint32_t)g_bs_cfg_override
                        : (g_fp16 ? (DPU_BS_CFG_BS_BYPASS(1) |
                                     DPU_BS_CFG_BS_RELU_BYPASS(1) |
                                     DPU_BS_CFG_BS_MUL_BYPASS(1) |
                                     DPU_BS_CFG_BS_ALU_BYPASS(1))
                                  : (DPU_BS_CFG_BS_ALU_ALGO(2) | DPU_BS_CFG_BS_ALU_SRC(1) |
                                     DPU_BS_CFG_BS_RELU_BYPASS(1) | DPU_BS_CFG_BS_MUL_BYPASS(1))));
   EMIT(REG_DPU_BS_ALU_CFG, g_bs_alu_override);
   EMIT(REG_DPU_BS_MUL_CFG, g_bs_mul_override);
   EMIT(REG_DPU_BS_RELUX_CMP_VALUE, 0);
   EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(1) | DPU_BS_OW_CFG_SIZE_E_1(1) |
                           DPU_BS_OW_CFG_SIZE_E_0(1));
   EMIT(REG_DPU_BS_OW_OP, DPU_BS_OW_OP_OW_OP(0));
   EMIT(REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA(c->oc - 1));
   EMIT(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_HEIGHT_WDMA(c->oh - 1) |
                             DPU_WDMA_SIZE_1_WIDTH_WDMA(c->ow - 1));
   EMIT(REG_DPU_BN_CFG, g_bn_cfg_override >= 0
                        ? (uint32_t)g_bn_cfg_override
                        : DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) |
                          DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));
   EMIT(REG_DPU_BN_ALU_CFG, g_bn_alu_override);
   EMIT(REG_DPU_BN_MUL_CFG, g_bn_mul_override);
   EMIT(REG_DPU_BN_RELUX_CMP_VALUE, g_bn_relux_override);
   EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) |
                        DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) |
                        DPU_EW_CFG_EW_BYPASS(1));
   EMIT(REG_DPU_EW_CVT_OFFSET_VALUE, 0);
   EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
   EMIT(REG_DPU_EW_RELUX_CMP_VALUE, 0);
   EMIT(REG_DPU_OUT_CVT_OFFSET, g_fp16 ? 0u : (uint32_t)offset);

   /* requant: conv_scale -> (scale, shift)，与 rnpu_coefs.c 同一套定点化 */
   float conv_scale = (c->in_scale * c->w_scale) / c->out_scale;
   uint32_t scale_bits = fui(conv_scale);
   unsigned shift = 127 + 31 - 32 - (scale_bits >> 23) + 16;
   if (c->truncate_bits > 0) shift--;
   unsigned scale = ((scale_bits >> 9) & 0x7fff) + 1;
   if (scale < (1u << 14)) scale |= 1u << 14;
   if (g_fp16) {
      /* fp16：DPU 输出直通 fp32->fp16，无重量化 */
      g_out_scale = 1; g_out_shift = 0; g_out_offset = 0;
      EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_FP32TOFP16_EN(1) |
                                  DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(1));
      EMIT(REG_DPU_OUT_CVT_SHIFT, 0);
   } else {
      g_out_scale = scale; g_out_shift = shift - 1; g_out_offset = offset;
      EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(scale));
      EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(shift - 1));
   }

   for (int i = 0; i < 8; i++) EMIT(REG_DPU_EW_OP_VALUE_0 + i * 4, 0);
   EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(surfaces_per_row));
   emit_raw(DPU | 0x1, 0x40c4, 0);
   for (int i = 0; i < 12; i++) EMIT(REG_DPU_LUT_ACCESS_CFG + i * 4, 0);

   /* ---- DPU_RDMA ---- */
   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH(c->ow - 1));
   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT(c->oh - 1));
   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL(c->oc - 1));
   EMIT(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0);
   EMIT(REG_DPU_RDMA_RDMA_BRDMA_CFG,
        g_brdma_cfg_override >= 0 ? (uint32_t)g_brdma_cfg_override
                                  : DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE(1));
   EMIT(REG_DPU_RDMA_RDMA_BS_BASE_ADDR, c->bias_addr);
   EMIT(REG_DPU_RDMA_RDMA_NRDMA_CFG,
        g_bn_nrdma_override >= 0 ? (uint32_t)g_bn_nrdma_override : 0);
   EMIT(REG_DPU_RDMA_RDMA_BN_BASE_ADDR,
        g_bn_nrdma_override >= 0
          ? (uint32_t)(c->wt_addr + g_bn_table_offset) : 0);
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

   /* ---- 实验寄存器覆盖（--ov）：必须落在 regcfg_amount 窗口内（trailer 之前），
    * 否则驱动取指 regcfg_amount+4 之外的条目不会被执行（第十四~十七轮 --ov 结论因此全部无效）----
    * §7.173：链尾写的 target 必须与 vendor 一致 —— PC_BASE/AMOUNTS → 0x101（PC bank），
    * VERSION → 0x41、OP_EN → 0x81（**下一 PC 实例 bank**，写当前 bank 会让链挂死：
    * 旧 §7.46 --chain-pc 超时的真正根因 = EMIT 把 VERSION/OP_EN 写到 0x101 ✗）。 */
   for (int i = 0; i < g_ov_n; i++) {
      if (g_ov_reg[i] == 0xFFFF)
         emit_raw(0, 0, 0);                 /* §7.186：终端 pad（vendor conv 终端首条，tgt0/reg0/val0） */
      else if (g_ov_reg[i] == REG_PC_VERSION)
         emit_raw(0x41, g_ov_reg[i], g_ov_val[i]);
      else if (g_ov_reg[i] == REG_PC_OPERATION_ENABLE)
         emit_raw(0x81, g_ov_reg[i], g_ov_val[i]);
      else
         EMIT(g_ov_reg[i], g_ov_val[i]);
   }

   /* ---- 收尾：链指针占位 + 触发标记（§7.176：链段自带尾写时跳过，防清链） ---- */
   if (!g_no_trailer) {
      EMIT(REG_PC_BASE_ADDRESS, 0);
      EMIT(REG_PC_REGISTER_AMOUNTS, 0);
      *rc++ = 0x0041000000000000ULL;
      emit_raw(0x81, REG_PC_OPERATION_ENABLE,
               PC_OPERATION_ENABLE_RESERVED_0(14) | PC_OPERATION_ENABLE_OP_EN(1));
   }

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
 * 5. 数据布局：NC1HWC2（group-major, y-major 行主序, c16）
 *
 * 硬件输入与输出都是行主序：atom(x,y) = g*w*h*FA + y*w*FA + x*FA。
 * 方阵 iw==ih（输入）/ ow==oh（输出）时行主序与列主序重合，故长期
 * 掩盖了该布局差异；矩形 iw≠ih 暴露。见台账 §7.11。
 * ------------------------------------------------------------------ */
static uint32_t npu_off(unsigned g, unsigned x, unsigned y, unsigned w, unsigned h)
{
   return g * w * h * FEATURE_ATOMIC_SIZE + y * w * FEATURE_ATOMIC_SIZE +
          x * FEATURE_ATOMIC_SIZE;
}
static uint32_t out_off(unsigned g, unsigned x, unsigned y, unsigned w, unsigned h)
{
   return npu_off(g, x, y, w, h);
}

/* fp16 输入面：单个像素装 c->ic 个 fp16（c->ic*2 字节），行主序，单组 */
static inline unsigned fp16_atom_elems(const struct conv_cfg *c)
{
   (void)c;
   return g_f16_pix_elems > 0 ? (unsigned)g_f16_pix_elems
                              : (unsigned)FP16_ATOM_ELEMS;
}
static inline unsigned fp16_pixel_bytes(const struct conv_cfg *c)
{
   return fp16_atom_elems(c) * 2;          /* 每像素 = 一个 16 B 通道平面 */
}
static inline unsigned fp16_planes(const struct conv_cfg *c)
{
   return DIV_ROUND_UP(c->ic, fp16_atom_elems(c));
}
static inline unsigned in_off(const struct conv_cfg *c, unsigned g, unsigned x, unsigned y)
{
   unsigned pb = fp16_pixel_bytes(c);
   return g * c->iw * c->ih * pb + y * c->iw * pb + x * pb;
}

/* ---- §7.31 CBUF bank / grains 口径（定义；原型见文件头部） ---- */
static unsigned cfg_row_bytes(const struct conv_cfg *c)
{
   unsigned pb = g_fp16 ? fp16_planes(c) * fp16_pixel_bytes(c)
                        : DIV_ROUND_UP(c->ic, FEATURE_ATOMIC_SIZE) * FEATURE_ATOMIC_SIZE;
   return c->iw * pb;
}
static unsigned cfg_in_bytes(const struct conv_cfg *c)
{
   return cfg_row_bytes(c) * c->ih;
}
static unsigned cfg_wbytes(const struct conv_cfg *c)
{
   if (!g_fp16) return c->kw * c->kh * c->ic * (c->depthwise ? 1 : c->oc);
   {
      unsigned wa = (unsigned)(g_f16_watom > 0 ? g_f16_watom : 32);
      unsigned icg = DIV_ROUND_UP(c->ic, wa) * MIN2(c->ic, wa);
      unsigned wk = g_w_kernels_override > 0
                    ? (unsigned)g_w_kernels_override
                    : ALIGN_UP(MAX2(c->oc, WEIGHT_ATOMIC_SIZE), 16);
      return c->kw * c->kh * icg * wk * 2u;
   }
}
static unsigned cfg_wbanks_need(const struct conv_cfg *c)
{
   return DIV_ROUND_UP(DIV_ROUND_UP(cfg_wbytes(c), 128), 256) + 1;
}
static unsigned cfg_data_banks_need(unsigned bytes)
{
   return DIV_ROUND_UP(bytes, 32768u);
}
static unsigned cfg_data_banks_avail(const struct conv_cfg *c)
{
   unsigned wneed = cfg_wbanks_need(c);
   unsigned a = (CBUF_BANKS > wneed) ? (CBUF_BANKS - wneed) : 1u;
   return a > 11u ? 11u : a;
}

static int g_pattern = 0;
static int g_deconv_flip = 0;   /* --deconv-flip：历史开关，§7.27 后无语义影响（保留兼容） */
static int g_wgrad_dot_mode = 0;
static int g_wgrad_conv3x3_mode = 0;
static unsigned g_wgrad_batch_count = 1;
static unsigned g_wgrad_ic_idx, g_wgrad_kx, g_wgrad_ky;

static int g_pat_asym = 0;
static int g_pat2 = 0;                          /* --pat2: x/y 双依赖 pattern */
static int g_w_tap_kx = -1, g_w_tap_ky = -1;   /* --w-tap KX:KY 单 tap one-hot */
static int8_t pat_in(unsigned x, unsigned y, unsigned ch)
{
   if (g_pat2)
      return (int8_t)(((int)(x * 3 + y * 5 + ch * 2) % 7) - 3);
   return (int8_t)(((int)(x * 7 + y * 5 + ch * 3) % 5) - 2);
}
static int8_t pat_w(unsigned x, unsigned y, unsigned oc, unsigned icc)
{
   if (g_w_tap_kx >= 0)
      return (int8_t)((x == (unsigned)g_w_tap_kx && y == (unsigned)g_w_tap_ky) ? 1 : 0);
   if (g_pat_asym)
      return (int8_t)(((int)(2 * x + y + oc + icc) % 3) - 1);
   return (int8_t)(((int)(x + y + oc + icc) % 3) - 1);
}
/* §7.192：外部输入（--in-file）时，CPU 参考改用 BO 里的真实输入而非内部 pattern。
 * BO 为硬件 y 主序布局：npu_off(g,x,y,w,h) + c16。 */
static const uint8_t *g_ref_in_base = NULL;
static int8_t ref_in_val(unsigned ix, unsigned iy, unsigned icc, unsigned iw, unsigned ih)
{
   if (g_ref_in_base)
      return (int8_t)g_ref_in_base[npu_off(icc / FEATURE_ATOMIC_SIZE, ix, iy, iw, ih)
                                   + (icc % FEATURE_ATOMIC_SIZE)];
   return pat_in(ix, iy, icc);
}

/* fp16 输入 pattern：取 1/3 的倍数（fp16 无法精确表示，CPU 参考直接读回 BO
 * 里的 fp16，从而同时验证数据通道与舍入）。 */
static float fp16_in_value(unsigned x, unsigned y, unsigned ch)
{
   if (g_in_cancel)
      return ch == 0 ? 2048.0f : (ch == 1 ? 1.0f : (ch == 2 ? -2048.0f : 0.0f));
   if (g_in_ones) return 1.0f;
   if (g_in_one_hot >= 0) return ((int)ch == g_in_one_hot) ? 1.0f : 0.0f;
   if (g_in_ch_id) return (float)(ch + 1);
   return (float)((int)((x * 3 + y * 5 + ch * 2) % 7) - 3) / 3.0f;
}
/* fp16 权重 pattern：1.1 × 整数（同样不可精确表示） */
static float fp16_w_value(unsigned kx, unsigned ky, unsigned oc, unsigned icc)
{
   if (g_w_ones) return 1.0f;
   if (g_w_ic_tap >= 0)
      return ((int)icc == g_w_ic_tap) ? 1.0f : 0.0f;
   return (float)pat_w(kx, ky, oc, icc) * 1.1f;
}

static int8_t wgrad_x(unsigned p)
{
   if (g_wgrad_conv3x3_mode) {
      unsigned sample = p / 36, pos = p % 36;
      unsigned ox = pos % 6, oy = pos / 6;
      unsigned ix = ox + g_wgrad_kx, iy = oy + g_wgrad_ky;
      return (int8_t)(((ix * 3 + iy * 5 + g_wgrad_ic_idx * 2 + sample) % 5) - 2);
   }
   return (int8_t)(((p * 5 + g_wgrad_ic_idx * 2 + g_wgrad_kx * 3 +
                     g_wgrad_ky * 7 + 1) % 3) - 1);
}

static int8_t wgrad_dy(unsigned p, unsigned oc)
{
   if (g_wgrad_conv3x3_mode) {
      unsigned sample = p / 36, pos = p % 36;
      return (int8_t)(((pos * 7 + oc * 2 + sample * 2 + 2) % 3) - 1);
   }
   return (int8_t)(((p * 7 + oc * 2 + 2) % 3) - 1);
}

static void fill_input(uint8_t *buf, const struct conv_cfg *c,
                       uint8_t byte, uint8_t pad_byte)
{
   if (c->ic_real == 1 && !g_fp16) {
      /* rnpu_convert_input() special case: 1 byte/point. 实测为行主序。
       * 0=x*ph+y 1=y*pw+x（line=pw） 2=y*iw+x（line=iw） 3=x*ih+y */
      unsigned ph = MAX2(c->ih, FEATURE_ATOMIC_SIZE);
      unsigned pw = MAX2(c->iw, FEATURE_ATOMIC_SIZE);
      for (unsigned yy = 0; yy < ph; yy++)
         for (unsigned xx = 0; xx < pw; xx++) {
            unsigned idx;
            switch (g_ic1_layout) {
               case 1:  idx = yy * pw + xx;    break;
               case 2:  idx = yy * c->iw + xx; break;
               case 3:  idx = xx * c->ih + yy; break;
               default: idx = xx * ph + yy;    break;
            }
            int real = (xx < c->iw && yy < c->ih);
            if (real && g_pattern) {
               int8_t logical = pat_in(xx, yy, 0);
               buf[idx] = (uint8_t)(logical + 128);
            } else if (real)
               buf[idx] = byte;
            else
               buf[idx] = pad_byte;
         }
      return;
   }

   unsigned atom_elems = g_fp16
         ? (g_f16_pix_elems > 0 ? (unsigned)g_f16_pix_elems : c->ic)
         : (unsigned)FEATURE_ATOMIC_SIZE;
   unsigned groups = DIV_ROUND_UP(c->ic, atom_elems);
   if (g_in_file && !g_fp16) {   /* §7.156：整块从文件读入（**不再逐点填 pattern**），布局与填充公式一致 */
      FILE *f = fopen(g_in_file, "rb");
      if (!f) { printf("IN-FILE: 打不开 %s\n", g_in_file); }
      else {
         size_t want = (size_t)groups * c->iw * c->ih * FEATURE_ATOMIC_SIZE;
         size_t got = fread(buf, 1, want, f);
         fclose(f);
         printf("IN-FILE: %s 读入 %zu/%zu B（%s 主序布局）\n", g_in_file, got, want,
                g_in_xiw ? "x（池化输出）" : "y（硬件行）");
         if (g_in_xiw) {   /* x 主序文件 → 硬件 y 主序缓冲：逐原子搬运 */
            static uint8_t tmp[1 << 22];
            if (got <= sizeof tmp) {
               memcpy(tmp, buf, got);
               memset(buf, 0, want);
               for (unsigned g = 0; g < groups; g++)
                  for (unsigned x = 0; x < c->iw; x++)
                     for (unsigned y = 0; y < c->ih; y++) {
                        size_t src = (size_t)g * c->iw * c->ih * FEATURE_ATOMIC_SIZE
                                     + (size_t)x * c->ih * FEATURE_ATOMIC_SIZE + (size_t)y * FEATURE_ATOMIC_SIZE;
                        size_t dst = (size_t)npu_off(g, x, y, c->iw, c->ih);
                        if (src + FEATURE_ATOMIC_SIZE <= got) memcpy(buf + dst, tmp + src, FEATURE_ATOMIC_SIZE);
                     }
            }
         }
         return;
      }
   }
   for (unsigned g = 0; g < groups; g++)
      for (unsigned x = 0; x < c->iw; x++)
         for (unsigned y = 0; y < c->ih; y++) {
            /* 硬件为行主序（row-major，见 §7.11）：atom(x,y) = base + y*iw*FA + x*FA。
             * --fill-xiw 为遗留实验开关：x-主序、步长=iw 原子。 */
            uint8_t *p = buf + (g_fill_xiw
                  ? (uint32_t)(g * (uint32_t)c->iw * c->ih * FEATURE_ATOMIC_SIZE
                               + x * (uint32_t)c->iw * FEATURE_ATOMIC_SIZE
                               + y * FEATURE_ATOMIC_SIZE)
                  : (g_fp16 ? in_off(c, g, x, y)
                            : npu_off(g, x, y, c->iw, c->ih)));
            for (unsigned k = 0; k < atom_elems; k++) {
               unsigned ch = g * atom_elems + k;
               if (g_fp16) {
                  /* fp16：每个原子 16 B = 8 个 fp16；pattern 用非二进制可精确
                   * 表示的有理数（1/3），CPU 参考改为直接读回 BO 里的 fp16。 */
                  uint16_t h;
                  if (g_in_ruler)
                     h = f32_to_f16((float)((unsigned)((p - buf) + k * 2) / 2u));
                  else if (ch >= c->ic_real) h = 0;
                  else if (g_wgrad_dot_mode)
                     /* wgrad：K 维（输入通道 p）= 归约点；conv3x3 仅 (0,0) 有效 */
                     h = f32_to_f16((float)((g_wgrad_conv3x3_mode &&
                                             (x != 0 || y != 0)) ? 0
                                            : wgrad_x(ch)));
                  else if (g_pattern) h = f32_to_f16(fp16_in_value(x, y, ch));
                  else h = f32_to_f16((float)byte);
                  p[k * 2 + 0] = (uint8_t)(h & 0xff);
                  p[k * 2 + 1] = (uint8_t)(h >> 8);
                  continue;
               }
               if (g_pattern && g_hi_zero && ch >= c->ic_real)
                  p[k] = 0;
               else if (g_pattern && g_wgrad_dot_mode)
                  p[k] = (uint8_t)((g_wgrad_conv3x3_mode && (x != 0 || y != 0))
                                      ? 0 : wgrad_x(ch));
               else if (g_pattern) {
                  int8_t logical = pat_in(x, y, ch);
                  /* Hypothesis for ic_real=1 CNA input CVT (scale=1,
                   * offset=-128, truncate=14 in Q14): output = byte-128.
                   * Encode logical signed inputs as value+128 to test it.
                   * The CPU oracle remains in signed value space. */
                  p[k] = c->ic_real == 1 ? (uint8_t)(logical + 128)
                                         : (uint8_t)logical;
               }
               else
                  p[k] = (ch < c->ic_real) ? byte : pad_byte;
            }
         }
}

/* fp16 权重元素序（layout A：oc1 → ic1 → ky → kx → oc2 → ic2，ic 原子 8 元素） */
static unsigned w_elem_index_f16(const struct conv_cfg *c, unsigned oc, unsigned icc,
                                 unsigned kx, unsigned ky, unsigned oc_w)
{
   unsigned wa   = (unsigned)(g_f16_watom > 0 ? g_f16_watom : 32);
   unsigned ic1  = DIV_ROUND_UP(c->ic, wa);
   unsigned ic2  = MIN2(c->ic, wa);
   unsigned oa   = (unsigned)(g_f16_ocatom > 0 ? g_f16_ocatom : 32);
   unsigned oc1n = DIV_ROUND_UP(oc_w, oa);
   unsigned oc2n = MIN2(oc_w, oa);
   unsigned oc1 = oc / oa, oc2 = oc % oa;
   unsigned i1 = icc / wa, i2 = icc % wa;
   (void)oc1n;
   if (g_f16_nest == 1)   /* [tap][oc1][i1][oc2][i2] */
      return (((((ky * c->kw + kx) * oc1n + oc1) * ic1 + i1) * oc2n + oc2) * ic2 + i2);
   if (g_f16_nest == 2)   /* [tap][oc1][oc2][i1][i2] */
      return (((((ky * c->kw + kx) * oc1n + oc1) * oc2n + oc2) * ic1 + i1) * ic2 + i2);
   return ((((oc1 * ic1 + i1) * c->kh + ky) * c->kw + kx) * oc2n + oc2) * ic2 + i2;
}

/* fp16 权重打包（layout A）：每 16 B 行 = 8 个 fp16，ic 越界槽填 0 */
static unsigned fill_weights_f16(uint8_t *dst, const struct conv_cfg *c,
                                 uint8_t w_byte, unsigned oc_w)
{
   unsigned n = 0;
   for (unsigned kx = 0; kx < c->kw; kx++)
    for (unsigned ky = 0; ky < c->kh; ky++)
     for (unsigned oc = 0; oc < oc_w; oc++)
      for (unsigned icc = 0; icc < c->ic; icc++) {
         uint16_t h;
         if (oc >= c->oc_real || icc >= c->ic_real) h = 0;
         else if (g_wgrad_dot_mode) h = f32_to_f16((float)wgrad_dy(icc + g_ic_offset, oc + g_oc_offset));
         else if (g_pattern) h = f32_to_f16(fp16_w_value(kx, ky, oc + g_oc_offset, icc + g_ic_offset));
         else h = f32_to_f16((float)w_byte);
         unsigned off = w_elem_index_f16(c, oc, icc, kx, ky, oc_w) * 2;
         dst[off] = (uint8_t)(h & 0xff);
         dst[off + 1] = (uint8_t)(h >> 8);
         n++;
      }
   return n;
}

/* fp16 位型的单调（monotonic）整数键：近零处正负跨越不再放大 ULP 距离 */
static inline long f16_ord(uint16_t h)
{
   return (h & 0x8000u) ? -(long)(h & 0x7fffu) : (long)(h & 0x7fffu);
}

static inline float rd_f16(const uint8_t *p)
{
   return f16_to_f32((uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8)));
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
           default: /* A: [oc1][i1][ky][kx][oc2][i2]，librocketnpu 原生写法
                     * tap 步进为 ky-major（kx 变化最快）：tap = ky*kw + kx。
                     * 依据 §7.15：vendor .rknn 权重 blob 中同一 oc 的 16B 行
                     * 依次是 (kh0,kw0..kw2),(kh1,kw0..)…，即 kx 内层。
                     * 旧写法 tap = kx*kh + ky 在方形核上与硬件互为转置，被对称
                     * pattern 抵消（--asym-w + 旧 y 盲区 pattern 亦抵消）；kw≠kh
                     * 且均 ≥2 时才暴露成 MISMATCH。 */
              idx = ((((oc1 * ic1 + i1) * c->kh + y) * c->kw + x) * oc2n + oc2) * ic2 + i2;
              break;
           }

           if (oc >= c->oc_real) dst[idx] = 0x00;
           else if (icc >= c->ic_real) dst[idx] = 0x00;  /* 权重零点 */
           else if (g_wgrad_dot_mode) dst[idx] = (uint8_t)wgrad_dy(icc + g_ic_offset, oc + g_oc_offset);
           else dst[idx] = g_pattern ? (uint8_t)pat_w(x, y, oc + g_oc_offset, icc + g_ic_offset) : w_byte;
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

   /* fp16 转换自检：打印一组 float 的 fp16 位型，供与 numpy.float16 对拍 */
   for (int i = 1; i < argc; i++)
      if (!strcmp(argv[i], "--f16-selftest")) {
         const float vals[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 0.1f, 1.0f/3.0f,
                               65504.0f, 65520.0f, 1e-8f, 5.96e-8f, -2.5f,
                               0.333251953125f, 1.5f, 100.0f, 0.00006103515625f};
         for (unsigned k = 0; k < sizeof(vals) / sizeof(vals[0]); k++) {
            uint16_t h = f32_to_f16(vals[k]);
            float r = f16_to_f32(h);
            printf("F16TEST %08x %04x %08x\n", fui(vals[k]), h, fui(r));
         }
         /* 全 16 bit 位型的 round-trip 覆盖（打印逐位型的 float 展开） */
         for (uint32_t h = 0; h < 65536; h++) {
            float r = f16_to_f32((uint16_t)h);
            printf("F16RT %04x %08x\n", h, fui(r));
         }
         return 0;
      }
   int iw_given = 0, ow_given = 0, oh_given = 0;
   int mode_deconv = 0, w_uniform = 0, w_oc2 = 0;

   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--dump")) g_dump = 1;
      else if (!strcmp(argv[i], "--dump-bin")) { g_dump_bin = 1; g_dump = 1; }
      else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
         i++; if (!strcmp(argv[i], "deconv")) mode_deconv = 1;
      }
      else if (!strcmp(argv[i], "--iw") && i + 1 < argc) { c.iw = atoi(argv[++i]); iw_given = 1; }
      else if (!strcmp(argv[i], "--ih") && i + 1 < argc) c.ih = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--ow") && i + 1 < argc) { c.ow = atoi(argv[++i]); ow_given = 1; }
      else if (!strcmp(argv[i], "--oh") && i + 1 < argc) { c.oh = atoi(argv[++i]); oh_given = 1; }
      else if (!strcmp(argv[i], "--kw") && i + 1 < argc) c.kw = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--fp16")) { g_fp16 = 1; g_pattern = 1; }
      else if (!strcmp(argv[i], "--f16-selftest")) g_f16_self_test = 1;
      else if (!strcmp(argv[i], "--f16-pix-elems") && i + 1 < argc) g_f16_pix_elems = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--f16-watom") && i + 1 < argc) g_f16_watom = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--f16-ocatom") && i + 1 < argc) g_f16_ocatom = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--f16-nest") && i + 1 < argc) g_f16_nest = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--w-ic-tap") && i + 1 < argc) { g_w_ic_tap = atoi(argv[++i]); g_pattern = 1; }
      else if (!strcmp(argv[i], "--in-ch-id")) { g_in_ch_id = 1; g_pattern = 1; }
      else if (!strcmp(argv[i], "--in-one-hot") && i + 1 < argc) { g_in_one_hot = atoi(argv[++i]); g_pattern = 1; }
      else if (!strcmp(argv[i], "--in-ones")) { g_in_ones = 1; g_pattern = 1; }
      else if (!strcmp(argv[i], "--w-ones")) { g_w_ones = 1; g_pattern = 1; }
      else if (!strcmp(argv[i], "--in-ruler")) { g_in_ruler = 1; g_pattern = 1; }
      else if (!strcmp(argv[i], "--in-cancel")) { g_in_cancel = 1; g_pattern = 1; }
      else if (!strcmp(argv[i], "--kh") && i + 1 < argc) c.kh = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--ic") && i + 1 < argc) c.ic_real = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--oc") && i + 1 < argc) c.oc_real = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--out-zp") && i + 1 < argc) c.output_zero_point = (unsigned)strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--bn-cfg") && i + 1 < argc) g_bn_cfg_override = (int)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--bn-alu") && i + 1 < argc) g_bn_alu_override = (uint32_t)strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--bn-mul") && i + 1 < argc) g_bn_mul_override = (uint32_t)strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--bn-relux") && i + 1 < argc) g_bn_relux_override = (uint32_t)strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--dpu-data-format") && i + 1 < argc) { g_dpu_data_format = (uint32_t)strtoul(argv[++i], NULL, 0); g_dpu_data_format_given = 1; }
      else if (!strcmp(argv[i], "--bn-nrdma") && i + 1 < argc) g_bn_nrdma_override = (int)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--bn-table-offset") && i + 1 < argc) g_bn_table_offset_override = (int32_t)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--bn-table-bias") && i + 1 < argc) g_bn_table_bias = (int32_t)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--bn-table-step") && i + 1 < argc) g_bn_table_step = (int32_t)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--bn-table-f32-bias") && i + 1 < argc) { g_bn_table_fp32 = 1; g_bn_table_f32_bias = strtof(argv[++i], NULL); }
      else if (!strcmp(argv[i], "--bn-table-f32-step") && i + 1 < argc) { g_bn_table_fp32 = 1; g_bn_table_f32_step = strtof(argv[++i], NULL); }
      else if (!strcmp(argv[i], "--bs-cfg") && i + 1 < argc) g_bs_cfg_override = (int)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--bs-alu") && i + 1 < argc) g_bs_alu_override = (uint32_t)strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--bs-mul") && i + 1 < argc) g_bs_mul_override = (uint32_t)strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--brdma-cfg") && i + 1 < argc) g_brdma_cfg_override = (int)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--brdma-bias") && i + 1 < argc) g_brdma_bias_base = (int32_t)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--brdma-bias-step") && i + 1 < argc) g_brdma_bias_step = (int32_t)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--brdma-mul") && i + 1 < argc) g_brdma_mul_value = (int16_t)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--brdma-mul-step") && i + 1 < argc) g_brdma_mul_step = (int16_t)strtol(argv[++i], NULL, 0);
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
      else if (!strcmp(argv[i], "--wgrad-dot")) { g_wgrad_dot_mode = 1; g_pattern = 1; }
      else if (!strcmp(argv[i], "--wgrad-conv3x3")) {
         g_wgrad_dot_mode = 1; g_wgrad_conv3x3_mode = 1; g_pattern = 1;
      }
      else if (!strcmp(argv[i], "--wgrad-ic") && i + 1 < argc) g_wgrad_ic_idx = (unsigned)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--wgrad-kx") && i + 1 < argc) g_wgrad_kx = (unsigned)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--wgrad-ky") && i + 1 < argc) g_wgrad_ky = (unsigned)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--wgrad-batch") && i + 1 < argc) g_wgrad_batch_count = (unsigned)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--deconv-flip")) g_deconv_flip = 1;
      else if (!strcmp(argv[i], "--enable-mask") && i + 1 < argc) enable_mask = (uint32_t)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--w-oc2") && i + 1 < argc) w_oc2 = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--w-kernels") && i + 1 < argc) { g_w_kernels_override = atoi(argv[++i]); g_wk_given = 1; }
      else if (!strcmp(argv[i], "--ic-pad") && i + 1 < argc) g_ic_pad_override = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--hi-zero")) g_hi_zero = 1;
      else if (!strcmp(argv[i], "--fill-ymaj")) { /* 行主序已为默认，保留兼容 */ }
       else if (!strcmp(argv[i], "--fill-xiw")) g_fill_xiw = 1;
       else if (!strcmp(argv[i], "--ic1-layout") && i + 1 < argc) g_ic1_layout = atoi(argv[++i]);
       else if (!strcmp(argv[i], "--in-pad-rows") && i + 1 < argc) g_in_pad_rows = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--out-ymaj")) { /* 已为默认布局，保留兼容 */ }
      else if (!strcmp(argv[i], "--asym-w")) { g_pat_asym = 1; g_pattern = 1; }
      else if (!strcmp(argv[i], "--pat2")) g_pat2 = 1;
      else if (!strcmp(argv[i], "--w-tap") && i + 1 < argc) {
         sscanf(argv[++i], "%d:%d", &g_w_tap_kx, &g_w_tap_ky);
         g_pattern = 1;
      }
      else if (!strcmp(argv[i], "--icreal-reg") && i + 1 < argc) g_icreal_reg = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--din-ch") && i + 1 < argc) g_din_ch = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--dma-ch") && i + 1 < argc) g_dma_ch = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--ws1") && i + 1 < argc) g_ws1 = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--kernel-group") && i + 1 < argc) g_kernel_group = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--weight-bank") && i + 1 < argc) g_weight_bank = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--grains") && i + 1 < argc) g_grains = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--pp-val") && i + 1 < argc) g_pp_val = (int)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--bs-table-in-wt") && i + 1 < argc) g_bs_tbl_wt_off = strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--job-flags") && i + 1 < argc) g_job_flags = (int)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--warmup-submits") && i + 1 < argc) g_warmup = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--entries") && i + 1 < argc) g_entries_override = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--subcore") && i + 1 < argc) g_subcore_mode = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--task-count") && i + 1 < argc) g_task_count = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--sp-prefix")) g_sp_prefix = 1;
      else if (!strcmp(argv[i], "--in-file") && i + 1 < argc) g_in_file = argv[++i];
      else if (!strcmp(argv[i], "--out-file") && i + 1 < argc) g_out_file_c = argv[++i];
      else if (!strcmp(argv[i], "--in-xiw")) g_in_xiw = 1;
      else if (!strcmp(argv[i], "--task-blocks")) g_task_blocks = 1;
      else if (!strcmp(argv[i], "--task-delta4")) { g_task_blocks = 1; g_task_delta4 = 1; }
      else if (!strcmp(argv[i], "--subcore-single0")) g_subcore_single0 = 1;
      else if (!strcmp(argv[i], "--subcore-hi")) g_subcore_hi = 1;
      else if (!strcmp(argv[i], "--task-out-step")) g_task_out_step = 1;
      else if (!strcmp(argv[i], "--subcore-range") && i + 1 < argc) g_subcore_range = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--chain-id")) g_chain_id = 1;
      else if (!strcmp(argv[i], "--chain-settle-ms") && i + 1 < argc) g_chain_settle_ms = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--multi-core")) g_multi_core = 1;
      else if (!strcmp(argv[i], "--chain-n") && i + 1 < argc) g_chain_n = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--chain-align8")) g_chain_align8 = 1;
      else if (!strcmp(argv[i], "--chain-pc")) g_chain_pc = 1;
      else if (!strcmp(argv[i], "--chain-stub")) g_chain_stub = 1;
      else if (!strcmp(argv[i], "--chain-op-en") && i + 1 < argc) g_chain_op_en = (uint32_t)strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--dump-regcmd") && i + 1 < argc) g_dump_regcmd = argv[++i];
      else if (!strcmp(argv[i], "--dump-tile") && i + 1 < argc) g_dump_tile = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--chain-stride") && i + 1 < argc)
         g_chain_stride = (unsigned)strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--chain-off") && i + 1 < argc)
         g_chain_off = (unsigned)strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--ov") && i + 1 < argc && g_ov_n < 16) {
         char *eq = strchr(argv[++i], '=');
         if (eq) { *eq = 0; g_ov_reg[g_ov_n] = (uint32_t)strtoul(argv[i], NULL, 0);
                   g_ov_val[g_ov_n++] = (uint32_t)strtoul(eq + 1, NULL, 0); }
      }
       else if (!strcmp(argv[i], "--line") && i + 1 < argc) g_line_override = atoi(argv[++i]);
       else if (!strcmp(argv[i], "--surf") && i + 1 < argc) g_surf_override = atoi(argv[++i]);
       else if (!strcmp(argv[i], "--act-size") && i + 1 < argc) g_act_size = (unsigned)strtoul(argv[++i], NULL, 0);
       else if (!strcmp(argv[i], "--out-off") && i + 1 < argc) g_out_off = (unsigned)strtoul(argv[++i], NULL, 0);
       else if (!strcmp(argv[i], "--tile-oh") && i + 1 < argc) g_tile_oh = atoi(argv[++i]);
       else if (!strcmp(argv[i], "--tile-auto")) g_tile_auto = 1;
       else if (!strcmp(argv[i], "--tile-ow") && i + 1 < argc) g_tile_ow = atoi(argv[++i]);
       else if (!strcmp(argv[i], "--tile-rows-max") && i + 1 < argc) g_tile_rows_max = atoi(argv[++i]);
       else if (!strcmp(argv[i], "--no-auto-cfg")) g_auto_cfg = 0;
       else if (!strcmp(argv[i], "--ic-groups") && i + 1 < argc) g_ic_groups = atoi(argv[++i]);
       else if (!strcmp(argv[i], "--oc-groups") && i + 1 < argc) g_oc_groups = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--poke") && i + 1 < argc && g_poke_n < 256) {
         char *colon = strchr(argv[++i], ':');
         if (colon) { *colon = 0;
            g_poke_atom[g_poke_n] = atoi(argv[i]);
            g_poke_val[g_poke_n] = (int)strtol(colon + 1, NULL, 0);
            g_poke_n++; }
      }
      else if (!strcmp(argv[i], "--auto-fix")) g_autofix = 1;
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
   if (g_fp16) {
      /* fp16 原子：16 B = 8 个 fp16；输入通道按 8 对齐，输出通道按 16 对齐
       * （vendor fp16 对照：DATAIN_CHANNEL 8、DATAOUT_CHANNEL 15=16-1）。 */
      /* 实测（§7.18）：fp16 权重 ic 行必须是完整的 16 元素（32 B）原子，
       * 否则「单 ic 平面 + 多 oc 组」会 submit ETIMEDOUT。
       * 实测（2026-09-30）：权重 ic 块（ic2=32 元素）不得是「半块」——硬件会把
       * c.ic 上取整到 32 的倍数。若 c.ic 不是 32 的倍数且 >16（如 40/48/72/80），
       * 硬件读到的第二个 ic 块错位 ⇒ 全量 pattern MISMATCH（`--ic-pad` 到 32 倍数即
       * BIT-EXACT）。故 ic≤16 → 16；否则按 32 对齐。 */
      c.ic = (c.ic_real <= 2u * FP16_ATOM_ELEMS)
             ? (2u * FP16_ATOM_ELEMS)
             : ALIGN_UP(c.ic_real, 4u * FP16_ATOM_ELEMS);
      /* 实测（§7.18）：fp16 每 tap 的 oc 跨度必须按 oc 原子（16）对齐，
       * 否则第二个 oc 组（不完整组）的 per-tap 步进会错位。 */
      if (g_autofix || w_oc2 == 0)
         w_oc2 = (int)ALIGN_UP(c.oc_real, 2u * FP16_ATOM_ELEMS);
      /* --auto-fix 的 int8 分支可能已把 kernels 设成非 16 对齐值，必须同步 */
      if (!g_wk_given) g_w_kernels_override = w_oc2;
      g_w_layout = 0;
   }
   if (g_ic_pad_override > 0) c.ic = (unsigned)g_ic_pad_override;
   c.oc = g_fp16 ? ALIGN_UP(MAX2(c.oc_real, 16), 16)
                 : ALIGN_UP(MAX2(c.oc_real, 32), 32);
   if (mode_deconv && !ow_given) { c.ow = c.iw; }
   if (mode_deconv && !iw_given) { /* keep */ }
   if (!ow_given && !mode_deconv) c.ow = c.iw;
   if (!oh_given) c.oh = c.ih;
   if (!ow_given && mode_deconv && c.deconv_stride) c.ow = c.iw * c.deconv_stride;
   if (!iw_given) { /* keep */ }

   printf("== min_conv: iw=%u ih=%u ic=%u(real %u) -> ow=%u oh=%u oc=%u(real %u) "
          "k=%ux%u s=%u pad=%u deconv=%d/%u in=0x%02x w=0x%02x prec=%s ==\n",
          c.iw, c.ih, c.ic, c.ic_real, c.ow, c.oh, c.oc, c.oc_real,
          c.kw, c.kh, c.stride_x, c.pad_left, c.deconv, c.deconv_stride,
          in_byte, w_byte, g_fp16 ? "fp16" : "int8");
   if (g_wgrad_dot_mode)
      printf("WGRAD_DOT: mode=%s batch=%u coord=(kx=%u,ky=%u,ic=%u) reduce P=%u activation-gradient products per output channel\n",
             g_wgrad_conv3x3_mode ? "conv3x3-synthetic" : "vector",
             g_wgrad_batch_count, g_wgrad_kx, g_wgrad_ky, g_wgrad_ic_idx, c.ic_real);

   int fd = open(dev, O_RDWR);
   if (fd < 0) { fprintf(stderr, "open %s: %s\n", dev, strerror(errno)); return 1; }
   printf("opened %s\n", dev);

   /* §7.33/§7.34：激活 BO 必须同时容纳「全局输入」与「输出面」；大 ic × 大 ih 的输入可达
    * 数 MB，旧默认 1 MB 会溢出（静默踩内存）。这里按几何自动放大（可用 --act-size/--out-off
    * 覆盖）。ic 切分时输出侧还需要 (G+1) 个独立 partial 面。 */
   {
      unsigned in_est = (c.ic_real == 1 && !g_fp16)
            ? MAX2(c.iw, FEATURE_ATOMIC_SIZE) * MAX2(c.ih, FEATURE_ATOMIC_SIZE)
            : c.iw * c.ih * (g_fp16 ? fp16_planes(&c) * fp16_pixel_bytes(&c)
                                    : DIV_ROUND_UP(c.ic, FEATURE_ATOMIC_SIZE) * FEATURE_ATOMIC_SIZE);
      if (g_out_off < in_est + 65536u) g_out_off = ALIGN_UP(in_est, 4096u) + 65536u;
      unsigned need = g_out_off + in_est + 65536u;
      if (g_act_size < need) {
         printf("ACT-BO: 按几何自动放大 %u -> %u B（输入 %u B，OUT_OFF=%u）\n",
                g_act_size, need, in_est, g_out_off);
         g_act_size = need;
      }
   }
   if (g_ic_groups > 1) {
      unsigned one = c.ow * c.oh *
                     DIV_ROUND_UP(c.oc, (unsigned)(g_fp16 ? FP16_ATOM_ELEMS
                                                          : FEATURE_ATOMIC_SIZE)) *
                     FEATURE_ATOMIC_SIZE;
      unsigned need2 = g_out_off + (unsigned)(g_ic_groups + 1) * one + 65536u;
      if (g_act_size < need2) {
         printf("IC-GROUPS: 激活 BO 再放大 %u -> %u B（%d 组 × 输出面 %u B）\n",
                g_act_size, need2, g_ic_groups, one);
         g_act_size = need2;
      }
   }

   struct bo act, wt, bias, rcbo, tk;
   if (bo_create(fd, g_act_size, 0, &act)) return 1;
   if (bo_create(fd, 1 << 20, 0, &wt)) return 1;   /* 权重区：容纳深 ic/oc 打包（ic64 oc64 fp16 描述符 73728 B） */
   if (bo_create(fd, 1 << 12, 0, &bias)) return 1;
   if (bo_create(fd, 1 << 13, 0, &rcbo)) return 1;
   if (bo_create(fd, 1 << 12, RKNPU_MEM_KERNEL_MAPPING, &tk)) return 1;

   const uint32_t OUT_OFF = g_out_off;

   /* 输出区清零：CMA 页在进程间复用时不会自动清零，残留上一次输出会污染
    * --poke/--dump-bin 的基线比较。统一填 0x80（量化零点）。 */
   {
      unsigned out_atoms = c.ow * c.oh *
                           DIV_ROUND_UP(c.oc, g_fp16 ? (unsigned)FP16_ATOM_ELEMS
                                                     : (unsigned)FEATURE_ATOMIC_SIZE);
      unsigned out_bytes = out_atoms * FEATURE_ATOMIC_SIZE;
      if (OUT_OFF + out_bytes > act.size)
         out_bytes = (OUT_OFF < act.size) ? act.size - OUT_OFF : 0;
      memset((uint8_t *)act.map + OUT_OFF, 0x80, out_bytes);
   }

   /* 输入 */
   unsigned in_atom_elems = g_fp16 ? (unsigned)FP16_ATOM_ELEMS
                                   : (unsigned)FEATURE_ATOMIC_SIZE;
   uint32_t in_size = (c.ic_real == 1 && !g_fp16)
         ? MAX2(c.iw, FEATURE_ATOMIC_SIZE) * MAX2(c.ih, FEATURE_ATOMIC_SIZE)
         : c.iw * c.ih * (g_fp16 ? fp16_planes(&c) * fp16_pixel_bytes(&c)
                                 : DIV_ROUND_UP(c.ic, in_atom_elems)
                                   * FEATURE_ATOMIC_SIZE);
    if (!pad_byte_given) pad_byte = 0x80;
    uint32_t in_shift = 0;
    if (g_in_pad_rows > 0 && !(c.ic_real == 1 && !g_fp16)) {
       in_shift = (uint32_t)g_in_pad_rows * c.iw *
                 (g_fp16 ? fp16_pixel_bytes(&c) : FEATURE_ATOMIC_SIZE);
       memset((uint8_t *)act.map, 0, in_shift);
    }
    fill_input((uint8_t *)act.map + in_shift, &c, in_byte, pad_byte);
    for (unsigned p = 0; p < g_poke_n; p++)
       ((uint8_t *)act.map)[g_poke_atom[p] * FEATURE_ATOMIC_SIZE] =
          (uint8_t)g_poke_val[p];
    if (g_poke_n)
       printf("POKE: %u atom(s) injected\n", g_poke_n);

   /* 权重：默认按 NPU 打包布局填写；--w-uniform 则整段同字节 */
   unsigned wskernels = g_w_kernels_override > 0
                        ? (unsigned)g_w_kernels_override
                        : ALIGN_UP(MAX2(c.oc, WEIGHT_ATOMIC_SIZE), 2);
   /* oc_w_eff 在打包与 fp16 CPU 参考中都要用，提到这里 */
   unsigned oc_w_eff = w_oc2 > 0 ? (unsigned)w_oc2 : ALIGN_UP(c.oc_real, 2);
   uint32_t wt_size;
   if (g_fp16) {
      unsigned wa = (unsigned)(g_f16_watom > 0 ? g_f16_watom : 32);
      unsigned icg = DIV_ROUND_UP(c.ic, wa) * MIN2(c.ic, wa);
      wt_size = c.kw * c.kh * icg * wskernels * 2u;
   } else {
      wt_size = c.kw * c.kh * c.ic * wskernels;
   }
   if (w_uniform) {
      memset(wt.map, w_byte, wt_size);
   } else {
      memset(wt.map, 0, wt_size);
      unsigned wn = g_fp16 ? fill_weights_f16((uint8_t *)wt.map, &c, w_byte, oc_w_eff)
                           : fill_weights((uint8_t *)wt.map, &c, w_byte, w_oc2);
      {
         const uint8_t *wp = (const uint8_t *)wt.map;
         uint32_t h = 2166136261u;
         for (uint32_t k = 0; k < wt_size; k++) { h ^= wp[k]; h *= 16777619u; }
         printf("weights packed: %u B of descriptor %u B  [w_kernels=%u w_oc2=%u layout=%c] fnv1a=%08x\n",
                wn, wt_size, wskernels, oc_w_eff, "ACD"[g_w_layout], h);
      }
   }

   /* ---------- §7.33/§7.36 ic / oc 切分规划 ----------
    * ic 组：切输入通道（各组 partial 输出 → host fp32 累加）。
    * oc 组（§7.36）：切输出通道（各组写**互不相交的输出平面** ⇒ **无需累加**），
    *   权重描述符按组缩小，用于 oc 大到权重 bank 放不下的情形。
    * 组描述符统一打进权重 BO 里 wt_size 之后的一个**共享 scratch 槽**（全局描述符保留在
    *   偏移 0，CPU 参考仍按全局描述符读权重）。 */
   struct ic_group { unsigned ic0, ic_real, ic; };
   struct oc_group { unsigned oc0, oc_real, oc, wk, oc_w; };
   static struct ic_group icpl[64];
   static struct oc_group ocpl[64];
   int n_icg = 1, n_ocg = 1;
   unsigned oc_atom = g_fp16 ? (unsigned)FP16_ATOM_ELEMS : (unsigned)FEATURE_ATOMIC_SIZE;
   unsigned grp_desc_off = ALIGN_UP(wt_size, 64);
   unsigned grp_desc_max = 0;

   if (g_ic_groups > 1 || g_oc_groups > 1) {
      if (g_ic_groups > 1 && !g_fp16) {
         fprintf(stderr, "--ic-groups 仅支持 fp16（int8 部分和会被重量化）\n"); return 2;
      }
      if (c.deconv) { fprintf(stderr, "--ic/oc-groups 不支持 deconv\n"); return 2; }
      if (g_bn_nrdma_override >= 0 || g_brdma_cfg_override >= 0) {
         fprintf(stderr, "--ic/oc-groups 暂不能与 BN/BRDMA 表同用（描述符区冲突）\n"); return 2;
      }
      if (g_bs_tbl_wt_off >= 0) { fprintf(stderr, "--ic/oc-groups 暂不能与 --bs-table-in-wt 同用\n"); return 2; }
   }

   if (g_ic_groups > 1) {
      if (c.ic_real <= 1) { fprintf(stderr, "--ic-groups 需要 ic_real > 1\n"); return 2; }
      unsigned gsz = ALIGN_UP(DIV_ROUND_UP(c.ic_real, (unsigned)g_ic_groups), 4u * FP16_ATOM_ELEMS);
      unsigned ic0 = 0;
      n_icg = 0;
      while (ic0 < c.ic_real && n_icg < 64) {
         unsigned n = MIN2(gsz, c.ic_real - ic0);
         icpl[n_icg].ic0 = ic0;
         icpl[n_icg].ic_real = n;
         icpl[n_icg].ic = (n <= 2u * FP16_ATOM_ELEMS)
                          ? (2u * FP16_ATOM_ELEMS) : ALIGN_UP(n, 4u * FP16_ATOM_ELEMS);
         ic0 += n; n_icg++;
      }
   }

   if (g_oc_groups > 1) {
      if (c.oc_real <= 1) { fprintf(stderr, "--oc-groups 需要 oc_real > 1\n"); return 2; }
      unsigned gsz = ALIGN_UP(DIV_ROUND_UP(c.oc_real, (unsigned)g_oc_groups),
                              (unsigned)FEATURE_ATOMIC_SIZE);
      unsigned oc0 = 0;
      n_ocg = 0;
      while (oc0 < c.oc_real && n_ocg < 64) {
         unsigned n = MIN2(gsz, c.oc_real - oc0);
         ocpl[n_ocg].oc0 = oc0;
         ocpl[n_ocg].oc_real = n;
         if (g_fp16) {
            ocpl[n_ocg].oc = ALIGN_UP(MAX2(n, 2u * FP16_ATOM_ELEMS), 2u * FP16_ATOM_ELEMS);
            ocpl[n_ocg].wk = ALIGN_UP(n, 2u * FP16_ATOM_ELEMS);
         } else {
            ocpl[n_ocg].oc = ALIGN_UP(MAX2(n, WEIGHT_ATOMIC_SIZE), WEIGHT_ATOMIC_SIZE);
            ocpl[n_ocg].wk = (n <= WEIGHT_ATOMIC_SIZE) ? ALIGN_UP(n, 2u)
                                                       : ALIGN_UP(n, WEIGHT_ATOMIC_SIZE);
         }
         ocpl[n_ocg].oc_w = ocpl[n_ocg].wk;
         oc0 += n; n_ocg++;
      }
   }

   if (n_icg > 1 || n_ocg > 1) {
      /* 最大组描述符（用于共享 scratch 槽的空间检查） */
      unsigned wa = (unsigned)(g_f16_watom > 0 ? g_f16_watom : 32);
      for (int oi = 0; oi < n_ocg; oi++)
         for (int gi = 0; gi < n_icg; gi++) {
            unsigned ic_g = (n_icg > 1) ? icpl[gi].ic : c.ic;
            unsigned wk_g = (n_ocg > 1) ? ocpl[oi].wk
                                        : (g_w_kernels_override > 0
                                           ? (unsigned)g_w_kernels_override
                                           : ALIGN_UP(MAX2(c.oc, WEIGHT_ATOMIC_SIZE), 2));
            unsigned icg_ = g_fp16 ? (DIV_ROUND_UP(ic_g, wa) * MIN2(ic_g, wa)) : ic_g;
            unsigned d = c.kw * c.kh * icg_ * wk_g * (g_fp16 ? 2u : 1u);
            if (d > grp_desc_max) grp_desc_max = d;
         }
      if (grp_desc_off + grp_desc_max > wt.size) {
         fprintf(stderr, "组描述符超出权重 BO（%u + %u > %u；用 --wt-size 放大）\n",
                 grp_desc_off, grp_desc_max, wt.size);
         return 1;
      }
      /* ic 组需要 (G+1) 个输出面（partial）*/
      if (n_icg > 1) {
         unsigned need = OUT_OFF + (unsigned)(n_icg + 1) *
                         (c.ow * c.oh * FEATURE_ATOMIC_SIZE *
                          DIV_ROUND_UP(c.oc, (unsigned)FP16_ATOM_ELEMS));
         if (need > act.size) {
            fprintf(stderr, "--ic-groups partial 区超出激活 BO（需 %u > %u；用 --act-size 放大）\n",
                    need, act.size);
            return 1;
         }
      }
      printf("GROUPS: ic=%d oc=%d  desc_scratch@%u (max %u B)  oc_atom=%u\n",
             n_icg, n_ocg, grp_desc_off, grp_desc_max, oc_atom);
      if (n_icg > 1) {
         printf("  IC:");
         for (int g = 0; g < n_icg; g++)
            printf(" [%u..%u real=%u align=%u]", icpl[g].ic0, icpl[g].ic0 + icpl[g].ic_real,
                   icpl[g].ic_real, icpl[g].ic);
         printf("\n");
      }
      if (n_ocg > 1) {
         printf("  OC:");
         for (int g = 0; g < n_ocg; g++)
            printf(" [%u..%u real=%u align=%u wk=%u]", ocpl[g].oc0, ocpl[g].oc0 + ocpl[g].oc_real,
                   ocpl[g].oc_real, ocpl[g].oc, ocpl[g].wk);
         printf("\n");
      }
   }

   if (g_bn_nrdma_override >= 0) {
      g_bn_table_offset = g_bn_table_offset_override > 0
                          ? (unsigned)g_bn_table_offset_override
                          : ALIGN_UP(wt_size, 64);
      g_bn_alu_dma_count = MIN2(c.oc, 256);
      int32_t *coeff = (int32_t *)((uint8_t *)wt.map + g_bn_table_offset);
      uint32_t hash = 2166136261u;
      for (unsigned ch = 0; ch < g_bn_alu_dma_count; ch++) {
         if (g_fp16 && g_bn_table_fp32) {
            float f = g_bn_table_f32_bias + (float)ch * g_bn_table_f32_step;
            memcpy(&coeff[ch], &f, sizeof(f));
            memcpy(&g_bn_alu_dma[ch], &f, sizeof(f));
         } else {
            g_bn_alu_dma[ch] = g_bn_table_bias + (int32_t)ch * g_bn_table_step;
            coeff[ch] = g_bn_alu_dma[ch];
         }
         uint32_t v = (uint32_t)coeff[ch];
         for (unsigned b = 0; b < 4; b++) { hash ^= (v >> (8 * b)) & 0xff; hash *= 16777619u; }
      }
      printf("DPU BN DMA table: NRDMA=0x%x base_offset=0x%x channels=%u bias=%d step=%d fp32=%d f32_bias=%g f32_step=%g fnv1a=%08x\n",
             (unsigned)g_bn_nrdma_override, g_bn_table_offset,
             g_bn_alu_dma_count, g_bn_table_bias, g_bn_table_step,
             g_fp16 && g_bn_table_fp32, g_bn_table_f32_bias, g_bn_table_f32_step, hash);
   }

   /* 偏置：全 0 */
   memset(bias.map, 0, bias.size);
   if (g_brdma_cfg_override >= 0) {
      unsigned groups = DIV_ROUND_UP(c.oc, 8);
      unsigned bytes = groups * 64;
      /* §7.19：可把表放进 weight BO（vendor/§7.12 的做法用 BN_BASE_ADDR 指 weight BO） */
      unsigned long bs_tbl_off = 0;
      uint8_t *bs_tbl_base = (uint8_t *)bias.map;
      uint32_t bs_tbl_addr = (uint32_t)bias.dma_addr;
      if (g_bs_tbl_wt_off >= 0) {
         bs_tbl_off = g_bs_tbl_wt_off == 0 ? ALIGN_UP(wt_size, 64) : (unsigned long)g_bs_tbl_wt_off;
         if (bs_tbl_off + bytes > wt.size) {
            fprintf(stderr, "BRDMA table in wt too large: %lu + %u > %u\n",
                    bs_tbl_off, bytes, wt.size);
            return 1;
         }
         bs_tbl_base = (uint8_t *)wt.map + bs_tbl_off;
         bs_tbl_addr = (uint32_t)wt.dma_addr + (uint32_t)bs_tbl_off;
      } else if (bytes > bias.size) {
         fprintf(stderr, "BRDMA table too large: %u > %u\n", bytes, bias.size);
         return 1;
      }
      g_brdma_table_channels = MIN2(c.oc, 256);
      for (unsigned group = 0; group < groups; group++) {
         uint8_t *chunk = bs_tbl_base + group * 64;
         int32_t *biases = (int32_t *)chunk;
         int16_t *multipliers = (int16_t *)(chunk + 48);
         for (unsigned lane = 0; lane < 8; lane++) {
            unsigned ch = group * 8 + lane;
            if (ch < c.oc) {
               g_brdma_bias_table[ch] = g_brdma_bias_base +
                                        (int32_t)ch * g_brdma_bias_step;
               g_brdma_mul_table[ch] = (int16_t)(g_brdma_mul_value +
                                                 (int32_t)ch * g_brdma_mul_step);
               biases[lane] = g_brdma_bias_table[ch];
               multipliers[lane] = g_brdma_mul_table[ch];
            }
         }
      }
      printf("DPU BRDMA table: cfg=0x%x base=0x%x channels=%u bias=%d bias_step=%d mul=%d shift=%u bytes=%u\n",
             (unsigned)g_brdma_cfg_override, (uint32_t)bias.dma_addr,
             g_brdma_table_channels, g_brdma_bias_base,
             g_brdma_bias_step, (int)g_brdma_mul_value,
             (g_bs_mul_override >> 8) & 0x3f, bytes);
   }

   c.in_addr   = (uint32_t)act.dma_addr + in_shift;
   c.out_addr  = (uint32_t)act.dma_addr + OUT_OFF;
   c.wt_addr   = (uint32_t)wt.dma_addr;
   c.bias_addr = (uint32_t)bias.dma_addr;
   if (g_bs_tbl_wt_off >= 0)
      c.bias_addr = (uint32_t)wt.dma_addr + (uint32_t)(g_bs_tbl_wt_off == 0
                       ? ALIGN_UP(wt_size, 64) : (unsigned long)g_bs_tbl_wt_off);

   printf("BO: act@%08x(0x%zx) wt@%08x bias@%08x regcmd@%08x task obj=0x%llx\n",
          (uint32_t)act.dma_addr, (size_t)act.size, (uint32_t)wt.dma_addr,
          (uint32_t)bias.dma_addr, (uint32_t)rcbo.dma_addr,
          (unsigned long long)tk.obj_addr);
   printf("layout: input=%u B (groups=%u) output@+%u  weight_desc=%u B\n",
          in_size, DIV_ROUND_UP(c.ic, FEATURE_ATOMIC_SIZE), OUT_OFF, wt_size);

   /* ---------- §7.31 输出行切分（row tiling）规划 ----------
    * 单次 submit 的 CNA 输入 feature tile ≤ DATA_BANK × 32768 B（CBUF bank，§7.31；旧
    * 硬编码 DATA_BANK=1 即 §7.30 的「32 KB」）。auto-cfg 已按 in_size 自动取足 bank；
    * 仅当「输入 bank + 权重 bank > 12」（真·CBUF 容量）时才必须切分：按输出行拆，把每个
    * tile 需要的输入行块 [g0, g0+rows) 拷成紧凑 stage，pad_top 相应下调（中间 tile 为 0）。
    * tile 输出面互不相交 ⇒ 无跨 submit 累加。 */
   unsigned plane_row_bytes = c.iw * (g_fp16 ? fp16_pixel_bytes(&c) : (unsigned)FEATURE_ATOMIC_SIZE);
   unsigned tile_planes     = g_fp16 ? fp16_planes(&c)
                                     : DIV_ROUND_UP(c.ic, FEATURE_ATOMIC_SIZE);
   unsigned tile_row_bytes  = plane_row_bytes * tile_planes;
   unsigned tile_groups     = DIV_ROUND_UP(c.oc, g_fp16 ? (unsigned)FP16_ATOM_ELEMS
                                                        : (unsigned)FEATURE_ATOMIC_SIZE);
   unsigned data_banks_avail = cfg_data_banks_avail(&c);
   unsigned weights_banks_need = cfg_wbanks_need(&c);
   if (n_icg > 1 || n_ocg > 1) {
      /* §7.36：切分后每组的权重描述符更小 ⇒ 按**最坏组**重算权重 bank 与可用 data bank */
      int save_wk = g_w_kernels_override;
      unsigned wmax = 1;
      for (int oi2 = 0; oi2 < n_ocg; oi2++)
         for (int gi2 = 0; gi2 < n_icg; gi2++) {
            struct conv_cfg cd = c;
            if (n_icg > 1) { cd.ic_real = icpl[gi2].ic_real; cd.ic = icpl[gi2].ic; }
            if (n_ocg > 1) { cd.oc_real = ocpl[oi2].oc_real; cd.oc = ocpl[oi2].oc;
                             g_w_kernels_override = (int)ocpl[oi2].wk; }
            unsigned wb = cfg_wbanks_need(&cd);
            if (wb > wmax) wmax = wb;
         }
      g_w_kernels_override = save_wk;
      weights_banks_need = wmax;
      data_banks_avail = (CBUF_BANKS > wmax) ? (unsigned)(CBUF_BANKS - wmax) : 1u;
      if (data_banks_avail > 11u) data_banks_avail = 11u;
      printf("GROUPS-BANKS: worst group needs %u weight bank(s) -> data banks available %u\n",
             wmax, data_banks_avail);
   }
   unsigned in_banks_full = cfg_data_banks_need(in_size);
   /* §7.34：单个 ic 组的「一行全通道」字节（组合切分时的真正行字节） */
   unsigned group_row_bytes = tile_row_bytes;
   if (n_icg > 1) {
      unsigned mx = 0;
      for (int g = 0; g < n_icg; g++) {
         unsigned rb = plane_row_bytes * (icpl[g].ic / FP16_ATOM_ELEMS);
         if (rb > mx) mx = rb;
      }
      group_row_bytes = mx;
      printf("IC-GROUPS: 单组行字节 %u B（全通道行 %u B）\n", mx, tile_row_bytes);
   }
   int want_tile = (g_tile_oh > 0 || g_tile_ow > 0 || g_tile_auto || g_tile_rows_max > 0);
   static struct tile_plan plan[2048];
   int n_tiles = 0;
   unsigned tile_maxT = c.oh;
   unsigned tile_max_plane = c.ow * c.oh;      /* max(Tc*T)：tile 输出面原子数 */

   printf("CBUF-PLAN: in=%u B needs %u data bank(s); weights need %u bank(s); "
          "data banks available = %u (auto_cfg=%d)\n",
          in_size, in_banks_full, weights_banks_need, data_banks_avail, g_auto_cfg);
   if (!c.deconv && !(c.ic_real == 1 && !g_fp16))
      printf("CBUF-PLAN: %s\n",
             in_banks_full + weights_banks_need <= (unsigned)CBUF_BANKS
                ? "单次 submit 可容纳（无需切分）"
                : "输入+权重 bank 超出 12 ⇒ 真·容量越界，需按行/ic 切分");

   if (want_tile && c.deconv) {
      fprintf(stderr, "--tile-* 不支持 deconv 模式\n"); return 2;
   }
   if (want_tile && (c.ic_real == 1 && !g_fp16)) {
      fprintf(stderr, "--tile-* 不支持 IC=1 专用输入布局\n"); return 2;
   }
   if (want_tile && g_fp16 && fp16_pixel_bytes(&c) != FEATURE_ATOMIC_SIZE) {
      fprintf(stderr, "--tile-* 需要 16 B 像素原子（fp16 请用 --f16-pix-elems 8）\n"); return 2;
   }

   g_tile_on = want_tile;
   if (!want_tile) {
      plan[0].oy0 = 0; plan[0].T = c.oh; plan[0].g0 = 0;
      plan[0].rows = c.ih; plan[0].pad_top = c.pad_top;
      plan[0].ox0 = 0; plan[0].Tc = c.ow; plan[0].x0 = 0;
      plan[0].cols = c.iw; plan[0].pad_left = c.pad_left;
      n_tiles = 1;
      g_tile_max_in = in_size;
   } else {
      /* §7.35：二维空间切分（输出行 × 输出列）。列切分覆盖「一整行都放不进 bank」的
       * 极宽输入：tile 的输入行字节 = 列 tile 的输入宽 × 通道字节。 */
      unsigned chbpp = n_icg > 1 ? (group_row_bytes / c.iw) : (tile_row_bytes / c.iw);
      unsigned cap = data_banks_avail * 32768u;
      unsigned T = c.oh, Tc = c.ow;
      if (g_tile_oh > 0 && !g_tile_auto && g_tile_ow <= 0) {
         T = (unsigned)g_tile_oh;                    /* 显式 tile_oh：不自动收紧 */
      } else if (g_tile_oh > 0 && g_tile_ow > 0 && !g_tile_auto) {
         T = (unsigned)g_tile_oh;
         Tc = (unsigned)g_tile_ow;
      } else {
         unsigned max_rows_budget;
         if (cap < chbpp * c.kw) {
            fprintf(stderr, "单行最小 tile（%u B）已超可用 bank（%u B）⇒ 请减小 ic 组（--ic-groups）\n",
                    chbpp * c.kw, cap);
            return 2;
         }
         /* §7.35：SURF_STRIDE = L·(staged_ih/4−1) 编码 4 行组 ⇒ staged 行数下限 4。
          * 若「4 行 × 全宽」都超 bank，则按列切，给 4 行留出预算。 */
         unsigned rows_min = 4u;
         if (group_row_bytes * rows_min > cap) {
            unsigned iw_max = chbpp ? ((cap / rows_min) / chbpp) : c.iw;
            if (iw_max < 1u) iw_max = 1u;
            if (iw_max > c.iw) iw_max = c.iw;
            Tc = (iw_max > c.kw)
               ? ((iw_max - c.kw) / (c.stride_x ? c.stride_x : 1u) + 1u) : 1u;
         } else {
            Tc = c.ow;
         }
         if (g_tile_ow > 0) Tc = (unsigned)g_tile_ow;
         if (Tc < 1) Tc = 1;
         if (Tc > c.ow) Tc = c.ow;
         /* 列 tile 的输入行字节（stage 后每行的字节） */
         unsigned col_iw = (Tc - 1) * (c.stride_x ? c.stride_x : 1u) + c.kw;
         if (col_iw > c.iw) col_iw = c.iw;
         unsigned row_bytes_tile = col_iw * chbpp;
         max_rows_budget = row_bytes_tile ? (cap / row_bytes_tile) : c.ih;
         if (max_rows_budget < c.kh) max_rows_budget = c.kh;
         while (max_rows_budget < rows_min) max_rows_budget += 1u;   /* 4 行组下限 */
         if (g_tile_rows_max > 0 && (unsigned)g_tile_rows_max < max_rows_budget)
            max_rows_budget = (unsigned)g_tile_rows_max;
         T = (max_rows_budget - c.kh) / (c.stride_y ? c.stride_y : 1u) + 1u;
         if (T < 1) T = 1;
         if (T > c.oh) T = c.oh;
      }
      if (g_tile_ow > 0) {
         Tc = (unsigned)g_tile_ow;
         if (Tc < 1) Tc = 1;
         if (Tc > c.ow) Tc = c.ow;
      }
      unsigned oy = 0;
      g_tile_max_in = 0;
      unsigned max_cols = 0;
      while (oy < c.oh && n_tiles < (int)(sizeof(plan) / sizeof(plan[0]))) {
         unsigned tt = MIN2(T, c.oh - oy);
         int iy0 = (int)(oy * c.stride_y) - (int)c.pad_top;
         int iy1 = iy0 + (int)((tt - 1) * c.stride_y + c.kh);
         unsigned g0 = iy0 < 0 ? 0u : (unsigned)iy0;
         unsigned g1 = iy1 <= 0 ? 0u : MIN2(c.ih, (unsigned)iy1);
         if (g1 <= g0) g1 = MIN2(c.ih, g0 + 1u);
         unsigned ox = 0;
         while (ox < c.ow && n_tiles < (int)(sizeof(plan) / sizeof(plan[0]))) {
            unsigned tc = MIN2(Tc, c.ow - ox);
            int ix0 = (int)(ox * c.stride_x) - (int)c.pad_left;
            int ix1 = ix0 + (int)((tc - 1) * c.stride_x + c.kw);
            unsigned x0 = ix0 < 0 ? 0u : (unsigned)ix0;
            unsigned x1 = ix1 <= 0 ? 0u : MIN2(c.iw, (unsigned)ix1);
            if (x1 <= x0) x1 = MIN2(c.iw, x0 + 1u);
            plan[n_tiles].oy0      = oy;
            plan[n_tiles].T        = tt;
            plan[n_tiles].g0       = g0;
            plan[n_tiles].rows     = g1 - g0;
            plan[n_tiles].pad_top  = (unsigned)((int)g0 - iy0);
            plan[n_tiles].ox0      = ox;
            plan[n_tiles].Tc       = tc;
            plan[n_tiles].x0       = x0;
            plan[n_tiles].cols     = x1 - x0;
            plan[n_tiles].pad_left = (unsigned)((int)x0 - ix0);
            if ((g1 - g0) < 4u && (g_tile_oh > 0 || g_tile_ow > 0) && !g_tile_auto && n_tiles == 0)
               printf("WARN: staged 行数 %u < 4（SURF_STRIDE 的 4 行组下限）⇒ 结果可能错\n",
                      g1 - g0);
            if (tt > tile_maxT) tile_maxT = tt;
            if (tc > max_cols) max_cols = tc;
            if (tt * tc > tile_max_plane) tile_max_plane = tt * tc;
            {
               unsigned b = (g1 - g0) * (x1 - x0) * chbpp;
               if (b > g_tile_max_in) g_tile_max_in = b;
            }
            ox += tc;
            n_tiles++;
         }
         oy += tt;
      }
      printf("TILE: %d tile(s)  tile=%ux%u  row_bytes(全通)=%u B  max staged=%u B  maxTc=%u\n",
             n_tiles, T, Tc, tile_row_bytes, g_tile_max_in, max_cols);
   }

   /* §7.31 容量判别：单次 submit 的输入 tile 字节必须 ≤ DATA_BANK × 32768，且
    * 输入 bank + 权重 bank ≤ 12。默认 auto-cfg 下 DATA_BANK = min(need, 12−权重 bank)，
    * 故判据即 need > 实际分配；--no-auto-cfg 时按旧口径（DATA_BANK=1，除 depthwise）。 */
   if (n_icg > 1 && !g_tile_on)
      g_tile_max_in = c.iw * c.ih * (group_row_bytes / plane_row_bytes);   /* 单组全图字节 */
   int ooe = 0;
   unsigned need_banks = cfg_data_banks_need(g_tile_max_in);
   if (!c.deconv && !(c.ic_real == 1 && !g_fp16)) {
      unsigned eff_in_banks;
      if (g_weight_bank >= 0)       eff_in_banks = (unsigned)(CBUF_BANKS - g_weight_bank);
      else if (g_auto_cfg)          eff_in_banks = MIN2(need_banks, data_banks_avail);
      else                          eff_in_banks = c.depthwise ? 7u : 1u;
      ooe = (need_banks > eff_in_banks);
   }
   if (ooe)
      printf("WARN: 单次 submit 需 %u data bank(s)（in tile %u B）而只分到 %u ⇒ CBUF 容量越界"
             "（§7.31），硬件静默算错；请切分（--tile-auto / --tile-oh）\n",
             need_banks, g_tile_max_in,
             g_weight_bank >= 0 ? (unsigned)(CBUF_BANKS - g_weight_bank)
                                : (g_auto_cfg ? MIN2(need_banks, data_banks_avail)
                                              : (c.depthwise ? 7u : 1u)));

   /* §7.31 staging BO：输入 stage @0，tile 输出 stage 之后（与输入不重叠） */
   struct bo tile;
   unsigned tile_in_off = 0, tile_out_off = 0, tile_in_bytes = 0, tile_out_bytes = 0;
   if (g_tile_on) {
      tile_in_bytes  = ALIGN_UP(g_tile_max_in ? g_tile_max_in : 4096u, 4096u);
      if (n_icg > 1) {                     /* §7.34：组合切分时 stage 只含本组平面 */
         unsigned mx = 0;
         for (int ti2 = 0; ti2 < n_tiles; ti2++) {
            unsigned b = plan[ti2].rows * group_row_bytes;
            if (b > mx) mx = b;
         }
         if (mx) tile_in_bytes = ALIGN_UP(mx, 4096u);
      }
      tile_out_bytes = ALIGN_UP(tile_max_plane * tile_groups * FEATURE_ATOMIC_SIZE, 4096u);
      tile_out_off   = tile_in_bytes + 16384u;
      unsigned tsize = tile_out_off + tile_out_bytes + 16384u;
      if (bo_create(fd, tsize, 0, &tile)) return 1;
      printf("TILE BO: %u B (in stage @0 %u B, out stage @%u %u B)\n",
             tsize, tile_in_bytes, tile_out_off, tile_out_bytes);
   }

   struct rknpu_task *t = (struct rknpu_task *)tk.map;
   memset(t, 0, sizeof(*t));
   t->op_idx        = 0;
   t->enable_mask   = enable_mask;
   t->int_mask      = 0x300;
   t->int_clear     = 0x1ffff;
   t->regcfg_amount = 0;
   t->regcmd_addr   = rcbo.dma_addr;
   if (g_task_count > 1) {
      for (int k = 0; k < g_task_count && k < 64; k++) {
         struct rknpu_task *tk2 = &t[k];
         tk2->op_idx      = 1;   /* §7.183：vendor task 事件 op_idx 全 1（含链首），对齐 */
         tk2->enable_mask = enable_mask;
         tk2->int_mask    = 0x300;
         tk2->int_clear   = 0x1ffff;
         /* §7.62：仿 vendor —— vendor 三个 task 的 regcfg_amount 全相同且**非零**
          *（69，对应 272 B 的块）⇒ 这里复制同一次 emit 的真实 amount，而不是 0。 */
         tk2->regcfg_amount = 0;   /* §7.110：稍后在 n 已知时统一写入（旧版在此取 t->regcfg_amount，
                                   * 但那一行在 n 计算**之前** ⇒ 实际写入的是 0 ✗ ——这正是
                                   * 「多 task 静默只跑 1 个」的直接原因）。 */
         tk2->regcmd_addr = rcbo.dma_addr;
      }
      printf("TASKS: %d identical task(s) per submit（驱动按 last task 的 int_mask 等中断）\n",
             g_task_count);
   }

   bo_sync(fd, &act, 1);
   bo_sync(fd, &wt, 1);
   bo_sync(fd, &bias, 1);
   bo_sync(fd, &rcbo, 1);
   bo_sync(fd, &tk, 1);
   if (g_tile_on) bo_sync(fd, &tile, 1);

    struct rknpu_submit sub;
    memset(&sub, 0, sizeof(sub));
    sub.flags          = g_job_flags >= 0 ? (uint32_t)g_job_flags
                          : (RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG);
    sub.timeout        = 6000;
    sub.task_start     = 0;
    sub.task_number    = (uint32_t)g_task_count;
    sub.task_obj_addr  = tk.obj_addr;
   /* §7.57：**实测**把 task_base_addr 设成 task obj 的 dma（driver 会写进 PC_DMA_BASE_ADDR）
    * 会让**单 task 提交也 6.3s 超时** ⇒ 正常路径必须保持 0（PC 不靠它取 task 描述符）。 */
    sub.core_mask      = 0;
    sub.fence_fd       = -1;
    for (int i = 0; i < 5; i++) { sub.subcore_task[i].task_start = 0; sub.subcore_task[i].task_number = 1; }
    if (g_subcore_mode == 1) { for (int i = 1; i < 5; i++) sub.subcore_task[i].task_number = 0; }        /* only subcore 0 */
    else if (g_subcore_mode == 2) { for (int i = 0; i < 5; i++) sub.subcore_task[i].task_number = (i == 1) ? 1 : 0; }  /* only subcore 1 */
    else if (g_subcore_mode == 3) { for (int i = 0; i < 5; i++) sub.subcore_task[i].task_number = (i == 2) ? 1 : 0; }  /* only subcore 2 */
    else if (g_subcore_mode == 4) { for (int i = 0; i < 5; i++) sub.subcore_task[i].task_start = 0, sub.subcore_task[i].task_number = 3; } /* 3x on all */
    else if (g_subcore_mode == 5) { sub.core_mask = 0x7; }                                              /* mask=0x7, default subcore_task */
    else if (g_subcore_mode == 6) { sub.core_mask = 0x7; for (int i = 1; i < 5; i++) sub.subcore_task[i].task_number = 0; } /* mask=0x7 + subcore0 only */
    /* §7.58：driver 用 subcore_task[core].task_number **覆盖** args->task_number（§7.45 根因）
     * ⇒ 只设 sub.task_number=N 时实际只跑 1 个 task。这里按 §7.45 的修法显式设成 N。 */
    if (g_task_count > 1 && g_subcore_range > 0) {   /* §7.149：每 subcore 连续区间（vendor 形态） */
       int sbase = g_subcore_hi ? 2 : 0;   /* §7.170：3 核（use_core_num==3）读槽位 2..4 */
       for (int k = 0; k < 5; k++) { sub.subcore_task[k].task_start = 0; sub.subcore_task[k].task_number = 0; }
       for (int k = 0; k < 3; k++) {
          unsigned st = (unsigned)k * (unsigned)g_subcore_range;
          if (st >= (unsigned)g_task_count) break;
          unsigned cnt = (unsigned)g_subcore_range;
          if (st + cnt > (unsigned)g_task_count) cnt = (unsigned)g_task_count - st;
          sub.subcore_task[sbase + k].task_start = st; sub.subcore_task[sbase + k].task_number = cnt;
       }
       if (g_subcore_hi) sub.core_mask = 0x7;
       printf("SUBCORE-RANGE: 每 subcore %d 个连续 task（core_mask=%u，槽位基 %d，总 %d）\n",
              g_subcore_range, sub.core_mask, sbase, g_task_count);
    } else if (g_task_count > 1 && g_subcore_single0) {   /* §7.127：全放 subcore0 */
       for (int k = 0; k < 5; k++) { sub.subcore_task[k].task_start = 0; sub.subcore_task[k].task_number = 0; }
       sub.subcore_task[0].task_number = (uint32_t)g_task_count;
       printf("SUBCORE-SINGLE0: %d 个 task 全放 subcore0（core_mask=%u）\n", g_task_count, sub.core_mask);
    } else if (g_task_count > 1 && g_subcore_hi) {
       /* §7.170（驱动反汇编定论）：`rknpu_job_subcore_commit_pc` 在 use_core_num==3 时
        * 读 subcore_task[core_index + 2]（槽位 **2..4**），∈{1,2} 时读 [0..1]。
        * 旧 §7.61 多核实验把 per-core 条目填在 0..2 ⇒ core0 跑 task#2、core1/2 拿到
        * task_number=0 的越界描述符 ⇒ 不稳定/部分写入。本分支把 per-core 条目填进 2..4。 */
       for (int k = 0; k < 5; k++) { sub.subcore_task[k].task_start = 0; sub.subcore_task[k].task_number = 0; }
       for (int k = 0; k < g_task_count && k < 3; k++) {
          sub.subcore_task[2 + k].task_start  = (uint32_t)k;
          sub.subcore_task[2 + k].task_number = 1u;
       }
       sub.core_mask = 0x7;
       printf("SUBCORE-HI: 槽位 2..4 = 每 core 1 个 task（core_mask=0x7，共 %d task）\n", g_task_count);
    } else if (g_task_count > 1 && g_subcore_mode == 0) {
       /* §7.115：**vendor 的真实形态**是「每个 subcore 各 1 个 task」
        *（`subcore_tasks=[[0,1],[0,1],[0,1]]` + args `task_number=3` + `core_mask=0`），
        * 而旧版把 N 个 task 全塞给 subcore0 ✗ ⇒ 那正是「单 subcore >1 task ⇒ 超时」的不支持路径 ✗。
        * 现在改为逐 subcore 分配 1 个 task（task_start=k）✓。 */
       for (int k = 0; k < 5; k++) {
          sub.subcore_task[k].task_start  = (uint32_t)k;
          sub.subcore_task[k].task_number = (k < g_task_count) ? 1u : 0u;
       }
       printf("SUBCORE-SPLIT: 每 subcore 1 个 task（k=%d 个 subcore 参与，core_mask=%u）\n",
              g_task_count, sub.core_mask);
    }

   /* §7.21：BS/BRDMA 表只在"上一次 submit 的 BS 阶段是激活的"时才被重载（且首个
    * BS 激活的 submit 可能读到旧表或挂住）。热身 submit（同一 regcmd）重新武装该
    * 状态；结果丢弃。BS 激活时默认自动 1 次。 */
   int bs_active = (g_bs_cfg_override >= 0) &&
                   !(((uint32_t)g_bs_cfg_override) & 0x01u);
   int warm = g_warmup >= 0 ? g_warmup : (bs_active ? 3 : 0);
   if (warm > 0)
      printf("WARMUP: %d BS-active submit(s) before the measured one (bs_active=%d)\n",
             warm, bs_active);

   /* ---------- §7.31 逐 tile 发射 + 提交（无切分时 n_tiles=1，等价旧路径） ---------- */
   uint8_t *gin  = (uint8_t *)act.map + in_shift;
   uint8_t *gout = (uint8_t *)act.map + OUT_OFF;
   unsigned out_plane_bytes = c.ow * c.oh * FEATURE_ATOMIC_SIZE;
   unsigned out_row_bytes   = c.ow * FEATURE_ATOMIC_SIZE;
   unsigned out_total_bytes = out_plane_bytes * tile_groups;   /* 全部通道平面（§7.33 partial 步长） */
   int ret = 0;
   double ms_total = 0;
   long long hw_total = 0;
   unsigned n = 0;

   unsigned oc_plane_off = 0;                       /* §7.36：oc 组的输出平面偏移 */
   for (int oi = 0; oi < n_ocg; oi++) {
   oc_plane_off = (n_ocg > 1) ? (ocpl[oi].oc0 / oc_atom) * out_plane_bytes : 0;
   for (int gi = 0; gi < n_icg; gi++) {
   struct conv_cfg cgv;
   const struct conv_cfg *cgb = &c;                  /* 本组基础 cfg */
   if (n_icg > 1 || n_ocg > 1) {
      cgv = c;
      if (n_icg > 1) {
         cgv.ic_real = icpl[gi].ic_real;
         cgv.ic      = icpl[gi].ic;
         cgv.in_addr = c.in_addr + (icpl[gi].ic0 / FP16_ATOM_ELEMS) * c.iw * c.ih * FEATURE_ATOMIC_SIZE;
      }
      if (n_ocg > 1) {
         cgv.oc_real = ocpl[oi].oc_real;
         cgv.oc      = ocpl[oi].oc;
      }
      cgv.wt_addr  = c.wt_addr + grp_desc_off;
      cgv.out_addr = c.out_addr + oc_plane_off
                   + (n_icg > 1 ? (size_t)(gi + 1) * out_total_bytes : 0);
      cgb = &cgv;
      /* 组描述符打进共享 scratch 槽（全局描述符保留在偏移 0） */
      {
         struct conv_cfg cd = *cgb;
         unsigned wk = (n_ocg > 1)
                     ? ocpl[oi].wk
                     : (g_w_kernels_override > 0
                        ? (unsigned)g_w_kernels_override
                        : ALIGN_UP(MAX2(c.oc, WEIGHT_ATOMIC_SIZE), 2));
         unsigned ow = g_fp16 ? ((n_ocg > 1) ? ocpl[oi].oc_w : oc_w_eff)
                              : ((n_ocg > 1) ? ocpl[oi].oc_w : (unsigned)(w_oc2 > 0 ? w_oc2 : 0));
         g_ic_offset = (n_icg > 1) ? (int)icpl[gi].ic0 : 0;
         g_oc_offset = (n_ocg > 1) ? (int)ocpl[oi].oc0 : 0;
         g_w_kernels_override = (int)wk;
         memset((uint8_t *)wt.map + grp_desc_off, 0, grp_desc_max);
         if (g_fp16) fill_weights_f16((uint8_t *)wt.map + grp_desc_off, &cd, w_byte, ow);
         else        fill_weights((uint8_t *)wt.map + grp_desc_off, &cd, w_byte, (int)ow);
         bo_sync(fd, &wt, 1);
      }
   }
   for (int ti = 0; ti < n_tiles; ti++) {
      struct conv_cfg ct = *cgb;
      const struct tile_plan *tp = &plan[ti];

      if (g_tile_on) {
         unsigned rows = tp->rows;
         /* stage 该 tile 的输入行块：布局 [plane][y][x][16 B]，与全局同构但高度=rows。
          * §7.34：组合 ic 切分时只 stage 本组的平面（源 = 组首平面 + tile 行区间）。 */
         unsigned st_planes = n_icg > 1 ? (cgb->ic / FP16_ATOM_ELEMS) : tile_planes;
         const uint8_t *gin_g = gin + (n_icg > 1
               ? (size_t)(icpl[gi].ic0 / FP16_ATOM_ELEMS) * c.ih * plane_row_bytes : 0);
         unsigned col_bytes = tp->cols * FEATURE_ATOMIC_SIZE;   /* §7.35：列 tile 的行内字节 */
         for (unsigned p = 0; p < st_planes; p++)
            for (unsigned r = 0; r < rows; r++)
               memcpy((uint8_t *)tile.map + tile_in_off + ((size_t)p * rows + r) * col_bytes,
                      gin_g + (size_t)p * c.ih * plane_row_bytes
                            + (size_t)(tp->g0 + r) * plane_row_bytes
                            + (size_t)tp->x0 * FEATURE_ATOMIC_SIZE,
                      col_bytes);
         ct.iw        = tp->cols;
         ct.ih        = rows;
         ct.ow        = tp->Tc;
         ct.oh        = tp->T;
         ct.pad_left  = tp->pad_left;
         ct.pad_top   = tp->pad_top;
         ct.in_addr  = (uint32_t)tile.dma_addr + tile_in_off;
         ct.out_addr = (uint32_t)tile.dma_addr + tile_out_off;
         /* tile 输出面清零（0x80 = 量化零点），用于暴露"未被写入"的区域 */
         memset((uint8_t *)tile.map + tile_out_off, 0x80,
                tp->Tc * tp->T * tile_groups * FEATURE_ATOMIC_SIZE);
         bo_sync(fd, &tile, 1);
      }

      rc = rc_buf;
      n = emit_conv(&ct);
      if (g_sp_prefix) {   /* §7.113：vendor 的每个块（含 4 条短块）都以 S_POINTER=0xe 开头 */
         memmove(rc_buf + 2, rc_buf, (size_t)n * 8);
         rc_buf[0] = ((uint64_t)0xeu << 16) | 0x4004u;
         rc_buf[1] = ((uint64_t)0xeu << 16) | 0x5004u;
         n += 2;
         printf("SP-PREFIX: 块首补写 S_POINTER=0xe（n=%u）\n", n);
      }
      if (ti == 0) {
         printf("OV: %u 组覆盖（已在 trailer 前发射，计入 regcfg_amount）\n", g_ov_n);
         for (int k2 = 0; k2 < g_ov_n; k2++)
            printf("OV: reg=%#x val=%#x\n", g_ov_reg[k2], g_ov_val[k2]);
         printf("regcmd: %u entries (%u uint64 = %u bytes), regcfg_amount=%u  (per tile)\n",
                n, n, n * 8, n - RKNPU_PC_DATA_EXTRA_AMOUNT);
      }
      memcpy(rcbo.map, rc_buf, n * 8);
      if (g_dump_regcmd && ti == g_dump_tile) {
         FILE *fr = fopen(g_dump_regcmd, "w");
         if (fr) {
            for (unsigned z = 0; z < n; z++)
               fprintf(fr, "0x%04x 0x%08x\n", (unsigned)(rc_buf[z] & 0xffffu),
                       (unsigned)((rc_buf[z] >> 16) & 0xffffffffu));
            fclose(fr);
            printf("DUMP-REGCMD: %u entries -> %s\n", n, g_dump_regcmd);
         }
      }
      t->regcfg_amount = n - RKNPU_PC_DATA_EXTRA_AMOUNT;
      if (g_task_count > 1) {   /* §7.110：把真实 amount 写进**全部**描述符（vendor 三描述符同值 ✓） */
         for (int k = 0; k < g_task_count && k < 64; k++)
            t[k].regcfg_amount = t->regcfg_amount;
         printf("TASKS: 已把 regcfg_amount=%u 写入 %d 个描述符（§7.110 顺序修正）\n",
                t->regcfg_amount, g_task_count);
         if (g_task_blocks) {   /* §7.114：复制 N 份块到 rcbo 不同偏移（64 B 对齐，vendor 同） */
            unsigned stride = (n * 8u + 63u) & ~63u;
            size_t need = (size_t)stride * (size_t)g_task_count;
            if (need > rcbo.size) {
               printf("TASK-BLOCKS: rcbo 太小（需 %zu > %zu）⇒ 放弃\n", need, rcbo.size);
               return 1;
            }
             for (int k = 0; k < g_task_count && k < 64; k++) {
                /* §7.132：真实负载里 80% 的 task 描述符 regcfg_amount = **9**（= 36 B = 4 条增量 ✓）
                 * ⇒ 后续 task 只写 4 条（这里取 emit 流的前 4 条 = 配置头 ✓，等价于"只改变量" ✓）。 */
                unsigned ents = (g_task_delta4 && k > 0) ? 4u : n;
                memcpy((uint8_t *)rcbo.map + (size_t)k * stride, rc_buf, (size_t)ents * 8);
                if (g_task_out_step && !g_tile_on && ents == n) {
                   /* §7.170：task k 的输出区 = OUT_OFF + k*step（patch 块内 DPU_DST_BASE_ADDR）。
                    * 各 task 输出面互不相交 ⇒ 提交后逐区对 task0 比对即可证明「该 task 真的执行了」。
                    * 条目格式 = (target<<48) | (val<<16) | reg ⇒ **必须保留 target 高位**。 */
                   uint64_t *blk = (uint64_t *)((uint8_t *)rcbo.map + (size_t)k * stride);
                   uint32_t step = (uint32_t)(ALIGN_UP(out_total_bytes, 4096u)) * (uint32_t)k;
                   for (unsigned z = 0; z < ents; z++) {
                      if ((unsigned)(blk[z] & 0xffffu) == REG_DPU_DST_BASE_ADDR) {
                         uint64_t tgt = blk[z] & 0xffff000000000000ull;
                         uint32_t v = (uint32_t)((blk[z] >> 16) & 0xffffffffu);
                         blk[z] = (tgt & ~(uint64_t)0xffffffffffffull)
                                | ((uint64_t)(uint32_t)(v + step) << 16) | (uint64_t)REG_DPU_DST_BASE_ADDR;
                      }
                   }
                }
             }
            for (int k = 0; k < g_task_count && k < 64; k++)
               t[k].regcmd_addr = (uint32_t)rcbo.dma_addr + (unsigned)k * stride;
            printf("TASK-BLOCKS: %d 块（首块 %u B%s，stride=%u，rcbo=%zu B）\n",
                   g_task_count, n * 8, g_task_delta4 ? "，后续 32 B 增量" : "", stride, rcbo.size);
         }
      }
      t->regcmd_addr   = rcbo.dma_addr;

      /* §7.42：真 2-task 链。task1 与 task0 几何相同、只改输出地址，用于验证 PC 按
       * task 描述符逐个取指（regcmd_addr/regcfg_amount 各自独立）与「等 last task 的中断」。
       * 依赖型链（task1 读 task0 的输出）留待下一轮。 */
      if (g_chain_id && !g_tile_on) {
         if (!g_chain_off) g_chain_off = OUT_OFF + 262144u;
         if (!g_chain_stride) g_chain_stride = ALIGN_UP(out_total_bytes, 4096u);
         if (g_chain_n < 1) g_chain_n = 1;
         if (g_chain_n > 8) g_chain_n = 8;
         /* §7.42 修正：PC 按「第一个 task 的 regcfg_amount + EXTRA」为步长**连续**取
          * task 的 regcmd 块（vendor trace 实测步长 = amount+11 或 与 emit 的 EXTRA 一致），
          * 因此 task1 的 regcmd 必须紧接在 task0 之后（offB = n*8），不能对齐填充。 */
         unsigned offB = n * 8;
         unsigned need = g_chain_off + out_total_bytes + 4096u;
         if (need > act.size) {
            printf("CHAIN: 激活 BO 不足（需 %u > %u）⇒ 跳过链（用 --act-size 放大）\n",
                   need, act.size);
         } else if (offB + n * 8 > rcbo.size) {
            printf("CHAIN: regcmd BO 不足（需 %u > %u）⇒ 跳过链\n", offB + n * 8, rcbo.size);
         } else {
            unsigned nA = (unsigned)(rc - rc_buf);      /* task0 段（= n） */
            t->op_idx = 1;                              /* §7.186：链首 op_idx=1（vendor 形） */
            unsigned L = nA;                            /* 每段长度（--chain-pc 时 = 含尾写） */
            unsigned pc_amt = 0;                        /* --chain-pc 时每段的 regcfg_amount */
            if (g_chain_pc) {
               /* §7.46：探出「带 4 条 PC 尾写」的段长 L（等长 ⇒ 可预先算出下一块地址），
                * 然后按 L 重放 task0（带指向 task1 的 PC 尾写）。
                * §7.173：OP_EN=0x18（链式延续）；§7.176：段内不追加标准 trailer。 */
               g_no_trailer = 1;
               g_ov_n = 4;
               g_ov_reg[0] = REG_PC_BASE_ADDRESS;      g_ov_val[0] = 0;
               g_ov_reg[1] = REG_PC_REGISTER_AMOUNTS;  g_ov_val[1] = 0;
               g_ov_reg[2] = REG_PC_VERSION;           g_ov_val[2] = 0;
               g_ov_reg[3] = REG_PC_OPERATION_ENABLE;  g_ov_val[3] = g_chain_op_en;   /* §7.181 旋钮 */
               rc = rc_buf;
               L = emit_conv(&ct);
               pc_amt = L - RKNPU_PC_DATA_EXTRA_AMOUNT;
               /* §7.179：vendor 所有 PC_BASE 目标全部 0x80（128 B=16 条）对齐——段长须 16 条对齐，
                * 否则段 2 地址非对齐（我方曾 +0x410）。段内零填充不进 fetch 窗（fetch=amount+4）。 */
               L = ALIGN_UP(L, 16);
               rc = rc_buf;
               uint32_t nxt_addr = (uint32_t)rcbo.dma_addr + L * 8;
               uint32_t next_amt = g_chain_stub ? 9u : pc_amt;  /* §7.181：下一段=stub 时 fetch 窗不同 */
               uint32_t nxt_amt  = (uint32_t)((next_amt + 4u + 2u - 1u) / 2u - 1u);
               g_ov_val[0] = nxt_addr & 0xfffffff0u;   /* PC_BASE_ADDRESS = 下一块地址 */
               g_ov_val[1] = nxt_amt;                  /* PC_REGISTER_AMOUNTS（下一块） */
               unsigned n0 = emit_conv(&ct);            /* 重放 task0（带尾写） */
               memcpy(rcbo.map, rc_buf, (size_t)n0 * 8);
               if (L > n0)   /* §7.179：段 0 的 0x80 对齐填充区必须清零（防陈旧字节） */
                  memset((uint8_t *)rcbo.map + n0 * 8, 0, (size_t)(L - n0) * 8);
               t->regcfg_amount = (uint32_t)pc_amt;
               rc = rc_buf + L;
               printf("CHAIN-PC: 每段 %u entries = %u B（amount=%u，下一块 @+%u，"
                      "PC_REGISTER_AMOUNTS=%u）\n", L, L * 8, pc_amt, L * 8, nxt_amt);
            }
            unsigned n_prev = nA;
            for (int k = 1; k < g_chain_n; k++) {
               struct conv_cfg cb = ct;
               cb.out_addr = (uint32_t)act.dma_addr + g_chain_off +
                             (unsigned)(k - 1) * g_chain_stride;   /* 区域 0..N-2 = task1..N-1 */
               unsigned off_k;
               if (g_chain_align8) {
                  /* vendor trace 实测：相邻 task 的 regcmd 块步长 = ALIGN_UP(amount+4, 8) 条
                   *（= 64 B 对齐）⇒ 块起点按 8 条对齐（中间留 0–7 条空隙）。 */
                  unsigned cur = (unsigned)(rc - rc_buf);
                  unsigned aligned = ALIGN_UP(cur, 8);
                  for (unsigned z = cur; z < aligned; z++) rc_buf[z] = 0;
                  rc += (aligned - cur);
                  off_k = aligned * 8;
               } else {
                  off_k = (unsigned)(rc - rc_buf) * 8;   /* 紧接前一段（连续） */
               }
               if (g_chain_pc) {
                  /* §7.46 vendor 实测：task k 尾部写「下一块地址 + 下一块 amount + VER + OP_EN」；
                   * **最后一个 task 不写 PC_BASE_ADDRESS**，只把 PC_REGISTER_AMOUNTS 置 0（终止）。 */
                  g_ov_reg[0] = REG_PC_BASE_ADDRESS;
                  g_ov_reg[1] = REG_PC_REGISTER_AMOUNTS;
                  g_ov_reg[2] = REG_PC_VERSION;
                  g_ov_reg[3] = REG_PC_OPERATION_ENABLE;
                  g_ov_val[2] = 0;
                  g_ov_val[3] = 0xd;   /* 探测段（不计入实验） */
                  if (k + 1 < g_chain_n) {
                     uint32_t nb_addr = (uint32_t)rcbo.dma_addr + (unsigned)(k + 1) * L * 8;
                     uint32_t nb_amt  = (uint32_t)((pc_amt + 4u + 2u - 1u) / 2u - 1u);
                     g_ov_val[0] = nb_addr & 0xfffffff0u;
                     g_ov_val[1] = nb_amt;
                     g_ov_val[3] = g_chain_op_en;   /* §7.181：链式延续（可调） */
                     g_ov_n = 4;
                  } else {
                     /* §7.186：vendor conv 终端真值（v2f 决定性）= [pad(tgt0/reg0/val0),
                      * AMOUNTS=0(tgt0x101), VERSION=0(tgt0x041), OP_EN=0x1d(tgt0x081)] —— pad 在首、
                      * OP_EN 含 bit0，AMOUNTS=0 负责停链。旧 attn 形态 [AMT,VER,OP=0x18,pad]
                      * ⇒ task1 配置写完但 DPU 永不启动（v2e 复现挂点、v2f 换形态即闭合）。 */
                     g_ov_n = 4;
                     g_ov_reg[0] = 0xFFFF;              /* PAD 哨兵 → emit_raw(0,0,0) */
                     g_ov_val[0] = 0;
                     g_ov_reg[1] = REG_PC_REGISTER_AMOUNTS;
                     g_ov_val[1] = 0;
                     g_ov_reg[2] = REG_PC_VERSION;
                     g_ov_val[2] = 0;
                     g_ov_reg[3] = REG_PC_OPERATION_ENABLE;
                     g_ov_val[3] = g_chain_op_en;
                  }
               }
               unsigned before = (unsigned)(rc - rc_buf);
               unsigned nk;
               unsigned cnt;
               if (g_chain_stub) {
                  /* §7.181：conv→stub 二分——段 1..N-1 = attn 增量块形状（9 配置 + 5 尾写），
                    * DST 指向 our 输出区（真写，可哨兵校验）；终端段 AMOUNTS=0 终止。
                    * 段长 = 14 条（fetch=round_up(9+4,2)=14）；段 0 的尾写 AMOUNTS 已按 9 推导。 */
                  uint32_t out_k = (uint32_t)(act.dma_addr + g_chain_off + (unsigned)(k - 1) * g_chain_stride);
                  emit_raw(0x1001, 0x4004, 0xe);
                  emit_raw(0x2001, 0x5004, 0xe);
                  emit_raw(0x1001, REG_DPU_DST_BASE_ADDR, out_k);
                  emit_raw(0x1001, REG_DPU_WDMA_SIZE_0, 0x30007);
                  emit_raw(0x1001, REG_DPU_WDMA_SIZE_1, 0x10000);
                  emit_raw(0x2001, 0x5018, (uint32_t)act.dma_addr);
                  emit_raw(0x2001, 0x5020, 0);
                  emit_raw(0x2001, 0x502c, 0);
                  emit_raw(0x2001, 0x5038, 0);
                  if (k + 1 < g_chain_n) {
                     emit_raw(0x101, REG_PC_BASE_ADDRESS, ((uint32_t)rcbo.dma_addr + (unsigned)(k + 1) * L * 8) & 0xfffffff0u);
                     emit_raw(0x101, REG_PC_REGISTER_AMOUNTS, (9u + 4u + 2u - 1u) / 2u - 1u);
                     emit_raw(0x41, REG_PC_VERSION, 0);
                     emit_raw(0x81, REG_PC_OPERATION_ENABLE, g_chain_op_en);
                  } else {
                     /* §7.183：终端 = AMOUNTS=0 + OP_EN=旋钮值。conv 块的执行触发依赖尾写
                      * OP_EN bit0（0x18 ⇒ task1 配置写完但 DPU 永不启动 ⇒ UNTOUCHED），
                      * vendor mul 终端 0x18 只适用其轻量 op；AMOUNTS=0 仍负责停链。 */
                     emit_raw(0x101, REG_PC_REGISTER_AMOUNTS, 0);
                     emit_raw(0x41, REG_PC_VERSION, 0);
                     emit_raw(0x81, REG_PC_OPERATION_ENABLE, g_chain_op_en);
                  }
                  emit_raw(0, 0, 0);
                  nk = (unsigned)(rc - rc_buf);
                  cnt = 14;
                  memcpy((uint8_t *)rcbo.map + off_k, rc_buf + before, (size_t)cnt * 8);
                } else {
                   nk = emit_conv(&cb);
                   /* §7.186：cnt/copy 一律以 before（本段 emit 起点）为准 —— 旧代码用
                    * n_prev（上一段未对齐端）当字节偏移，chain-pc 光标复位到 L 后源错位
                    * 14 条，段 1 尾写永远落不进 BO（seg1=字节错位垃圾 ⇒ 长期假象
                    * 「PC 不发起 seg1 取指」）。 */
                   cnt = nk - before;
                }
               /* §7.59：链的「下一块地址」按首段长度 L 线性推进（(k+1)*L*8）⇒ 每个段必须
                * **恰好 L 条**。否则 PC 跳到段中间 ⇒ 解析到垃圾寄存器 ⇒ 提交挂（hw_elapse=0）。
                * 这里把不一致的段补零/截断到 L，并打印警告。 */
               if (g_chain_pc && cnt != L && !g_chain_stub) {
                  printf("CHAIN-WARN: 段 %d 长度 %u ≠ 首段 %u ⇒ 规整到 %u（补零）\n",
                         k, cnt, L, L);
                  for (unsigned z = before + cnt; z < before + L; z++) rc_buf[z] = 0;
                  cnt = L;
                  rc = rc_buf + before + L;
                  nk = (unsigned)(rc - rc_buf);
               }
               if (g_chain_pc && before != (unsigned)k * L && !g_chain_stub)
                  printf("CHAIN-WARN: 段 %d 起点 %u ≠ k*L=%u（PC 跳转仍会错）\n",
                         k, before, (unsigned)k * L);
                if (!g_chain_stub)
                   memcpy((uint8_t *)rcbo.map + off_k, rc_buf + before, (size_t)cnt * 8);
               memset((uint8_t *)act.map + g_chain_off + (unsigned)(k - 1) * g_chain_stride,
                      g_fp16 ? 0x00 : 0x80, out_total_bytes);
               struct rknpu_task *tk2 = &t[k];
               memset(tk2, 0, sizeof(*tk2));
               tk2->op_idx        = (uint32_t)k + 1;   /* §7.186：vendor op_idx=[1,2]（从不 0；
                                                          §7.183 的排除实验在 copy-bug 修复前，无效） */
               tk2->enable_mask   = enable_mask;
               tk2->int_mask      = 0x300;
               tk2->int_clear     = 0x1ffff;
               tk2->regcfg_amount = cnt - RKNPU_PC_DATA_EXTRA_AMOUNT;
               if (g_chain_pc) tk2->regcfg_amount = (uint32_t)pc_amt;
               if (g_chain_stub) tk2->regcfg_amount = 9;
               tk2->regcmd_addr   = (uint32_t)rcbo.dma_addr + off_k;
               n_prev = nk;
            }
            unsigned nB = g_chain_pc ? (unsigned)(rc - rc_buf) : n_prev;
            if (g_chain_pc) { rc = rc_buf; g_no_trailer = 0; }   /* 复位游标 + 恢复标准 trailer（§7.176） */
            sub.task_number   = (uint32_t)g_chain_n;
            /* §7.45 根因：driver 的 rknpu_get_task_number() 在 num_irqs>1 时用
             * subcore_task[core].task_number *覆盖* args->task_number ⇒ 只把 task_number
             * 设成 N 而 subcore_task 仍是默认 {0,1} 时，实际只提交 **1** 个 task
            /* §7.45 根因：driver 的 rknpu_get_task_number() 在 num_irqs>1 时用
             * subcore_task[core].task_number *覆盖* args->task_number ⇒ 只把 task_number
             * 设成 N 而 subcore_task 仍是默认 {0,1} 时，实际只提交 **1** 个 task
             *（这正是 §7.42–§7.44「后续 task 从不执行」的原因）。
             * §7.184：vendor conv 链（conv_bn_conv_fp16）真值 = slots {{0,2},{0,2},{0,2},{0,0},{0,0}}
             * + 6 描述符（3 核 × 每核 2-task 链）；这里默认全槽=N 对 AUTO 鲁棒，
             * --subcore-single0 时仅槽 0={0,N}（单核链，其余 0）。 */
            for (int k = 0; k < 5; k++) {
               sub.subcore_task[k].task_start  = 0;
               sub.subcore_task[k].task_number = (uint32_t)g_chain_n;
            }
            if (g_subcore_single0) {
               for (int k = 1; k < 5; k++) sub.subcore_task[k].task_number = 0;
            }
            if (g_multi_core) {
               /* §7.43/§7.44：driver 侧 use_core_num = popcount(core_mask)；
                * ∈{1,2} ⇒ 核 i 取 subcore_task[i]；==3 ⇒ 核 i 取 subcore_task[i+2]（3 核用下标 2..4，
                * 即 vendor 的填法）。每核一个连续 task 区间。 */
               if (g_chain_n >= 3) {
                  sub.core_mask = 0x7;
                  for (int k = 0; k < 3; k++) {
                     sub.subcore_task[k + 2].task_start  = (uint32_t)k;
                     sub.subcore_task[k + 2].task_number = 1;
                  }
                  for (int k = 0; k < 2; k++) {
                     sub.subcore_task[k].task_start  = 0;
                     sub.subcore_task[k].task_number = 0;
                  }
                  printf("MULTICORE: core_mask=0x7 (use_core_num=3) 核0/1/2 → task0/1/2"
                         "（subcore_task[2..4]）\n");
               } else {
                  sub.core_mask = 0x3;
                  sub.subcore_task[0].task_start  = 0;
                  sub.subcore_task[0].task_number = 1;
                  sub.subcore_task[1].task_start  = 1;
                  sub.subcore_task[1].task_number = 1;
                  for (int k = 2; k < 5; k++) {
                     sub.subcore_task[k].task_start  = 0;
                     sub.subcore_task[k].task_number = 0;
                  }
                  printf("MULTICORE: core_mask=0x3 (use_core_num=2) core0→task0 core1→task1\n");
               }
            }
            /* 驱动把 args->task_base_addr 写进 PC_DMA_BASE_ADDR：task0 的 regcmd 地址由
             * PC_DATA_ADDR 直接给，但 task1 描述符需要 PC 自己按基址取 ⇒ 必须给对。 */
            /* 注意：把 task_base_addr 设成 task BO 的 DMA 地址会让 PC-DMA 取指失败并
             * 超时（§7.42 实测），故保持 0（驱动默认）；多 task 语义见台账。 */
            g_chain_armed = 1;
            printf("CHAIN: %d task(s)，每段 regcmd %u entries（连续排布），输出 out@+%u 起、"
                   "步长 %u；task BO dma=%#llx（task_base_addr 保持 0）\n",
                   g_chain_n, nB - nA, g_chain_off, g_chain_stride,
                   (unsigned long long)tk.dma_addr);
         }
      }
      if (g_task_out_step && g_task_blocks && g_task_count > 1 && !g_tile_on) {
         /* §7.170：task k≥1 的输出区预填哨兵（fp16=0x00 / int8=0x80）；提交后逐区对 task0 比对。 */
         unsigned tstep = ALIGN_UP(out_total_bytes, 4096u);
         size_t need = (size_t)OUT_OFF + (size_t)tstep * (size_t)(g_task_count - 1) + out_total_bytes;
         if (need > act.size) {
            printf("TASK-OUT-STEP: 激活 BO 不足（需 %zu > %zu）⇒ 放弃\n", need, act.size);
            return 1;
         }
         for (int k = 1; k < g_task_count; k++)
            memset((uint8_t *)act.map + OUT_OFF + (size_t)tstep * k, g_fp16 ? 0x00 : 0x80,
                   out_total_bytes);
         bo_sync(fd, &act, 1);
         printf("TASK-OUT-STEP: task k → OUT_OFF + %u·k（哨兵 %s）\n",
                tstep, g_fp16 ? "0x00" : "0x80");
      }
      bo_sync(fd, &rcbo, 1);
      bo_sync(fd, &tk, 1);

      for (int w = 0; w < warm; w++) {
         int wret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub);
         printf("WARMUP %d/%d ret=%d %s\n", w + 1, warm, wret,
                wret ? strerror(errno) : "ok");   /* §7.39 诊断：热身提交的逐次结果 */
         bo_sync(fd, &act, 2);
      }

      struct timespec t0, t1;
      clock_gettime(CLOCK_MONOTONIC, &t0);
      ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub);
      clock_gettime(CLOCK_MONOTONIC, &t1);
      double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
      ms_total += ms;
      hw_total += (long long)sub.hw_elapse_time;
      if (g_tile_on)
         printf("SUBMIT tile %d/%d (out rows [%u,%u), in rows [%u,%u)) ret=%d errno=%s  "
                "%.3f ms  hw_elapse=%lld ns\n",
                ti, n_tiles, tp->oy0, tp->oy0 + tp->T, tp->g0, tp->g0 + tp->rows,
                ret, ret ? strerror(errno) : "-", ms, (long long)sub.hw_elapse_time);
      else
         printf("TASK-COUNTER: args->task_counter=%u (args->task_number=%u, subcore_task[0].task_number=%u)\n",
                sub.task_counter, sub.task_number, sub.subcore_task[0].task_number);
         printf("SUBMIT ret=%d errno=%s  %.3f ms  hw_elapse=%lld ns\n",
                ret, ret ? strerror(errno) : "-", ms, (long long)sub.hw_elapse_time);
      if (ret && g_chain_id) {
         /* §7.182：超时路径也做哨兵检查（诊断链各段是否已写输出区）——BO 仍有效 */
         bo_sync(fd, &act, 2);
         for (int k = 0; k < g_chain_n && k < 8; k++) {
            uint32_t off = (k == 0) ? (uint32_t)OUT_OFF
                                    : g_chain_off + (unsigned)(k - 1) * g_chain_stride;
            const uint8_t *om = (const uint8_t *)act.map + off;
            long diff = 0;
            for (unsigned i = 0; i < out_total_bytes; i++)
               if (om[i] != (uint8_t)(g_fp16 ? 0x00 : 0x80)) diff++;
            printf("TIMEOUT-OUT %d @+%u: %s (%ld/%u bytes nonzero-diff)\n",
                   k, (k == 0) ? (unsigned)OUT_OFF : g_chain_off + (unsigned)(k - 1) * g_chain_stride,
                   diff ? "HIT" : "UNTOUCHED", diff, out_total_bytes);
         }
      }
      if (ret) return 1;

      if (g_tile_on) {
         /* 把 tile 输出面 scatter 回目标面（互不相交，无需累加）；§7.34：ic 组时目标是
          * 本组的 partial 区，最后由 host fp32 累加各组的 partial。 */
         bo_sync(fd, &tile, 2);
         uint8_t *gout_g = gout + oc_plane_off
                         + (n_icg > 1 ? (size_t)(gi + 1) * out_total_bytes : 0);
         const uint8_t *tout = (const uint8_t *)tile.map + tile_out_off;
         unsigned t_row = tp->Tc * FEATURE_ATOMIC_SIZE;          /* tile 输出行内字节 */
         for (unsigned gg = 0; gg < tile_groups; gg++)
            for (unsigned r = 0; r < tp->T; r++)
               memcpy(gout_g + (size_t)gg * out_plane_bytes
                            + (size_t)(tp->oy0 + r) * out_row_bytes
                            + (size_t)tp->ox0 * FEATURE_ATOMIC_SIZE,
                      tout + (size_t)gg * tp->T * t_row + (size_t)r * t_row,
                      t_row);
      }
   }
   }   /* ic 组结束 */
   if (n_tiles > 1)
      printf("TILE TOTAL: %d submits  %.3f ms  hw_elapse_total=%lld ns\n",
             n_tiles, ms_total, hw_total);

    bo_sync(fd, &act, 2);

    if (g_chain_settle_ms > 0 && g_chain_armed) {
       /* §7.188 读早/排空判别：在正常读点（已 bo_sync from dev）先探一次，再 sleep +
        * 重新 bo_sync from dev 探第二次。若 after 与 before 不同（且 after==task0）⇒
        * task1 损坏来自 WDMA 排空/读早；若二者一致仍≠task0 ⇒ 计算/CBUF 侧竞态。
        * 纯读 + 延时，不改提交与协议。 */
       for (int phase = 0; phase < 2; phase++) {
          if (phase == 1) {
             usleep((useconds_t)g_chain_settle_ms * 1000);
             bo_sync(fd, &act, 2);
          }
          const uint8_t *oa = (const uint8_t *)act.map + OUT_OFF;
          for (int k = 1; k < g_chain_n; k++) {
             const uint8_t *ob = (const uint8_t *)act.map + g_chain_off +
                                 (unsigned)(k - 1) * g_chain_stride;
             uint32_t hb = 2166136261u;
             unsigned diff = 0, nz = 0;
             for (unsigned j = 0; j < out_total_bytes; j++) {
                hb ^= ob[j]; hb *= 16777619u;
                if (oa[j] != ob[j]) diff++;
                if (ob[j] != 0) nz++;
             }
             printf("CHAIN-SETTLE[%s] task%d hash=%08x nonzero=%u diff_vs_task0=%u %s\n",
                    phase ? "after" : "before", k, hb, nz, diff,
                    diff ? (nz ? "PARTIAL/OTHER" : "UNTOUCHED") : "EQUAL-TASK0");
          }
       }
    }


    if (n_icg > 1) {
       unsigned oc_bytes = (n_ocg > 1) ? (ocpl[oi].oc / oc_atom) * out_plane_bytes : out_total_bytes;
       unsigned elems = oc_bytes / 2;                    /* fp16 元素数 */
       float  *accf = (float *)calloc(elems, sizeof(float));
       double *accd = (double *)calloc(elems, sizeof(double));
       if (!accf || !accd) { fprintf(stderr, "ic-split accum alloc 失败\n"); return 1; }
       for (int g = 0; g < n_icg; g++) {
          const uint8_t *pp = (const uint8_t *)act.map + OUT_OFF + (size_t)(g + 1) * out_total_bytes
                            + oc_plane_off;
          for (unsigned k = 0; k < elems; k++) {
             uint16_t h = (uint16_t)(pp[2 * k] | ((uint16_t)pp[2 * k + 1] << 8));
             float fv = f16_to_f32(h);
             accf[k] += fv;
             accd[k] += (double)fv;
          }
       }
       uint8_t *dst = (uint8_t *)act.map + OUT_OFF + oc_plane_off;
       long f32_vs_f64 = 0; double maxad = 0;
       for (unsigned k = 0; k < elems; k++) {
          uint16_t h32 = f32_to_f16(accf[k]);
          uint16_t h64 = f32_to_f16((float)accd[k]);
          dst[2 * k]     = (uint8_t)(h32 & 0xff);
          dst[2 * k + 1] = (uint8_t)(h32 >> 8);
          long u = f16_ord(h32) - f16_ord(h64);
          if (u < 0) u = -u;
          if (u > f32_vs_f64) f32_vs_f64 = u;
          double d = (double)f16_to_f32(h32) - (double)f16_to_f32(h64);
          if (d < 0) d = -d;
          if (d > maxad) maxad = d;
       }
       printf("IC-SPLIT: %d partial(s), %u elems; host fp32 累加 vs fp64 累加: max ulp=%ld max|diff|=%.4g\n",
              n_icg, elems, f32_vs_f64, maxad);
       free(accf); free(accd);
    }
    }   /* §7.36 oc 组结束 */
    g_ic_offset = 0; g_oc_offset = 0;
    {
       const uint8_t *o = (const uint8_t *)act.map + OUT_OFF;
       uint32_t hh = 2166136261u;
       for (unsigned k = 0; k < out_total_bytes; k++) { hh ^= o[k]; hh *= 16777619u; }
   if (g_out_file_c) {   /* §7.158：提交后把输出逐字节写文件（OUT_OFF 起，out_total_bytes ✓） */
      FILE *fo = fopen(g_out_file_c, "wb");
      if (fo) {
         fwrite((uint8_t *)act.map + OUT_OFF, 1, out_total_bytes, fo);
         fclose(fo);
         printf("OUT-FILE: %s (%u B, OUT_OFF=%d)\n", g_out_file_c, out_total_bytes, (int)OUT_OFF);
      }
   }

       printf("OUT-HASH: %08x  (%u B, ic_groups=%d, oc_groups=%d, tiles=%d)\n",
              hh, out_total_bytes, n_icg, n_ocg, n_tiles);
       if (g_chain_armed) {
          const uint8_t *oa = (const uint8_t *)act.map + OUT_OFF;
          unsigned eq = 0, written = 0;
          for (int k = 1; k < g_chain_n; k++) {      /* 区域 = task1..N-1 */
             const uint8_t *ob = (const uint8_t *)act.map + g_chain_off +
                                 (unsigned)(k - 1) * g_chain_stride;
             uint32_t hb = 2166136261u;
             unsigned diff = 0, nz = 0;
             for (unsigned j = 0; j < out_total_bytes; j++) {
                hb ^= ob[j];
                hb *= 16777619u;
                if (oa[j] != ob[j]) diff++;
                if (ob[j] != 0) nz++;
             }
             if (!diff) eq++;
             if (nz) written++;
             printf("CHAIN-TASK%d: hash=%08x nonzero_bytes=%u diff_vs_task0=%u %s\n",
                    k, hb, nz, diff, diff ? (nz ? "PARTIAL/OTHER" : "UNTOUCHED") : "EQUAL-TASK0");
          }
          printf("CHAIN-SUMMARY: task(s)=%d（task0 + %d 个后续）已写入=%u 与 task0 逐位相同=%u "
                 "RESULT: %s\n", g_chain_n, g_chain_n - 1, written, eq,
                 (eq == (unsigned)(g_chain_n - 1)) ? "ALL-TASKS-EQUAL (后续 task 全部执行且一致)"
                                                   : "INCOMPLETE");
       }
       if (g_task_out_step && g_task_blocks && g_task_count > 1 && !g_tile_on) {
          /* §7.170：逐 task 输出区校验 —— 与 task0 逐字节比对；全哨兵 ⇒ 该 task 未执行。 */
          unsigned tstep = ALIGN_UP(out_total_bytes, 4096u);
          uint8_t sent = g_fp16 ? 0x00 : 0x80;
          const uint8_t *oa = (const uint8_t *)act.map + OUT_OFF;
          unsigned eq = 0, untouched = 0;
          for (int k = 1; k < g_task_count && k < 64; k++) {
             const uint8_t *ob = (const uint8_t *)act.map + OUT_OFF + (size_t)tstep * k;
             uint32_t hb = 2166136261u;
             unsigned diff = 0, nsent = 0;
             for (unsigned j = 0; j < out_total_bytes; j++) {
                hb ^= ob[j]; hb *= 16777619u;
                if (oa[j] != ob[j]) diff++;
                if (ob[j] != sent) nsent++;
             }
             if (!diff) eq++;
             if (!nsent) untouched++;
             printf("TASK-OUT %d: hash=%08x diff_vs_task0=%u %s\n", k, hb, diff,
                    diff ? (nsent ? "PARTIAL/OTHER" : "UNTOUCHED") : "EQUAL-TASK0");
          }
          printf("TASK-OUT-SUMMARY: task(s)=%d 与 task0 一致=%u 未写=%u ⇒ %s\n",
                 g_task_count, eq, untouched,
                 (eq == (unsigned)(g_task_count - 1))
                    ? "ALL-TASKS-EXECUTED (每 task 独立输出且一致)"
                    : "INCOMPLETE (部分 task 未执行/输出错)");
       }
    }

    if (g_dump_bin) {
       FILE *fa = fopen("mc_act_bo.bin", "wb");
       if (fa) { fwrite(act.map, 1, act.size, fa); fclose(fa); }
       FILE *fw = fopen("mc_wt_bo.bin", "wb");
       if (fw) { fwrite(wt.map, 1, wt.size, fw); fclose(fw); }
       printf("DUMP_BIN: act(%u B, out@%u) + wt(%u B) -> mc_act_bo.bin / mc_wt_bo.bin\n",
              act.size, OUT_OFF, wt.size);
       if (g_tile_on) {
          FILE *ft = fopen("mc_tile_bo.bin", "wb");
          if (ft) { fwrite(tile.map, 1, tile.size, ft); fclose(ft); }
          printf("DUMP_BIN: tile(%u B, in@%u, out@%u) -> mc_tile_bo.bin\n",
                 tile.size, tile_in_off, tile_out_off);
       }
    }

    /* 打印输出（group 0，通道 0..min(oc_real,16)-1），NHWC 语义 */
   uint8_t *outnpu = (uint8_t *)act.map + OUT_OFF;
   unsigned shown = MIN2(c.oc_real, FEATURE_ATOMIC_SIZE);
   printf("\n-- output group0 (raw bytes, shown as int8 / uint8) --\n");
   for (unsigned ch = 0; ch < shown; ch++) {
      printf("ch%02u: ", ch);
      for (unsigned y = 0; y < c.oh; y++) {
         printf("[");
         for (unsigned x = 0; x < c.ow; x++) {
            int8_t v = (int8_t)outnpu[out_off(0, x, y, c.ow, c.oh) + ch];
            printf("%5d", v);
            if (x + 1 < c.ow) printf(",");
         }
         printf("]");
      }
      printf("\n");
      if (ch >= 3 && !g_wgrad_dot_mode) { printf("  ... (rest omitted)\n"); break; }
   }

   /* ---- fp16 CPU 参考：直接读回 BO 里的 fp16（激活 + 权重），float 累加 ---- */
   if (g_pattern && g_fp16) {
      const uint8_t *in_base = (const uint8_t *)act.map + in_shift;
      const uint8_t *wbase = (const uint8_t *)wt.map;
      printf("\n-- CPU reference check (fp16, atom=8 fp16) --\n");
      long total = 0, bad = 0, badf = 0, badf_matches = 0;
      long maxulp = 0, maxulpf = 0;
      double maxabs = 0;
      static long p_oc16[256];
      memset(p_oc16, 0, sizeof(p_oc16));
      for (unsigned oc = 0; oc < c.oc_real; oc++)
        for (unsigned oy = 0; oy < c.oh; oy++)
          for (unsigned ox = 0; ox < c.ow; ox++) {
             double acc = 0.0;
             float accf = 0.0f;
             if (c.deconv && c.deconv_stride) {
                /* 转置卷积 gather（§7.27 修正口径，与 int8 参考同一口径）：
                 * ox = ix*S + pad - kx  ⇒  tx = ox - pad + kx，须能被 S 整除；
                 * 权重 tap 直接用 kx（无语义翻转）。旧口径 tx = ox + pad - kx 仅
                 * 在 2·pad = kw-1（如 3x3 pad1）时与硬件巧合重合，其余核全 MISMATCH。 */
                int S = (int)c.deconv_stride + 1;
                unsigned ae = fp16_atom_elems(&c);
                for (unsigned ky = 0; ky < c.kh; ky++)
                 for (unsigned kx = 0; kx < c.kw; kx++) {
                    int ty = (int)oy - (int)c.pad_top + (int)ky;
                    int tx = (int)ox - (int)c.pad_left + (int)kx;
                    if (ty % S || tx % S) continue;
                    int iy = ty / S, ix = tx / S;
                    if (ix < 0 || iy < 0 || ix >= (int)c.iw || iy >= (int)c.ih) continue;
                    unsigned wx = kx;
                    unsigned wy = ky;
                    for (unsigned icc = 0; icc < c.ic_real; icc++) {
                       const uint8_t *ip = in_base + in_off(&c, icc / ae, ix, iy) +
                                           (icc % ae) * 2;
                       const uint8_t *wp = wbase +
                          w_elem_index_f16(&c, oc, icc, wx, wy, oc_w_eff) * 2;
                       acc  += (double)rd_f16(ip) * (double)rd_f16(wp);
                       accf += rd_f16(ip) * rd_f16(wp);
                    }
                 }
             } else {
             for (unsigned ky = 0; ky < c.kh; ky++)
              for (unsigned kx = 0; kx < c.kw; kx++) {
                 int ix = (int)ox * (int)c.stride_x + (int)kx - (int)c.pad_left;
                 int iy = (int)oy * (int)c.stride_y + (int)ky - (int)c.pad_top;
                 if (ix < 0 || iy < 0 || ix >= (int)c.iw || iy >= (int)c.ih) continue;
                 for (unsigned icc = 0; icc < c.ic_real; icc++) {
                    unsigned ae = fp16_atom_elems(&c);
                    const uint8_t *ip = in_base + in_off(&c, icc / ae, ix, iy) +
                                        (icc % ae) * 2;
                    const uint8_t *wp = wbase +
                       w_elem_index_f16(&c, oc, icc, kx, ky, oc_w_eff) * 2;
                    acc  += (double)rd_f16(ip) * (double)rd_f16(wp);
                    accf += rd_f16(ip) * rd_f16(wp);
                 }
              }
             }
             /* §7.19/§7.24：fp16 BS/BRDMA —— ALU addend 表/fp32（槽=0x40a00000
              * 时 hw=conv+5.0），MUL factor=fp16@[31:16]/表内 fp16；先加后乘。 */
             if (g_bs_cfg_override >= 0) {
                float af = apply_bs_fp16(accf, oc);
                if (af == af) { accf = af; acc = (double)af; }
             }
             /* FP16 BN/NRDMA uses a per-channel fp32 bias table (vendor bn_only trace). */
             if (g_fp16 && g_bn_cfg_override >= 0 &&
                 !(((uint32_t)g_bn_cfg_override) & 0x01u) &&
                 (((uint32_t)g_bn_cfg_override) & 0x100u) &&
                 g_bn_nrdma_override >= 0 && oc < g_bn_alu_dma_count &&
                 !( ((uint32_t)g_bn_cfg_override) & 0x02u ) &&
                 (((uint32_t)g_bn_cfg_override >> 16) & 0xfu) == 2u) {
                float bf;
                memcpy(&bf, &g_bn_alu_dma[oc], sizeof(bf));
                acc += (double)bf; accf += bf;
             }
             uint16_t rh = f32_to_f16((float)acc);
             uint16_t rf = f32_to_f16(accf);
             unsigned og = oc / FP16_ATOM_ELEMS, occ = oc % FP16_ATOM_ELEMS;
             unsigned off = out_off(og, ox, oy, c.ow, c.oh) + occ * 2;
             uint16_t hh = (uint16_t)((uint16_t)outnpu[off] |
                                      ((uint16_t)outnpu[off + 1] << 8));
             total++;
             if (g_wgrad_dot_mode)
                printf("  WGRAD_F16 oc=%u dot=%g hw=%g\n", oc,
                       (double)accf, (double)f16_to_f32(hh));
             if (oc < 4 && ox < 4 && oy < 4)
                printf("  P[oc=%u x=%u y=%u] hw=%g ref=%g\n", oc, ox, oy,
                       (double)f16_to_f32(hh), (double)f16_to_f32(rh));
             if (hh == rf && rf != rh) badf_matches++;
             if (hh != rh) {
                bad++; if (oc < 256) p_oc16[oc]++;
                if (bad <= 24)
                   printf("  mismatch oc=%u (x=%u,y=%u): ref=%.9g(0x%04x) "
                          "hw=%.9g(0x%04x)\n", oc, ox, oy,
                          (double)f16_to_f32(rh), rh, (double)f16_to_f32(hh), hh);
             }
             double d = (double)f16_to_f32(hh) - (double)f16_to_f32(rh);
             if (d < 0) d = -d;
             if (d > maxabs) maxabs = d;
             if (hh != rf) badf++;
             long u = f16_ord(hh) - f16_ord(rh);
             if (u < 0) u = -u;
             if (u > maxulp) maxulp = u;
             long uf = f16_ord(hh) - f16_ord(rf);
             if (uf < 0) uf = -uf;
             if (uf > maxulpf) maxulpf = uf;
          }
      printf("  total=%ld  fp16-mismatch=%ld  max|diff|=%.6g  max ulp=%ld\n",
             total, bad, maxabs, maxulp);
      printf("  vs float32-accumulate reference: mismatch=%ld  max ulp=%ld  "
             "(hw==fp32ref but !=fp64ref: %ld)\n", badf, maxulpf, badf_matches);
      if (bad) {
         printf("  mismatches per oc:");
         for (unsigned k = 0; k < c.oc_real && k < 256; k++)
            if (p_oc16[k]) printf(" oc%u=%ld", k, p_oc16[k]);
         printf("\n");
      }
      printf("  CFG: ic_real=%u ic=%u oc_real=%u oc=%u w_kernels=%u w_oc2=%u\n",
             c.ic_real, c.ic, c.oc_real, c.oc, wskernels, oc_w_eff);
      /* §7.30 修正 §7.29：硬件包线并非「多 tap/1x1 区分 c.ic 上限」，而是 CNA
       * 输入 feature tile 字节数上限 = 32768 B（即 in_size）。控制变量实测：
       * 同一 iw/ih/ic 下 1x1 与 3x3 同为 OK/BAD，仅改变 ih（或 iw）即可跨越
       * 边界（ic128 iw16: ih8=32768 OK, ih9=36864 BAD；iw 亦然）。fp16 与 int8
       * 同受此限（int8 iw16 ih16 ic128=32768 OK，ic160=40960 BAD）。超限硬件
       * 静默算错（max|diff| 可达数千）。修复：按输出行（或 ic 组）切分，使单次
       * submit 的 in_size ≤ 32768。 */
      int ooe_local = ooe;
      if (ooe_local)
         printf("  WARN: 单次 submit 的输入 feature tile %u B > 32768 B 超实测包线（§7.30），硬件静默算错\n",
                g_tile_max_in);
      /* 判据：fp64 参考逐位一致 = BIT-EXACT；否则 fp32 累加参考一致；否则
       * 与 fp64 参考的位型差 ≤16 ULP 视为多元素求和顺序差异（fp16 输出粒度）。 */
      printf("  RESULT: %s\n", ooe_local ? "OUT-OF-ENVELOPE (CBUF banks: input+weight > 12, §7.31)"
             : (bad == 0 ? "BIT-EXACT (fp16)"
             : (badf == 0 ? "BIT-EXACT (fp32-acc)"
             : (maxulp <= 16 ? "ULP-EXACT (<=16 ulp)" : "MISMATCH"))));
   }

   /* ---- CPU 参考对照 ---- */
   if (g_pattern && !g_fp16) {   /* §7.192：需 --pattern（权重=pat_w）；若同时 --in-file，参考读 BO 真实输入 */
      g_ref_in_base = g_in_file ? ((const uint8_t *)act.map + in_shift) : NULL;
      printf("\n-- CPU reference check (scale=%u shift=%u offset=%d, input=%s) --\n",
             g_out_scale, g_out_shift, g_out_offset, g_ref_in_base ? "BO/--in-file" : "pattern");
      long total = 0, bad_floor = 0, bad_round = 0, maxd = 0;
      static long per_oc[256];
      static long per_oy[512], per_ox[512];
      memset(per_oc, 0, sizeof(per_oc));
      memset(per_oy, 0, sizeof(per_oy));
      memset(per_ox, 0, sizeof(per_ox));
      for (unsigned oc = 0; oc < c.oc_real; oc++)
        for (unsigned oy = 0; oy < c.oh; oy++)
          for (unsigned ox = 0; ox < c.ow; ox++) {
             int32_t acc = 0;
             if (g_wgrad_dot_mode) {
                for (unsigned p = 0; p < c.ic_real; p++)
                   acc += (int)wgrad_x(p) * (int)wgrad_dy(p, oc);
             } else if (c.deconv && c.deconv_stride) {
                /* 转置卷积 gather（§7.27 修正口径）：ox = ix*S + pad - kx
                 * ⇒ tx = ox - pad + kx（须被 S 整除），权重 tap 直接用 kx。 */
                int S = (int)c.deconv_stride + 1;
                for (unsigned ky = 0; ky < c.kh; ky++)
                  for (unsigned kx = 0; kx < c.kw; kx++) {
                     int ty = (int)oy - (int)c.pad_top + (int)ky;
                     int tx = (int)ox - (int)c.pad_left + (int)kx;
                     if (ty % S || tx % S) continue;
                     int iy = ty / S, ix = tx / S;
                     if (ix < 0 || iy < 0 || ix >= (int)c.iw || iy >= (int)c.ih) continue;
                     unsigned wx = kx;
                     unsigned wy = ky;
                     for (unsigned icc = 0; icc < c.ic_real; icc++)
                        acc += (int)ref_in_val(ix, iy, icc, c.iw, c.ih) * (int)pat_w(wx, wy, oc, icc);
                  }
             } else {
                for (unsigned ky = 0; ky < c.kh; ky++)
                  for (unsigned kx = 0; kx < c.kw; kx++) {
                     int ix = (int)ox * (int)c.stride_x + (int)kx - (int)c.pad_left;
                     int iy = (int)oy * (int)c.stride_y + (int)ky - (int)c.pad_top;
                     if (ix < 0 || iy < 0 || ix >= (int)c.iw || iy >= (int)c.ih) continue;
                     for (unsigned icc = 0; icc < c.ic_real; icc++)
                        acc += (int)ref_in_val(ix, iy, icc, c.iw, c.ih) * (int)pat_w(kx, ky, oc, icc);
                  }
             }
             acc = apply_bs_brdma(acc, oc);
             acc = apply_bn_immediate(acc, oc);
             int64_t vf = (((int64_t)acc * g_out_scale) >> g_out_shift) + g_out_offset;
             if (vf < -128) vf = -128; if (vf > 127) vf = 127;
             int64_t vr = shift_right_round64((int64_t)acc * g_out_scale, g_out_shift) + g_out_offset;
             if (vr < -128) vr = -128; if (vr > 127) vr = 127;
             /* 输出是 NC1HWC2，通道原子 16：oc>=16 落在下一个 group */
             unsigned og = oc / FEATURE_ATOMIC_SIZE;
             unsigned occ = oc % FEATURE_ATOMIC_SIZE;
             int hw = (int)(int8_t)outnpu[out_off(og, ox, oy, c.ow, c.oh) + occ];
             total++;
             if (g_wgrad_dot_mode)
                printf("  WGRAD_VECTOR oc=%u dot=%d npu=%d\n", oc, acc, hw);
             if (hw != (int)vf) { bad_floor++; if (oc < 256) per_oc[oc]++;
                if (oy < 512) per_oy[oy]++; if (ox < 512) per_ox[ox]++; }
             if (hw != (int)vr) bad_round++;
             long d = labs((long)hw - (long)vf);
             if (d > maxd) maxd = d;
              if (hw != (int)vf && bad_floor <= 500)
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
         printf("  mismatches per oy:");
         for (unsigned k = 0; k < c.oh && k < 512; k++)
            if (per_oy[k]) printf(" oy%u=%ld", k, per_oy[k]);
         printf("\n");
         printf("  mismatches per ox:");
         for (unsigned k = 0; k < c.ow && k < 512; k++)
            if (per_ox[k]) printf(" ox%u=%ld", k, per_ox[k]);
         printf("\n");
      }
      printf("  CFG: ic_real=%u ic=%u oc_real=%u oc=%u w_kernels=%u w_oc2=%u layout=%c "
             "pad_con1=%d\n",
             c.ic_real, c.ic, c.oc_real, c.oc, wskernels,
             w_oc2 > 0 ? (unsigned)w_oc2 : ALIGN_UP(c.oc_real, 2), "ACD"[g_w_layout],
             g_effective_pad_con1);
      printf("  AUX: hi_zero=%d icreal_reg=%d ic_pad=%u\n", g_hi_zero, g_icreal_reg, c.ic);
      printf("  DPU_BN: cfg=%s alu=0x%08x mul=0x%08x relux=0x%08x\n",
             g_bn_cfg_override >= 0 ? "override" : "bypass",
             g_bn_alu_override, g_bn_mul_override, g_bn_relux_override);
      printf("  DPU_DATA_FORMAT=0x%08x\n", g_dpu_data_format);
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
