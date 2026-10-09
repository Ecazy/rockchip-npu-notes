/*
 * min_pool.c — 手写 PPU MaxPool 寄存器程序（不依赖任何 NPU 运行时库）
 *
 * 目的：验证「构建一个算子」的完整过程。做法是两步——
 *   1. 用 LD_PRELOAD hook 抓到厂商 RKNN 对 maxpool 的 regcmd（标准答案，30 条）；
 *   2. 照抄进自己的发射器，换成自己的 BO 地址，与 CPU maxpool 逐元素对照。
 *
 * 与 min_conv.c 的关系：BO 管理、内核 UAPI、NC1HWC2 布局沿用同一套；
 * 区别是这个 task **完全不用 CNA / DPU / CORE**，只写 PPU(0x4001) 与
 * PPU_RDMA(0x8001) 两个 block。
 *
 * 编译： gcc -O2 -o min_pool min_pool.c -lm
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
 * 1. 内核 UAPI（与 min_conv.c 相同）
 * ------------------------------------------------------------------ */
struct rknpu_mem_create {
   uint32_t handle; uint32_t flags; uint64_t size; uint64_t obj_addr;
   uint64_t dma_addr; uint64_t sram_size; int32_t iommu_domain_id; uint32_t core_mask;
};
struct rknpu_mem_map     { uint32_t handle; uint32_t reserved; uint64_t offset; };
struct rknpu_mem_destroy { uint32_t handle; uint32_t reserved; uint64_t obj_addr; };
struct rknpu_mem_sync    { uint32_t flags; uint32_t reserved; uint64_t obj_addr;
                           uint64_t offset; uint64_t size; };
struct rknpu_task {
   uint32_t flags, op_idx, enable_mask, int_mask, int_clear, int_status;
   uint32_t regcfg_amount, regcfg_offset;
   uint64_t regcmd_addr;
} __attribute__((packed));
struct rknpu_subcore_task { uint32_t task_start, task_number; };
struct rknpu_submit {
   uint32_t flags, timeout, task_start, task_number, task_counter;
   int32_t  priority;
   uint64_t task_obj_addr;
   uint32_t iommu_domain_id, reserved;
   uint64_t task_base_addr;
   int64_t  hw_elapse_time;
   uint32_t core_mask;
   int32_t  fence_fd;
   struct rknpu_subcore_task subcore_task[5];
};

#define DRM_IOCTL_BASE    'd'
#define DRM_COMMAND_BASE  0x40
#define DRM_IOWR(nr, type) _IOWR(DRM_IOCTL_BASE, nr, type)
#define RKNPU_MEM_NON_CONTIGUOUS (1 << 0)
#define RKNPU_MEM_CACHEABLE      (1 << 1)
#define RKNPU_MEM_KERNEL_MAPPING (1 << 3)
#define RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT (1 << 10)
#define RKNPU_JOB_PC       (1 << 0)
#define RKNPU_JOB_BLOCK    0
#define RKNPU_JOB_PINGPONG (1 << 2)

#define DRM_IOCTL_RKNPU_SUBMIT      DRM_IOWR(DRM_COMMAND_BASE + 0x01, struct rknpu_submit)
#define DRM_IOCTL_RKNPU_MEM_CREATE  DRM_IOWR(DRM_COMMAND_BASE + 0x02, struct rknpu_mem_create)
#define DRM_IOCTL_RKNPU_MEM_MAP     DRM_IOWR(DRM_COMMAND_BASE + 0x03, struct rknpu_mem_map)
#define DRM_IOCTL_RKNPU_MEM_DESTROY DRM_IOWR(DRM_COMMAND_BASE + 0x04, struct rknpu_mem_destroy)
#define DRM_IOCTL_RKNPU_MEM_SYNC    DRM_IOWR(DRM_COMMAND_BASE + 0x05, struct rknpu_mem_sync)

#define RKNPU_PC_DATA_EXTRA_AMOUNT 4

/* ------------------------------------------------------------------ */
#define FEATURE_ATOMIC_SIZE 16
#define IC_PAD              16          /* PPU 的通道按 16 对齐 */

#define ALIGN_UP(x, a) (((x) + (a) - 1) & ~((a) - 1))
#define MIN2(a, b)     ((a) < (b) ? (a) : (b))

/* ------------------------------------------------------------------ *
 * 2. regcmd 发射
 * ------------------------------------------------------------------ */
static uint64_t rc_buf[4096];
static uint64_t *rc = rc_buf;
static int g_dump = 0;
static int g_miss_detail = 0;   /* §7.63：打印失配位置窗口 */
static int g_miss_hist = 0;     /* §7.64：按 ox / oy 统计失配数 */
static int miss_shown = 0;
static int g_ppu_op_en = 0;
static int g_recip_w = -1, g_recip_h = -1;   /* 覆盖 RECIP_KERNEL_* */
static int g_surf_len = -1;                   /* 覆盖 MISC_CTRL.SURF_LEN */
static int g_mc_surf_out = 0;

static void emit_raw(uint32_t op, uint32_t reg, uint32_t value)
{
   *rc++ = ((uint64_t)op << 48) | ((uint64_t)value << 16) | (uint64_t)reg;
}

/* vendor 的 PPU / PPU_RDMA block 编码 */
#define OP_PPU       0x4001
#define OP_PPU_RDMA  0x8001
#define OP_NONE      0x0000
#define OP_PC        0x0101
#define OP_SYNC      0x0041
#define OP_OP_EN     0x0081

static void EMIT_PPU(uint32_t reg, uint32_t val)
{
   if (g_dump) fprintf(stderr, "  PPU      reg=0x%04x val=0x%08x\n", reg, val);
   emit_raw(OP_PPU, reg, val);
}
static void EMIT_RDMA(uint32_t reg, uint32_t val)
{
   if (g_dump) fprintf(stderr, "  PPU_RDMA reg=0x%04x val=0x%08x\n", reg, val);
   emit_raw(OP_PPU_RDMA, reg, val);
}

struct pool_cfg {
   unsigned iw, ih, ic;          /* 输入 W/H/补齐后通道 */
   unsigned ow, oh;              /* 输出 W/H */
   unsigned kw, kh, sx, sy;
   unsigned pad_left, pad_top, pad_right, pad_bottom;
   int      method;              /* PPU_OPERATION_MODE_CFG.POOLING_METHOD */
   unsigned precision;           /* PPU_DATA_FORMAT.PROC_PRECISION */
   int      transpose;           /* 1 = PPU 的 W 取 ih（内层维） */
   uint32_t in_addr, out_addr;
};

static unsigned ppu_in_w(const struct pool_cfg *c) { return c->transpose ? c->ih : c->iw; }
static unsigned ppu_in_h(const struct pool_cfg *c) { return c->transpose ? c->iw : c->ih; }
static unsigned ppu_out_w(const struct pool_cfg *c){ return c->transpose ? c->oh : c->ow; }
static unsigned ppu_out_h(const struct pool_cfg *c){ return c->transpose ? c->ow : c->oh; }

/* 完全按厂商 maxpool 捕获的 26 条顺序组装 */
static unsigned emit_pool(const struct pool_cfg *c)
{
   rc = rc_buf;   /* §7.71 根因修复：每次 emit 必须复位游标（旧版**追加**写入 ⇒
                   * 每次都把"第一段"拷进 rcbo，导致 re-emit / 每个 tile 提交的都是旧流）。 */
   unsigned in_w = ppu_in_w(c), in_h = ppu_in_h(c);
   unsigned out_w = ppu_out_w(c), out_h = ppu_out_h(c);
   unsigned kw = c->transpose ? c->kh : c->kw;
   unsigned kh = c->transpose ? c->kw : c->kh;
   unsigned sx = c->transpose ? c->sy : c->sx;
   unsigned sy = c->transpose ? c->sx : c->sy;
   unsigned kx0 = c->transpose ? c->pad_top  : c->pad_left;
   unsigned kx1 = c->transpose ? c->pad_bottom : c->pad_right;
   unsigned ky0 = c->transpose ? c->pad_left : c->pad_top;
   unsigned ky1 = c->transpose ? c->pad_right : c->pad_bottom;

   /* S_POINTER：POINTER_PP_MODE | EXECUTER_PP_EN | POINTER_PP_EN = 0x0e */
   EMIT_PPU(REG_PPU_S_POINTER, 0x0e);
   EMIT_RDMA(REG_PPU_RDMA_RDMA_S_POINTER, 0x0e);

   EMIT_PPU(REG_PPU_DATA_CUBE_IN_WIDTH,   in_w - 1);
   EMIT_PPU(REG_PPU_DATA_CUBE_IN_HEIGHT,  in_h - 1);
   EMIT_PPU(REG_PPU_DATA_CUBE_IN_CHANNEL, c->ic - 1);
   EMIT_PPU(REG_PPU_DATA_CUBE_OUT_WIDTH,  out_w - 1);
   EMIT_PPU(REG_PPU_DATA_CUBE_OUT_HEIGHT, out_h - 1);
   EMIT_PPU(REG_PPU_DATA_CUBE_OUT_CHANNEL, c->ic - 1);

   /* FLYING_MODE(1) | POOLING_METHOD */
   EMIT_PPU(REG_PPU_OPERATION_MODE_CFG,
            PPU_OPERATION_MODE_CFG_FLYING_MODE(1) |
            PPU_OPERATION_MODE_CFG_POOLING_METHOD(c->method));

   EMIT_PPU(REG_PPU_POOLING_KERNEL_CFG,
            PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_HEIGHT((sy - 1) & 0xf) |
            PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_WIDTH((sx - 1) & 0xf) |
            PPU_POOLING_KERNEL_CFG_KERNEL_HEIGHT((kh - 1) & 0xf) |
            PPU_POOLING_KERNEL_CFG_KERNEL_WIDTH((kw - 1) & 0xf));
   EMIT_PPU(REG_PPU_RECIP_KERNEL_WIDTH,  g_recip_w >= 0 ? (unsigned)g_recip_w : 0);
   EMIT_PPU(REG_PPU_RECIP_KERNEL_HEIGHT, g_recip_h >= 0 ? (unsigned)g_recip_h : 0);
   EMIT_PPU(REG_PPU_POOLING_PADDING_CFG,
            PPU_POOLING_PADDING_CFG_PAD_BOTTOM(ky1 & 0x7) |
            PPU_POOLING_PADDING_CFG_PAD_RIGHT(kx1 & 0x7) |
            PPU_POOLING_PADDING_CFG_PAD_TOP(ky0 & 0x7) |
            PPU_POOLING_PADDING_CFG_PAD_LEFT(kx0 & 0x7));
   EMIT_PPU(REG_PPU_PADDING_VALUE_1_CFG, 0);
   EMIT_PPU(REG_PPU_PADDING_VALUE_2_CFG, 0);
   EMIT_PPU(REG_PPU_DST_BASE_ADDR, c->out_addr);
   EMIT_PPU(REG_PPU_DST_SURF_STRIDE, out_w * out_h * FEATURE_ATOMIC_SIZE);
   /* 厂商此处 = 输出面字节数；INDEX_EN=0，语义待定，先照抄 */
   EMIT_PPU(REG_PPU_DATA_FORMAT,
            PPU_DATA_FORMAT_INDEX_ADD(out_w * out_h) |
            PPU_DATA_FORMAT_PROC_PRECISION(c->precision));
   EMIT_PPU(REG_PPU_MISC_CTRL,
            (g_surf_len >= 0 ? PPU_MISC_CTRL_SURF_LEN((unsigned)g_surf_len) : 0) |
            (g_mc_surf_out ? PPU_MISC_CTRL_MC_SURF_OUT(1) : 0) |
            PPU_MISC_CTRL_BURST_LEN(3));
   if (g_ppu_op_en) {
      EMIT_PPU(REG_PPU_OPERATION_ENABLE, 1);
      EMIT_RDMA(REG_PPU_RDMA_RDMA_OPERATION_ENABLE, 1);
   }

   EMIT_RDMA(REG_PPU_RDMA_RDMA_CUBE_IN_WIDTH,   in_w - 1);
   EMIT_RDMA(REG_PPU_RDMA_RDMA_CUBE_IN_HEIGHT,  in_h - 1);
   EMIT_RDMA(REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL, c->ic - 1);
   EMIT_RDMA(REG_PPU_RDMA_RDMA_SRC_BASE_ADDR, c->in_addr);
   EMIT_RDMA(REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE, in_w * FEATURE_ATOMIC_SIZE);
   EMIT_RDMA(REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE,
             in_w * in_h * FEATURE_ATOMIC_SIZE);
   EMIT_RDMA(REG_PPU_RDMA_RDMA_DATA_FORMAT,
             PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION(1));

   /* 收尾 4 条：与厂商捕获逐条一致 */
   emit_raw(OP_NONE, 0x0000, 0);
   emit_raw(OP_PC, REG_PC_REGISTER_AMOUNTS, 0);
   *rc++ = (uint64_t)OP_SYNC << 48;
   emit_raw(OP_OP_EN, REG_PC_OPERATION_ENABLE,
            PC_OPERATION_ENABLE_RESERVED_0(0x30));   /* 厂商值 0x60，OP_EN=0 */

   return (unsigned)(rc - rc_buf);
}

/* ------------------------------------------------------------------ *
 * 3. BO
 * ------------------------------------------------------------------ */
struct bo { uint32_t handle; uint64_t obj_addr; uint64_t dma_addr; void *map; uint32_t size; };

static int bo_create(int fd, uint32_t size, uint32_t extra, struct bo *b)
{
   memset(b, 0, sizeof(*b));
   b->size = size;
   struct rknpu_mem_create mc = {
      .size = size,
      .flags = RKNPU_MEM_NON_CONTIGUOUS | RKNPU_MEM_CACHEABLE |
               RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT | extra,
      .iommu_domain_id = 0,
   };
   if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mc)) {
      fprintf(stderr, "MEM_CREATE(%u) failed: %s\n", size, strerror(errno));
      return -1;
   }
   b->handle = mc.handle; b->obj_addr = mc.obj_addr; b->dma_addr = mc.dma_addr;
   struct rknpu_mem_map mm = { .handle = mc.handle };
   if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mm)) {
      fprintf(stderr, "MEM_MAP failed: %s\n", strerror(errno)); return -1;
   }
   b->map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mm.offset);
   if (b->map == MAP_FAILED) { fprintf(stderr, "mmap: %s\n", strerror(errno)); return -1; }
   memset(b->map, 0, size);
   return 0;
}

static void bo_sync(int fd, struct bo *b, uint32_t dir)
{
   struct rknpu_mem_sync ms = { .flags = dir, .obj_addr = b->obj_addr,
                                .offset = 0, .size = b->size };
   ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, &ms);
}

/* ------------------------------------------------------------------ *
 * 4. 数据：NC1HWC2，offset = x*ih*16 + y*16 + c
 * ------------------------------------------------------------------ */
/* NC1HWC2：group 维在最外层，group 内 offset = x*ih*16 + y*16 + c16 */
static unsigned g_iw, g_ih;
static uint32_t lay(unsigned x, unsigned y, unsigned c, unsigned ih)
{
   unsigned g = c / FEATURE_ATOMIC_SIZE;
   unsigned cc = c % FEATURE_ATOMIC_SIZE;
   return g * g_iw * g_ih * FEATURE_ATOMIC_SIZE
         + x * ih * FEATURE_ATOMIC_SIZE + y * FEATURE_ATOMIC_SIZE + cc;
}

/* 输出读取：组间距 = ow*oh*16（DST_SURF_STRIDE 口径），与输入面无关 */
static int g_stage_in = 0;
static int g_stage_out = 0;
static int g_no_clear_out = 0;
static int g_dump_submit = 0;
static const char *g_out_file = NULL;   /* §7.155：把池化输出（逐字节）写文件，供下游算子复用 */
static const char *g_in_file = NULL;   /* §7.192：从文件读输入（与 min_conv_exp --out-file 管线对接） */
static int g_in_yiw = 0;               /* §7.192：文件为 y 主序（卷积输出布局），需转置到池化输入 x 主序 */
static unsigned g_in_off = 0;     /* §7.79：输入放在 in BO 内非零偏移（单变量） */
static unsigned g_out_off = 0;
static int g_two_submit = 0;
static int g_tile_warm = 0;       /* §7.86：每个 tile 前先做 N 次热身提交（结果丢弃） */
static int g_two_same = 0;        /* §7.68：两次提交都用**同一个新地址**（判定"首次不生效"） */      /* §7.68：同进程两次提交、不同 out_addr ⇒ 判定地址是否锁存 */    /* §7.67：输出放在 out BO 内偏移处（单变量：基址 vs 偏移） */
static unsigned g_out_used_off = 0;       /* §7.66：把输出放进独立 BO（单变量拆分：只换输出地址） */
static struct bo g_out2;        /* §7.66：把输入放进独立 BO（单变量拆分：只换输入地址） */
static unsigned g_tile_oh = 0;   /* §7.65：PPU 按输出行切分（突破 ih≤63 的 6-bit 高度限制） */
static unsigned g_ow, g_oh;
static unsigned out_lay(unsigned ox, unsigned oy, unsigned ch)
{
   unsigned g = ch / FEATURE_ATOMIC_SIZE;
   unsigned cc = ch % FEATURE_ATOMIC_SIZE;
   return g * g_ow * g_oh * FEATURE_ATOMIC_SIZE
        + ox * g_oh * FEATURE_ATOMIC_SIZE + oy * FEATURE_ATOMIC_SIZE + cc;
}

/* §7.65：tile 局部布局（与 lay/out_lay 同构，但用 tile 自己的 ih_t/iw_t） */
static uint32_t lay_t(unsigned x, unsigned y, unsigned c, unsigned ih_t, unsigned iw_t)
{
   unsigned g = c / FEATURE_ATOMIC_SIZE;
   unsigned cc = c % FEATURE_ATOMIC_SIZE;
   return g * iw_t * ih_t * FEATURE_ATOMIC_SIZE + x * ih_t * FEATURE_ATOMIC_SIZE
        + y * FEATURE_ATOMIC_SIZE + cc;
}

static int8_t pat_in(unsigned x, unsigned y, unsigned c)
{
   /* 覆盖整个 int8 值域，便于暴露符号/截断问题 */
   return (int8_t)(((int)(x * 37 + y * 53 + c * 29) % 256) - 128);
}

/* §7.68：扫描一块 BO，返回「与参考输出一致」的最佳偏移（命中粗采样点数） */
static long scan_hits(const uint8_t *buf, size_t sz, const struct pool_cfg *c,
                      const uint8_t *inp, unsigned *best_off_out)
{
   unsigned best_off = 0; long best_hit = -1;
   unsigned out_need = c->ow * c->oh * c->ic;
   for (unsigned off = 0; off + out_need <= sz; off += 16) {
      long hit = 0;
      for (unsigned ch = 0; ch < c->ic && hit >= best_hit / 2; ch += 8)
         for (unsigned oy = 0; oy < c->oh; oy += 4)
            for (unsigned ox = 0; ox < c->ow; ox += 4) {
               int8_t best = -128;
               for (unsigned ky = 0; ky < c->kh; ky++)
                  for (unsigned kx = 0; kx < c->kw; kx++) {
                     int ix = (int)(ox * c->sx + kx) - (int)c->pad_left;
                     int iy = (int)(oy * c->sy + ky) - (int)c->pad_top;
                     if (ix < 0 || iy < 0 || ix >= (int)c->iw || iy >= (int)c->ih) continue;
                     int8_t v = (int8_t)inp[lay(ix, iy, ch, c->ih)];
                     if (v > best) best = v;
                  }
               if ((int8_t)buf[off + out_lay(ox, oy, ch)] == best) hit++;
            }
      if (hit > best_hit) { best_hit = hit; best_off = off; }
   }
   if (best_off_out) *best_off_out = best_off;
   return best_hit;
}

/* §7.71：打印一次提交的完整信息（对照两条流程的差异） */
static void dump_submit(const char *tag, const struct rknpu_submit *sub,
                        const struct rknpu_task *t, const uint64_t *rc, unsigned n)
{
   printf("  [%s] flags=0x%x timeout=%u task_start=%u task_number=%u task_obj=0x%llx "
          "core_mask=%u fence=%d task_base=0x%x\n", tag, sub->flags, sub->timeout,
          sub->task_start, sub->task_number, (unsigned long long)sub->task_obj_addr,
          sub->core_mask, sub->fence_fd, sub->task_base_addr);
   printf("  [%s] subcore=", tag);
   for (int i = 0; i < 5; i++) printf("{%u,%u}", sub->subcore_task[i].task_start, sub->subcore_task[i].task_number);
   printf("\n  [%s] task: op_idx=%u en=0x%x int_mask=0x%x int_clear=0x%x amount=%u addr=0x%x\n", tag,
          t->op_idx, t->enable_mask, t->int_mask, t->int_clear, t->regcfg_amount, t->regcmd_addr);
   printf("  [%s] rc[0..2]=", tag);
   for (unsigned i = 0; i < 3 && i < n; i++) printf("(%04x:%08x)", (unsigned)(rc[i] & 0xffff), (unsigned)(rc[i] >> 32));
   printf(" rc[%u..]=", n > 3 ? n - 3 : 0);
   for (unsigned i = (n > 3 ? n - 3 : 0); i < n; i++) printf("(%04x:%08x)", (unsigned)(rc[i] & 0xffff), (unsigned)(rc[i] >> 32));
   printf("\n");
   for (unsigned i = 0; i < n; i++)
      if ((unsigned)(rc[i] & 0xffff) == 0x6070)
         printf("  [%s] 流内 PPU_DST_BASE_ADDR(0x6070) = 0x%08x\n", tag, (unsigned)(rc[i] >> 32));
}

int main(int argc, char **argv)
{
   struct pool_cfg c = {
      .iw = 64, .ih = 64, .ic = IC_PAD,
      .ow = 32, .oh = 32,
      .kw = 2, .kh = 2, .sx = 2, .sy = 2,
      .pad_left = 0, .pad_top = 0, .pad_right = 0, .pad_bottom = 0,
      .method = 1, .precision = 0, .transpose = 1,
   };
   const char *dev = "/dev/dri/card1";
   uint32_t enable_mask = 0xf;
   uint32_t int_mask = 0xc00;        /* PPU 的完成信号是 bit10，fuzz 后为 0xc00 */
   uint32_t int_clear = 0x1ffff;
   uint32_t timeout_ms = 500;
   int ow_given = 0, oh_given = 0, ic_given = 0;
   int ppu_op_en = 0;

   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--dump")) g_dump = 1;
      else if (!strcmp(argv[i], "--miss-detail")) g_miss_detail = 1;
      else if (!strcmp(argv[i], "--tile-oh") && i + 1 < argc) g_tile_oh = (unsigned)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--stage-in")) g_stage_in = 1;
      else if (!strcmp(argv[i], "--stage-out")) g_stage_out = 1;
      else if (!strcmp(argv[i], "--no-clear-out")) g_no_clear_out = 1;
      else if (!strcmp(argv[i], "--dump-submit")) g_dump_submit = 1;
      else if (!strcmp(argv[i], "--out-file") && i + 1 < argc) g_out_file = argv[++i];
      else if (!strcmp(argv[i], "--in-file") && i + 1 < argc) g_in_file = argv[++i];
      else if (!strcmp(argv[i], "--in-yiw")) g_in_yiw = 1;
      else if (!strcmp(argv[i], "--out-off") && i + 1 < argc) g_out_off = (unsigned)strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--in-off") && i + 1 < argc) g_in_off = (unsigned)strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--two-submit")) g_two_submit = 1;
      else if (!strcmp(argv[i], "--tile-warm") && i + 1 < argc) g_tile_warm = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--two-same")) { g_two_submit = 1; g_two_same = 1; }
      else if (!strcmp(argv[i], "--miss-hist")) g_miss_hist = 1;
      else if (!strcmp(argv[i], "--iw") && i + 1 < argc) c.iw = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--ih") && i + 1 < argc) c.ih = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--ic") && i + 1 < argc) { c.ic = atoi(argv[++i]); ic_given = 1; }
      else if (!strcmp(argv[i], "--ow") && i + 1 < argc) { c.ow = atoi(argv[++i]); ow_given = 1; }
      else if (!strcmp(argv[i], "--oh") && i + 1 < argc) { c.oh = atoi(argv[++i]); oh_given = 1; }
      else if (!strcmp(argv[i], "--kw") && i + 1 < argc) c.kw = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--kh") && i + 1 < argc) c.kh = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--sx") && i + 1 < argc) c.sx = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--sy") && i + 1 < argc) c.sy = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--pad") && i + 1 < argc) {
         c.pad_left = c.pad_top = c.pad_right = c.pad_bottom = atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "--pad-l") && i + 1 < argc) c.pad_left = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--pad-t") && i + 1 < argc) c.pad_top = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--pad-r") && i + 1 < argc) c.pad_right = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--pad-b") && i + 1 < argc) c.pad_bottom = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--method") && i + 1 < argc) c.method = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--precision") && i + 1 < argc) c.precision = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--enable-mask") && i + 1 < argc)
         enable_mask = (uint32_t)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--int-mask") && i + 1 < argc)
         int_mask = (uint32_t)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--int-clear") && i + 1 < argc)
         int_clear = (uint32_t)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--timeout") && i + 1 < argc)
         timeout_ms = (uint32_t)strtol(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--ppu-op-en")) { ppu_op_en = 1; g_ppu_op_en = 1; }
      else if (!strcmp(argv[i], "--recip-w") && i + 1 < argc) g_recip_w = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--recip-h") && i + 1 < argc) g_recip_h = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--surf-len") && i + 1 < argc) g_surf_len = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--mc-surf-out")) g_mc_surf_out = 1;
      else if (!strcmp(argv[i], "--no-transpose")) c.transpose = 0;
      else if (!strcmp(argv[i], "--dev") && i + 1 < argc) dev = argv[++i];
      else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
   }

   if (ic_given) c.ic = ALIGN_UP(c.ic, FEATURE_ATOMIC_SIZE);
   else c.ic = IC_PAD;
   if (!ow_given) c.ow = (c.iw + 2 * c.pad_right - c.kw) / c.sx + 1;
   if (!oh_given) c.oh = (c.ih + 2 * c.pad_bottom - c.kh) / c.sy + 1;

   printf("== min_pool: in %ux%ux%u -> out %ux%ux%u  k=%ux%u s=%ux%u pad=%u "
          "method=%d transpose=%d ==\n",
          c.iw, c.ih, c.ic, c.ow, c.oh, c.ic, c.kw, c.kh, c.sx, c.sy,
          c.pad_left, c.method, c.transpose);
   printf("   PPU 视口: in %ux%u out %ux%u\n",
          ppu_in_w(&c), ppu_in_h(&c), ppu_out_w(&c), ppu_out_h(&c));

   g_iw = c.iw; g_ih = c.ih;
   g_ow = c.ow; g_oh = c.oh;

   int fd = open(dev, O_RDWR);
   if (fd < 0) { fprintf(stderr, "open %s: %s\n", dev, strerror(errno)); return 1; }
   printf("opened %s\n", dev);

   struct bo in, out, rcbo, tk;
   unsigned in_need = (unsigned)c.iw * c.ih * c.ic;
   unsigned out_need = c.ow * c.oh * c.ic;
   if (bo_create(fd, in_need > (1u << 18) ? (in_need + 4095) & ~4095u : 1u << 18, 0, &in)) return 1;
   unsigned out_bo_size = out_need > (1u << 17) ? (out_need + 4095) & ~4095u : 1u << 17;
   if (g_tile_oh) out_bo_size += (out_need + 8191u) & ~4095u;   /* §7.67：给 tile 输出暂存留位置 */
   if (bo_create(fd, out_bo_size, 0, &out)) return 1;
   if (bo_create(fd, 1 << 13, 0, &rcbo)) return 1;
   if (bo_create(fd, 1 << 12, RKNPU_MEM_KERNEL_MAPPING, &tk)) return 1;

   /* 输入图案 */
   unsigned in_bytes = c.iw * c.ih * c.ic;
   if (in_bytes > in.size) { fprintf(stderr, "input too big\n"); return 1; }
   uint8_t *inp = (uint8_t *)in.map;
   if (g_in_file) {   /* §7.192：整块读入外部输入，供 conv→pool 管线对接 */
      FILE *f = fopen(g_in_file, "rb");
      if (!f) printf("IN-FILE: 打不开 %s\n", g_in_file);
      else {
         size_t want = (size_t)(c.ic / FEATURE_ATOMIC_SIZE) * c.iw * c.ih * FEATURE_ATOMIC_SIZE;
         size_t got = fread(inp, 1, want, f);
         fclose(f);
         printf("IN-FILE: %s 读入 %zu/%zu B（%s）\n", g_in_file, got, want,
                g_in_yiw ? "y 主序（卷积输出）→ x 主序" : "x 主序（池化原生）");
         if (g_in_yiw) {   /* conv out: g*w*h*16 + y*w*16 + x*16  →  pool in: g*iw*ih*16 + x*ih*16 + y*16 */
            static uint8_t tmp[1 << 22];
            if (got <= sizeof tmp) {
               memcpy(tmp, inp, got);
               memset(inp, 0, want);
               unsigned groups = c.ic / FEATURE_ATOMIC_SIZE;
               for (unsigned g = 0; g < groups; g++)
                  for (unsigned x = 0; x < c.iw; x++)
                     for (unsigned y = 0; y < c.ih; y++) {
                        size_t src = (size_t)g * c.iw * c.ih * FEATURE_ATOMIC_SIZE
                                     + (size_t)y * c.iw * FEATURE_ATOMIC_SIZE + (size_t)x * FEATURE_ATOMIC_SIZE;
                        size_t dst = (size_t)g * c.iw * c.ih * FEATURE_ATOMIC_SIZE
                                     + (size_t)x * c.ih * FEATURE_ATOMIC_SIZE + (size_t)y * FEATURE_ATOMIC_SIZE;
                        if (src + FEATURE_ATOMIC_SIZE <= got) memcpy(inp + dst, tmp + src, FEATURE_ATOMIC_SIZE);
                     }
            }
         }
      }
   } else
   for (unsigned x = 0; x < c.iw; x++)
      for (unsigned y = 0; y < c.ih; y++)
         for (unsigned ch = 0; ch < c.ic; ch++)
            inp[lay(x, y, ch, c.ih)] = (uint8_t)pat_in(x, y, ch);

   c.in_addr  = (uint32_t)in.dma_addr;
   c.out_addr = (uint32_t)out.dma_addr;
   if (g_in_off) {
      /* §7.79：把输入图案挪到 in BO 内的非零偏移处（同几何、同字节、只改偏移） */
      unsigned need = g_in_off + in_bytes;
      if (need > in.size) { printf("IN-OFF: 需要 %u > in BO %zu ⇒ 放弃\n", need, in.size); return 1; }
      memmove((uint8_t *)in.map + g_in_off, in.map, in_bytes);
      c.in_addr = (uint32_t)in.dma_addr + g_in_off;
      printf("IN-OFF: in@0x%08x + %u\n", (uint32_t)in.dma_addr, g_in_off);
   }
   struct bo in2;
   if (g_stage_in) {
      /* §7.66：同一份字节、同一几何、只换 BO/地址 ⇒ 判定「换 BO」本身是否致命 */
      if (bo_create(fd, (in_bytes + 4095u) & ~4095u, 0, &in2)) return 1;
      memcpy(in2.map, in.map, in_bytes);
      bo_sync(fd, &in2, 1);
      c.in_addr = (uint32_t)in2.dma_addr;
      printf("STAGE-IN: in@0x%08x -> in2@0x%08x (%u B 复制)\n",
             (uint32_t)in.dma_addr, (uint32_t)in2.dma_addr, in_bytes);
   }

   if (g_two_submit) {
      /* §7.68：同进程两次提交，第二次用**另一块 BO** 的地址 ⇒ 结果落在哪块即为答案 */
      if (bo_create(fd, (out_need + 4095u) & ~4095u, 0, &g_out2)) return 1;
      printf("TWO-SUBMIT: out@0x%08x out2@0x%08x\n",
             (uint32_t)out.dma_addr, (uint32_t)g_out2.dma_addr);
      for (int pass = 0; pass < 2; pass++) {
         uint32_t addr = g_two_same ? (uint32_t)g_out2.dma_addr
                                    : (pass ? (uint32_t)g_out2.dma_addr : (uint32_t)out.dma_addr);
         memset(out.map, 0x80, out_need);
         memset(g_out2.map, 0x80, out_need);
         bo_sync(fd, &in, 1);          /* 输入必须先落盘（原路径在提交前 sync in） */
         bo_sync(fd, &out, 1); bo_sync(fd, &g_out2, 1);
         struct pool_cfg cc = c; cc.out_addr = addr;
         unsigned nn = emit_pool(&cc);
         memcpy(rcbo.map, rc_buf, (size_t)nn * 8);
         struct rknpu_task *tt = (struct rknpu_task *)tk.map;
         memset(tt, 0, sizeof(*tt));
         tt->op_idx = 1; tt->enable_mask = enable_mask; tt->int_mask = int_mask;
         tt->int_clear = int_clear; tt->regcfg_amount = nn - RKNPU_PC_DATA_EXTRA_AMOUNT;
         tt->regcmd_addr = rcbo.dma_addr;
         bo_sync(fd, &rcbo, 1); bo_sync(fd, &tk, 1);
         struct rknpu_submit sub2;
         memset(&sub2, 0, sizeof(sub2));
         sub2.flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG;
         sub2.timeout = timeout_ms; sub2.task_number = 1;
         sub2.task_obj_addr = tk.obj_addr; sub2.core_mask = 0; sub2.fence_fd = -1;
         for (int i = 0; i < 5; i++) { sub2.subcore_task[i].task_start = 0; sub2.subcore_task[i].task_number = 1; }
         if (g_dump_submit) dump_submit("two", &sub2, tt, rc_buf, nn);
         int rr = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub2);
         bo_sync(fd, &out, 2); bo_sync(fd, &g_out2, 2);
         unsigned b1 = 0, b2 = 0;
         long h1 = scan_hits((uint8_t *)out.map, out.size, &c, inp, &b1);
         long h2 = scan_hits((uint8_t *)g_out2.map, g_out2.size, &c, inp, &b2);
         printf("PASS%d addr=0x%08x ret=%d: out 命中=%ld@%u, out2 命中=%ld@%u ⇒ 结果在 %s\n",
                pass, addr, rr, h1, b1, h2, b2,
                (h2 > h1) ? "out2（地址生效）" : "out（地址未生效/被锁存）");
      }
      return 0;
   }

   unsigned n = emit_pool(&c);
   if (g_stage_out) {
      /* §7.66：输出换到独立 BO（同一几何、同一输入） */
      if (bo_create(fd, (out_need + 4095u) & ~4095u, 0, &g_out2)) return 1;
      if (!g_no_clear_out) memset(g_out2.map, 0x80, out_need);
      bo_sync(fd, &g_out2, 1);
      c.out_addr = (uint32_t)g_out2.dma_addr;
      printf("STAGE-OUT: out@0x%08x -> out2@0x%08x (%u B)\n",
             (uint32_t)out.dma_addr, (uint32_t)g_out2.dma_addr, out_need);
   }
   if (g_out_off) {
      unsigned need = g_out_off + out_need;
      if (need > out.size) {
         printf("OUT-OFF: 需要 %u > out BO %zu ⇒ 放弃\n", need, out.size);
         return 1;
      }
      memset((uint8_t *)out.map + g_out_off, 0x80, out_need);
      bo_sync(fd, &out, 1);
      c.out_addr = (uint32_t)out.dma_addr + g_out_off;
      g_out_used_off = g_out_off;
      printf("OUT-OFF: out@0x%08x + %u\n", (uint32_t)out.dma_addr, g_out_off);
   }
   if (g_out_off || g_stage_out) {
      /* §7.67 修正：地址变更发生在首次 emit **之后** ⇒ 必须用新地址重新 emit（否则流里仍是旧地址，
       * 之前的"偏移/换 BO 不生效"结论正是这个顺序错误造成的假象）。 */
      unsigned n_re = emit_pool(&c);
      memcpy(rcbo.map, rc_buf, (size_t)n_re * 8);
      printf("RE-EMIT: 用新地址重发 regcmd（%u entries）\n", n_re);
   }
   printf("regcmd: %u entries (%u bytes), regcfg_amount=%u\n",
          n, n * 8, n - RKNPU_PC_DATA_EXTRA_AMOUNT);
   memcpy(rcbo.map, rc_buf, n * 8);

   /* ---------- §7.65 tiled 路径：按输出行切分，绕开 ih>63 ---------- */
   if (g_tile_oh) {
      /* §7.77：**不再新建 BO** —— 输入暂存放 `in` BO 尾部、输出暂存放 `out` BO 尾部，
       * 即"同一块 BO 内偏移"（§7.70 曾试过但当时还有 emit 游标 bug，结论失效，现重试）。
       * 目标是排除"不同 BO/不同尺寸"这一族差异。 */
      unsigned tin_need  = (unsigned)c.iw * (g_tile_oh * c.sy + c.kh) * c.ic;
      unsigned tout_need = (unsigned)c.ow * (g_tile_oh + 2u) * c.ic;   /* §7.78：容纳推导多出的行 */
      unsigned tin_al  = (tin_need + 4095u) & ~4095u;
      unsigned tout_al = (tout_need + 4095u) & ~4095u;
      unsigned in_stage_off  = (in_bytes + 4095u) & ~4095u;
      unsigned out_stage_off = (out_need + 4095u) & ~4095u;
      if (in_stage_off + tin_al > in.size) {
         printf("TILE-WARN: in BO 放不下输入暂存（%u+%u > %zu）\n", in_stage_off, tin_al, in.size);
         return 1;
      }
      if (out_stage_off + tout_al > out.size) {
         printf("TILE-WARN: out BO 放不下输出暂存（%u+%u > %zu）\n", out_stage_off, tout_al, out.size);
         return 1;
      }
      /* §7.80 根因：PPU 的**输入地址偏移不被遵守**（实测 in BO +4096 ⇒ 3072/4096 失配 ✗，
       * 偏移 0 ⇒ 0 ✓；而输出偏移是遵守的 ✓）⇒ 输入暂存必须放在**自己 BO 的基址** ✓。 */
      struct bo stage_in;
      /* §7.81：试探"BO 尺寸/对齐"假设 —— 与通过的 --stage-in 用同一量级（1<<18） */
      unsigned stage_in_size = tin_al > (1u << 18) ? tin_al : (1u << 18);
      if (bo_create(fd, stage_in_size, 0, &stage_in)) return 1;
      printf("TILE-BO: stage_in size=%u (tin_al=%u)\n", stage_in_size, tin_al);
      uint8_t *sb   = (uint8_t *)stage_in.map;             /* 输入暂存（独立 BO 基址） */
      uint8_t *sout = (uint8_t *)out.map + out_stage_off;  /* 输出暂存（out BO 尾部） */
      printf("TILE-STAGE: in@0x%08x+%u, out@0x%08x+%u（同一 BO 内偏移）\n",
             (uint32_t)in.dma_addr, in_stage_off, (uint32_t)out.dma_addr, out_stage_off);
      unsigned tin = tin_need, tout = tout_need;
      uint8_t *inb = (uint8_t *)in.map;
      uint8_t *outb = (uint8_t *)out.map;
      int ret_t = 0; double ms_t = 0; long long hw_t = 0;
      for (unsigned r0 = 0; r0 < c.oh && !ret_t; ) {
         unsigned nt = MIN2(g_tile_oh, c.oh - r0);
         unsigned r0_loop = r0;
         /* §7.90：lead 行方案（§7.89 设计）——保留**硬件 pad 语义**（pt = pad_top 恒定），
          * 同时让每次提交的形状一致：base = r0*sy - pad_top、rows = (nt+pad_top-1)*sy + kh（恒定）、
          * 越界行填 pad 值；有用的输出是 j = pad_top + t ⇒ 散回时丢弃前 pad_top 行。 */
         unsigned pt = c.pad_top;
         int base = (int)(r0 * c.sy) - (int)pt;
         unsigned rows = (nt + pt - 1u) * c.sy + c.kh;
         if (rows > 63u) {
            printf("TILE-WARN: tile 需 %u 输入行（>63，PPU 高度上限）⇒ 请减小 --tile-oh\n", rows);
            ret_t = 1; break;
         }
         memset(sb, 0x80, (size_t)c.iw * rows * c.ic);
         for (unsigned x = 0; x < c.iw; x++)
            for (unsigned y = 0; y < rows; y++) {
               int gy = base + (int)y;
               if (gy < 0 || gy >= (int)c.ih) continue;
               for (unsigned ch = 0; ch < c.ic; ch++)
                  sb[lay_t(x, y, ch, rows, c.iw)] = inb[lay(x, (unsigned)gy, ch, c.ih)];
            }
         {   /* §7.65 自检：整块覆盖（src0=0、rows=ih、pt=pad_top）时暂存应与不切分输入逐字节相同 */
            unsigned cmp_bytes = (unsigned)c.iw * rows * c.ic;
            int same = (base == 0 && rows == c.ih)
                     ? (memcmp(sb, inb, cmp_bytes) == 0) : -1;
            printf("TILE-CHECK: staged vs untiled-input memcmp=%s (%u B)%s\n",
                   same == 1 ? "SAME" : (same == -1 ? "N/A" : "DIFF"), cmp_bytes,
                   same == -1 ? "" : (same == 1 ? "" : " ← 打包布局不一致"));
            if (same == 1) {   /* 打印第一个差异字节 */
               unsigned k = 0; while (k < cmp_bytes && sb[k] == inb[k]) k++;
               if (k < cmp_bytes)
                  printf("TILE-CHECK-DIFF: off=%u staged=%d untiled=%d\n",
                         k, (int)(int8_t)sb[k], (int)(int8_t)inb[k]);
            }
         }
         /* §7.78：tile 的 oh 取 emitter 的**推导值**（与 untiled 同 cfg 的对照一致）；
          * 推导值可能 > nt（多出的行照算、散回时丢弃），这才让视口真正自洽。 */
         unsigned oh_t = (rows + 2u * pt - c.kh) / (c.sy ? c.sy : 1u) + 1u;
         struct pool_cfg ct = c;
         ct.ih = rows; ct.oh = oh_t; ct.pad_top = pt;
         ct.in_addr  = (uint32_t)stage_in.dma_addr;
         ct.out_addr = (uint32_t)out.dma_addr + out_stage_off;
         unsigned nn = emit_pool(&ct);
         printf("TILE r0=%u nt=%u base=%d pt=%u rows=%u oh=%u [lead] regcmd=%u\n", r0, nt, base, pt, rows, oh_t, nn);
         memcpy(rcbo.map, rc_buf, (size_t)nn * 8);
         struct rknpu_task *tt = (struct rknpu_task *)tk.map;
         memset(tt, 0, sizeof(*tt));
         tt->op_idx = 1; tt->enable_mask = enable_mask;
         tt->int_mask = int_mask; tt->int_clear = int_clear;
         tt->regcfg_amount = nn - RKNPU_PC_DATA_EXTRA_AMOUNT;
         tt->regcmd_addr = rcbo.dma_addr;
         memset(sout, 0x80, tout_al);   /* 每次提交前清输出暂存 */
         bo_sync(fd, &stage_in, 1); bo_sync(fd, &out, 1); bo_sync(fd, &rcbo, 1); bo_sync(fd, &tk, 1);
         struct rknpu_submit sub2;
         memset(&sub2, 0, sizeof(sub2));
         sub2.flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG;
         sub2.timeout = timeout_ms; sub2.task_start = 0; sub2.task_number = 1;
         sub2.task_obj_addr = tk.obj_addr; sub2.core_mask = 0; sub2.fence_fd = -1;
         for (int i = 0; i < 5; i++) { sub2.subcore_task[i].task_start = 0; sub2.subcore_task[i].task_number = 1; }
         for (int w = 0; w < g_tile_warm; w++) {   /* §7.86：热身提交（结果丢弃） */
            memset(sout, 0x80, tout_al);
            bo_sync(fd, &stage_in, 1); bo_sync(fd, &out, 1); bo_sync(fd, &rcbo, 1); bo_sync(fd, &tk, 1);
            (void)ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub2);
            bo_sync(fd, &out, 2);
         }
         memset(sout, 0x80, tout_al);
         bo_sync(fd, &stage_in, 1); bo_sync(fd, &out, 1); bo_sync(fd, &rcbo, 1); bo_sync(fd, &tk, 1);
         struct timespec t0, t1;
         clock_gettime(CLOCK_MONOTONIC, &t0);
         int rr = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub2);
         clock_gettime(CLOCK_MONOTONIC, &t1);
         ms_t += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
         hw_t += (long long)sub2.hw_elapse_time;
         if (rr) { ret_t = rr; break; }
         bo_sync(fd, &out, 2);
         {   /* §7.69 逐 tile 自检：暂存输出 vs「按全局逻辑输入算的参考」 */
            long bad_t = 0, maxd_t = 0, ok_oh = 0;
            long probe[6] = {0, 0, 0, 0, 0, 0};   /* [stride: nt|oh_t] × [d: -1|0|+1] */
            for (unsigned ch = 0; ch < c.ic; ch++)
               for (unsigned oy = 0; oy < nt; oy++)
                  for (unsigned ox = 0; ox < c.ow; ox++) {
                     int8_t best = -128;
                     for (unsigned ky = 0; ky < c.kh; ky++)
                        for (unsigned kx = 0; kx < c.kw; kx++) {
                           int ix = (int)(ox * c.sx + kx) - (int)c.pad_left;
                           int iy = (int)((r0 + oy) * c.sy + ky) - (int)c.pad_top;
                           if (ix < 0 || iy < 0 || ix >= (int)c.iw || iy >= (int)c.ih) continue;
                           int8_t v = (int8_t)inp[lay(ix, iy, ch, c.ih)];
                           if (v > best) best = v;
                        }
                     int8_t hw = (int8_t)sout[lay_t(ox, oy + pt, ch, oh_t, c.ow)];
                     if (hw != best) { bad_t++; int d = abs((int)hw - (int)best); if (d > maxd_t) maxd_t = d; }
                     int8_t hw2 = (int8_t)sout[lay_t(ox, oy + pt, ch, oh_t, c.ow)];
                     if (hw2 == best) ok_oh++;
                     /* §7.83：探测"读回映射"——步长 × 行偏移 共 6 种（后续 tile 疑似错位） */
                     for (int si = 0; si < 2; si++) {
                        unsigned st = si ? oh_t : nt;
                        for (int di = 0; di < 3; di++) {
                           int d = di - 1;
                           int yy = (int)oy + d;
                           if (yy < 0 || yy >= (int)st) continue;
                           if ((int8_t)sout[lay_t(ox, (unsigned)yy, ch, st, c.ow)] == best)
                              probe[si * 3 + di]++;
                        }
                     }
                  }
            printf("  TILE-CMP r0=%u nt=%u: bad=%ld/%u (max %ld) | 按 oh_t=%u 读命中=%ld/%u%s\n",
                   r0, nt, bad_t, nt * c.ow * c.ic, maxd_t, oh_t, ok_oh, nt * c.ow * c.ic,
                   bad_t ? (ok_oh * 2 > (long)nt * c.ow * c.ic
                            ? " ⇒ 暂存布局其实是 oh_t（读回步长错）" : " ← tile 计算本身不对")
                         : " ⇒ tile 计算正确");
            printf("    PROBE[%s]: nt(d-1/0/+1)=%ld/%ld/%ld  oh_t(d-1/0/+1)=%ld/%ld/%ld  (总 %u)\n",
                   r0 == 0 ? "首片" : "后续", probe[0], probe[1], probe[2], probe[3], probe[4], probe[5],
                   nt * c.ow * c.ic);
         }
         for (unsigned g = 0; g < (c.ic + FEATURE_ATOMIC_SIZE - 1) / FEATURE_ATOMIC_SIZE; g++)
            for (unsigned ox = 0; ox < c.ow; ox++)
               for (unsigned oy = 0; oy < nt; oy++)
                  memcpy(outb + out_lay(ox, r0 + oy, g * FEATURE_ATOMIC_SIZE),
                         sout + lay_t(ox, oy + pt, g * FEATURE_ATOMIC_SIZE, oh_t, c.ow),
                         FEATURE_ATOMIC_SIZE);
         /* §7.82：按**推导出的**输出行数前进（硬件只产出 oh_t 行；旧版按 nt 前进 ⇒
          * 内层 tile 的几何与产出错配，且最后一行永远缺）。 */
         printf("  TILE-ADV: r0 %u -> %u (nt=%u oh_t=%u rows=%u pt=%u)\n",
                r0_loop, r0_loop + oh_t, nt, oh_t, rows, pt);
         r0 = r0_loop + nt;   /* §7.90：每片固定 nt 个有用行 */
      }
      printf("SUBMIT ret=%d  tiled total %.3f ms  hw_elapse=%lld ns\n", ret_t, ms_t, hw_t);
      if (ret_t) return 1;
      goto after_submit;
   }

   struct rknpu_task *t = (struct rknpu_task *)tk.map;
   memset(t, 0, sizeof(*t));
   t->op_idx        = 1;
   t->enable_mask   = enable_mask;
   t->int_mask      = int_mask;
   t->int_clear     = int_clear;
   t->regcfg_amount = n - RKNPU_PC_DATA_EXTRA_AMOUNT;
   t->regcmd_addr   = rcbo.dma_addr;

   bo_sync(fd, &in, 1);
   bo_sync(fd, &out, 1);
   bo_sync(fd, &rcbo, 1);
   bo_sync(fd, &tk, 1);

   struct rknpu_submit sub;
   memset(&sub, 0, sizeof(sub));
   sub.flags       = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG;
   sub.timeout     = timeout_ms;
   sub.task_start  = 0;
   sub.task_number = 1;
   sub.task_obj_addr = tk.obj_addr;
   sub.core_mask   = 0;
   sub.fence_fd    = -1;
   for (int i = 0; i < 5; i++) { sub.subcore_task[i].task_start = 0; sub.subcore_task[i].task_number = 1; }

   if (g_out_file) {   /* §7.155：在提交前写"输入"，提交后写"输出" ⇒ 便于管线对接 */
      FILE *f = fopen(g_out_file, "wb");
      if (f) { fwrite(in.map, 1, in_need, f); fclose(f); }
   }
   if (g_dump_submit) dump_submit("normal", &sub, (struct rknpu_task *)tk.map, rc_buf, n);
   struct timespec t0, t1;
   clock_gettime(CLOCK_MONOTONIC, &t0);
   int ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub);
   clock_gettime(CLOCK_MONOTONIC, &t1);
   double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
   if (c.ih > 63) {
      /* §7.64 实测：输入高度 ≥64 时 PPU 结果错（ih=63 全对、ih=64 起大面积错）⇒ 高度字段 6 bit。
       * 这是**硬边界**，直接告警（要跑更高的图必须按行切分）。 */
      printf("WARN: 输入高度 ih=%u > 63 —— PPU 高度字段 6 bit，结果不可信（实测 ih=64 起大面积失配）\n",
             c.ih);
   }
   printf("SUBMIT ret=%d errno=%s  %.3f ms  hw_elapse=%lld ns\n",
          ret, ret ? strerror(errno) : "-", ms, (long long)sub.hw_elapse_time);
   if (ret) return 1;
after_submit:
   bo_sync(fd, g_stage_out ? &g_out2 : &out, 2);
   if (g_out_file) {   /* §7.192：必须在 from-device 同步之后 dump，否则读到未失效的 CPU 缓存（旧版缺陷）*/
      const uint8_t *src = g_stage_out ? (const uint8_t *)g_out2.map : (const uint8_t *)out.map;
      char nb[512]; snprintf(nb, sizeof nb, "%s.out", g_out_file);
      FILE *f = fopen(nb, "wb");
      if (f) { fwrite(src, 1, out_need, f); fclose(f); printf("OUT-FILE: %s (%u B)\n", nb, out_need); }
   }

   /* ---- CPU 参考 ---- */
   uint8_t *op = (uint8_t *)(g_stage_out ? g_out2.map : ((uint8_t *)out.map + g_out_used_off));
   long total = 0, bad = 0, maxd = 0;
   long bad_min = 0, bad_avgf = 0, bad_avgr = 0;
   long *hist_x = g_miss_hist ? (long *)calloc(c.ow + 2, sizeof(long)) : NULL;
   long *hist_y = g_miss_hist ? (long *)calloc(c.oh + 2, sizeof(long)) : NULL;
   long hist_is_min = 0;
   for (unsigned ch = 0; ch < c.ic; ch++)
      for (unsigned oy = 0; oy < c.oh; oy++)
         for (unsigned ox = 0; ox < c.ow; ox++) {
            int8_t best = -128, worst = 127;
            long sum = 0; int cnt = 0;
            for (unsigned ky = 0; ky < c.kh; ky++)
               for (unsigned kx = 0; kx < c.kw; kx++) {
                  int ix = (int)(ox * c.sx + kx) - (int)c.pad_left;
                  int iy = (int)(oy * c.sy + ky) - (int)c.pad_top;
                  if (ix < 0 || iy < 0 || ix >= (int)c.iw || iy >= (int)c.ih) continue;
                  int8_t v = (int8_t)inp[lay(ix, iy, ch, c.ih)];
                  if (v > best) best = v;
                  if (v < worst) worst = v;
                  sum += v; cnt++;
               }
            if (!cnt) { best = -128; worst = 127; }
            int8_t hw = (int8_t)op[out_lay(ox, oy, ch)];
            int8_t avgf = cnt ? (int8_t)(sum / cnt) : 0;
            int8_t avgr = cnt ? (int8_t)((sum >= 0) ? (sum + cnt / 2) / cnt
                                                   : -((-sum + cnt / 2) / cnt)) : 0;
            total++;
            if (hw != best && g_miss_detail && miss_shown < 8) {
               /* §7.63：打印失配位置的窗口值，判定「PPU 在 x≡0(mod7) 处取到了什么」 */
               miss_shown++;
               printf("MISS oy=%u ox=%u ch=%u hw=%d best=%d worst=%d avgr=%d win=[", oy, ox, ch,
                      (int)hw, (int)best, (int)worst, (int)avgr);
               for (unsigned ky = 0; ky < c.kh; ky++)
                  for (unsigned kx = 0; kx < c.kw; kx++) {
                     int ix2 = (int)(ox * c.sx + kx) - (int)c.pad_left;
                     int iy2 = (int)(oy * c.sy + ky) - (int)c.pad_top;
                     if (ix2 < 0 || iy2 < 0 || ix2 >= (int)c.iw || iy2 >= (int)c.ih) continue;
                     printf("%d,", (int)(int8_t)inp[lay(ix2, iy2, ch, c.ih)]);
                  }
               printf("]\n");
            }
            if (hw != best) {
               bad++; int d = abs((int)hw - (int)best); if (d > maxd) maxd = d;
               if (g_miss_hist) { hist_x[ox]++; hist_y[oy]++; if (hw == worst) hist_is_min++; }
            }
            if (hw != worst)   bad_min++;
            if (hw != avgf)    bad_avgf++;
            if (hw != avgr)    bad_avgr++;
         }
   if (g_miss_hist && bad) {   /* §7.64：周期结构 */
      printf("  HIST-IS-MIN: hw==窗口最小值的失配 = %ld/%ld\n", hist_is_min, bad);
      printf("  HIST-X:"); for (unsigned x = 0; x < c.ow && x < 80; x++) if (hist_x[x]) printf(" %u:%ld", x, hist_x[x]);
      printf("\n  HIST-Y:"); for (unsigned y = 0; y < c.oh && y < 80; y++) if (hist_y[y]) printf(" %u:%ld", y, hist_y[y]);
      printf("\n");
      free(hist_x); free(hist_y);
   }
   printf("  total=%ld  vs-max mismatch=%ld (max abs diff %ld)\n", total, bad, maxd);
   if (g_out_off || g_stage_out) {   /* §7.67：扫描 BO，找出 PPU 究竟把结果写到了哪个偏移 */
      uint8_t *scanbuf = g_stage_out ? (uint8_t *)g_out2.map : (uint8_t *)out.map;
      size_t scansz = g_stage_out ? g_out2.size : out.size;
      unsigned bo = 0;
      long bh = scan_hits(scanbuf, scansz, &c, inp, &bo);
      printf("  OUT-SCAN[%s]: 最佳匹配偏移=%u（命中粗采样 %ld）⇒ PPU 实际写到这里\n",
             g_stage_out ? "out2" : "out", bo, bh);
   }
   if (g_stage_out) {   /* §7.66：对照旧 out BO —— 判定 PPU 究竟写到哪块 */
      long bad_old = 0, maxd_old = 0;
      for (unsigned ch = 0; ch < c.ic; ch++)
         for (unsigned oy = 0; oy < c.oh; oy++)
            for (unsigned ox = 0; ox < c.ow; ox++) {
               int8_t best = -128;
               for (unsigned ky = 0; ky < c.kh; ky++)
                  for (unsigned kx = 0; kx < c.kw; kx++) {
                     int ix = (int)(ox * c.sx + kx) - (int)c.pad_left;
                     int iy = (int)(oy * c.sy + ky) - (int)c.pad_top;
                     if (ix < 0 || iy < 0 || ix >= (int)c.iw || iy >= (int)c.ih) continue;
                     int8_t v = (int8_t)inp[lay(ix, iy, ch, c.ih)];
                     if (v > best) best = v;
                  }
               int8_t hw2 = (int8_t)((uint8_t *)out.map)[out_lay(ox, oy, ch)];
               if (hw2 != best) { bad_old++; int d = abs((int)hw2 - (int)best); if (d > maxd_old) maxd_old = d; }
            }
      printf("  STAGE-OUT-CMP: 旧 out BO 的失配=%ld（max %ld）⇒ %s\n", bad_old, maxd_old,
             bad_old == 0 ? "PPU 实际写到了旧 out BO（地址寄存器未生效？）" : "旧 BO 也不是正确输出");
   }
   printf("  refs: vs-min=%ld  vs-avg_floor=%ld  vs-avg_round=%ld\n",
          bad_min, bad_avgf, bad_avgr);
   if (bad && bad_min && bad_avgf && bad_avgr) {
      int shown = 0;
      for (unsigned ch = 0; ch < c.ic && shown < 6; ch++)
         for (unsigned oy = 0; oy < c.oh && shown < 6; oy++)
            for (unsigned ox = 0; ox < c.ow && shown < 6; ox++) {
               int8_t best = -128, worst = 127; long sum = 0; int cnt = 0;
               for (unsigned ky = 0; ky < c.kh; ky++)
                  for (unsigned kx = 0; kx < c.kw; kx++) {
                     int ix = (int)(ox * c.sx + kx) - (int)c.pad_left;
                     int iy = (int)(oy * c.sy + ky) - (int)c.pad_top;
                     if (ix < 0 || iy < 0 || ix >= (int)c.iw || iy >= (int)c.ih) continue;
                     int8_t v = (int8_t)inp[lay(ix, iy, ch, c.ih)];
                     if (v > best) best = v; if (v < worst) worst = v; sum += v; cnt++;
                  }
               int8_t hw = (int8_t)op[out_lay(ox, oy, ch)];
               printf("   sample c=%u x=%u y=%u: hw=%4d  max=%4d min=%4d sum=%ld n=%d\n",
                      ch, ox, oy, hw, best, worst, sum, cnt);
               shown++;
            }
   }
   printf("  RESULT: %s\n", bad == 0 ? "BIT-EXACT(max)" :
          bad_min == 0 ? "MATCHES-MIN" :
          (bad_avgf == 0 ? "MATCHES-AVG(floor)" :
           (bad_avgr == 0 ? "MATCHES-AVG(round)" : "MISMATCH")));
   printf("  CFG: iw=%u ih=%u ic=%u ow=%u oh=%u kw=%u kh=%u sx=%u sy=%u int_mask=0x%x ppu_op_en=%d\n",
          c.iw, c.ih, c.ic, c.ow, c.oh, c.kw, c.kh, c.sx, c.sy, int_mask, g_ppu_op_en);

   /* 前若干元素对照，便于定位 */
   if (bad) {
      int shown = 0;
      for (unsigned ch = 0; ch < c.ic && shown < 8; ch++)
         for (unsigned oy = 0; oy < c.oh && shown < 8; oy++)
            for (unsigned ox = 0; ox < c.ow && shown < 8; ox++) {
               int8_t best = -128;
               for (unsigned ky = 0; ky < c.kh; ky++)
                  for (unsigned kx = 0; kx < c.kw; kx++) {
                     int ix = (int)(ox * c.sx + kx) - (int)c.pad_left;
                     int iy = (int)(oy * c.sy + ky) - (int)c.pad_top;
                     if (ix < 0 || iy < 0 || ix >= (int)c.iw || iy >= (int)c.ih) continue;
                     int8_t v = (int8_t)inp[lay(ix, iy, ch, c.ih)];
                     if (v > best) best = v;
                  }
               int8_t hw = (int8_t)op[out_lay(ox, oy, ch)];
               if (hw != best) {
                  printf("   mismatch c=%u x=%u y=%u: cpu=%d hw=%d\n", ch, ox, oy, best, hw);
                  shown++;
               }
            }
   }
   return (bad && bad_min && bad_avgf && bad_avgr) ? 3 : 0;
}
