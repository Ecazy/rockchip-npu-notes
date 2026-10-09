/*
 * vendor_replay.c — vendor regcmd 块字节级字面回放（§7.172 / 路线图 Task 1）
 *
 * 数据源：extract_task_blocks.py 从 attn_relu_vendor.jsonl(seq1) 提取的 tasks 0/1/2
 *   捕获字节（69/69/9 u32 words；34.5/34.5/4.5 条目；DST 步进 0x400）。
 * 槽位几何（vendor 实测）：full=0x280 B、increment=0x80 B。
 *
 * 变体：
 *   --variant 0  纯字面：槽内只有捕获字节+0 填充，无尾写
 *                预期：单 task 执行（slot.task_number=1 ⇒ PC_TASK_CONTROL 编程 1），
 *                task0 输出区 HIT、task1/2 哨兵 UNTOUCHED、ret=0。
 *   --variant 1  字面 + 重建尾写（链协议假设，§7.59/§7.60 真值 + 槽几何）：
 *                slot0/1 尾 = [PC_BASE_ADDRESS=下一槽, PC_REGISTER_AMOUNTS=(下一槽字节/16)-1,
 *                             PC_VERSION=0(target 0x41), PC_OPERATION_ENABLE=0xd(target 0x81), pad]
 *                slot2 尾 = mul trace 终端真值 [AMOUNTS=0, VERSION=0, OP_EN=0x18, pad]
 *   --variant 2  全字面（§7.173）：blocks.bin = 重抓 trace 的完整 fetch 窗口
 *                （148/148/28 words = round_up(amount+4,2) 条目，**尾写已含在块内**）；
 *                只 patch DST(0x4020)/SRC(0x5018) 与尾写 PC_BASE(reg 0x10)；
 *                描述符 = vendor 真值（enable_mask=0x18, int_clear=0x1ffff, int_mask=0x300,
 *                slots={{0,1}×3,{0,0}×2}, timeout=36630）。
 *
 * 描述符/submit = §7.171 vendor 规格：flags=0x5 / task_number=3 / task_base=0 /
 * core_mask=0(AUTO) / subcore_task[0..4]={{0,1}×5} / regcfg_amount=69/69/9 / int_mask=0x300。
 * 输出区预填非均匀 pattern 哨兵，提交后逐区比对（HIT/UNTOUCHED）——假成功防线。
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

struct rknpu_mem_create {
   uint32_t handle, flags; uint64_t size, obj_addr, dma_addr, sram_size;
   int32_t iommu_domain_id; uint32_t core_mask;
};
struct rknpu_mem_map { uint32_t handle, reserved; uint64_t offset; };
struct rknpu_mem_sync { uint32_t flags, reserved; uint64_t obj_addr, offset, size; };

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
   uint32_t core_mask; int32_t fence_fd;
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
#define RKNPU_JOB_PINGPONG   (1 << 2)
#define DRM_IOCTL_RKNPU_SUBMIT      DRM_IOWR(DRM_COMMAND_BASE + 0x01, struct rknpu_submit)
#define DRM_IOCTL_RKNPU_MEM_CREATE  DRM_IOWR(DRM_COMMAND_BASE + 0x02, struct rknpu_mem_create)
#define DRM_IOCTL_RKNPU_MEM_MAP     DRM_IOWR(DRM_COMMAND_BASE + 0x03, struct rknpu_mem_map)
#define DRM_IOCTL_RKNPU_MEM_DESTROY DRM_IOWR(DRM_COMMAND_BASE + 0x04, struct rknpu_mem_destroy)
#define DRM_IOCTL_RKNPU_MEM_SYNC    DRM_IOWR(DRM_COMMAND_BASE + 0x05, struct rknpu_mem_sync)

#define SLOT_FULL 0x280
#define SLOT_INCR 0x80
#define N_TASKS   3
#define OUT_STEP  0x400

struct bo { uint32_t handle; uint64_t obj_addr, dma_addr; void *map; uint32_t size; };

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

static void bo_sync(int fd, struct bo *b, uint32_t dir)
{
   struct rknpu_mem_sync ms = { .flags = dir, .obj_addr = b->obj_addr,
                                .offset = 0, .size = b->size };
   ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, &ms);
}

/* PC 寄存器 target：mesa 单元号 | 0x1（与 min_conv_exp 同源） */
static uint32_t tgt(uint32_t reg) { return rkt_get_target(reg) + 0x1; }

static uint64_t mk(uint32_t target, uint32_t reg, uint32_t val)
{
   return ((uint64_t)target << 48) | ((uint64_t)val << 16) | (uint64_t)reg;
}

int main(int argc, char **argv)
{
   const char *dev = "/dev/dri/card1";
   const char *blocks_file = "blocks.bin";
   int variant = 0;
   uint32_t int_mask = 0x300, timeout_ms = 6000;
   uint32_t enable_mask = 0xf, int_clear_v = 0x300;
   uint32_t bwords[3] = { 69, 69, 9 };
   int have_src = 0;
   uint32_t slot_task_number = 1;
   uint32_t amounts[3] = { 0, 0, 0 };        /* 0 = 用 bwords/2−(奇数?5:4) 推导；v2 默认 vendor 69/69/9 */
   uint32_t strides[3] = { 0x280, 0x280, 0x80 };
   uint32_t out_step = OUT_STEP;
   int src_patch = -1;                       /* -1 = 按 variant 默认 */
   /* §7.185：--vpatch VENDOR_VAL:OFF —— 通用指针自适应：所有槽内 val==VENDOR_VAL 的条目
    * 改写为 src.dma+OFF（按值匹配，比按寄存器匹配稳健）。 */
   struct { uint32_t val; uint32_t off; } vpatch[8];
   int npatch = 0;
   int emask_override = 0;                   /* --enable-mask 显式给出时不被 variant 默认覆盖 */
   /* §7.185 v2：--vmem-file/--task-file 完全忠实复刻（真实 blob 数据 + blob[8] 描述符） */
   const char *vmem_file = NULL, *task_file = NULL, *dump_file = NULL;
   /* §7.186：--blk-flip REG:VAL —— 流区内 reg==REG 的条目 val 改为 VAL（保留 tgt/reg），
    * 二分控制位用（v2b 基线上 vendor→ours 翻转） */
   struct { uint32_t reg, val; } bflip[16];
   int nflip = 0;

   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--blocks-file") && i + 1 < argc) blocks_file = argv[++i];
      else if (!strcmp(argv[i], "--vmem-file") && i + 1 < argc) vmem_file = argv[++i];
      else if (!strcmp(argv[i], "--task-file") && i + 1 < argc) task_file = argv[++i];
      else if (!strcmp(argv[i], "--dump-vmem") && i + 1 < argc) dump_file = argv[++i];
      else if (!strcmp(argv[i], "--blk-flip") && i + 1 < argc && nflip < 16) {
         char *s = argv[++i];
         bflip[nflip].reg = (uint32_t)strtoul(s, &s, 0);
         bflip[nflip].val = 0;
         if (*s == ':' || *s == '=') bflip[nflip].val = (uint32_t)strtoul(s + 1, NULL, 0);
         nflip++;
      }
      else if (!strcmp(argv[i], "--variant") && i + 1 < argc) variant = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--timeout") && i + 1 < argc) timeout_ms = (uint32_t)strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--words") && i + 1 < argc) {
         char *s = argv[++i];
         bwords[0] = (uint32_t)strtoul(s, &s, 0); if (*s == ',') s++;
         bwords[1] = (uint32_t)strtoul(s, &s, 0); if (*s == ',') s++;
         bwords[2] = (uint32_t)strtoul(s, &s, 0);
      }
      else if (!strcmp(argv[i], "--subcore-num") && i + 1 < argc) slot_task_number = (uint32_t)strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--amounts") && i + 1 < argc) {
         char *s = argv[++i];
         amounts[0] = (uint32_t)strtoul(s, &s, 0); if (*s == ',') s++;
         amounts[1] = (uint32_t)strtoul(s, &s, 0); if (*s == ',') s++;
         amounts[2] = (uint32_t)strtoul(s, &s, 0);
      }
      else if (!strcmp(argv[i], "--slot-strides") && i + 1 < argc) {
         char *s = argv[++i];
         strides[0] = (uint32_t)strtoul(s, &s, 0); if (*s == ',') s++;
         strides[1] = (uint32_t)strtoul(s, &s, 0); if (*s == ',') s++;
         strides[2] = (uint32_t)strtoul(s, &s, 0);
      }
      else if (!strcmp(argv[i], "--out-step") && i + 1 < argc) out_step = (uint32_t)strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--src-patch") && i + 1 < argc) src_patch = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--vpatch") && i + 1 < argc && npatch < 8) {
         char *s = argv[++i];
         vpatch[npatch].val = (uint32_t)strtoul(s, &s, 0);
         vpatch[npatch].off = 0;
         if (*s == ':' || *s == '+') vpatch[npatch].off = (uint32_t)strtoul(s + 1, NULL, 0);
         npatch++;
      }
      else if (!strcmp(argv[i], "--enable-mask") && i + 1 < argc) {
         char *s = argv[++i];
         uint32_t v = (uint32_t)strtoul(s, NULL, 0);
         emask_override = 1; enable_mask = v;
      }
      else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
   }
   if (variant >= 2) {
      if (bwords[0] == 69 && bwords[1] == 69 && bwords[2] == 9) {
         bwords[0] = 148; bwords[1] = 148; bwords[2] = 28;   /* 完整 fetch 窗口（--words 未给时） */
      }
      if (!emask_override) enable_mask = 0x18;          /* vendor attn 真值；conv 用 --enable-mask 0x1d */
      int_clear_v = 0x1ffff;
      have_src = 1;
      if (amounts[0] == 0) { amounts[0] = 69; amounts[1] = 69; amounts[2] = 9; }
   }
   for (int k = 0; k < 3; k++)
      if (amounts[k] == 0) {  /* 由块长推导：fetch = round_up(amount+4,2) = words/2 ⇒ amount = entries−4 或 −5 */
         uint32_t entries = bwords[k] / 2;
         amounts[k] = (entries & 1) ? entries - 5 : entries - 4;
      }
   if (src_patch >= 0) have_src = src_patch;

   /* blocks.bin：3 块捕获字节 verbatim（words 由 --words/--variant 决定）；v2 模式跳过 */
   static uint8_t blk[N_TASKS][2048];
   if (!vmem_file) {
      FILE *f = fopen(blocks_file, "rb");
      if (!f) { fprintf(stderr, "open %s: %s\n", blocks_file, strerror(errno)); return 1; }
      size_t total = 0;
      for (int k = 0; k < N_TASKS; k++) {
         size_t want = bwords[k] * 4;
         if (fread(blk[k], 1, want, f) != want) { fprintf(stderr, "blocks.bin short\n"); return 1; }
         total += want;
      }
      fclose(f);
      printf("blocks.bin: %zu B (words %u/%u/%u)\n", total, bwords[0], bwords[1], bwords[2]);
   }

   int fd = open(dev, O_RDWR);
   if (fd < 0) { fprintf(stderr, "open %s: %s\n", dev, strerror(errno)); return 1; }

   struct bo stream, out, tk, src;
   if (bo_create(fd, 0x1000, 0, &stream)) return 1;   /* 3 槽：0x280+0x280+0x80 = 0x580 */
   if (bo_create(fd, out_step * N_TASKS + 0x400, 0, &out)) return 1;
   if (bo_create(fd, 0x100, RKNPU_MEM_KERNEL_MAPPING, &tk)) return 1;
   if (bo_create(fd, 0x1400, 0, &src)) return 1;      /* 源数据区（3×0x400 + 余量） */

   for (int k = 0; k < 3; k++)
      if (amounts[k] == 0) {  /* 由块长推导：fetch = round_up(amount+4,2) = words/2 ⇒ amount = entries−4 或 −5 */
         uint32_t entries = bwords[k] / 2;
         amounts[k] = (entries & 1) ? entries - 5 : entries - 4;
      }
   if (src_patch >= 0) have_src = src_patch;

   /* ---- variant 3：vendor conv 链复刻回放（§7.185）----
    * cbc_fp16_full（conv→BN，6 task/1 submit，AUTO slots [[0,2]×3,[0,0]×2]）。
    * 单 BO vmem 复刻 vendor IOVA 镜像 [0xffff6000, 0xfffff000)：
    *   0x0000..0x1000  = t1 DST（哨兵 A）        0x1000..0x2000 = t0 CNA 输入（ramp）
    *   0x2180..0x3000  = t0 DST / t1 FEAT（哨兵 B） 0x4000..      = BS/DCOMP 表区（0）
    *   0x4940..        = 6 个 regcmd 块（vendor 原相对位置）
    * patch = 均匀 delta：所有 val≥0xffff0000 的条目 += (vmem.dma − 0xffff6000)
    * （数据指针与 PC_BASE 尾写指针同一镜像偏移 ⇒ 链结构原样保持）
    * v2（--vmem-file + --task-file）：vmem.bin 含真实权重/BS 表（blob[1..6]）与
    * trace 终值块；task.bin = blob[8] 17 描述符 verbatim（regcmd 为流内相对偏移，
    * 运行时 += vmem.dma+0x4940）；delta patch 仅作用于流区 [0x4940,0x7ec0)
    * —— 权重/表数据（0x4000..0x4940）不 patch，避免破坏真实权重。 */
   if (variant == 3) {
      const uint32_t CBC_BASE = 0xffff6000u;
      const uint32_t baddr[6] = { 0xffffa940u, 0xffffad80u, 0xffffb1c0u,
                                  0xffffb5c0u, 0xffffb9c0u, 0xffffbac0u };
      const uint32_t bamt[6]  = { 126, 126, 122, 122, 26, 122 };
      const uint32_t bents[6] = { 130, 130, 126, 126, 30, 126 };
      const uint32_t bem[6]   = { 0x1d, 0x1d, 0x1d, 0x1d, 0x60, 0x1d };
      const uint32_t bim[6]   = { 0x300, 0x300, 0x300, 0x300, 0xc00, 0x300 };
      const uint32_t bop[6]   = { 1, 2, 1, 2, 2, 1 };
      const uint32_t VMEM_SZ = 0x9000;
      const uint32_t STREAM_OFF = 0x4940, STREAM_SZ = 13696;

      static uint8_t blk3[6][1040];
      if (!vmem_file) {
         FILE *f3 = fopen(blocks_file, "rb");
         if (!f3) { fprintf(stderr, "open %s: %s\n", blocks_file, strerror(errno)); return 1; }
         for (int k = 0; k < 6; k++) {
            size_t want = (size_t)bents[k] * 8;
            if (fread(blk3[k], 1, want, f3) != want) { fprintf(stderr, "blocks.bin short (task%d)\n", k); return 1; }
         }
         fclose(f3);
      }

      int fd = open(dev, O_RDWR);
      if (fd < 0) { fprintf(stderr, "open %s: %s\n", dev, strerror(errno)); return 1; }
      struct bo vmem, tk;
      if (bo_create(fd, VMEM_SZ, 0, &vmem)) return 1;
      if (bo_create(fd, 0x400, RKNPU_MEM_KERNEL_MAPPING, &tk)) return 1;

      uint8_t *vm = (uint8_t *)vmem.map;
      if (vmem_file) {
         FILE *fv = fopen(vmem_file, "rb");
         if (!fv) { fprintf(stderr, "open %s: %s\n", vmem_file, strerror(errno)); return 1; }
         if (fread(vm, 1, VMEM_SZ, fv) != VMEM_SZ) { fprintf(stderr, "vmem short\n"); return 1; }
         fclose(fv);
      } else {
         memset(vm, 0, VMEM_SZ);
         for (int k = 0; k < 6; k++)
            memcpy(vm + (baddr[k] - CBC_BASE), blk3[k], (size_t)bents[k] * 8);
         /* 预填：t1 DST 哨兵 A（0x0）、t0 CNA 输入 ramp、t0 DST/哨兵 B（0x2000..0x3000） */
         for (uint32_t i = 0x0000; i < 0x1000; i++) vm[i] = (uint8_t)((i * 37 + 0x11) & 0xFF);
         for (uint32_t i = 0x1000; i < 0x2000; i++) vm[i] = (uint8_t)(i & 0xFF);
         for (uint32_t i = 0x2000; i < 0x3000; i++) vm[i] = (uint8_t)((i * 53 + 0x2A) & 0xFF);
      }

      /* 均匀 delta：流区 [STREAM_OFF, STREAM_OFF+STREAM_SZ) 内 val≥0xffff0000 → +delta
       * （v2 全图复刻：pc/数据/表指针同镜像；权重数据区不碰） */
      uint32_t delta = (uint32_t)vmem.dma_addr - CBC_BASE;
      uint32_t pstart = vmem_file ? STREAM_OFF : (uint32_t)(baddr[0] - CBC_BASE);
      uint32_t pend   = vmem_file ? STREAM_OFF + STREAM_SZ : (baddr[5] - CBC_BASE) + bents[5] * 8;
      for (uint32_t off = pstart; off < pend; off += 8) {
         uint64_t w64; memcpy(&w64, vm + off, 8);
         uint32_t val = (uint32_t)(w64 >> 16);
         if (val >= 0xffff0000u) {
            /* §7.185 v2 修正：保留 [63:48]=target 与 [15:0]=reg（原 mask 0x0000FFFF0000FFFF
             * 把 target 清零 ⇒ 配置写全丢进 target 0 ⇒ hw 永不启动 ⇒ counter=0 假阴性） */
            uint32_t nv = val + delta;
            uint64_t nw = (w64 & 0xFFFF000000000000ull) | ((uint64_t)nv << 16) | (w64 & 0xFFFF);
            memcpy(vm + off, &nw, 8);
            printf("  [%06x] reg=0x%04x val %08x -> %08x\n",
                   off, (uint32_t)(w64 & 0xFFFF), val, nv);
         }
      }

      /* §7.186：--blk-flip 控制位二分——delta patch 之后，流区内 reg 匹配条目改 val */
      for (int p2 = 0; p2 < nflip; p2++) {
         for (uint32_t off = pstart; off < pend; off += 8) {
            uint64_t w64; memcpy(&w64, vm + off, 8);
            if ((w64 & 0xFFFF) == bflip[p2].reg) {
               uint64_t nw = (w64 & 0xFFFFFFFF00000000ull) | ((uint64_t)bflip[p2].val << 16) | (w64 & 0xFFFF);
               if (nw != w64) {
                  memcpy(vm + off, &nw, 8);
                  printf("  bflip [%06x] reg=0x%04x val -> %08x (tgt 0x%x)\n",
                         off, bflip[p2].reg, bflip[p2].val, (uint32_t)(w64 >> 48));
               }
            }
         }
      }

      struct rknpu_task *t = (struct rknpu_task *)tk.map;
      memset(t, 0, 0x400);
      int ndesc = 6;
      if (task_file) {
         FILE *ft = fopen(task_file, "rb");
         if (!ft) { fprintf(stderr, "open %s: %s\n", task_file, strerror(errno)); return 1; }
         if (fread(tk.map, 1, 680, ft) != 680) { fprintf(stderr, "task.bin short\n"); return 1; }
         fclose(ft);
         for (int k = 0; k < 17; k++)
            t[k].regcmd_addr += (uint64_t)vmem.dma_addr + STREAM_OFF;
         ndesc = 17;
      } else {
         for (int k = 0; k < 6; k++) {
            t[k].op_idx = bop[k]; t[k].enable_mask = bem[k];
            t[k].int_mask = bim[k]; t[k].int_clear = 0x1ffff;
            t[k].regcfg_amount = bamt[k];
            t[k].regcmd_addr = (uint64_t)vmem.dma_addr + (baddr[k] - CBC_BASE);
         }
      }
      bo_sync(fd, &vmem, 1); bo_sync(fd, &tk, 1);

      struct rknpu_submit sub;
      memset(&sub, 0, sizeof(sub));
      sub.flags = RKNPU_JOB_PC | RKNPU_JOB_PINGPONG;   /* vendor flags=0x5 */
      sub.timeout = timeout_ms;
      sub.task_start = 0; sub.task_number = 6;
      sub.task_obj_addr = tk.obj_addr;
      sub.task_base_addr = 0; sub.core_mask = 0; sub.fence_fd = -1;
      for (int i = 0; i < 3; i++) { sub.subcore_task[i].task_start = 0; sub.subcore_task[i].task_number = 2; }

      printf("cbc replay v%d: %s vmem@%08x delta=%+d t0blk=%08x desc=%d timeout=%u slots=[[0,2]x3]\n",
             vmem_file ? 2 : 1, vmem_file ? vmem_file : blocks_file,
             (uint32_t)vmem.dma_addr, (int32_t)delta,
             (uint32_t)(vmem.dma_addr + STREAM_OFF), ndesc, timeout_ms);

      struct timespec t0, t1;
      clock_gettime(CLOCK_MONOTONIC, &t0);
      int ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub);
      clock_gettime(CLOCK_MONOTONIC, &t1);
      double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
      printf("SUBMIT ret=%d errno=%s %.3f ms hw_elapse=%lld ns task_counter=%u\n",
             ret, ret ? strerror(errno) : "-", ms,
             (long long)sub.hw_elapse_time, sub.task_counter);

      bo_sync(fd, &vmem, 2);
      if (dump_file) {
         FILE *fdump = fopen(dump_file, "wb");
         if (fdump) { fwrite(vm, 1, VMEM_SZ, fdump); fclose(fdump); printf("dump: %s\n", dump_file); }
      }
      struct { uint32_t off, len; const char *name; } zones[2] = {
         { 0x2180, 0x400, "DST0 conv-out" },
         { 0x0000, 0x400, "DST1 bn-out" },
      };
      int hit = 0;
      for (int z = 0; z < 2; z++) {
         uint32_t step = zones[z].off ? 53 : 37, bias = zones[z].off ? 0x2A : 0x11;
         long diff = 0;
         for (uint32_t i = 0; i < zones[z].len; i++)
            if (vm[zones[z].off + i] != (uint8_t)(((zones[z].off + i) * step + bias) & 0xFF)) diff++;
         printf("ZONE %s: %s (%ld/%u changed) first8:", zones[z].name,
                diff ? "HIT" : "UNTOUCHED", diff, zones[z].len);
         for (int b = 0; b < 8; b++) printf(" %02x", vm[zones[z].off + b]);
         printf("\n");
         if (diff) hit++;
      }
      printf("RESULT: %s (variant 3 cbc v%d)\n",
             ret == 0 ? (hit == 2 ? "ALL-TASKS-EXECUTED" : "PARTIAL") : "SUBMIT-FAILED",
             vmem_file ? 2 : 1);
      return ret ? 1 : 0;
   }

   uint32_t slot_off[N_TASKS];
   slot_off[0] = 0; slot_off[1] = strides[0]; slot_off[2] = strides[0] + strides[1];
   const uint32_t slot_sz[N_TASKS]  = { strides[0], strides[1], strides[2] };

   /* 填槽：捕获字节 verbatim + 0 填充；patch DST/SRC（保留 target 位） */
   uint8_t *sm = (uint8_t *)stream.map;
   memset(sm, 0, stream.size);
   for (int k = 0; k < N_TASKS; k++) {
      memcpy(sm + slot_off[k], blk[k], bwords[k] * 4);
      for (uint32_t w = 0; w < bwords[k] / 2; w++) {
         uint64_t w64;
         memcpy(&w64, sm + slot_off[k] + (size_t)w * 8, 8);
         uint32_t reg = w64 & 0xFFFF;
         uint32_t tgt = (uint32_t)(w64 >> 48);
         int handled = 0;
         for (int p = 0; p < npatch; p++) {
            if ((uint32_t)(w64 >> 16) == vpatch[p].val) {
               uint64_t nw = mk(tgt, reg, (uint32_t)(src.dma_addr + vpatch[p].off));
               memcpy(sm + slot_off[k] + (size_t)w * 8, &nw, 8);
               printf("slot%d vpatch val=0x%08x -> 0x%08x (reg 0x%04x tgt 0x%x)\n",
                      k, vpatch[p].val, (uint32_t)(src.dma_addr + vpatch[p].off), reg, tgt);
               handled = 1;
               break;
            }
         }
         if (handled) continue;
         if (reg == REG_DPU_DST_BASE_ADDR) {
            uint64_t nw = mk(tgt, reg, (uint32_t)(out.dma_addr + (uint64_t)k * out_step));
            memcpy(sm + slot_off[k] + (size_t)w * 8, &nw, 8);
            printf("slot%d patch 0x%04x -> DST 0x%08x (target 0x%x)\n",
                   k, reg, (uint32_t)(out.dma_addr + k * out_step), tgt);
         } else if (have_src && reg == 0x5018) {
            uint64_t nw = mk(tgt, reg, (uint32_t)(src.dma_addr + (uint64_t)k * out_step));
            memcpy(sm + slot_off[k] + (size_t)w * 8, &nw, 8);
            printf("slot%d patch 0x5018 (SRC) -> 0x%08x (target 0x%x)\n",
                   k, (uint32_t)(src.dma_addr + k * OUT_STEP), tgt);
         } else if (variant >= 2 && reg == REG_PC_BASE_ADDRESS && tgt == 0x101) {
            /* 尾写链指针：指向流内下一槽（vendor 原值 = vendor 布局地址，必须改到我们的流） */
            uint32_t next = (k < 2) ? (uint32_t)(stream.dma_addr + slot_off[k + 1]) : 0;
            uint64_t nw = mk(tgt, reg, next);
            memcpy(sm + slot_off[k] + (size_t)w * 8, &nw, 8);
            printf("slot%d patch tail PC_BASE -> 0x%08x (target 0x%x)\n", k, next, tgt);
         } else if (variant >= 2 && k == N_TASKS - 1 && reg == REG_PC_REGISTER_AMOUNTS && tgt == 0x101) {
            /* 终端：AMOUNTS=0（vendor 尾块 AMOUNTS=0x6 指向 task3，须归零防续链） */
            uint64_t nw = mk(tgt, reg, 0);
            memcpy(sm + slot_off[k] + (size_t)w * 8, &nw, 8);
            printf("slot%d patch tail PC_REGISTER_AMOUNTS -> 0 (terminal)\n", k);
         }
      }
   }

   /* 变体 1：重建尾写（链协议），slot0/1 链下一槽，slot2 终止（mul trace 终端真值）
    * 变体 2 不走这里 —— 尾写已在 blocks 内（verbatim），只 patch PC_BASE/SRC/DST */
   if (variant == 1) {
      const uint32_t tail_off[N_TASKS] = { 35 * 8, 35 * 8, 5 * 8 };
      struct { uint32_t tgt, reg, val; } tails[N_TASKS][5] = {
         { /* slot0 → slot1（0x280 = 40 单位） */
            { 0x101, REG_PC_BASE_ADDRESS,     0 },               /* val 运行期填 */
            { 0x101, REG_PC_REGISTER_AMOUNTS, (SLOT_FULL / 16) - 1 },
            { 0x41,  REG_PC_VERSION,          0 },
            { 0x81,  REG_PC_OPERATION_ENABLE, 0xd },
            { 0, 0, 0 },
         },
         { /* slot1 → slot2（0x80 = 8 单位） */
            { 0x101, REG_PC_BASE_ADDRESS,     0 },
            { 0x101, REG_PC_REGISTER_AMOUNTS, (SLOT_INCR / 16) - 1 },
            { 0x41,  REG_PC_VERSION,          0 },
            { 0x81,  REG_PC_OPERATION_ENABLE, 0xd },
            { 0, 0, 0 },
         },
         { /* slot2 终止：无 PC_BASE，AMOUNTS=0，OP_EN=0x18（mul trace 尾写真值） */
            { 0x101, REG_PC_REGISTER_AMOUNTS, 0 },
            { 0x41,  REG_PC_VERSION,          0 },
            { 0x81,  REG_PC_OPERATION_ENABLE, 0x18 },
            { 0, 0, 0 },
            { 0, 0, 0 },
         },
      };
      tails[0][0].val = (uint32_t)(stream.dma_addr + slot_off[1]);
      tails[1][0].val = (uint32_t)(stream.dma_addr + slot_off[2]);
      for (int k = 0; k < N_TASKS; k++) {
         uint64_t *p = (uint64_t *)(sm + slot_off[k] + tail_off[k]);
         for (int e = 0; e < 5; e++)
            p[e] = mk(tails[k][e].tgt, tails[k][e].reg, tails[k][e].val);
      }
      printf("variant 1: tails @ slot0+280 slot1+280 slot2+40 ; chain %08x -> %08x -> term\n",
             (uint32_t)stream.dma_addr + slot_off[1], (uint32_t)stream.dma_addr + slot_off[2]);
   }

   /* 描述符（vendor 形态）+ 哨兵预填 */
   struct rknpu_task *t = (struct rknpu_task *)tk.map;
   memset(t, 0, N_TASKS * sizeof(*t));
   for (int k = 0; k < N_TASKS; k++) {
      t[k].op_idx = 1; t[k].enable_mask = enable_mask;
      t[k].int_mask = int_mask; t[k].int_clear = int_clear_v;
      t[k].regcfg_amount = amounts[k];
      t[k].regcmd_addr = (uint64_t)stream.dma_addr + slot_off[k];
   }
   uint8_t *om = (uint8_t *)out.map;
   for (uint32_t i = 0; i < out.size; i++) om[i] = (uint8_t)((i * 37 + 0x11) & 0xFF);
   uint8_t *srcm = NULL;
   if (have_src) {
      srcm = (uint8_t *)src.map;
      for (uint32_t i = 0; i < src.size; i++) srcm[i] = (uint8_t)((i * 53 + 0x2A) & 0xFF);
      bo_sync(fd, &src, 1);
   }
   bo_sync(fd, &stream, 1); bo_sync(fd, &out, 1); bo_sync(fd, &tk, 1);

   struct rknpu_submit sub;
   memset(&sub, 0, sizeof(sub));
   sub.flags = RKNPU_JOB_PC | RKNPU_JOB_PINGPONG;   /* vendor flags=0x5 */
   sub.timeout = timeout_ms;
   sub.task_start = 0;
   sub.task_number = N_TASKS;
   sub.task_obj_addr = tk.obj_addr;
   sub.task_base_addr = 0;
   sub.core_mask = 0;                               /* AUTO */
   sub.fence_fd = -1;
   for (int i = 0; i < 5; i++) { sub.subcore_task[i].task_start = 0; sub.subcore_task[i].task_number = slot_task_number; }
   if (variant >= 2) {                               /* vendor 真值：仅槽 0..2 填值 */
      for (int i = 3; i < 5; i++) { sub.subcore_task[i].task_start = 0; sub.subcore_task[i].task_number = 0; }
   }

   printf("submit: task_number=3 core_mask=0 flags=0x5 slots={{0,%u}x3,{0,0}x2} amounts=%u/%u/%u variant=%d\n",
          slot_task_number, bwords[0], bwords[1], bwords[2], variant);
   printf("BO: stream@%08x out@%08x task_obj=0x%llx\n", (uint32_t)stream.dma_addr,
          (uint32_t)out.dma_addr, (unsigned long long)tk.obj_addr);

   struct timespec t0, t1;
   clock_gettime(CLOCK_MONOTONIC, &t0);
   int ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub);
   clock_gettime(CLOCK_MONOTONIC, &t1);
   double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
   printf("SUBMIT ret=%d errno=%s %.3f ms hw_elapse=%lld ns task_counter=%u\n",
          ret, ret ? strerror(errno) : "-", ms,
          (long long)sub.hw_elapse_time, sub.task_counter);

   int hit = 0, hit_all = 1;
   if (ret == 0) {
      bo_sync(fd, &out, 2);
      for (int k = 0; k < N_TASKS; k++) {
         long diff = 0;
         for (uint32_t i = 0; i < out_step; i++)
            if (om[k * out_step + i] != (uint8_t)((i * 37 + 0x11) & 0xFF)) diff++;
         printf("TASK-OUT %d: %s (%ld/%d bytes changed) first8:", k,
                diff ? "HIT" : "UNTOUCHED", diff, OUT_STEP);
         for (int b = 0; b < 8; b++) printf(" %02x", om[k * OUT_STEP + b]);
         printf("\n");
         if (diff) hit++;
         else hit_all = 0;
      }
      printf("RESULT: %s (variant %d)\n", hit_all ? "ALL-TASKS-EXECUTED" : "PARTIAL", variant);
   } else {
      printf("RESULT: SUBMIT-FAILED (timeout family; check dmesg) (variant %d)\n", variant);
   }
   return ret ? 1 : 0;
}
