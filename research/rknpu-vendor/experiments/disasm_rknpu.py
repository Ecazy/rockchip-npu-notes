#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§7.170: 反汇编板上内核 (/boot/Image) 中的 rknpu 驱动函数。

用法（本机，Image 已拉到本地）:
  python3 experiments/disasm_rknpu.py <Image 文件> <符号 VA 十六进制> [字节数]

偏移基准: 文件偏移 = VA - 0xffffffc008000000
  - 校验: linux_banner（System.map rodata 首符号）在 Image 内偏移吻合 ✓
  - /proc/kallsyms 运行时地址 == System.map 静态地址 ⇒ KASLR 关闭 ✓
前置: 板上拉取 —— .tmp/board.sh 'base64 /boot/Image' | base64 -d > Image
      .tmp/board.sh 'cat /boot/System.map-6.1.43-rockchip-rk3588-rknpu098' > System.map

关键符号（本板实测）:
  rknpu_get_task_number        0xffffffc009002910
  rknpu_job_subcore_commit     0xffffffc009002e30 (.isra.0, 内联 commit_pc)
  rknpu_job_schedule           0xffffffc0090033d0
产物: experiments/results/driver_disasm_20261002/rknpu_disasm.txt
"""
import sys
import capstone

BASE = 0xffffffc008000000  # VA -> Image 文件偏移（KASLR off，由 linux_banner 校验）


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/opencode/Image'
    img = open(path, 'rb').read()
    md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_LITTLE_ENDIAN)
    # "name va size" 三元组直接写在脚本里（符号地址随内核版本固定）
    FUNCS = [
        ("rknpu_get_task_number",    0xffffffc009002910, 0x80),
        ("rknpu_job_subcore_commit", 0xffffffc009002e30, 0x400),
        ("rknpu_job_schedule",       0xffffffc0090033d0, 0x280),
    ]
    for name, va, sz in FUNCS:
        print(f"===== {name} @ {va:#x} =====")
        code = img[va - BASE: va - BASE + sz]
        for ins in md.disasm(code, va):
            print(f"{ins.address:x}:  {ins.mnemonic:<8}{ins.op_str}")


if __name__ == '__main__':
    main()
