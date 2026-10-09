#!/usr/bin/env python3
"""提取 vendor trace 的 regcmd 块原始字节（§7.172 / 路线图 Task 1 Step 1）。

用法：extract_task_blocks.py <trace.jsonl> [--seq N] [--tasks K]
         [--raw-out blocks.bin] [--meta-out meta.txt]

输出（--raw-out）：blocks.bin = 逐块拼接捕获载荷（verbatim 字节，不做任何改写）。
meta：每块 u32 字数、条目数、regcfg_amount、DPU_DST_BASE_ADDR(0x4020) 值。

说明：attn 系 hook 捕获字节数 = regcfg_amount × 4 B，尾写在槽位未捕获区（§7.172）；
blocks.bin = 捕获字节 verbatim，尾写由 vendor_replay 按变体重建。
"""
import json
import sys


def main():
    path = sys.argv[1]
    seq, ntasks, raw_out, meta_out = 1, 3, None, None
    argv = sys.argv[2:]
    i = 0
    while i < len(argv):
        if argv[i] == "--seq":
            seq = int(argv[i + 1]); i += 2
        elif argv[i] == "--tasks":
            ntasks = int(argv[i + 1]); i += 2
        elif argv[i] == "--raw-out":
            raw_out = argv[i + 1]; i += 2
        elif argv[i] == "--meta-out":
            meta_out = argv[i + 1]; i += 2
        else:
            print("unknown arg", argv[i]); return 2

    lst = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if '"event":"regcmd"' not in line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("sequence") != seq:
            continue
        raw = bytes.fromhex(ev["hex"])
        ents = []
        for k in range(len(raw) // 8):
            w = int.from_bytes(raw[k * 8:k * 8 + 8], "little")
            ents.append((w & 0xFFFF, (w >> 16) & 0xFFFFFFFF, w >> 48))
        dst = [v for o, v, tg in ents if o == 0x4020]
        lst.append({
            "task_index": ev.get("task_index"),
            "op_idx": ev.get("op_idx"),
            "amount": ev.get("regcfg_amount"),
            "raw": raw,
            "u32_words": len(raw) // 4,
            "entries": len(raw) / 8,
            "dst": dst,
            "first_reg": ents[0][0] if ents else None,
            "last_reg": ents[-1][0] if ents else None,
        })

    print(f"trace={path} seq={seq} regcmd events: {len(lst)}")
    use = lst[:ntasks]
    out = b"".join(m["raw"] for m in use)
    if raw_out:
        open(raw_out, "wb").write(out)
    lines = []
    for m in use:
        lines.append(
            "task%-4d amount=%-4d u32_words=%-4d entries=%.1f DST=%s first=0x%04x last=0x%04x"
            % (m["task_index"], m["amount"], m["u32_words"], m["entries"],
               [hex(x) for x in m["dst"]] if m["dst"] else "-", m["first_reg"], m["last_reg"]))
    txt = "\n".join(lines) + f"\ntotal raw bytes = {len(out)}\n"
    if meta_out:
        open(meta_out, "w").write(txt)
    sys.stdout.write(txt)
    dsts = [m["dst"][0] for m in use if m["dst"]]
    print("DST per task:", dsts)
    if len(dsts) >= 2:
        step = dsts[1] - dsts[0]
        print(f"DST step = {step} B (0x{step:x})")
    return 0


if __name__ == "__main__":
    main()
