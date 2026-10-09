#!/usr/bin/env python3
"""从 hook trace（rknpu_ioctl 的 regcmd 事件）抽 CNA 的 CBUF bank / FEATURE_GRAINS 规划。

用途：验证 §7.31 的结论（真包线 = `DATA_BANK × 32768 B` 的输入 tile + 权重 bank，
`FEATURE_GRAINS` 随行字节放大）是否与 vendor runtime 的实际寄存器一致。

用法：
    analyze_vendor_cbuf_banks.py <trace.jsonl> [<trace.jsonl> ...]
    analyze_vendor_cbuf_banks.py --all            # 扫描仓库内全部 trace

对每个 regcmd 事件解 NPUOP（64-bit LE：target=(w>>48), value=(w>>16), reg=w&0xffff），
读：
  CNA_CBUF_CON0 0x1040 → WEIGHT_BANK(bits7:4) / DATA_BANK(bits3:0) / FC_DATA_BANK(bits10:8)
  CNA_CONV_CON2 0x1010 → FEATURE_GRAINS(bits13:4) / KERNEL_GROUP(bits23:16)
  CNA_DATA_SIZE0/1/2/3  → DATAIN_WIDTH/HEIGHT、CHANNEL(_REAL)、DATAOUT_WIDTH、ATOMICS
  CNA_WEIGHT_SIZE0/1/2  → 权重描述符字节 / (kw,kh,kernels)
  CNA_DMA_CON1/CON2     → LINE_STRIDE / SURF_STRIDE（32-bit 字）
并打印 `DATA_BANK != 1` 或 `WEIGHT_BANK != 11` 的 task（旧发射器的假设之外）。
"""
import glob
import json
import struct
import sys

R_CNA_CBUF_CON0 = 0x1040
R_CNA_CONV_CON2 = 0x1010
R_CNA_DATA_SIZE0 = 0x1020
R_CNA_DATA_SIZE1 = 0x1024
R_CNA_DATA_SIZE2 = 0x1028
R_CNA_DATA_SIZE3 = 0x102c
R_CNA_WEIGHT_SIZE0 = 0x1030
R_CNA_WEIGHT_SIZE1 = 0x1034
R_CNA_WEIGHT_SIZE2 = 0x1038
R_CNA_DMA_CON1 = 0x1074
R_CNA_DMA_CON2 = 0x1078
R_CNA_CONV_CON1 = 0x100c
R_CNA_CONV_CON3 = 0x1014


def decode(blob):
    """hex 串 → {reg: value}（后写覆盖先写，与硬件 Group0 语义近似）。"""
    regs = {}
    raw = bytes.fromhex(blob)
    for i in range(len(raw) // 8):
        w = struct.unpack_from("<Q", raw, i * 8)[0]
        reg = w & 0xFFFF
        val = (w >> 16) & 0xFFFFFFFF
        regs[reg] = val
    return regs


def f(val, mask, shift):
    return (val & mask) >> shift if val is not None else None


def scan(path):
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if '"regcmd"' not in line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("event") != "regcmd" or "hex" not in ev:
            continue
        r = decode(ev["hex"])
        cbuf = r.get(R_CNA_CBUF_CON0)
        con2 = r.get(R_CNA_CONV_CON2)
        if cbuf is None or con2 is None:
            continue
        ds0, ds1 = r.get(R_CNA_DATA_SIZE0), r.get(R_CNA_DATA_SIZE1)
        ws0, ws2 = r.get(R_CNA_WEIGHT_SIZE0), r.get(R_CNA_WEIGHT_SIZE2)
        rows.append(dict(
            op=ev.get("op_idx"), seq=ev.get("sequence"),
            data_bank=f(cbuf, 0xF, 0), weight_bank=f(cbuf, 0xF0, 4),
            fc_data_bank=f(cbuf, 0x700, 8),
            grains=f(con2, 0x3FF0, 4), kernel_group=f(con2, 0xFF0000, 16),
            iw=f(ds0, 0xFFF, 16), ih=f(ds0, 0xFFF, 0),
            ic=f(ds1, 0xFFF, 0), ic_real=(f(ds1, 0xFFF, 16) or 0) + 1 if ds1 is not None else None,
            ow=f(r.get(R_CNA_DATA_SIZE2), 0xFFF, 0), atoms=f(r.get(R_CNA_DATA_SIZE3), 0xFFFFFF, 0),
            wbytes=ws0,
            kw=f(ws2, 0xFF, 0), kh=f(ws2, 0xFF00, 8), kernels=f(ws2, 0xFF0000, 16),
            line_stride=r.get(R_CNA_DMA_CON1), surf_stride=r.get(R_CNA_DMA_CON2),
            conv_con1=r.get(R_CNA_CONV_CON1), conv_con3=r.get(R_CNA_CONV_CON3),
        ))
    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--all" in sys.argv or not args:
        args = sorted(set(glob.glob("experiments/**/*.jsonl", recursive=True)
                          + glob.glob("findings/**/*.jsonl", recursive=True)))
    grand = 0
    interesting = 0
    print(f"{'trace':<58} {'tasks':>6} {'DB!=1':>6} {'WB!=11':>7} {'grains set':>22}")
    for p in args:
        try:
            rows = scan(p)
        except FileNotFoundError:
            continue
        if not rows:
            continue
        grand += len(rows)
        dbn = sum(1 for r in rows if r["data_bank"] != 1)
        wbn = sum(1 for r in rows if r["weight_bank"] != 11)
        gs = sorted({r["grains"] for r in rows})
        gtxt = (", ".join(str(g) for g in gs[:6]) + ("…" if len(gs) > 6 else "")) if gs else "-"
        print(f"{p:<58} {len(rows):>6} {dbn:>6} {wbn:>7} {gtxt:>22}")
        for r in rows:
            if r["data_bank"] != 1 or r["weight_bank"] != 11:
                interesting += 1
                print("    op={op} seq={seq} DATA_BANK={data_bank} WEIGHT_BANK={weight_bank} "
                      "FC_DATA_BANK={fc_data_bank} grains={grains} KG={kernel_group} "
                      "in=({iw}x{ih})x{ic} out_w={ow} atoms={atoms} wbytes={wbytes} "
                      "k={kw}x{kh}x{kernels} line={line_stride} surf={surf_stride}".format(**r))
    print(f"\ntotal regcmd tasks={grand}  非默认 bank 的 task={interesting}")


if __name__ == "__main__":
    main()
