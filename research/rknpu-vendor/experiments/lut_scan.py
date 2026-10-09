#!/usr/bin/env python3
"""LUT 扫描器（RKNPU ioctl trace → DPU LUT 写入统计）。

用法:
  python3 lut_scan.py <trace.jsonl> [...]            # 统计模式（默认）
  python3 lut_scan.py --fit attention <trace.jsonl>  # attention 大表拟合（§7.190）
  python3 lut_scan.py --fit sigmoid|tanh <trace.jsonl>  # 激活表拟合（§7.191）

背景（§7.134）：
  * trace 必须带 regcmd 载荷 ⇒ 捕获时必须设 `EPERF_RKNPU_REGCMD_MAX_BYTES`（否则 hex 为空 ✗）；
  * 寄存器写在 **`event == "regcmd"`** 事件里（**不是** task 事件 ✗）；
  * 条目编码 = **8 B**：`reg(u16 LE) @0` + `val(u32 LE) @2`（后 2 B 为域外 ✓）。
统计模式输出: 每个 trace 的 LUT 区写入次数、0x4104 唯一值数/样例、LUT 配置寄存器取值。

--fit attention（§7.189 真实喂数语料）：
  * 一个 regcmd 事件里连续 `0x4104` > `BURST_MIN` 条 = **整表突发**，其**写入序 = 表索引序**
    （实测严格非降 ⇒ 非分块乱序 ✓）；
  * 拟合整数四舍五入指数 `value[i] = round(A·exp(k·i))`（网格搜 k + 线性解 A），
    报 mean|err|/max/exact%；并给「平台」（等值游程）表与游程律 `run(v)≈(1/k)·ln((v+.5)/(v-.5))`。
"""
import sys, json, struct, collections, os, math, gzip

LUT_LO, LUT_HI = 0x4100, 0x411c
CFG_REGS = (0x4108, 0x410c, 0x4110, 0x4114, 0x4118, 0x411c)
BURST_MIN = 10  # 单 regcmd 事件内 0x4104 条数阈值，判整表突发

def _open(path):
    return gzip.open(path, "rt", errors="ignore") if path.endswith(".gz") else open(path, errors="ignore")

def scan(path):
    n = 0; area = collections.Counter(); vals = collections.Counter(); cfg = {}
    for ln in _open(path):
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            ev = json.loads(ln)
        except Exception:
            continue
        if ev.get("event") != "regcmd":
            continue
        n += 1
        b = bytes.fromhex(ev.get("hex") or "")
        for i in range(0, len(b) - 7, 8):
            reg = struct.unpack_from("<H", b, i)[0]
            val = struct.unpack_from("<I", b, i + 2)[0]
            if LUT_LO <= reg <= LUT_HI:
                area[reg] += 1
                if reg == 0x4104:
                    vals[val] += 1
                if reg in CFG_REGS:
                    cfg[reg] = val
    return n, area, vals, cfg

def bursts(path):
    """返回每个整表突发的 0x4104 取值序列（写入序 = 索引序候选）。"""
    out = []
    for ln in _open(path):
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            ev = json.loads(ln)
        except Exception:
            continue
        if ev.get("event") != "regcmd":
            continue
        b = bytes.fromhex(ev.get("hex") or "")
        seq = [struct.unpack_from("<I", b, i + 2)[0]
               for i in range(0, len(b) - 7, 8)
               if struct.unpack_from("<H", b, i)[0] == 0x4104]
        if len(seq) > BURST_MIN:
            out.append(seq)
    return out


def plateaus(seq):
    """连续等值游程 → [(start_idx, end_idx, value, run_len)]。"""
    runs = []
    start = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            runs.append((start, i - 1, seq[start], i - start))
            start = i
    return runs


def fit_exp_round(seq):
    """整数四舍五入指数拟合 value[i]=round(A·exp(k·i))。网格 k，A 线性解。"""
    n = len(seq)
    # 跳过首部填充 0（真实表从第一个非零起）
    i0 = next((i for i, v in enumerate(seq) if v > 0), 0)
    xs = list(range(i0, n))
    ys = [float(v) for v in seq[i0:]]
    best = None
    k = 0.0
    # 粗扫精扫两轮
    for lo, hi, steps in ((1e-4, 5e-2, 400), (None, None, 0)):
        if lo is None:
            k0 = best[1]
            lo, hi, steps = k0 * 0.97, k0 * 1.03, 400
        for kk in [lo + (hi - lo) * j / (steps - 1) for j in range(steps)]:
            e = [math.exp(kk * x) for x in xs]
            denom = sum(v * v for v in e)
            A = sum(y * v for y, v in zip(ys, e)) / denom if denom else 0.0
            err = [abs(math.floor(A * v + 0.5) - y) for y, v in zip(ys, e)]
            mean = sum(err) / len(err)
            if best is None or mean < best[0]:
                best = (mean, kk, A, max(err), sum(1 for x in err if x == 0) / len(err), e, xs, ys)
    mean, k, A, mx, exact, e, xs, ys = best
    return dict(k=k, A=A, mean_abs=mean, max_abs=mx, exact_frac=exact, i0=i0, n=n)


def fit_attention(path, csv_path=None):
    bs = bursts(path)
    print("== %s == 整表突发=%d 规模=%s" % (os.path.basename(path), len(bs), [len(b) for b in bs]))
    if not bs:
        print("   (无 >%d 条的 0x4104 突发 ⇒ 非整表加载或未捕获载荷)" % BURST_MIN)
        return
    seq = bs[0]
    same = all(b == seq for b in bs)
    mono = all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
    runs = plateaus(seq)
    nz = [v for v in seq if v > 0]
    print("   各突发相同=%s 非降(写序=索引序)=%s" % (same, mono))
    print("   条目=%d 非零唯一=%d 值域=%d..%d 平台数=%d 最长平台=%d"
          % (len(seq), len(set(nz)), min(nz), max(nz), len(runs), max(r[3] for r in runs)))
    f = fit_exp_round(seq)
    print("   指数拟合 value[i]=round(A·exp(k·i)): A=%.6f k=%.7f | mean|err|=%.4f max=%.0f exact=%.1f%%"
          % (f["A"], f["k"], f["mean_abs"], f["max_abs"], f["exact_frac"] * 100))
    print("   参考: ln2/k=%.3f (每 %.3f 索引翻倍)  1/k=%.2f" % (math.log(2) / f["k"], math.log(2) / f["k"], 1 / f["k"]))
    # 游程律抽样核对
    print("   游程律 run(v)≈(1/k)·ln((v+.5)/(v-.5)) 抽样 [v: 实测/预测]:", end="")
    d = {v: r for _, _, v, r in runs}
    for v in (2, 3, 4, 5, 10, 100, 1000):
        if v in d:
            pred = (1 / f["k"]) * math.log((v + 0.5) / (v - 0.5))
            print("  %d: %d/%.1f" % (v, d[v], pred), end="")
    print()
    # 压缩：原始 u32 → 2 参数 vs 平台表
    print("   压缩: 原始 %d×u32=%dB | 指数 2 参数 | 平台表 %d×(idx,val)"
          % (len(seq), len(seq) * 4, len(runs)))
    if csv_path:
        with open(csv_path, "w") as fo:
            fo.write("idx,value\n")
            for i, v in enumerate(seq):
                fo.write("%d,%d\n" % (i, v))
        print("   wrote %s" % csv_path)


def _lin(x, z, w=None):
    """加权最小二乘 z ≈ b·x + a（w 为权重；使 y 残差而非 z 残差最小）。"""
    if w is None:
        w = 1.0
    W = w
    Sw = (W).sum(); Swx = (W * x).sum(); Swz = (W * z).sum()
    Swxx = (W * x * x).sum(); Swxz = (W * x * z).sum()
    den = Sw * Swxx - Swx * Swx
    if den == 0:
        return 0.0, Swz / Sw
    b = (Sw * Swxz - Swx * Swz) / den
    a = (Swz - b * Swx) / Sw
    return b, a


def fit_activation(path, kind, csv_path=None):
    """sigmoid/tanh 表拟合（§3b-4②，§7.191）：整表突发内写序 = 索引序（实测单调非降）。

    模型 `y[i] = round(S·f(a·i+c))`，f=σ (sigmoid) 或 tanh；对 S 一维网格 + z 线性回归。
    另报固定 S=32767（int16 max，Q15 口径）。
    """
    import numpy as np
    bs = bursts(path)
    print("== %s == 整表突发=%d 规模=%s" % (os.path.basename(path), len(bs), [len(b) for b in bs]))
    if not bs:
        print("   (无整表突发)"); return
    seq = bs[0]
    vals = [(v if v < 2**31 else v - 2**32) for v in seq]
    i0 = next((i for i, v in enumerate(vals) if v != 0), 0)
    y = np.array(vals[i0:], float); x = np.arange(len(y))
    mono = all(y[i] <= y[i + 1] for i in range(len(y) - 1))
    print("   条目=%d 起点=%d n=%d 值域=%.0f..%.0f 单调非降=%s"
          % (len(seq), i0, len(y), y[0], y[-1], mono))

    def model(S, a, c):
        if kind == "sigmoid":
            return S / (1 + np.exp(-(a * x + c)))
        return S * np.tanh(a * x + c)

    def fit_at(S):
        r = y / S
        if kind == "sigmoid":
            if r.min() <= 0 or r.max() >= 1: return None
            z = np.log(r / (1 - r)); dy = S * (r * (1 - r))          # dy/dz
        else:
            if np.abs(r).max() >= 1: return None
            z = np.arctanh(r); dy = S * (1 - r * r)                  # dy/dz = S·sech²
        a, c = _lin(x, z, w=dy * dy); pred = model(S, a, c); e = np.abs(pred - y)
        return (float((e ** 2).mean() ** .5), float(e.max()), a, c)

    gmax = y.max() if kind == "sigmoid" else np.abs(y).max()
    best = None
    for S in np.concatenate([[32767.0], np.linspace(gmax + 1e-6, gmax * 6, 20000)]):
        f = fit_at(S)
        if f and (best is None or f[0] < best[0]):
            best = (f[0], S) + f[1:]
    rmse, S, mx, a, c = best
    print("   %s 拟合 y=round(S·%s(a·i+c)): S=%.1f a=%.6f c=%.4f rmse=%.3f max=%.1f"
          % (kind, "σ" if kind == "sigmoid" else "tanh", S, a, c, rmse, mx))
    f32767 = fit_at(32767)
    if f32767:
        r, m, a2, c2 = f32767
        print("   固定 S=32767: a=%.6f c=%.4f rmse=%.3f max=%.2f exact(≤0.5)=%.1f%%"
              % (a2, c2, r, m, (np.abs(model(32767, a2, c2) - y) <= 0.5).mean() * 100))
    if csv_path:
        with open(csv_path, "w") as fo:
            fo.write("idx,value\n")
            for i, v in enumerate(vals):
                fo.write("%d,%d\n" % (i, v))
        print("   wrote %s" % csv_path)


def main():
    args = sys.argv[1:]
    fit = None
    if len(args) >= 2 and args[0] == "--fit":
        fit = args[1]
        args = args[2:]
    if not args:
        print(__doc__); return 1
    csv_path = None
    if len(args) >= 2 and args[0] == "--csv":
        csv_path = args[1]
        args = args[2:]
    for p in args:
        if fit == "attention":
            fit_attention(p, csv_path)
        elif fit in ("tanh", "sigmoid"):
            fit_activation(p, fit, csv_path)
        else:
            n, area, vals, cfg = scan(p)
            print("== %s == regcmd 事件=%d" % (os.path.basename(p), n))
            print("   LUT 区写入:", {"0x%04x" % k: v for k, v in sorted(area.items())} or "{}")
            print("   0x4104 唯一值=%d top3=%s" % (len(vals), [(hex(v), c) for v, c in vals.most_common(3)]))
            print("   配置:", " ".join("0x%04x=0x%08x" % (k, v) for k, v in sorted(cfg.items())) or "(无)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
