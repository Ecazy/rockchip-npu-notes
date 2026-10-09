#!/usr/bin/env python3
"""LUT 逐项拟合（§3b-4②，路线图 Task 4，纯离线）。

数据: 表 = 0x4104 写序唯一值（u32，低 16 位 = fp16 条目），来自 2026-10-02 重抓：
  sigmoid_1x16_fp16 (439), tanh_1x16_fp16 (442), f_Softmax_1x64_fp16 (110)
  —— 与 §7.137 普查的表规模一致（440/443/111 系含 0 条目的差 1）。

候选模型（计划规定）:
  A. 近指数 A·exp(b·i + c·i²)（log 域全局二次，sigmoid 同族假设）
  B. 分段线性（以 5% 相对跳变为节点的分段线性）
判据: log 域 RMSE/max 残差 vs fp16 相对精度 eps=2^-11（log 域 ≈ 4.9e-4）+ AIC。
附注: tanh 表（442 项）写入序单调（无平台），§7.153 的平台证据属 attention 大表（1027 项），
      本轮未重抓（attn_softmax 需多输入喂数，属 Task 3）。
"""
import csv
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FP16_LOG_EPS = math.log1p(2.0 ** -11)  # fp16 相对精度在 log 域 ≈ 4.883e-4


def fp16_to_float(x):
    s = (x >> 15) & 1
    e = (x >> 10) & 0x1F
    m = x & 0x3FF
    if e == 0:
        f = m * 2.0 ** -24
    elif e == 31:
        f = float("inf")
    else:
        f = (1 + m / 1024) * 2.0 ** (e - 15)
    return -f if s else f


def fit_table(name, u):
    vals = [fp16_to_float(v & 0xFFFF) for v in u]
    y = np.abs(np.array(vals))
    n = len(y)
    idx = np.arange(n, dtype=float)
    logy = np.log(y)

    # A: 近指数（log 域全局二次）
    c, b, a = np.polyfit(idx, logy, 2)
    lr = logy - (a + b * idx + c * idx ** 2)
    rmse_log = float(np.sqrt(np.mean(lr ** 2)))
    maxres_log = float(np.max(np.abs(lr)))
    aic_quad = float(n * np.log(float(np.mean(lr ** 2))) + 2 * 3)

    # B: 分段线性（5% 相对跳变切段；fp16 相邻可分辨 ≈ 1/2048，5% ⇒ 真实弯折点）
    d = np.abs(np.diff(y)) / np.maximum(np.abs(y[1:]), 1e-12)
    brk = [k for k in range(n - 1) if d[k] > 0.05]
    segs = [0] + [x + 1 for x in brk] + [n]
    pieces = len(segs) - 1
    maxres_lin = 0.0
    for s, e in zip(segs[:-1], segs[1:]):
        if e - s >= 3:
            p = np.polyfit(idx[s:e], y[s:e], 1)
            r = float(np.max(np.abs(y[s:e] - np.polyval(p, idx[s:e]))) / max(abs(float(y[s])), 1e-12))
            maxres_lin = max(maxres_lin, r)
    k_lin = 2 * pieces2 if (pieces2 := pieces) else 0
    aic_lin = n * np.log(max(maxres_lin / max(float(np.mean(y)), 1e-12), 1e-12)) + 2 * k_lin

    return {
        "n": n,
        "quad_log": {"a": a, "b": b, "c": c,
                     "rmse_log": float(np.sqrt(np.mean(lr ** 2))),
                     "maxres_log": float(np.max(np.abs(lr)))},
        "piecewise": {"pieces_at_5pct": pieces2,
                      "longest_piece": max(e - s for s, e in zip(segs, segs[1:])) if False else None,
                      "maxrel_residual": maxres_lin},
        "aic": {"quad_log": aic_quad, "piecewise_5pct": aic_lin},
        "fp16_log_eps": FP16_LOG_EPS,
        "winner": "quad-log" if aic_quad < aic_lin else "piecewise",
    }, y


def main():
    tables = json.load(open("/tmp/opencode/tables.json"))
    os.chdir(HERE)
    res = {}
    for name, vs in tables.items():
        u = [int(v, 16) for v in vs]
        out, y = fit_table(name, u)
        res[name] = out
        with open(f"table_{name}.csv", "w") as fo:
            w = __import__("csv").writer(fo)
            w.writerow(["idx", "raw_u32", "value_abs"])
            for k, (v, yy) in enumerate(zip(u, y)):
                w.writerow([k, hex(v), yy])
        print(name, json.dumps(out, ensure_ascii=False))
    json.dump(res, open("fit_summary.json", "w"), indent=1)
    print("saved fit_summary.json + table_*.csv")


if __name__ == "__main__":
    main()
