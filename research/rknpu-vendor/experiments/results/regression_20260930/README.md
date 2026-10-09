# 全量回归（第 17 轮末，§7.46 之后）

板端 Orange Pi 5 Plus，driver v0.9.8；`min_conv_exp.c` sha `984755db…`（含 §7.46 `--chain-pc` 与终止约定），
`wgrad_real.c` sha `b14e487a…`（含 §7.41 int8 深通道）。

| 套件 | 结果 |
| --- | --- |
| `exp2/run_fp16_regression.sh` | **pass=79 fail=0 envfail=0 dirtyevents=0** |
| `exp2/run_regression_hard.sh`（int8） | **pass=74 fail=0 envfail=0** |
| `exp2/run_deconv_smoke.sh` | **pass=12 fail=0** |
| `exp3/run_fp16_two_layer_loop.sh` | `CLOSED-LOOP OK` |
| `exp3/run_wgrad_real_loop.sh`（int8） | `CLOSED-LOOP OK` |
| `exp3/run_fp16_wgrad_real_loop.sh` | 前若干相 `CLOSED-LOOP OK`；最后一次 `FWD SUBMIT FAILED (b=0)`（6.4 s ETIMEDOUT = 已知随机 env 失败，单跑复验 `CLOSED-LOOP OK`） |
| 健康探针 `min_conv_exp 3x3p1 ic16 ih16` | `SUBMIT ret=0` + ULP-EXACT（`OUT-HASH 533d6b86`） |
| `min_conv_exp` 切分臂 `iw1024 ic256 tile-auto` | `OUT-HASH e86f99c0` + BIT-EXACT（14 tiles） |
| 楔死签名（`job commit failed` 计数） | **0**（soft reset 累计 27，均为多 task 尝试所致） |

结论：§7.42–§7.46 对发射器/训练工具的所有改动**未破坏既有能力**；多 task / 多核路径仍是唯一
未闭合项（`--chain-id`/`--multi-core`/`--chain-pc` 全部 opt-in，不影响默认路径）。
