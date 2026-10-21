# 实验证据的收录规则

本目录下 `.gitignore` 里有一条 `results/*`，作用是**防止批量原始数据被误提交**。
实际收录规则是：

| 类别 | 是否进 git | 说明 |
|---|---|---|
| 各实验目录的 `README.md` | **是**（force-add） | 结论、口径、撤回记录、证据索引 |
| 各实验目录的 `analysis*/` | **是**（force-add） | `report.md` / `summary.json` / `*.csv`，由分析器生成 |
| 文档里直接引用的个别原始样例 | **是**（force-add） | 例如 `board_intra_20261021/wide/w01.csv`、`board_ftrace_20261021/ftrace/t1.txt` |
| 其余逐次运行的原始 CSV / trace | 否（留在本地） | 可由本目录脚本重新生成；体量大（单个目录可达 14 MB / 2300 文件） |

新增证据时：

```sh
# 只提交结论与分析摘要
git add -f research/rknpu-vendor/experiments/results/<实验>/README.md \
           research/rknpu-vendor/experiments/results/<实验>/analysis*
```

## 口径提醒

这些实验只支持 `research/scheduling/REGCMD-TIMING-MEASUREMENT-MODEL.md` 里
逐条标注过证据等级的结论。**per-regcmd timing 未建立，P0/P1 未通过**；
本轮只可验收为"完成归档的可观测性探索"，不可验收为目标能力已实现。
