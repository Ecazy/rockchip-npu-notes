# 上游独立组件源码

获取日期：2026-10-09。按 [上游使用指南](../../guide/README.md) 的组件列表获取 `gregordinary` 下的 6 个仓库，作为独立 Git checkout 保存在当前目录。

每个仓库采用默认分支的 `--depth 1` 浅克隆，包含该 commit 的源码树，不包含完整历史。获取时全部工作树干净，均无 `.gitmodules`。本次未编译、安装或运行组件，也未获取各前端的第三方构建依赖。

| 仓库 | 用途 | 获取 commit |
| --- | --- | --- |
| [rocket-userspace](rocket-userspace/README.md) | 用户态驱动、librocketnpu、matmul/算子库与 graph 实现 | `f86cf52c666b` |
| [ggml-rocket](ggml-rocket/README.md) | ggml 后端，LLM/Whisper 对接 | `550986d2212e` |
| [tflite-rocket](tflite-rocket/README.md) | TensorFlow Lite delegate，CNN/检测与 Frigate 对接 | `d4c00ee8ce88` |
| [ort-rocket](ort-rocket/README.md) | ONNX Runtime execution provider，视觉及文本编码器对接 | `2d1136d68b8b` |
| [rknpu-submit](rknpu-submit/README.md) | 厂商 rknpu 内核驱动提交适配层 | `6b144b19bb67` |
| [patches](patches/README.md) | 内核与平台补丁；本指南重点为 rocket/ 子目录 | `7b945e71fd55` |

完整 URL、commit、分支、commit 时间及浅克隆状态见 [provenance.json](provenance.json)。获取的是 2026-10-09 可见的默认分支版本，未假定与笔记导入 commit 对应同一时刻。

## 目录与构建关系

6 个组件互为同级目录；本目录的 `patches/rocket/` 已随完整 patches 源码树获取。上游指南假设笔记仓库也与组件同级，本地查阅时从本文件进入即可。构建时按各组件 README 设置源码、安装前缀和依赖路径。

仓库目录已被主笔记仓库的 `.gitignore` 排除，避免把独立 Git 仓库意外纳入提交；本索引和来源清单正常跟踪。换机器时可用清单恢复同一 commit：

```bash
python3 - <<'PY'
from pathlib import Path
import json, subprocess
base = Path("research/upstream-components")
manifest = json.loads((base / "provenance.json").read_text())
for repo in manifest["repositories"]:
    dest = base / repo["name"]
    if dest.exists():
        print(f"skip existing: {dest}")
        continue
    subprocess.run(["git", "init", str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", repo["url"]], check=True)
    subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", repo["commit"]], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--detach", "FETCH_HEAD"], check=True)
PY
```

以上命令在笔记仓库根目录执行。已有目录会保留，不自动覆盖或切换版本。
