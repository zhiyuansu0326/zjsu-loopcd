# 当前环境

项目只保留根目录的 `venv/`。它由实际完成七组实验的 `venv-stage3c-h/` 原地改名而来，未升级或重装包；未使用的 `venv-cpu/` 已删除。

| 项目 | 当前值 |
|---|---|
| Python | 3.12.9 |
| PyTorch 运行时 | 2.6.0+cu124 |
| CUDA 构建版本 | 12.4 |
| Transformers | 4.56.2 |
| 核心依赖 | [requirements.txt](../requirements.txt) |
| 全部已安装包版本 | [requirements.freeze.txt](../requirements.freeze.txt) |

使用现有环境：

```bash
cd /work/zs220/zjsu-loopcd
source venv/bin/activate
python --version
deactivate

# 不激活也可以直接使用，Makefile 默认采用这个解释器。
venv/bin/python -B scripts/evaluate_arc_suite.py --help
make check-env
```

改名时已修正命令入口的 shebang、三种激活脚本、提示名称及 `pyvenv.cfg` 中的路径，并同步更新相关安装记录的文件哈希。基础 Python 仍来自本机 `/admin/apps/rhel9/miniconda3/bin/python3.12`，因此此目录不是可直接搬到其他机器的独立 Python 发行版。

交互节点没有分配 GPU 时，`make check-env` 显示 `CUDA visible: False` 是正常的；完整评测应通过 Slurm 申请 GPU。CPU 单元测试与保存结果的复核不需要 GPU。

## 需要在其他机器重建时

当前目录无需安装任何包。以下仅用于在兼容的 Linux、Python 3.12、NVIDIA 驱动环境中重建；应先安装 CUDA 12.4 的 PyTorch，再安装其余固定依赖：

```bash
python3.12 -m venv venv
venv/bin/python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
venv/bin/python -m pip install -r requirements.txt \
  -c requirements.freeze.txt
venv/bin/python -m pip check
```

CUDA 12.4 索引仅用于 PyTorch 安装步骤；其余包使用正常的包索引。依据：[PyTorch 2.6.0 官方安装说明](https://pytorch.org/get-started/previous-versions/#v260)。

`requirements.freeze.txt` 是当前安装版本的快照，不包含 wheel 哈希或原始下载索引，不能单凭它保证另一台机器数值逐位相同。正式实验还记录了模型版本、数值精度、随机初态、GPU 和代码快照。

Parcae 使用保存在 `sources/parcae/` 中的官方推理代码与 SDPA，无需安装其完整训练依赖。测试使用标准库 `unittest`。
