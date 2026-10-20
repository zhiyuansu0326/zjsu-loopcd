# 代码库结构

模型、任务和评分逻辑分别维护。ARC-Easy、ARC-Challenge、SciQ、PIQA 使用相同数据行格式、模型适配器和评测入口。四任务 × 七模型配置共 28 项完整评测均已完成并通过独立审计，覆盖 44,702 次题目评测和 153,048 条候选评分。原 ARC-Easy 七项完整结果保留原位置；最终状态见 `results/multitask/completion.json`。

```text
zjsu-loopcd/
├── README.md                    项目概览和常用入口
├── AGENTS.md                    GPU 分区、环境和结果保留约束
├── Makefile                     环境、数据、测试、GPU 作业、审计和报告入口
├── requirements*.txt            核心依赖和环境版本快照
├── docs/                        环境、复现流程和目录说明
├── configs/
│   ├── models.json              七模型配置；循环、引导强度、数值默认值
│   ├── tasks.json               四任务拆分、示例数、数据路径和主指标
│   ├── paper_mc_reference.json  新增三任务的论文数值及核对依据
│   └── arc_all_models.json      历史 ARC-Easy 配置
├── src/                         注册接口、数据构造、适配器、评分和完整性检查
├── scripts/                     下载、统一准备/评测/汇总及 Slurm 入口
├── tests/                       CPU 功能、数据模板和完整性测试
├── data/
│   ├── raw/                     官方原始数据、来源及许可
│   ├── arc_easy_{0,8}shot.jsonl  保留历史 ARC-Easy 提示路径
│   ├── protocol*.json           历史 ARC-Easy 协议
│   └── tasks/
│       ├── arc_easy/            protocol_0shot.json、protocol_8shot.json
│       ├── arc_challenge/       0shot/25shot.jsonl、protocol.json、长度预检
│       ├── sciq/                0shot.jsonl、protocol.json
│       └── piqa/                0shot.jsonl、protocol.json
├── models/                      六份官方权重和版本/哈希清单
├── sources/                     固定官方源码、harness YAML、许可和证据
├── results/
│   ├── multitask/               新运行、调度历史及统一任务/总报告
│   ├── arc_all_models/          原 ARC-Easy 七配置汇总和六项实验
│   ├── arc_easy/                原 Ouro-1.4B ARC-Easy 完整实验
│   ├── repeats/                 显式指定的独立重跑
│   └── maintenance/             运行审计命令时生成的本地输出，不纳入版本控制
├── venv/                        本地唯一 Python 环境，不纳入版本控制
├── .cache/                      运行时生成的本地缓存，不纳入版本控制
└── Liu et al. ... .pdf          原始论文
```

## 当前接口与职责

| 文件 | 职责 |
|---|---|
| `src/registry.py` | `load_registry()`、`resolve_run(task, model)` 合并模型和任务配置 |
| `scripts/prepare_tasks.py` | `--task all|具体任务`，按注册协议准备或核验数据 |
| `src/data_arc_challenge.py` | ARC-Challenge 原始数据、0/25-shot 构造与审计 |
| `src/data_external.py` | SciQ/PIQA 官方源、固定模板；`prepare_external()`、只读 `audit_prepared()` |
| `src/evaluation.py` | 通用数据行检查、候选构造、模型窗口内上下文处理与截断记录 |
| `src/loopcd.py` | LoopCD-Logits 公式和 continuation 编码；保留初次 Ouro 实现 |
| `src/adapter_ouro.py` | Ouro 第一/最终完整循环读出 |
| `src/adapter_huginn.py` | Huginn 随机初态、循环和完整 coda |
| `src/adapter_parcae.py` | Parcae 原生实现加载、随机初态及循环读出 |
| `src/adapter_looped_qwen.py` | Qwen3 八步阻尼循环窗口及完整后续层读出 |
| `src/suite_scoring.py` | 各任务和模型共用的四策略评分与数值检查 |
| `src/run_integrity.py` | 续跑配置、原生来源及已有候选记录的兼容性核验 |
| `scripts/evaluate.py` | `--task`、`--model`，兼容 `--name`，统一 GPU 评测 |
| `scripts/run_suite.slurm` | `scavenger-h200` GPU 作业入口，离线加载权重和数据 |
| `scripts/analyze.py` | 按任务审计原始数据、保存分数与完整结果，生成统一报告 |
| `scripts/finalize_results.py` | 全套测试和四任务审计后生成完成标记、最终哈希并更新 README |
| `scripts/finalize_results.slurm` | 依赖评测结束的 CPU 收尾作业，仍只使用允许分区，不申请 GPU |

`download_*.py` 按固定版本获取各家权重。ARC-Easy 仍由 `prepare_arc.py`、`prepare_arc_zero.py` 构造，统一准备入口调用这两个实现，并在 `data/tasks/arc_easy/` 提供协议路径。

`evaluate_arc_suite.py` 是兼容包装，默认任务为 ARC-Easy。`evaluate_arc.py`、`analyze_results.py`、`run_arc.slurm` 是首轮 Ouro 的历史入口；`analyze_arc_suite.py` 保留旧 ARC-Easy 聚合逻辑。新任务使用 `prepare_tasks.py`、`evaluate.py`、`analyze.py` 和 `run_suite.slurm`。

## 数据合同

公共 JSONL 字段为 `id, question, choices, labels, gold, context, fewshot_ids`。`choices`、`labels` 是字符串列表；`gold` 是零起始整数索引；`context` 已包含任务要求的支持段落或训练集示例。额外原始字段用于追溯，不改变答案顺序。

四任务分别保留 ARC-Easy 2376 题、ARC-Challenge 1172 题、SciQ 1000 题和 PIQA validation 1838 题。SciQ 包含支持段落，116 条空段落题也全部保留。ARC-Challenge 对非 Huginn 模型使用 25-shot，Huginn 使用 0-shot；PIQA 不使用未公开测试标签。完整示例数矩阵见 [REPRODUCING.md](REPRODUCING.md)。

协议记录固定数据哈希、来源、提示模板、选项/标签规则及审计结果。SciQ/PIQA 的 `prepare_external(task, shots=0, seed=1234)` 返回所写 `protocol.json` 的字典；`audit_prepared()` 只读验证官方来源、固定 harness、提示与协议。ARC-Challenge 协议同时记录两种示例数输出。

预先指定 SciQ 主指标为 `acc`，其余为 `acc_norm`，四任务同时保留两者。`acc_norm` 按原始答案字符数归一化，不按 token 数。长输入仅在超过模型窗口时左截上下文，保留全部答案 token，具体截断写入 manifest；准备文件本身完整保留原文。本次完成的 28 项评测均未发生上下文截断。

## 结果与来源记录

新运行默认位于 `results/multitask/<任务>/<模型>/`：

```text
candidate_scores.jsonl    四策略答案 token 分数、目标 ID 和候选总分
predictions.jsonl         原始/字符归一化分数及逐题预测
summary.json              正确数、准确率及本次调用资源
run_manifest.json         任务/模型、协议/源码哈希、数值设置、截断记录
source_snapshot/          实际执行源码与注册配置
validation.json           实际权重上的读出及数值一致性检查
data_protocol.json        此次运行使用的数据协议快照
```

统一任务汇总位于 `results/multitask/<任务>/comparison.json`、`REPORT.md`；总汇总位于 `results/multitask/comparison.json`、`comparison.csv`、`REPORT.md`，只包含本次 `--task` 选择。单任务报告命令也会重写总索引，四任务总表应使用 `TASK=all` 生成。`--partial` 仅打印进行中的审计结果；正式报告要求所选完整结果通过核验。

ARC-Easy 原评分不迁移：Ouro-1.4B 来自 `results/arc_easy/`，其余六项来自 `results/arc_all_models/<模型>/`，由统一分析读取。`results/repeats/` 的独立重跑不会自动替换正式来源。审计依据原执行快照，不以当前源码覆盖历史版本。

所有正式评测的候选记录、预测、清单及源码快照均保留。实际作业、替换和分区历史记录在 `results/multitask/jobs.json`，不删改已发生的调度事实。`resume_history/` 保存被续跑替换的原清单和验证文件；`precision_decisions.json` 引用的七份失败日志仍保留，审计会核验其哈希。

## 清理和维护

发布整理删除了早期试跑、重复文档备份、过时进度与预检输出、重复测试日志和无需保留的调度日志。历史源码快照及正式评分保持原样。旧阶段的文件哈希清单是当时的记录，不能作为当前工作树的完整性清单；当前清单由 `make finalize` 生成。

官方原始数据压缩包属于审计来源，SciQ/PIQA 的只读数据审计会核验它们，不能作为可随意删除的下载缓存处理。六份完整权重和 `venv/` 留在本地；新克隆需要按主 README 安装环境并下载固定权重。

```bash
make download-models
make prepare-data TASK=all
make test
make submit TASK=sciq MODEL=ouro_26 OUTPUT=results/repeats/sciq/ouro_26
make audit TASK=all
make report TASK=all
```

准备、审计、报告的 `TASK` 默认 `all`；提交必须指定具体任务。`make submit` 从已有数值决策记录选取该任务/模型已验证的精度，也可通过 `MATMUL_PRECISION` 显式指定。**继续评测只允许 `scavenger-h200`，每作业一张 H200；不能通过切换分区绕过排队。** `AGENTS.md` 持久化此约束，Makefile 显式指定允许的 partition/account 及 `qos=normal`，Slurm 入口检查实际分区。历史调度记录不改变这一当前约束。

`make audit` 不重写正式结果，但保存审计 JSON，并在所选结果不完整时返回失败。`make report` 从完整保存结果重新生成汇总。增加模型修改 `configs/models.json` 并提供适配器；增加任务修改 `configs/tasks.json` 并提供相同数据合同、协议及原始来源审计。论文参照和本地实测分别保存，新实验使用独立目录保留已有来源链。
