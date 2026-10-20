# 四任务复现指南

统一入口覆盖 ARC-Easy、ARC-Challenge、SciQ、PIQA 和七个模型配置，复现完整循环深度的 LoopCD-Logits。每项保存普通解码、固定引导、自适应引导和第一循环弱参考；不包含 Thinking、隐藏态引导或加速实验。

ARC-Easy 复用已有七项完整结果，新增三个任务的 21 项完整评测也已完成并通过审计。全套共 28 项、44,702 次题目评测、153,048 条候选评分。最终完成标记为 `results/multitask/completion.json` 中的 `complete: true`；各项数值见统一报告和 `summary.json`。常用入口见 [英文 README](../README.md)，目录职责见 [STRUCTURE.md](STRUCTURE.md)。

## 模型和任务配置

模型配置位于 [configs/models.json](../configs/models.json)，任务拆分、示例数、数据路径和主指标位于 [configs/tasks.json](../configs/tasks.json)。`src.registry.resolve_run(task, model)` 合并两者。旧 `configs/arc_all_models.json` 保留历史 ARC-Easy 配置。

| 模型名称 | 官方模型 | 循环次数 | 固定强度 | 自适应上限 | 默认矩阵乘法精度 |
|---|---|---:|---:|---:|---|
| `ouro_14` | ByteDance/Ouro-1.4B | 4 | 0.5 | 1.0 | `high` |
| `ouro_26` | ByteDance/Ouro-2.6B | 4 | 0.5 | 1.0 | `high` |
| `huginn_16` | tomg-group-umd/huginn-0125 | 16 | 0.5 | 0.5 | `highest` |
| `huginn_32` | tomg-group-umd/huginn-0125 | 32 | 0.5 | 0.5 | `highest` |
| `parcae_370` | SandyResearch/parcae-370m | 8 | 0.5 | 1.0 | `high` |
| `parcae_13` | SandyResearch/parcae-1.3b | 8 | 0.5 | 1.0 | `high` |
| `looped_qwen` | Qwen/Qwen3-4B，论文指定循环包装 | 8 | 0.5 | 1.0 | `highest` |

| 任务名称 | 评测拆分 | 题数 | 候选总数 | 示例数 | 预先指定主指标 |
|---|---|---:|---:|---|---|
| `arc_easy` | test | 2376 | 9501 | Ouro 8-shot；其余 0-shot | `acc_norm` |
| `arc_challenge` | test | 1172 | 4687 | Huginn 0-shot；其余 25-shot | `acc_norm` |
| `sciq` | test | 1000 | 4000 | 全部 0-shot | `acc` |
| `piqa` | validation | 1838 | 3676 | 全部 0-shot | `acc_norm` |

四任务均保留 `acc` 和 `acc_norm`。前者对完整候选答案的全部 token 条件对数概率求和后取最大值；后者再除以原答案文本的 Python `len()`，即 Unicode 字符数，不含额外添加的前导空格，**不是 token 数**。平局选择原选项顺序中的第一个最大值。主指标在新任务评分前固定；论文没有逐任务明确区分两种口径，因此这是复现假设，不能按实测结果切换。SciQ 实测 `acc_norm` 更接近论文，但没有据此更换预先确定的 `acc` 主指标。

新增三任务的论文对照值及 Table 2/A2/A4 核对依据保存在 [configs/paper_mc_reference.json](../configs/paper_mc_reference.json)。数组顺序为普通解码、固定引导、自适应引导；论文参照与本地实测分开保存。

## 数据与提示

固定 lm-evaluation-harness 版本为 `d6de81643928d653435c431bae19945d41d32520`，任务 YAML、README 和许可保存在 `sources/lm_eval/`。

- **ARC-Easy / ARC-Challenge**：使用完整官方 AI2 测试拆分。当前题提示为 `Question: …\nAnswer:`。few-shot 示例只从相应训练集抽取，固定 `random.Random(1234)`，按测试集原顺序每题调用 `sample(train, shots)`，随机状态持续推进。示例展示完整正确答案，不依据测试标签选示例。
- **SciQ**：使用官方完整 `test.json`。模板为 `{{support.lstrip()}}\nQuestion: {{question}}\nAnswer:`，包含支持段落。全部 1000 题保留，其中 116 题支持段落为空，仍保留模板开头的换行。选项顺序为 `distractor1, distractor2, distractor3, correct_answer`，目标索引为 3，不打乱选项。
- **PIQA**：使用官方压缩包的 `dev.jsonl` 和逐行对应的 `dev-labels.lst`，即公开有标签的 validation。1838 题的文本、顺序和标签另与作者网站 `valid.jsonl`、`valid-labels.lst` 交叉核对。模板为 `Question: {{goal}}\nAnswer:`，选项为 `[sol1, sol2]`，标签为 0/1。没有使用或推断未公开的测试标签。

所有任务以一个前导空格加完整候选文本作为 continuation，不评分选项字母，不使用聊天模板，也不抽样生成答案。公共数据行包含 `id, question, choices, labels, gold, context, fewshot_ids`；其中 `choices`、`labels` 是字符串列表，`gold` 是整数索引。SciQ 另存原始支持段落，PIQA 保留官方 UUID。

| 任务 | 提示文件 | 协议文件 |
|---|---|---|
| ARC-Easy | `data/arc_easy_{0,8}shot.jsonl` | `data/tasks/arc_easy/protocol_{0,8}shot.json` |
| ARC-Challenge | `data/tasks/arc_challenge/{0,25}shot.jsonl` | `data/tasks/arc_challenge/protocol.json` |
| SciQ | `data/tasks/sciq/0shot.jsonl` | `data/tasks/sciq/protocol.json` |
| PIQA | `data/tasks/piqa/0shot.jsonl` | `data/tasks/piqa/protocol.json` |

协议记录来源、固定哈希、提示和选项规则、主指标及完整性检查。SciQ 许可为 CC-BY-NC-3.0；PIQA 作者数据页注明 AFL-3.0。许可或作者页面保存在对应 `data/raw/` 子目录。官方压缩包没有不可变版本号时，以完整 SHA256 固定字节内容。

准备文件保留完整上下文。评测时仅在超过模型窗口时按固定 harness 规则从左侧截去上下文，完整保留答案 token 及其预测位置；答案本身无法容纳时直接报错。`run_manifest.json` 的 `truncation_details` 记录受影响题目、候选及删除数量。本次全部 28 项正式评测均未发生上下文截断；真实 tokenizer 的长度预检保存在各新增任务的 `tokenizer_lengths.json`。

## 模型读出与数值设置

引导公式为 `z′ = zR + ω(zR − z1)`；自适应强度为 `ω = cap × (1 − (pR,top1 − pR,top2))`，概率在完整词表上计算。弱参考来自第一次完整循环更新，经该模型完整输出路径：Huginn 的 coda、Parcae 的投影和 coda、Looped-Qwen 的全部后续层。

Huginn 和 Parcae 保留官方随机初态分布，按题目 ID 和固定种子生成初态，使同题候选、引导方法及 Huginn 两种深度共享相同位置前缀。论文未指定随机流，这属于显式记录的复现假设。Huginn 加入一次官方 BOS，其余模型不额外添加 BOS。

Looped-Qwen 使用 36 层 Qwen3-4B：前 15 层、索引 `15:19` 的 4 层窗口、后 17 层；执行 `h ← (7/8)h + (1/8)g(h)` 共 8 次，不添加 LoopCD 附录未指定的 anchor。上游包装仓库获取时返回 404，来源和实现假设见 [evidence.json](../sources/looped_qwen/evidence.json)。

默认计算使用 FP32、SDPA；logits 引导和 log-softmax 也使用 FP32。矩阵乘法默认设置来自真实权重数值检查：`high` 允许较快内部实现，`highest` 使用最高精度。实际设置写入 manifest。既有决策见 [precision_decisions.json](../results/arc_all_models/precision_decisions.json)，没有按准确率选择精度。

新增任务中，七项 Ouro/Parcae 配置在 `high` 下未通过正式评分前的批量/单条一致性检查。仅这些配置改用 `--matmul-precision highest` 重试，均通过原始检查并完成正式评分；原容差、权重、提示、种子与引导强度保持不变。逐任务决策、失败日志哈希和“重试前无正式评分”记录见 [新任务精度记录](../results/multitask/precision_decisions.json)。记录中的重试申请文字保留为当时事实，最终接受证据是各项 `validation.json`、`run_manifest.json` 及完整审计。

`make submit` 自动应用该任务/模型的已验证精度；需要显式覆盖时使用 `MATMUL_PRECISION=highest`。直接调用 `scripts/evaluate.py` 或 `sbatch scripts/run_suite.slurm` 时，应自行传入相应的 `--matmul-precision`，不能假设它们会读取任务级精度决策。

## 准备与提交

本地继续使用唯一保留的 `venv/` 环境。新克隆先按 [英文 README](../README.md) 创建环境和安装依赖；版本信息见 [ENVIRONMENT.md](ENVIRONMENT.md)。以下命令从仓库根目录执行：

```bash
make check-env
make download-models
make prepare-data TASK=all
make test
```

数据准备、审计和报告的 `TASK` 默认 `all`，也可指定 `arc_easy`、`arc_challenge`、`sciq` 或 `piqa`。例如 `make prepare-data TASK=sciq`。`make download-models` 获取或校验六份固定权重，合计约 37 GiB；模型文件不随 Git 仓库分发。CPU 测试需要真实 tokenizer 和原生配置/源码；完整结果审计还会检查权重文件是否存在及大小，因此新克隆需先完成下载。审计不在 CPU 上加载完整模型。

**仅允许在 `scavenger-h200` 分区继续评测，每个作业一张 H200。不得改用其他分区绕过排队。** 此约束已写入 [AGENTS.md](../AGENTS.md)。`make submit` 显式指定 partition/account 为 `scavenger-h200`、qos 为 `normal`；`scripts/run_suite.slurm` 也检查实际分区，不符合时退出。每作业申请 6 个 CPU、64 GB 主机内存、最多 60 分钟，模型和数据齐备后离线评分。Slurm 包装使用 `LOOPCD_ROOT`（若提供）或 `SLURM_SUBMIT_DIR` 定位仓库，不依赖原工作站的绝对路径；默认从仓库根目录提交。

新一轮独立重跑示例：

```bash
make submit TASK=sciq MODEL=ouro_26 \
  OUTPUT=results/repeats/sciq/ouro_26
```

提交必须指定具体任务，`TASK=all` 不能作为单个 GPU 作业。需要额外评测参数时仍使用允许的默认分区：

```bash
mkdir -p results/multitask/logs
sbatch scripts/run_suite.slurm \
  --task piqa --model huginn_32 \
  --output results/repeats/piqa/huginn_32
```

统一 Python 入口是 `scripts/evaluate.py --task <任务> --model <模型>`，应在已分配 GPU 的作业中运行。`--name` 兼容 `--model`。`--limit 0` 默认评测完整拆分；正数 `--limit` 试跑必须显式指定独立 `--output`。不指定输出时写入 `results/multitask/<任务>/<模型>/`。

`scripts/evaluate_arc_suite.py` 是兼容包装，默认仍为 ARC-Easy，并转交同一逻辑。`evaluate_arc.py`、`analyze_results.py`、`run_arc.slurm` 保留首轮 Ouro 历史入口；新任务使用统一入口。

以后若有部分结果需要续跑，保持原任务、模型、数值参数和输出目录；评测器先核对数据、协议、执行源码、原生实现及已保存候选，再处理未完成项。本次 28 项已有完整结果应保留，不重复当作待跑任务。实际分区历史、作业替换和调度信息保存在 `results/multitask/jobs.json`；历史记录不构成使用其他分区的授权。缺少必要来源字段的旧 manifest 不能直接续跑，应使用独立新目录。历史结果按实际保存快照审计，不要求等于后续修改的当前源码。

## 审计与报告

统一分析入口为 `scripts/analyze.py --task all|<任务>`：

```bash
make audit TASK=all
make report TASK=all

# 也可只检查或汇总一个任务
make audit TASK=sciq
make report TASK=sciq
```

`make audit` 使用只读 `--partial` 分析，把 JSON 写入 `results/maintenance/latest_audit.json`，随后要求所选结果全部完成；有缺失时不会显示“审计通过”。评测进行中可直接查看：

```bash
venv/bin/python -B scripts/analyze.py --task all --partial
```

`--partial` 不重写正式报告。完整任务通过核验后，报告位于 `results/multitask/<任务>/comparison.json`、`REPORT.md`；总汇总位于 `results/multitask/comparison.json`、`comparison.csv`、`REPORT.md`。总汇总只包含本次 `--task` 选择的任务：单任务命令会重写总索引为该任务，四任务总表需用 `TASK=all` 生成。分析读取注册的默认结果位置，`results/repeats/` 不会自动替换正式来源。

ARC-Easy 直接复用 `results/arc_easy/` 的 Ouro-1.4B 和 `results/arc_all_models/<模型>/` 的其余六项结果，保留其原始数值与来源。每个新运行目录保存候选 token 分数、逐题预测、summary、manifest、实际源码快照、协议快照和真实权重验证。

独立分析从官方原始数据核对完整题目和提示，重新分词，重算候选求和、字符归一化、预测和准确率，并检查重复、缺失与来源。配对改对/改错、置信区间及显著性须按具体报告的统计范围解读；单次固定随机流不能代表跨种子的稳定性。

四种策略共享循环轨迹，记录的是联合实验耗时，不能据此计算 LoopCD 单独加速比。最终数值以已完成的审计报告为准；结果有升有降，不能概括为全部任务获得提升。

## 自动收尾

本轮评测结束后的依赖作业已完成 CPU 测试和四任务完整审计，未申请 GPU。历史依赖关系保存在 `results/multitask/finalization_job.json`。任一评测不完整或检查失败时，收尾脚本写入 `completion.json` 的错误信息和 `complete: false`；只有全部 28 个配置通过后才生成最终报告、CSV 和文件哈希清单。

整理仓库或重建报告后可手动执行 `make finalize`。输出为 `results/multitask/completion.json`、`artifact_manifest.json`、`tests_final.log`，以及前述统一报告。后续如果替换或重提交失败的 GPU 作业，应同步更新收尾作业的依赖，或在所有结果齐备后手动收尾。

发布清理移除了过时进度 JSON、重复备份、早期试跑和无需保留的日志。正式评分、预测、源代码快照、恢复历史和精度决策所依赖的失败日志保留。SciQ/PIQA 原始 ZIP 是哈希核验的正式来源，也必须保留或通过数据准备命令重新获取。
