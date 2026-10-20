# 结果目录

四任务 × 七模型配置共 **28 项完整评测均已通过审计**，合计 44,702 次题目评测、153,048 条候选评分。统一入口为 [四任务报告](multitask/REPORT.md)、[机器可读结果](multitask/comparison.json) 和 [CSV](multitask/comparison.csv)。

| 目录 | 内容 |
|---|---|
| `multitask/` | 四任务统一审计、汇总；三个新任务各七配置的正式评测 |
| `multitask/jobs.json` | 新任务的 Slurm 提交及实际分区记录 |
| `multitask/finalization_job.json` | 评测结束后的自动测试、审计与报告作业依赖 |
| `multitask/completion.json` | 最新测试和完整审计的完成状态；失败或不完整时为 false |
| `multitask/artifact_manifest.json` | 最新完整审计生成的文件清单与 SHA256 |
| `multitask/tests_final.log` | 最新收尾过程的 CPU 测试输出 |
| `multitask/precision_decisions.json` | 正式评分前的数值精度决策及失败日志哈希 |
| `multitask/resume_history/` | 续跑前原始 manifest 和 validation 来源记录 |
| [arc_all_models/REPORT.md](arc_all_models/REPORT.md) | ARC-Easy 七配置的正式汇总、论文对照和统计 |
| [arc_easy/REPORT.md](arc_easy/REPORT.md) | 复用的首轮 Ouro-1.4B 完整结果 |
| `maintenance/` | 执行只读审计命令时生成的本地输出，不纳入版本控制 |

七项 ARC-Easy 实验的数据与源码快照保留原位置。新任务统一存放于 `multitask/<task>/<model>/`，每组保存完整候选 token 分数、逐题预测、配置、数据协议和执行源码快照。旧环境名称在历史记录中保持原样，当前环境路径为 `../venv/`。`repeats/` 用于用户显式提交的独立重跑，不会自动替换正式结果。

发布整理删除了早期试跑、重复文档备份、过时进度和预检输出、重复测试日志及无需保留的 Slurm 日志。正式结果与来源快照、实际作业/分区历史、恢复历史均保留。`multitask/logs/` 中精度决策引用的七份失败日志是审计证据，不是可删除的临时日志。历史 `artifact_manifest.json` 仅代表其创建时的文件状态；当前完整性清单位于 `multitask/artifact_manifest.json`。

从项目根目录执行 `make audit TASK=all` 可复核四个任务，或用 `TASK=arc_easy` 单独检查旧任务。新克隆先按主 README 安装环境、下载固定权重并准备数据。输出写入 `maintenance/latest_audit.json`，不会改写原始评分或正式报告；最终审计要求选定任务的全部七配置完成。`make report TASK=all` 重建统一报告，`make finalize` 运行测试、完整审计并更新完成状态和当前文件清单。
