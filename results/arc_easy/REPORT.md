# Ouro-1.4B × ARC-Easy：LoopCD 功能复现

完整测试集 2376 题。唯一模型为 ByteDance/Ouro-1.4B base；8-shot 示例来自训练集，seed=1234，4次完整循环。
无训练、无参数搜索；固定采用论文 Table A4 的 ω=0.5、自适应上限1.0。所有策略使用相同问题、示例、权重和循环轨迹。

## 实测与论文

下表预先选择 acc_norm 作为展示口径（答案条件对数似然除以答案文本字符数）。论文只明确 likelihood scoring，未明确写出 ARC-E 列的归一化口径，因此同时保留未归一化 acc，不能声称是作者原评测代码的逐位复现。

| 策略 | 答对数 | 本次 acc_norm | 论文 Table 2 | 本次 acc |
|---|---:|---:|---:|---:|
| 普通解码（第4次循环） | 1998/2376 | 84.09% | 84.09% | 83.80% |
| LoopCD 固定强度 ω=0.5 | 2005/2376 | 84.39% | 84.39% | 84.43% |
| LoopCD 自适应上限=1.0 | 1999/2376 | 84.13% | 84.26% | 84.55% |
| 弱参考（第1次循环） | 1386/2376 | 58.33% | 未比较 | 64.27% |

## 同题配对变化

| 策略 | 改对 | 改错 | 净增答对 | 提升百分点 | 配对 bootstrap 95%区间 | McNemar p |
|---|---:|---:|---:|---:|---|---:|
| LoopCD 固定强度 ω=0.5 | 38 | 31 | +7 | +0.29 | [-0.38, +0.97] | 0.4704 |
| LoopCD 自适应上限=1.0 | 48 | 47 | +1 | +0.04 | [-0.76, +0.84] | 1.0000 |

区间使用固定种子1234、20000次同题配对重采样；它描述测试题抽样不确定性，不是多次模型运行的波动。单模型单任务的结果不能推广为整篇论文所有模型、生成任务或降循环速度结论。

本次两种 LoopCD 的 acc_norm 改善区间都包含零；这次实测提升较小，不能据此认定存在稳定、显著的准确率提升。

## 实现和验证

官方 Ouro 在每次完整24层循环后执行 RMSNorm，再继续下一循环。这里直接读取官方模型返回的第1/第4次循环状态，通过共享 lm_head 得到 z₁、z₄，以 z′=z₄+ω(z₄−z₁) 评分。自适应 ω=1−(p₄,top1−p₄,top2)，概率在完整词表上计算。
模型计算使用 float32（FP32 matmul precision=high），FP32 logits 运算和 log_softmax；对每个候选答案的全部 token 做 teacher-forced 条件似然评分。不使用选项字母打分，不截断上下文，不抽样生成。

已检查 ω=0 与官方第4循环 logits 一致、早晚循环输出不同、单条/补齐批量输入一致性，另有微型官方 Ouro 的全答案位置测试。真实模型检查见 validation.json，测试源码见 ../../tests/test_loopcd.py。

数值预检中，BF16 的单条/批量答案 token 对数概率最大差异为0.28756，未用于正式结果。正式配置的同项检查误差为零。FP32 matmul precision=high 允许较快的内部矩阵乘法实现，不等同于强制最高精度的 highest；这里选择精度依据是数值检查，没有按测试准确率调参。

三种策略和弱参考共用一次4循环计算，因此记录的总耗时是联合实验耗时，不能据此宣称 LoopCD 的单独加速比。

- 模型加载和验证：13.0 秒。
- 本次评分调用：147.9 秒。
- 峰值已分配 GPU 显存：6.66 GiB。

## 直观例子

以下按原测试集顺序分别取每种策略最早两条改对/改错案例；完整逐题结果保存在 predictions.jsonl。

### Mercury_7239733 · LoopCD 固定强度 ω=0.5 · 改对

Meiosis is a type of cell division in which germ cells divide to produce haploid cells. Where does meiosis occur?

- A. brain cells
- B. bone cells
- C. muscle cells
- D. ovary cells

正确答案：D；普通解码：C；LoopCD：D。

### Mercury_7166145 · LoopCD 固定强度 ω=0.5 · 改对

The circulatory system and the endocrine system work together in the human body. Which describes one way in which these systems interact?

- A. taking in oxygen and transporting it to cells of the body
- B. releasing hormones and transporting them to cells of the body
- C. absorbing nutrients from food and transporting them to cells in the body
- D. collecting waste products from the cells and transporting it out of the body

正确答案：B；普通解码：C；LoopCD：B。

### Mercury_SC_405199 · LoopCD 固定强度 ω=0.5 · 改错

Which of these will most likely harm a habitat?

- A. planting trees
- B. water pollution
- C. rainfall
- D. sunlight

正确答案：B；普通解码：B；LoopCD：A。

### Mercury_7146195 · LoopCD 固定强度 ω=0.5 · 改错

A research scientist writes a paper on the initial regrowth of a forest after a fire has damaged the entire ecosystem. Which title would be best for the paper?

- A. Primary Succession
- B. Stable Communities
- C. Climax Communities
- D. Secondary Succession

正确答案：D；普通解码：D；LoopCD：A。

### Mercury_7166145 · LoopCD 自适应上限=1.0 · 改对

The circulatory system and the endocrine system work together in the human body. Which describes one way in which these systems interact?

- A. taking in oxygen and transporting it to cells of the body
- B. releasing hormones and transporting them to cells of the body
- C. absorbing nutrients from food and transporting them to cells in the body
- D. collecting waste products from the cells and transporting it out of the body

正确答案：B；普通解码：C；LoopCD：B。

### Mercury_SC_402980 · LoopCD 自适应上限=1.0 · 改对

During which activity should a student wear goggles?

- A. writing a science report
- B. mixing baking soda with water
- C. measuring the length of a shadow
- D. examining a leaf with a microscope

正确答案：B；普通解码：D；LoopCD：B。

### Mercury_SC_405199 · LoopCD 自适应上限=1.0 · 改错

Which of these will most likely harm a habitat?

- A. planting trees
- B. water pollution
- C. rainfall
- D. sunlight

正确答案：B；普通解码：B；LoopCD：A。

### VASoL_2011_5_34 · LoopCD 自适应上限=1.0 · 改错

Which layer of Earth is the thinnest?

- A. Inner core
- B. Crust
- C. Outer core
- D. Mantle

正确答案：B；普通解码：B；LoopCD：C。

## 可比性限制

论文没有公布完整 few-shot 实例/随机种子、模型精确 revision 和所有数值实现细节。本次采用冻结的官方模型版本、固定的官方 lm-eval 提示构造与采样规则，未以测试结果选择种子或强度。绝对分数的差异可能来自这些未公开细节、数值精度/attention 实现和归一化口径。即便接近论文分数，也只代表这个设置的功能复现。

来源：本地论文第6页 Table 2、第17页 Table A2、第18页 Table A4；[论文](https://arxiv.org/html/2610.02185v1)、[官方模型](https://huggingface.co/ByteDance/Ouro-1.4B)、[官方 lm-eval](https://github.com/EleutherAI/lm-evaluation-harness)。精确版本和文件哈希见 ../../models/model_manifest.json、../../data/protocol.json 和 run_manifest.json。
