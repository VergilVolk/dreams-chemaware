# B 锚定双 encoder／多重排器互补性诊断（2026-10-23）

## 裁决

`run_2400157` 只证明当前“专家分数特征差值 + 不感知 B 状态的有界残差头”没有超过 B；它没有证明 Stage-1、Phase-A 或 MSG 缺乏互补信息。相反，严格 nested formula-OOF 逐查询结果显示，这些模块能够纠正一批彼此不完全重叠的 B 错误。失败发生在**如何识别何时采用这些证据**，而不是互补头寸不存在。

B 仍是当前可用排序器，但不能把本轮结果扩大解释为“双 encoder 和 MSG 已被充分检验并应退出统一算法”。

## 证据边界

- 数据：GNPS Gold/Silver 10-ppm，已消费开发集。
- identity-disjoint：10,995 queries；formula-disjoint：5,261 queries。
- 输入：`run_2400157/modeling_nested_oof/candidate_predictions.npz` 及同一次运行的两个 frozen evidence bundles。
- 所有下述 oracle 数字均使用真值，仅用于测量信息上界，绝不是可部署成绩。
- 未访问 Enveda，未训练模型，未选择权重或阈值。

## 模块确有 B 之外的信息

| 面板 | B 错误 | Stage-1 可纠正 | Phase-A 可纠正 | 两 encoder 并集 | Stage-1 独有 | Phase-A 独有 | MSG 在两 encoder 之外独有 | 全部四专家 oracle 并集 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| identity | 1,254 | 231 | 244 | 286 | 42 | 55 | 109 | 438 |
| formula | 534 | 129 | 132 | 152 | 20 | 23 | 55 | 229 |

两 encoder 的真值 oracle 头寸相当于在 B 上最多再增加 `+2.60/+2.89 pp`；MSG 在该并集之外另有 `+0.99/+1.05 pp` 的独有头寸。四个原始专家的不可实现 oracle 上界为 B 上 `+3.98/+4.35 pp`，对应 R@1 `92.58%/94.20%`。这些数值说明继续研究“如何可靠兑现互补信息”有依据，但绝不能作为模型成绩。

## 为什么当前接法失败

### 1. 原始专家不能无条件替换 B

| 专家 | identity：纠正／误导 B | formula：纠正／误导 B | 真值 oracle 决策精度（identity/formula） |
|---|---:|---:|---:|
| Stage-1 | 231 / 554 | 129 / 278 | 29.4% / 31.7% |
| Phase-A | 244 / 575 | 132 / 271 | 29.8% / 32.8% |
| MSG | 277 / 666 | 150 / 350 | 29.4% / 30.0% |
| RRF | 206 / 547 | 108 / 266 | 27.4% / 28.9% |

因此模块有信息不等于应直接投票；真正问题是学习候选级证据可靠性。

### 2. 当前残差头看不到最关键的 anchor 状态

当前 `build_features` 只提供 Stage-1−Noise、Phase-A−Noise、MSG−P2b、RRF−P2b 的 raw/gap/margin 差值和 availability。B 仅在网络输出外部作为加法 anchor，网络没有直接看到 B 的候选分数、query gap 或 Top1–Top2 margin。

实测 B margin 本身正是区分“专家纠正”和“专家误导”的主要可观测信号：

| 专家 | `−B margin` AUROC，identity | `−B margin` AUROC，formula | 专家自身 margin AUROC，identity/formula |
|---|---:|---:|---:|
| Stage-1 | 0.647 | 0.694 | 0.534 / 0.521 |
| Phase-A | 0.678 | 0.691 | 0.534 / 0.541 |
| MSG | **0.708** | **0.721** | 0.553 / 0.548 |
| RRF | 0.671 | 0.642 | 0.623 / 0.639 |

这是标签条件诊断，不是已经验证的 router；但它明确指出实现遗漏了一个部署时可观察、且比专家自身置信度更有辨别力的变量。

例如 MSG 原始纠正发生时 B margin 中位数为 `0.215/0.210`，误导发生时为 `0.375/0.357`。当前 complete arm 的 introduced 查询同样集中在较高 B margin（中位数 `0.360/0.346`），说明模型正在不必要地推翻较确定的 B。

### 3. MSG 与 P2b 不是同量纲专家

MSG 是高度饱和的分类概率；identity/formula 的候选分数中位数为 `0.0012/0.0043`，Top1–Top2 margin 中位数约 `0.996/0.999`。P2b 的分数中位数为 `0.153/0.232`，margin 为 `0.589/0.649`。把两套分数摘要先相减再 RMS 缩放，不等于完成概率校准或语义对齐。

因此，`complete_spectral` 的显著退化只能归因于当前 MSG 接法，不能归因于 MSG 没有独有纠错信息。

### 4. 残差上界不是本轮主要瓶颈

complete arm 中达到至少 95% 残差上界的候选仅 `0.168%/0.168%`，出现任一近饱和候选的 query 仅 `1.26%/0.76%`。失败不是模型普遍被 bound 卡住；继续扩大 bound 不会修复核心问题。

### 5. 现有头确实筛掉一部分坏动作，但筛选仍不够准

双 encoder arm 捕获了 `184/102` 个纠正，同时引入 `254/116` 个错误；complete arm 捕获 `260/147` 个纠正，却引入 `530/285` 个错误。纠正占所有正确／错误交换的比例从双 encoder 的 `42.0%/46.8%` 下降到 complete 的 `32.9%/34.0%`。MSG 的加入扩大了覆盖，却没有学会尊重 anchor 可靠性。

## 唯一有依据的下一次修复

下一轮不再换融合器名称，也不扫描阈值。只修复本诊断确认的三个结构缺陷：

1. **显式 anchor conditioning**：将 B 的 candidate score、query-relative gap、Top1–Top2 margin 输入残差模型，使“何时可以推翻 B”成为模型可学习变量，而不是事后门控。
2. **按证据家族校准**：Stage-1、Phase-A、MSG 分别保留自己的 raw/gap/margin；不再对 MSG 与 P2b 直接做分数差。所有尺度和单调校准只在每个 outer-train fold 内拟合。
3. **排名目标直接对齐错误交换**：在相同 nested formula-OOF 下，以 positive-vs-current-top competitor 的候选对损失训练，并同时报告 corrected/introduced；不以 listwise loss 下降替代 Top-1 改善。

固定比较仅需三项：B、当前失败的 complete arm、修复后的 anchor-conditioned family model。若修复模型仍不能在两个面板同时超过 B，停止这条分数级融合线；这时才有依据转向使用 1024 维 embedding 邻域几何的独立表示实验。若胜出，再用同一架构做 encoder、MSG、V2 留一消融，不能先用真值 oracle 选择专家。

## 可复现入口

只读诊断脚本：`tasks/analyze_b_dual_multireranker_complementarity.py`。

输出：`data/validation/B_dual_encoder_multireranker_run_2400157/modeling_nested_oof/complementarity_audit_v2.json`。

