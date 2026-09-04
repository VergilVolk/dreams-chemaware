# MTBLS13729 同患者唾液酸游离池—活化供体解耦审计

## 结论先行

在原始补充表所覆盖的 10 对黏液型结直肠癌（Rmu）肿瘤—匹配癌旁组织中，游离
N-acetylneuraminic acid（Neu5Ac）在 10/10 患者中升高，平均配对变化为
`+2.249 log2`；但 CMP-Neu5Ac 与 UDP-GlcNAc 分别仅为 `+0.556` 和 `+0.327 log2`，
均未达到名义显著。更重要的是，在同一患者内直接比较变化量后，游离 Neu5Ac 的升幅
显著大于两个活化供体/上游核苷酸糖节点。

这支持一个比“唾液酸化整体升高”更窄、更可检验的发现：**Rmu 中扩大的游离
Neu5Ac pool 与 CMP-Neu5Ac 活化供体及 UDP-GlcNAc 上游核苷酸糖并不同步**。它为
`free-pool-to-activated-donor decoupling` 提供了同患者证据，但不区分合成、回收、
去 O-乙酰化、糖链释放、摄取或糖链掺入下降，也不建立通量或酶活性。

## 数据与预设比较

- 数据源：原研究 HILIC(-) 代谢物补充表与样本元数据；Rmu 为患者 P21–P30。
- 设计：每位患者的奇数组织号为肿瘤，后续偶数组织号为匹配癌旁。
- 节点：
  - free Neu5Ac：HMDB0000230，Level 1，`m/z 308.09838`，RT `7.185 min`；
  - CMP-Neu5Ac：HMDB0001176，Level 2，`m/z 613.13904`，RT `8.745 min`；
  - UDP-GlcNAc：HMDB0000290，Level 1，`m/z 606.07362`，RT `8.371 min`。
- 两个比较在读入结局前固定：
  1. `delta free Neu5Ac - delta CMP-Neu5Ac`；
  2. `delta free Neu5Ac - delta UDP-GlcNAc`。
- 每个节点和比较均使用患者配对 log2 差值；报告 t 检验、Wilcoxon、sign test 和
  100,000 次 bootstrap。两个主要 Wilcoxon 检验使用 Holm 校正。

## 节点级结果

| 节点 | 身份 | 升高患者 | 平均配对 log2 变化 | bootstrap 95% CI | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| free Neu5Ac | Level 1 | 10/10 | +2.249 | [1.641, 2.866] | 0.00195 |
| CMP-Neu5Ac | Level 2 | 6/10 | +0.556 | [-0.157, 1.410] | 0.275 |
| UDP-GlcNAc | Level 1 | 5/10 | +0.327 | [-0.993, 1.687] | 0.492 |

free Neu5Ac 的平均变化约为 `4.75-fold`，且 sign test p=`0.00195`。另外两个节点的
置信区间跨零，患者方向也不一致。

## 同患者变化差异

| 预设比较 | free pool 增幅更大的患者 | 平均 log2 差异 | bootstrap 95% CI | Wilcoxon p | Holm p |
|---|---:|---:|---:|---:|---:|
| free Neu5Ac − CMP-Neu5Ac | 8/10 | +1.693 | [0.710, 2.633] | 0.0137 | 0.0273 |
| free Neu5Ac − UDP-GlcNAc | 9/10 | +1.922 | [0.527, 3.357] | 0.0273 | 0.0273 |

两个比较的 bootstrap 下界均大于零，且 Holm 校正后的 Wilcoxon p 均为 `0.0273`。
因此结论不是由“一个显著、另一个不显著”的错误逻辑产生，而是由患者内差值直接支持。

## 对论文机制图的影响

原先 `donor–carrier–core–linkage decoupling` 中的 donor 层需要修订：RNA 层面的
GNE/NANS/SLC35A1 富集不能等同于 CMP-Neu5Ac 实际扩增。本审计显示，至少在现有静态
代谢物层面，**free pool 扩张强于 activated donor/precursor pool**。因此主图应区分：

1. free Neu5Ac pool：强同患者升高；
2. CMP-Neu5Ac/UDP-GlcNAc：未同步升高；
3. secretory-mucin carrier：转录背景富集；
4. core/linkage：外部 O-glycomics 支持选择性结构重塑；
5. 这些层之间使用虚线关联，不能画成已证实的实线通量。

## 仍不能回答的问题

- CMP-Neu5Ac 当前为 Level 2，必须用标准品/高质量 MS2 进一步确认；
- 没有 ManNAc、胞质/高尔基体分区或同一样本 glycan destination；
- 静态丰度不能区分 NEU1/3/4 介导释放、CMAS 活化、SLC35A1 转运、摄取或糖链利用；
- 这是同一发现队列的再分解，不是独立 Rmu 患者复现；
- 不能据此声称 flux、enzyme activity、causality 或 global hypersialylation。

## 可复核工件

- 脚本：`tasks/audit_mtbls13729_sialic_donor_decoupling.py`
- 报告：`data/mtbls13729/sialic_donor_decoupling_v1/report.json`
- 患者级明细：`data/mtbls13729/sialic_donor_decoupling_v1/rmu_patient_sialic_donor_deltas.csv`

