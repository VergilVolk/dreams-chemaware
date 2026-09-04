# E4-A 三臂共享 embedding 因果归因：预汇总结果

日期：2026-09-01  
作业：2328357（array 0–2）  
状态：三个训练臂均完整通过；逐 query paired summary 尚未生成，因此本文不是最终因果裁决。

## 1. 协议完整性

- 三臂：clean duplicate、matched random、targeted mature noise；
- held formula fold：0，共 5,923 queries；
- 三臂均为同一官方初始化、同一训练配置、同一候选图；
- train action rows 均为 28,509，train identities 均为 1,562；
- 四轮 action/safety sampling schedule SHA256 三臂完全相同；
- initialization rank mismatch 均为 0；
- R0、candidate graph、官方 checkpoint 和训练脚本 SHA256 完全相同；
- matched-random 两个冻结对照的使用数为 18,437 / 18,497，差异 0.162%；
- P2b 禁止，P3 未使用，推理输入为 clean spectrum。

## 2. 单臂相对 official 的结果

| 训练臂 | Recall@1 | 相对 official | corrected / introduced | risk net | near 增量 | MRR 增量 | preservation |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean duplicate | 0.938038 | +0.5065 pp | 37 / 7 | 23 | +0.5539 pp | +0.3560 pp | 0.995137 |
| matched random | 0.938038 | +0.5065 pp | 37 / 7 | 23 | +0.5539 pp | +0.3560 pp | 0.995125 |
| targeted | 0.938376 | +0.5403 pp | 38 / 6 | 26 | +0.5816 pp | +0.3735 pp | 0.995245 |

注意：这些 corrected/introduced 是各臂相对 official 的转移，不是 targeted 相对 matched-random 的配对转移。

## 3. 已经成立的事实

1. clean duplicate 与 matched random 的 Recall@1、near、MRR 和 37/7 转移完全相同；本轮约 +0.5065 pp 的主体来自三臂共有的 clean/listwise continuation，而不是随机删峰。
2. targeted 相对两个对照的聚合 Recall@1 仅多 2 / 5,923，即 +0.03377 pp；MRR 约多 +0.01745 pp，near 约多 +0.02769 pp。
3. 定向动作在 action view 上确实有效：第 4 轮 targeted 的 action margin 从 clean-view 0.30062 提高到 augmented-view 0.31290；matched-random 则从 0.30197 降到 0.29793。
4. 但 held-action clean-query 子集三臂都为 30 corrected / 2 introduced。现有 action-view 优势几乎没有转移为该子集的 clean-input 排名优势。
5. 三臂的 full-graph mean margin 增量分别为 0.013160、0.013094、0.013134；targeted 没有显示出比 clean duplicate 更大的整体 clean margin 移动。

## 4. 当前科学解释边界

本结果支持“成熟 targeted action 在被修改后的谱图上具有方向性”，但不支持把共享 encoder 的 +0.5403 pp 全部归因于定向噪声。现阶段最符合数据的解释是：普通 clean/listwise continuation 贡献了绝大多数增益，现有损失只把很小一部分 action-specific signal 转移到 clean embedding。

最终 Attribution gate 必须读取三个 `held_per_query.csv.gz`，计算 targeted−matched-random 与 targeted−clean 的 formula-cluster paired CI、配对 corrected/introduced 和 candidate-switch。自动汇总未因共享文件系统完成时序触发，使用只读恢复脚本生成，不重新训练也不覆盖三臂权重。

## 5. 下一步硬边界

- paired CI 未生成前，不做 multifold、LR/layer/epoch 扫描，也不开始 P3；
- 若 targeted−matched-random 的 formula-cluster CI 下界不大于 0，则当前 mature targeted family 未通过共享-encoder 因果归因门；
- 即使失败，也不能解释为峰级动作无效，而应定位为“action view 有效、clean-input transfer 未成立”；随后按既定顺序进入可学习性诊断，而不是扩大曝光或继续 dose 扫描。
