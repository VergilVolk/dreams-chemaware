# Noise-v3 S1a 单峰正交矩阵结果

日期：2026-08-24  
作业：2323835  
状态：`noise_v3_s1a_matrix_validation_passed`

## 1. 实现完整性

- 全量 23,876 个 strict-10ppm 查询；
- 3 个选择器 × 4 个固定剂量全部完成；
- 192,296 个 query-action 配对单元；
- 同一 selector 内 query 集跨剂量完全一致；
- 同一 query-selector 的目标峰与随机对照跨剂量完全复用；
- 前体 token 保护、角色方向、严格同角色敏感性和共享 eligibility 比较全部通过；
- 官方权重、候选图、HDF5、embedding cache 与核心脚本均有哈希记录。

## 2. 全量动作结果

| 选择器 | 衰减 | 覆盖 query | 修正 | 新增错误 | 净修正 |
|---|---:|---:|---:|---:|---:|
| candidate-gradient | 25% | 20,876 | 70 | 51 | +19 |
| candidate-gradient | 50% | 20,876 | 138 | 113 | +25 |
| candidate-gradient | 75% | 20,876 | 198 | 258 | -60 |
| candidate-gradient | 100% | 20,876 | 288 | 560 | -272 |
| role-confounder | 25% | 6,114 | 7 | 2 | +5 |
| role-confounder | 50% | 6,114 | 23 | 6 | +17 |
| role-confounder | 75% | 6,114 | 41 | 9 | +32 |
| role-confounder | 100% | 6,114 | 99 | 18 | **+81** |
| role-identity | 25% | 21,084 | 15 | 31 | -16 |
| role-identity | 50% | 21,084 | 25 | 80 | -55 |
| role-identity | 75% | 21,084 | 39 | 140 | -101 |
| role-identity | 100% | 21,084 | 43 | 503 | -460 |

这里的“修正/新增错误”是动作直接施加在 clean query 后的变化，不是模型训练结果。

## 3. 三个决定性发现

### 3.1 候选梯度具有纠错能力，但不能全局执行

在 1,605 个 baseline-wrong 查询上：

- 25% 修正 70；
- 50% 修正 138；
- 75% 修正 198；
- 100% 修正 288。

四个剂量的 target-minus-random margin 与 Top-1 方向均为正，说明梯度选择不是随机效应。但 baseline-correct 的损害随剂量迅速增加：51、113、258、560。候选梯度适合作为“需要学习何时执行”的候选动作，不适合作为全局增强。

### 3.2 role-confounder 是当前最干净的固定方向

错误候选专属峰的四个剂量均保持正净收益，且随剂量增强：+5、+17、+32、+81。100% 删除时：

- baseline-wrong：99 个修正；
- baseline-correct：18 个新增错误；
- near：75 个修正、15 个新增错误，净 +60；
- target-minus-random 的 margin 与 Top-1 在 identity/formula bootstrap 下均为正。

严格同角色对照子集中结果仍保持方向：100% 时 baseline-wrong 修正 37 个、baseline-correct 新增 6 个，说明主结果并非完全由对照角色回退造成。

### 3.3 identity-only 方向负对照通过

50% 衰减的 target-minus-random margin 为 -0.00807：

- identity-cluster CI：[-0.00915, -0.00758]；
- formula-cluster CI：[-0.01065, -0.00850]。

随剂量增加，净损害由 -16 扩大到 -460。正确身份专属峰确实承载正向身份信息。这验证了角色定义与实验方向敏感性，也解释了过去粗暴删峰为何容易破坏同分子聚集。

## 4. 对原假设的裁决

1. “随机或整组删峰能够纠正 DreaMS”不成立；
2. “任何梯度最高峰都可删除”不成立，强剂量会大量伤害正确查询；
3. “候选关系决定的峰角色包含可利用方向信号”成立；
4. “错误候选专属峰可作为定向训练视图”得到当前最强支持；
5. 直接全局执行动作仍不能进入训练，必须先解决动作选择与 clean-view 迁移。

## 5. 下一步锁定

### S1b：动作 oracle 与可学习性上限

在 no-op 加 12 个动作中计算每个 query 的最大可修正头寸、动作重叠和互补性。该步骤回答这些动作组合是否足以支持至少 1–2 个百分点的 clean 检索提升。

### S2：非线性动作策略

策略只允许从 `no-op / role-confounder / candidate-gradient` 中选择。输入只能使用动作前可得信息；按分子式 OOF 评价。规则特征仅作为 with/without-rules 增量消融。

### S3：query-side peak adapter

将通过策略选择的纠错视图作为 stop-gradient teacher，训练学生在原始 clean query 上复现纠正后的候选排序；安全流大量保留 baseline-correct 查询。禁止只在被删峰谱图上优化而不迁移回 clean 表征。

本轮还不能宣称模型性能提高，但已经从“盲目加噪”推进到“峰角色方向成立、剂量风险明确、存在可学习动作空间”。
