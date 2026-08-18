# 峰级增强策略审计：删除峰 / 掩盖峰 / 加噪声 —— 到底是什么、不足在哪

> 目的：把"进一步扩大那 ~1–2pp 困难样本改善"所依赖的峰级增强策略读透，
> 定位到代码，找出不足，提出改进。本文只做分析与建议，**不承诺"必然提高"**。

## 一句话结论

当前所谓"删除峰 / 掩盖峰 / 加噪声"其实是**两条独立脚本 + 三类干预**，其中
**"加噪声"在训练侧并不存在**（只有验证侧的扰动压力测试）。真正的两个训练策略是：

| 用户叫法 | 代码实际做法 | 文件 |
|---|---|---|
| 删除峰（反事实） | 把 query 谱的"identity 峰 / confounder 峰 / 随机对照峰"强度置 0，并施加**方向性**反事实损失 | `tasks/train_counterfactual_dreams.py` |
| 掩盖峰 | 把 anchor/positive 中"相对另一方唯一的峰"按强度置 0 | `tasks/train_causal_chemmask_head.py` `mask_unique_peaks` |
| 加噪声 | **训练代码里没有**；只有 `pilot_rule_noise_stress.py` 等**验证**压力测试 | 见 §3 |

---

## 1. 策略一：反事实"删除峰"（train_counterfactual_dreams.py）

这是产出 `best_counterfactual.pt`（head/seed_20260813）的脚本，也是"2pp"的直接来源。

### 1.1 数据侧：每个 query 生成 4 个视图（`CounterfactualDataset.__getitem__`）
- `clean`：原始 query 谱。
- `identity_masked`：删掉"identity 峰"——与**同分子 positive** 匹配上的峰（`target_tokens(..., 0.005)`，5 mDa 容差）。
- `confounder_masked`：删掉"confounder 峰"——与**不同分子 negative** 匹配上的峰。
- `random_masked`：删掉"随机对照峰"——`matched_control` 用 log-intensity + 强度 + m/z 距离做最优匹配 + `gumbel(0, 0.015)` 抖动，选一批**强度相近**的峰删掉。

### 1.2 损失侧：方向性因果约束（`batch_objective`）
设 `margin = cos(q,pos) - cos(q,neg)`：
- `triplet = relu(margin_clean)`：标准三元组。
- `identity_cf = relu(cf_margin + identity_margin - clean_margin)`：**删"真峰"→ margin 必须下降**（下降幅度 ≥ 0.02）。
- `confounder_cf = relu(cf_margin + clean_margin - confounder_margin)`：**删"假峰"→ margin 必须上升**。
- `preserve`：与官方 teacher 的余弦保真（权重 5.0，最强）。
- `random_consistency`：删随机对照峰 → embedding 不变（`cos(q, qr)→1`）。
- `sample_weight`：`transition ∈ {fixed_oof, residual_wrong}` 的"困难"样本权重 1.5。

**本质**：用"删峰是否让 margin 往正确方向移动"作为监督，让模型学会"真峰支撑正例、假峰误导负例"。

---

## 2. 策略二：掩盖峰 + 困难负例（train_causal_chemmask_head.py）

这是较新的"ChemMask"脚本（`causal_chemmask_head` / `strict_counterfactual_full_cpu`）。

### 2.1 掩盖峰 `mask_unique_peaks`
对 anchor/positive 二选一（50/50），把该侧**相对另一侧不匹配（0.02 Da）的"唯一峰"**按强度从高到低置 0：
```
capacity = min(#唯一峰, max_peaks=12, ceil(0.3 * #峰), #峰 - 3)
```
只掩**一侧**，采样率 `identity_mask_prob=0.3`。

### 2.2 困难负例 `shared_major_score`
以 0.5 概率从 `negative_probe_size`（8 或 32）个随机负例里，选 `shared_major_score`
最高者 = `min(共享强度占比) + top-10 重叠`。即选"与 anchor 共享峰最多"的负例当难负例。

### 2.3 损失
`triplet + λ_preserve·(1 - cos(current, teacher))`，backbone 冻结，只训 1024×1024 head。

**本质**：FN 方向（掩唯一峰→忽略条件特异峰）+ FP 方向（共享峰难负例→拒绝撞脸分子），
但**没有方向性反事实损失**，只是普通三元组 + 数据增强。

---

## 3. "加噪声"在哪？

全局搜 `noise/噪声/perturb/gaussian/jitter/random_intensity` 后确认：
- **训练侧（train_*）没有任何强度/mz 噪声注入**。
- "噪声"只出现在**验证侧**：`pilot_rule_noise_stress.py`（"intensity-proportional m/z
  masking, 30% mask fraction, precursor protected, mask token -1"）、`run_external_peak_mask_stress.py`、
  `run_large_targeted_peak_occlusion.py`、`run_residual_pair_peak_occlusion.py`。
- 最接近"加噪声"的 `matched_control` 里的 `gumbel(0,0.015)` 只是**匹配分配的抖动**，不是加到谱上。

> 关键发现：DreaMS 预训练用的正是"强度正比 m/z 掩码 + mask token -1"（见
> `pilot_rule_noise_stress.py:548`），而**微调完全没有复现这套输入扰动**，存在分布漂移。

---

## 4. 不足（按严重度排序）

1. **重归一化污染因果信号**（两脚本都中）。`preprocess_spectrum`（`train_e1_identity.py:99-101`）
   在**置 0 之后**才取 max 归一化，删掉一个高强度峰 → 新 max 下降 → 其余所有峰被放大。
   "删一个峰"实际变成"删一个峰 + 放大其余峰"，反事实不干净。且微调用 `0.0` 当掩码，
   预训练用 `-1.0` 掩码 token，二者不一致。

2. **"加噪声"缺失**。预训练有强度正比掩码 + mask token，微调没有 → 训练/推理分布漂移，
   直接对应错误图谱里的 FN 机制（仪器/CE 特异峰）。

3. **反事实损失是单边下界，且与评估指标错配**。`identity_cf/confounder_cf` 只在方向效应
   **太小**时惩罚，不奖励更大效应、也不惩罚错误方向的超大效应；而 checkpoint 选择用
   **符号准确率**（`identity_effect>0`），优化的是**阈值化幅度**（cf_margin=0.02），二者不一致。

4. **反事实峰集合来自预计算 CSV + 5 mDa 容差**，噪声大、覆盖小，且不随 10-ppm pool 动态更新。
   反事实信号只作用在小规模 curated split 上，没有铺到全量 pool。

5. **掩盖单边、低采样率，与真实检索不匹配**。只掩 anchor/positive 之一，真实检索里
   query 和候选**双方**都各有条件特异峰；且 `identity_mask_prob=0.3`，70% 时间无掩盖。

6. **困难负例弱且离散**。`shared_major_score` 是离散分（top_overlap 最多 10 档），
   只在 8/32 个随机负例里 argmax；没有用 **embedding 空间最近邻**挖难负例。

7. **保真项全局均匀，封顶了增益**。`preserve_weight=5.0` 对 `counterfactual_weight=0.7`
   压倒性占优，embedding 余弦保真 0.9985，Top-1 增益只有 +0.0057——保真把反事实信号压死了。

8. **两条策略未统一**。方向性反事实损失只在 `train_counterfactual_dreams.py`（小 CSV split），
   掩峰 + 难负例只在 `train_causal_chemmask_head.py`（大 pool 但无方向损失）。最强做法是把
   反事实方向损失搬到大 pool、峰集合动态算。

9. **随机对照不"随机"**。`matched_control` 按强度匹配，删的是强度相近的峰（正是模型
   关心的峰），不是中性对照，`random_consistency` 目标被弱化。

---

## 5. 改进方向（按性价比排序，均需消融验证，不保证必然提高）

1. **补上"加噪声"训练增强**（低成本、直接打 FN 机制）：复现预训练的强度正比 m/z 掩码
   （30%、precursor 保护、mask token -1），对 anchor 和 positive 双侧施加；再加小幅
   强度乘法噪声 + 容差内 m/z 抖动。
2. **修重归一化 + 用 -1 掩码 token**：删峰前先记 max，掩码用 -1.0（与预训练一致），
   让"删一个峰"真的等于"少一个峰"。
3. **把方向性反事实损失搬到大 pool + 峰集合动态算**：用 `greedy_peak_matches(0.02)` 对
   每个 anchor-positive-negative 三元组实时划分 shared(identity)/unique(confounder) 峰，
   在 10-ppm pool 上施加 §1.2 的方向损失，把小 CSV split 的信号放大到全量。
4. **embedding 挖难负例**：预计算官方 embedding，选与 anchor 余弦最高的负例当难负例，
   替代/结合 `shared_major_score` 离散 probe。
5. **双侧掩盖 + 提高采样率**：对 anchor 和 positive 各自掩唯一峰，`identity_mask_prob` 上调。
6. **选择性/方向性保真**：只惩罚"伤害检索"的移动、放开"提升反事实顺序"的方向，
   替代全局均匀保真，把被压死的增益放出来。
7. **统一脚本**：把 1+2 合并成一个目标，逐项消融 identity / +hard / +mask / +counterfactual。

---

## 6. 结论边界
- 本文是**读代码后的静态分析**，每一条"不足"都对应具体行号，但"改进是否扩大那 1–2pp"
  **必须靠消融实验测出来**，不能口头保证。
- 下一步建议：先做成本最低、风险最小的 #1+#2（补噪声 + 修重归一化），跑一个 head-only
  smoke，用 500 例困难样本 + 100 硬三元组评估是否正向，再决定是否上 #3/#4/#6。
