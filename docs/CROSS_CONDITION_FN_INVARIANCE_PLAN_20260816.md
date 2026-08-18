# 跨条件同分子分离（FN）系统攻坚方案 —— 机理·评估·数据·方法·层级

> 目标：让"同一分子、不同采集条件（仪器/碰撞能）"的谱图对，在 embedding 空间不再被错误分离。
> 这是四个因果验证机制里效应最大、最"必胜"的一个（§5 效应 0.170，confirmation 95% CI 0.146–0.194）。
> **口径**：本文给的是"有梯度压力、可验证"的方案，不是"必然提高"的承诺。成功与否由一个
> 直接测量条件 gap 的 locked benchmark 判定。

---

## 0. 一句话目标 + 可证伪判据

**目标**：条件（instrument / CE）成为 embedding 的 nuisance，不再决定同分子谱图之间的距离。

**判据（可证伪）**：训练前后，同一批同分子谱图对的
`condition_gap = mean(cos(同分子·跨条件)) − mean(cos(同分子·同条件))`
显著缩小（分子聚类 bootstrap 的 95% CI 上界 < 0），同时"不同分子"相似度不上升。

---

## 1. 机理背景（联网查证汇报）

### 1.1 碰撞能/仪器对 MS2 的影响（FN 的直接来源）
- **碰撞能（CE）对碎片峰强度和跨仪器可复现性有决定性影响**。[JASMS 2016《Quantitative
  Comparison of Tandem Mass Spectra Obtained on Various Instruments》](https://pubmed.ncbi.nlm.nih.gov/27206510/)
  跨 5 家厂商 QTOF/Orbitrap：碎片 m/z 一致性在 **20 eV 最高**，0/10/40 eV 显著下降。
  即"条件特异峰"确实存在，且随 CE 系统性出现/消失——这正是我们错误图谱里 FN 的物理根源。
- [JASMS 2024《Advancing the Prediction of MS/MS Spectra Using Machine Learning》](https://www.x-mol.com/paper/1834053465732788224)
  明确建议：**对 CE 范围/归一化要格外谨慎**，并谨慎筛选数据集——从预测侧印证了"CE 是主要 nuisance"。

### 1.2 现有"跨条件不变表征"方法（直接对口，说明这条路走得通）
| 方法 | 关键机制 | 出处 |
|---|---|---|
| **MVP（多视图对比）** | 用**共识谱作 anchor**，强制同一分子不同仪器条件的光谱得到**一致表征** | [bioRxiv MVP](https://www.biorxiv.org/content/10.1101/2025.11.12.688047v1.full) |
| **VAE 解耦** | 把 CE / 电离模式 / 仪器类型**解耦成独立潜因子**，得"仪器无关"表征 | MDPI 生成建模综述（对应文献） |
| **ChemEmbed** | **跨多 CE 合并光谱** + 中性丢失，CNN→Mol2vec | [SEES:lab ChemEmbed](https://seeslab.info/publications/chemembed-deep-learning-framework-metabolite-identification-using-enhanced-msms-data-and-multidimensional-molecular-embeddings/) |
| **CSU-MS²** | 跨模态对比，覆盖低/中/高 CE，预训练合成+微调实验 | 对应论文 |
| **MSBERT** | 掩码 + 对比（与 DreaMS 同族） | [Anal Chem 2024](https://pubs.acs.org/doi/10.1021/acs.analchem.4c02426) |
| **Spec2Vec** | Word2Vec 峰共现，库检索鲁棒（无专门 CE 定量） | [PLOS CB 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7909622/) |

结论：**"跨仪器/跨 CE 的不变质谱表征"是已发表、可复现的方向**，不是我们发明新范式，而是
把已证方法嫁接到 DreaMS 基座上。

### 1.3 通用"强制不变性"工具（必然性压力的来源）
- **DANN / 梯度反转层（GRL）**：Ganin et al. 2016（[arXiv 1409.7495](https://arxiv.org/abs/1409.7495)）。
  特征提取器 + 域判别器做 minimax：判别器预测 nuisance（这里=仪器/CE），GRL 把梯度取反传回，
  **强迫 embedding 对 nuisance 不可判别**。这就是"必然性调整 embedding space"的标准工具。
  参数惯例：反转权重从 0 退火到 1。
- **稳定化（纯 GRL 会抖）**：[Calibrated Domain-Invariant Learning (CaNE)](https://ar5iv.labs.arxiv.org/html/1911.11314)
  用校准负熵替代原始 GRL 处理类别不平衡；[Iwasawa IJCAI 2020](https://mlanthology.org/ijcai/2020/iwasawa2020ijcai-stabilizing/)
  用判别器匹配；[Adversarial Forgetting AAAI 2020](https://mlanthology.org/aaai/2020/jaiswal2020aaai-invariant/)
  把不变性视为信息瓶颈。

---

## 2. 直接评估（无可争议）

**建一个 locked condition-invariance benchmark，冻结后不再改。**

1. **样本**：固定一批同分子谱图对，标注每条谱的 instrument / CE 状态（沿用错误图谱 §5.1 的
   同仪器/跨仪器、CE 差 ≤10 / >10 分层）。
2. **主指标**：`condition_gap`（§0），**按分子聚类 bootstrap**（不是按谱图对），95% CI 要 excludes 0。
3. **非目标 guard（防引入新错误）**：不同分子余弦相似度、Top-1、MRR、Pearson、Spearman **不得下降**。
4. **逐对审计**：fixes（跨条件对距离缩小）vs regressions（同条件对/不同分子对距离恶化）。
5. **test 完全不读**，只到最后测一次。

---

## 3. 数据构建

1. **正样本对 = 同分子 × 跨条件**：跨仪器、大 CE 差、其他已验证条件特异（沿用 v2 §7.1 的 324 对
   + 从全池扩展，若 hdf5 有 instrument/CE 元数据）。
2. **负样本对 = 不同分子 × 质量近邻（10 ppm）**，维持 FP 方向的共享峰难负例。
3. **严格 IK14 隔离**：train/val 无谱图泄漏，test 不读；机制禁用清单继续排除"低共享碎裂"。

---

## 4. 针对性增强设计（改丰度为主，遮蔽为辅）

FN 的根源是"条件特异峰"，但**条件改变的是峰的相对丰度比，而非峰的有无**：
- CE 升高 → 低质量碎片占比上升、各碎片**相对强度重新分配**（丰度比变化，不是峰消失）；
- 仪器差异 → 质量精度（微小 m/z 偏移）+ 电离灵敏度（强度变化）。

所以主增强应是**改丰度（强度抖动）**，不是**改有无（遮蔽）**。这已被业内基准方法证实：
**MS2DeepScore（Huber 2021，PMC8556919）的三大增强之一就是"强度抖动 ±40%"**，另两个是
低强度峰删除（0–20%）和新峰添加——"改丰度"是业内标准做法，且机理上最贴合 CE。

### 4.1 主增强：乘法强度抖动（改丰度，主攻 CE）
1. 对非 precursor 的真实峰（intensity ∈ (0,1]）强度乘 `exp(ε)`，`ε ~ N(0, σ²)`，σ≈0.15–0.25
   （对应 ±15–25%，落在 MS2DeepScore 的 ±40% 内）；
2. **整谱再归一化**（base peak 回到 1.0），保住 precursor 行、不改 m/z；
3. 只对 anchor/positive 双侧施加（模拟同一分子两次采集的丰度波动），negative 不施加；
4. 施加**一致性损失** `L_jitter = 1 − cos(z_full, z_jitter)`，逼嵌入对丰度波动不变。

### 4.2 辅增强：定向条件遮蔽（改有无，主攻 FP/共享峰过依赖）
1. 贪心峰匹配（0.02 Da）找 anchor 相对 positive 的**唯一峰**，置 -1；
2. 一致性损失同上。这治"共享峰过依赖"（FP），对 FN 是补充而非主力。

### 4.3 仪器专用：m/z 微抖动
模拟质量精度差：对真实峰 m/z 加 `δ ~ N(0, σ_mz²)`，σ_mz 对应 ±0.001–0.005 Da。

> 顺序：先上 4.1（改丰度，直接对症 CE 这个最大 gap），再视情况加 4.2/4.3。

---

## 5. 必然性调整 embedding space（三层递进，压力递增）

| 层 | 手段 | 必然性压力 | 风险 |
|---|---|---|---|
| L1 数据 | 跨条件正对 + 定向条件遮蔽 | 低（靠采样） | 低 |
| L2 损失 | `L_mask` 遮蔽一致性 | 中（直接梯度） | 低 |
| L3 对抗 | 条件判别器 + GRL（DANN），nuisance=instrument/CE | **高（强制不可判别）** | 中（GRL 会抖→用 CaNE/判别器匹配） |

**参数（有据可查）**：GRL 反转权重 0→1 退火；对抗项 `λ_adv` 0.1~1.0；`λ_mask` 0.1~1.0；
**`λ_preserve` 从 5.0 降到 0.1~1.0**（5.0 已证明把头钉死在官方，输出 0.0）。

**顺序**：先 L1+L2（稳、低风险）跑通看 condition_gap 是否动；若不够再上 L3。不要一上来押最不稳定的对抗。

---

## 6. 在哪个层级训练、训几层

**关键判断：head-only 可能不够。** 条件特异峰的信息**纠缠在 backbone 的峰级表征里**，
1024×1024 的 head 只做近线性变换，无法"移除"已经纠缠的条件信息。所以：

1. **第一阶段 head-only（1024×1024，104 万参数）**，3 种子，看 condition_gap 是否缩小；
2. 若不够 → **解冻最后 1 层 Transformer**，3 种子；
3. 仍不够 → **解冻最后 2 层**（封顶，不碰前 5 层，保 backbone 的碎裂化学基础）。

每步都配 §2 的 non-target guard；preserve 保持在 0.1~1.0 防坍缩，但不钉死。

---

## 7. 怎么保证不出错、不引入新错误

1. **非目标 guard 全绿**：不同分子相似度 / Top-1 / MRR / Pearson / Spearman 不下降。
2. **fixes > regressions** 逐对审计，且 regressions 不集中在某一结构区间。
3. **preserve ≥ 0.995** 但不再 0.999（允许头动）。
4. **3 种子方向一致**才宣称（§8 门槛）。
5. **机制禁用清单**（低共享碎裂）+ **test 只测一次**。

---

## 8. 怎么保证改正了错误（验证链）

因果验证（已做：删除条件特异峰→同分子相似度恢复 0.09–0.17）→ 定向干预（L1/L2/L3）→
**直接测 condition_gap**。若 gap 显著缩小 + 非目标不降 + fixes>regressions → **改正成立**。

诚实边界：L3（对抗）给的是"**必然性的梯度压力**"，不是"必然成功"——它是当前把"条件不变性"
写进目标函数的最强手段，但最终仍以 §0 的判据为准，数字说了算。

---

## 9. 里程碑（按依赖排序）

1. **M1 建 benchmark**：写 locked condition-invariance 评估脚本 + 分子聚类 bootstrap（先有尺子）。
   ✅ 脚本已写：`tasks/eval_condition_invariance_benchmark.py`（real-only、same-adduct、cross+same 配对、
   分子聚类 bootstrap、post-head 嵌入、locked manifest+sha256）。
   **基线已在 8/11 测出（官方模型 layer-7 precursor，500 分子队列）**：CE 差 ≥10 的 gap −0.190、
   跨仪器 gap −0.043、两者叠加 −0.297（对照同条件余弦 0.866）。新尺子在 post-head 嵌入上重测并扩到 ~1300 分子。
2. **M2 数据**：✅ 已建（2026-08-17，`tasks/step3_m2_build_cross_condition_pairs.py`）。train fold real 谱
   93,499 张 / 17,561 同分子组，枚举 **56,887 对跨条件同分子对**（另 1,084 同条件对），确定性步进采样
   2,000 对并配条件匹配负例（同仪器+同加合物优先、最近 m/z 的不同分子），manifest sha256 锁定。
   **口径差异（非 bug）**：56,887 vs Step 1 的 86,390 = Step 1 是 `[M+H]+` 专属且不截成员数；M2 是全 adduct
   + `representative_members` 截 40 成员（防高重复分子独占训练信号）。56,887 仍远超"≥数千"门槛。
3. **M3 L1+L2**：✅ 脚本已建（2026-08-17，`tasks/step3_m3_cross_condition_train.py`）+ smoke 跑通。
   **首版只上 L1（真实跨条件对），L2 遮蔽暂缓**（见下）。loss =
   `1−cos(anchor, cross-condition positive)`（跨条件一致性，与 §4.1 的 `1−cos` 同形，但作用在**真实跨条件对**
   而非已证无效的强度抖动）+ 负例守卫 `relu(cos(neg)−ceiling)`（ceiling = 官方负例余弦 + 0.03）
   + preserve 铰链 `relu(0.995 − cos(anchor, teacher))`（对应 §7.3"≥0.995 但不再 0.999"）。
   head-only、3 种子、backbone 冻结、embedding 跨种子缓存。
   **smoke（150 对 / 1 epoch）**：head 动了 —— train 折叠 cross_cosine **0.682→0.716**、margin 0.329→0.353、
   preserve_cosine 0.998（未触发 0.995 地板）；`cond=0.31` 有**真梯度**。证实"真实跨条件对的一致性损失有
   headroom"，对照 Step 3 强度抖动无 headroom（`docs/WEIGHTED_RULE_NOISE_TRAINING_PLAN` §3.8）。
   L2 遮蔽（§4.2）对 FN 是补充、主攻 FP，等 L1 在 locked benchmark 上看到 gap 动之后再上。
4. **M4 层级**：按需解冻 last-1/last-2 层，重复测。
5. **M5 L3**：条件判别器 + GRL（CaNE 稳定化），仅在 L1+L2 不够时上。
6. **M6 锁定判定**：§8 验证链全绿 → 才宣称改正；否则回 M3 改机制。

> 先做 M1——**没有直接测 condition_gap 的尺子之前，任何训练都是盲跑**。这是上一轮 full-pool
> 输出 0.0 的根本教训：我们用错了尺子（容易 val），现在先把对的尺子造出来。
