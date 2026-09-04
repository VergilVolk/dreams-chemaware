# 阶段汇报逐页讲稿：第 6 页以后

版本：2026-08-29  
定位：组会/阶段答辩型学术汇报。前 1–5 页沿用现有“非靶向代谢组学—代谢物注释—DreaMS 原理”内容，本稿从第 6 页开始。  
总任务：让听众理解我们不是简单叠加化学规则，而是先建立严格评价体系，定位 DreaMS 的真实残余错误，再分别从共享 embedding、候选专家、化学解释和生物学应用四个层面给出经过验证且边界清楚的改进。

## 统一视觉规范

- 标题使用结论句，不使用“实验结果一/二”之类目录式标题。
- 正式结果用深紫或深蓝；开发结果用蓝色；理论上界用虚线橙色；负结果与安全边界用灰红色。
- 所有性能图必须同时标出基线、样本数、数据划分和证据层级：`W` 表示共享权重，`X` 表示 embedding 后专家，`A` 表示动作/上界，`D` 表示开发，`B` 表示封存或外部验证。
- Recall@1 的变化统一写“百分点（pp）”，不得写成相对百分比。
- 每张统计图尽量给 95% CI；存在配对 Top-1 变化时同时报告 corrected/introduced。
- 讲稿中的“官方 DreaMS”特指 `official_embedding_slim.pt` 对应的官方谱图相似度微调权重，不与 DreaMS 论文中的其他任务专用模型混用。

---

## 第 6 页｜先统一评价语言：不同任务不能由一个“准确率”概括

### 版面上写什么

标题：**DreaMS 的能力需要按任务、候选协议和评价单位分别衡量**

左半页“DreaMS 原论文的评价”：

| 任务 | 输出 | 主要指标 |
|---|---|---|
| 谱图相似度近似 | 谱图对相似度 | Pearson correlation |
| 谱库身份判别 | 同分子/异分子谱图对 | AUROC |
| 结构类似物检索 | MCES 阈值下的类似物 | AUROC |
| 结构库检索专用模型 | 候选结构排序 | Accuracy@k |

右半页“本项目的评价”：

| 任务 | 评价单位 | 主要指标 |
|---|---|---|
| 严格谱库检索 | 每个 query 的候选组 | Recall@1、MRR |
| near/isomer 压力测试 | MCES 分层候选组 | 分层 Recall@1 |
| 模型改动的成败 | 同一 query 前后配对 | corrected / introduced |
| 统计稳定性 | formula/scaffold 簇 | cluster bootstrap CI、McNemar |

页脚一句：**AUROC 衡量谱图对可分性；Recall@1 衡量候选组内第一名是否正确；二者不能互相替代。**

### 配什么图/公式

中间放一个很小的“评价单位变换”示意：

`谱图对分数 → 候选组排序 → Top-1 身份 → 生物样本中的注释证据`

右下角写严格排序定义：

\[
\operatorname{rank}(q)=1+\#\{c\neq c^+:s(q,c)\ge s(q,c^+)\},\qquad
\mathrm{R@1}=\frac{1}{N}\sum_q\mathbf 1[\operatorname{rank}(q)=1]
\]

并注明：平分计错，候选先按分子聚合。

### 讲稿

“在继续汇报之前，我先统一一下评价语言。DreaMS 原论文并不是只用一个准确率评价模型，而是分别考察谱图相似度近似、谱库身份判别、结构类似物检索，以及另外训练的结构库检索模型。这里必须特别注意，Pearson、AUROC 和 Recall@1 回答的是不同问题。AUROC 关注大量谱图对总体能不能分开；而我们更关心真实注释时，同一个质量窗口内的多个候选中，正确分子能不能排在第一位。

因此，本项目把评价单位固定为 query 候选组：先做 strict 10 ppm、同加合物筛选，再按分子聚合，最后计算 Recall@1 和 MRR。模型改动不能只报一个平均分，还要报告修正了多少原错误、又引入了多少新错误，并用分子式簇 bootstrap 和 McNemar 检验配对变化。后面出现的每一个百分点，我都会说明它属于哪一个任务、哪一种协议以及哪一级证据。”

### 证据来源

- DreaMS 原论文：[Self-supervised learning of molecular representations from millions of tandem mass spectra using DreaMS](https://www.nature.com/articles/s41587-025-02663-3)
- 项目基线账本：`docs/OFFICIAL_DREAMS_CHECKPOINT_BASELINE_LEDGER_20260829.md`

---

## 第 7 页｜官方权重已经很强，但任务难度差异巨大

### 版面上写什么

标题：**同一官方 checkpoint 的 Recall@1 从 0.49 到 0.92，说明“任务协议”决定了难度**

只保留六个关键数：

| 面板 | n | 官方 Recall@1 | 含义 |
|---|---:|---:|---|
| full error graph | 23,876 | 0.9244 | 大规模常规严格检索 |
| P2/P2b OOF graph | 5,037 | 0.8606 | 多候选困难开发图 |
| g8r locked | 620 | 0.8081 | 跨条件困难正例 |
| P3 main | 3,000 | 0.8793 | 封存常规主面板 |
| P3 isomer | 1,989 | 0.7949 | 同分子式异构体 |
| P3 near-core | 496 | 0.4879 | MCES 0–2 核心难例 |

页脚：**不能把不同面板上的百分点直接相加，也不能用常规主面板掩盖 near-core 退化。**

### 配什么图/表

主图用一张横向“难度阶梯”条形图，横轴为 Recall@1，按照 0.9244→0.4879 排序。每根柱旁标 n；P3 near-core 用深红色强调。

右上角小字标官方 checkpoint SHA256 前 12 位：`8928f908606c…`，说明所有基线来自同一冻结工件。

### 讲稿

“接下来先看官方权重本身。它在大规模常规检索图上的 Recall@1 已经达到 0.9244，说明我们不是在改造一个弱模型。但是一旦把任务收缩到跨采集条件正例、同分子式异构体和 MCES 0–2 的 near-core，性能会分别降到 0.808、0.795，最困难的 near-core 只有 0.488。

这个结果决定了后续研究视角：我们不能再笼统地问‘DreaMS 好不好’，而要问‘它在什么局部化学空间、什么候选关系和什么采集条件下失败’。它也解释了为什么某个方法在常规检索上提升，并不代表真正解决了异构体问题。后面的所有结果都同时给出 overall 和 near，必要时还要单独看跨仪器、跨碰撞能以及异构体子群。”

### 证据来源

- `docs/OFFICIAL_DREAMS_CHECKPOINT_BASELINE_LEDGER_20260829.md`

---

## 第 8 页｜早期设想覆盖了三条路线，但“合理”不等于“有效”

### 版面上写什么

标题：**我们最初提出三类改进：化学规则、谱图噪声与候选后验专家**

中间画三条分叉路线：

1. **ChemAware embedding**：规则/概念监督 → 改变共享 embedding；
2. **Noise embedding**：困难样本与定向峰扰动 → 改变共享 embedding；
3. **Post-embedding experts**：P2b 谱学证据、BioAware 网络证据 → 候选级后验修正。

底部明确边界：

- 共享 embedding：推理时只输入原始谱图，query/reference 使用同一新权重；
- 下游专家：需要候选组、原始谱峰或队列网络，不等于模型权重改善；
- 双重映射：负责解释，不是另一个排序头。

### 配什么图

画一个单一的数据流图：

`clean MS/MS → shared encoder z → candidate retrieval → P2b → BioAware → annotation`

从 `z` 向下分出：`chemical concept ↔ peak evidence` 双重映射。用不同颜色表示“改变 embedding”和“embedding 后模块”。

### 讲稿

“前期我们的思路比较宏观：一方面希望把化学规则注入 DreaMS，另一方面希望通过噪声增强提高模型鲁棒性，同时也考虑在 embedding 之后加入谱学或生化网络专家。这里后来最重要的认识，是三条路线必须严格分工。

噪声微调和 ChemAware 微调都必须真正改变共享 embedding；也就是说，推理时只给一张原始谱图，就能得到新的表示。P2b 和 BioAware 则是 embedding 之后的候选专家，它们可以提高系统性能，但不能被表述为模型权重变好了。双重映射的任务又不同，它要把 embedding 方向连接到化学概念和具体峰，并用干预验证解释是否忠实。把这三个层次分开，是我们后续避免偷换概念的前提。”

### 证据来源

- `docs/PROJECT_RESEARCH_MASTER_SUMMARY_20260829.md`
- `docs/BIOAWARE_NETWORK_EXPERT_AND_CONTEXT_ADAPTER_PLAN_20260827.md`

---

## 第 9 页｜两条“看起来合理”的路线被数据否定，研究因此转向错误机制

### 版面上写什么

标题：**规则重叠不能直接充当标签，随机删峰也不能代表有效噪声**

放一个“假设—检验—裁决”表：

| 初始假设 | 关键检验 | 结果 | 裁决 |
|---|---|---|---|
| 规则重叠高就是正样本 | 规则误差检测与因果删峰 | AUC 有信号，但严格特异性门 0 个动作通过 | 不作距离标签 |
| 随机删除 20% 峰可增强鲁棒性 | 目标删峰 vs 强度/mz 匹配随机删峰 | 多数总体位移不是错误特异信号 | 不作核心方法 |
| 开发集重排提升可直接泛化 | RAW v1 冻结测试 | 开发 +4.35pp；Test-A/B 仅 +0.45/+0.73pp，均不显著 | 终止通用 RAW-v1 |

右下角结论：**先分析真实错误，再设计与错误机制方向一致的噪声。**

### 配什么图

使用一张三段漏斗：宏观先验 → 匹配对照 → 可复现机制。前两段大量候选被淘汰，最后只保留通过反事实检验的动作。

### 讲稿

“这一步是整个项目的转折。最初我们认为，规则重叠高的谱图可能就是正样本，或者随机删掉一部分峰就能迫使模型学得更稳健。但系统检验后发现，规则确实与错误风险相关，却不足以定义分子身份；更严格的匹配因果删峰中，没有任何规则动作通过错误特异性门。随机删峰也会改变 embedding，但这种变化往往同样发生在本来正确的样本上，不能证明它在修正错误。

与此同时，RAW 重排器在开发集上看起来提高了 4.35 个百分点，但冻结测试只剩 0.45 和 0.73 个百分点，而且都不显著。这些负结果不是无效工作，它们共同说明：我们不能从一个宏观合理的先验直接跳到训练，而必须先建立真实错误图谱，再找方向正确、相对匹配随机对照仍然成立的峰级动作。”

### 证据来源

- `docs/PROJECT_RESEARCH_MASTER_SUMMARY_20260829.md`
- `docs/RAW_RERANKER_V1_FINAL_VERDICT_20260822.md`

---

## 第 10 页｜23,876 个真实 query 揭示：多数错误来自正例证据不足

### 版面上写什么

标题：**1,805 个官方错误不是同一种错误：positive deficit 是主体**

核心数字：

- 23,876 queries；2,522 identities；1,082 formulas；
- official Recall@1 = 0.9244；1,805 errors；
- near 子群 13,784 queries，1,446 errors。

错误机制表：

| 机制 | 数量 | 比例 |
|---|---:|---:|
| positive deficit only | 1,242 | 68.8% |
| positive deficit + negative excess | 197 | 10.9% |
| negative excess only | 188 | 10.4% |
| comparative boundary | 178 | 9.9% |

一句结论：**只压低错误候选最多覆盖一小部分错误；主体需要恢复同分子跨条件正证据。**

### 配什么图/公式

左侧用 100% 堆叠条形图显示四类错误；右侧用 margin 分解：

\[
m(q)=s(q,c^+)-\max_{c^-}s(q,c^-)
\]

并画两种错误：

- positive deficit：`s(q,c+)` 异常偏低；
- negative excess：`max s(q,c-)` 异常偏高。

### 讲稿

“在严格候选图上，我们对 23,876 个真实 query 做了统一错误分解。官方 DreaMS 一共有 1,805 个 Top-1 错误，其中 1,446 个发生在 near 子群。更重要的是，68.8% 属于单纯的 positive deficit，也就是正确分子的跨条件参考谱与 query 相似度不足；另有 10.9% 同时存在正例不足和错误候选过高。只有大约一成是纯粹的 negative excess。

这个分解直接纠正了我们的训练直觉。此前很多删峰策略主要试图把错误候选推远，但如果错误主体是正确候选被推得过远，仅做负臂并不能解决问题。因此后续噪声微调必须分成两条方向：N-arm 处理错误候选过近，P-arm 处理同分子跨仪器、跨碰撞能导致的正例证据不足，并且二者都要保留本来正确样本的安全约束。”

### 证据来源

- `docs/PROJECT_RESEARCH_MASTER_SUMMARY_20260829.md`
- `tasks/analyze_g8r_real_error_atlas.py`

---

## 第 11 页｜峰级机制具有因果方向，但“删峰有效”必须相对匹配随机对照成立

### 版面上写什么

标题：**定向峰干预优于匹配随机干预，证明部分错误来自可重复的峰证据偏差**

放三组结果：

1. confirmation：删除混淆峰纠正率 28.1%，匹配随机删峰 7.6%；
2. 一次性机制面板：定向删除纠正 10.74%，匹配随机 4.52%；
3. 公式平衡 margin 净改善 0.0305，95% CI [0.0229, 0.0386]。

关键反向对照：删除身份特异峰会降低正确排序 margin。

### 配什么图/公式

主图使用配对森林图：每个动作画 `target − matched random` 的 margin 效应与 95% CI。

页中写反事实量：

\[
\Delta_{\text{specific}}
=\Delta m_{\text{target deletion}}
-\mathbb E[\Delta m_{\text{intensity/mz-matched random deletion}}]
\]

再配一张真实谱图小示意：红色为混淆峰、蓝色为身份峰、灰色为匹配随机控制峰。

### 讲稿

“这里我们把相关性推进到反事实验证。仅仅看到删峰后分数变化是不够的，因为删除任何高强度峰都会扰动 embedding。真正有意义的量，是目标删峰相对于峰数、强度和质量位置匹配的随机删峰，是否产生额外的方向性 margin 改善。

在 confirmation 数据中，删除混淆峰可以纠正 28.1% 的错误，而匹配随机删除只有 7.6%；冻结机制面板中，定向删除仍然是 10.74% 对 4.52%。同时，删除身份特异峰会产生相反效应，进一步降低正确候选的 margin。这说明至少一部分 near 错误不是随机波动，而是模型对具体峰证据使用失衡。需要强调的是，这些动作依赖已知正负候选，仍然不是部署算法；它们的价值是为训练提供有方向的干预信号。”

### 证据来源

- `docs/PROJECT_RESEARCH_MASTER_SUMMARY_20260829.md`
- `docs/COUNTERFACTUAL_PEAK_FINETUNE_STAGE_REPORT.md`

---

## 第 12 页｜严格的数据协议先于模型：避免重复谱、候选泄漏和容易正例造成虚高

### 版面上写什么

标题：**我们将训练、开发和封存评价按 identity 与 formula 双重隔离**

左侧列正式协议：

- strict 10 ppm、same adduct、exclude self；
- 候选按 IK14 分子聚合，谱图分数取组内最大；
- 正例：同一 IK14 的真实跨条件谱；
- 负例：不同 IK14，按 MCES/结构难度分层；
- formula-group folds；P3 query identity overlap = 0；
- tie counts against positive；测试时只加载冻结工件。

右侧列 P3 面板：main 3,000；isomer 1,989；near-core 496；near+mid 661；exposed 851；sim-to-real 609。

### 配什么图

画一个数据流与隔离图：

`全 HDF5 → 训练允许身份 → formula folds → 开发 OOF`  
另一条完全分离：`P3 pristine identities → 一次性封存评价`

用红叉标出曾被抓住的问题：A/B 重叠、重复谱富集、先 global top-k 再 ppm 过滤、测试现场重训。

### 讲稿

“在开始训练之前，我们先把候选和拆分协议固定下来。每个 query 只在 strict 10 ppm、同加合物的候选中比较，并先按分子聚合，避免某个分子因为重复谱多而获得额外优势。正例必须是同一 IK14 的真实参考谱，困难性通过跨条件和 MCES 分层定义，而不是由规则重叠自创标签。

训练与开发采用分子式分组折，P3 则在 identity 层面与训练完全零重叠。所有并列都对正例不利，测试只加载冻结工件，任何缺行、无正例或哈希不符都会 fail-closed。这个基础设施曾经真实抓住过 A/B 面板重叠、重复谱导致的 95% 虚高、候选生成顺序错误和现场重训等问题。因此它不是附属工程，而是后面所有性能数字可信的前提。”

### 证据来源

- `docs/PROJECT_RESEARCH_MASTER_SUMMARY_20260829.md`
- `docs/OFFICIAL_DREAMS_CHECKPOINT_BASELINE_LEDGER_20260829.md`

---

## 第 13 页｜噪声不再是随机遮峰，而是沿错误机制构造的方向性训练动作

### 版面上写什么

标题：**定向噪声的目标不是让谱图更乱，而是把错误 margin 沿正确方向推回去**

展示四类成熟动作：

1. **candidate-gradient**：削弱使错误候选过近的查询峰；
2. **role-confounder**：去除对错误候选有利、对正例无利的峰；
3. **consensus projection**：把现有峰强度向同分子真实参考谱的共识强度投影；
4. **recurrent peak transfer / union mix**：仅迁移真实同分子参考中反复出现的少量峰。

明确禁用：role-shared 强删除、无条件高剂量 dropout、用动作结果作为训练权重、P2b 参与定义标签。

### 配什么图/公式

用一张四行谱图 before/after 图，每行只突出 3–5 个变化峰。

训练动作的目标写成：

\[
m(q')=s(f_\theta(q'),f_\theta(c^+))-max_{c^-}s(f_\theta(q'),f_\theta(c^-))
\]

动作选择阶段检验 `Δm_target − EΔm_control > 0`；真正推理仍为 `z=fθ(clean spectrum)`。

### 讲稿

“因此，我们后续所说的噪声，不是随机删除 20% 或 30% 的峰，而是沿已验证错误机制构造的方向性动作。candidate-gradient 和 role-confounder 主要处理 negative excess；consensus projection 和 recurrent peak transfer 主要补充 positive deficit。后两类动作也不是凭空生成峰，而是只利用真实同分子参考谱中稳定重复出现的峰或强度关系。

动作阶段允许利用候选关系来判断方向，但最终模型的推理输入仍然只有 clean spectrum。换句话说，候选信息只用于构造训练压力，不能在部署时泄漏进 embedding。我们还明确淘汰了 role-shared 的强删除，因为它会同时破坏正例证据；也禁止把事后动作结果直接作为样本权重，否则会把上界信息伪装成可部署监督。”

### 证据来源

- `docs/NOISE_FINETUNE_SYSTEMATIC_STRATEGY_20260824.md`
- `docs/PROJECT_RESEARCH_MASTER_SUMMARY_20260829.md`

---

## 第 14 页｜动作矩阵找到了大梯度，也同时揭示了危险动作

### 版面上写什么

标题：**同样是峰扰动，candidate-gradient 净修正 +98，而强 role-shared 净损失 −874**

左侧用精选矩阵数据：

| 动作 | 最佳/代表剂量 | corrected | introduced | net |
|---|---:|---:|---:|---:|
| candidate-gradient | 0.50，step 6 | 140 | 42 | +98 |
| role-confounder | 1.00 | 24 | 0 | +24 |
| role-unmatched | 1.00，step 5 | 41 | 26 | +15 |
| role-shared | 1.00，step 6 | 151 | 1,025 | −874 |

右侧放容量阶梯：

- S1c+S2+S3A oracle：799 errors，+3.35pp；
- A4 history union：920 errors，+3.85pp；
- P/N union：922 errors，+3.86pp；
- 加 positive-guided + recurrent transfer：1,257 errors，名义容量 >5pp。

醒目标注：**这是 outcome-aware action headroom，不是训练后模型性能。**

### 配什么图

复用已有 S3A 四象限热图，但把主图裁成“corrected / introduced / net”三幅，不必再放第四幅覆盖图。右侧用瀑布图显示可恢复错误并集从 799→920→922→1,145→1,257。

### 讲稿

“这张矩阵回答了两个问题。第一，错误空间中确实存在足够大的方向性梯度。candidate-gradient 在第六步可以修正 140 个错误、引入 42 个，净修正 98；role-confounder 的覆盖较小，但风险很低。第二，不是所有看起来有化学意义的删峰都安全。role-shared 在强剂量下虽然修正 151 个错误，却引入 1,025 个新错误，净损失达到 874。

将不同动作逐 query 事后选择并允许 no-op，动作空间曾覆盖 920 个错误，对应约 3.85 个百分点；加入正例引导和真实重复峰迁移后，名义并集超过 5 个百分点。但这里必须严格称为 headroom：它看过每个动作的结果再挑最优，不是一个模型可以在未知样本上直接获得的成绩。它证明‘数据里有梯度’，不证明‘学生已经学会这条梯度’。”

### 证据来源

- `docs/PROJECT_RESEARCH_MASTER_SUMMARY_20260829.md`
- 既有 S3A action-matrix 图及相应结果目录

---

## 第 15 页｜真正的目标是一个共享的新 encoder，而不是逐 query 动作选择器

### 版面上写什么

标题：**训练时使用定向噪声，推理时只保留一个共享 DreaMS encoder**

主流程：

`clean query + clean references`  
`↘ fixed directional noisy views`  
`→ same shared encoder fθ → candidate-group ranking loss + safety preservation`  
`→ 新 checkpoint`

固定实现：

- 解冻最后 1/7 Transformer block + 官方 projection head；
- backbone LR = 2×10⁻⁶；head LR = 1×10⁻⁵；4 epochs；gradient clip 1.0；
- identity-equal weighting；4 views/identity；
- query/reference 同一 encoder；P2b 禁用；推理 clean-only。

### 配什么公式

用“结构化目标”而非伪造精确权重：

\[
\mathcal L=
\mathcal L_{\text{candidate-group rank}}
+\lambda_{\text{noise}}\mathcal L_{\text{directional views}}
+\lambda_{\text{safe}}\mathcal L_{\text{protected correct}}
+\lambda_{\text{pres}}\left(1-\cos(z_\theta,z_{\text{official}})\right)
\]

旁注：公式表示训练职责分解，具体流与权重以冻结脚本为准。

### 讲稿

“动作上界之后，真正需要解决的问题是：能否把这些局部动作迁移进一个共享 encoder。我们的正式实现不是为每个 query 训练动作选择器，也不是在 embedding 后再拟合一个残差头，而是直接更新 DreaMS 的最后一个 Transformer block 和官方 projection head。query 和 reference 始终使用同一套新权重，推理时只输入原始 clean spectrum。

训练目标同时承担三个责任：提高候选组内正确分子的 margin；让定向噪声视图产生预期的鲁棒或纠错方向；保护原本正确的 query，限制 embedding 发生无边界漂移。最终固定配置使用较小的 backbone 学习率和较大的 head 学习率，训练四个 epoch。P2b、动作事后结果和封存 P3 都不进入训练。这一步的输出才可以称为一个新的 embedding checkpoint。”

### 证据来源

- `docs/NOISE_E4A_HIGHLR_MULTIFOLD_RESULT_20260827.md`
- `tasks/train_noise_final_e4a_direct_augmentation.py`

---

## 第 16 页｜共享权重获得稳定但有限的真实提升：overall +0.635pp

### 版面上写什么

标题：**5 个 formula folds × 3 个 seeds 均为正，证明定向噪声可以迁移进共享 embedding**

大号结果：

- Recall@1：平均 **+0.635pp**；
- near Recall@1：平均 **+0.522pp**；
- 每 seed 平均 **约 186 corrected / 35 introduced**；
- 风险净收益 `corrected − 2×introduced ≈117`；
- **15/15** fold 的 formula-cluster CI 下界 >0；
- 平均 preservation >0.995，p01 约 0.980–0.982。

三 seed 表：

| seed | overall Δ | near Δ | corrected/introduced |
|---:|---:|---:|---:|
| 20260828 | +0.611pp | +0.508pp | 183/37 |
| 20260829 | +0.662pp | +0.537pp | 191/33 |
| 20260830 | +0.632pp | +0.522pp | 185/34 |

页脚：**开发图多折多种子权重结果；尚不是新的外部盲测结论。**

### 配什么图

主图用 15 个点的 forest/strip plot：横轴为每 fold 的 Recall@1 Δ，零线清楚标出；点按 seed 着色，误差线为 formula-cluster CI。

右下角放 corrected vs introduced 的三组配对柱。

### 讲稿

“这是目前噪声微调路线最关键的真实结果。固定训练配方在五个 formula fold、三个独立 seed 上全部得到正向提升。整体 Recall@1 平均提高 0.635 个百分点，near 提高 0.522 个百分点。每个 seed 平均修正约 186 个原错误，同时引入约 35 个新错误；十五个 fold 的公式簇置信区间下界全部大于零。

这说明定向峰级噪声不只是冻结模型上的动作现象，确实可以迁移进同一个共享 embedding。另一方面，我们也不把它夸大为全面超越 DreaMS：这是开发图上的多折多种子证据，还没有在一个全新的外部封存面板上完成最终总评。更重要的是，0.635 个百分点显著低于动作矩阵的 3.85 个百分点，这个差距本身就是下一阶段需要解释的科学问题。”

### 证据来源

- `docs/NOISE_E4A_HIGHLR_MULTIFOLD_RESULT_20260827.md`

---

## 第 17 页｜3.85pp 上界为何只迁移出 0.635pp：瓶颈是共享表示学习，不是梯度不存在

### 版面上写什么

标题：**动作空间有容量，但局部、特权和事后方向难以被一个共享函数完全吸收**

中央放“上界—模型”漏斗：

`A4 outcome-aware union +3.853pp`  
↓ 去除看答案选择、候选特权、逐 query no-op  
`shared encoder OOF +0.635pp`

描述性迁移比例：约 16.5%，注明“非正式效率指标，仅用于说明量级差”。

四个瓶颈：

1. oracle 对每个 query 看结果选动作，模型不能；
2. 动作依赖已知正/错候选，推理只有单张谱；
3. 1,439 个 positive-deficit 错误需要恢复证据，删除类 N-arm 覆盖不足；
4. 共享参数必须同时保护 22,071 个原本正确 query，强更新会产生新错误。

### 配什么图/数学

定义 oracle 与可部署模型的差别：

\[
\Delta_{oracle}=\frac1N\sum_q\max_{a\in\mathcal A\cup\{0\}}
\left[y_q(a)-y_q(0)\right]
\]

\[
\Delta_{model}=R@1(f_{\theta^*})-R@1(f_{\theta_0}),
\quad a(q)\text{ 在推理时不可见}
\]

### 讲稿

“为什么动作上界接近 3.85 个百分点，而共享权重只有 0.635？首先，两者不是同一种对象。动作上界对每个 query 看过结果以后，可以从多个动作和 no-op 中挑最有利的一个；共享模型面对未知谱图，没有这个答案。其次，candidate-gradient 等动作显式利用了正确候选和错误候选的关系，而部署时 encoder 只能看到单张原始谱图。

第三，错误图谱表明主体是 positive deficit，单纯删除混淆峰只能处理一部分 negative excess。第四，共享模型不能只服务 1,805 个错误，还必须保护两万多个原本正确的 query，因此学习率和表示漂移都受到安全约束。换句话说，当前瓶颈不是数据里没有大梯度，而是如何把局部、特权、逐 query 的方向压缩成一个候选无关的共享表示。下一阶段应该扩大真实跨条件 P-arm 和可学习但不泄漏的局部峰策略，而不是继续提高粗暴删峰剂量。”

### 证据来源

- `docs/NOISE_E4A_HIGHLR_MULTIFOLD_RESULT_20260827.md`
- `docs/PROJECT_RESEARCH_MASTER_SUMMARY_20260829.md`

---

## 第 18 页｜共享 embedding 与下游专家是互补层，而不是同一条成绩线

### 版面上写什么

标题：**新 embedding 改变通用谱图表示；候选专家只在检索上下文中追加证据**

系统图：

1. `MS/MS → official/new shared embedding`：W，clean-only；
2. `candidate group + raw peak evidence → P2b`：X，谱学专家；
3. `cohort feature graph + reaction evidence → BioAware`：X，生化网络专家；
4. `embedding/peak tokens → concepts → peaks`：解释层。

历史对照小表：

| 模块 | 开发结果 | 封存/外部结果 | 裁决 |
|---|---|---|---|
| RAW-v1 | +4.35pp | A +0.45pp；B +0.73pp，均不显著 | 归档 |
| P1 risk gate | 29/7 | 不优于 disagreement-only | 终止学习门控 |
| P2b | OOF +3.91pp | P3 main +1.07pp；near −4.23pp | 有条件保留 |
| BioAware | 小面板局部正信号 | 无显著外部增益 | 暂作解释/研究线 |

### 配什么图

主图用四层纵向架构，不用四个孤立卡片。每层右侧标输入、输出和能否改变 embedding。

### 讲稿

“噪声微调解决的是共享表示问题，但真实注释还可以利用候选上下文中的额外证据。因此我们保留一个正交的后处理层。P2b 使用原始峰和中性丢失信息，在候选组内重新排序；BioAware 使用实验 feature graph 和反应网络，输出网络支持、冲突或弃权。它们都不应该被说成新的 embedding。

这条路线也经历了严格淘汰。RAW-v1 的开发提升没有在冻结测试上显著复现；P1 学习门控没有超过简单 disagreement-only。最终保留下来的是 P2b，但它只在常规主面板成立，在 near-core 上反而退化。这里还要纠正一个术语：OOF 是交叉验证协议，不是一个专家模块。我们后面汇报 P2b 时，会把它作为系统性能层单独讨论，而不与共享权重的 0.635 个百分点相加。”

### 证据来源

- `docs/P2B_RANK_FUSION_FORMAL_RECORD_20260823.md`
- `docs/RAW_RERANKER_V1_FINAL_VERDICT_20260822.md`

---

## 第 19 页｜P2b 在常规候选检索上成立，但必须对 near-core 设置安全回退

### 版面上写什么

标题：**冻结 P2b 在 P3 main 显著提高 +1.07pp，却在 near-core 降低 −4.23pp**

方法：预注册 rank fusion，经候选组内标准化后

\[
S_{P2b}=0.1S_{DreaMS}+0.1S_{entropy}+0.8S_{neutral\ loss}
\]

开发 OOF：

- 5,037 queries：0.8606→0.8997，+3.91pp；
- MRR +0.0250；near +5.83pp；280/83；
- formula bootstrap 95% CI [+2.91,+4.87]pp。

封存 P3：

- main：0.8793→0.8900，+1.07pp；89/57；CI [+0.235,+1.889]pp；McNemar p=0.0101；
- near-core：0.4879→0.4456，−4.23pp；20/41；CI [−8.09,−0.61]pp。

### 配什么图

左右两幅 slope chart：左图 P3 main 上升，右图 near-core 下降。下方用 corrected/introduced 的红蓝配对柱解释方向。

### 讲稿

“P2b 是目前最成熟的 embedding 后谱学专家。它不是重新训练 DreaMS，而是在候选组内融合 DreaMS、entropy 和 neutral-loss 证据。开发 OOF 中，它把 Recall@1 从 0.8606 提到 0.8997，提升 3.91 个百分点；冻结 P3 main 中仍保留 1.07 个百分点的显著增益，修正 89 个、引入 57 个。

但是 near-core 给出了完全相反的结果：Recall@1 从 0.4879 降到 0.4456，修正 20 个却引入 41 个，而且置信区间整体为负。这说明传统峰匹配和中性丢失在常规候选上提供互补证据，但面对最接近的异构体时会被共享碎片误导。因此 P2b 可以作为常规检索保底模块，但不能全局无条件启用；部署时必须对 near-core 或高结构歧义候选回退到共享 embedding，或者明确弃权。”

### 证据来源

- `docs/P2B_RANK_FUSION_FORMAL_RECORD_20260823.md`
- `docs/PROJECT_RESEARCH_MASTER_SUMMARY_20260829.md`

---

## 第 20 页｜ChemAware 的阶段成果是“证据语言”，不是已经成功的规则微调

### 版面上写什么

标题：**3,486 条规则能描述风险和峰证据，但尚不能直接监督 embedding 距离**

规则资产：

- 25,275 张可达谱；3,472 identities；
- 335 核心规则 + 3,151 MassBank 衍生规则；
- NL 293、CF 3,174、ISO 8，其余 11。

已成立：

- 不同分子错误检测 AUC 0.647；
- 同分子一错一对 AUC 0.599；
- 谱图概念可作为解释标签和候选证据。

未成立：

- 规则重叠作为正负样本标签；
- 规则距离直接替代结构距离；
- 匹配因果删除 eligible action = 0；
- 规则微调带来稳定检索提升。

### 配什么图

左侧做“规则库组成”矩形树图；右侧做证据等级梯子：

`规则命中 → 错误风险相关 → 候选特异证据 → 因果峰忠实性 → embedding 性能`

只把前两级填满，第三部分填一半，后两级留空并标下一步。

### 讲稿

“化学规则路线并不是完全失败，但它的正确位置发生了变化。我们已经建立覆盖 25,275 张谱、3,472 个身份的 3,486 条规则库，并修正了中性丢失、同位素和前体质量偏移的语义。规则对于错误风险有中等信号：不同分子错误检测 AUC 为 0.647，同分子一错一对为 0.599。

但是这些数值不足以把规则重叠直接当成分子身份或 embedding 距离标签。更严格的匹配因果删除中，没有动作通过正式特异性门。因此现在 ChemAware 的合理定位是：把规则作为峰证据语言、概念监督、样本分层和冲突解释，而不是把 3,000 多条规则全部硬注入距离。未来真正的 ChemAware embedding 必须证明规则/概念监督能在结构隔离测试上提高共享表示，而不能只报规则数量或训练 loss。”

### 证据来源

- `docs/PROJECT_RESEARCH_MASTER_SUMMARY_20260829.md`
- `docs/CHEMAWARE_FINETUNE_RESET_20260824.md`

---

## 第 21 页｜双重映射已经证明“可解码”和“峰是输入来源”，尚未证明控制 Top-1

### 版面上写什么

标题：**解释链初步闭合：embedding → 化学概念/结构环境 → 具体峰 → 定向删峰**

两层结果：

| 层级 | 结果 |
|---|---|
| 266 个谱图概念 | macro-AUPRC 0.659，基线 0.200；254/266 ≥2×基线 |
| 469 个局部结构环境 | macro-AUPRC 0.240，基线 0.0447；396/469 ≥2×基线 |
| 跨层桥梁 | 1,260 个跨发现/确认复现；方向门后保留 175 条 |
| 峰 token 因子 | 10 个稳定；8 个峰质量复现；2 个结构环境复现 |
| 忠实性 | factor117、176 定向删峰额外 embedding 位移 CI >0 |

边界：两个因子的严格检索 margin CI 均跨 0。

### 配什么图/数学

主图画一条具体证据链：

`z direction ↔ r1:N(C)(C)C ↔ m/z 70.0651 ↔ 删除该峰 → concept logit/embedding 改变`

并写三个门：

\[
\text{decode} \land \text{localize} \land \text{faithful intervention}
\]

只有再通过 `retrieval-direction` 才能解释 Top-1。

### 讲稿

“双重映射的目标不是给 embedding 贴一个化学名称，而是建立一条可证伪链：首先从全局或峰级 embedding 解码化学概念和局部结构环境；然后把概念定位到具体峰或质量差；最后删除被定位的峰，验证对应概念得分或 embedding 是否比匹配随机删除下降更多。

目前，266 个谱图概念的 macro-AUPRC 为 0.659，469 个结构环境为 0.240，分别显著高于流行率基线；跨发现和确认集复现后保留 175 条结构环境—谱图概念桥。两个峰因子通过了定向删峰导致额外 embedding 位移的忠实性检验。但它们对严格检索 margin 的置信区间仍跨零。因此我们已经可以说某些峰是 embedding 因子的真实输入来源，还不能说这些因子已经解释了 Top-1 决策。这个边界会在汇报中明确保留。”

### 证据来源

- `docs/DOUBLE_MAPPING_STATUS_20260816.md`

---

## 第 22 页｜BioAware 完成了严谨工程闭环，但外部准确率增益尚未成立

### 版面上写什么

标题：**生化网络“有路径”不等于“当前候选正确”，BioAware 必须以弃权为默认**

方法图：

`candidate spectral evidence + observed feature graph + reaction hyperedges + orthogonal evidence → supported/conflicted/abstain`

关键结果：

- MTBLS1905：36 queries，DreaMS 0.750；evaluation-only 0.778，1/0，但 CI 下界为 0；可部署版本 0 改变；
- MTBLS13729 v1：20/21→19/21，0/1；v2 two-layer：0/0；
- MetDNA3 117-query development：DreaMS 95/117；G3-v2 2/0、+1.71pp，但 CI 下界 0；
- 当前结论：系统与负对照成立，显著外部增益未成立。

### 配什么图

左侧画反应超图与 query leave-out；右侧画三列结果：supported、conflicted、abstain。不要画夸张的全局 diffusion 动画。

底部用一条风险公式：

\[
\text{override only if }\sum_{k\in\text{independent families}}\mathbf 1[e_k\text{ supports}]\ge 2
\quad\text{and}\quad \widehat{\Delta risk}>0
\]

### 讲稿

“BioAware 解决的是另一个问题：当谱学候选接近时，队列中的共检出 feature 和生化反应网络能否提供正交证据。我们已经实现了显式反应超图、leave-query-out、leave-truth-out、货币代谢物过滤、度数保持 rewiring 和无证据回退。

但外部结果要求我们保持克制。MTBLS1905 中 evaluation-only 只修正 1 个样本，置信区间下界为零；可部署版本因为种子不足完全不干预。MTBLS13729 v1 反而引入一个错误，v2 选择全部弃权。MetDNA3 开发集最新安全门得到 2 修正、0 新增，但样本太小，置信区间仍接触零。这说明‘存在一条生化路径’不能证明当前 feature 的身份。BioAware 现在是一条有创新潜力但尚未过性能门的研究线；在至少两个独立证据族共同支持之前，正确输出应当是弃权。”

### 证据来源

- `docs/BIOAWARE_V1_IMPLEMENTATION_AND_PILOT_RESULT_20260827.md`
- `docs/BIOAWARE_METDNA3_DEVELOPMENT_RESULT_20260828.md`
- `docs/BIOAWARE_10PP_NETWORK_HEADROOM_PLAN_20260828.md`

---

## 第 23 页｜真实生物学应用采用“先冻结注释、后看表型”，避免循环发现

### 版面上写什么

标题：**MTBLS13729 用于检验算法能否把谱学改进转化为可审计的代谢候选**

工作流：

1. 统一正/负离子参考库与 precursor-mass 候选图；
2. official DreaMS / 新 shared embedding / P2b 三路注释；
3. 在查看 Rmu/RN 表型前冻结候选、置信度和 EIC 参数；
4. 5 ppm EIC、12 s apex，患者内配对；
5. raw、PQN、PQN+drift 三种归一化敏感性；
6. FDR 后做离子家族去冗余；Level 2 结论封顶。

数据规模：neg_rp 62 个冻结定量目标；pos_rp 555 个；Rmu/RN 主终点 10 对。

### 配什么图

画一条从“谱图”到“候选”再到“患者配对丰度”的完整流程；注释冻结点用一把锁表示，锁之后才出现组别标签。

右下角画一个患者内配对的小图，强调不是把全部样本当独立个体。

### 讲稿

“生物学部分现在不是算法主线的替代品，而是检验算法输出能否进入真实研究流程。我们选择 MTBLS13729，先用统一谱库和正确的 precursor-mass 候选图完成三路注释，然后在查看表型差异之前冻结候选、置信度和定量参数。之后才做 5 ppm EIC 重提取、患者内配对和多归一化敏感性分析。

这个顺序非常重要。如果先看到肿瘤与癌旁差异，再挑最顺眼的结构，就会形成循环论证。最终负离子冻结 62 个定量目标，正离子 555 个，主终点是 10 对 Rmu 与癌旁样本。身份结论统一封顶在 Level 2；没有标准品共洗脱和正交结构确认时，不会写成唯一化学结构。”

### 证据来源

- `docs/MTBLS13729_FROZEN_P2B_APPLICATION_PROTOCOL_20260828.md`
- `docs/MTBLS13729_FROZEN_BIOLOGY_RESULT_20260829.md`

---

## 第 24 页｜当前最稳的生物学锚点是 C20:4 酰基肉碱样信号，而不是机制定论

### 版面上写什么

标题：**C20:4 arachidonoylcarnitine-like 在 Rmu 配对组织中稳定升高 2.34–3.39 倍**

主结果：

- feature 3222，m/z 448.339463，59/59 正离子样本检出；
- 三路注释指向同一 IK14；19 张谱/样本支持；
- official max/median similarity = 0.8505/0.8091；实验性单折 E6 shared embedding = 0.8568/0.8166；
- raw：log2FC +1.760，p=.00204，q=.0266；
- PQN：+1.228，p=.01099，q=.0925；
- PQN+drift：+1.256，p=.00977，q=.0861。

其他候选：核苷样未知离子家族、黄嘌呤样、N1,N8-diacetylspermidine-like。六个 FDR10 feature 去冗余后最多五个离子家族。

结论边界：Level 2；支持“酰基肉碱稳态积累/脂肪酸利用异常假设”，不证明 β-氧化通量或具体酶。

### 配什么图

主图用患者配对 spaghetti plot，三种归一化各放一个小面板；旁边放 MS/MS mirror plot，标出支持峰与主要竞争异构体。

底部放 ion-family 网络小图，显示 feature1597 与 7489 合并，避免“六个 feature=六个代谢物”。

### 讲稿

“目前最稳的单分子生物学锚点是 feature 3222，对应 C20:4 arachidonoylcarnitine-like 候选。它在 59 个正离子样本中全部检出，三路注释给出同一个候选身份。官方 embedding 的最大和中位相似度分别为 0.8505 和 0.8091，实验性单折 E6 shared embedding 提高到 0.8568 和 0.8166，说明新表示主要增强了谱学证据一致性，而不是把候选翻成另一个分子。这里的 E6 只作真实样本证据一致性的应用复核，不能替代 E4-A 的多折权重结论。

在患者配对分析中，raw、PQN 和漂移校正三种处理的 log2 fold change 都为正，对应 2.34 到 3.39 倍升高。与此同时，我们对显著 feature 做了离子家族去冗余，六个 FDR10 feature 最多对应五个独立离子家族。这里可提出‘Rmu 中酰基肉碱稳态积累和脂肪酸利用异常’的假设，但身份仍是 Level 2，静态丰度也不能证明 β-氧化通量下降或某个具体酶发生改变。”

### 证据来源

- `docs/MTBLS13729_FROZEN_BIOLOGY_RESULT_20260829.md`
- `docs/PROJECT_RESEARCH_MASTER_SUMMARY_20260829.md`

---

## 第 25 页｜平台把研究方法固化为可追溯的注释流程，而不只是一组脚本

### 版面上写什么

标题：**MS/MS 注释平台已形成端到端原型：检索、证据、解释、统计与导出统一审计**

流程：

`上传 mzML/MGF`  
→ `precursor-mass candidate graph`  
→ `DreaMS/new embedding retrieval`  
→ `P2b safe routing`  
→ `confidence/FDR + Schymanski level`  
→ `peak/concept explanation`  
→ `paired statistics / pathway interface / export`

工程事实：

- feature/showspace 分支；前端与后端原型已成型；
- 正离子参考库 265,011 张、负离子 29,564 张；
- 约 207,787 个唯一 InChIKey；
- 先按 precursor mass 建图，再在窗口内排序；
- 每条结果保留模型版本、候选、得分、证据、弃权与哈希。

### 配什么图

主图应使用平台真实截图：候选列表 + 镜像谱图 + 证据解释侧栏。不要使用通用 UI 示意图。

下方配一个小型 provenance chain：checkpoint hash、library hash、candidate graph hash、result version。

### 讲稿

“除了模型实验，我们还把方法固化成一个可追溯的 MS/MS 注释平台原型。平台不是简单调用 cosine Top-k，而是先根据 precursor mass 构建候选图，再在质量窗口内执行 embedding 检索和候选证据融合。之后统一输出校准置信度、Schymanski 分级、峰级解释、配对统计和导出结果。

目前参考库包含约 26.5 万张正离子谱和 3 万张负离子谱，覆盖约 20.8 万个唯一 InChIKey。前后端已经在 feature/showspace 分支成型，但汇报中应表述为‘正在开发并准备开源的原型’，而不是已经完成正式发布。平台最重要的价值，是把 checkpoint、参考库、候选图、阈值和每条候选证据全部落盘，使算法结果能够被复查和复现。”

### 证据来源

- `docs/PROJECT_RESEARCH_MASTER_SUMMARY_20260829.md`
- `docs/ANNOTATION_TOOL_DELIVERY_ROADMAP_20260817.md`

---

## 第 26 页｜最终系统由一个共享表示和三个有边界的证据层组成

### 版面上写什么

标题：**我们形成的不是单一“化学规则模型”，而是可回退、可解释的分层注释系统**

全页系统图：

1. **Noise/ChemAware shared encoder**：通用、样本无关的新 embedding；
2. **P2b spectral expert**：常规候选中的峰/中性丢失证据；near-core 回退；
3. **Double mapping**：概念、结构环境和具体峰的解释证据；
4. **BioAware expert**：队列上下文与反应网络；默认弃权；
5. **Annotation platform**：候选生成、置信度、FDR、审计与生物统计。

输出不是一个强制结构，而是：`candidate ranking + confidence + evidence + conflict + abstention`。

### 配什么图

建议做唯一一张全系统架构图：左侧一张谱图，中央共享 encoder，右侧候选列表；P2b、双重映射、BioAware 分别从不同方向提供证据，最终汇入“校准与弃权”，再到生物学分析。

图中用实线表示已成立组件，用虚线表示未来 ChemAware embedding 与 BioAware context adapter。

### 讲稿

“把这些工作放在一起，最终系统并不是把规则、噪声和网络全部塞进一个黑盒。底层是一个通用、样本无关的共享 encoder；目前定向噪声已经使这个 embedding 在开发图上稳定改善。上层的 P2b 只提供候选组内谱学证据，并对 near-core 设置回退。双重映射不负责强制改名，而是解释模型依赖了哪些概念和峰。BioAware 则只在真实实验图和至少两个独立证据族共同支持时提供生化上下文，否则弃权。

最终输出也不应只是一个结构名称，而应包括候选排序、校准置信度、支持峰、冲突证据和弃权状态。这样的分层设计既保留了性能改进，也允许我们清楚地指出每一次改动来自共享表示、谱学候选证据还是生化上下文，避免再把不同层的收益混为一谈。”

### 证据来源

- `docs/PROJECT_RESEARCH_MASTER_SUMMARY_20260829.md`
- `docs/BIOAWARE_NETWORK_EXPERT_AND_CONTEXT_ADAPTER_PLAN_20260827.md`

---

## 第 27 页｜当前结论与下一步：先完成新权重的外部总评，再决定论文主张

### 版面上写什么

标题：**已证明“错误可定位、动作可迁移、系统可审计”；全面 SOTA 仍需新外部冻结总评**

左侧“现在可以说”：

1. DreaMS 残余错误集中在 near 与跨条件正例不足；
2. 匹配反事实实验确认部分峰证据偏差具有因果方向；
3. 定向噪声使共享 embedding 多折多种子稳定提高 +0.635pp；
4. P2b 在 P3 main 提高 +1.07pp，但 near-core 存在明确负边界；
5. embedding 中存在可解码概念与可定位峰证据；
6. 生物学应用获得 C20:4、核苷样和多胺候选。

右侧“现在不能说”：

- 新模型全面超过 DreaMS/SOTA；
- 3.85–5pp 动作上界已被模型学到；
- P2b 的 +3.91pp 可直接外推；
- ChemAware 规则微调已经成功；
- BioAware 已显著提高外部注释；
- Level 2 候选等于结构鉴定或机制证明。

底部下一步三项：

1. 冻结当前最优 shared encoder，构建全新 identity/formula 隔离外部测试；
2. 扩充真实跨条件 P-arm，并对 action-to-weight transfer 做机制消融；
3. 对 175 条双重映射桥完成概念删峰忠实性与检索方向四门验证。

### 配什么图

使用一张“证据阶梯”收尾：

`错误描述 → 匹配反事实 → 动作上界 → 共享权重 → 封存外部 → 生物学验证`

把前四级填实，第五级标“下一步”，第六级显示当前 case study 而非终局鉴定。

### 讲稿

“最后总结。目前我们已经证明三件核心事情：第一，DreaMS 的残余错误不是均匀随机，而是集中在 near、跨条件正例不足和局部峰证据失衡；第二，这些错误中存在相对匹配随机对照仍然成立的方向性峰动作；第三，其中一部分动作已经能够迁移进同一个共享 encoder，在五折三 seed 上稳定提高约 0.635 个百分点。

与此同时，P2b 在常规 P3 主面板上提供 1.07 个百分点的系统增益，但 near-core 的负结果明确规定了它的使用边界。化学双重映射已经建立可解码与峰输入忠实性，BioAware 和生物学应用则分别形成了严谨的负结果与候选发现。

下一步最重要的不是继续堆模块，而是冻结当前最优权重，在一个全新的 identity 和 formula 隔离外部基准上做一次性总评；同时扩大真实跨条件正臂，解释为什么动作上界只能部分迁移。只有这一步通过之后，我们才能把‘共享 embedding 改善’升级为‘全面优于官方 DreaMS’的论文主张。”

### 证据来源

- `docs/PROJECT_RESEARCH_MASTER_SUMMARY_20260829.md`
- `docs/NOISE_E4A_HIGHLR_MULTIFOLD_RESULT_20260827.md`

---

## 建议的附录页（不进入主讲，问答时调用）

### 附录 A｜官方 checkpoint 全部基线表

直接复用 `docs/OFFICIAL_DREAMS_CHECKPOINT_BASELINE_LEDGER_20260829.md`，按任务分组，不把不同协议混在同一排名表中。

### 附录 B｜错误机制分层的完整数量与阈值

展示 positive deficit、negative excess、shared-major-peak、neutral-loss convergence、cross-condition 的定义、阈值和重叠关系。

### 附录 C｜噪声动作完整 44-cell 矩阵与 max-T 多重性校正

主文只展示代表动作；附录保留所有预注册 cell、匹配控制比例、公式簇 sign-flip max-T 和被淘汰动作。

### 附录 D｜E4-A 15 个 fold-seed 的逐项结果

报告 Recall@1、near、MRR、corrected/introduced、preservation mean/p01、公式簇 CI。

### 附录 E｜P2b P3 全面板

同时展示 main、isomer、near-core、near+mid 和 secondary 面板，避免只展示最有利的 main。

### 附录 F｜BioAware 负对照

展示 leave-out、degree-preserving rewiring、seed scarcity、conflict/abstain 以及 post-hoc 结果为何不能作为正式性能。

### 附录 G｜生物学候选完整证据表

每个候选列 m/z、RT、加合物、IK14、谱数、相似度、归一化敏感性、q 值、离子家族和 Schymanski level。
