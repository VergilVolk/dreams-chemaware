# MS2DeepScore 加噪方法调研与迁移依据（Step 3 前置）

> 日期：2026-08-17。目的：为 Step 3 的"MS2DeepScore 三件套"噪声小实验提供精确参数、
> 迁移到 DreaMS 的设计判断、以及与近期方法的结合依据。
> **口径约束**：只给"方法出处 + 迁移依据 + 潜力来源"，不承诺"必然提高"；最终以 G3 消融
> 数字为准。本文与 `NOISE_AUGMENTATION_LITERATURE_JUSTIFICATION_20260816.md`（掩码峰建模）
> 是**两条不同的噪声腿**：那份讲 DreaMS 自身的掩码预训练，这份讲 MS2DeepScore 的强度域噪声。

---

## 1. MS2DeepScore 的加噪方法（精确参数）

来源（原文 + 官方实现）：MS2DeepScore，Huber et al., *J. Cheminform.* 2021
（biorxiv [10.1101/2021.04.18.440324](https://www.biorxiv.org/content/10.1101/2021.04.18.440324v1.full)；
PMC 版 [PMC8556919](https://pmc.ncbi.nlm.nih.gov/articles/PMC8556919/)；代码
[matchms/ms2deepscore](https://github.com/matchms/ms2deepscore)，[PyPI 0.2.0](https://pypi.org/project/ms2deepscore/)）。

### 1.1 输入预处理（噪声施加在哪个域上，必须先讲清）

1. **基础过滤**：删掉强度 < 最大峰 0.1% 的峰（去噪）。
2. **binning**：m/z 10.0–1000.0 Da，等宽 bin（原文"up to 10,000 bins"；`SpectrumBinner`
   常用 1000–10000 bin）。
3. **强度变换 = 开根号**：`peak_scaling=0.5` → `intensity^0.5`。这是官方 API 参数
   （PyPI 文档 `peak_scaling=0.5` 即开根号）。

### 1.2 三件套（每个训练样本都施加，训练数据生成时 on-the-fly）

| # | 名称 | 精确参数 | 作用域 |
|---|---|---|---|
| 1 | **删弱峰** | 随机选 **0–20%** 的、强度 **< 0.4**（**变换前**、归一化 0–1 域）的非零 bin，置 0 | 原强度域 |
| 2 | **强度抖动** | 每个非零 bin 强度 ×随机因子 **[0.6, 1.4]**（即 ±40%） | 变换后（根号）域 |
| 3 | **加峰** | 随机选 **0–10** 个零强度 bin，强度置 **[0, 0.01]** 随机值 | 变换后（根号）域 |

关键语义：**删弱峰**的 0.4 阈值是在**归一化强度（0–1，base peak=1）**上判的——即"随机
删掉至多 20% 的、低于 40% 基峰的弱峰"；**抖动**和**加峰**是在**开根号后**的强度上做。

### 1.3 训练上下文（为什么这样做）

- 架构：Siamese 网络，base 500→500→200，head = 余弦相似度；输入是 binning+根号后的谱。
- 目标：预测谱对的**分子结构相似度（Tanimoto，RDKit 2048-bit）**，MSE 损失。
- 超参：Adam lr=0.001、batch=32、dropout=0.2、batch norm、L1/L2=1e-6、早停 5 epoch。
- 数据：>100k 谱 / ~14k 唯一分子。

**加噪的目的**：逼 Siamese 的嵌入对这三种扰动**不变**（余弦不随删峰/抖动/加峰漂移）。
这正是我们 Step 3 门 G3"锁定尺上噪声一致性↑（原谱 vs 抖动谱余弦）"的**文献级定义**。

---

## 2. 迁移到 DreaMS 的设计判断（关键，别照抄）

MS2DeepScore 在 **binning + 开根号** 的谱上做噪声；DreaMS 用**自己的 token 表示**
（precursor token + 峰 token），峰强度已是**归一化 0–1**（base peak=1.0，见
`NOISE_AUGMENTATION_LITERATURE_JUSTIFICATION_20260816.md` 的"可掩峰 = intensity∈(0,1)"）。

因此：**迁移"菜单"，不迁移 binning/开根号**。

| MS2DeepScore 原始 | 迁移到 DreaMS（token 峰） |
|---|---|
| 删弱峰：删 0–20% 的强度<0.4 的 bin | 删 0–20% 的归一化强度<0.4 的峰 token |
| 抖动 ±40%（根号域） | 峰强度 ×[0.6, 1.4]（0–1 域，幅度照搬） |
| 加峰 0–10 个、强度 0–0.01 | 加 0–10 个弱峰 token、强度 ~0–0.01 |

**与 DreaMS 自身掩码预训练的关系（重要）**：
- **重叠**：MS2DeepScore"删弱峰" ≈ DreaMS 掩码预训练的"掩峰"。但方向相反——DreaMS
  预训练**强度正比地优先掩强峰**（30%），MS2DeepScore 删的是**弱峰**（<0.4，≤20%）。
  两者补的是不同象限（强峰鲁棒 vs 弱峰鲁棒），不冗余。
- **全新轴（本次真正的增量）**：**强度抖动 ±40%** 和**加峰**。DreaMS 预训练只掩峰、
  **从不抖动、从不加峰**。抖动 = "改丰度"，正是 Step 1 证过的"跨 CE gap（0.676→0.19）
  真实数据几乎为零、只能靠噪声兜底"的那个杠杆——**抖动是 CE 代理**。

---

## 3. 近期（2024–2026）噪声/增强方法（结合候选）

计划要求"与近期加噪声方法结合"，调研到的可结合项：

| 来源 | 方法 | 可结合点 | 出处 |
|---|---|---|---|
| MALDI Transformers（2024） | "峰判别"预训练：峰级损失 + 物种损失（λ~Bernoulli(0.01)） | 峰级监督 + 概率混合损失 | [biorxiv 2024.01.18.576189](https://www.biorxiv.org/content/10.1101/2024.01.18.576189v1.full) |
| 光谱对比学习（Mendeleev Comm. 2024） | 合成谱对比学习；增强=高斯噪声 + **峰位移 + 强度变 + 峰宽变**；triplet + 硬样本挖掘、余弦、单位球 | 与我们的"triplet + 硬挖掘"同款范式；补 **m/z 位移**轴 | 见检索摘要 |
| 空间质谱增强（Frontiers 2025） | **m/z 位移增强**（±X m/z）稀疏数据集 | m/z 域新轴（DreaMS 预训练没有位移增强） | [fspas.2025.1706125](https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2025.1706125/full) |
| CoRe-Gen（2026） | "频率感知指纹损坏"——在**标签/解码侧**建模结构噪声 | 概念迁移：噪声建模在**标签侧**而非仅输入侧 | [arXiv 2605.12980](https://arxivlens.com/paperview/details/core-gen-robust-spectrum-to-structure-generation-under-imperfect-fingerprint-conditions-2873-88681ae4) |
| DreaMS（2026，我们自己） | 掩码峰预测（30% 峰掩）+ 保留序预测 | 已有：掩峰=删峰；缺：抖动/加峰/位移 | [Nat. Biotech. s41587-025-02663-3](https://www.nature.com/articles/s41587-025-02663-3) |

**结合结论**：MS2DeepScore 三件套（删弱峰/抖动/加峰）打**强度域**；近期方法补的是
**m/z 位移轴**（Frontiers、光谱对比学习）和**标签侧噪声**（CoRe-Gen 概念）。Step 3 小实验
先只上 MS2DeepScore 强度域三件套（可解析、可归因），m/z 位移作为**消融扩展项**，不混着上。

---

## 4. 诚实边界（必须写清）

1. **无一篇文献直接证明**"加噪 → 提高同分异构体区分 / 10ppm 检索"这条因果链。MS2DeepScore
   的加噪目的是**结构相似度预测的噪声不变性**，不是"提高身份判别"。迁移到 DreaMS 的
   "抖动→跨 CE 鲁棒"是我们**待验证的假设**（Step 1 已给出跨 CE 真实数据≈0 的动机）。
2. 三件套参数（0.4 / ±40% / 0.01）是 MS2DeepScore 在其 binning+根号域上定的，迁移到
   DreaMS 的 0–1 峰域后**阈值是否最优需重扫**（Step 3 小实验调噪声强度/菜单）。
3. 结论以 G3 门数字为准：噪声一致性↑ + 检索/子群指标不降 + 3 种子方向一致。不过 → 调
   强度/菜单重跑，仍不过 → 停。
