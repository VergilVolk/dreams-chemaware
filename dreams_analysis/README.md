# DreaMS 完整复现分析

> Self-supervised learning of molecular representations from millions of tandem mass spectra using DreaMS
> Bushuiev et al., *Nature Biotechnology*, 2025

本目录对 DreaMS 的核心机制进行逐层拆解，涵盖架构、注意力、预训练目标、微调策略和数据预处理。

---

## 目录结构

```
dreams_analysis/
├── README.md                        ← 本文件：总览与核心公式
├── 01_architecture.md               ← 架构全景：PeakEncoder → SpectrumEncoder → PeakDecoder
├── 02_attention_graphormer.md       ← 核心：Graphormer 自注意力机制（关键创新）
├── 03_pretraining.md                ← 预训练目标：Masked Peak Prediction + Retention Order
├── 04_finetuning.md                 ← 微调策略：对比学习、指纹预测、氟检测
├── 05_data_preprocessing.md         ← 数据流水线：MGF→HDF5→SpectrumPreprocessor
├── 06_training_config.md            ← 训练配置：Noam调度、Focal Loss、批次策略
├── 07_code_structure.md             ← 代码架构：关键文件与数据流
├── fig1_architecture.png            ← 图1：三阶段架构全景图
├── fig2_graphormer.png              ← 图2：Graphormer 注意力公式 + 化学解释
├── fig3_pretraining.png             ← 图3：双预训练目标示意
└── fig4_comparison.png              ← 图4：Graphormer vs ChemAware 对比
```

---

## 一句话总结

**DreaMS 是一个在 7 亿张未标注 MS/MS 谱图上自监督预训练的 Transformer，通过预测被遮住的质谱峰和学习保留顺序，自发涌现出分子结构表征。**

---

## 核心创新点

1. **Graphormer 注意力**：将峰对 m/z 差显式编码为注意力偏置，使模型天然关注化学合理的碎裂路径
2. **Fourier 特征编码**：m/z 值通过傅里叶特征编码，而非离散 tokenization，保留精确质量信息
3. **双预训练目标**：掩码峰预测（学习碎裂化学）+ 保留顺序预测（学习分子极性）
4. **分类式掩码预测**：将 m/z 值分箱为 one-hot 类别，避免回归的平均效应
5. **解释性涌现**：预训练后可用线性探测预测 MACCS 指纹位点，证明模型"理解"了分子结构

---

## 核心公式

### 输入表示

每个谱图表示为 n 个峰的序列：

$$X = \{ (m_1, i_1), (m_2, i_2), ..., (m_n, i_n) \}$$

每个峰编码为：

$$\text{PeakEncoder}(m, i) = \text{FFN}_F(\Phi(m)) \parallel \text{FFN}_P(m, i)$$

其中 $\Phi(m) \in \mathbb{R}^{d_F}$ 是 Fourier 特征，$\parallel$ 表示拼接。

### Graphormer 注意力

这是 DreaMS **最关键的创新**。第 $l$ 层的注意力分数计算为：

$$\alpha_{ij}^{(l)} = \frac{q_i^{(l)} \cdot k_j^{(l)}}{\sqrt{d_k}} + \phi^{(l)}\big(F_i^{(l)} - F_j^{(l)}\big)$$

- 第一项：标准 dot-product 注意力
- 第二项：**Graphormer 项**——峰 i 和 j 的 Fourier 特征差经过线性投影 $\phi^{(l)}$ 后加到注意力分数上
- $F_i^{(l)} \in \mathbb{R}^{d_F}$：第 i 个峰的 Fourier 特征在当前层的表示

**意义**：模型不需要"猜"哪些质量差重要——m/z 差被显式编码为注意力偏置。这等价于在自注意力中嵌入了**连续版本的相对位置编码**，但因为编码的是化学质量差而非序列位置，它自动学会了关注中性丢失等化学有意义的峰对。

### 预训练损失

Masked Peak Prediction（分类式）：

$$\mathcal{L}_{\text{mass}} = -\frac{1}{|M|} \sum_{k \in M} \sum_{c=1}^{C} y_{k,c} \log \hat{y}_{k,c}$$

- $M$：被遮住的峰集合（30% 峰值）
- $C$：m/z 分类的 bin 数
- 使用 Focal Loss 处理类别不平衡

Retention Order Prediction：

$$\mathcal{L}_{\text{order}} = -\big[z \log \hat{p} + (1-z) \log (1-\hat{p})\big]$$

- 两张谱图比较保留时间先后，$z \in \{0, 1\}$

总损失：

$$\mathcal{L} = \mathcal{L}_{\text{mass}} + w_{\text{order}} \cdot \mathcal{L}_{\text{order}}$$

---

## 关键数据事实

| 项目 | 数值 |
|------|:---:|
| 预训练数据 | GeMS：7 亿张 GNPS/MassIVE 谱图 |
| 谱图长度 | 60 个峰（取前 n 高） |
| 嵌入维度 | d_model = 1024 |
| 注意力头数 | 8 |
| Transformer 层数 | 7 |
| Fourier 特征维数 | 980 |
| 峰值嵌入维数 | 44 |

---

## 为什么 DreaMS 有效

```
传统方法：
  MIST, SIRIUS, CANOPUS — 需要化学式标注 / 组合碎裂树 / SVM 分类器

DreaMS 方法：
  7亿谱图 → 自监督预训练 → 1024维嵌入 → 下游任务只需少量微调

关键insight：
  掩码峰预测这个任务，如果做的是"分类"而非"回归"，
  模型需要理解峰与峰之间的化学关系才不容易被混杂。
  这种化学关系的学习，自然形成了分子结构表征。
```

---

## 与你的 chem_aware 框架的关联

| DreaMS | chem_aware |
|--------|-----------|
| Graphormer：m/z 差 → 注意力偏置 | ChemicalRuleEngine：化学规则 → 注意力偏置 |
| 从数据学习偏置映射 | 从化学第一性原理编码偏置 |
| 需要大量数据训练 $\phi^{(l)}$ | 规则库可零样本工作 |
| 学到的是"统计上常见"的碎裂 | 编码的是"化学上合理"的碎裂 |

**DreaMS 的 Graphormer 是你的 chem_aware 框架的重要基线**——两者都在注意力层面注入化学先验，但一个是数据驱动的（学出来的），一个是知识驱动的（编码进去的）。理解 Graphormer 的优劣是对抗协同框架设计的前提。
