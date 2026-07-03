# Graphormer 自注意力机制 — DreaMS 核心创新

## 论文原文

> "we explicitly enrich the attention mechanism with all pairwise m/z differences including neutral losses"

> "This enables the transformer to attend directly to neutral losses without extra tokens or modifications to the dot-product attention mechanism"

---

## 公式

### 完整注意力计算

第 $l$ 层，峰 $i$ 和峰 $j$ 之间的注意力分数：

$$\alpha_{ij}^{(l)} = \underbrace{\frac{q_i^{(l)} \cdot k_j^{(l)}}{\sqrt{d_k}}}_{\text{标准 dot-product}} + \underbrace{\phi^{(l)}\big(F_i^{(l)} - F_j^{(l)}\big)}_{\text{Graphormer 项}}$$

其中：
- $q_i, k_j \in \mathbb{R}^{d_k}$：第 i 峰的 query，第 j 峰的 key（$d_k = d_{model}/n_{heads}$）
- $F_i, F_j \in \mathbb{R}^{d_F}$：第 i 峰和第 j 峰的 Fourier 特征在当前层的表示
- $\phi^{(l)}: \mathbb{R}^{d_F} \to \mathbb{R}$：**可学习的线性投影**，将 Fourier 特征差映射为标量偏置

### 代码实现

```python
# layers.py MultiheadAttention.forward()

# 1. 标准 dot-product 注意力
att_weights = torch.einsum('bhnd,bhdm->bhnm', q, k.transpose(-2, -1))
att_weights = att_weights * self.scale

# 2. Graphormer 项
if graphormer_dists is not None:
    if self.d_graphormer_params:  # parameterized mode
        # (bs, n, n, d_F) → (bs, n, n, n_heads) → (bs, n_heads, n, n)
        att_bias = self.lin_graphormer(graphormer_dists).permute(0, 3, 1, 2)
    else:  # simple mode
        # (bs, n, n, d_F) → (bs, 1, n, n)
        att_bias = graphormer_dists.sum(dim=-1).unsqueeze(1)
    att_weights = att_weights + att_bias

# 3. Padding mask + softmax
att_weights.masked_fill_(mask, -1e9)
att_weights = F.softmax(att_weights, dim=-1)
```

### graphormer_dists 的来源

```python
# dreams.py DreaMS.forward()

# Fourier 特征差（逐元素）
if self.d_fourier:
    graphormer_dists = fourier_features.unsqueeze(2) - fourier_features.unsqueeze(1)
    # shape: (batch, n, n, d_fourier=980)
```

---

## 为什么 Graphormer 比标准 Self-Attention 更适合质谱

### 标准 Self-Attention

$$
\alpha_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}
$$

峰之间**没有显式的结构关系编码**。模型必须通过数据隐式学习哪些质量差的峰对是相关的。这需要大量数据，且对罕见碎裂模式学习不充分。

### Graphormer Self-Attention

$$
\alpha_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} + \phi(F_i - F_j)
$$

**峰对的 m/z 差直接参与注意力计算**。$\phi$ 是一个线性层，学习"什么质量差值得高注意力"。

### 化学直觉

| m/z 差 (Da) | 化学意义 | Graphormer 学到的 $\phi$ 值 |
|:---:|------|:---:|
| 0 | 自身 | 0（被 Softmax 前排除） |
| 18.01 | H₂O 丢失（极常见） | 高正偏置 |
| 27.99 | CO 丢失（极常见） | 高正偏置 |
| 17.03 | NH₃ 丢失（含 N 化合物） | 中等正偏置 |
| 25.00 | 无化学意义 | 零或负偏置 |

**Graphormer 从 7 亿谱图中学习到了化学上合理的质量差→注意力偏置映射**——这与人类质谱专家总结的中性丢失列表高度一致。

---

## Graphormer vs 标准位置编码

| | 标准 Positional Encoding | Graphormer (DreaMS) |
|---|---|---|
| 编码什么 | 序列中的位置 i | 峰的质量 m_i |
| 距离度量 | |i - j|（离散） | |m_i - m_j|（连续） |
| 编码方式 | sin/cos 固定频率 | Fourier 特征 + 可学习投影 $\phi$ |
| 物理意义 | "两个 token 有多近" | "两个峰的碎裂关系" |

**关键差异**：质谱峰没有自然顺序——峰的"位置"就是它的 m/z 值。所以 Graphormer 本质上是**利用物理量（m/z）作为结构先验**。

---

## 论文中的实验证据

### Fig 3c: Linear Probing（线性探测）

```
预训练进度 → 冻结 backbone → 训练线性分类器预测 MACCS 指纹位点
结果：预训练越充分，MACCS 预测越准
证明：模型自发学会了分子子结构信息
```

### Fig 3d: Attention Visualization（注意力可视化）

```
谱图上的每个峰着色 = 所有注意力头对该峰的最大注意力值
蓝色 = 高注意力，黄色 = 低注意力
结果：高注意力集中在代表分子碎片的高强度峰上，噪声峰被忽略
证明：Graphormer 驱动模型关注化学有意义的峰
```

---

## 局限

1. **$\phi$ 是线性投影**——只能学到单调的质量差→偏置关系，无法捕获非线性碎裂规律
2. **需要数据学习**——罕见碎裂模式的 $\phi$ 值学不充分
3. **无因果知识**——$\phi$ 学到"Δm=18 经常重要"，但不知道"为什么重要"（H₂O 丢失）
4. **对异构体不敏感**——Δm 相同的两种不同碎裂路径无法区分
