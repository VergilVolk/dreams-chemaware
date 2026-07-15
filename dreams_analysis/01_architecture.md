# DreaMS 架构全景

## 三阶段流水线

```
原始谱图 X (n, 2)
    │ [m/z, intensity]
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    PeakEncoder                               │
│                                                             │
│   m/z ──→ FourierFeatures (sin/cos) ──→ FFN_F ──→ F_j      │
│                                                             │
│   [m/z, intensity] ──→ FFN_P ──→ P_j                       │
│                                                             │
│   PeakEmb_j = F_j ∥ P_j    (Fourier ∥ Peak)                 │
│   维度: d_model = d_fou![alt text](image.png)rier + d_peak = 980 + 44 = 1024       │
└───────────────────────────┬─────────────────────────────────┘
                            │ (n, 1024)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  SpectrumEncoder                             │
│                                                             │
│   7 层 Transformer Encoder (Pre-Norm, ScaleNorm)              │
│                                                             │
│   每层:                                                      │
│     ┌──────────────────────────────────┐                    │
│     │  MultiheadAttention (8 heads)     │                   │
│     │  + Graphormer bias term          │  ← 核心创新        │
│     │  + Residual + Norm               │                    │
│     └──────────────┬───────────────────┘                    │
│                    ▼                                         │
│     ┌──────────────────────────────────┐                    │
│     │  FeedForward (ReLU, 4× expansion) │                   │
│     │  + Residual + Norm               │                    │
│     └──────────────────────────────────┘                    │
│                                                             │
│   输入:  (n, 1024)                                           │
│   输出:  (n, 1024)  — 精炼后的峰值表征                        │
│   s_0 (precursor token) = 谱图级表征                          │
└───────────────────────────┬─────────────────────────────────┘
                            │ (n, 1024)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    PeakDecoder                               │
│                                                             │
│   根据任务选择不同的输出头:                                    │
│                                                             │
│   预训练:  s_k → W_mass ∈ R^{C × 1024} → m/z one-hot         │
│            s_k → W_intensity → intensity bin                │
│                                                             │
│   指纹预测: s_0 → FFN → MACCS bits                           │
│   氟检测:   s_0 → FFN → [0, 1]                               │
│   对比学习: s_0 → Linear → triplet loss                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 关键设计决策

### 1. Fourier 特征替代 Tokenization

**为什么不用 tokenization？**
- m/z 值精度要求高（0.01 Da），token 数量会爆炸
- 相近的 m/z 值应该有相近的嵌入——Fourier 特征天然满足连续性

**实现**：
```python
Φ(m) = [cos(2π·m·b₁), sin(2π·m·b₁), ..., cos(2π·m·b_k), sin(2π·m·b_k)]
```
其中 b₁,...,b_k 是可学习的频率参数（初始化为对数均匀分布）。

### 2. Precursor Token

人工添加的前置峰（m/z = precursor_mz, intensity = 1.1），代表母离子。在所有层中作为"主节点"聚合谱图信息。谱图级嵌入从 s_0 提取。

### 3. ScaleNorm

使用 ScaleNorm 而非 LayerNorm：
```python
ScaleNorm(x) = x * (g / ||x||)
```
其中 g 是初始化为 √d_model 的可学习标量参数。

### 4. Pre-Norm 架构

```python
x = norm(x)
x = Attention(x) + residual     # norm → attention → residual
x = norm(x)
x = FFN(x) + residual           # norm → FFN → residual
```

Pre-Norm 比 Post-Norm 在大规模训练中更稳定，不易梯度爆炸。

---

## 维度变化追踪

```
输入:     (batch, n=60, 2)              # [m/z, intensity]
Precursor:(batch, n=61, 2)              # 插入前置峰
PeakEmb:  (batch, n=61, 1024)           # Fourier(980) + Peak(44)
SpecEnc:  (batch, n=61, 1024)           # 7层 Transformer 精炼
Precursor:(batch, 1024)                 # s_0 = 谱图表征
输出:     取决于任务
```

## 参数总量

- PeakEncoder: Fourier(512 freqs × 2) + FFN_F(1024→512→980) + FFN_P(2→44→44) ≈ 1M
- SpectrumEncoder: 7 × (MultiheadAttn(≈4M) + FFN(≈8M)) ≈ 84M
- PeakDecoder: 取决于任务 ≈ 1-20M
- 总计 ≈ 100M 参数
