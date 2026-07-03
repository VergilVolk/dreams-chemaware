# 预训练目标

DreaMS 使用两个互补的自监督预训练目标。

---

## 目标 1: Masked Peak Prediction（掩码峰预测）

### 任务描述

```
原始谱图:  [mz₁, mz₂, mz₃, ..., mz₆₀]
              ↓ 随机遮住 30% 的峰
遮罩谱图:  [mz₁, ???, mz₃, ..., ???]
              ↓ 模型预测被遮住的峰
预测输出:  [P(mz₂), P(mz₆₀), ...]  → 每个遮罩位的 m/z 概率分布
```

### 关键设计：分类而非回归

**为什么不用回归（MSE）？**

> "A regression model may converge at predicting the average value"

质谱中可能存在多个合理的 m/z 值补全一个遮罩位。回归会预测平均值（可能不对应任何合理值），分类则输出完整概率分布。

**实现**：

```python
# dream.py spec_ssl_step()

# Step 1: 把真实 m/z 映射到 one-hot 类别
real_mz = su.to_hot(real[..., [0]], max_val=self.dformat.max_mz, 
                     bin_size=self.hot_mz_bin_size)

# Step 2: 预测 m/z 类别
pred_mz = self.ff_out(pred_embs[mask])  # (masked_peaks, C)

# Step 3: Focal Loss（处理类别不平衡）
loss, p_mz = self.mz_masking_loss(pred_mz, real_mz)
# FocalLoss(gamma=args.focal_loss_gamma, return_softmax_out=True)
```

### 遮罩策略

```
1. 随机选择 30% 的峰（排除 precursor）
2. 将被遮峰的 m/z 替换为特殊值（mask_val = 0 或 -1）
3. 强度保持不变 → 模型仍能看到"这里有个峰"
4. 可选：同时预测强度（mask_peak_hot 模式）
```

参考 BERT 的 80-10-10 策略（可选）：
- 80%：替换为 mask token
- 10%：替换为随机 m/z
- 10%：保持不变（但仍然预测）

---

## 目标 2: Retention Order Prediction（保留顺序预测）

### 任务描述

```
谱图 A（部分遮罩）  +  谱图 B（部分遮罩）
              ↓
         模型预测：A 还是 B 先洗脱？
              ↓
     label = 1 if A 先于 B, else 0
```

### 化学直觉

保留时间（RT）主要受分子极性影响。极性大的分子在反相色谱中先洗脱。预测保留顺序迫使模型学习**分子极性**信息——这是与碎裂化学**正交**的物理维度。

### 实现

```python
# dreams.py step()

# Step 1: 分别对两张谱图做 mask prediction
embs1 = model(spec_mask_1)
embs2 = model(spec_mask_2)

# Step 2: 拼接 precursor embeddings
prec_embs12 = torch.cat([embs1[:, 0, :], embs2[:, 0, :]], dim=-1)

# Step 3: 二分类预测保留顺序
loss_ro = F.binary_cross_entropy(
    F.sigmoid(self.ro_out(prec_embs12)).squeeze(), ro_label
)
```

### 总损失

```python
L_total = (1 - w_ro) * L_mask + w_ro * L_ro
```

其中 $w_{ro} \approx 0.3$ 控制保留顺序损失的权重。

---

## 为什么这两个目标互补

| | Masked Peak Prediction | Retention Order |
|---|---|---|
| 学什么 | 碎裂化学（峰间关系） | 分子极性（分子间关系） |
| 输入 | 单张谱图 | 两张谱图对 |
| 标签 | 自监督（被遮的 m/z） | 自监督（RT 先后） |
| 空间 | 嵌入空间内部结构 | 嵌入空间全局组织 |

两者共同作用 → 嵌入既编码了局部碎裂信息，又有全局理化性质结构。
