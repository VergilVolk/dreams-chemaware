# 微调策略

DreaMS 展示了多种微调范式，针对不同的下游任务。

---

## 微调范式概览

| 范式 | 任务 | 头结构 | 损失函数 | 训练数据 |
|------|------|--------|---------|---------|
| **对比微调** | 谱图相似度 | Linear(1024 → 1024) | Triplet Margin | MoNA 5,500 分子 |
| **指纹预测** | 分子指纹 | FFN(1024 → 2048) | CosSim/SmoothIoU | NIST20/MoNA |
| **二分类** | 氟检测 | FFN(1024 → 1) | Focal Loss (binary) | 含氟/不含氟标注 |
| **回归** | 分子性质 | FFN(1024 → 1) | MSE | 10 种理化性质 |

---

## 1. 对比微调（Contrastive Fine-tuning）

### 论文动机

> "we refine the embedding space through contrastive fine-tuning"

预训练的嵌入在区分同分异构体方面不够好。对比微调通过 triplet loss 拉近同分子谱图、推远不同分子谱图。

### 实现

```python
# heads.py ContrastiveHead

# 输入: (anchor, positive, negative) 三元组
emb   = model(spec)          # anchor (bs, 1024)
pos   = model(pos_specs)     # positives (bs, n_pos, 1024)
neg   = model(neg_specs)     # negatives (bs, n_neg, 1024)

# Triplet Margin Loss
cos_sim_pos = F.cosine_similarity(emb, pos, dim=-1)
cos_sim_neg = F.cosine_similarity(emb, neg, dim=-1)
loss = clamp(margin + (-cos_sim_pos) - (-cos_sim_neg), 0)
# margin = 0.3
```

**效果**：Fig 4b 显示，对比微调后的 DreaMS 在 Tanimoto 相似度预测上超越了 MS2DeepScore 等专用方法。

---

## 2. 指纹预测（Fingerprint Prediction）

### 任务

从谱图直接预测分子指纹（MACCS 166-bit 或 Morgan 2048-bit）。

### 实现

```python
# heads.py FingerprintHead

emb = self.backbone(spec)           # (bs, 1024) precursor embed
pred_fp = self.head(emb)            # (bs, 2048) fingerprint bits
loss = CosSimLoss(pred_fp, real_fp) # cosine similarity loss
```

### 评测：FingerprintInChIRetrieval

```
预测指纹 → 余弦相似度搜索候选分子库 → 
看 Top-K 中是否包含正确分子（InChI connectivity block 匹配）
```

这是最接近"分子结构注释"的下游评测任务。

---

## 3. 氟检测（Fluorine Detection）

### 设计动机

论文特别提到氟检测是因为含氟化合物在自然界中极少，但在药物中常见。DreaMS 在含氟检测上达到 SOTA，证明了预训练嵌入对化学性质的敏感性。

### 实现

```python
# heads.py BinClassificationHead

emb = self.backbone(spec)
f_prob = F.sigmoid(self.head(emb))  # [0, 1]
loss = FocalLoss(f_prob, label)     # binary focal loss
```

---

## 4. 线性探测（Linear Probing）

### 最简微调

冻结 backbone → 训练单层 Logistic Regression → 预测 MACCS 指纹位点。

**这是论文中最核心的解释性实验**——证明预训练嵌入已经包含分子子结构信息，不需要额外微调。

```python
# No head training needed - just:
log_reg = LogisticRegression()
log_reg.fit(embeddings, MACCS_labels)
recall = log_reg.score(test_embeddings, test_labels)
```

---

## 微调策略对比

| | 线性探测 | 对比微调 | 全微调 |
|---|---|---|---|
| 冻结 backbone | ✅ | ✅（部分） | ❌ |
| 训练参数 | ~2K | ~2M | 全量 |
| 训练时间 | 分钟 | 小时 | 天 |
| 效果 | 验证预训练质量 | 最好的嵌入质量 | 任务特定最优 |

论文主要使用线性探测验证预训练，对比微调优化嵌入。
