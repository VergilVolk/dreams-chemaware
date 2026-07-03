# 训练配置

## 预训练配置

| 参数 | 值 | 说明 |
|------|:---:|------|
| 预训练数据 | GeMS (7 亿谱图) | GNPS/MassIVE 仓库 |
| Batch Size | 256 | 跨 8 × A100 GPU |
| 学习率 | 3e-4 | Adam 优化器 |
| 学习率调度 | Noam Scheduler | warmup 4000 steps → inverse sqrt decay |
| 序列长度 | 60 峰 | 取前 n 高强度 |
| 掩码率 | 30% | 随机遮罩 |
| 掩码策略 | 确定性随机 | 非 BERT 80-10-10 |
| 保留顺序权重 | 0.3 | w_ro in total loss |
| 精度 | FP32 | 训练精度 |
| 训练时长 | ~2 周 | 8 × A100 80GB |

## Noam 学习率调度

```python
# schedulers.py NoamScheduler
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
```

- Warmup 阶段：线性增长
- Decay 阶段：反平方根衰减
- 经验上比 Cosine Annealing 更适合 Transformer 大规模预训练

## Focal Loss

用于掩码峰预测，处理 m/z bin 的极端类别不平衡：

```python
FocalLoss(gamma=2.0, alpha=None)
# FL = -alpha * (1 - p_t)^gamma * log(p_t)
```

- gamma=2：降低易分类样本的权重
- 大多数 m/z bin 的概率接近 0，Focal Loss 让模型聚焦于"难预测"的 bin

## 批次构建策略

### MaxVarBatchSampler

```python
# 优先选取"峰数量变化大"的谱图组成 batch
# 让模型适应不同峰数的输入
MaxVarBatchSampler(dataset, batch_size=256, 
                   max_var_features='n_peaks')
```

### 保留顺序对构造

```python
# 20% 概率：同一样本内交换正负标签
# 防止模型利用谱图质量等表面特征判断 RT
```

## 验证策略

训练时每 epoch 末自动运行多种验证：

```python
# dreams.py on_validation_epoch_end()
SpecRetrievalValidation  (AUC)
CorrelationValidation    (Pearson r)
KNNValidation            (k-NN 准确率)
```

全部在 NIST20/MoNA 标注数据上评测，不参与训练。

## 微调配置

| 参数 | 对比微调 | 指纹预测 | 氟检测 |
|------|:---:|:---:|:---:|
| 学习率 | 5e-5 | 3e-5 | 5e-5 |
| Epochs | 5-10 | 10-20 | 30 |
| Batch | 64 | 64 | 64 |
| 冻结 backbone | 前几 epoch | 前几 epoch | 部分解冻 |
| 优化器 | Adam | Adam | Adam |
