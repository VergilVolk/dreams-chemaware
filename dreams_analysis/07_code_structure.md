# 代码架构 — 关键文件与数据流

## 核心文件地图

```
dreams/
├── models/
│   ├── dreams/
│   │   ├── dreams.py          ← DreaMS 主模型（PyTorch Lightning）
│   │   └── layers.py          ← MultiheadAttention + TransformerEncoder + FeedForward + ScaleNorm
│   ├── heads/
│   │   └── heads.py           ← 微调头：ContrastiveHead, FingerprintHead, BinClassificationHead, RegressionHead
│   ├── layers/
│   │   ├── fourier_features.py ← FourierFeatures 编码器
│   │   └── feed_forward.py    ← 通用 FFN 构建器
│   └── optimization/
│       ├── schedulers.py      ← NoamScheduler
│       ├── losses_metrics.py  ← FocalLoss, CosSimLoss, FingerprintMetrics
│       └── samplers.py        ← MaxVarBatchSampler
├── api.py                     ← 用户入口：dreams_embeddings(), DreaMSAtlas, DreaMSSearch
├── utils/
│   ├── data.py                ← SpectrumPreprocessor, MSData, MaskedSpectraDataset
│   ├── spectra.py             ← 谱图工具：parse, trim, pad, to_hot, from_hot
│   ├── dformats.py            ← DataFormatA/B/C 质量过滤器
│   └── io.py                  ← MGF/MSP/MzML 读写
├── training/
│   ├── train.py               ← 训练主入口
│   └── train_argparse.py     ← 参数解析 + 验证
└── definitions.py             ← 全局常量/路径定义
```

## 关键数据流

### 推理流程

```
dreams_embeddings('spectra.mgf')                   # api.py:222
  → dreams_predictions(model, spectra)              # api.py:132
    → PreTrainedModel.from_name('DreaMS_embedding') # 加载 ContrastiveHead
    → SpectrumPreprocessor(spec)                    # 预处理谱图
    → model(spec)                                   # 前向传播
      → PeakEncoder (Fourier + FFN)                 # dreams.py:152-169
      → SpectrumEncoder (7× Graphormer Attention)   # dreams.py:179-183
        → MultiheadAttention(q,k,v, graphormer)     # layers.py:53-110
      → s_0 (precursor embedding)                   # 1024-dim
    → cosine_sim(query, reference)                  # 检索
```

### 训练流程

```
python train.py --dataset_path data.hdf5            # train.py
  → MaskedSpectraDataset(hdf5, preproc, n_samples)  # data.py
    → __getitem__: 动态遮罩 + RT 对
  → DreaMS(args, spec_preproc)                       # dreams.py:23
  → Trainer.fit(model, datamodule)                   # PyTorch Lightning
    → training_step: L_mask + L_ro                   # dreams.py:332
    → validation_step: SpecRetrievalValidation       # dreams.py:404
```

### 微调流程

```
python train.py --dataset_path labeled.hdf5          # train.py:74
                --train_regime fine-tuning
  → FineTuningHead(backbone=pretrained, lr=5e-5)     # heads.py:27
  → Trainer.fit(head, datamodule)
    → training_step: L_task (contrastive/fingerprint/...)  # heads.py:XXX
```

## DreaMS 类结构

```python
class DreaMS(pl.LightningModule):
    # === 构造 ===
    def __init__(self, args, spec_preproc):
        # Fourier 特征
        self.fourier_enc = FourierFeatures(...)       # m/z → 正弦/余弦特征
        self.ff_fourier = FeedForward(1024→512→980)   # Fourier 特征降维
        
        # 峰值嵌入
        self.ff_peak = FeedForward(2→44→44)          # [mz, int] → 44-dim
        
        # Transformer编码器
        self.transformer_encoder = TransformerEncoder(args)  
        # 7层 × 8头 MultiheadAttention + FeedForward
        
        # 输出头
        self.ff_out = FeedForward(1024→1024→C)       # 预训练用
        self.ro_out = Linear(2048→1)                  # 保留顺序用
    
    # === 前向 ===
    def forward(self, spec, charge=None):
        # 1. Padding mask
        # 2. Peak embedding
        # 3. Fourier encoding
        # 4. Concatenate
        # 5. Transformer (with Graphormer bias)
        # 6. Return embeddings
    
    # === 训练 ===
    def spec_ssl_step(self, spec_mask, spec_real, mask, charge):
        # 掩码峰预测 + 损失
    
    def step(self, data, batch_idx, log_prefix):
        # 训练/验证步骤
    
    # === 优化器 ===
    def configure_optimizers(self):
        # Adam + NoamScheduler
```

## 核心层实现

### MultiheadAttention

```python
class MultiheadAttention(nn.Module):
    def __init__(self, args):
        self.weights = Parameter(4*d_model, d_model)  # Q,K,V,O 合并
        self.lin_graphormer = Linear(d_graphormer_params, n_heads)  # φ
    
    def forward(self, q, k, v, mask, graphormer_dists, 
                chem_bias=None, gate_weights=None):
        # 1. QKV 投影
        # 2. Split heads
        # 3. dot-product attention
        # 4. + Graphormer bias
        # 5. + chem_bias (模块一新增)
        # 6. × gate_weights (模块一新增)
        # 7. Softmax + Dropout
        # 8. Weighted sum + output projection
```

### TransformerEncoder

```python
class TransformerEncoder(nn.Module):
    def __init__(self, args):
        self.atts = [MultiheadAttention × 7]    # 注意力子层
        self.ffs  = [FeedForward × 7]            # FFN 子层
        self.scales = [ScaleNorm × 15]           # Pre-Norm × 2 + final Norm
    
    def _layer_forward(self, i, x, mask, graphormer, chem_bias, gate_w):
        # Pre-Norm → Attention → Residual → Pre-Norm → FFN → Residual
```

## 与 chem_aware 模块的集成点

```python
# chem_aware_dreams.py ChemAwareDreaMS(DreaMS)
def forward(self, spec, charge=None, lambda_override=None):
    # ... 原始 DreaMS 流程 ...
    
    # [模块一插入点] 计算化学偏置
    mz_diffs = compute_peak_pair_mz_diffs(raw_mz)
    chem_bias = self.chem_rule_engine(mz_diffs, ...)
    
    # [模块一插入点] LambdaController 生成 λ
    lambda_frag, lambda_layer, lambda_conf = self.lambda_controller(...)
    chem_bias = chem_bias * lambda_effective
    
    # [模块一插入点] Gate weights
    gate_weights = self.gate_network(spec_emb)
    
    # 传入 Transformer
    spec = self.transformer_encoder(spec_emb, ..., chem_bias, gate_weights)
```

集成完全通过 `MultiheadAttention.forward()` 的两个可选参数实现——不需要修改 Transformer 架构本身。
