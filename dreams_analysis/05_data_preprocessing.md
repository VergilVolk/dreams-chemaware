# 数据预处理流水线

## 完整流程

```
.mgf / .mzML / .msp
    │
    ▼
MSData.load()  ← 统一的数据加载接口
    │
    ▼
HDF5 存储  ← ML-friendly 格式，支持内存映射
    │
    ▼
SpectrumPreprocessor  ← 标准化处理
    │
    ▼
PyTorch Dataset  ← 训练用
```

---

## SpectrumPreprocessor 详细步骤

```python
def __call__(spec, prec_mz, high_form):
    # 1. 选前 n 高强度的峰
    spec = trim_peak_list(spec, n_highest_peaks=60)
    
    # 2. 补齐到固定长度
    spec = pad_peak_list(spec, target_len=60)
    
    # 3. 强度归一化到 [0, 1]
    spec = to_rel_intensity(spec)
    
    # 4. 前置 precursor peak
    spec = prepend_precursor_peak(spec, prec_mz=precursor_mz, 
                                   intensity=1.1)
    # precursor 峰放在位置 0，强度为 1.1（比任何真实峰都高）
    
    return spec  # shape: (n_peaks=61, 2)
```

---

## 关键设计决策

### 1. 输入格式 (2, n) vs (n, 2)

DreaMS 内部统一使用 wide format (2, n_peaks)——第 0 行 m/z，第 1 行强度。

### 2. Precursor Peak

```
真实峰: (mz=353.21, int=0.87)
真实峰: (mz=208.15, int=0.45)
  ...
precursor: (mz=parent_ion, int=1.1)  ← 人工插入，位置 0
```

precursor 峰作为"主节点"聚合全局信息。在所有层的自注意力中不受限制（bias=0）。

### 3. Normalization

m/z 在 forward 中归一化（除以 max_mz），强度在预处理时归一化（除以 base peak）。

```python
# dreams.py DreaMS.__normalize_spec()
spec = spec / tensor([max_mz, 1.0])
# m/z 归一化到 [0, 1]，强度保持不变
```

### 4. 数据增强

训练时可选用：
- m/z 偏移增强（模拟仪器误差）
- 随机丢弃低强度峰（增强鲁棒性）

---

## DataFormat 类层次

```python
DataFormatA:  # LC-MS 高质量
    max_mz: 1000.0
    max_peaks_n: 128
    max_tbxic_stdev: 0.0001  # 极严格的 TBXIC 质量过滤
    min_charge: 1
    max_charge: 1
    min_intensity_ampl: 1000

# 其他格式（B, C, ...）用于不同质量级别的数据
```

DataFormat 定义了数据质量过滤器——在构建 GeMS 数据集时筛除低质量谱图。

---

## HDF5 存储格式

DreaMS 使用自定义的 HDF5 格式存储谱图数据：

```
.hdf5 文件结构:
├── spectrum/       (N, n_peaks, 2) — 所有谱图（压缩存储）
├── precursor_mz/   (N,) — 母离子 m/z
├── charge/         (N,) — 电荷数
├── adduct/         (N,) — 加合物类型
├── smiles/         (N,) — SMILES（如有标注）
├── fold/           (N,) — 训练/验证/测试划分
└── ...
```

使用 h5py 的 chunked storage + 内存映射，支持超过内存大小的大数据集随机访问。

---

## 训练时的批处理

```python
# train.py
dataset = MaskedSpectraDataset(
    in_pth=hdf5_path,
    spec_preproc=spec_preproc,
    frac_masks=0.3,           # 遮住 30% 的峰
    deterministic_mask=False,  # 随机遮罩
    ret_order_pairs=True,     # 是否生成保留顺序对
)
```

MaskedSpectraDataset 在 `__getitem__` 中动态生成遮罩和保留顺序对，不需要提前计算。
