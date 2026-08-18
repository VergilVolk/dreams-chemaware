# ChemMask 峰级增强修正 #1+#2 —— 实施记录

> 承接 [[PEAK_AUGMENTATION_STRATEGY_AUDIT_20260816]] 的 #1（补噪声）+ #2（修重归一化）。
> 本文记录"改了什么、为什么、怎么验证的"。**不承诺必然提高**，最终靠 500 例困难样本 / 100 硬三元组评估说话。

## 一句话
把 ChemMask 训练数据增强从"先 mask（仅 intensity→0，m/z 不动）→ 再重归一化"，
改成与 DreaMS 预训练一致：**先归一化 → 再 mask（m/z 和 intensity 都置 -1）**，并新增预训练的
**强度正比 m/z 掩码**作为"加噪声"增强。

## 改动的文件
- `tasks/train_causal_chemmask_head.py`（核心）
- `tasks/train_causal_chemmask_full_cpu.py`（同步 CLI 参数）
- `tasks/smoke_causal_chemmask_augment.py`（新增单元冒烟）

## 改了什么

### #2a：mask token 从 `0.0`（仅 intensity）改成 `-1.0`（m/z + intensity 都置 -1）
预训练 `MaskedSpectraDataset` 用 `mask_val=-1`，对 `mask_mz` 和 `mask_intensity` 两个维度都置 -1
（`dreams/utils/data.py:1178-1184`）。之前 chemask 只把 intensity 置 0、m/z 留着，模型看到的是
"m/z 存在但强度为 0"的模糊信号，不是预训练学过的"被掩峰" token。

### #2b：先归一化、后 mask（消除重归一化污染）
预训练顺序是 `spec_preproc`（trim→pad→`to_rel_intensity` 除以原 max→prepend precursor）**之后**再 mask。
之前 chemask 是 `mask_unique_peaks` 在原始谱上置 0，**然后** `preprocess_spectrum` 才取 max 归一化——
删掉一个高强度峰会拉低新 max，导致其余所有峰被放大（"删一个峰"变成"删一个峰 + 放大其余峰"）。
现在改为：先 `preprocess_spectrum`（在干净谱上归一化），再对张量上的特定行做 mask。

### #1：新增 `mask_noise` 强度正比掩码
复现预训练 `MaskedSpectraDataset`（`intens_p`、`frac_masks=0.3`、`mask_val=-1`）：
- 可掩峰 = intensity ∈ (0, 1)（排除 precursor 行 0、base peak 1.0、padding 0）；
- 采样数 `n_masks = max(2, round(n * fraction))`，按强度比例采样；
- 命中行的 m/z 和 intensity 都置 -1；
- 作用于 anchor 和 positive 双侧（negative 不打噪声）。

### 新 CLI 参数
- `--noise-mask-prob`（默认 0.5）：噪声增强采样概率。
- `--noise-mask-fraction`（默认 0.3）：单条谱掩多少比例的可掩峰（对齐预训练 30%）。

数据集构造新增两个关键字参数 `noise_mask_prob=0.0`、`noise_mask_fraction=0.3`（**默认关**），
因此 `eval_causal_cpu_paired.py` / `train_chemaware_multitask_head.py` / `smoke_causal_chemmask_data.py`
等既有调用方不受影响、噪声默认关闭。

## 验证结果（全部通过）
1. `py_compile` 两个脚本 → OK。
2. `smoke_causal_chemmask_augment.py`（合成张量）→ ALL PASS：
   - `mask_noise`：precursor/base peak/padding 保护，命中行 m/z+intensity 均 -1；
   - `mask_unique_peaks`：只掩"无 m/z 匹配"的高强度峰，匹配峰位级不变（无重归一化）。
3. `smoke_causal_chemmask_data.py`（真实 hdf5 + pool，`identity_mask_prob=1.0`）→ protocol_errors=0，
   masked 计数 min=0/median=1/max=12 合理。
4. 端到端噪声（`noise_mask_prob=1.0`，16 样本）→ anchor 13/16、positive 13/16 有 -1 token，
   negative 0/16（噪声只打双侧，符合预期；个别样本可掩峰不足而跳过）。

## 结论边界
- 以上只证明"改动语义正确、不破坏数据管线"。是否扩大那 1–2pp **未验证**。
- 下一步：head-only smoke 训练（噪声开）→ 500 例困难样本 / 100 硬三元组评估，与
  `best_counterfactual.pt` 基线对比。数字决定结论，不预判。
