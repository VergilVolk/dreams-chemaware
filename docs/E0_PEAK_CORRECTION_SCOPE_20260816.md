# E0 峰级纠错脚本 范围与结论边界（防止误读）

## 脚本
`tasks/eval_e0_peak_correction.py`

## 它吃什么（全部是项目自身的数据/接口，无 DeepMet）
- `data/validation/e0_baseline/e0_pair_arrays.npz` → 有向三元组对：`primary__pair_i/j/labels/scores/query_ids`（1,697,022 对，positive rate 0.5446，21,163 queries）
- `data/validation/e0_baseline/e0_manifest.json` → `spectrum_id`（= hdf5 `IDENTIFIER`）
- `data/models/MassSpecGym_MurckoHist_split.hdf5` → `spectrum` (2,128) [m/z, intensity]

## 它复用了什么（关键：不是重写，是复用已验证的接口）
`train_causal_chemmask_head.py` 的 `raw_peaks` / `greedy_peak_matches` / `shared_major_score`，fragment_tolerance=0.02 Da。

## 它做什么
对每个候选对算 4 个峰特征（n_matched / matched_frac / shared_intensity / top_overlap），
先透明诊断（per-label 均值、AUC、FP-FN 签名），再 query-clustered 5-fold CV 对比
cos 单独 vs cos+峰特征的 R@1/MRR，报告 fix/regress。

## 冒烟测试已证明（dry-run，~4k pairs / 2k queries）
- self-check = 0.0（脚本重算的 shared_major 与 shared_major_score 逐位相等）
- FP 签名：高 cos 的错误候选 shared_int 0.653 vs 低 cos 0.167 → 确认「共享主峰 → FP」
- FN 签名：低 cos 的正确候选 matched_frac 0.554 vs 高 cos 0.752 → 确认「仪器/CE 特异峰 → FN」
- 峰特征 pair 级 AUC > cos（top_overlap 0.718 / shared_int 0.710 vs cos 0.636）；
  元数据 pair 级 AUC < cos（ppm 0.396 / ce_diff 0.459 / inst_match 0.530）

## 结论边界（它最多能证明这么多，不要越界）
- dry-run 的 +4.4pp **不是真值**（baseline 0.50 而非 0.90，是抽样扭曲的冒烟值）。
- 完整集 baseline R@1 = 0.902（同 query_metrics，与官方 recall@1=0.9003 一致）。
- 峰信号是「直接因果信号」，方向已在诊断里确认。
- **完整集结果（已跑）**：LR（cos+4 峰特征，连续分）R@1 = 0.9224（+2.04pp），1237 fix / 805 reg，5/5 fold 全正。
- ⚠ **重要修正**：raw `top_overlap` 单独当排序键会**输给 cos**（官方协议 -1.5pp），
  之前 B 里 `topov_only 0.9482` 是 **tie-breaking 假象**（离散分数同分多、npz pos-first
  插入顺序偏向 query IK）。详见 [[../docs/E0_PEAK_OVERVLAP_TIEBREAK_RESOLUTION_20260816]]。
- 不构成「必然提高」的承诺。脚本只负责测量，数字决定结论。

## 完整运行命令（CPU，约 3–6 分钟）
```
D:\dreams_env\python.exe tasks\eval_e0_peak_correction.py
```
