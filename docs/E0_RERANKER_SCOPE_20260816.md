# E0 Re-ranker 脚本范围与结论边界（防止误读）

## 脚本
`tasks/eval_e0_reranker.py`

## 它实际吃了什么数据（全部是你自己的，无 DeepMet）
- `data/validation/e0_baseline/e0_pair_arrays.npz` → DreaMS 余弦 `cos` + `(pair_i, pair_j, label, query_id)`
- `data/validation/e0_baseline/e0_manifest.json` → `precursor_mz`、`ce`、`instrument`、`ik14`

## 它做了什么
逻辑回归把 5 个特征（cos、ppm 质量误差、碰撞能差、碰撞能缺失标记、仪器是否匹配）揉成一个排序分数；query-clustered 5-fold CV；分子级（IK14）聚合后重排序。

## 它没用什么
- 没有 DeepMet 的任何代码 / 数据 / 权重（DeepMet 在此贡献为零）
- 没有峰级数据（共享主峰、条件特异峰都没用）

## 结论边界（它最多只能证明这么多）
- 你自己的粗元数据（仪器 / 碰撞能 / 质量误差）里，存在一点点与 DreaMS 余弦**不正交**的信号。
- 这点信号能微弱纠错：R@1 +0.46pp，434 例修正 / 336 例新增错误（净 +98）。
- 这个结果 **不** 证明 DeepMet 有效；**不** 证明融合路线能带来想要的提升；**不** 证明这些元数据是纠错的正确来源。

## 一句话
这是"用你自己的元数据测一下『融合思想』"的骨架实验，结论是"方向对但很弱、且带新增错误"，与 DeepMet 无关，也不构成性能提升。
