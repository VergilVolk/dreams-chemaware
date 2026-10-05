# BioAware B4 直接共享 embedding：本地预检节点（2026-09-05）

## 已通过的数据门

- 冻结查询：548；真值身份：164；真值分子式：136。
- 候选分子对：2,003；候选参考谱行：13,645；官方错误：163。
- 化学过滤前 595 条查询张量中仅使用正式 548 条，47 条被拒绝张量不进入 manifest。
- v4 化学完整性候选特征与正式 benchmark 的 `(query_id, candidate_id)` 集合为
  `2003/2003` 完全一致；旧 v3 表多出的 124 个候选对已被识别并禁止使用。
- 冻结历史兼容门：`full_bioaware = 17/0`；`full_no_edge_gate = 24/2`。
- 所有候选参考行均来自结构—质量一致的 approved `[M-H]-` 清单；候选公式取
  RDKit 结构计算公式，不再使用存在记录级冲突的 HDF5 原始 FORMULA 字段。

## 外层分子式隔离后的训练路由覆盖

每个 outer fold 内，BioAware 路由器只在其余公式上再次 inner-formula cross-fit；
当前 outer fold 的公式既不参与路由器拟合，也不产生训练路由。

| outer fold | held queries | SAFE corrective/harm | RECALL corrective/harm |
|---:|---:|---:|---:|
| 0 | 109 | 17 / 0 | 25 / 1 |
| 1 | 108 | 15 / 0 | 23 / 2 |
| 2 | 105 | 15 / 1 | 22 / 2 |
| 3 | 85 | 15 / 0 | 19 / 1 |
| 4 | 141 | 19 / 2 | 26 / 3 |

这里的 `harm` 是 nested router 在训练部分造成的已知错误，进入高权重 margin
保护臂，不进入纠错臂。该表只证明路由和覆盖可运行，不是 embedding 性能。

## 预检中抓到并修复的问题

1. 单元张量缓存是 595-query 上游全集，不能要求与 548-query benchmark 完全相等；
   已改为 frozen benchmark 必须是其完整子集，并显式报告 47 条未使用张量。
2. HDF5 的同一 IK14 参考记录可能出现 FORMULA 字段冲突；已改用化学完整性审计中
   的 `calculated_formula`，并要求每个进入候选图的参考行均为 approved `[M-H]-`。
3. 已有 8-unit OOF 转换不能直接充当 formula-OOF 的训练路由；已改为每个 outer
   formula fold 内重新 inner-formula cross-fit，堵住路由器间接见过 held formula 的风险。
4. 2026-09-05 首次服务器 smoke 在零步复现门失败（285/548 个 rank 不一致，最大
   候选分数误差 0.9278）。根因不是 checkpoint 或训练，而是外部负离子 benchmark 的
   `library_row` 指向 MoNA-negative MGF，我却把该整数误作 MassSpecGym HDF5 行号。
   已删除这条跨数据库行号别名：manifest 现在从原 MoNA MGF 流式提取 2,997 张实际
   候选参考谱，并同时冻结原 2,997 个参考 embedding 和 548 个查询 embedding。
   GPU smoke 必须先分别通过 query/reference embedding 重放，再以 0 rank mismatch
   和候选分数误差不超过 `5e-4` 通过总复现门；任何一门失败均禁止训练。

## 尚待 GPU 作业回答

本地无法运行官方 116M DreaMS 的真实前向与反向；服务器作业将先执行：

1. manifest 语义验证；
2. 官方 checkpoint 对 548 个查询、2,997 张 MoNA 参考谱、全部 2,003 个候选分数
   和 548 个 strict rank 的 FP32 复现；
3. 一个完整 smoke optimizer step 和共享查询/参考重算；
4. 两策略各五个 formula-OOF 模型；
5. pooled formula-cluster CI、corrected/introduced、MRR 和 preservation。

在上述结果完成前，不声明共享 embedding 已提高，也不承诺复现后处理的 3–4 pp。
