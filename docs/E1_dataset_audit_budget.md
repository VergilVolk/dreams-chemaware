# E1 数据源严格审查与低预算决策

日期：2026-08-07

## 结论

E1（同分子谱图身份对比微调）应以 MassSpecGym 的 `train` fold、`[M+H]+`
为主训练集。`annotated01.mgf` 不能原样进入 E1，也不能在训练后继续把
MassSpecGym `val/test` 当作独立评估集。

这不是因为 annotated01 小或化学覆盖不足，而是因为它在合并时丢失了
adduct、collision energy、instrument 和 source provenance，并且把 MassSpecGym
自身的数据一并并入，形成了直接的数据泄漏。

## 实测规模与质量

| 项目 | annotated01 | MassSpecGym HDF5 |
|---|---:|---:|
| 谱图数 | 3,263,214 | 231,104 |
| 唯一 IK14 | 76,157 | 28,929 |
| 每分子谱图中位数 | 33 | 3 |
| 分子式覆盖 | 90.5% | 100% |
| adduct 字段 | 0% | 100% |
| collision energy 字段 | 0% | 有字段，52.7% 有值 |
| instrument 字段 | 0% | 97.7% 有值 |
| source provenance | 0% | 当前 HDF5 未显式保存，但 fold 明确 |
| 独立 train/val molecule split | 无 | 有，train/val IK14 重叠为 0 |
| 具有 >=3 个 10% 基峰强度峰的质量代理 | 74.1% | train 63.9% |

质量代理只能说明谱峰是否足够，不代表标签可靠。annotated01 的 74.1% 较高，
部分原因是生成脚本已经先删除少于 3 个原始峰的谱图；它不能补偿 adduct 和来源
字段缺失。

MassSpecGym train 中共有 156,568 张 `[M+H]+` 谱图、22,525 个进入当前候选池的
分子，以及 145,512 个可构造 triplet 的 anchor。进一步施加严格谱峰质量代理后，
仍有 101,835 张谱图、19,723 个分子；其中 13,383 个分子有至少两张合格谱图，
对应 95,495 张谱图。这一规模足以完成 E1，不需要为“数据量”引入 annotated01。

## 泄漏审查

| 重叠 | IK14 数 |
|---|---:|
| annotated01 与 MassSpecGym HDF5 train | 22,348 |
| annotated01 与 MassSpecGym HDF5 val | 5,481 |
| annotated01 与 MassSpecGym metadata test | 4,600 |
| annotated01 与任一 MassSpecGym 集合的并集 | 27,829（占 annotated01 分子的 36.5%） |

annotated01 覆盖了约 96% 的 MassSpecGym 验证/测试分子。因此只按随机谱图拆分、
或先在 annotated01 上训练再评估 MassSpecGym，都会产生严重乐观偏差。

## annotated01 的工程风险

`build_annotated01.py` 会递归读取 `data` 下的所有 MGF/MSP，却不保存输入文件清单。
它只输出 SMILES、InChIKey、可选 formula、precursor mass、粗粒度 ion mode 和峰表。
去重键是 `(InChIKey, 前 20 个最低 m/z 峰按 0.1 Da 取整后的哈希)`，不使用完整峰表
与强度。脚本声称读取 MSP，但主体解析器主要按 `KEY=VALUE` 读取，而标准 MSP 常用
`KEY: VALUE`。这些设计使数据来源、加合物一致性、真正重复谱和解析完整性都无法
从最终 MGF 中可靠恢复。

同一 IK14 中，29,911 个分子的 precursor mass 跨度超过 1 Da，23,713 个分子同时
出现正、负离子谱。这些现象本身不一定是化学错误，但在 adduct 丢失后，不能把它们
直接当作“同条件正样本”。

## 低预算训练闸门

当前三 seed、每 seed 最多 10 epoch 的正式脚本，每 epoch 为 9,095 个 micro-batch
（batch size 16，每个 triplet 前向 3 张谱图）。完整计划上限约为：

- 272,850 个 micro-batch；
- 136,440 次 optimizer update（gradient accumulation 2）；
- 13,096,080 张谱图前向。

预算紧张时不应直接提交这个作业。建议按以下闸门执行：

1. **R0-OfficialFT**：只加载官方 `embedding_model.ckpt` 做一次统一 E0 评估，不训练。
2. **E1-Pilot-A**：raw SSL 起点，1 个 seed，每 epoch 1,000--2,000 个 train batch，
   最多 3 epoch；用于判断 identity triplet 是否方向正确。
3. **E1-Pilot-B**：官方 fine-tuned 起点，使用更小学习率，执行同预算 continuation；
   与 A 使用完全相同的样本和验证协议。
4. 只有 pooled 10-ppm ROC-AUC、query-macro AUC 和 separation 至少两项改善，才扩大
   到 1 个完整 seed；只有胜出配置才跑 3 seeds。

第一轮 pilot 采用 2,000 batch x 3 epoch 时约处理 288,000 张训练谱图前向，约为
原三 seed 完整上限的 2.2%。

## annotated01 的正确用途

annotated01 有约 48,328 个不属于 MassSpecGym 的额外分子，具有后续扩展价值，
但必须从原始来源重建 `annotated01-clean`：

1. 保存 source file、library、adduct、polarity、collision energy、instrument；
2. 强制排除 MassSpecGym val/test 的 IK14；增量价值实验最好连 MassSpecGym train
   的重复谱也移除；
3. 正样本必须是同 IK14、同 adduct、同 polarity、不同峰哈希；
4. 用完整、容差感知的峰表和强度哈希去重；
5. 按分子限制最大谱图数，避免每分子最多 1,003 张谱图的高重复分子主导梯度；
6. 建立 source-group / molecule-group split，并保存可复现 manifest。

在完成重建前，annotated01 只适合探索规则覆盖和构建候选化学概念，不适合作为
E1 的主训练集或独立基准。

## 审查产物

机器可读结果：`data/e1/dataset_audit.json`。
复查脚本：`tasks/audit_e1_data_sources.py`。
