# ChemAware MoNA 双极性迁移评测协议（2026-09-28）

## 1. 原有 +2.1266 pp 到底在哪个数据上

该结果不是 MoNA 外部结果。它来自本地 `MassSpecGym_MurckoHist_split.hdf5`
构成的 ChemAware corrected candidate graph 中的 **formula role 2**：

- query 数：1,975；
- official DreaMS Recall@1：0.8962025；
- ChemAware Phase-A step-2000 Recall@1：0.9174684；
- 绝对增量：+0.0212658，即 **+2.1266 percentage points**；
- top-1 corrected / introduced：54 / 12；
- formula-cluster bootstrap 95% CI：+1.2761 至 +3.0303 pp；
- 证据资格：role-2 development result，不是 role-3/outer confirmation。

因此它能证明“该微调方法在冻结的内部开发角色上有效”，但不能单独证明跨库迁移。

## 2. 新 MoNA 面板的目的

新评测只回答一个问题：冻结的 ChemAware shared embedding 相对 official DreaMS，
能否把在 MassSpecGym 上得到的提升迁移到另一个谱库的同分子式候选检索。

它不是 reranker，不在推理时使用分子式、结构或规则；这些字段只用于离线构造
候选集合和真值。query 与所有 reference 均由同一个冻结 encoder 独立编码。

## 3. 为什么不能直接把约 8 万条 MoNA 谱称为 8 万条测试样本

有效检索 query 必须同时满足：

1. 有合法结构、InChIKey 和 precursor m/z；
2. 身份不出现在本地 MassSpecGym 微调/开发 HDF5；
3. 同一身份至少有两条不同的归一化谱，一条作 query、另一条作 positive reference；
4. 同 polarity、同分子式、20 ppm precursor 窗内至少存在一个其他身份作为 negative；
5. 完全相同或仅强度等比例缩放的重复谱先按 DreaMS 输入 token 哈希去重；
6. positive 和 negative polarity 永不混入同一 query 的候选集。

在每个身份最多取 4 条 query 后，本地严格审计规模为：

| polarity | raw MGF records | queries | query identities | formulas |
|---|---:|---:|---:|---:|
| positive | 42,622 | 921 | 290 | 122 |
| negative | 36,663 | 1,770 | 569 | 225 |
| combined | 79,285 | **2,691** | 580（去重合计） | 347 polarity-formula clusters |

2,691 比原 role-2 的 1,975 queries 多 716 条（约 +36.3%）。面板构造不读取任何
embedding 或模型分数，避免按模型表现选择样本。

## 4. 评测指标与配对统计

候选 molecule 的分数定义为 query 与该 molecule 所有 reference spectra 的最大余弦相似度。
并列分数按不利于 positive 计数。报告：

- Recall@1/3/5/10/20/50；
- MRR、mean positive margin；
- micro AUC、macro AUC；
- 各 top-k corrected / introduced；
- paired delta；
- 以 `polarity:molecular_formula` 为 cluster 的 Recall@1 bootstrap 95% CI；
- positive、negative 两个 polarity 的分层结果；
- 微调 embedding 与 official embedding 的平均 cosine preservation。

MoNA 结果不得用于选择 checkpoint 或调参；本次只比较已冻结 official 与已冻结
Phase-A checkpoint。完整 query-level rank 和 margin 会保存到
`evaluation_paired_outcomes.npz`。

## 5. 运行入口

服务器上直接提交：

```bash
sbatch tasks/run_chemaware_mona_polarity_transfer.sbatch
```

脚本固定为一个 GPU，不指定内存。默认验证并使用：

```text
data/validation/chemaware_online_role_budget_native/run_2345481/
phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt
```

默认 checkpoint SHA256 为
`a8428329ca1d12bfe735f3f8b848ed020a8db18d0a654d4f62cdbeb006e61135`。
如服务器路径不同，必须同时显式给出 checkpoint 和对应哈希：

```bash
sbatch --export=ALL,CHEMAWARE_CHECKPOINT=/absolute/path/model.ckpt,CHEMAWARE_CHECKPOINT_SHA256=<sha256> \
  tasks/run_chemaware_mona_polarity_transfer.sbatch
```

输出目录：

```text
data/validation/chemaware_mona_polarity_transfer/run_<SLURM_JOB_ID>/
```

核心文件为 `panel/report.json`、`evaluation.json` 和
`evaluation_paired_outcomes.npz`。

## 6. 声明边界

该面板相对本地 MassSpecGym 微调/开发身份严格不重叠，因此可称为
“identity-disjoint cross-library transfer”。但是 MoNA 可能参与过 DreaMS 预训练，
所以不能称为“对 DreaMS 预训练语料完全未见”的外部测试。若要主张预训练级独立性，
仍应使用已经做过 MassSpecGym 与 MoNA 双重去重的 MSnLib benchmark；其严格
mass-competitive query 数较少，适合作为更独立但更小的互补验证。
