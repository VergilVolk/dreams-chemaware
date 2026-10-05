# GLM：GNPS 绝对基线尺与全项目增益总账

日期：2026-09-30（GNPS 基线 run 2347472 完成后）
性质：结果账本。全部数字直接抄自 run 2347472 的两份 evaluator 报告
（`data/validation/gnps_absolute_baselines_run_2347472/gnps/paired_official_vs_stage1/report.json`
与 `.../paired_official_vs_v1/report.json`），配对 CI 为公式簇 bootstrap
（10,000 次重采样，seed 20261001，每面板族 24 假设）。

---

## 1. 一句话答案

**当前最强模型（V1，T1/T3 关系课程冠军）相对官方原始 DreaMS：外部 GNPS +1.20 / +1.26pp（双面板统计显著），内部 held fold-0 +1.05pp。距 +5pp 目标还差约 4pp。**

## 2. GNPS 绝对数字（首次产出，模型盲外部基准）

基准：GNPS Gold/Silver 10ppm（自建，9/28 封存），52,871 谱，[M+H]+，查询/参考按 identity 或 formula 不相交。

| checkpoint | identity-disjoint R@1（10,995 题） | formula-disjoint R@1（5,261 题） | identity near 子集（7,592） | formula near 子集（2,944） |
|---|---:|---:|---:|---:|
| official | 85.357% | 86.809% | 79.939%（错 20.06pp） | 77.989%（错 22.01pp） |
| Stage-1 | 86.066% | 87.797% | 80.822% | 79.416% |
| **V1** | **86.558%** | **88.063%** | **81.441%** | **79.959%** |

- "卡在 20 几个 pp"跨数据集复现：official 的 near 子集错误率 = 20.1 / 22.0pp。
- V1 自己的 near 错误率仍 18.6 / 20.0pp——最难段没有被现有课程实质解决。
- 全库 pooled pairwise AUROC：identity 0.9287→0.9413，formula 0.9033→0.9299（V1）。

## 3. 配对增益（可声明口径，vs official 基线）

| 对比 | 面板 | ΔR@1 | 95% CI（pp） | corrected/introduced | risk_net(λ=2) |
|---|---|---:|---|---|---:|
| Stage-1 vs official | identity | +0.709 | [−0.005, +1.399]（不显著） | 273/195 | −117 |
| Stage-1 vs official | formula | +0.988 | [+0.134, +1.886]（显著） | 130/78 | −26 |
| **V1 vs official** | **identity** | **+1.201** | **[+0.460, +1.890]（显著）** | 328/196 | −64 |
| **V1 vs official** | **formula** | **+1.255** | **[+0.229, +2.299]（显著）** | 156/90 | −24 |
| V1 vs official（near 子集） | identity | +1.502 | [+0.589, +2.464] | near 295/181 | −67 |
| V1 vs official（near 子集） | formula | +1.970 | [+0.251, +3.674] | near 138/80 | −22 |

分步归因（由绝对值相减，点估计）：

- official → Stage-1（噪声课程）：identity +0.709pp，formula +0.988pp；
- Stage-1 → V1（T1/T3 关系课程）：identity 再 +0.491pp，formula 再 +0.267pp。

## 4. 内部口径（MassSpecGym corrected held fold-0，18,333 题）

| 模型 | R@1 | vs official |
|---|---:|---:|
| official | 93.1875%（1,249 错） | — |
| Stage-1 | 93.6835%（1,158 错） | +0.496pp |
| **V1** | **94.2399%（1,056 错）** | **+1.052pp**（两步 CI 分别为 [+0.88,+1.65] 因果、[+0.11,+1.05] held） |

关键交叉验证：**held fold-0 上的 +0.56pp（V1 vs Stage-1）在 GNPS 外部方向一致（+0.49/+0.27pp）**——V1 的增益是真实泛化，不是折内过拟合。

## 5. 与 +5pp 目标的差距算术

- 内部：+5pp = 98.19% = 约 917 个净纠正（官方错误 1,249 的 74.6%）；V1 现有净纠正 193 个（1,249→1,056）。
- 外部：GNPS 总错误率仍有 13.4/11.9pp（V1 口径），near 段 18.6/20.0pp。
- 交换比：V1 在 GNPS 的 C/I = 1.67（identity）/ 1.73（formula），全部 risk_net(λ=2) 为负——修正/引入比仍不过 2 的保险线，这是 V5 保护设计的直接依据。

## 6. 结论与边界

1. **可声明**：共享 encoder 经噪声+关系两段课程后，在模型盲外部 GNPS 基准上获得 +1.20/+1.26pp（显著），near 段 +1.50/+1.97pp（显著）；两段课程均方向正确。
2. **不可声明**：GNPS 是迁移基准（NIST20 式 10ppm 配对数学，非论文复现；仅 [M+H]+）；不是 MassSpecGym leaderboard 协议；内部 93-94% 是公式候选图内排名任务，两者数值不可互换。
3. **战略含义**：累计 ~1.2pp 确认了天花板审计的判断——现有信息池（不变性监督）按迭代堆不到 +5pp。后续小步改动（V5 minimal：atlas 选中 1,132 事件、零 clean 流、matched control）以本尺子为晋级门（G1 dev 不降 + G2/G3 GNPS 双面板非负），本 run 的 embeddings 与报告即 `BASELINE_RUN=data/validation/gnps_absolute_baselines_run_2347472`。

## 7. 溯源

- 作业：`run_gnps_absolute_baselines_1gpu.sbatch`（run 2347472；零训练）
- checkpoints：official `data/e1/official_embedding_slim.pt`；Stage-1 run 2344820 `targeted_final.ckpt`（slim 转换）；V1 run 2347055 `primary_seed_3407_slim.pt`
- 编码器/评估器：`tasks/encode_gnps_gold_silver_10ppm_checkpoint.py` / `tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py`
- 产物：`$RUN_ROOT/gnps/{official,stage1,v1}_embeddings.npz` + 两份 paired 报告（绝对数字在各报告 baseline/candidate 块内）
- GNPS 约法：仅评估，任何 GNPS 谱不得进入训练（否则模型盲资格作废）
