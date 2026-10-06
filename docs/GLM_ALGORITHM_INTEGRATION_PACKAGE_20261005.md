# 算法整合打包（任务一交付骨架 · 整合不封存）

**日期：** 2026-10-05
**性质：** 算法部分的梳理与整合文档。定位：把多模块系统、它们的证据资格、统一外部天梯和已知边界装订成一个可交给评审的完整包。**整合 ≠ 封存**——末节列出已排队的前向升级路径。
**状态：** 天梯数字槽位待 12 方法终版落地后填充（监督链自动完成）。

---

## 1. 一句话定位

> 一个以外部验证为纪律的谱图信息提取系统：稳健谱图表示（Noise V1）+ 候选特异谱学证据层（P2b 通道族）+ 冻结重排层，在模型盲、身份/公式双隔离的 GNPS 基准上以 12 方法统一天梯接受检验。

## 2. 四层架构

```
┌─ 评估层 ─────────────────────────────────────────────┐
│ GNPS Gold/Silver 模型盲双面板（10,995 + 5,261）       │
│ 12 方法统一天梯 · 双源核验（报告 vs 逐query重算）      │
├─ 决策层 ─────────────────────────────────────────────┤
│ 冻结重排器：P2b(official) / P2b(Noise V1)             │
├─ 证据层 ─────────────────────────────────────────────┤
│ P2b 通道族：sqrt-cosine / 熵 / 中性丢失               │
│ 经典基线：WSE / cosine / modified cosine              │
│ 公开神经基线：MS2DeepScore 2.x / SpecVec              │
├─ 表示层 ─────────────────────────────────────────────┤
│ official DreaMS → Noise V1 课程微调                  │
│ （真值盲关系课程 · 原生运行时 · 外部 CI>0）            │
└──────────────────────────────────────────────────────┘
```

## 3. 模块证据资格表

| 模块 | 角色 | 证据资格 | 已知边界 |
|---|---|---|---|
| Noise V1 encoder | 表示层底座 | **外部 E2**：GNPS 双面板 +1.20/+1.25pp，CI 下界>0 | Top-1 与 WSE 统计上难分（点估计 −0.81）；判别类指标第一 |
| P2b 通道族 | 候选特异证据 | 外部方向成立（official 基底 +1.62pp CI>0） | near-core −4.23pp（共享碎片重复计分） |
| P2b 冻结重排器 | 决策层 | 封存 P3-main +1.07pp（E3） | 无新独立确认面板 |
| ChemAware V2 | 候选特异化学重排 | 内部 12 门全过（role-2 +3.24 / role-3 内部 +3.99，胜六对照） | World B：encoder 线增益不外迁；V2 作重排器待外测 |
| GNPS 双面板基准 | 评估基础设施 | 封存、模型盲、双隔离 | 与 DreaMS 预训练共享数据生态；[M+H]+ 单加合物 |
| RRF | 已关闭负对照 | 外部确认失败（run 2347032） | 只进消融表 |
| BioAware 静态网络 | 冻结负链 | B44−0.56 / B45+0.23 n.s. / B46−0.77 | B47 外部门未开 |

## 4. 统一天梯（引用 + 读数槽位）

完整表见 `GLM_GNPS_ARTICLE_LADDER_20261005.md`（12 方法 × 双面板 × 全指标，双源核验）。三个 headline 读数：

- **读数 1（外推性）**：Noise V1 86.56% vs official DreaMS 85.36%（+1.20pp，双面板 CI 下界>0）；对公开神经基线：**vs MS2DeepScore 2.x +7.44/+6.60pp（双面板 CI 全正，显著）**；vs Spec2Vec +0.26/+0.63pp（点估占优，CI 跨零，统计打平；判别类指标差距更大：pooled AUROC 0.9413 vs 0.9336）；
- **读数 2（指标劈裂）**：Top-1 第一 = WSE（87.37，零训练）；pooled pairwise AUROC 第一 = Noise V1（0.9413 vs WSE 0.9402 vs official 0.9287）——判别最强的表示与 Top-1 最强的分数不是同一家；
- **读数 3（神经基线对照）**：SpecVec 86.29（第 8，胜 official 但低于 Noise V1）；**MS2DeepScore 2.x 79.12（第 12）**——其公开 dual-mode 模型以化学相似度（Tanimoto 类）为训练目标，会把结构类似物排在真结构之前：**相似度目标 ≠ 鉴定目标**，与读数 2 互为印证，且打分经官方 `.pair()` 审计（worst 6.09e-08），是模型真实行为而非实现偏差。

## 5. 必须随行的边界知识

1. **World B**：同分布微调增益不外迁（12 个 R@1 CI 全跨零）——所有内部数字须挂同语料星号；
2. **指标劈裂**：DreaMS 论文 0.85 为 pairwise 口径；cosine 在 Top-1 上几乎追平 official；
3. **near-core 墓地**：所有全局重排在极近异构体上回退；
4. **公开基线披露**：MS2DeepScore/SpecVec 训练于 GNPS 生态，零重叠保证仅覆盖 MSG/MoNA。

## 5.5 方法互补性（12 方法 oracle 分析，本地已验证计算）

- **最佳二元组**：noise V1 + WSE → identity **89.98%**（比最佳单方法 +2.61pp）；formula 面板最佳对是 **cosine + noise V1 → 91.52%**（+3.25pp）——互补性来自**分歧**而非强度（谦逊的 cosine 在 formula 面板是最有价值的搭档）；
- **三元组** WSE+Spec2Vec+noise V1：90.98 / 92.45pp；
- **全体 12 方法 oracle**：92.90 / 94.28pp（+5.53 / +6.01pp）；
- **最难子集上互补性最大**：near 子集 family oracle vs 最佳单方法 = identity **+4.40pp**（86.91 vs 82.51）、formula **+6.18pp**（86.28 vs 80.10）——**方法们在不同的分子上失败**，这是候选层路由的立项实证（此前 grand fusion 文档引用的 +3.22/+3.67pp 三专家 oracle 现在有 12 方法版验证数据支撑）；
- 工件：`method_complementarity.json` + `method_complementarity_pairs.csv`（66 对全组合）。

## 6. 整合不封存：已排队的前向路径

| 层 | 状态 | 入口工件 |
|---|---|---|
| 参数层 | task-vector 七臂已产出（诊断：近正交 0.037），**内部保护门评价待跑** → 选一臂 → 一次性 GNPS | run 2349962 |
| 候选层 | grand router 已排队（job 2349964 日志未回收）；V2 分数已导出 | run 2349963 |
| 应用层 | 三层主张体制（实体/类别/结构）+ 反向代谢组学杠杆，LCNEC 暗特征清单待建 | docs/URSEA 系列（降级为统计工具） |

## 7. 工件索引

- 天梯：`docs/GLM_GNPS_ARTICLE_LADDER_20261005.md` + `deliverables/GLM_gnps_article_ladder/` + 图（svg/pdf/png）
- 重建认证：`tasks/GLM_reconstruct_gnps_identity_panel.py` → `data/validation/GLM_gnps_identity_panel_reconstruction/`
- 公开模型打分：`tasks/GLM_score_public_models_on_gnps.py`（嵌入语义经官方 .pair() 审计）
- 监督链：`tasks/GLM_public_models_chain.ps1`（日志 `run_local/chain.log`）
- git：`cdb153a`（管线脚本）→ 本次链完成后自动追加提交
