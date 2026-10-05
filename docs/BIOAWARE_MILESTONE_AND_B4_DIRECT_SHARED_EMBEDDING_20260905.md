# BioAware 里程碑与 B4 官方共享 embedding 直接微调契约（2026-09-05）

## 1. 本节点冻结的事实

### 已经成立

1. 在化学完整性过滤后的 548 条外部 Level-1 `[M-H]-` 查询上，冻结官方
   DreaMS 后接的 BioAware v2 候选专家具有稳定的开发期增量：
   - `full_bioaware`：Recall@1 `+3.47 pp`，`19 corrected / 0 introduced`，
     formula-cluster 95% CI 为正；
   - `full_no_edge_gate`：Recall@1 `+4.01 pp`，`23 / 1`，是高召回但风险更高的消融；
   - `spectral_plus_known_topology`：`+2.92 pp`，`16 / 0`。
2. 增量不是 DreaMS cosine 的单调再校准。质量覆盖、已知反应拓扑、原始 MS2
   step-0 边证据均提供了可测增量；候选联合置换的经验 `p=0.0099`。
3. 当前结果证明的是候选上下文的增量价值，不是共享谱图 embedding 已经改善。

### 尚未成立

1. 四个生物来源均已被开发流程查看；上述数值不是独立盲测或 SOTA 证明。
2. B0-M0 在严格反应特异匹配下没有同时达到覆盖与协变量平衡门，不能把
   “Rhea 相邻”直接解释为单谱 embedding 中可识别的因果方向。
3. B2/B3 的候选上下文 embedding 尚未形成显著、身份隔离的通用增益。
4. 不得把反应邻居当作同分子正样本；不同代谢物即使相邻仍需保持可区分。

## 2. 为什么仍值得做 B4

BioAware v2 找到了一组高精度、真实困难的候选竞争关系。B4 不蒸馏后处理分数，
也不要求不同反应邻居相互接近，而是仅把网络专家当作**训练样本路由器**：

- 正例始终是同一真实分子的查询谱与参考谱；
- 负例始终是同一候选图中真实的异分子候选；
- BioAware 只决定哪些官方错误应获得更高训练预算；
- 原本正确及被高召回专家误导的查询进入安全臂，约束其官方 margin 不下降；
- 查询与参考谱使用同一个可训练 DreaMS 编码器；部署只输入干净谱图。

因此，B4 测试的是：**候选上下文发现的高价值错误，是否能以训练期特权信息的
形式，迁移为候选无关的共享谱图几何。**

## 3. 两个冻结策略臂

### B4-SAFE：`full_bioaware`

只使用带 raw step-0 完整边硬门的保守专家。其已观察性能为 `19/0`（最严格
来源与分子式净化报告）或同协议消融中的 `17/0`。它优先检验高精度路由能否
产生可迁移的 embedding 梯度。

### B4-RECALL：`full_no_edge_gate`

使用高召回专家。训练时：

- 被它正确修正的官方错误进入纠错臂；
- 它新引入的错误不得进入纠错臂，而必须作为高权重安全反例；
- 不在结果出来后修改 gate、损失或剂量。

该臂检验多覆盖的错误是否能提供更大共享梯度，同时显式惩罚其已知风险。

## 4. 固定工程协议

1. 初始化：`data/e1/official_embedding_slim.pt`，架构来自
   `dreams/models/pretrained/ssl_model_server.pt`。
2. 容量：只解冻官方 projection head 和最后一个 Transformer block。
3. dropout：训练时保持 `model.eval()`，但不关闭 autograd。
4. 损失：候选组内 listwise + hardest-negative margin；安全查询使用官方 margin
   floor；查询与参考谱均使用官方 embedding preservation。
5. 推理：干净谱图单输入；同一共享 encoder；无候选特征、Rhea、疾病标签、
   BioAware 分数、P2b 或可丢弃教师。
6. 评估：5-fold truth-formula OOF；同一公式只属于一个 fold；held formula 的
   候选谱不得进入训练梯度；全部候选参考谱重新编码并按 IK14 取最大分数；并列算错。
7. 训练：固定 epoch 和优化器；held fold 不用于选 checkpoint、学习率或门控。
8. 基线：必须由同一执行器从官方 checkpoint 重新前向，并与冻结候选分数和
   Top-1 对账，否则 fail closed。

## 5. 首轮裁决门

B4 是开发期机制迁移试验，不承诺复现后处理的 `+3.47–4.01 pp`。只有满足以下
条件，才允许进入独立数据复现：

- pooled formula-OOF Recall@1 增量的 formula-cluster 95% CI 下界 `> 0`；
- corrected `> introduced`，且 `corrected - 2*introduced > 0`；
- MRR 不下降；
- 每个 fold 的 embedding preservation `>= 0.995`；
- 两个臂均完整报告，不按结果删除失败臂；
- 必须另跑等规模标签路由对照，才能把增益归因给 BioAware 路由本身。

即使通过，上述结果仍是打开队列上的 formula-OOF 开发证据，不能称为独立验证或
SOTA。最终模型必须在未参与本节点任何选择的新队列上一次性冻结验证。

## 6. 冻结证据与工件

- 算法证据：`docs/BIOAWARE_NEGATIVE_NETWORK_EXPERT_V2_CHEMICAL_INTEGRITY_20260901.md`
- 主专家工件：
  `data/validation/bioaware_metdna3_negative_network_expert_v2_chemically_filtered/artifact.json`
- 主专家工件 SHA256：
  `a04f9a7d02f726702f1c03ec4bac2e9ac2e471a3f722422165555774e7944c74`
- 固定消融与逐查询转换：
  `data/validation/bioaware_metdna3_external_negative_loso_ablation_v3_chemically_filtered/`
- 官方候选协议：
  `data/validation/bioaware_metdna3_external_negative_dreams_v2_chemically_filtered/`

本文件是 B4 的预先契约。结果产生后只能追加结果文件，不得回写本契约以迁就结果。
