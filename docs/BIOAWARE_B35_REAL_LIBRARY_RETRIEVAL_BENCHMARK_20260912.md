# BioAware B35：真实大谱库检索基准

日期：2026-09-12  
状态：本地完整复算与独立验证脚本通过；服务器正式复现入口已冻结

## 1. 这次回答的问题

B35 不再只问“BioAware 改对了多少个 Top-1”，而是在同一批真实 MS/MS
查询、同一参考谱库、同一候选图上，比较：

1. 官方 DreaMS 完整分子排名；
2. 官方 DreaMS + 冻结 BioAware B30 动作后的完整分子排名。

主分析对每个物理查询谱只计一票，覆盖六个来源、两种极性，共 `1,631`
个真实 MS/MS 查询、`398` 个真值分子身份、`328` 个分子式。上游搜索库含
`259,176` 张参考谱和 `30,984` 个分子身份；查询局部候选使用 strict
10 ppm、同加合物过滤。分子分数是该分子全部参考谱与查询的最大官方
DreaMS cosine。

## 2. BioAware 如何形成完整排名

B30 是候选动作路由器，不是给所有候选输出连续分数的模型。为了避免凭空
发明一个新的打分器，B35 使用最小充分扩展：

- B30 干预时，把其冻结选择的候选提升为唯一第 1 名；
- 其余候选之间的 DreaMS 相对顺序完全不变；
- B30 没有干预时，完整 DreaMS 排名完全不变；
- B30 只在已准入的负离子域工作，正离子查询明确弃权；
- 真值与错误候选并列时，仍按严格协议把并列计为真值之后。

因此 Recall@2/5/10/20、MRR 和 query-level AUROC 都有确定且可审计的定义。

## 3. 主结果：1,631 个物理真实查询

| 指标 | 官方 DreaMS | DreaMS + BioAware B30 | 差值 |
|---|---:|---:|---:|
| Recall@1 | 0.75414 | 0.78296 | **+0.02882** |
| Recall@2 | 0.88289 | 0.88473 | +0.00184 |
| Recall@5 | 0.98529 | 0.98590 | +0.00061 |
| Recall@10 | 0.99755 | 0.99755 | 0 |
| Recall@20 | 1.00000 | 1.00000 | 0 |
| MRR | 0.85236 | 0.86707 | **+0.01471** |
| macro-query AUROC | 0.85475 | 0.86820 | **+0.01345** |
| micro within-query AUROC | 0.84106 | 0.85181 | **+0.01075** |
| pooled candidate AUROC | 0.82828 | 0.82879 | +0.00052 |

Recall@1 有 `50` 个修正、`3` 个新增，净修正 `47`；McNemar exact
`p=5.52e-12`。公式簇 bootstrap 95% CI 为 `[+0.01553,+0.04388]`，
身份簇 bootstrap 95% CI 为 `[+0.01623,+0.04320]`。

MRR 的公式簇 CI 为 `[+0.00767,+0.02263]`，macro-query AUROC 的公式簇
CI 为 `[+0.00626,+0.02155]`。Recall@2 的点估计为正但 CI 跨零；
Recall@5 只剩一个净修正；Recall@10/20 已接近或达到天花板。

## 4. 效果来源分解

### 4.1 负离子准入域

`753` 个去重物理查询：

- Recall@1：`0.64409 -> 0.70651`，`+6.2417 pp`；
- corrected / introduced：`50 / 3`；
- MRR：`+3.1873 pp`；
- macro-query AUROC：`+2.9135 pp`。

### 4.2 正离子弃权域

`878` 个真实查询：DreaMS 与 DreaMS+BioAware 的每个 rank 完全相同。
Recall@1 均为 `0.84852`。这证明综合主值没有通过正离子现场重训或迁移
调参获得；B30 在未准入域严格回退到官方 DreaMS。

### 4.3 为什么综合增益是 +2.88 pp 而不是 +5.93/+6.24 pp

`+5.93 pp` 是 860 行历史 OOF 口径；`+6.24 pp` 是相同负离子动作按物理
谱去重后的适用域口径。B35 的 `+2.88 pp` 是把真实部署中的 878 个正离子
弃权查询也纳入后得到的预注册主值。三者的分母和科学问题不同，不能互换。

## 5. AUC 口径必须分开

DreaMS 论文约 `0.85` 是 NIST20 个体谱图对的 pooled ROC-AUC：分数是两张
谱的 DreaMS cosine，标签是两张谱是否对应同一 IK14。B35 则是分子候选
检索，候选分数先在一个分子的多张参考谱上取最大值；BioAware 又是候选
动作而非新的谱图对 cosine。因此：

- `macro-query AUROC` 是本任务最直接的候选检索 AUC；
- `micro within-query AUROC` 按每个真值—错误候选比较计权；
- `pooled candidate AUROC` 是展平候选行后的辅助指标；
- 以上都不能称作“精确复现 DreaMS 论文 0.85”。

主表中官方 DreaMS 的 macro-query AUROC `0.85475` 与论文的 `0.85` 数值接近
纯属不同协议下的巧合，绝不能据此宣称复现论文。

## 6. 复现门

B35 在输出任何性能数字之前强制验证：

1. B30 860 个负离子查询的官方 Recall@1 精确为 `0.6604651163`；
2. B30 完整结果精确为 `0.7197674419`；
3. B33 的 `3,314` 个候选分数与全部参考谱重算图一致，最大允许误差
   `2e-6`；
4. B33 保留 `22,737` 条候选—参考谱链接及 `3,515` 张实际参考谱；
5. 正离子 878 个查询逐查询 rank 不变；
6. 1,738 行历史表正确折叠为 1,631 个物理查询；
7. 所有输入与输出写入 SHA256；评估期间不拟合模型。

## 7. 可以与不可以声称什么

当前可以说：

> 在一个含 259,176 张参考谱、30,984 个分子身份的真实大库检索协议中，
> 冻结 BioAware B30 作为负离子专用、正离子自动弃权模块，在 1,631 个真实
> 物理 MS/MS 查询上把官方 DreaMS Recall@1 从 75.41% 提高到 78.30%，
> 提升 2.88 个百分点；公式簇与身份簇 95% CI 均严格大于零。

当前不可以说：

- 独立外部盲测已完成；
- 精确复现或超过了 DreaMS 论文 NIST20 `0.85` AUROC；
- BioAware 已普遍改善正负两种极性；
- 反应路径是 B30 增益的因果来源；
- 共享 DreaMS embedding 已改善；
- 已经达到跨数据库 SOTA。

根本限制是这六个来源都已用于 BioAware 动作开发，虽然使用了 source-LOSO
和公式/身份隔离，仍属于 opened cross-fitted development benchmark。下一篇
主张“真实外部泛化”需要一个从未参与动作选择的新负离子标准品/Level-1
队列，并原样加载 B30 工件。

## 8. 固定实现

- 评估：`tasks/evaluate_bioaware_b35_real_library_retrieval.py`
- 单测：`tasks/test_bioaware_b35_real_library_retrieval.py`
- 独立验证：`tasks/validate_bioaware_b35_real_library_retrieval.py`
- 服务器入口：`tasks/run_bioaware_b35_real_library_retrieval.sbatch`

服务器只需提交：

```bash
sbatch tasks/run_bioaware_b35_real_library_retrieval.sbatch
```

日志直接写入仓库根目录：`bioaware_b35_real_<jobid>.out/.err`；结果写入
`data/validation/bioaware_b35_real_library_retrieval_<jobid>/`，拒绝覆盖已有
目录。
