# 课题交接记录（2026-08-22 晚）

**当前一句话**：RAW 组内重排器（v1）已冻结，最终大规模盲测（Test-A/B/C）脚本与 sbatch 就绪，等待在服务器提交并读取结果。

---

## 一、本阶段走完的路线

```
G8R M1b（head-only 微调）
  → 失败（正负一起抬升，margin 压缩）
  → P0 dropout 单变量（发现冻结 backbone 仍开 dropout 的工程 bug）
  → 结论：head-only 未过门，降级为非优先路线（不是"head 数学上不可能"）
  → 转向 RAW 候选组内重排器（不动 DreaMS 权重）
  → 开发集 +4.35pp（显著）→ 分层 + CI + McNemar → near 净+13（显著）
  → 冻结 v1 → 锁定大规模最终测试 A/B/C
```

## 二、核心结论（已有多重证据，开发集）

| 指标 | 值 | 显著性 |
|---|---|---|
| 整体 Recall@1 | 0.8081 → 0.8516（+4.35pp） | bootstrap CI>0、McNemar p=0.0009 |
| unseen-formula | +4.33pp | 与 seen 一致 |
| near（MCES 0–2）净修正 | +13（20 修正/7 引入） | CI>0、McNemar p=0.019（唯一显著层） |
| gate 覆盖率 | 46.5%，gate-off 反事实仅 1/1 | gate 有效 |

**严谨表述**：RAW 重排器是"通用检索改善恰好覆盖 near"，**不是** near 特异突破；最难 near 硬负例（预挖面板）仍净+3 不显著。**没动 DreaMS 权重**，不是"微调出更好的 DreaMS"。

## 三、冻结的 RAW-v1（唯一正式版本）

- 逻辑回归，12 维特征（`dreams_similarity` + 11 个 RAW 峰匹配特征），fit_intercept=False。
- C=0.01、hard_k=5、峰匹配容差 0.02 Da、gate 阈值 0.24098341166973114（Top1−Top2 分差，OOF 选定）、require_disagreement=false。
- 训练= g8r_train（formula-group OOF），开发= g8r_val。
- 正式转换表 = gated 44/17（ungated 45/18，差异 2 query 见 reconcile 脚本）。
- 冻结文档：`docs/RAW_RERANKER_V1_FREEZE_20260822.md`。

## 四、最终测试（大规模，已锁定）

- 候选库 = 全 HDF5（231,104 谱）减去 g8r IK14 → 166,842 可用谱。
- 有效 query = **102,449**（十万级，数据上限；"几十万"物理不可达）。
- 均衡三面板（按 IK14，面板间零重叠）：

| 面板 | n_queries | n_ik14 | 用途 |
|---|---:|---:|---|
| Test-A | 35,662 | 4,287 | 最终盲测 |
| Test-B | 32,096 | 4,287 | 最终盲测 |
| Test-C | 34,691 | 4,288 | 下一代 reserve（不复用作 RAW-v1 盲测） |

- 数据：`data/validation/g8r_final_test_large/`（manifest.json + queries.csv）。

## 五、待办（服务器）

```bash
sbatch tasks/run_g8r_raw_reranker_reconcile.sbatch      # 对账 44/17 vs 45/18（秒级）
sbatch tasks/run_g8r_raw_reranker_final_test.sbatch      # 最终盲测（~1–1.5h，ONE-SHOT）
```

- 看 `data/validation/g8r_raw_reranker_final_test.json` 的 `panels.{a,b,c}.reranker_recall1` vs `base_recall1`。
- 通过标准：Recall@1 95% CI 下界 > 0，且 MRR 不退化。

## 六、下一步方向（最终测试通过后）

1. 若最终测试通过 → 得到"官方 DreaMS + RAW 重排器"两阶段检索系统，写论文。
2. 近异构体 hardest 仍未解 → 下一代方向：**DreaMS peak tokens + 完整候选组 listwise 排序 + 峰级 RAW 证据**（用 Test-C）。
3. 噪声/反事实遮蔽仅作增强项，不充当主架构；反事实峰遮蔽需先确认"共享主峰是 near 误判因果来源"。

## 七、纪律（全程反复被纠正，写死）

1. 不把"证据不足"写成"已证明"；不把"head 配方未过门"写成"head 数学不可能"。
2. 训练/开发/测试 IK14 零重叠；分子式层必须分层报告 seen/unseen。
3. 配对比较用 corrected/introduced + McNemar + cluster 配对 bootstrap，不用两个独立二项区间。
4. 预挖 near 硬负例（压力测试）与真实检索 near 错误是两个口径，必须分开报告。
5. 服务器上跑的一切必须写 sbatch，不裸命令。
