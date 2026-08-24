# P0 Dropout 单变量结论 + RAW 组内重排器首版结果（2026-08-22）

**状态**：结果记录与决策存档；后续分层评价（seen/unseen-formula、near/mid、CI）未完成前，不得据此宣布"突破"。
**前序文档**：`G8R_M1_POSTMORTEM_AND_M1B_DECISION_20260822.md`、`DREAMS_ERROR_GUIDED_FINETUNING_MASTER_PLAN_20260821.md`。

---

## 一、P0：dropout 单变量实验（3 seed 正式结论）

### 1.1 结果表（3 seed 汇总，方向全部一致）

| 指标（3 seed mean） | dropout_on | dropout_off |
|---|---:|---:|
| margin Δ | **−0.0056**（3 seed CI 全负） | **+0.0004**（CI 跨 0） |
| pairwise acc Δ | +0.0032 | **−0.0042** |
| near margin Δ | −0.0027 | +0.0010 |
| Recall@1 Δ | +0.0043 | +0.0032 |
| macro-AUC Δ | +0.0031（部分 seed 不显著） | **+0.0035（3 seed CI 全 >0）** |
| corrected / introduced | 12 / 4（净 +8） | 9 / 3（净 +6） |

### 1.2 结论（严格口径）

1. **dropout 是真实混杂因子**：关掉后 margin 压缩彻底消失（−0.0056→+0.0004），`L_preserve` 初始值从 0.111 回到 0，preservation 从 0.996→0.9987。→ **"冻结 backbone 必须保持 eval()" 固定为后续所有 head-only 训练规范。**

2. **head-only 未通过预设门**：margin 只是持平（非正）、near 仅微升 +0.001、pairwise 不升反降（−0.0042）。→ 表述为 **"当前冻结 backbone＋线性 head＋M1b 损失未过门，降级为非优先路线"**，**不是**"线性 head 数学上无法分离近异构体"。

3. **最终决定**：head-only 主线正式停止继续烧预算，转向"保持 DreaMS 表征、仅对低置信候选做 RAW 峰级组内重排"的 reranker 主线。

---

## 二、RAW 组内重排器首版结果（g8r_val = 开发集）

### 2.1 配置

- 特征：`dreams_similarity` + 11 个 RAW 谱峰特征（sqrt/linear cosine、entropy_similarity、intensity_coverage、matched_peak_fraction、top10_match_fraction、neutral_loss_*、peak_count_ratio）。
- **不含 token、不含旧 8 项 panel**。
- 训练：within-query 差分（正例减 Top-5 困难负例）+ formula-group OOF 选 C。
- 门控：DreaMS Top1−Top2 分差，**阈值在 formula-group OOF 预测上选择**（修复了训练集内选 gate 的泄漏）。
- hard_k=5 固定，C=0.01。

### 2.2 结果

| 指标 | baseline | reranker | Δ |
|---|---:|---:|---:|
| Recall@1 | 0.8081 | **0.8516** | **+4.35 pp** |
| MRR | 0.8956 | **0.9195** | +2.39 pp |
| corrected / introduced | — | 44 / 17 | 净 **+27** |
| gate 覆盖率 | 0 | **46.5%** | — |

### 2.3 必须诚实标注的 caveat（未解决前不算"突破"）

1. **seen-formula 重叠 75.7%**：g8r_val 799 分子式中 605 个在 g8r_train 出现。当前 +4.35pp 主要是"见过分子式"的泛化，不能外推为"全新分子式"泛化。**必须拆 seen/unseen-formula 分层。**
2. **无 CI**：+4.35pp 需要 formula-cluster bootstrap CI 确认显著。
3. **无 near/mid 分层**：不知道增益来自 easy 还是 hard near 异构体。
4. **gate 覆盖 46.5%**（近一半 query），比旧 19.7% 宽得多，"只碰低置信"初衷被削弱，需评估是否可接受。
5. **g8r_val 是开发集**，已反复用于路线判断（元层面过拟合）；最终结论必须等一次性锁定测试集（Test-A/Test-B）。

### 2.4 当前可表述

> RAW 组内重排器在 g8r_val（开发集、75.7% 分子式重叠）上给出 +4.35pp 的初步正信号，但尚未通过 seen/unseen-formula、near/mid 分层与 CI 的门。

---

## 三、最终测试集（已锁定，两面板）

| 面板 | 用途 | n_queries | n_ik14 | n_formula | 异构体 query |
|---|---|---:|---:|---:|---:|
| **Test-A** | 代表性总体检索 | 2000 | 2000（每分子 1 个） | 1277 | 1415 |
| **Test-B** | 同分异构体挑战 | 1792 | 1792 | 746 | 1792 |

- 来源：HDF5 原始 `val` fold（45,185 谱 / 5,706 IK14），与 g8r 的 IK14 重叠 = 0。
- 抽样：按 IK14（多谱图分子不占权重）、seed 随机、**不按模型得分或异构体难度**。
- 每个 query 保存完整候选列表 + 三重哈希（候选图 / 构建脚本 / query 清单）。
- 文件：`data/validation/g8r_final_test/test_{a,b}_manifest.json`。
- **纪律**：只允许检查数量/缺失字段/候选覆盖，禁止提前计算任何模型优劣；模型与阈值完全冻结后才开启。

---

## 四、本次修正/收口的实现

1. `step4_m1b_train.py`：新增 `--backbone-eval`（P0 单变量开关）。
2. `train_g8r_raw_reranker.py`：RAW 组内重排器；gate 阈值改为 OOF 选择（修泄漏）；hard_k 固定 5。
3. `build_g8r_final_test.py`：废止 v0（全选异构体），重锁 Test-A + Test-B。
4. `audit_g8r_field_overlap.py`：五层重叠审计（spectrum row / IK14 / full InChIKey / formula / Murcko）。
5. `eval_g8r_d04_psd_probe.py` + `fix_psd_same_anchor_314.py`：PSD 探针同 anchor 指标修复到 314。
6. 服务器入口（sbatch）：`run_m1b_p0_dropout.sbatch`（已跑）、`run_summarize_p0_dropout.sbatch`、`run_g8r_raw_reranker.sbatch`。

---

## 五、下一步（分层评价，未完成）

复用 g8r 特征缓存，秒级重跑模型阶段，补齐裁决标准：

1. **seen / unseen-formula 分层**；
2. **near（MCES 0–2）/ mid（MCES 3–5）分层**；
3. **Recall@1 / MRR 的 formula-cluster bootstrap CI** + corrected/introduced 的公式组分布；
4. 分层结果通过后，才评估是否把 RAW reranker 作为近期主模型，并（最终）开启 Test-A/Test-B。

**一句话现状**：

> P0 证明 dropout 是混杂并应关闭、head-only 当前配方未过门而停止；RAW 组内重排器在开发集上给出大的初步正信号，但离"可采信的突破"还差分层与 CI 两道硬门。
