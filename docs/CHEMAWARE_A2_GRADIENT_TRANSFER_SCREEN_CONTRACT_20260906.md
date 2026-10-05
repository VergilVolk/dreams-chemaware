# ChemAware A2 动作到干净谱图的梯度迁移审计合同

**冻结日期**：2026-09-06  
**性质**：机制筛选；不更新权重；不产生性能 pp 主张。

**结果更新**：24-action 筛选已完成并判为
`CHEMAWARE_ACTION_TRANSFER_GRADIENT_FAIL`。三个非 naive 路由均有绝对正的
clean-margin influence，但均未显著超过 candidate-swapped 与 peak-permuted；
`naive_action_minus_clean` 为严格负向。完整结果、数学诊断及禁止把对照梯度作为
负监督的修正见
`docs/CHEMAWARE_A2_GRADIENT_TRANSFER_SCREEN_RESULT_20260906.md`。

## 1. 保留与否决对象

保留动作库已经通过的 117 个 identity-unique 化学动作。冻结的动作设置为
`conflict_attenuate / strength=0.75 / top_k=3`。它们属于 E1：在冻结
official DreaMS 上，正确结构动作相对 candidate-swapped 与 peak-permuted
对照具有候选边界优势。

否决的是旧 `direct_projected_guarded` 的迁移解释。最近的同预算 Stage-1
中，correct action 与 clean duplicate 的 Recall@1 增量相同，且两臂自身均
退化；所以该结果否决“原始 action full-list CE 能提供化学特异增量”，不否决
上述 117 个动作本身。

## 2. 数学目标

设干净谱图 margin 为

```text
m_c(theta) = s_true(theta; x_clean) - max_j!=true s_j(theta; x_clean)
```

候选分子的分数严格采用部署规则：该分子所有参考谱相似度的最大值。若候选
注入损失的参数梯度为 `g_a`，按训练器 head/backbone 学习率缩放后的更新方向
为 `u_a`，则一步梯度下降对干净 margin 的一阶影响是

```text
Delta m_c / step ~= - <grad_theta m_c, u_a>
```

审计的主指标为

```text
I_a = - <grad_theta m_c, u_a> / ||u_a||
```

`I_a > 0` 才表示等更新范数下，该动作梯度预计改善未修改的干净谱图边界。
动作后的当前排名、动作 loss 较小、梯度非零和共享参数都不能替代该条件。

## 3. 同时比较的注入路线

1. `action_query_only`：动作视图参与前向与反向，但候选参考 embedding 全部
   detach，用于隔离旧训练中候选侧重复梯度的干扰。
2. `action_forward_clean_backward`：前向分数严格等于动作视图；通过
   straight-through 路由，只把该 listwise 梯度送入同一谱图的 clean-query
   编码路径。它不回归教师分数、margin 或 embedding。
3. `action_routed_clean_pair`：动作只负责在真实候选列表中选出一条正参考—
   最难负参考边；pairwise loss 完全计算在 clean query 上，候选侧 detach。
4. `naive_action_minus_clean`：仅作为反例审计。动作通常已经是容易样本，故
   `g_action - g_clean` 可能反向降低 clean margin，不能未经实证直接训练。

## 4. 匹配对照与晋级门

每个 route 都在完全相同的 query、候选列表、动作算子、strength 与 top-k 下
比较：

- correct structure action；
- candidate-swapped action；
- peak-permuted action。

主筛选使用 24 个 formula-stratified、identity-unique 的已入库动作。对每个
route，以下四项必须同时成立：

1. correct action 的 `I_a` formula-cluster bootstrap 95% CI 下界大于 0；
2. correct minus candidate-swapped 的区间下界大于 0；
3. correct minus peak-permuted 的区间下界大于 0；
4. 至少 60% 的被审计动作具有正 clean-margin influence。

24-action 结果只允许晋级到全部 117 个动作的梯度确认。全部动作确认通过也只
允许构建一个有 clean-duplicate 与两种 matched action control 的小规模
embedding pilot；不直接授权正式训练或 outer evaluation。

## 5. 防止再次浪费 GPU 的运行约束

- CPU 单元测试和 py_compile 先于模型构造；
- preflight 在加载 117M DreaMS 前验证 graph、teacher、action-bank 哈希、数组
  形状、HDF5 行覆盖、official embedding 行覆盖和选定 action setting；
- 动作库的 peak-permuted 随机种子固定为 `20260935 + 41`。旧微调脚本曾错误
  地让该对照跟随 optimizer seed；现已拆分为独立的
  `--action-generation-seed 20260935`；
- 第一条动作在求梯度前必须以最大 `5e-4` margin 误差重放 action-bank 中的
  correct/swapped/permuted 结果；不一致立即停止；
- 审计不构建全数据 frozen-prefix cache，只编码 24 组 clean/action query，候选
  直接读取冻结 official embedding；
- 每条结果 fsync 到 JSONL；最终报告、CSV 与 COMPLETE 标志原子写入；
- 任一路线产生零范数或非有限 influence 时记为该路线 FAIL，不在末尾使整个
  已完成审计崩溃。

## 6. 唯一入口

```bash
sbatch tasks/run_chemaware_action_transfer_gradient_screen.sbatch
```

结果位于：

```text
data/validation/chemaware_action_transfer_gradients/run_<job_id>/screen24/
```

只读取 `report.json` 中的 `winner`、`pass_to_full_gradient_confirmation` 与各
route 的公式簇区间。`performance_pp_claim` 在该阶段必须为 `null`。

24-action 筛选完成后的逐 action 特异性分解必须通过 CPU Slurm 作业提交，禁止在
登录节点直接运行：

```bash
sbatch tasks/run_chemaware_action_transfer_specificity_summary.sbatch
```

该入口自动解析最新完整 screen，不接受手工 job id，申请恰好 1 张 GPU，并且不
手动指定任何内存参数。
