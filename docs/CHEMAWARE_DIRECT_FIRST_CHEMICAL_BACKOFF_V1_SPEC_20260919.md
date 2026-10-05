# ChemAware Direct-First Chemical Backoff V1

日期：2026-09-19  
状态：`POST-OUTER DEVELOPMENT / NEW EXTERNAL CONFIRMATION REQUIRED`  
适用范围：未来新的独立评价面板；不得重评已消费的 formula role 4。

## 1. 失败事实与方法重置

已完成的 canonical outer 表明，residual ChemAware 相对 official DreaMS 有正的总增益，
但没有达到3 pp注册门，也没有证明相对 nuisance-only 的化学独立增量或相对
same-feature-direct 的 residualization 增量。same-feature-direct 的点估计和风险效用更好。

因此，新方法不再让化学残差无条件改写主决策，也不把提高全局阈值当作根本修复。
它把问题改写成两个冻结专家的选择性覆盖：

1. direct expert 负责主候选干预；
2. residual ChemAware 只覆盖 direct expert 明确拒绝的查询；
3. 两者同时行动或发生候选冲突时，direct expert 优先；
4. 两者均拒绝时保持 official DreaMS。

## 2. 数学定义

对查询 `q`，direct 与 residual expert 分别输出候选效用 `u_D(q,c)`、`u_C(q,c)`，
以及在 formula-role-2 冻结的阈值 `tau_D`、`tau_C`：

```text
a_D(q) = argmax_c u_D(q,c),  if max_c u_D(q,c) >= tau_D; otherwise abstain
a_C(q) = argmax_c u_C(q,c),  if max_c u_C(q,c) >= tau_C; otherwise abstain

a(q) = a_D(q),               if direct acts
       a_C(q),               if direct abstains and ChemAware acts
       official baseline,    otherwise
```

这个裁决不读取标签、正确候选、旧排名、公式身份或任何 action outcome。化学专家不能覆盖
direct 已接受的候选，只能提供增量覆盖。

## 3. 当前开发证据

在 canonical run 2338337 的 role-3 development ledger 上：

- direct：63 corrected / 5 introduced，Recall@1 `+3.0067 pp`；
- residual：64 / 4，`+3.1104 pp`；
- direct-first residual-backoff：78 / 6，`+3.7325 pp`；
- formula-cluster 95% CI：`[+2.6927,+4.8339] pp`。

这是在已使用 role 3 上形成的新组合，只是方法开发证据。它不能更新、替代或“修复”已完成
的 role-4 outer 结果。

## 4. 已否定的简单修复

五折 formula cross-calibration 将 residual 阈值从 `0.8358` 提高到 `1.2311`，在 role 3
把 introduced 从4降到3，但 corrected 从64降到54，Recall@1 降为 `+2.6439 pp`；role-2
OOF 中 correct 相对 alignment-permuted 的风险优势区间下界仍为0。故“只抬阈值”不能作为
下一代主算法。

## 5. 下一验证合同

下一次评价必须使用新的独立来源或新冻结面板，同时比较：

1. official DreaMS；
2. same-feature-direct；
3. residual ChemAware；
4. direct-first residual-backoff；
5. candidate-count、mass、reference multiplicity相匹配的非化学控制。

主比较是 backoff 相对 direct，而不是相对 official。要求：

- Recall@1 增量的 formula/source-cluster CI 下界严格大于0；
- corrected 严格大于两倍 introduced；
- Recall@3/5/10/20/50、MRR、Micro-AUC、Macro-AUC均不下降；
- residual fallback 的新增动作必须单独报告覆盖、纠正、误伤和公式支持；
- 不允许根据新面板结果重新选择专家顺序或阈值。

未通过时，结论应为：化学残差存在开发期互补覆盖，但不能稳定转化为独立部署增量。
