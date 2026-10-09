# 损坏长对话续作恢复状态（2026-10-09）

## 仓库边界

- 恢复时最新提交：`bc49f85`（2026-10-08，DreaMS embedding OOM batch 修复）。
- 2026-10-08/09 的 Enveda、统一模型与综合文档均仍为未跟踪工作区文件；本轮没有把它们误报为已提交版本，也没有覆盖其他既有改动。
- 本轮验证：核心与 Enveda 构建合同 `15 passed`；另有外测封存/一次性 opening 端到端合同 `1 passed`，合计 `16 passed`。

## Enveda-180 作业 2354608

- 已知输入文件：`285,632,591` bytes，MD5 `6d06a2c916aefe1c0b2d86bf40628441`；总谱数 `1,175,202`。这些数字来自恢复输入，当前轮未能重新访问服务器验证。
- 尝试的直连 `scv7tsl@101.6.121.161`、`huzplab@101.6.121.161` 与既有跳板路径均被 `Permission denied (publickey,...)` 拒绝，因此当前不能确认作业状态、远端日志或远端脚本哈希。
- 2026-10-09 泄漏审计发现旧链将若干已消费排除源设为 optional，且没有统一处理不同谱峰哈希合同。修订现在同时改变 prepare、builder、registry 与 sbatch；因此不得在运行中的作业内只替换某一个脚本。
- 作业 `2354608` 的产物至多作为构建审计，不能自动成为最终外测。获得服务器认证后应先保留其日志和工件，再以完整修订链提交一个新作业；新链缺任一 required source 即停止。

## 统一模型保真修复

本轮只完成代码修复与 smoke，不产生可引用的统一算法性能主张：

1. ChemAware Stage-1、Phase-A、V2 拆为独立资产；默认仅启用已合格 Stage-1。
2. `noise_rrf_1_7` 恢复为 `retired_audit_only`；provisional/development-only/conditional 模块不会被默认作业静默启用。
3. availability 从 query×module 改为 module×candidate；ChemAware V2/BioAware 等局部证据不再全候选广播。
4. 额外分数必须同时提供完全匹配的 `query_ids` 和 `candidate_ids`；候选顺序错位直接失败；bundle 记录 panel/base/extra SHA-256。
5. 可靠性允许精确零权重；默认取消 anti-collapse。
6. 训练器新增 R@1/3/5/10/20/50、MRR、candidate macro/micro/pooled AUROC、最强单模块冻结对照、corrected/introduced、`corrected-2*introduced` 和 formula-cluster bootstrap CI。

真实 GNPS identity-disjoint smoke 使用 `10,995` queries 的开发面板；固定公式验证折为 `2,175` queries。单 epoch 且缺 ChemAware Stage-1 时，融合 R@1 `86.8966%`，最强单模块对照为 P2b-on-Noise R@1 `86.7586%`，差值 `+0.1379 pp`，`17/14`，风险净收益 `-11`，formula-cluster 95% CI `[-0.3758,+0.6602] pp`。角色为 `development / consumed`，且只是工程 smoke；CI 跨零、风险为负，不能算正向统一模型结果，更不能据此打开 Enveda。

## 下一执行门

1. 获得可用服务器认证后，先检查 `squeue/sacct`、日志末尾和远端脚本 SHA-256；保留作业 `2354608` 的结果，但不得中途混用新旧脚本。
2. 完整同步 prepare、builder、registry、sbatch 后新提交 fail-closed Enveda 构建，回传 `report.json`、`checksums.sha256`、exclusion-source hashes 和完整文件清单；在此之前不计算任何模型分数。
3. 在 GNPS `development / consumed` 双面板上补齐 ChemAware Stage-1、Phase-A、V2 的主键化分数和逐候选适用域，运行多 seed、逐模块消融与最强单方法准入门。
4. 只有统一模型在两个 GNPS 开发面板均通过预注册门并冻结所有哈希后，才允许一次性打开 Enveda。
