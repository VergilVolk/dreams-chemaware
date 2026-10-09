# 统一算法数据泄漏与外测开启总账

日期：2026-10-09  
状态：算法整合的强制边界；优先于较早的融合、路由与“全模块激活”说明。

## 1. 当前裁决

截至本日，统一算法没有合格的最终外测结果。MassSpecGym、GNPS Gold/Silver、MassBank 2026.03、项目内 MoNA 与 MSnLib 均已被开发、选择、解释或压力测试消费，只能承担 `development/consumed`。Enveda-180 仍是计划中的唯一终局外测，但旧构建链缺少完整的已消费谱峰哈希排除，并把若干排除源设为 optional；因此由旧链生成的 Enveda 工件不能用于最终主张。

服务器作业 `2354608` 在本次修订前启动，不能自动继承新合同。不得在它运行中覆盖已启动的 prepare 脚本；其可能产出的工件至多作为构建审计。最终 Enveda 工件必须由修订后的 fail-closed 链重新生成，并出现 `exclusion_policy_complete=true`、全部 required source 为 loaded、选中 identity overlap=0、选中 normalized-spectrum overlap=0 和完整 SHA-256 链。

## 2. 已确认问题及永久处置

| 问题 | 类型 | 证据/影响 | 永久处置 |
|---|---|---|---|
| S1 以 positive-vs-best-negative margin 选择方法 | 直接读取真值的特征泄漏 | 曾产生 identity/formula `+4.93/+5.44 pp` | 正式撤回；数字不得再引用为性能 |
| router v2 两面板共享 5,237/5,261 formula queries | 开发/评价面板交叉污染 | 曾产生 `+2.93/+4.03 pp` | 正式撤回；query/formula/source 交集必须在拟合前为零 |
| B47 U1 旧候选顺序信息 | 候选轴泄漏 | 修复后 U1b v2 为 0/0、R@1 0 | 旧 U1 增益不得使用；所有额外数组按 query/candidate ID 精确对齐 |
| P0-G no-match 阳性未从候选中移除 | 标签/任务构造失效 | 旧 no-match AUC 不回答开放集问题 | 旧结果失效；重做必须物理删除真候选 |
| LCNEC P2 v2 entropy 实现错误 | 实现失效，不是数据泄漏 | 23 个 medium/high 与 91% dark 不可复现 | 全部失效待重算 |
| P2b raw 三通道、P2b composite 及其派生重排同时入模 | 相关证据重复计票 | 夸大同一谱学证据，不是独立多模态 | 一次实验只能用 raw family 或 frozen composite；默认只允许 composite |
| retired RRF 被重新设为 active | 资格回退 | Stage-1/GNPS 已失败仍被强制正权 | 保持 `retired_audit_only`，生产 bundle 禁止出现 |
| 全模块强制正权 | 有害模块不能退出 | 失败、域外和冗余模块必然污染 | 允许 family 精确零权；缺失为 unavailable，不是零证据 |
| 训练器同时接受 train 与 final test | 外测反复探测通道 | 可按 Enveda 结果反复换 seed/模块/超参 | 训练与外测拆成两个进程；训练器拒绝 `--test-bundle` |
| Enveda exclusion sources optional | 去污染不完整 | 缺源仍可能生成“外部集” | registry 中所有已消费源 required，缺一即停 |
| Enveda 仅排 identity/formula，未完整汇总 consumed peak hash | 精确谱复制风险 | identity-disjoint 不能排除同谱异注释/命名差异 | 对所有可用已消费源计算 normalized-spectrum hash 并排除 |

撤回证据见 `GLM_ENSEMBLE_COMPLEMENTARITY_RETRACTION_AND_TRUTHBLIND_RESULT_20261007.md`；当前算法资格见 `ALGORITHM_INTEGRATION_AND_BIOLOGY_CURRENT_STATE_20261009.md`。

## 3. 新的 fail-closed 数据链

1. `tasks/unified_consumed_sources_v1.json` 是唯一已消费源 registry。MassSpecGym、GNPS、MoNA 正/负、MassBank 2026.03、MSnLib 全部为 required。
2. `tasks/prepare_enveda180_scoreblind_manifest.py` 在不加载模型、不计算性能的条件下汇总 identity、formula 与 normalized-spectrum hash，并记录每个源文件 SHA-256。任何 required 文件缺失即失败。
3. `tasks/build_enveda180_cross_condition_benchmark.py` 验证 audit report、manifest、conflict ledger、源 MGF 与 registry 哈希，排除 consumed identity 和 spectrum overlap，才允许写入 `allow_final_claim=true`。
4. `tasks/train_grand_unified_evidence_model.py` 只读取 development bundle，保存模型、训练 bundle、registry、超参数与验证结果；禁止接收 final test。
5. `tasks/seal_grand_unified_external_bundle.py` 将 Enveda component scores 与冻结 panel、冻结模型、registry、benchmark report/checksums 全部按 SHA-256 绑定；只允许 registry status=`enabled` 的模块。
6. `tasks/evaluate_frozen_grand_unified_external.py` 在读取标签前以 exclusive-create 写 opening ledger。成功或失败都保留账本；同一路径不能重试。它逐项复核 component bundle、panel、模型、freeze report、registry 和 benchmark 哈希，并要求预先声明唯一主比较基线。

这套链只阻止可机械检测的泄漏；它不把“运行了脚本”误写成科学独立性。模块分数是否通过人工查看 Enveda truth 后生成，仍需作业日志、提交时间和 score-generation manifest 证明。

## 4. Enveda 开启前的硬门

以下条件必须同时满足，否则不得运行 one-time evaluator：

- GNPS development/consumed 上唯一模型、模块清单、seed、超参数、abstention、阈值和主比较基线已冻结；
- 与 WSE、Noise V1、ChemAware Stage-1、P2b-on-Noise 的同候选轴比较和 leave-one-family-out 消融完整；
- 统一模型在两个 GNPS 面板均未出现不可接受的 corrected/introduced 风险，并报告 formula-cluster paired CI；
- 不含 RRF、BioAware 静态目录、ChemAware V2 development-only 或 MSG provisional；除非它们在 Enveda 开启前按独立合同晋级并更新 registry；
- Enveda 新 benchmark 是由本文件第3节链路重新生成，不是作业 `2354608` 的旧链遗留产物；
- opening ledger 路径从未存在，输出路径从未存在；
- 运行命令与全部输入 SHA-256 在揭盲前写入预注册记录。

## 5. 不能消除的残余边界

项目当前无法完整枚举 DreaMS 无监督预训练语料的结构命名空间。因此，新链可以严格排除仓库可得的已标注 identity/formula 和 normalized-spectrum 精确重叠，但不能证明 Enveda 与 DreaMS 预训练语料“零曝光”。最终论文必须写为“对可枚举已消费标注源及可得谱峰哈希去重”，不得写成“对全部预训练数据完全独立”。

Enveda 还是标准谱库跨碰撞能检索，不包含真实样本事件上下文。它可以终局检验 Noise、ChemAware encoder、WSE、P2b 及合格谱学融合，但不能验证 BioAware B47。B47 必须在独立、事件级、带真实样本上下文且 truth/seed 隔离的外部面板上另行一次性评价。

## 6. 当前允许的下一步

只允许在 GNPS/MSG 等已消费开发集上修正、消融和冻结；这些结果必须显式标记 `development/consumed`。不允许为了“补齐模块”复活负结果，不允许将不同面板百分点相加，也不允许依据 Enveda 的首次结果再选模型。若 Enveda 首次结果为负，科学结论就是该冻结统一模型外推失败；后续修改必须等待新的、未打开外部来源。
