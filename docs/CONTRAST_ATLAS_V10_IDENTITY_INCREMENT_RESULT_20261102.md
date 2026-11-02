# Contrast Atlas v10：Full23 同轴身份增量结果

## 结论

Full23 已经在 Atlas v10 的完整模型无关候选轴上完成真实应用，但在当前隐藏身份锚点上**没有通过相对最强基线的身份增量门**。

它显著优于 Official DreaMS、ChemAware Stage-1、ChemAware Phase-A 和 Noise-ChemAware FullMSG；这说明统一系统确实兑现了部分互补信息。然而，它没有超过 WSE，也没有超过冻结 P2b。当前结果支持“Full23 比若干 encoder 更强”，不支持“Full23 在 Atlas 跨研究身份恢复中总体最优”。

数据角色为 `development_consumed`。本结果不是外部验证，也不是 MSI level-1 身份确认。

## 固定评测对象

- Full23 真值盲评分作业：`2433291`
- 身份解盲分析作业：`2433751`
- 来源 ID：16
- 研究级 contrasts：53
- 模型无关定向候选对：294,236
- query × target-source blocks：826,905
- 有至少一个候选的 blocks：141,332
- 隐藏身份真值合格 blocks：2,257
- 定向来源 dyads：186

谱学评分阶段没有读取响应 `beta/SE`、表型、登记关系或身份真值。身份真值只在九种方法全部完成候选块决策后打开。

隐藏真值定义为共同库策略下获得支持、具有合法完整 InChIKey，且目标研究候选块中存在相同完整 InChIKey 的 observed-ion block。该真值来自 GNPS/WSE 库匹配流程，对 WSE 可能有结构性优势；它是开发期身份锚点，不是标准品确认。

## 九方法身份结果

| 方法 | 正确 | 错误 | 弃权 | 无条件 R@1 | 条件准确率 | MRR | utility |
|---|---:|---:|---:|---:|---:|---:|---:|
| WSE | 2,144 | 107 | 6 | 94.993% | 95.247% | 97.331% | 0.8551 |
| Official DreaMS | 2,071 | 171 | 15 | 91.759% | 92.373% | 95.019% | 0.7661 |
| Noise V1 | 2,084 | 157 | 16 | 92.335% | 92.994% | 95.565% | 0.7842 |
| P2b Noise | 2,133 | 109 | 15 | 94.506% | 95.138% | 96.741% | 0.8485 |
| Noise MSG | 2,089 | 151 | 17 | 92.556% | 93.259% | 95.699% | 0.7918 |
| ChemAware Stage-1 | 2,072 | 170 | 15 | 91.803% | 92.417% | 95.088% | 0.7674 |
| ChemAware Phase-A | 2,055 | 187 | 15 | 91.050% | 91.659% | 94.724% | 0.7448 |
| Noise-ChemAware FullMSG | 2,078 | 164 | 15 | 92.069% | 92.685% | 95.388% | 0.7754 |
| **Full23 v1** | **2,132** | **110** | **15** | **94.462%** | **95.094%** | **96.762%** | **0.8471** |

其中 `utility=(correct−2×wrong)/N`。表中 utility 是 block-micro 值；正式配对比较采用来源 dyad macro 值。

回答数、边数或覆盖率不能替代准确性。Full23 回答 136,498/141,332 个非空块并形成 38,358 条互惠边，但其互惠边数并非九方法最多。

## Full23 相对各方法的来源配对效用

| 基线 | Full23 − 基线 | 普通95% CI | simultaneous 95% lower | leave-one-source 最低值 | 判定 |
|---|---:|---:|---:|---:|---|
| WSE | −0.0261 | [−0.0677, +0.0064] | −0.0848 | −0.0369 | 不通过 |
| Official DreaMS | +0.0690 | [+0.0163, +0.1218] | +0.0103 | +0.0426 | 通过 |
| Noise V1 | +0.0420 | [−0.0050, +0.0837] | −0.0167 | +0.0245 | 不通过 simultaneous CI |
| P2b Noise | +0.0019 | [−0.0188, +0.0235] | −0.0567 | −0.0038 | 不通过 |
| Noise MSG | +0.0350 | [−0.0150, +0.0805] | −0.0237 | +0.0164 | 不通过 simultaneous CI |
| ChemAware Stage-1 | +0.0715 | [+0.0395, +0.1086] | +0.0128 | +0.0511 | 通过 |
| ChemAware Phase-A | +0.0752 | [+0.0253, +0.1235] | +0.0166 | +0.0584 | 通过 |
| Noise-ChemAware FullMSG | +0.0822 | [+0.0494, +0.1176] | +0.0235 | +0.0640 | 通过 |

预定门要求 Full23 相对所有非集成方法的 simultaneous 95% CI 下界大于零，且所有 leave-one-source 增量为正。正式状态为：

`FULL23_FAILS_ATLAS_IDENTITY_INCREMENT_GATE`

## Full23 与 WSE 的错误交换

| WSE → Full23 | blocks |
|---|---:|
| wrong → correct | 9 |
| abstain → correct | 5 |
| correct → wrong | 12 |
| correct → abstain | 14 |
| correct → correct | 2,118 |
| wrong → wrong | 98 |
| abstain → abstain | 1 |

因此，Full23 的确纠正了一部分 WSE 错误，也增加了 WSE 无法回答的正确身份；但这些收益不足以抵消新引入的错误和丢失的正确回答。

相对 P2b，Full23 为 `wrong→correct=13`、`correct→wrong=14`，其总体表现近乎相同而略低。当前 Atlas 身份轨最合理的结论是 WSE 与 P2b 仍为最强基线，Full23 尚未证明净优势。

## 科学边界

1. 当前轴只评价同观测前体 10-ppm 候选，不评价跨加合物中性实体，也不评价跨质量化学关系。
2. Full23 在此轴上没有候选结构输入，ChemAware V2 按合同为 unavailable no-op；这不是 Full23 在完整结构候选任务中的全能力测试。
3. GNPS/WSE 库支持身份可能偏向 WSE，因此本结果是保守的开发比较。仍需独立标准品、来源原生身份或未参与建图的身份真值确认。
4. 身份层失败不能自动否定研究响应层；反过来，改变更多研究响应也不能证明实体身份正确。
5. 本结果不能支持 unknown 信息增量、反向代谢组学增量或新生物学发现。

## 下一项决定性实验

使用九种方法各自形成的互惠边，在完全相同的53个 contrast、parent-project 排除、响应算子和固定分母上重算：

- double-centered known-anchor ↔ unresolved response correlation；
- leave-one-study-out 最低相关；
- axis-label permutation；
- 开放 Top-1 相对随机增量；
- known-only、measurement-matched known、shuffled-unresolved-MS2 与 all 四世界。

每种方法必须使用自己的候选决策，不得继承 WSE 边。不同方法的 raw score 不具有共同概率含义，因此主分析使用不加权的互惠边 membership，双向 raw score 只作敏感性分析。

只有研究响应特异性、unknown 条件增量和算法增量同时通过，才能将 Atlas 写成服务一线非靶向或反向代谢组学的方法；否则应保留为规模化开发资源和负结果。
