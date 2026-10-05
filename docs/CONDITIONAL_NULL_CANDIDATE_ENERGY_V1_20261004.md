# 条件空对照锚定候选能量：Noise × ChemAware 集成开发契约

**冻结日期：2026-10-04**  
**服务器入口：** `tasks/run_conditional_null_energy_stage1_1gpu.sbatch`  
**性质：** 方法开发与一次性方案筛选，不是独立外部结论。

## 1. 为什么不是“多路投票”

本方法不训练路由器，不按 query 选择专家，也不把若干历史准确率直接相加。对每个候选只产生一个最终能量：

```text
final score = frozen V1 spectral score + bounded spectral residual
            + bounded chemistry increment conditioned on spectral state
```

其中化学增量被定义为：

```text
f(spectral, chemical contrast) - f(spectral, zero chemical contrast)
```

因此，当真实化学证据与三类空对照没有差别时，化学贡献在数值上严格为零；模型不能仅凭“这是 baseline 候选”或“此处缺少化学动作”改写排序。

## 2. 实际纳入的已有组件

- **Noise V1/T1-T3 encoder：** 作为冻结基础谱图分数；
- **P2b：** 作为已验证的局部谱学重排证据；
- **P2b neutral-loss 通道：** 作为独立谱学补充；
- **ChemAware multi-null V2：** 不输入已经阈值化的最终动作，只输入正确规则相对 zero、reversed、candidate-ID deranged 三类空对照的候选级中心化残差；
- **三通道 RRF：** 明确保留为历史负对照。它在 Stage-1/GNPS 确认中失败，因此不会作为输入或部署分支；“用上”它的正确方式是保留否证边界，而不是重新包装进总分；
- **BioAware：** B47 外部门通过前不进入排序器。其接口保留，但本阶段不拟合、不计入“协同”。

## 3. 三个关键防泄漏设计

1. **候选顺序不变量空对照。** 原候选图把正确分子固定在第一位，因此按数组位置循环移位并不盲。新实现使用候选 InChIKey 和 query row 建立规范顺序，再做 query-keyed 无固定点置换；任意重排候选并逆置换输出，结果必须完全一致。
2. **只使用空对照中心化化学残差。** 原始 ChemAware total utility 含 nuisance、参考谱数量和官方分数，不直接进入模型。若 correct 与所有 null 相同，全部化学特征必须严格为零。
3. **逐行来源核对。** exporter 同时读取 frozen evidence、corrected graph 和 ChemAware manifest，严格比较 `query_row`、`query_ptr`、`molecule_ptr`、`pair_candidate_row`、候选身份、标签账本和图文件哈希；不能只靠浮点分数“看起来一样”。

## 4. 四个固定臂

| 臂 | 定义 | 回答的问题 |
|---|---|---|
| `joint` | 谱学残差 + 以谱学状态为条件的化学增量 | 完整方法 |
| `no_interaction` | 两个独立网络分别有界后相加 | 学到的跨证据交互是否必要 |
| `spectral_only` | 只保留谱学残差 | 化学模块是否有独立增量 |
| `chem_only` | 只保留零锚定化学残差 | 谱学模块是否有独立增量 |

`no_interaction` 的两个输出分别经过非线性再相加；不存在共同 `tanh(s+c)`，因此交叉偏导严格为零。

## 5. 训练目标

- 分子式五折 OOF；
- query 内 listwise cross-entropy；
- 对九个注册空对照分别计算完整 query 的 listwise loss；
- 要求真实证据损失优于每个空对照，优化最差空对照臂；
- 残差平方信任惩罚与统一有界幅度，防止覆盖整个 V1 基线。

九个空对照为三种谱学候选-ID置换、三种化学空对照，以及一一配对的三种联合空对照。

## 6. 统计裁决与“1+1>2”的严格定义

所有对比均为 query 配对差，并按分子式簇 bootstrap 10,000 次。五个预注册对比使用 Bonferroni family-wise 95% 区间：

1. `joint - baseline`；
2. `joint - spectral_only`；
3. `joint - chem_only`；
4. `joint - no_interaction`；
5. `joint + baseline - spectral_only - chem_only`。

只有以下条件同时成立，内部结果才记为 `SUPERADDITIVE_DEVELOPMENT_SIGNAL_EXTERNAL_CONFIRMATION_REQUIRED`：

- 第 1--5 项相应主门下界均严格大于 0；
- `corrected > 2 × introduced`，且风险净收益为正；
- 五个 OOF fold 的 joint-baseline 均不为负；
- near-structure 子集不回退；
- 在所有候选参考谱数量一致的 query 子集中，`joint - spectral_only` 的公式簇 CI 下界严格大于 0；
- joint 独有纠正数大于 joint 独有新增错误数。

若 joint 只优于 baseline，只能称“联合模型有开发增益”；若 joint 优于两个单模块但严格超加和式未过门，只能称“互补”，不能称协同。

## 7. 参考谱多重性与 RRF 边界

主报告必须同时给出：

- reference-count-only Top-1；
- 候选参考谱数量完全匹配子集；
- near-structure 子集；
- corrected / introduced / risk-net。

这不能完全替代未来的单参考谱重复抽样，但足以阻止“参考谱越多，max 越大”被误写成化学贡献。若匹配子集的化学增量门失败，停止化学协同主张。

RRF 不参与最终能量。其独立确认失败是方法选择依据，不是被遗忘的正结果。

## 8. 结果资格边界

MassSpecGym 上的二级 formula-OOF 仍使用了已经参与上游 P2b、Noise 和 ChemAware 开发的面板。因此，无论数值多高，都只能用于冻结一个候选方法；不能称为独立泛化或论文最终性能。

真正文章级结论必须把冻结后的唯一方法一次性应用到未被任何组件训练、阈值选择或架构选择消费的外部面板，并至少满足：

- identity-disjoint 与 formula-disjoint 均不回退；
- 至少一个主外部面板的公式簇 CI 下界严格大于 0；
- near、MRR 与风险门不出现实质回退；
- 外部候选池、参考谱聚合和空对照生成均保持同一冻结协议。

## 9. 唯一运行方式

```bash
sbatch --export=ALL,EVIDENCE_DIR=data/validation/noise_msg_fusion_run_<JOB_ID>/evidence \
  tasks/run_conditional_null_energy_stage1_1gpu.sbatch
```

该作业只申请一张 GPU，在同一 allocation 内依次完成契约测试、ChemAware 候选空对照导出、四臂训练和统计裁决；不会继续提交子作业，也不会运行已关闭的 RRF 或未过门的 BioAware。
