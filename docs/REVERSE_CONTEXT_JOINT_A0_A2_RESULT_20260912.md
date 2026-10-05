# 反向语境联合建模 A0-A2 首轮结果

> **2026-09-12 撤回声明：本结果不再作为生物学应用主线或方法创新证据。**  
> A0–A2 使用公共库预计算谱匹配和 Morgan/ClassyFire 特征，没有在 MTBLS13729 的逐个真实样本中执行标准/参考谱反向检索，没有构建“参考谱×样本”检测与丰度矩阵，也没有比较官方 DreaMS、改进 shared embedding 和 P2b。它只保留为“公共元数据/可检测性偏差负对照”。
>
> 替代路线见 `REVERSE_STANDARD_ANCHORED_BIOLOGY_RESET_20260912.md`。

**日期：2026-09-12**  
**结论：数据基础成立；结构语境信号存在；首版 M1 未超过强技术基线，不进入 DreaMS embedding 扩展。**

## 1. A0 与 A0b/A0c

初始 64-anchor 字典序抽样没有通过可行性门：23 个 anchor 有严格公共命中、17 个跨项目、10 个跨生物样本类型。审计显示字典序抽中了大量罕见复杂天然产物，因此该失败结果保留，不作为公共数据不足的最终结论。

保持 cosine >= 0.85、matching peaks >= 4 和全部通过门不变，使用固定 seed 20260912 的伪随机类内抽样：

- A0b：256 anchors；88 个有命中、64 个跨项目、45 个跨生物样本类型，A0 门全过；
- A0c：487 anchors（七类各 64，核苷类全部 39）；173 个有命中、112 个跨项目、76 个跨样本类型；
- 公共机会分母：2,427 项目、412,159 文件、662,147,232 条报告 MS2；
- 33 个 anchor 在 blank/QC 中有命中，证明技术污染/背景不能忽略。

## 2. A1 双轴冻结

固定 SHA-256 seed 拆分，不按结果重抽：

- 1,956 development projects / 471 test projects；
- 392 development identities / 95 test identities；
- project holdout：541 条生物正观察；
- identity holdout：226 条；
- joint holdout：60 条，覆盖 17 个有正观察身份。

HealthStatus 在公共元数据中大面积缺失，因此第一版只建模样本大类，不做疾病预测。

## 3. A2 四模型对照

- B0：公共库暴露量与全局观察率；
- B1：离子模式、加合物家族、前体质量段的技术可检测性；
- M0：ClassyFire superclass 语境分布；
- M1：Morgan fingerprint 邻域迁移的结构—语境模型。

### 主要结果

在 project holdout：

- B1 project-weighted MRR = 0.3213；M1 = 0.3204，基本持平；
- M1 相对 M0 的 identity-macro MRR 差 = +0.0306，95% bootstrap CI `[+0.0066,+0.0602]`；
- M1 的观察过程 Poisson NLL = 0.7181，略差于 B1 的 0.7076。

在 identity holdout：

- M1 project-weighted MRR = 0.3061，高于 B1 的 0.2931；
- 但 identity-macro 差为 -0.0254，CI 跨 0；
- M1 相对 M0 的 identity-macro 差 = +0.0256，CI `[+0.0025,+0.0482]`。

在 joint holdout：

- M1 project-weighted MRR = 0.3732，高于 B1 的 0.3130 和 M0 的 0.3522；
- 只有 17 个有正观察身份，M1-B1 identity-macro CI 很宽并跨 0，不足以下稳定结论。

## 4. 严格裁决

1. **成立**：结构邻域相对粗化学大类有可迁移语境信号；这不是单纯把化合物分成八类。
2. **未成立**：该信号在未见项目上没有稳定超过技术可检测性基线 B1。
3. **不得声称**：疾病关联、代谢机制、真实生物学缺失、代谢物身份确认、优于 DreaMS 注释。
4. **当前停线**：不得直接上更大的 DreaMS/图网络模型；否则是在用容量掩盖可辨识性问题。

## 5. 下一步唯一合理方向

下一代目标不是预测原始命中，而是预测经技术可检测性校正后的残差：

`chemical-context residual = observed public occurrence - expected detectability(B1)`。

在 development projects 内部做嵌套选择，冻结后再进入新的外部项目集合。必须比较：

- technical-only B1；
- superclass residual M0-R；
- structure-fingerprint residual M1-R；
- frozen official DreaMS embedding residual M2-R；
- 改进 embedding residual M3-R（只有 M2-R 先显示可辨识信号后才允许）。

只有 M2-R/M3-R 在新项目与新 IK14 双留出上改善校准似然和语境排序，且 blank/QC 风险不升，才能构成“算法把标准谱转化为可泛化生物语境证据”的方法学结果。

## 产物

- `tasks/preflight_reverse_context_joint_model.py`
- `tasks/build_reverse_context_joint_a1_manifest.py`
- `tasks/pilot_reverse_context_joint_a2.py`
- `data/validation/reverse_context_joint_a0*`
- `data/validation/reverse_context_joint_a1c`
- `data/validation/reverse_context_joint_a2/report.json`
