# ST003325 结构隐藏扰动复现结果

日期：2026-10-12  
状态：正式 development/consumed 负结果；按预注册停止

## 一句话裁决

在 ST003325 中，预注册的跨 STS 编辑机制 response signature 未能区分留出编辑 clone 与留出 WT；该数据当前不能作为谱学算法或公共证据层的事件级正控，Noise、P2b、ChemAware 与公共参考不应在这个终点上继续追加。

## 数据、分母、角色与泄漏

- 数据集：Metabolomics Workbench ST003325 / PR001844。
- 角色：`development/consumed`；`sealed=false`；不是独立外测。
- 结构/通路泄漏：无。feature universe 由 pooled-QC、blank、study detection 和 DDA linkage 表型盲构建；训练/测试签名不使用结构名、library hit 或 steroid/STS pathway 列表。
- 独立单位：12 个 clone，CE/KO/PTC/WT 各 3 个。每面板 72 次进样不是 72 个独立生物重复。
- 设计限制：Basal 固定在 plate 1--3，Basal+DHEAS 固定在 plate 4--6，因此终点只能称 design-labelled plate-set response，不能称无混杂 DHEA-S 因果效应。

## 预注册主结果

外层完整留出 CE、KO 或 PTC；内层留出 1 个 WT clone。训练签名来自另外两种编辑机制的 6 个 clone 与剩余 2 个 WT clone。三个留出编辑 clone 分别与留出 WT 比较，合计 27 个原子比较；机会方向为 50%。

`MS1-all` 使用 3,080 个 measurement-qualified features（AN005444 1,029；AN005445 2,051）：

- combined correct：`10/27 = 37.04%`；
- CE：`2/9 = 22.22%`；
- KO：`4/9 = 44.44%`；
- PTC：`4/9 = 44.44%`；
- combined median margin：`-0.03970`；
- 2,000 次保持 12-clone 结构的 clone-label permutation，right-tail `p = 0.7126`；null mean correct `13.761/27`，2.5--97.5% quantile `3--25/27`。

固定的 raw-log1p 敏感性得到同样的 `10/27`，三个编辑类分解也完全相同。因此结果不是逐进样 median centering 导致的符号翻转。

## DDA 选择控制

只使用 804 个 replicated-MS2-linked abundance features、但完全不使用碎裂关系时：

- combined correct：`6/27 = 22.22%`；
- CE/KO/PTC：`1/9、2/9、3/9`；
- combined median margin：`-0.26009`。

这说明仅限制到“具有 pooled-QC MS/MS 的 features”不会改善复现，反而更差。它不评价 WSE、Noise 或 P2b；但根据冻结合同，完整 MS1 response 未过最低正控门后，不允许继续叠加这些模型来寻找阳性。

## 科学解释

本结果否定的是一个具体命题：CE、KO、PTC 三种 STS 编辑在该 plate-set contrast 上共享一个可由全体无偏 features 学习并迁移的稳定方向。它不证明 STS 没有代谢表型，也不证明稀疏的单代谢物效应不存在。可能原因包括编辑机制/clone 异质性、条件与 plate 绑定、真实响应稀疏而全局方向不稳定，或 processed peak intensity 未保留合适的响应尺度。

但这些可能原因不能作为在同一结果上继续调 top-K、选择 pathway 或换相似度的许可。任何稀疏效应、单编辑机制或已知 steroid 分析都必须成为新的、有独立科学问题和外层验证的新合同，且不能覆盖本次失败。

## 对总方法的影响

1. ST003325 从“候选事件级正控”降级为设计受限负结果与数据工程资产。
2. 暂停在该终点上运行统一谱学模型和公共 reference/Atlas 层。
3. 下一事件级数据优先要求：条件不与 plate/batch 完全绑定、多独立扰动或明确 mechanism truth、真实 biological replicates、pooled-QC 或 sample-linked MS/MS，以及可做 gene/source/instrument 外层留出。
4. 公共数据不再是主线前提；只有主 response object 先成立后，才问公共证据是否提供条件增量。

## 机器可读结果

`data/validation/st003325_structure_hidden_ms1_baseline_20261012/report.json`

