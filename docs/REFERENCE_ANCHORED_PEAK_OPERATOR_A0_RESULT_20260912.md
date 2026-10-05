# 标准品锚定峰级化学算子图谱 A0 结果

## 裁决

A0 找到了一个值得继续扩大验证的**局部正信号**，但没有通过通用方法门，不能宣称新 Atlas 已成立。

- 在全部 2,707 个可评估的“作者 Level-1 身份 × 生物样本”查询上，冻结的最高 sqrt-cosine 基线 Recall@1 为 `0.7281`。
- 多参考谱身份聚合本身下降 `-1.63 pp`，证明“多放几张标准谱再平均”不是答案。
- 不含 MS1 的谱学峰算子模型达到 `0.7385`，即 `+1.03 pp`，修正 30、引入 2。
- 在 521 个真正存在多个候选 feature 的困难查询上，谱学峰算子从 `0.5854` 提高到 `0.6392`，即 `+5.37 pp`。
- 加入 MS1 强度/队列可检测性后，全体提高 `+1.66 pp`，困难查询提高 `+8.64 pp`，修正 50、引入 5。
- 但是谱学峰算子的身份聚类 bootstrap 95% CI 为 `[0, 0.0227]`；全张量为 `[-0.00149, 0.0360]`。两者均未满足“下界严格大于零”。

## 为什么不能直接庆祝

### 1. 有效困难身份太少

57 个校准身份中只有 15 个出现多个候选 feature；510/521 个困难查询来自正离子 panel。谱学模型实际改变了 6 个身份，其中正收益主要集中在：

- linoleyl carnitine：净修正 22 个样本；
- myristoylcarnitine：净修正 4 个样本。

因此当前结果更像“酰基肉碱系列的峰级证据有用”，还不是跨化学类别的通用算子。

### 2. MS1 增益不能冒充峰级结构解释

仅使用 MS1 与可检测性特征的模型整体下降 `-1.96 pp`，并产生 121 个新增错误；说明这些先验单独并不可靠。它们与谱学证据联合时能帮助两个酰基肉碱身份，是因为正确 feature 在全队列中具有更一致的检出/丰度支持。这属于 cohort-level analyte support，不是结构真值，也不是峰级转化定位。

### 3. 当前“transformation channel”只是代理变量

A0 复用了已经算好的匹配峰比例、top-10 匹配、intensity coverage 和 neutral-loss 相似度。它尚未建立标准品多条件谱的保守峰集合，也没有输出“哪个峰发生了何种质量平移”。所以它验证的是多通道分解的可行性，不是完整的化学转化图。

### 4. 候选生成仍限制上限

正确 feature 至少被一张参考谱选中的查询比例仅为 `0.7610`。当正确 feature 不在候选集合中，后续算子不可能修正。这说明正式版本必须保留每张标准谱在每个样本中的多个 precursor-compatible 候选，而不能像现有缓存一样只保留每张参考谱的单个最佳候选。

## A0 实际证明了什么

可以支持的表述：

> 在身份隔离的外层交叉验证中，将参考谱—样本匹配分解为匹配峰覆盖、top-intensity 峰保留、neutral-loss 与多参考谱支持后，在多候选 feature 子集上比最高 sqrt-cosine 获得 5.37 pp 的初步增益，并显著降低查询级新增错误；但该增益集中于少数酰基肉碱身份，尚未形成跨身份显著性。

不能支持的表述：

- 新图谱已经超过 StructureMASST、DreaMS Atlas 或 GNPS；
- 已发现新的代谢物或结构转化；
- MS1 高频 feature 就是真实身份；
- +5.37 pp 是全体样本或外部队列性能；
- 已经实现峰级结构位点解释。

## 下一阶段必须怎样改变

下一阶段不是继续调这个小排序器，而是补齐真正的算子对象：

1. 从标准品的多碰撞能/多仪器参考谱中构建 conserved-core peak set；
2. 对每个样本保留 top-k precursor-compatible analyte 候选，不再只存每张参考谱的 top-1；
3. 对每个候选输出 retained、shifted、lost 和 gained peak 四类证据；
4. 用隐藏的标准身份验证化学轴归属和峰变化定位；
5. 至少扩展到 100 个多候选身份、四个化学家族，并采用 study/instrument held-out；
6. 只有身份/研究聚类 CI 下界大于零，才进入表型重编程图。

## 可复核工件

- 实现：`tasks/build_reference_anchored_peak_operator_a0.py`
- 单元测试：`tasks/test_reference_anchored_peak_operator_a0.py`
- 结果校验：`tasks/validate_reference_anchored_peak_operator_a0.py`
- 结果目录：`data/mtbls13729/reference_anchored_peak_operator_a0_v2/`
- 候选张量：`operator_candidate_tensor.csv.gz`
- 隐藏真值逐查询结果：`per_query_hidden_truth.csv.gz`
- 汇总：`report.json`
