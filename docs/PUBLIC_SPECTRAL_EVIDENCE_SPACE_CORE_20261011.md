# 公共谱学证据空间：可选证据层与最小实现

日期：2026-10-11  
状态：核心方法假设与可运行证书引擎；尚无阳性生物学结论

## 核心对象

公共空间不定义为 embedding kNN 图，而定义为一组 **public spectral evidence units**：每个单元必须由多个公共来源中的谱图、至少两个预先冻结的证据族共同支持，并保留来源集中度、仪器覆盖、冲突和 dustbin。谱学邻近只产生有限分辨率的 evidence unit，不自动升级为结构、反应或功能。

新研究中的 feature `i` 得到带弃权的单位概率/证书 `c(i,u)`；无标签样本的表示为

`X(s,u) = sum_i log1p(abundance_si) * reliability_si * c(i,u)`。

同一个冻结对象支持三种入口：单谱查询返回证书；无标签样本形成单位占用与异质性；有设计研究在同一单位上估计方向和不确定性。表型从不参与单位构建。

## 与现有方法的硬边界

- DreaMS Atlas 已提供 2.01 亿谱图、每节点三近邻及元数据查询；本项目不能把“更大 kNN”当创新。
- MEMO 已能把 MS2 peak/loss presence 向量化用于大样本比较；本项目必须证明公共、跨来源证书相对研究内向量有增量。
- MS2LDA 2.0 已能发现和注释重复碎裂模式；motif 只能作为一个 evidence family，不能把 motif overlap 写成身份。
- reference-data-driven analysis 已证明元数据标注的公共源数据能扩大谱图利用；本项目必须显式控制公共来源与仪器偏差，并用 matched-shuffle 公共层证明增量。

## 四世界决定性消融

对同一冻结研究同时计算：研究内 MS1/MEMO；研究内谱学但无公共层；真实公共证据单元；来源/仪器/质量/峰数匹配但 unit 置乱的公共层。主结果必须是“真实公共层减去前三者最强值”，并按 biological subject 或 source 聚类给 CI。只恢复仪器、TIC、MS2 数量或 batch 即失败。

## 已实现

`tasks/build_public_spectral_evidence_certificates.py` 已实现 label-blind、structure-truth-blind 的长表接口、多证据族/多来源门、来源支配审计、冲突 tie 和显式 `__NO_MATCH__`。它是证书核心，不训练 Noise/ChemAware/P2b，也不读取疾病或药物标签。

`tasks/project_public_evidence_certificates_to_samples.py` 已实现证书到样本级 unit occupancy 的表型盲投影。它同时输出 certified coverage 与 dustbin fraction；归一化的 unit composition 只在已认证信号内部成立，不能掩盖低覆盖。两项合同测试均已通过。

`tasks/shuffle_public_evidence_certificates.py` 生成匹配伪公共层：在 polarity、mass、峰数和 instrument 等预冻结技术分层内置换已认证的 public-unit 标签，同时逐 query 保持是否弃权和 certificate score。它直接否证“只要加一个稀疏公共标签就会改善”的解释；正式比较使用全部 null repeats，而不挑最弱的一次。

## 在总方法中的资格

公共层不是总方法的必要条件。科学主线是结构隐藏的化学响应能否在独立生物学单位、编辑机制、研究和仪器间复现；公共参考只在它对这种复现提供超过研究内 MS1、研究内谱学和 matched shuffle 的条件增量时保留。不得为了利用 Atlas 而改变生物学问题、候选总体或终点。

## 压力测试而非主实验

LCNEC 可作为 negative-mode、低公共覆盖压力测试，但不再是下一主实验。先隐藏 tumor/adjacent 标签构建单位占用；只有覆盖门通过时才评价技术重复/同患者可恢复性与 batch/instrument 混杂，再打开配对标签。LCNEC 是 development/consumed；其作用是暴露失败域，不负责证明通用性。

具体冻结顺序、四世界、`1/67` mate-retrieval 机会水平、34-patient 聚类推断与停止门见 `LCNEC_PUBLIC_EVIDENCE_SPACE_M0_CONTRACT_20261011.md`。当前完整 85 x 263 EIC matrix 仅有 `.filepart`，真实运行仍为 `INPUT_NOT_EVALUABLE`。

停止规则：公共层不优于 MEMO/研究内谱学，增量由 source dominance 解释，或 unresolved 加入 known 后无条件增量，则从生产生物学方法中删除；最多保留为证据查询工具。公共层失败不自动否定不依赖公共参考的事件级结构隐藏方法。
