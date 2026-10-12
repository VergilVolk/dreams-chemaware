# MSV000096214 结构隐藏 feature--MS2 linkage 合同

日期：2026-10-12  
状态：在扫描 pooled-QC 谱峰、计算任何谱学相似性或打开 annotation strata 前冻结；`development/consumed`、`sealed=false`

## 科学目的

已经通过的 MS1 结果只证明作者处理表中存在强、可复现的遗传响应。下一步不是寻找更多差异峰，而是建立完全不读结构注释的 feature--MS2 映射，检验谱学关系能否在相同 49-sample abundance graph 上增加遗传反事实信息。

Atlas、公共谱库和跨研究检索均不是本步前提。若研究内部的谱学关系没有超过 MS1-only、m/z/RT 和 raw cosine，公共证据层不启动。

## 输入与泄漏边界

- 数据集：MassIVE `MSV000096214`；3 个 positive pooled-QC、3 个 negative pooled-QC，共 66,958 MS2 scans；只用 pooled-QC 构建谱学证据。
- feature anchor：Table S1 的 920 个 feature，仅使用 ID、ion mode、m/z、RT；49-sample abundance 只在后续冻结效应任务使用。
- 禁止读取：annotation、metabolite family、annotation level、作者 molecular-network cluster、作者差异 feature 集及 Figure 5 cluster。
- Table S1 是作者 MZmine 处理并按 apigenin internal standard 与 seed weight 归一化的 feature universe；它不是我们从 raw data 无偏发现的全 feature universe。除非完成 genotype-blind raw reprocessing，不得把结果称为端到端 unbiased feature discovery。

## 冻结映射规则

1. 按 ion polarity 匹配；precursor tolerance `10 ppm`，retention-time half-window `0.10 min`（6 s）。同时报告 5 ppm/3 s 与 20 ppm/12 s 的纯覆盖敏感性，但不据其结果改主映射。
2. 每个 MS2 scan 对阈值内 feature 计算 `sqrt((ppm/10)^2 + (delta_rt_sec/6)^2)`；主链接取距离最小者。若同一扫描有多个阈值内 feature，保留完整 ambiguity ledger，不把它复制成多个独立谱学样本。
3. 谱峰仅做结构非依赖清洗：有限值、正 intensity、至少 5 peaks；记录 collision energy、TIC、precursor、RT、文件与 scan number。不得按 genotype effect 或 annotation 选择代表谱。
4. 同一 feature 可有多张 pooled-QC/碰撞能谱；全部保留为 replicate spectra。后续 embedding 先逐谱计算，再在 feature 内取中位表示。raw cosine 也先做跨 replicate 汇总，不能挑对生物学结论最有利的一张谱。
5. 规范化 peak-list hash 相同者只形成一个 spectral unit；共享谱或多 feature 冲突不得重复计票。
6. feature linkage 的 QC 只依据 ppm、RT、peak count、replicate 文件覆盖、谱间一致性和 acquisition 属性；不得读取 49-sample genotype effect。

## 进入谱学增量的门

- 主规则下至少覆盖 300/920 features，且正负离子各至少 100；
- 至少 70% 已覆盖 feature 在 3 个同极性 pooled-QC 中有两个或以上文件支持；
- 5 ppm/3 s、10 ppm/6 s、20 ppm/12 s 的覆盖单调且主规则不是仅靠极宽窗口获得；
- ambiguity、共享 peak hash、collision-energy 和 polarity 分布完整报告；
- 任一失败只说明该数据的 feature--MS2 linkage 不适合正式增量实验，不允许按 genotype effect 调窗口救结果。

通过后才构建相同 feature universe 的 `MS1-MS2-observed`、m/z/RT、raw cosine、WSE、official DreaMS、Noise V1、P2b 与合格 ChemAware 比较。任何谱学增量必须在三条 held allele、三个 alternative mutants 与三条 double-mutant lineage 上同时成立。
