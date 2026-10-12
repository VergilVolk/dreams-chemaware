# MSV000096214 结构隐藏扰动实验预注册合同

日期：2026-10-12
状态：**冻结**。本合同在任何效应结果（T1–T4）打开之前写就并提交。特征表构建（下载、定标、检测、对齐、补空、审计）只允许使用表型盲信息；任何阶段不得读取基因型列。

> **修订 v1.1（2026-10-12，定标运行之前）**：首批 QC mzXML 落地后立即测得 MS1 点强度量级 p50=68 / p90=156 / p99=912（SJ-Mutant-Flavo-CQ2-NEG，16.4M 点）。§3.2 原网格 {2e3…5e5} 沿用自另一仪器的绝对强度尺度，与本数据不符；机械执行将导致 G1 必然失败——这是尺度假设错误，不是数据失败。修正**仅限网格数值**：改为 {50, 100, 200, 400, 800, 1600, 3200}。选择规则（最小 T 使 3 个 QC 特征数中位数 ≥ 2,000）、目标数、定标输入（仅 pooled-QC）、全部其他条款不变。修订发生时无任何 study 文件被检测、无基因型被读取。RT 单位已确认秒制（梯度约 2,100 s），§3.3 trace 参数不变。

> **修订 v1.2（2026-10-12，表审计运行之前）**：§3.5 G4 原文"每基因型×极性缺失率"与 §1"表审计脚本不得读取 genotype 列"矛盾。修正为样本级：**gap-filling 后每 study 样本缺失率 ≤ 50%（每极性超限样本 ≤ 2 个）**。基因型级缺失率在效应阶段按 tube join 后作描述性报告，不作门。此修正比原意更严（样本级能抓出个体坏样本），且不违反表型盲原则。

> **修订 v1.3（2026-10-12，任何效应结果打开之前）**：独立代码审计发现原 T2 把同一 15 个 tt7 样本在 5 个 WT 轮换折中的重复计分当成 75 个独立判定，并把总计 90 个相关判定代入二项区间；这属于伪重复。另发现实现先生成 `log2` 矩阵、随后在 signature/score 中再次 `log2`，会把负的中心化值变成 NaN。现冻结如下修正：(i) 每个生物样本跨适用 WT folds 聚合后只计一次；(ii) 主推断改为 6 个 single-mutant genotype 中选 3 个伪 target 的全部 `C(6,3)=20` 种簇分配穷举，最小可达 `p=0.05`；(iii) 每极性主预处理为正强度 `log2` 后 sample-median centering，raw log2 为 sensitivity；(iv) 删除一切二次 log；(v) feature bootstrap 只称为条件于本表的稳定性区间，不当作独立生物学推断。修订时只有 raw 下载和 pooled-QC 噪声定标在运行，study feature table、T1--T4 与 genotype effect 均未生成或打开。

> **修订 v1.4（2026-10-12，任何 study feature/effect 生成之前）**：首次 pooled-QC 定标暴露出 §3.2 选择方向错误。feature count 随 `noise_threshold_int` 上升而非增加；原“满足中位 feature count ≥2,000 的最小阈值”必然优先选择最宽松阈值。实际阈值 50 在 3 个正离子 QC 产生 141,468--153,791 features，在 3 个负离子 QC 产生 133,225--133,729 features，显然不是仅保留信息下限的严格定标。已停止尚未产出任何 per-file feature 的 study extraction，归档该 calibration；规则更正为：从 3,200 向下扫描冻结网格，选择仍使 3 个 QC feature count 中位数 ≥2,000 的**最高阈值**。目标、网格、QC-only 输入和检测算法不变；genotype/effect 从未读取。

## 0. 一句话

在拟南芥类黄酮突变体 LC-MS/MS 数据上，检验**无偏、结构无关的 MS1 特征表**能否在不用任何结构注释的前提下复现已知酶级联通论的结构（上位塌缩、等位基因一致性、分支分离），并只在最后一步打开 MS2 做生化一致性检查。

## 1. 数据、角色与泄漏

- 数据集：MassIVE `MSV000096214`（拟南芥 WT 与 tt3/tt4/tt7/fls1 系列突变体，正负双极性 DDA LC-MS/MS）。
- 角色：`development/consumed`；`sealed=false`；不是独立外测，不产生最终主张。
- 入库清单（冻结）：`data/validation/msv000096214_readiness_v2/report.json`，`sample_manifest.csv.gz`（含每文件 bytes 与 `.scans` SHA-256）。
- 98 个 study mzXML + 6 个 pooled-QC mzXML（3 个独立 QC 池 × 2 极性），tube 33（fls1_3_EMS）两个极性缺失，fls1 仅 4 个重复。
- 泄漏控制：下载、定标、检测、对齐、补空、表审计的所有脚本**不得读取** genotype 列；genotype 映射仅在效应阶段（T1–T4）由独立脚本按 tube 号 join。

## 2. 已知混杂与反混杂设计

**基因型与进样顺序整块绑定**：tube 1–5=WT、6–10=tt7_115、11–15=tt7_8、16–20=tt7_9、21–25=tt4_11、26–30=tt3、31–35=fls1_3、36–40=tt7_8×tt4_11、41–45=tt7_9×tt4_11、46–50=tt7_115×tt4_11。这与 ST003325 的 plate 混杂同构，是本数据最大的有效性风险。

**反混杂结构**（本设计独有，ST003325 没有）：

1. **三条独立 tt7 等位基因**（EMS、SALK_039417、SALK_seq_064306）位于三个分离的进样位置（6–10、11–15、16–20）。若三者对 WT 的效应签名同向且互相高相关，单调漂移无法同时满足。
2. **上位塌缩**：tt4 是 CHS（查尔酮合酶）null，类黄酮全通路上游阻断；三个 tt7×tt4 双突位于序列**最末**（36–50），tt4 单突位于**中部**（21–25）。若双突签名塌缩到 tt4 而非 tt7，则相隔 20+ 进样位的两个块表现一致而中间块（26–35 的 tt3/fls1）不塌缩，漂移无法解释。
3. tt3（DFR，花青素分支）与 fls1（FLS，黄酮醇合酶）位于 tt4 与双突之间，作为位置诱饵：漂移会把 36–50 拉向 26–35 的近邻，上位预测则拉向 21–25 的 tt4。

## 3. 表型盲特征表构建（参数冻结）

### 3.1 下载

- 端点：MassIVE `DownloadResultFile?file=f.MSV000096214/peak/<filename>`。
- 每文件校验 bytes 与冻结 manifest 一致；sha256 落盘记录；断点续传（.part 临时文件）。
- 目录：`data/external/msv000096214/mzxml/{positive,negative}/`。

### 3.2 噪声阈值定标（基因型盲：只用 pooled-QC）

- 每极性取其 3 个 pooled-QC 文件，只读 MS1，收集全部正强度点。
- 候选网格（v1.1 尺度修正后）：`T ∈ {50, 100, 200, 400, 800, 1600, 3200}`，其余检测参数按 §3.3 冻结。
- 选择规则：按 `3200→1600→800→400→200→100→50` 扫描，选择**最高的 T**使 3 个 QC 文件的检测特征数中位数 ≥ 2,000；若无一达到，取中位数特征数最大者。
- 定标结果连同输入 sha256 写入 `calibration.json`，之后不可更改。

### 3.3 检测（OpenMS 冻结配方，与 MTBLS13729 管线同源）

- 只读 MS1；`MassTraceDetection(mass_error_ppm=5.0, noise_threshold_int=T, min_trace_length=5.0s, max_trace_length=120.0s, trace_termination_criterion=outlier, trace_termination_outliers=5)`
- `ElutionPeakDetection(width_filtering=auto)`
- `FeatureFindingMetabo(isotope_filtering_model=none, mz_scoring_13C=true, remove_single_traces=false, report_summed_ints=false)`
- 输出 per-file：mz、rt_sec、intensity、charge、quality、width_sec；同时抽取 MS2 precursor 表（precursor mz、RT、charge、TIC）作为 DDA 联结层。
- 逐文件 checkpoint，失败文件记录不中断。

### 3.4 一致性对齐（每极性独立）

- 参照文件：该极性 3 个 QC 文件中特征数中位者（表型盲）。
- RT 校正：非参照文件对参照做 m/z 种子匹配（5 ppm、初始 RT 窗 ±60 s），Theil-Sen 线性校正。
- 分组：全文件特征按强度降序贪心锚定，组内容差 5 ppm + 10 s（校正后 RT）。
- 补空：对组内缺失文件，取 ±5 ppm、组中位 RT ±15 s 的原始 EIC 做 apex 积分；低于该文件噪声点集 Q90 则记缺失（**缺失不是零**）。

### 3.5 特征表审计门（打开基因型标签之前）

- G1：每极性一致性特征 ≥ 2,000。
- G2：study 文件原始检测特征数 < 1,000 的文件每极性 ≤ 3 个。
- G3：3 个 QC 文件两两间特征 log2 强度中位差 ≤ 1（描述性；不足为硬门时须报告）。
- G4：gap-filling 后每 study sample 缺失率 ≤ 50%，每极性超限样本 ≤ 2 个；效应阶段 join 后另报告超限样本是否集中于同一 genotype。
- G5：代码审查记录：构建阶段无脚本读 genotype（合同即记录）。
- 任一硬门（G1/G2/G4）失败 → **停止**，不叠加任何模型层，按失败报告。

## 4. 预注册效应检验（打开基因型后）

特征合格标准：在 ≥4/5 该基因型重复与 ≥4/5 WT 重复中非缺失（同极性）；fls1 用 ≥3/4。
丰度：主分析为正强度 log2 后在每个 sample、每个 polarity 内减去 feature 中位数；raw log2 为固定 sensitivity。S(g) = 各合格特征上基因型中位中心化 log2FC(g vs WT) 的向量。任何一步只允许一次 log 变换。

- **T1 上位塌缩**（主）：对每个双突 D：Spearman ρ(S(D), S(tt4)) > Spearman ρ(S(D), S(三条 tt7 均值签名))。预注册预测：3/3 成立。显著性：特征标签置换（n=2,000）。
- **T2 留出等位基因恢复与路径特异性**（对照 ST003325 死因）：完整留出一条 target genotype，训练签名 w = 另两条 target genotype vs WT 的中位中心化 log2FC，**全部合格特征、无 top-K/显著性筛选**。WT 样本使用 leave-one-WT-sample-out signature；其他 genotype 从不进入自己的计分模型。分数为 Spearman(x_s − 训练 WT 中心, w)。主统计量是每个 held target 相对 WT 和 3 个 complement single mutants 的 12 个 genotype-balanced pairwise AUROC 的最小值。对 6 个 single-mutant genotype 中任取 3 个为伪 target 的全部 `C(6,3)=20` 种分配重复完整流程；真实 tt7 集合必须唯一第一，精确右尾 `p=1/20=0.05`。三条 held tt7 相对 WT 的 AUROC 还必须各自 >0.5，且相对 tt3/tt4/fls1 的 9 个均值 margin 全部 >0。样本级分数跨适用 WT folds 聚合后，每个真实生物样本只计一次；不再报告伪独立的 90 次二项区间。
- **T3 拓扑分离**：(i) min_D [ρ(D, tt4) − ρ(D, 配对 tt7)] > 0；(ii) 三 tt7 互相中位 ρ > 中位 ρ(tt7, tt3) 且 > 中位 ρ(tt7, fls1)。树状图仅描述性。
- **T4 生化一致性（最后打开，只做一次）**：
  - 类黄酮预测集 F：log2FC(tt4 vs WT) ≤ −1 且 BH-FDR<0.05（Mann-Whitney 5v5），且 ≥2/3 双突 log2FC ≤ −0.5。
  - 对照集 C：所有基因型 |log2FC| ≤ 0.25 且 FDR>0.5，按 m/z ±50 Da 匹配。
  - 打开 MS2（DDA 联结：precursor 5 ppm、apex RT ±20 s）：F vs C 的糖苷中性丢失（162.0528、146.0579、308.1107、176.0321）携带率 Fisher 精确检验（双侧）；m/z 400–900 Da 富集。
  - tt7 分支证据（不用谱库）：在 tt7 下降集（log2FC ≤ −1、FDR<0.05、∉F）中，统计存在 +15.9949 Da 伴侣（5 ppm 内、且该伴侣在 tt7 中 log2FC ≥ 0）的特征比例，与 C 中同统计比较。
- 正负极性**全程独立**，分别报告，不做合并主张。

## 5. 停止规则

- 表审计硬门失败 → 停。
- T2 未满足三条 held-vs-WT AUROC >0.5、9 个路径特异性 margin >0、且真实集合在 20 种簇分配中唯一第一 → **封存该对比**；禁止在同表上换相似度/预处理/top-K 补救（ST003325 纪律）。
- T1–T4 任何结果打开后，禁止回改阈值、检测参数或合格标准。

## 6. 与定位宣言的关系

本实验是宣言"事件级扰动复现"线的第一个合格候选：多独立扰动（3 等位基因 × 2 通路位置 + 上位对）、真实生物重复、pooled-QC、可外层留出。成功 → 结构无关效应对接公共证据证书层（四世界消融）；失败 → 记入停止门有效性证据，不影响主线其余部分。

## 7. 机器可读锚点

- 审计报告：`data/validation/msv000096214_readiness_v2/report.json`（gates 全过）
- 本合同提交后，后续所有脚本与结果以本文件版本为唯一参照。
