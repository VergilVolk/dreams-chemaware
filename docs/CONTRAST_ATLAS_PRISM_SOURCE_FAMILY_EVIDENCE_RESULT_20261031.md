# Metabolomic Contrast Atlas：PRISM 候选的源离子家族与公共谱证据

日期：2026-10-31

数据角色：**development / consumed**。本结果不是 Enveda 封存外测，不是外部疾病机制确认，也不是标准品鉴定。

## 结论

这轮完成的不是另一个小型检索任务，而是对全局 Atlas 输出中一个可证伪问题的实证回答：

> 当研究级响应检索由若干 unresolved 化学边支持时，这些边究竟代表可跨研究累积的同一实体、同一分析离子家族，还是偶然共质量/共洗脱？

成人 IBD200、儿童 VEO-IBD 与 ST001000 两参与者队列的开发筛查曾优先化 13 个源实体对。目标队列的质量差和 RT 将其中 7 对初步折为 3 组；冻结源 LC/MS 证据证明三组不能同样处理：

| 质量组 | 源数据裁决 | 独立公共谱证据 | 允许结论 |
|---|---|---|---|
| `595/617` | 成人/儿童源 RT 分别相差 72.10/138.07 s；5% 基峰阈值均无共有碎片 | `MSV000079651` 两质量均有谱，但 raw cosine `0.000824`、2 个匹配峰 | 保留为两个不同源实体假说；ST001000 中共洗脱不能反推同一实体 |
| `817/834/839` | 两源队列共洗脱、同相关组；二聚体 H/NH4/Na 中性质量跨度 0.19/0.37 ppm；诊断谱 cosine `0.940/0.959` | `MSV000079651` 无 10-ppm 谱，属于无覆盖而非反证 | 折为一个分析离子家族；结构仍未确认 |
| `371/389` | 两源队列共洗脱、同相关组；H2O 质量差；诊断谱 cosine `0.960/0.935` | `MSV000079651` 两质量均有谱，raw cosine `0.8289`、13 个匹配峰 | 强支持一个中性分析家族/源内脱水关系；不能据此命名结构 |

因此，7 条相关源边当前只对应 4 个可辩护分析家族单位。其余 6 条暂按独立计算时，13 个源实体对的当前上限是 **10 个分析家族**，不是此前目标队列质量/RT 筛查给出的 9，也不是 13 个已确认代谢物。

## 对 unknown 增量的影响

同一个源证据裁决被应用到完整的 101 个可测试源实体对，而不是只修改 13 个阳性候选：

| 分母 | unresolved | known | mixed | unresolved vs known |
|---|---:|---:|---:|---:|
| 源实体对 | 9/82 | 2/12 | 2/7 | OR `0.616`，单侧 Fisher `p=0.855` |
| 源证据纠正后的家族 | **7/73** | **1/10** | **2/8** | OR `0.955`，单侧 Fisher `p=0.754` |

完整分母由 101 对折为 91 个工作家族；10 个家族含至少一个严格优先成员。当前仍**没有 unresolved 相对 known 富集的证据**。这不等于 unresolved 没有生物学价值；它说明这个开发面板尚未证明条件增量，不能用更多 unresolved 数量替代正确答案。

## 为什么这对 Atlas 方法有用

1. **身份轨得到一个真实反例。** `595/617` 在目标研究中质量兼容且共洗脱，但两个源研究的 RT 与碎片都反对合并。仅靠目标队列加合物关系会把两个不同源响应错误压成一个“复现分子”。
2. **关系轨得到一个可复算正例。** `371/389` 在两源队列和独立公共粪便谱资产中均显示强碎片关系；它适合支持化学响应模式，但两条 beta/SE 仍不能被当成同一 observed ion 直接合并。
3. **家族折叠必须使用双端证据。** 研究级响应关系不能只在目标端判断冗余；源端与目标端都要有 LC/MS 证据，且结构身份、离子家族和一般化学关系必须分层输出。

这三点是 Atlas 全局算子的资格规则，不把课题缩成一个 IBD 峰案例。大型主体仍是 15 个来源、51 个研究级 contrast 和完整开放候选轴；本轮提供的是其中关系轨首次得到外部谱证据支持、身份轨同时出现可证伪反例的实测结果。

## 证据边界

- ST001000 的四种公开方法是高分辨率 full-scan MS1；没有可用于本轮的样本链接 DDA/DIA MS/MS。因此它只能提供质量、RT 和丰度响应，不能独立确认源谱身份。[官方研究页](https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST001000)；[原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6342642/)
- `MSV000079651` 是独立公共粪便背景谱资产；本轮目标质量不在公开的 145 个加标标准清单中。因此它提供独立 MS/MS 关系证据，不是 authentic-standard 或 IBD 响应确认。[MGF](https://gnps.ucsd.edu/ProteoSAFe/DownloadResultFile?file=f.MSV000079651%2Fclustered%2FMSV000079651_specs_ms.mgf&forceDownload=true)；[标准清单](https://gnps.ucsd.edu/ProteoSAFe/DownloadResultFile?file=f.MSV000079651%2Fmethods%2F145_standards_fecal_BK.csv&forceDownload=true)
- 13 个优先源实体对中，当前共同 Gold/Silver 政策只允许 2 对作条件性 observed-ion 合并；`0/13` 达到 MSI level 1。
- 响应方向的总体置换门仍未通过：检验层面 `p=0.0522`，源实体对层面 `p=0.0856`。本轮化学核验不能反过来把总体方向预测写成成功。

## 下一项决定性实验

不再围绕这 13 对继续调参。下一项全局实验保持固定：在 291,330 条模型无关身份候选轴上，分别让 WSE、Official、Noise 和统一算法形成自己的选择/弃权，再用同一 51-contrast 响应算子比较 retained、gained、lost、abstained 及独立支持。`371/389` 作为关系轨正控，`595/617` 作为禁止错误合并的反例；两者不能替代全局结果。

## 可复算产物

- `tasks/adjudicate_contrast_atlas_prism_source_ion_families.py`
- `data/validation/contrast_atlas_prism_source_ion_family_adjudication_v1_20261030/`
- `tasks/analyze_contrast_atlas_prism_family_level_increment.py`
- `data/validation/contrast_atlas_prism_family_level_increment_v2_source_adjudicated_20261030/`
- `tasks/validate_contrast_atlas_prism_public_mgf_family_evidence.py`
- `data/validation/contrast_atlas_prism_public_mgf_family_evidence_v1_20261030/`

三份脚本通过 `py_compile`；各输出目录含输入/脚本 SHA256、结果报告和输出校验和。
