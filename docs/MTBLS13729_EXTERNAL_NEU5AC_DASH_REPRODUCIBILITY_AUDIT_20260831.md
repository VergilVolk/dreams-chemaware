# MTBLS13729 外部 Neu5Ac 网页数据复现性审计（2026-08-31）

## 结论

Jain 等人的公开 CRC metabolome 网页确实返回了 `N-Acetylneuraminic acid` 在七个结直肠
亚部位的肿瘤和正常组织绘图值，但这些值**不能当作论文372对患者的分析级数据重新统计**。
网页结果只保留为方向性可视化背景；外部队列的正式统计仍以同行评议论文及其补充表为准。

## 冻结方法

- 数据端点：`https://colorectal-cancer-metabolome.com/_dash-update-component`
- 固定输入：`compound-dropdown-linear.value = N-Acetylneuraminic acid`
- 固定输出：tumour/normal linear-plot figures
- 原始 HTTP 响应、请求、整洁表、脚本和 SHA256 均已冻结。
- 脚本：`tasks/freeze_external_crc_neu5ac_dash.py`
- 产物：`data/external/CRC_metabolic_biogeography_PMC11438248_20260831/neu5ac_dash_patient_level_v1/`

## 样本量审计

网页对每一种组织均返回：

- 7个亚部位；
- 每个亚部位恰好53个值；
- 总计371个值；
- 没有 patient ID、pair ID 或其他配对键；
- 每种组织的371个值均唯一，不是简单的重复填充。

这与来源文件存在两层不一致：

1. 论文和方法报告372对肿瘤/正常组织，而网页为每种组织371个值；
2. 补充人口学表中的亚部位样本量明显不相等，stage cells分别为63、45、28、32、76、38、92，
   且总和为374；网页却固定为每组53。

因此，网页可能经过未公开的平衡、重排、截取或图形层转换。没有映射表时，不能把
`within_trace_index`解释为患者，也不能把肿瘤和正常的同位置值配成一对。

## 回归复算

按照方法中规定的 `cecum=0 ... rectum=6` 对网页值做简单线性回归：

| 组织 | 网页 raw slope/step | 网页标准化 beta（Pearson r） | 网页 p | 补充表 slope | 补充表 p |
|---|---:|---:|---:|---:|---:|
| normal | +0.07528 | +0.39144 | 4.92e-15 | +0.349 | <0.001 |
| tumour | +0.03694 | +0.17897 | 5.33e-4 | +0.088 | 0.091 |

网页值不论按 raw slope 还是标准化 beta，都不能精确复现补充表。尤其是 tumour 的网页
回归为显著，而补充表报告不显著，二者不能混用。

描述性方向仍一致：

- normal 的 rectum-minus-cecum 均值为 +0.35886；
- tumour 为 +0.19086；
- 两套网页分布都向远端结直肠升高，且 normal 的梯度更强。

这支持“疾病状态改变 Neu5Ac 的解剖梯度”这一宽泛背景，但不增加一个独立、精确复现的
统计结果。

## 对主论文的影响

保留：

- 外部队列用分析标准确认 Neu5Ac 为 Level 1；
- 正式补充表报告 normal slope +0.349、p<0.001，tumour slope +0.088、p=0.091；
- 该结果说明位置是 Neu5Ac 分析的重要协变量。

禁止：

- 把网页371个值称为完整372对患者级源数据；
- 将肿瘤与正常数组按位置配对；
- 用网页复算的 p 值替换论文补充统计；
- 把该队列称为黏液型 CRC 的独立丰度复现；
- 从空间梯度推断通量、酶活性或因果机制。

## 裁决

这次审计没有削弱 MTBLS13729 的 Neu5Ac 主轴，但降低了公开网页可承担的证据等级。
外部队列继续作为 **Level-1 身份和疾病相关空间背景**；患者级精确复算与独立黏液型复现
仍属于缺失证据。

