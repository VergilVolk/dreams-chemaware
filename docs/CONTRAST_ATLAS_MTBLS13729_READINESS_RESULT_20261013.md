# MTBLS13729 → Contrast Atlas 本地准入结果

日期：2026-10-13  
数据角色：`development/consumed`  
结论：**FAIL-CLOSED；0 个面板获准生成正式 contrast record**

机器报告：`data/validation/contrast_atlas_mtbls13729_readiness_20261013/report.json`

## 分母与结果

| 面板 | 登记样本/患者 | 完整 normal-tumor 配对 | 本地 mzML | 本地 HDF5 | 现有定量表 | MS1-MS2 link | 准入 |
|---|---:|---:|---:|---:|---|---|---|
| neg-RP | 60 / 30 | 30 | 20/60 | 19/60 | 18 features × 60 samples；另有 3,798-feature `discovery_missingness_pilot` | 无统一 link 表 | FAIL |
| pos-RP | 59 / 30 | 29 | 13/59 | 16/59 | 77 features × 59 samples | 5,834 rows、315 features、59 query files | FAIL |

四个本地面板合计虽有 63 mzML 和 72 HDF5，但分散为 neg-HILIC 16/18、neg-RP 20/19、
pos-HILIC 14/19、pos-RP 13/16，不能把总数包装成任一完整面板。

## 为什么已有矩阵仍不合格

1. 18/77-feature EIC 表是小型既定 target 集，不是无偏全 feature universe。
2. neg-RP 3,798-feature 图明确登记为 `discovery_missingness_pilot`、`formal=false`；只可证明
   ion-family pilot 存在，不能替代统一重定量。
3. 当前本地缺失生成该 pilot 的完整 source matrix/target bundle，不能从报告反向恢复可复现
   contrast。
4. pos-RP 虽有 315 个 feature 的 MS1-MS2 links，但定量宇宙、样本分母和 link 覆盖没有同时
   闭合；且 sample manifest 本身缺 1 个肿瘤样本。

## 允许与禁止

允许继续使用既有 MTBLS13729 结果作为 consumed 生物学案例、定向 EIC 或代码接口核查。

禁止：

- 把 63 个本地 mzML 写成完整 60-sample panel 或完整 240-file study；
- 用 18/77 个预选 feature 代表 unbiased unknown chemistry；
- 将 discovery missingness matrix 与 pos-RP link 表跨面板拼接；
- 把 MTBLS13729 当作第二来源正向 Atlas 结果或 sealed validation。

重新准入至少需要同一面板同时具备：完整登记样本、可重放的统一定量矩阵、独立 ion-family
单位、样本可追溯 MS1-MS2 links、检测/QC 边界和冻结效应模型。恢复原始文件或可验证的全量
中间产物后，应重跑同一审计，而不是放宽门槛。
