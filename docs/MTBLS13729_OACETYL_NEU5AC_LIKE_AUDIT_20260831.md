# MTBLS13729 mono-O-acetyl-Neu5Ac-like 精确质量审计（2026-08-31）

## 一、裁决

现有 negative-HILIC 原始数据在 `m/z 350.109269 [M-H]-` 上存在两个可重复、色谱分离且带
`m/z 87.0088` 碎片的信号，但**没有证据表明它们在 Rmu 中稳定升高，也没有证据表明它们随
free Neu5Ac 的患者内变化同步**。

因此当前数据不支持：

> `free Neu5Ac pool up -> bulk mono-O-acetyl-Neu5Ac pool up`

这个负结果进一步强化 donor–carrier–core–linkage 解耦边界，但它不能排除糖链结合态
O-acetyl-Neu5Ac、未被当前方法稳定保留的异构体，或 NXPE1/CASD1/SIAE 的蛋白/酶活改变。

## 二、为什么不能直接叫 9-O-acetyl-Neu5Ac

`C13H21NO10 [M-H]-` 的理论精确质量为 `350.109269`。4-、7-、8-和9-O-acetyl-Neu5Ac
互为位置异构体，普通精确质量无法区分。O-acetyl 基团还可能迁移或在样本处理和离子源中不稳定。

已有标准品与离子淌度研究都强调，需要合成标准、保留时间、位置敏感碎片或碰撞截面才能定位
O-acetyl 位点：

- quantitative 4-O/9-O-acetyl-Neu5Ac standards：
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9303589/
- isomer-resolving ion-mobility MS：
  https://pubmed.ncbi.nlm.nih.gov/37880209/

因此本文统一使用 `mono-O-acetyl-Neu5Ac-like exact-mass feature`，不使用 Level 1/2 身份。

## 三、分析协议

### 3.1 表型盲色谱发现

- 输入：negative-HILIC 的60个原始 mzML；
- MS1：`m/z 350.109269 ±5 ppm`；
- 每个样本独立基线校正并按99.5百分位归一化，确保每个样本等权；
- 在结果分组完全不可见的总体共识EIC上冻结色谱峰；
- 峰间最小距离18秒；
- 每个峰必须在至少12个样本中有独立局部峰支持；
- 冻结RT后才读取Rmu/Rtu/Ltu表型并进行配对统计。

该协议替代了早期8秒complete-linkage试运行。后者会把RT漂移峰带切成14个相邻簇并产生积分窗
重叠，已判定为不合格开发结果，不进入任何生物学结论。

### 3.2 定量与统计

- 每个冻结RT在每个样本中使用独立9秒半窗和局部峰界积分；
- 主分析只使用肿瘤与正常均检出的完整配对；
- 两侧exact sign-flip检验，以患者为随机化单元；
- 两个表型盲RT峰共同做BH校正；
- half-minimum-detected-area floor仅作缺失敏感性，不替代主分析；
- 另检验Rmu-vs-Rtu interaction，并计算Rmu患者变化与Level-1 free Neu5Ac变化的Spearman相关。

### 3.3 MS2

MS2从同一批样本的HDF5逐谱恢复，要求前体在10 ppm内并落在冻结RT峰的±14秒内。报告
`87.0088/128.0350/170.0459/308.0987/332.0987`在基峰1%以上的出现频率。这些碎片只作
谱学相容性描述，不能定位O-acetyl位置。

## 四、结果

### 4.1 两个表型盲RT峰

| feature | RT | 表型盲峰支持 | RT分层MS2 | m/z 87 | m/z 170 |
|---|---:|---:|---:|---:|---:|
| OAc-like-01 | 257.69 s / 4.29 min | 50/60 | 47谱/19样本 | 47/47 | 12/47 |
| OAc-like-02 | 333.19 s / 5.55 min | 54/60 | 56谱/30样本 | 54/56 | 6/56 |

两个峰前体质量误差中位数约为 `-1.88/-1.74 ppm`。强m/z 87支持它们属于同一个可重复的
含该碎片的化学家族，但m/z 128、308和332均未达到1%基峰门，故不能把它们终证为某个
O-acetyl-Neu5Ac位置异构体。

### 4.2 Rmu配对丰度

| feature | 完整检出配对 | 正向 | mean log2 delta | exact p | BH q | floor敏感性mean |
|---|---:|---:|---:|---:|---:|---:|
| OAc-like-01 | 4/10 | 2/4 | +1.057 | 0.750 | 0.930 | +0.033 |
| OAc-like-02 | 8/10 | 4/8 | +0.176 | 0.930 | 0.930 | +0.314 |

两个峰都没有稳定Rmu主效应。floor敏感性中分别只有4/10和5/10患者为正，仍不支持统一升高。

### 4.3 亚型与free Neu5Ac耦合

- OAc-like-01 的 `(Rmu-RN)-(Rtu-RN)` 为 `+1.057 log2`，BH q=`0.593`；
- OAc-like-02 为 `+2.589 log2`，BH q=`0.441`；
- 与free Neu5Ac患者变化的Spearman rho分别为`0.170`和`-0.067`，均不显著。

因此不能用这两个exact-mass峰解释Rmu中free Neu5Ac的扩张。

## 五、对主模型的影响

### 可增加的结论

> The expanded free Neu5Ac pool was not accompanied by a reproducible increase in either of two
> phenotype-blind mono-O-acetyl-Neu5Ac-like exact-mass features, further arguing against a single
> coordinated sialic-acid pool.

### 仍然禁止

- `9-O-acetyl-Neu5Ac is unchanged in mucinous CRC`；
- `NXPE1/CASD1/SIAE activity is unchanged`；
- `O-acetylated MUC2 glycans are absent`；
- `m/z 350.109 is definitively O-acetyl-Neu5Ac`；
- 从bulk游离峰否定glycan-bound、空间或细胞型特异O-acetylation。

## 六、下一步价值排序

1. 若能获得标准：4-O和9-O-acetyl-Neu5Ac标准共同进样，比较RT、MS2和spike-in；单一标准不足以
   排除另一个位置异构体。
2. 若不能获得标准：优先做linkage-aware O-glycomics或MUC2 glycopeptide，而不是继续把
   `m/z 350.109`当作身份终证。
3. 若有组织材料：NXPE1/CASD1/SIAE蛋白/活性与O-acetyl-sensitive lectin/抗体共定位只能作为
   机制升级，不能由当前bulk RNA替代。

## 七、可复核工件

- `data/mtbls13729/oacetyl_neu5ac_like_v2/report.json`
- `data/mtbls13729/oacetyl_neu5ac_like_v2/frozen_rt_features.csv`
- `data/mtbls13729/oacetyl_neu5ac_like_v2/per_sample_eic.csv.gz`
- `data/mtbls13729/oacetyl_neu5ac_like_v2/rt_resolved_ms2_spectra.csv.gz`
- `data/mtbls13729/oacetyl_neu5ac_like_figure_v1/oacetyl_neu5ac_like_audit.png`
- `tasks/audit_mtbls13729_oacetyl_neu5ac_like.py`
- `tasks/plot_mtbls13729_oacetyl_neu5ac_like.py`

