# ChemAware GNPS 统一基准预注册协议(2026-10-03)

状态:**预注册。两个作业(`run_GLM_chemaware_gnps_unified_benchmark`、`run_GLM_chemaware_gnps_public_model_benchmark`)的结果目录在本文档定稿时尚不存在;端点与判决语言先于任何结果开箱固定。**

## 0. 与既有协议的关系

本协议与 Noise 线 `GLM_GNPS_ARTICLE_BENCHMARK_PREREGISTRATION_20261003.md` 共享同一冻结基准、同一基线、同一种子;方法分层沿用其 S 系记号。ChemAware 侧的增量端点编号为 CA-x,不重复对方 E-x。两线冲突时以各自预注册文本为准,不得互相改写。

## 1. 冻结资产表

| 资产 | 身份 |
|---|---|
| GNPS Gold/Silver 10ppm benchmark v1 | `checksums.sha256` 全表;关键指纹:panel_identity `3d53b8a1…`,panel_formula `efbd24ca…`,manifest `ab1d0441…`,spectra `b813c9e8…`;运行时另经 `gnps_pair_score_cache.benchmark_fingerprint` 逐字节校验 |
| official / Noise V1 embedding 缓存 | run_2347472(`gnps_absolute_baselines_run_2347472/gnps/*.npz`),编码身份以作业内 SHA256SUMS 落盘为准 |
| chemaware_stage1 | `run_2340721_resume_2340524/training/best.ckpt`(已发布 role-3 确认 +1.8144);编码身份:run_2347471 电池已测 slim SHA `dd4bf63e…` |
| chemaware_phaseA | `run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt`(受保护开发冠军 +2.1266,从未 role-3);电池已测 slim SHA `b4871646…` |
| official 基线编码 | 电池已测 SHA `8928f908…` |
| S4 公开模型 | Spec2Vec `spec2vec_gnps_2019`:Zenodo 10.5281/zenodo.3978054,三文件 md5(a733…/cd77…/0962…);MS2DeepScore `ms2deepscore_dual_2024`:Zenodo 10.5281/zenodo.13897744 v2,两文件 md5(d5cb…/5d60…) |
| 包版本钉死(2026-10-03 平台实测修正) | 节点为 gcc 4.8.5/glibc 2.17，仅使用已核实存在的 cp311 manylinux2014 wheel。采用同代且依赖区间可同时满足的隔离栈：numpy 1.26.4 / scipy 1.12.0 / pandas 2.2.3 / scikit-learn 1.5.2 / matplotlib 3.7.2 / numba 0.59.1 / llvmlite 0.42.0 / matchms 0.27.0 / sparsestack 0.5.0 / spec2vec 0.8.0 / gensim 4.3.3 / smart_open 7.1.0 / ms2deepscore 2.4.0。SciPy 1.13 删除 `scipy.linalg.triu`，与 gensim 4.3.3 的实际导入路径冲突，因此不能使用。MS2DeepScore 2.4.0 与公开模型（2024-10-07）同代；两种公开模型均先对52,871张注册谱图各编码一次，再按冻结边计算余弦，禁止逐边重复前向。 |
| 评估协议 | `evaluate_noise_gnps_article_benchmark.py`,bootstrap 10,000,seed 20261003,公式簇 24 假设族,负并列对真值不利 |

## 2. 方法分层(本协议覆盖的增量)

- **S2-DreaMS 家族扩展**:`chemaware_stage1`、`chemaware_phaseA`(经 SHA 锁定扩展层进入;零选择、零调参)。
- **S4 公开神经基线(实用榜)**:`spec2vec_gnps_2019`、`ms2deepscore_dual_2024`。两者公开权重的训练语料均含 GNPS 家族数据,**只进现成实用榜,不进任何严格 OOD 结论**。
- S1/S1b/S2 基础配方 = Noise 冻结配方的逐字节重建(同代码同输入同旗标);S3 冻结重排器**不在本作业、永不与 encoder 层混排**。

## 3. 预指定端点

### CA-1 跨管线一致性闸(实现事故熔断)

本链经 bundle pair-score 路径重推的绝对 Recall@1 必须复现已测值(舍入差内),已测值来源 = 电池作业 run_2347471(embedding 评估器路径):

| 方法 | identity R@1 | formula R@1 |
|---|---|---|
| official | 0.8535698 | 0.8680859 |
| chemaware_stage1 | 0.8565712 | 0.8701768 |
| chemaware_phaseA | 0.8558436 | 0.8720776 |

任一偏离超过舍入差 → **全表作废,先查实现,不谈结果**。合并出图层(`plot_GLM_chemaware_gnps_unified_benchmark.py`)对跨作业同名方法的绝对值冲突同样 fail-closed——该闸在代码里执行。

### CA-2 主端点(判决语言先写死)

对 `chemaware_stage1` 与 `chemaware_phaseA` 各自 vs official,双面板 R@1 配对差 + 公式簇 CI:

- (a) 任一面板 CI 下界 > 0:判"该检查点在分布外 GNPS 存在正迁移"。**与既有证据(§6.2:run_2347471 八个 CI 全跨零)冲突时,必须先按 CA-1 查实现;两条独立路径一致才允许升级声明。**
- (b) CI 跨零(先验预期):判"ChemAware 微调线的 Recall@1 增益为训练分布内资产,分布外迁移为零/未证明",定位句照此写入论文;不得使用"接近显著"字样。
- (c) CI 上界 < 0:判负迁移,写入限制表。

### CA-3 S4 实用榜

只报绝对名次与 vs official 差;不写 OOD 结论;S4 未落榜前禁止"优于全部近年方法"句式(与 Noise 线红线一致)。

## 4. 声明红线(沿袭 + ChemAware 侧新增)

1. GNPS AUROC 禁止与 NIST20 0.85 跨集比较;
2. S3 重排器与 encoder 永不混排一张表;
3. pp 不得跨基线/面板/信息层相加;
4. 化学归因声明只允许来自配对对照证据,本作业不产生任何"化学导致增益"的因果句;
5. `chemaware_phaseA` 保持"开发面板冠军、未获 role-3 确认"身份标签,任何表格不得省略该限定。

## 5. 交付与修订

- 合并图与汇总表:`tasks/run_GLM_chemaware_gnps_merged_figures.sbatch`(集群登录节点禁跑程序,收口同样走 sbatch;三个已落地评估目录经 `EVAL_NOISE/EVAL_CHEMAWARE/EVAL_PUBLIC` 环境变量注入,缺一即 fail-closed;作业内执行 `plot_GLM_chemaware_gnps_unified_benchmark.py`,CA-1 闸在代码里强制)。
- 结果开箱后按 CA-2 模板填数;**无预注册修正案不得新增端点**;任何偏离按本文档语言记录,不得回改本文。
