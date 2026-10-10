# Noise × ChemAware TSV-Merge：GNPS 开发结果与算法裁决

日期：2026-10-10

作业：Slurm `2358006`

状态：**完整执行；GNPS development/consumed；未达到稳定 +3 pp；不得替代 Noise V1；不得作为 Enveda 外测候选。**

## 1. 一句话结论

官方 TSV-Merge 的 DreaMS 适配执行正确，并且在两个固定参数合并构造中一致胜过简单线性平均；但是相对真正需要战胜的 Noise V1，identity-disjoint Recall@1 只增加 `+0.0091 pp`，formula-disjoint 只增加 `+0.2281 pp`，两个公式簇配对区间均跨零，同时 micro-candidate 和 pooled-pairwise AUC/AUPRC 多数下降。TSV 因而只是**合并臂内部赢家**，不是新的共享 encoder 冠军。

本轮没有达到两面板各 `+3 pp` 的目标。停止 TSV scale/rank 扫描，保留 Noise V1 作为统一系统的共享 encoder；TSV checkpoint 仅保留为开发消融。

## 2. 数据集、分母和证据资格

本轮只使用已经被开发流程消费的 GNPS Gold/Silver 10-ppm 双面板：

| 面板 | queries | candidate molecules | directed spectrum pairs | 数据资格 | 泄漏状态 |
|---|---:|---:|---:|---|---|
| identity-disjoint | 10,995 | 87,518 | 175,171 | development / consumed | 已用于多轮模型开发与比较，不是最终外测 |
| formula-disjoint | 5,261 | 24,401 | 47,724 | development / consumed | 已用于多轮模型开发与比较，不是最终外测 |

两个面板共同需要编码 `52,871` 张谱，来源 MGF 共 `329,607` 条记录。全部谱使用现有原生 DreaMS 预处理：100 peaks、precursor intensity `1.1`、FP32、无增强，输出 1,024 维单位向量。

- MassSpecGym：本作业未读取；其状态仍为 `development / consumed / leaked-for-selection`。
- GNPS：用于本轮两个固定合并构造的开发比较；不能称为 external confirmation。
- Enveda-180：本作业未读取，继续保持最终未开封模型外测资格。

## 3. 固定输入与输出

### 3.1 三个共同来源 checkpoint

| 角色 | checkpoint | SHA-256 | 资格 |
|---|---|---|---|
| common base | `data/e1/official_embedding_slim.pt` | `8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245` | official DreaMS 基线 |
| Noise constituent | `data/validation/noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt` | `01ac8c7bfa75d89862e0b1eb2b5ffb297955cffae471fd483ad6c38d13f33beb` | 当前最佳共享 Noise encoder |
| Chem constituent | `run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt` | `a8428329ca1d12bfe735f3f8b848ed020a8db18d0a654d4f62cdbeb006e61135` | ChemAware Phase-A role-2 开发冠军；未独立 role-3 确认 |

ChemAware 原生 checkpoint 只做格式归一化，没有训练：

```text
source kind: official_embedding
source SHA-256: a8428329ca1d12bfe735f3f8b848ed020a8db18d0a654d4f62cdbeb006e61135
normalized slim SHA-256: d8d72db672db277f15a6444ea734fd7722da5cfbed32c4f977316d136bbb448d
```

### 3.2 两个固定构造

令共同起点、Noise 和 ChemAware 参数分别为 \(\theta_0,\theta_N,\theta_C\)，任务更新为：

\[
\Delta_N=\theta_N-\theta_0,\qquad
\Delta_C=\theta_C-\theta_0.
\]

仅构造两个模型，没有 scale、rank、density 或随机种子扫描：

1. **Linear control**

\[
\theta_{linear}=\theta_0+\frac{1}{2}(\Delta_N+\Delta_C).
\]

2. **TSV-Merge**

对每个二维参数分别对 \(\Delta_N\) 与 \(\Delta_C\) 做薄 SVD；两个任务各保留前 `1/2` 奇异分量，按作者实现拼接左右奇异向量和奇异值，再分别正交化左右空间并重建。非二维浮点更新取算术平均，最终固定：

\[
\theta_{TSV}=\theta_0+1.0\times TSV(\Delta_N,\Delta_C).
\]

实际处理了：

- backbone：31 个二维浮点 tensor、37 个非二维浮点 tensor；
- projection head：1 个二维权重、1 个一维 bias；
- 无非浮点 buffer 漂移；
- DreaMS 不含 CLIP 的 `text_projection` 参数名，因此不存在误跳过检索投影头的问题。

官方实现锚定为：

```text
repository: https://github.com/AntoAndGar/task_singular_vectors
commit: a6c97188f7aa0bb20753de4f2939be89dd64f1c3
function: compute_and_sum_svd_mem_reduction
```

本地对作者函数与项目适配函数做了直接数值比较；在 `(7,5)`、`(8,8)`、`(5,9)` 三种矩阵上最大逐元素误差均为 `0.0`。

输出 checkpoint：

| 构造 | SHA-256 |
|---|---|
| Linear | `0d4cd66e27de95058246092c105bb195a2bc266b7cce0e29bdbde120c4ffa1db` |
| TSV scale=1 | `26b31aa270d19ea15739e84ae74641223f5ff7d36ce79370136796662847113e` |

作业目录：

```text
data/validation/noise_chemaware_tsv_gnps_run_2358006
```

## 4. 四个 encoder 的绝对结果

### 4.1 Identity-disjoint：10,995 queries，87,518 candidates，175,171 directed pairs

| 方法 | Recall@1 | MRR | macro-query AUROC | micro-candidate AUROC | micro-candidate AUPRC | pooled-pairwise AUROC | pooled-pairwise AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| official | 85.3570% | 0.905164 | 0.937330 | 0.931927 | 0.681486 | 0.928697 | 0.651907 |
| Noise V1 | 86.5575% | **0.912741** | 0.941290 | **0.943576** | **0.752298** | **0.941279** | **0.733654** |
| Linear | 86.5393% | 0.912416 | **0.942000** | 0.942420 | 0.736530 | 0.940278 | 0.715238 |
| TSV | **86.5666%** | 0.912553 | 0.941543 | 0.943117 | 0.744047 | 0.941272 | 0.723946 |

TSV 获得最高 Recall@1，但只比 Noise 多约一个净正确 query；Noise 仍在 MRR、micro-candidate 和 pooled-pairwise 指标上更强。

### 4.2 Formula-disjoint：5,261 queries，24,401 candidates，47,724 directed pairs

| 方法 | Recall@1 | MRR | macro-query AUROC | micro-candidate AUROC | micro-candidate AUPRC | pooled-pairwise AUROC | pooled-pairwise AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| official | 86.8086% | 0.920242 | 0.931864 | 0.907215 | 0.680116 | 0.903277 | 0.648430 |
| Noise V1 | 88.0631% | 0.928331 | 0.938052 | **0.931447** | **0.769246** | **0.929902** | **0.750432** |
| Linear | 88.2342% | 0.929189 | **0.939358** | 0.926638 | 0.744761 | 0.924026 | 0.720984 |
| TSV | **88.2912%** | **0.929617** | 0.939309 | 0.929165 | 0.756825 | 0.926932 | 0.734519 |

TSV 在 Recall@1/MRR 上最高，但仍低于 Noise 的 micro-candidate 与 pooled-pairwise AUROC/AUPRC。这说明少量 Top-1 重排收益伴随全候选区分能力下降。

## 5. 相对 official 的开发增益

### 5.1 Linear vs official

| 面板 | ΔRecall@1 | formula-cluster CI | corrected / introduced | risk `C−2I` | Δmacro AUROC | CI |
|---|---:|---:|---:|---:|---:|---:|
| identity, n=10,995 | +1.1824 pp | `[+0.5918,+1.7699]` | 260 / 130 | 0 | +0.4671 pp | `[+0.1391,+0.7852]` |
| formula, n=5,261 | +1.4256 pp | `[+0.5602,+2.3323]` | 125 / 50 | +25 | +0.7494 pp | `[+0.1715,+1.3050]` |

### 5.2 TSV vs official

| 面板 | ΔRecall@1 | formula-cluster CI | corrected / introduced | risk `C−2I` | Δmacro AUROC | CI |
|---|---:|---:|---:|---:|---:|---:|
| identity, n=10,995 | +1.2096 pp | `[+0.5689,+1.8182]` | 280 / 147 | −14 | +0.4214 pp | `[+0.0700,+0.7672]` |
| formula, n=5,261 | +1.4826 pp | `[+0.6494,+2.3967]` | 136 / 58 | +20 | +0.7444 pp | `[+0.1660,+1.3291]` |

TSV 对 official 的 Recall@1、MRR、macro AUROC/AUPRC 均有严格正的公式簇配对区间。这证明 TSV 仍是一套有效的 DreaMS 改良表示，但不能回答它是否超过强 Noise V1；后者必须由直接配对比较决定。

TSV 相对 official 的其他 AUC 增益：

| 面板 | micro AUROC | micro AUPRC | pooled AUROC | pooled AUPRC |
|---|---:|---:|---:|---:|
| identity | +1.1190 pp | +6.2561 pp | +1.2575 pp | +7.2039 pp |
| formula | +2.1950 pp | +7.6709 pp | +2.3655 pp | +8.6089 pp |

这些绝对增益很强，但其中大部分已由 Noise V1 实现，不能误写成 ChemAware 与 TSV 新增贡献。

## 6. 决定资格的直接比较：TSV vs Noise V1

### 6.1 Top-1 与配对稳定性

| 面板 | ΔRecall@1 | formula-cluster CI | corrected / introduced | 净纠正 | risk `C−2I` |
|---|---:|---:|---:|---:|---:|
| identity, n=10,995 | **+0.0091 pp** | `[-0.4869,+0.5053]` | 147 / 146 | +1 | −145 |
| formula, n=5,261 | **+0.2281 pp** | `[-0.4260,+0.8954]` | 72 / 60 | +12 | −48 |

两个面板的区间都跨零。Identity 中几乎是 147 个纠正换来 146 个新错误；formula 虽有 12 个净增，但仍伴随 60 个新错误，不能称为稳定互补。

Near-subset 同样没有独立增量支持：

| 面板 | near queries | Δnear Recall@1 | formula-cluster CI |
|---|---:|---:|---:|
| identity | 7,592 | +0.0395 pp | `[-0.6997,+0.7181]` |
| formula | 2,944 | +0.3397 pp | `[-0.7773,+1.4954]` |

### 6.2 AUC：局部 Top-1 改善没有转化为全局区分增益

| 指标 | identity：TSV−Noise | formula：TSV−Noise |
|---|---:|---:|
| MRR | −0.0188 pp | +0.1286 pp |
| macro-query AUROC | +0.0253 pp | +0.1257 pp |
| macro-query AUPRC | −0.0188 pp | +0.1286 pp |
| micro-candidate AUROC | −0.0459 pp | −0.2282 pp |
| micro-candidate AUPRC | −0.8251 pp | −1.2421 pp |
| pooled-pairwise AUROC | −0.0007 pp | −0.2970 pp |
| pooled-pairwise AUPRC | −0.9707 pp | −1.5913 pp |

Macro-query AUROC 的配对区间也跨零：identity `[-0.2361,+0.2886] pp`，formula `[-0.2972,+0.5371] pp`。Micro 和 pooled 指标没有 query-cluster CI，但点估计在两个面板上均显示：TSV 没有保住 Noise 的全局候选区分优势。

## 7. TSV vs Linear：官方子空间处理是否有用

| 面板 | ΔRecall@1 | CI | corrected / introduced | Δmicro AUROC | Δpooled AUROC |
|---|---:|---:|---:|---:|---:|
| identity, n=10,995 | +0.0273 pp | `[-0.3352,+0.3757]` | 74 / 71 | +0.0697 pp | +0.0993 pp |
| formula, n=5,261 | +0.0570 pp | `[-0.4676,+0.5972]` | 36 / 33 | +0.2527 pp | +0.2906 pp |

TSV 在两个面板上均略高于 Linear 的 Recall@1、micro-candidate AUC/AUPRC 和 pooled-pairwise AUC/AUPRC，因此“层内低秩子空间去干扰优于无脑参数平均”的方向与结果一致。但增量极小、Recall@1 区间跨零，无法把方法层面的合理性升级为 DreaMS 上已经确认的显著优势。

## 8. 为什么自动输出 `winner=tsv` 不等于新冠军

本作业预先固定的选择器只在两个新构造 `Linear` 与 `TSV` 之间选择，并按以下字典序比较：

1. identity/formula 两面板中较小的 Recall@1 增益；
2. 若相同，再比较两面板平均增益。

因此输出：

```json
{
  "winner": "tsv",
  "stable_plus_3pp_target_attained": false
}
```

其严格含义仅为：**TSV 是两个固定合并构造中的开发赢家。** 选择器没有把 Noise V1 纳入可淘汰候选，故生成的：

```text
data/validation/noise_chemaware_tsv_gnps_run_2358006/selected_checkpoint.pt
```

不得标记为 production encoder、final encoder 或 Noise V1 replacement。它只是 TSV checkpoint 的 hardlink。

## 9. 目标判定与最终裁决

预定目标是相对 official 在两个 GNPS 开发面板上均达到 `+3 pp` Recall@1。实际 TSV 为：

- identity：`+1.2096 pp`；
- formula：`+1.4826 pp`；
- `stable_plus_3pp_target_attained=false`。

最终判定：

| 问题 | 裁决 |
|---|---|
| 作业是否完整成功 | 是；归一化、合并、四次编码、四组比较、选择均完成 |
| 是否使用作者官方 TSV 核心 | 是；固定作者 commit，数值直接对照误差为 0 |
| TSV 是否优于线性平均 | 点估计一致略优，但增量很小且 Recall CI 跨零 |
| TSV 是否显著优于 official | 是，GNPS 两开发面板 Recall/MRR/macro AUC 配对 CI 均严格为正 |
| TSV 是否显著优于 Noise V1 | 否；Recall 和 macro AUC CI 跨零，全局 AUC 多数下降 |
| 是否达到稳定 +3 pp | 否 |
| 是否继续扫描 scale/rank | 否；不为小点估计展开新调参 |
| 是否允许打开 Enveda-180 测这个 encoder | 否；不消耗最终外测资格 |
| 当前共享 encoder 主线 | 继续使用 Noise V1 |
| TSV 的保留方式 | development ablation / fixed negative-to-small result |

## 10. 对统一算法的直接意义

本轮排除了一个高性价比假设：ChemAware Phase-A 的 MSG 内部收益不能靠无训练 checkpoint 合并稳定迁移为 GNPS 上超过 Noise 的共享表示。TSV 比线性平均更能减少参数干扰，但仍无法把两个历史增益相加，更没有产生指数叠加。

因此统一算法不再继续消耗时间于：

- TSV scale/rank 扫描；
- 更多线性系数组合；
- 把同一实验换名字重复执行；
- 在 MassSpecGym 上重新选择合并模型；
- 用 Enveda 为这一弱增量做事后选择。

下一步恢复最有证据的系统结构：

1. 共享表示固定为 Noise V1；
2. WSE 作为独立全谱证据；
3. P2b 使用既有冻结最佳综合版本，不与其原始通道重复计票；
4. ChemAware V2 作为候选条件化下游残差证据，而不是继续压入单一 embedding；
5. 已失败的 RRF 不复活；
6. 在 GNPS development/consumed 上完成一次统一系统开发选择后，才将唯一固定系统一次性送入 Enveda-180。

这不是否定 ChemAware：Phase-A 在原开发域上的 `+2.1266 pp` 仍是有效的域内开发结果，ChemAware V2 重排器的强开发结果也仍保留。这里被否定的只是“通过无训练参数合并即可把 Phase-A 与 Noise V1 的历史增益稳定相加”这一具体假设。

## 11. 可复现入口

```bash
sbatch tasks/run_noise_chemaware_tsv_gnps_1gpu.sbatch
```

实现与结果解析：

- `tasks/merge_dreams_tsv.py`
- `tasks/run_noise_chemaware_tsv_gnps_1gpu.sbatch`
- `tasks/select_dreams_tsv_gnps.py`
- `tests/test_dreams_tsv_merge.py`
- `tests/test_dreams_tsv_selection.py`
