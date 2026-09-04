# Noise final direct：制品盘点与下载边界

日期：2026-09-04  
状态：本地只读盘点；服务器存在性尚未在本轮验证

本清单服务于 `NOISE_FINAL_DYNAMIC_CONDITIONAL_DIRECT_FINETUNING_CONTRACT_20260904.md`。路径来自当前仓库脚本的实际默认参数和已完成作业记录；下载前仍须在服务器只读确认存在、大小与 SHA。服务器项目根在既有日志中为 `/data02/run01/scv7tsl/DreaMS`，下列均为相对此根的路径。

## 1. 本地已经存在

- `data/models/MassSpecGym_MurckoHist_split.hdf5`
- `data/e1/official_embedding_slim.pt`
- `dreams/models/pretrained/ssl_model_server.pt`
- `data/validation/g8r_noise_final_e4a_direct/`
  - 其中 mature multifold 的 `seed_20260830/fold_0..4/decision.json` 与 `final_shared_encoder.pt` 本地均存在；`held_per_query.csv.gz` 不存在，不能假定应当存在。
- `data/validation/g8r_noise_final_e8_direct_transfer/`

## 2. 本地缺失、Phase A 前需要从服务器核实并取得

### 2.1 冻结任务边界

- `data/validation/g8r_error_atlas_listwise_cache.npz`
- `data/validation/g8r_p2_official_embeddings.npz`

### 2.2 label-free contextual token cache

- 目录：`data/validation/g8r_noise_final_f1_full_tokens/`
- 最少文件：`report.json`、`rows.npy`、`tokens_f16.npy`、`mz_f32.npy`、`intensity_f32.npy`、`valid.npy`

本地的 `data/validation/official_peak_tokens/` 不能未经 SHA/schema 对齐就冒充该目录。

### 2.3 成熟 N action 与 clean-visible OOF

- `data/validation/g8r_noise_final_r0_faithful_s3a/`
  - `report.json`
  - `training_actions.csv.gz`
  - `outcome_audit_only.csv.gz`
  - `cell_fidelity.csv`
- `data/validation/g8r_noise_final_e4a_causal_attribution/`
  - L0/L1 provenance 指向的 clean-duplicate checkpoint、decision 和配对制品；必须整目录核实，不能只拿一个 checkpoint。
- `data/validation/g8r_noise_final_l0_action_learnability_ledger/`
  - `report.json`
  - `action_labels.csv.gz`
  - `cell_summary.csv`
- `data/validation/g8r_noise_final_l1_clean_action_learnability/`
  - `report.json`
  - `action_oof_predictions.csv.gz`
  - `primary_per_query.csv.gz`
  - `primary_false_positive_audit.csv.gz`
  - `clean_query_features.npz`

### 2.4 P recipe 的历史证据与方向对照

Phase A 账本直接重放以下两个已经完成的全图固定矩阵；它们提供每个 query 的真实同身份参考和 wrong-direction control，缺一不可：

- `data/validation/g8r_noise_final_positive_guided_matrix/`
  - `report.json`、`action_manifest.csv.gz`、`matrix_results.h5`
- `data/validation/g8r_noise_final_positive_peak_transfer/`
  - `report.json`、`action_manifest.csv.gz`、`matrix_results.h5`

下列更晚制品用于复核 relaxed recurrence/reference-diversity 的历史容量；它们不替代上述两个全图 recipe manifest：

- `data/validation/g8r_noise_final_e9_action_staleness/`
  - `report.json`、`per_action.csv.gz`
- `data/validation/g8r_noise_final_e10b_positive_action_expansion/`
  - `report.json`、`cell_summary.csv`、`matrix.npz`、`oracle_per_query.csv.gz`
- `data/validation/g8r_noise_final_e11_reference_diversity/`
  - `report.json`、`cell_summary.csv`、`matrix.npz`、`oracle_per_query.csv.gz`
- `data/validation/g8r_noise_final_e12b_relaxed_recurrence/`
  - `report.json`、`cell_summary.csv`、`matrix.npz`、`oracle_per_query.csv.gz`

这些目录的 held outcomes 只用于复核历史容量与 provenance。新训练只复用 recipe/reference-selection/control 实现；P actions 必须在 outer-train 重新物化并 formula-crossfit。

### 2.5 首轮初始化的唯一合法路径

- `data/validation/g8r_noise_final_e4a_causal_attribution/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e4a_causal_v1_20260901_causal_clean_duplicate/seed_20260828/fold_0/`
  - `decision.json`
  - `final_shared_encoder.pt`

不得以 high-LR multifold 的其他 checkpoint 代替。原因不是它们一定更差，而是 L0/L1 的动作标签和 clean-visible 预测在上述 clean-duplicate geometry 上定义；混用会造成跨 geometry 标签错配。

## 3. 首轮不需要下载

- CPG0 的大型逐候选 residual 中间缓存；
- E14/E15 的 outcome-mined selected-action 训练目录；
- B0/B1/B2 的全梯度缓存；
- P2b、ChemAware 或 P3 结果目录。

它们可以保留作失败审计，但不是 direct Phase A 的输入，避免继续为错误主线支付 I/O 与训练成本。

## 4. 下载完成后的唯一下一步

先生成 machine-readable artifact inventory，逐项记录：绝对路径、是否存在、文件大小、SHA256、schema、shape、formula fold、checkpoint provenance 和来源 status。任何一项不一致时只报告缺口，不加载 117M 模型、不提交正式训练。
