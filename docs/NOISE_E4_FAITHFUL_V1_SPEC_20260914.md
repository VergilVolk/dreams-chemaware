# Noise E4 Faithful V1：历史 E4 原法逐字节复原合同

日期：2026-09-14  
状态：实现完成，独立反方审计 PASS，可提交服务器执行

## 1. 这次只做什么

本任务只恢复已经取得稳定正增益的历史 E4 直接微调。它不开发新动作、不使用 Injector、不蒸馏、不从 E8 或任何后续 checkpoint warm start，也不把后来的多来源动作或 optimizer-boundary 机制混入训练。

历史训练入口已经从 2026-08-27 的会话补丁链重放恢复，且命中历史结果 `decision.json` 中记录的唯一验收哈希：

- 训练器 SHA-256：`28c3b375d270fc2030783938d9390710c2c5ab8f0926a0b4ff507d62afa26885`
- 训练器按 Git blob 算法得到的内容 SHA-1：`89307d572c02f4409bcd669e088659717a72f85c`

该 SHA-1 是字节内容身份，不表示这些恢复文件已经作为 Git object 提交到当前仓库。validator、test、core 与 historical layer 均应表述为“从会话补丁链恢复并字节锁定”，不能冒充仓库提交 provenance。

此前暂存的 `987e550d...` 是后来加入 causal、materialized 等接口后的版本，不能称为产出历史 E4 结果的原始程序；本合同已经将它从执行路径移除。

同时逐字节封存：

- 历史 validator：`95b0e6a5...`
- 历史 test：`c20e5a9e...`
- 历史 `noise_v3_core.py`：`a7dd0b07...`
- 2026-08-24 的 `dreams/models/dreams/layers.py`：`ab5e4497...`
- `noise_final_core.py`、`train_e1_identity.py`、`train_noise_final_r2_shared_encoder.py` 已核实与历史会话最终字节完全一致。

SBATCH 在独立 source snapshot 中把这些封存文件恢复成历史 import 文件名，不覆盖工作树中供后续实验使用的新文件。

## 2. 冻结输入及其验收哈希

- 历史候选图：`data/validation/g8r_error_atlas_listwise_cache.npz`  
  SHA-256 `5f2340751c7521c5a93114e2b134d5796f157148736ad9162d545b84c11d9f71`
- 历史 outcome-free R0 报告：`data/validation/g8r_noise_final_r0_faithful_s3a/report.json`  
  SHA-256 `4f68bfd950d44d02664f67ae9d7ac6700fd3ea33ae0f914f6e623146c57749b6`
- 历史 R0 动作表：`data/validation/g8r_noise_final_r0_faithful_s3a/training_actions.csv.gz`  
  SHA-256 `35d52f13e4441141d622c3d60208e3ae7f11dc554bfc129fef2baa4a1cd27843`
- 历史 official embedding cache：`data/validation/g8r_p2_official_embeddings.npz`  
  SHA-256 `86b7e60194b059b839d266e9e6b52a84b185e5b7160aa934531d4c1c026a5486`
- MassSpecGym HDF5：`data/models/MassSpecGym_MurckoHist_split.hdf5`  
  SHA-256 `ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f`
- 官方初始化 checkpoint：`data/e1/official_embedding_slim.pt`  
  SHA-256 `8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245`
- 架构 checkpoint：`dreams/models/pretrained/ssl_model_server.pt`  
  SHA-256 `9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2`
- corrected graph：SHA-256 `8a57bb3a9cccdf69a738f2b093bfad8f6fc4393b23f1342c7a035ac54ce58fa1`
- corrected official cache：SHA-256 `18d7632adc67dcda5d650a5c0bc344d8db3878f9e1ff3f8d51a6d4918b96bbda`
- corrected metadata：SHA-256 `504ce0a570ac1bc461e76cad81acb3ca3560e58b008ec2d8ff14036e944599f5`

任何已知哈希不一致都会在构造模型之前 fail closed；不会先训练数小时再发现输入漂移。

## 3. 原始 E4 动作信息

只使用 R0 中预先冻结、无 outcome 字段的九个 curriculum 单元：

- `candidate_gradient`，attenuation `0.50`，step `3/4/5/6`
- `role_confounder`，attenuation `1.00`，step `1/2/3/4/5`

完整 R0 为 36,934 行。fold 0 训练侧为 28,509 行、1,562 个 identity、689 个 formula。每行在训练前复核 query row、IK14、formula、hard negative、target path；动作表禁止 `corrected`、`introduced`、target outcome 等后验字段。

这里没有 matched-random 或 clean-duplicate 训练臂。原因不是删除控制，而是产出历史结果的 `28c3...` 原始训练器本身尚无 `--causal-arm` 接口；把后来接口加回去便不再是逐字节原法复现。历史 causal attribution 作为既有外部证据保留，不混入这次复原。

## 4. 原始 E4 训练机制

- `policy=curriculum`
- `action_scope=all`
- fold `0`，formula-fold seed `20260825`
- 每个 identity、每个 epoch 恰好 4 个 action view；训练 4 epochs
- batch-actions 4，positive spectra 4，negative molecules 8
- clean、action、positive、negative 全部由同一个可训练 shared encoder 编码
- clean-rank `1.0`，action-rank `1.0`，consistency `0.25`
- official margin floor `2.0`，clean embedding preservation `5.0`
- safety ratio `1.0`
- 仅解冻最后一个 Transformer block 和官方 projection head
- backbone LR `2e-6`，head LR `1e-5`，weight decay `1e-4`
- full FP32，合并 action 与 safety loss 后执行 global gradient clip `1.0`，再使用普通 AdamW
- 训练结束才评估 query-formula-held-out fold；指标不选择 checkpoint、不调动作、不改变 epoch

这是历史 E4 的固有限制：它只按 query formula 划分 action/query；positive 与 negative reference spectra 没有额外执行 formula/spectrum 隔离。因此不能把这里称为“参考端也完全 formula-held-out”。

明确禁止：Injector V1、restored AdamW、PCGrad、materialized actions、multi-action union、teacher embedding、teacher margin、P2b/P3、warm start、动态权重及每步固定百分比注入。

## 5. 两张 GPU 如何使用

SBATCH 恰好申请两张 GPU，且没有手工内存参数：

- GPU 0：预先固定的 primary replay，seed `20260830`
- GPU 1：独立 replica replay，seed `20260829`

两个 seed 都是历史 E4 已实际执行过的配置。primary 身份在 corrected graph 评估前固定；replica 只检验稳定性，不构成从 held 结果中挑最好 checkpoint 的候选池。

## 6. 训练后外部评估

原始训练完成后，独立只读评估器在 corrected MassSpecGym graph 上计算：

- Recall@1/2/3/5/10/20
- MRR、mean rank、median rank
- macro-query AUROC/AUPRC
- micro-candidate AUROC/AUPRC
- positive-vs-best-negative margin
- Top1-Top2 raw/signed gap
- corrected、introduced、risk-net(lambda=2)
- near subset 全套指标
- formula-cluster paired CI
- all-adduct 与 `[M+H]+` 10-ppm pooled pairwise AUROC/AUPRC

corrected graph 不进入训练、梯度、采样或 checkpoint 选择。这里的 pooled AUROC 明确不是 NIST20 `0.85` 的精确复现。

该 corrected graph 的登记角色是 train-side development graph，不是 P3 test；所以本次完整指标只能形成 development 证据，不能形成最终 held test claim。

## 7. 已知事实和裁决边界

历史 seed `20260830` 在原 E4 graph 上的实际结果是 Recall@1 `+0.5740 pp`、38 corrected / 4 introduced、formula-cluster CI 下界 `+0.2192 pp`。历史 E4 自身的 `decision.json` 明确把 `3.85 pp` 写成 action-oracle headroom；后来 best-action union 出现过约 `5.33 pp` 的 training-geometry headroom。两者都不是已经训练得到的 shared embedding 增益，不能写成“E4 已提升 5 pp”。

因此本次首先回答“原始有利 E4 信号能否不经暗改重新进入 shared encoder”。只有服务器新结果出来后，才能依据 corrected graph 的完整指标报告实际增益；代码不会预写或保证 4--5 pp。

“逐字节复原”只限定为恢复源码和上述登记资产；CUDA、PyTorch、驱动与硬件运行栈没有被冻结，因此不承诺 checkpoint 或浮点结果逐 bit 相同，只检验科学与训练合同一致。

正式入口：

```bash
sbatch tasks/run_noise_e4_faithful_v1_2gpu.sbatch
```

该命令只能在服务器上提交；所有 Python 检查、训练和评估均在 Slurm allocation 内执行，禁止在登录节点直接运行测试或训练脚本。
