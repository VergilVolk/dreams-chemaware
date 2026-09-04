# E4-A 定向噪声独立贡献三臂因果实验

日期：2026-09-01  
状态：实现完成、等待集群正式运行  
上游裁决：`NOISE_FINETUNING_PPT_TO_IMPLEMENTATION_ROOT_CAUSE_AUDIT_20260901.md` v3

## 1. 唯一问题

当前 E4-A 共享 encoder 在五折三 seed 上平均 Recall@1 `+0.635 pp`、near `+0.522 pp`，但历史训练没有把 R0 已保存的 matched-control paths 送入配对学生。因此本实验只回答：在相同训练预算下，定向 action 是否比等剂量 matched-random action 给共享 clean-spectrum encoder 带来额外、可跨 formula 转移的收益。

本实验不扩大动作空间、不加入 P-arm、不训练 selector、不调整学习率或解冻层数。

## 2. 三个训练臂

三臂都使用 R0 `curriculum` 的九个固定 cell、`action_scope=all`、同一 query/action row 和同一 candidate references：

1. `clean_duplicate`：action view 是 clean query 的精确复制；
2. `matched_random`：从 R0 每条 action 预先冻结的两个 matched controls 中，以 query/selector/dose/step 的 outcome-free SHA256 规则固定选择一个；
3. `targeted`：使用 R0 原始 candidate-gradient/confounder target path。

三臂唯一允许不同的是 action view 的 peak path。hard negative、正负参考谱、identity/query/policy sampler、batch order、训练步数、优化器和评价候选图必须一致。

## 3. 冻结训练配置

- development formula fold：0；
- formula fold seed：20260825；
- training seed：20260828；
- official initialization；
- shared query/reference encoder；
- curriculum、all queries、4 views/identity、4 epochs；
- positive spectra 4、negative molecules 8；
- last Transformer block 1；
- backbone LR `2e-6`、head LR `1e-5`；
- clean/aug rank `1/1`、consistency `0.25`、margin floor `2`、preservation `5`；
- safety ratio/weight `1/1`；
- FP32、grad clip `1.0`；
- P-arm、guided action、outcome mining、P2b、P3 全部禁止。

每个 epoch 写出 action 与 safety sampler SHA256；三个臂不一致时汇总器拒绝比较。

## 4. 评价和裁决

模型推理只输入 clean spectrum。held fold 使用完整 candidate molecule/reference graph 重新排名，并保存：

- strict rank、MRR、near Recall@1；
- full-list positive-vs-best-negative margin；
- corrected、introduced、risk-net；
- top candidate molecule switch 与 wrong-to-different-wrong；
- formula-cluster paired bootstrap CI。

主比较是 `targeted - matched_random`；次比较是 `targeted - clean_duplicate`。只有以下条件全部成立才进入 action-learnability：

1. targeted-vs-random Top-1 formula CI lower bound > 0；
2. targeted-vs-clean Top-1 formula CI lower bound > 0；
3. targeted-vs-random corrected > introduced、risk-net > 0；
4. near 和 MRR 不下降；
5. 三臂输入、初始化、候选、配置和逐 epoch sampler hash 完全一致。

若失败，不调 LR、不加 epoch、不扩 fold；先裁决现有成熟 action 的独立语义贡献不足。若通过，下一阶段才是 clean-input action-learnability audit，而不是直接 multifold。

## 5. 实现与运行

- trainer：`tasks/train_noise_final_e4a_direct_augmentation.py`
- real-ledger preflight：`tasks/preflight_noise_final_e4a_causal_attribution.py`
- unit test：`tasks/test_noise_final_e4a_causal_attribution.py`
- paired summary：`tasks/summarize_noise_final_e4a_causal_attribution.py`
- validator：`tasks/validate_noise_final_e4a_causal_attribution.py`
- batch job：`tasks/run_noise_final_e4a_causal_attribution.sbatch`

集群只需提交一次：

```text
sbatch tasks/run_noise_final_e4a_causal_attribution.sbatch
```

该命令创建三个 GPU array members；每个 member 明确申请 `--gpus=1`。最后完成的 member 使用原子锁自动生成三臂配对汇总。
