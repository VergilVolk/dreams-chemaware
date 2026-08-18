# Causal ChemMask GPU 运行与决策规程

## 0. 先在 CPU 环境验证

CPU 预检和最小端到端运行：

```bat
cd /d D:\DreaMS
python tasks\run_causal_chemmask.py --stage cpu-check
python tasks\run_causal_chemmask.py --stage cpu-smoke --cpu-threads 8
```

`cpu-smoke` 只运行 1 个 epoch、8 个训练 batch 和 4 个验证 batch，用于确认模型加载、前向、反向、保存 checkpoint 全部跑通。它不能用来判断算法是否有效。

最小运行通过后，可执行较大的 CPU 方向性实验：

```bat
python tasks\run_causal_chemmask.py --stage cpu-pilot --cpu-threads 8
```

默认 CPU pilot 为 2 个 epoch，每个 epoch 100 个训练 batch、25 个验证 batch；它仍然只是方向性结果，不替代 GPU 全量实验。

## 1. 这次训练回答什么

只回答一个问题：在保持 DreaMS 官方检索能力的前提下，针对已经由独立峰删除实验确认的错误机制进行训练，能否同时改善质量近邻中的身份区分和 embedding—结构相似度关系。

训练标签仍然来自真实分子身份（IK14）；峰级证据只用于困难负样本选择和同分子谱图增强，不作为结构标签。规则重叠不定义正负样本。

## 2. 固定数据

- train：112,601 个严格 10 ppm anchors；5,125,411 条同身份正边；4,828,326 条不同身份负边。
- validation：21,163 个 anchors；924,136 条正边；772,886 条负边。
- discovery 结构面板：57,566 对、3,776 个分子，用于方法开发。
- confirmation 结构面板：15,652 对、1,196 个分子，只用于独立确认，不进入训练。
- test：本阶段不读取。

## 3. 运行顺序

在带 CUDA 的 `dreams_env` 中执行：

```bat
cd /d D:\DreaMS
python tasks\run_causal_chemmask.py --stage check
python tasks\run_causal_chemmask.py --stage pilot --seed 20260815 --workers 4
```

若显存不足，将 batch size 降为 8：

```bat
python tasks\run_causal_chemmask.py --stage pilot --seed 20260815 --workers 4 --batch-size 8
```

pilot 会依次完成：

1. 训练 3 个 epoch，每个 epoch 1,000 个 batch；
2. 保存每个 epoch 的轻量 projection head；
3. 在完全相同的 10 ppm 检索协议上计算 AUC、Top-1；
4. 编码一次冻结主干 precursor token；
5. 只在 discovery 的固定 57,566 对面板上比较每个 epoch 的 Pearson、Spearman 和同分子稳定性；pilot 不读取 confirmation。

## 4. pilot 的判定

官方权重在 discovery 固定面板上的参照值用于 pilot 选择；confirmation 数值保持封存，模型锁定后才读取。

| 范围 | Pearson r | Spearman rho |
|---|---:|---:|
| 不同分子 | 0.5747 | 0.5527 |

仅当至少一个 epoch 同时满足以下条件，才进入正式训练：

1. 10 ppm AUC/Top-1 没有明显下降；
2. 不同分子 Pearson 或 Spearman 至少一项改善，另一项不出现明显下降；
3. 同分子式局部相关性不下降；
4. 同分子重复谱图稳定性不下降；
5. 改善不只出现在训练 triplet accuracy。

若只见 loss/accuracy 改善，而 AUC、Pearson 或同分子稳定性下降，立即停止，不扩大训练。

## 5. 正式训练

pilot 通过后，先固定所选超参数，再运行：

```bat
python tasks\run_causal_chemmask.py --stage formal --seed 20260815 --workers 4
```

之后再补两个随机种子。三个种子的方向一致，才形成正式结论。模型和 epoch 锁定后，才执行一次：

```bat
python tasks\run_causal_chemmask.py --stage confirm --checkpoint D:\DreaMS\data\e1\causal_chemmask_formal\seed_20260815\epoch_XX_causal_head.pt
```

confirmation 只报告最终锁定模型，不用于反复挑选超参数。

## 6. 关键输出

- 训练历史：`data/e1/causal_chemmask_pilot/seed_20260815/history.json`
- 每个 epoch 的 head：同目录下 `epoch_XX_causal_head.pt`
- 检索结果：`data/validation/causal_chemmask_pilot_seed_20260815/`
- 官方权重与候选 head 的结构对比：`data/validation/causal_chemmask_pilot_seed_20260815_structure/official_vs_causal_structure.json`
- 各 epoch 固定面板结果：上述目录下 `epoch_head_comparison/epoch_head_metrics.csv`
