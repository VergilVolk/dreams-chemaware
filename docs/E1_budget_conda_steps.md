# E1 低预算 Conda 执行步骤

所有命令都应在 DreaMS 项目根目录执行。脚本使用当前激活环境中的 Python，
Windows、Linux 和 SLURM 交互节点都可以使用。

## 0. 激活环境

```powershell
conda activate D:\dreams_env
cd D:\DreaMS
```

如果服务器上的环境有名称而不是路径，可用 `conda activate dreams_env`；Linux 服务器
同时把第二行换成实际项目目录。

## 1. 查看计划但不运行

```powershell
python tasks/run_e1_budget.py --stage all --dry-run
```

## 2. 构建候选池

如果 `data/e1/e1_train_triplet_pool.npz` 和 `e1_val_triplet_pool.npz` 已存在，
可以跳过；重新构建不会消耗 GPU。

```powershell
python tasks/run_e1_budget.py --stage pools
```

## 3. 精简官方 checkpoint（只做一次）

```powershell
python tasks/run_e1_budget.py --stage prepare
```

原始官方文件包含不再需要的 optimizer/trainer 状态。该步骤生成
`data/e1/official_embedding_slim.pt`；Windows 首次转换可能需要数分钟，之后官方
评估和 Pilot B 都读取精简版，不重复支付加载成本。

## 4. 环境预检

```powershell
python tasks/run_e1_budget.py --stage check
```

必须看到 `CUDA available: True`、train/val IK14 overlap 为 0，以及两个候选池
均无 missing keys。若 CUDA 为 False，不要开始正式训练。

## 5. 官方微调模型基线（不训练）

```powershell
python tasks/run_e1_budget.py --stage official
```

输出目录：`data/validation/e1_budget/r0_official/`。
首次读取 1.24 GB 官方 checkpoint 会慢一些。

## 6. Pilot A：raw SSL 起点

```powershell
python tasks/run_e1_budget.py --stage pilot-a
```

默认预算为 1 个 seed、3 epochs、每 epoch 最多 2,000 个 train batch。
训练结束后自动用完整 val fold 评估 best checkpoint。

## 7. Pilot B：官方微调起点继续训练

```powershell
python tasks/run_e1_budget.py --stage pilot-b
```

Pilot B 会同时复用官方 backbone 和 1024→1024 投影头，学习率为 `1e-6`；
并非重新初始化一个随机头。

## 8. 汇总和闸门判断

```powershell
python tasks/run_e1_budget.py --stage summary
```

输出：`data/validation/e1_budget/comparison.json`。自动比较：

- Pilot A vs raw SSL；
- Pilot B vs official FT；
- pooled AUC、macro AUC、separation 三项中至少两项改善才显示 PASS。

## 一次性顺序运行

确认预检无误后，也可以运行：

```powershell
python tasks/run_e1_budget.py --stage all
```

预算更紧时，可将 pilot 缩小到 1,000 batch、2 epoch：

```powershell
python tasks/run_e1_budget.py --stage pilot-a --train-batches 1000 --epochs 2
python tasks/run_e1_budget.py --stage pilot-b --train-batches 1000 --epochs 2
```

不要在 A 和 B 使用不同的 batch/epoch/seed 设置，否则失去公平比较意义。

## 推荐的官方模型继续微调路线（不跑 Pilot A）

预算有限时，可以把科学问题限定为“官方最佳 embedding 在本项目训练集上继续微调后
能否改善”，不再训练 raw SSL 分支：

```powershell
python tasks/run_e1_budget.py --stage official
python tasks/run_e1_budget.py --stage pilot-b --train-batches 500 --epochs 2
python tasks/run_e1_budget.py --stage summary
```

也可以一次执行同一路线：

```powershell
python tasks/run_e1_budget.py --stage lean --train-batches 500 --epochs 2
```

该路线仍然严格使用 MassSpecGym train 训练、val 评估，并以“继续微调前的官方模型”
作为直接对照。它不能回答“相对 raw SSL 的增益由哪一步产生”，因此论文中应称为
official-DreaMS continuation/domain adaptation，而不是从 SSL 起点的完整消融。
