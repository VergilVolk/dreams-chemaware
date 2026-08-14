# ChemAware反事实峰微调：Anaconda Prompt运行说明

这套训练从 `official_embedding_slim.pt` 开始，不从原始SSL权重重训。训练/内部验证按分子式隔离；确认集和测试集均不参与训练及阶段选择。

## 1. 打开Anaconda Prompt

```bat
conda activate dreams_env
cd /d D:\DreaMS
```

先确认当前环境真的有CUDA：

```bat
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

若显示 `False`，不要启动正式训练。当前轻薄本的CPU环境只能做预检；应将同一仓库和数据放到CUDA服务器上运行。

## 2. 正式预检

```bat
python tasks\run_counterfactual_training.py --stage check
```

必须看到：

- CUDA available: True
- Formula overlap: 0
- invalid HDF5 rows: 0
- Preflight passed

## 3. 推荐：自动三阶段激进训练

8–12 GB显存先使用：

```bat
python tasks\run_counterfactual_training.py --stage aggressive --batch-size 2 --grad-accum 8 --eval-batch-size 16 --epochs 8 --workers 0
```

16 GB及以上显存可使用：

```bat
python tasks\run_counterfactual_training.py --stage aggressive --batch-size 4 --grad-accum 4 --eval-batch-size 32 --epochs 10 --workers 0
```

流程依次执行：

1. 只训练官方embedding投影头；
2. 继承最佳头权重，解冻最后1层Transformer；
3. 继承上一阶段，解冻最后2层Transformer；
4. 每阶段在100个隔离分子式的完整±10 ppm候选池重新计算Top-1、MRR、困难负样本ROC-AUC和bootstrap区间；
5. 表征保持低于0.98，或完整检索退化时自动停止，不继续扩大解冻范围。

若显存溢出，将 `--batch-size` 改为1，将 `--grad-accum` 改为16。不要降低 `--n-highest-peaks`，否则训练协议会改变。

## 4. 分阶段手动运行

只训练投影头：

```bat
python tasks\run_counterfactual_training.py --stage head --batch-size 8 --grad-accum 2 --epochs 8
```

头阶段通过后解冻最后1层：

```bat
python tasks\run_counterfactual_training.py --stage last1 --batch-size 2 --grad-accum 8 --epochs 8
```

最后1层通过后解冻最后2层：

```bat
python tasks\run_counterfactual_training.py --stage last2 --batch-size 2 --grad-accum 8 --epochs 8
```

## 5. 输出位置

最佳权重：

```text
data\e1\counterfactual_formal\head\seed_20260813\best_counterfactual.pt
data\e1\counterfactual_formal\last1\seed_20260813\best_counterfactual.pt
data\e1\counterfactual_formal\last2\seed_20260813\best_counterfactual.pt
```

完整候选池评价：

```text
data\validation\counterfactual_formal\<stage>\seed_20260813\report.json
```

训练日志：

```text
data\e1\counterfactual_formal\<stage>\seed_20260813\history.json
```

## 6. 当前损失

```text
L = L_identity
  + 0.7 * (L_remove-identity + L_remove-confounder) / 2
  + 5.0 * L_preserve
  + 0.2 * L_random-mask-consistency
```

- `L_identity`：真实候选得分高于同分子式误候选；
- `L_remove-identity`：删除真实候选独有峰后，正负分差应下降；
- `L_remove-confounder`：删除误候选独有峰后，正负分差应上升；
- `L_preserve`：保护官方DreaMS已有表征；
- `L_random-mask-consistency`：随机遮蔽约20%非目标峰后保持语义稳定。

规则库只用于解释峰模式，不参与正负标签定义。

## 7. 暂时不要做的事

- 不运行 `--stage all`；它会解冻全部116M参数，只有最后两层稳定获益后才有讨论价值。
- 不改用随机负样本。
- 不使用确认集反复调参。
- 不提前打开测试集。测试集只在架构、权重和停止门槛全部冻结后使用一次。
