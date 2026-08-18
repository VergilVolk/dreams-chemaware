# ChemAware 完整流水线

## 当前执行方式

统一入口：

```powershell
D:\dreams_env\python.exe tasks\run_chemaware_pipeline.py --mode status
```

它会检查每个阶段的真实产物和关键字段，而不是只看文件是否存在。已经完成的阶段会自动跳过，中断后可直接重跑。

## CPU 数据准备与验证

```powershell
D:\dreams_env\python.exe tasks\run_chemaware_pipeline.py --mode cpu --mces-workers 4
```

顺序包括：MCES候选清单 → 可恢复MCES缓存 → MCES排序三元组 → 谱图规则标签 → 冻结化学概念探针 → 结构环境探针 → 双重映射候选 → 模块2峰级证据。

## P1训练结束后的正式评估

将最终选定的P1 checkpoint代入：

```powershell
D:\dreams_env\python.exe tasks\run_chemaware_pipeline.py --mode gpu --device cuda --batch-size 32 --p1-checkpoint D:\path\to\best_checkpoint.pt
```

该命令会完成：

1. P1固定10 ppm检索、困难负样本、遮蔽鲁棒性与结构连续性评估；
2. 在通过P1门槛后，联合身份、MCES局部排序、冻结概念解码器与P1保持约束；
3. 对最终ChemAware权重重复同一套锁定评估；
4. 对最终权重执行`embedding → 化学概念 → 具体谱峰`的支持峰删除/匹配对照因果检验；
5. 写入统一状态报告。

多任务阶段的规则解码器保持冻结。这样，规则损失只能推动embedding朝已验证的化学概念方向移动，无法由解码器单独适配来制造“可解码性提升”。

独立确认集默认保持锁定。模型、epoch和阈值冻结以后，才在正式命令中显式解锁一次。

## 唯一总状态文件

`data/pipeline/chemaware_pipeline_status.json`

其中同时记录：

- 每个阶段是完成、运行中、待GPU还是被上游阻塞；
- MCES监督是否可用；
- 化学概念和结构环境是否可解码；
- 双重映射是否在独立分组中复现；
- 概念—谱峰因果闭环是否成立；
- P1是否已通过正式锁定评估。

## 当前决策规则

- P1没有通过锁定检索与安全性评估前，不接入最终模型。
- MCES只提供局部相对排序，不回归绝对距离，不把阈值截断值当连续标签。
- 规则负责概念解码、证据定位和冲突发现，不再直接定义embedding距离。
- 模块2只把质量匹配称为“候选化学证据”；没有干预证据时，不宣称唯一碎片结构或唯一机理。
