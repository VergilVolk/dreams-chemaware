"""
化学感知注意力模块 (Chemical-Aware Attention Module) — 模块一核心

本包实现了"数据驱动 + 化学规则引导"的双轨注意力机制：
  - 模块 A（数据驱动）：HeadGatingNetwork — 动态注意力头门控，让模型自适应调节各头权重
  - 模块 B（知识引导）：ChemicalRuleEngine — 化学碎裂规则编码为注意力偏置矩阵，注入 Transformer

组件清单：
  chem_rules.py          — ChemicalRuleEngine：化学规则引擎（中性丢失/特征离子/质量亏损 → 偏置矩阵）
  gating.py              — HeadGatingNetwork：注意力头动态门控网络
  chem_aware_dreams.py   — ChemAwareDreaMS：继承 DreaMS，注入 A+B 协同机制的主模型
  losses.py              — 协同/对抗损失函数（当前阶段：注意力熵正则化 + 对抗损失桩）

对比开关：
  ChemAwareDreaMS.chem_attn_enabled = False → 行为与原版 DreaMS 完全一致
  ChemAwareDreaMS.chem_attn_enabled = True  → 启用化学感知注意力（chem_bias + gate_weights）
"""

from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine, NEUTRAL_LOSSES
from dreams.models.chem_aware.gating import HeadGatingNetwork
from dreams.models.chem_aware.chem_aware_dreams import ChemAwareDreaMS
from dreams.models.chem_aware.losses import attention_entropy_loss
