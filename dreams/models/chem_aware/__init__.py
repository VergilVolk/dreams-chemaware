"""
化学感知注意力模块 (Chemical-Aware Attention Module) — 模块一核心 [v3]

本包实现了化学规则引导的注意力奖励机制：
  - ChemicalRuleEngine v3：奖励式 + 每维度独立权重，将质谱碎裂化学先验编码为注意力偏置
  - ChemAwareDreaMS v3：继承 DreaMS，仅最后一层注入化学偏置，避免跨层复合

组件清单：
  chem_rules.py          — ChemicalRuleEngine v4：6 类化学规则（NL/CF/ISO/NR/EE/HR，~335 条）→ 注意力奖励偏置
  chem_aware_dreams.py   — ChemAwareDreaMS v3：简化主模型（最后一层注入）
  gating.py              — [遗留] HeadGatingNetwork 等（v3 不再使用，保留供参考）
  losses.py              — [遗留] 协同/对抗损失函数（v3 不再使用，保留供参考）
  train_chem_aware.py    — v3 轻量微调脚本

对比开关：
  ChemAwareDreaMS.chem_attn_enabled = False → 行为与原版 DreaMS 完全一致
  ChemAwareDreaMS.chem_attn_enabled = True  → 最后一层注入化学奖励偏置
"""

from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
from dreams.models.chem_aware.chem_aware_dreams import ChemAwareDreaMS
