"""
mil_interpretable — Attention MIL for Chemical Rule Interpretability

用 Attention-based Multiple Instance Learning (Ilse et al. 2018) 聚合化学规则
匹配结果，预测谱图对间的结构相似度，并通过 attention weights 解释哪些规则
对判断贡献最大。

与 chem_aware 的关系：
  chem_aware: 规则→对比学习信号→嵌入空间优化（间接）
  mil_interpretable: 规则→MIL聚合→直接预测结构相似度 + 可解释性（直接）
"""

from dreams.models.mil_interpretable.mil_model import RuleAttentionMIL
