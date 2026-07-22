"""
RuleAttentionMIL — Gated Attention MIL for Chemical Rule Bag Classification

基于 Ilse et al. 2018 (ICML) "Attention-based Deep Multiple Instance Learning"
的 gated attention 变体，适配化学规则匹配向量。

输入：一张谱图对共同命中的规则列表（每个 instance 是 12 维特征向量）
输出：结构相似概率 + 每条规则的 attention weight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RuleAttentionMIL(nn.Module):
    """
    Gated Attention MIL for chemical rule interpretability.

    参数：
        instance_dim: int — 每个规则 instance 的特征维度（默认 12）
        hidden_dim: int — 隐层维度（默认 32）

    输入：
        instances: (n, instance_dim) — bag 中 n 条命中规则的特征
        calibrated: bool — 是否应用 temperature scaling

    返回：
        prob: 标量 — P(结构相似)
        attn: (n,) — 每条规则的 attention weight
    """

    def __init__(self, instance_dim=12, hidden_dim=32):
        super().__init__()

        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(instance_dim, hidden_dim),
            nn.ReLU(),
        )

        # Gated attention（两路门控：Tanh + Sigmoid → 逐元素乘）
        self.attn_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attn_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attn_w = nn.Linear(hidden_dim, 1)

        # 分类器
        self.classifier = nn.Linear(hidden_dim, 1)

        # 空 bag 的 embedding（可学习）
        self.no_evidence_embedding = nn.Parameter(torch.zeros(hidden_dim))

        # Temperature scaling 参数（阶段 B 单独训练，阶段 A 冻结）
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def encode_bag(self, instances):
        """
        将 bag 编码为单个向量。

        参数：
            instances: (n, instance_dim) — 可能为 n=0（空 bag）

        返回：
            bag_repr: (hidden_dim,) — 加权聚合后的 bag 表示
            attn: (n,) — 每条规则的 attention weight
        """
        if instances.shape[0] == 0:
            return self.no_evidence_embedding, torch.zeros(0, device=instances.device)

        h = self.feature_extractor(instances)                       # (n, hidden_dim)
        a_raw = self.attn_w(self.attn_V(h) * self.attn_U(h))        # (n, 1)
        a = torch.softmax(a_raw, dim=0)                              # (n, 1)
        bag_repr = (a * h).sum(dim=0)                                # (hidden_dim,)
        return bag_repr, a.squeeze(-1)

    def forward(self, instances):
        """
        前向传播（回归模式）。返回预测 Tanimoto 值和 attention weights。
        """
        bag_repr, attn = self.encode_bag(instances)
        pred = self.classifier(bag_repr)  # 标量，无 sigmoid
        return pred, attn
