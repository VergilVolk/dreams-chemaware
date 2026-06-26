"""
注意力头门控网络 (Head Gating Network) — 模块 A 核心组件 [v2 阶段二]

功能：
  - HeadGatingNetwork: 基于 token 嵌入动态生成每个注意力头的权重 (0~1)
  - StateExtractor: 从各层注意力熵提取"学习状态向量"
  - LambdaGenerator: [阶段二新增] 根据状态向量生成化学规则约束强度 λ ∈ [0,1]

A-B 协同路径（阶段二）：
  输入谱图 → Transformer 前向
    ├── 各层注意力矩阵 → 逐头熵计算 → StateExtractor → state_vector (64-dim)
    │                                                       ↓
    │                                              LambdaGenerator
    │                                                       ↓
    │                                                  λ ∈ [0, 1]
    │                                                       ↓
    └── ChemRuleEngine(..., λ) → chem_bias → 注入 softmax 前

作者：module1-chem-attn 开发分支
"""

import torch
import torch.nn as nn
from typing import Optional


class HeadGatingNetwork(nn.Module):
    """
    注意力头动态门控网络（模块 A）

    通过对 token 嵌入的池化统计量进行小型 MLP 推理，为每个注意力头输出
    一个 [0, 1] 范围的权重系数。训练过程中自动学习哪些头在何种输入下更重要。

    架构：
        Input: (batch, n_peaks, d_model) — 当前层输入 token 序列
          ↓
        平均池化 (n_peaks 维度) → (batch, d_model)
          ↓
        两层 MLP: d_model → d_model/4 → n_heads
          ↓
        Sigmoid → (batch, n_heads) 门控权重 ∈ [0, 1]

    参数：
        d_model: int — token 嵌入维度（例如 512 或 1024）
        n_heads: int — 注意力头数量（例如 8）
        hidden_ratio: int — 隐藏层降采样比率，默认 4（即 hidden_dim = d_model / 4）

    使用示例：
        >>> gate_net = HeadGatingNetwork(d_model=512, n_heads=8)
        >>> x = torch.randn(4, 100, 512)  # (batch=4, n_peaks=100, d_model=512)
        >>> gate_w = gate_net(x)           # (4, 8)
        >>> # gate_w[i, j] = 第 i 个样本的第 j 个注意力头权重
    """

    def __init__(self, d_model: int, n_heads: int, hidden_ratio: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        hidden_dim = max(d_model // hidden_ratio, n_heads * 2)

        # 两层 MLP，Sigmoid 输出保证权重在 [0, 1]
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_heads),
            nn.Sigmoid()
        )

        # 参数初始化：偏置初始化为 1.0，使初始时所有头接近等权
        nn.init.xavier_uniform_(self.gate_net[0].weight)
        nn.init.zeros_(self.gate_net[0].bias)
        nn.init.xavier_uniform_(self.gate_net[3].weight)
        nn.init.ones_(self.gate_net[3].bias)  # Sigmoid 前 bias=1 → 初始输出 ~0.73

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算注意力头门控权重

        参数：
            x: (batch, n_peaks, d_model) — 当前层的 token 嵌入序列

        返回：
            gate_weights: (batch, n_heads) — 每个注意力头的门控权重，值域 [0, 1]
        """
        # 沿峰值维度平均池化，得到谱图级表征
        x_pooled = x.mean(dim=1)  # (batch, d_model)
        return self.gate_net(x_pooled)  # (batch, n_heads)


class StateExtractor(nn.Module):
    """
    学习状态提取器（阶段二预备组件）

    从各层的注意力分布中提取"学习状态向量"，用于反馈给模块 B
    以动态调节化学规则的约束强度 λ。

    目前为桩实现（stub），阶段二将扩展为完整的状态编码器。

    参数：
        n_heads: int — 注意力头数量
        n_layers: int — Transformer 层数
        state_dim: int — 输出状态向量维度，默认 64
    """

    def __init__(self, n_heads: int, n_layers: int, state_dim: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.state_dim = state_dim

        # 桩实现：一个简单的线性投影（阶段二将替换为 LSTM/Transformer）
        self.proj = nn.Linear(n_heads * n_layers, state_dim)

    def forward(self, attn_entropies: torch.Tensor) -> torch.Tensor:
        """
        提取学习状态向量

        参数：
            attn_entropies: (batch, n_layers, n_heads) — 每层每头的注意力熵值

        返回：
            state: (batch, state_dim) — 压缩后的学习状态向量
        """
        batch = attn_entropies.shape[0]
        flat = attn_entropies.reshape(batch, -1)  # (batch, n_layers * n_heads)
        return self.proj(flat)  # (batch, state_dim)


class LambdaGenerator(nn.Module):
    """
    动态 λ 生成器（阶段二新增 — A→B 反馈通道的核心）

    根据从注意力分布提取的"学习状态向量"，生成化学规则约束强度 λ ∈ [0, 1]。
    这是 A 告诉 B "我需要多大力度化学引导"的通信通道。

    机制：
      - 训练早期：注意力散乱 + mask loss 高 → state 表示"困惑" → λ ≈ 1.0（强力引导）
      - 训练后期：注意力聚焦 + mask loss 低 → state 表示"熟练" → λ ≈ 0.2（温和建议）
      - λ 的梯度来自 mask loss：化学偏置帮到预测 → λ 保持或增大；干扰预测 → λ 被压小

    架构：
        Input: state_vector (batch, state_dim=64)
          ↓
        Linear(64 → 16) → ReLU → Dropout(0.1)
          ↓
        Linear(16 → 1) → Sigmoid
          ↓
        λ: (batch, 1) ∈ [0, 1]

    参数：
        state_dim: int — 状态向量维度，需与 StateExtractor.state_dim 一致（默认 64）
    """

    def __init__(self, state_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim

        # 小型 MLP: 64 → 16 → 1, Sigmoid 输出 [0, 1]
        self.net = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # 初始化：偏置设为 2.0，使 Sigmoid(2.0) ≈ 0.88，初始时相信化学规则
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[3].weight)
        nn.init.constant_(self.net[3].bias, 2.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        根据学习状态生成 λ

        参数：
            state: (batch, state_dim) — 来自 StateExtractor 的学习状态向量

        返回：
            lambda_val: (batch, 1) — 化学规则约束强度，值域 [0, 1]
        """
        return self.net(state)

