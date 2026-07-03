"""
注意力头门控与 λ 控制器 — 模块 A 核心 [v3 对抗协同版]

组件：
  - HeadGatingNetwork: 每层每个注意力头的动态权重 (0~1)
  - LambdaController:  [阶段二/三核心] A-B 对话中枢，从模型状态生成分组 λ

λ 设计（v3 改进）：
  λ 不再是全局标量，而是：
    λ_frag (2-dim): NL + CF 共用的"碎裂路径"λ, ISO/NR/EE 共用的"分子属性"λ
    λ_layer (n_layers,): 每层独立 λ，浅层/深层可差异化
    综合 λ = λ_frag × λ_layer → (2, n_layers) 最终映射到 5 维规则分组

课程调度（Curriculum）：
  Phase WARMUP  (0-W steps):   λ 固定 0.6，给 B 足够的存在感
  Phase RELEASE (W-R steps):   λ 逐步释放给 LambdaController 自动调节
  Phase AUTO    (R+ steps):    完全自动，梯度驱动 λ 收敛

A-B 通信路径：
  Attn_A (纯数据 forward) → 熵分布 + 规则命中率 + 谱图特征
       ↓
  LambdaController → λ_frag, λ_layer, confidence
       ↓
  chem_bias = RuleEngine * λ → 注入协同 forward

作者：module1-chem-attn 开发分支
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class HeadGatingNetwork(nn.Module):
    """
    注意力头动态门控网络（未变，同 v2）
    对 token 嵌入平均池化后经两层 MLP 输出每头权重 [0,1]
    """
    def __init__(self, d_model: int, n_heads: int, hidden_ratio: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        hidden_dim = max(d_model // hidden_ratio, n_heads * 2)

        self.gate_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_heads),
            nn.Sigmoid()
        )
        nn.init.xavier_uniform_(self.gate_net[0].weight)
        nn.init.zeros_(self.gate_net[0].bias)
        nn.init.xavier_uniform_(self.gate_net[3].weight)
        nn.init.ones_(self.gate_net[3].bias)  # Sigmoid(1) ≈ 0.73

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pooled = x.mean(dim=1)  # (batch, d_model)
        return self.gate_net(x_pooled)  # (batch, n_heads)


class LambdaController(nn.Module):
    """
    λ 控制器 — A-B 对抗协同中枢 [v3]

    接收"模型当前学习状态"和"化学规则匹配统计"，输出精细化的分组 λ：
      - λ_frag: (2,) 碎裂规则组 + 分子属性组
      - λ_layer: (n_layers,) 每层独立加权
      - confidence: 标量，B 对当前建议的信心度

    架构：
      Input:  attn_entropy (n_layers * n_heads) — A 的状态
              rule_hit_rates  (n_rules,)          — B 的建议质量
              spec_features   (3,)                — 谱图难度
         ↓
      MLP (state_dim → 32 → 16 → output)
         ↓
      λ_frag = Sigmoid → [0.2, 0.9] 范围（避免极端 0/1）
      λ_layer = Sigmoid × 0.8 + 0.1 → [0.1, 0.9]
      confidence = Sigmoid → [0, 1]

    初始化：bias 设使 λ_frag ≈ 0.6, λ_layer ≈ 0.5, confidence ≈ 0.7
            对应"初始信任化学规则"的状态
    """

    def __init__(
        self,
        n_layers: int = 7,
        n_heads: int = 8,
        n_rules: int = 5,
        state_dim: int = 64
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_rules = n_rules

        # 输入维度：每层每头熵 + 规则命中率 + 谱图特征
        input_dim = n_layers * n_heads + n_rules + 3

        # 共享编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # 三个输出头
        self.head_frag = nn.Sequential(
            nn.Linear(16, 2),
            nn.Sigmoid()
        )  # λ 分组: (frag_path, mol_prop)
        self.head_layer = nn.Sequential(
            nn.Linear(16, n_layers),
            nn.Sigmoid()
        )  # 每层缩放因子
        self.head_confidence = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )  # B 信心

        self._init_weights()

    def _init_weights(self):
        """初始化使 λ 从合理中间值出发"""
        for module in [self.encoder, self.head_confidence]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        # λ_frag: Sigmoid 前 bias=0.5 → 初始 ≈ 0.62
        nn.init.xavier_uniform_(self.head_frag[0].weight)
        nn.init.constant_(self.head_frag[0].bias, 0.5)

        # λ_layer: Sigmoid 前 bias=0 → 初始 ≈ 0.5
        nn.init.xavier_uniform_(self.head_layer[0].weight)
        nn.init.zeros_(self.head_layer[0].bias)

        # confidence: Sigmoid 前 bias=1.0 → 初始 ≈ 0.73
        nn.init.xavier_uniform_(self.head_confidence[0].weight)
        nn.init.constant_(self.head_confidence[0].bias, 1.0)

    # =========================================================================
    # 课程调度（静态工具，不参与梯度）
    # =========================================================================

    @staticmethod
    def curriculum_lambda(
        global_step: int,
        warmup_steps: int = 500,
        release_steps: int = 2500,
        lambda_base: float = 0.6
    ) -> float:
        """
        课程调度：返回当前步应该使用的 λ 上限

        Phase WARMUP  (0 ~ warmup_steps):         λ = lambda_base 固定
        Phase RELEASE (warmup ~ release_steps):   λ 线性释放到底
        Phase AUTO    (release_steps+):            λ = 完全由 LambdaController 输出
        """
        if global_step < warmup_steps:
            return lambda_base
        elif global_step < release_steps:
            progress = (global_step - warmup_steps) / (release_steps - warmup_steps)
            # 线性缩放: 从 lambda_base * 0.9 逐渐降到 0，即释放控制权
            return lambda_base * 0.9 * (1.0 - progress) + 0.02 * progress
        else:
            return 0.02  # 极小值，λ 几乎完全由控制器决定

    # =========================================================================
    # 前向传播
    # =========================================================================

    def forward(
        self,
        attn_entropy_per_layer: torch.Tensor,
        rule_hit_rates: torch.Tensor,
        spec_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成分组 λ 和信心度

        参数：
            attn_entropy_per_layer: (batch, n_layers * n_heads)
                — 纯数据 forward 提取的各层各头熵值
            rule_hit_rates: (batch, n_rules)
                — 每种化学规则在当前谱图上的匹配比例 [0, 1]
            spec_features: (batch, 3)
                — (n_peaks, mz_mean, mz_std) 谱图难度特征

        返回：
            lambda_frag:  (batch, 2) — [frag_path, mol_prop] 分组 λ，值域 [0.1, 0.9]
            lambda_layer: (batch, n_layers) — 每层 λ 缩放，值域 [0.1, 0.9]
            confidence:   (batch, 1) — B 对当前建议的自信度 [0, 1]
        """
        # 拼接输入
        features = torch.cat([
            attn_entropy_per_layer,
            rule_hit_rates,
            spec_features
        ], dim=-1)  # (batch, n_layers*n_heads + n_rules + 3)

        # 共享编码
        encoded = self.encoder(features)  # (batch, 16)

        # 三个输出头
        lambda_frag_raw = self.head_frag(encoded)      # (batch, 2) ∈ [0, 1]
        lambda_layer_raw = self.head_layer(encoded)     # (batch, n_layers) ∈ [0, 1]
        confidence = self.head_confidence(encoded)       # (batch, 1) ∈ [0, 1]

        # 缩放到 [0.1, 0.9] 避免极端值
        lambda_frag = lambda_frag_raw * 0.8 + 0.1
        lambda_layer = lambda_layer_raw * 0.8 + 0.1

        return lambda_frag, lambda_layer, confidence


class StateExtractor(nn.Module):
    """
    学习状态提取器（v3 简化版 — 直接拼接熵向量）

    从各层注意力熵提取"模型学习状态"，供 LambdaController 使用。
    v3 版本不再做 learnable projection，直接输出原始拼接向量，
    让 LambdaController 自己学编码。
    """
    def __init__(self, n_heads: int = 8, n_layers: int = 7):
        super().__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.output_dim = n_heads * n_layers

    def forward(self, attn_entropies: torch.Tensor) -> torch.Tensor:
        """
        参数：
            attn_entropies: (batch, n_layers, n_heads) — 各层各头熵
        返回：
            state: (batch, n_layers * n_heads)
        """
        batch = attn_entropies.shape[0]
        return attn_entropies.reshape(batch, -1)


class LambdaGenerator(nn.Module):
    """
    动态 λ 生成器（v2 遗留，v3 被 LambdaController 取代）
    保留以兼容现有 checkpoint，但推荐迁移到 LambdaController。
    """
    def __init__(self, state_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 16), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(16, 1), nn.Sigmoid()
        )
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[3].weight)
        nn.init.constant_(self.net[3].bias, 2.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
