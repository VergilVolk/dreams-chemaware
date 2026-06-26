"""
化学感知注意力相关损失函数 [v2 阶段二]

功能：
  - attention_entropy_loss: 注意力熵正则化，鼓励适度聚焦
  - lambda_regularization_loss: λ 正则化，防止 λ 坍塌到 0
  - compute_per_layer_entropy: 从 hook 抓取的各层注意力计算熵向量（供 StateExtractor 使用）

设计原则：
  - 熵正则化引导注意力聚焦，但不强制（软约束）
  - λ 正则化保持化学规则的底线参与度
  - 所有辅助损失通过很小的权重（~0.01）与主任务损失（mask loss）混合

作者：module1-chem-attn 开发分支
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


def attention_entropy_loss(
    attn_weights: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    target_entropy: float = 3.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    注意力熵正则化损失

    对注意力权重矩阵计算逐行熵，与目标熵值做 MSE 损失。
    - 熵过低 → 注意力过度集中（可能过拟合到少数几个峰）
    - 熵过高 → 注意力过度平坦（没有真正聚焦到关键碎裂路径）
    - target_entropy ≈ log(n_valid_peaks) / 2

    参数：
        attn_weights: (batch, n_heads, n_peaks, n_peaks) — softmax 后的注意力权重
        padding_mask: (batch, n_peaks) — padding 掩码（True=填充位），可选
        target_entropy: float — 目标熵值，默认 3.0
        reduction: str — 'mean'（批平均）或 'none'（返回逐样本损失）

    返回：
        loss: 标量或逐样本张量
    """
    eps = 1e-8
    log_attn = torch.log(attn_weights + eps)
    entropy = -(attn_weights * log_attn).sum(dim=-1)  # (bs, n_heads, n)

    if padding_mask is not None:
        pad_expanded = ~padding_mask.unsqueeze(1)  # (bs, 1, n)
        entropy = entropy * pad_expanded.float()
        n_valid = pad_expanded.sum(dim=-1, keepdim=True).clamp(min=1).float()
        avg_entropy = (entropy.sum(dim=-1) / n_valid.squeeze(-1))  # (bs, n_heads)
    else:
        avg_entropy = entropy.mean(dim=-1)  # (bs, n_heads)

    avg_entropy = avg_entropy.mean(dim=-1)  # (bs,)  对所有头取平均

    loss = F.mse_loss(avg_entropy, torch.full_like(avg_entropy, target_entropy),
                      reduction=reduction)
    return loss


def lambda_regularization_loss(lambda_val: torch.Tensor, min_lambda: float = 0.1) -> torch.Tensor:
    """
    λ 正则化损失 — 防止 λ 在训练中坍塌到 0

    如果 λ → 0，化学规则完全失效。通过施加 min_lambda 的软下界，
    确保化学先验始终有最低限度的参与。

    参数：
        lambda_val: (batch, 1) 或 标量 — 当前的 λ 值
        min_lambda: float — λ 软下界，默认 0.1

    返回：
        loss: 标量 — λ 低于下界时产生的惩罚
    """
    return F.relu(min_lambda - lambda_val).mean()


def compute_per_layer_entropy(
    attn_matrices: List[torch.Tensor],
    padding_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    从各层注意力矩阵计算逐头熵向量（供 StateExtractor 使用）

    参数：
        attn_matrices: List[Tensor] — 每层的注意力权重，
            每个元素形状为 (batch, n_heads, n, n) 或 (batch*n_heads, n, n)
        padding_mask: (batch, n) — padding 掩码

    返回：
        entropies: (batch, n_layers, n_heads) — 每层每头的平均熵
    """
    eps = 1e-8
    n_layers = len(attn_matrices)
    batch = attn_matrices[0].shape[0]
    n_heads = attn_matrices[0].shape[1] if attn_matrices[0].dim() == 4 else None

    layer_entropies = []

    for layer_attn in attn_matrices:
        # 统一形状为 (batch, n_heads, n, n)
        if layer_attn.dim() == 3:
            # (batch*n_heads, n, n) → 需要 n_heads 信息
            # 回退：假设 batch=1 或从外部传入 n_heads
            if n_heads is None:
                raise ValueError('Cannot infer n_heads from 3D attention tensor')
            layer_attn = layer_attn.reshape(batch, n_heads, *layer_attn.shape[1:])

        log_a = torch.log(layer_attn + eps)
        H = -(layer_attn * log_a).sum(dim=-1)  # (batch, n_heads, n)

        if padding_mask is not None:
            pad_exp = ~padding_mask.unsqueeze(1)  # (batch, 1, n)
            H = H * pad_exp.float()
            n_valid = pad_exp.sum(dim=-1, keepdim=True).clamp(min=1).float()
            H_mean = (H.sum(dim=-1) / n_valid.squeeze(-1))  # (batch, n_heads)
        else:
            H_mean = H.mean(dim=-1)  # (batch, n_heads)

        layer_entropies.append(H_mean)

    return torch.stack(layer_entropies, dim=1)  # (batch, n_layers, n_heads)


class AdversarialLoss(nn.Module):
    """
    对抗损失（阶段三桩实现）

    借鉴 GAN 思想：判别器 D 判断注意力分布是"规则引导的"还是"数据自学的"，
    生成器（即注意力头）试图骗过判别器。

    参数：
        lambda_adv: float — 对抗损失的权重系数，默认 0.01
    """

    def __init__(self, lambda_adv: float = 0.01):
        super().__init__()
        self.lambda_adv = lambda_adv

    def forward(
        self,
        attn_weights: torch.Tensor,
        chem_bias: torch.Tensor,
        discriminator: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """当前返回 0，阶段三实现"""
        if discriminator is None:
            return torch.tensor(0.0, device=attn_weights.device)
        raise NotImplementedError(
            "对抗判别器将在阶段三实现。当前使用 attention_entropy_loss 和 "
            "lambda_regularization_loss 进行训练正则化。"
        )
