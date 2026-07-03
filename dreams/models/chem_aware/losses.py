"""
化学感知协同损失函数 [v3 对抗协同版]

设计原则：
  - 化学规则应帮助模型做得更好，而非仅仅"不添乱"
  - λ 的各维度应差异化（不同规则/不同层权重不同）
  - 模型不能过度依赖化学规则（保留发现新碎裂路径的能力）

三合一协同损失：
  L_collab = L_consistency + α * L_novelty + β * L_overmix

作者：module1-chem-attn 开发分支
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


# ==============================================================================
# 协同损失组件
# ==============================================================================

def consistency_loss(
    L_pure: torch.Tensor,      # (batch,) 纯数据 mask loss
    L_chem: torch.Tensor,      # (batch,) 带化学偏置 mask loss
    lambda_conf: torch.Tensor, # (batch, 1) B 的信心度
    margin: float = 0.1
) -> torch.Tensor:
    """
    一致性损失 — A 和 C 不能走太远

    A = 纯数据路径, C = 化学协同路径

    逻辑：
      如果 L_chem < L_pure（加规则更好）→ 不惩罚，鼓励这种偏离
      如果 L_chem > L_pure + margin（加规则明显更差）→ 惩罚，按 B 信心度加权：
          信心高但效果差 → B 错了 → 重罚
          信心低但效果差 → B 知道自己不确定 → 轻罚

    参数：
        L_pure: (batch,) 纯数据 mask loss
        L_chem: (batch,) 协同 mask loss
        lambda_conf: (batch, 1) B 对当前样本的信心
        margin: float — 允许的"变差"容忍度
    """
    # 计算 C 比 A 差多少
    degradation = F.relu(L_chem - L_pure - margin)  # (batch,)
    # B 信心越高 → 惩罚越重
    weighted_degradation = degradation * lambda_conf.squeeze(-1)
    return weighted_degradation.mean()


def novelty_bonus(
    attn_pure: torch.Tensor,        # (batch, n_heads, n, n) 纯数据注意力
    chem_bias: torch.Tensor,        # (batch, n, n) 化学偏置
    max_cosine: float = 0.90
) -> torch.Tensor:
    """
    新颖性奖励 — 防止 A 过度模仿 B

    如果 A 的注意力分布与 B 的偏置高度一致（余弦 > max_cosine），
    说明 A 没有自己发现任何超越规则库的东西 → 惩罚。

    参数：
        attn_pure: (batch, n_heads, n, n) 纯数据注意力
        chem_bias: (batch, n, n) 化学偏置（已压缩到 2D）
        max_cosine: float — 允许的最大一致性
    """
    # 平均所有头
    attn_mean = attn_pure.mean(dim=1)  # (batch, n, n)

    # 展平后计算余弦相似度
    attn_flat = attn_mean.reshape(attn_mean.shape[0], -1)  # (batch, n*n)
    bias_flat = chem_bias.reshape(chem_bias.shape[0], -1)  # (batch, n*n)

    cosine = F.cosine_similarity(attn_flat, bias_flat, dim=-1)  # (batch,)

    # 超过阈值的部分惩罚
    bonus = F.relu(cosine - max_cosine)
    return bonus.mean()


def rule_overmix_loss(
    lambda_frag: torch.Tensor,   # (batch, 2)
    lambda_layer: torch.Tensor,  # (batch, n_layers)
    min_var: float = 0.05
) -> torch.Tensor:
    """
    规则过度混合损失 — λ 不能全 0 或全 1

    三个约束：
      1. λ_frag 两维之间应有差异（不能两个规则组永远一样）
      2. λ_layer 各层之间应有差异（浅层深层不能完全同权）
      3. 总平均值不应极端（不能全部 > 0.9 或 < 0.2）

    参数：
        lambda_frag: (batch, 2) 分组 λ
        lambda_layer: (batch, n_layers) 层级 λ
        min_var: float — 最低允许方差
    """
    # 约束 1: frag 组间方差
    frag_var = lambda_frag.var(dim=-1).mean()  # (batch,) → 标量
    loss_frag = F.relu(min_var - frag_var)

    # 约束 2: layer 间方差
    layer_var = lambda_layer.var(dim=-1).mean()
    loss_layer = F.relu(min_var - layer_var)

    # 约束 3: 总平均值应在 [0.25, 0.85] 之间
    all_mean = lambda_frag.mean() * 0.3 + lambda_layer.mean() * 0.7  # 加权
    loss_mean = F.relu(0.25 - all_mean) + F.relu(all_mean - 0.85)

    return loss_frag + loss_layer + loss_mean


# ==============================================================================
# 三合一协同损失
# ==============================================================================

def collaborative_loss(
    L_pure: torch.Tensor,
    L_chem: torch.Tensor,
    lambda_frag: torch.Tensor,
    lambda_layer: torch.Tensor,
    lambda_conf: torch.Tensor,
    attn_pure: torch.Tensor,
    chem_bias: torch.Tensor,
    alpha_novelty: float = 0.05,
    beta_overmix: float = 0.02,
) -> Tuple[torch.Tensor, dict]:
    """
    三合一协同损失

    L_collab = L_consistency + α * L_novelty + β * L_overmix

    返回：
        total_loss: 标量
        components: dict — 各组件值供日志
    """
    L_cons = consistency_loss(L_pure, L_chem, lambda_conf)
    L_nov = novelty_bonus(attn_pure, chem_bias)
    L_omix = rule_overmix_loss(lambda_frag, lambda_layer)

    total = L_cons + alpha_novelty * L_nov + beta_overmix * L_omix

    components = {
        'consistency': L_cons.item() if torch.is_tensor(L_cons) else L_cons,
        'novelty': L_nov.item() if torch.is_tensor(L_nov) else L_nov,
        'overmix': L_omix.item() if torch.is_tensor(L_omix) else L_omix,
    }
    return total, components


# ==============================================================================
# 辅助函数（阶段一遗留，保留兼容）
# ==============================================================================

def attention_entropy_loss(
    attn_weights: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    target_entropy: float = 3.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """注意力熵正则化（阶段一兼容）"""
    eps = 1e-8
    log_attn = torch.log(attn_weights + eps)
    entropy = -(attn_weights * log_attn).sum(dim=-1)
    if padding_mask is not None:
        pad_expanded = ~padding_mask.unsqueeze(1)
        entropy = entropy * pad_expanded.float()
        n_valid = pad_expanded.sum(dim=-1, keepdim=True).clamp(min=1).float()
        avg_entropy = (entropy.sum(dim=-1) / n_valid.squeeze(-1))
    else:
        avg_entropy = entropy.mean(dim=-1)
    avg_entropy = avg_entropy.mean(dim=-1)
    return F.mse_loss(avg_entropy, torch.full_like(avg_entropy, target_entropy), reduction=reduction)


def lambda_regularization_loss(lambda_val: torch.Tensor, min_lambda: float = 0.1) -> torch.Tensor:
    """λ 正则化（阶段二兼容）"""
    return F.relu(min_lambda - lambda_val).mean()


def compute_per_layer_entropy(
    attn_matrices: List[torch.Tensor],
    padding_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """从各层注意力计算逐头熵（阶段一兼容）"""
    eps = 1e-8
    layer_entropies = []
    for layer_attn in attn_matrices:
        if layer_attn.dim() == 3:
            batch = layer_attn.shape[0]
            n_valid = layer_attn.shape[1]
            n_heads_guess = 1
            layer_attn = layer_attn.reshape(batch, n_heads_guess, n_valid, n_valid)
        log_a = torch.log(layer_attn + eps)
        H = -(layer_attn * log_a).sum(dim=-1)
        if padding_mask is not None:
            pad_exp = ~padding_mask.unsqueeze(1)
            H = H * pad_exp.float()
            n_valid_count = pad_exp.sum(dim=-1, keepdim=True).clamp(min=1).float()
            H_mean = (H.sum(dim=-1) / n_valid_count.squeeze(-1))
        else:
            H_mean = H.mean(dim=-1)
        layer_entropies.append(H_mean)
    return torch.stack(layer_entropies, dim=1)


class AdversarialLoss(nn.Module):
    """对抗损失桩（阶段三）"""
    def __init__(self, lambda_adv: float = 0.01):
        super().__init__()
        self.lambda_adv = lambda_adv

    def forward(self, attn_weights, chem_bias, discriminator=None) -> torch.Tensor:
        if discriminator is None:
            return torch.tensor(0.0, device=attn_weights.device)
        raise NotImplementedError("阶段三实现")
