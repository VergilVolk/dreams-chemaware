"""
化学感知 DreaMS (Chemical-Aware DreaMS) — 真注意力修复版

核心改动（v2 → v3）：
  1. 移除 LambdaController / StateExtractor / LambdaGenerator（过度工程化）
  2. 移除 HeadGatingNetwork（门控权重未证明有效）
  3. 化学偏置真正加到指定 Transformer 层的 pre-softmax logits
  4. 零初始化全局尺度保证启用时仍精确复现官方基线
  5. 旧 post-Transformer 残差保留为显式 legacy control

设计理念：
  - 非目标层保持原版 Graphormer 注意力不受干预
  - 指定层接收化学规则"建议"：匹配碎裂规则的峰对获得正向注意力偏置
  - 每条规则维度的权重独立学习，好规则不会被坏规则拖累
  - 不匹配规则的峰对不受惩罚，保持 DreaMS 原有注意力自由

对比开关：
  chem_attn_enabled = False → 行为与原版 DreaMS 完全一致
      chem_attn_enabled = True  → 指定层注入化学奖励偏置

作者：module1-chem-attn 开发分支
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Dict, List
from dreams.models.dreams.dreams import DreaMS
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine


def route_chemical_bias_to_layer(
    chem_bias: torch.Tensor,
    n_layers: int,
    target_layer: int,
) -> tuple[list[Optional[torch.Tensor]], int]:
    """Route one candidate-independent rule bias to exactly one encoder layer.

    Negative indices follow normal Python indexing.  Returning the resolved
    index makes the scientific intervention auditable and prevents the old
    failure mode where ``chem_attn_layer`` was accepted but never used.
    """

    if n_layers < 1:
        raise ValueError("n_layers must be positive")
    resolved = int(target_layer)
    if resolved < 0:
        resolved += int(n_layers)
    if resolved < 0 or resolved >= int(n_layers):
        raise ValueError(
            f"chem_attn_layer {target_layer} is outside an encoder with {n_layers} layers"
        )
    routed: list[Optional[torch.Tensor]] = [None] * int(n_layers)
    routed[resolved] = chem_bias
    return routed, resolved


class ChemAwareDreaMS(DreaMS):
    """
    化学感知 DreaMS — 模块一主模型 [v3 简化版]

    继承 DreaMS 的全部功能，仅在最后一层 Transformer 自注意力中注入
    化学规则奖励偏置。

    [对比开关]
      chem_attn_enabled: bool — True = 启用化学感知, False = 完全等效原版 DreaMS

    [新增子模块]
      chem_rule_engine: ChemicalRuleEngine v3 — 化学规则 → 注意力奖励偏置
        - 5 维度独立可学习权重
        - 默认全零（不惩罚），匹配规则加分

    [新增参数（通过 args 传入）]
      args.chem_attn: bool — 是否启用化学感知注意力（默认 False）
      args.chem_attn_tolerance: float — 质量匹配容差 Da（默认 0.02）
      args.chem_attn_layer: int — 注入化学偏置的层索引（默认 -1 = 最后一层）
    """

    def __init__(self, args, spec_preproc):
        """
        初始化化学感知 DreaMS [v3]

        参数：
            args: Namespace — 除原版 DreaMS 参数外，额外支持：
                chem_attn (bool): 化学感知注意力开关，默认 False
                chem_attn_tolerance (float): 质量容差 Da，默认 0.02
                chem_attn_layer (int): 注入层索引，默认 -1（最后一层）
            spec_preproc: SpectrumPreprocessor — 谱图预处理器
        """
        # ---- 提取模块一专属参数（必须在 super().__init__ 之前） ----
        self.chem_attn_enabled = getattr(args, 'chem_attn', False)
        self.chem_attn_tolerance = getattr(args, 'chem_attn_tolerance', 0.02)
        self.chem_attn_layer = getattr(args, 'chem_attn_layer', -1)
        self.chem_attn_mode = getattr(args, 'chem_attn_mode', 'attention')
        if self.chem_attn_mode not in {'attention', 'residual'}:
            raise ValueError("chem_attn_mode must be 'attention' or 'residual'")
        self.chem_attn_categories = tuple(
            getattr(args, 'chem_attn_categories', ('NL', 'CF', 'ISO'))
        )
        self.chem_attn_use_massbank = getattr(args, 'chem_attn_use_massbank', False)  # 默认屏蔽 MassBank 噪声

        # ---- 调用父类初始化 ----
        super().__init__(args, spec_preproc)

        # ---- 构建化学规则引擎 v3 ----
        if self.chem_attn_enabled:
            self.chem_rule_engine = ChemicalRuleEngine(
                tolerance=self.chem_attn_tolerance,
                enable_categories=None,  # 全部 6 类启用（NL/CF/ISO/NR/EE/HR）
                use_massbank=self.chem_attn_use_massbank
            )
            # The true attention route is exactly the official model at init.
            # A zero gate first learns whether the aggregate rule graph helps;
            # individual softplus rule weights receive gradients once it opens.
            self.chem_attention_scale = (
                nn.Parameter(torch.tensor(0.0))
                if self.chem_attn_mode == 'attention' else None
            )
            # Retain the old post-Transformer residual only as an explicit
            # legacy control; it is no longer mislabeled as attention.
            self.chem_residual_scale = (
                nn.Parameter(torch.tensor(1.0))
                if self.chem_attn_mode == 'residual' else None
            )
        else:
            self.chem_rule_engine = None
            self.chem_attention_scale = None
            self.chem_residual_scale = None

        # 缓存最近一次的化学分析数据
        self._last_chem_analysis: Optional[Dict] = None

    # =========================================================================
    # 前向传播
    # =========================================================================

    def forward(self, spec, charge=None):
        """
        化学感知前向传播 [v3 简化版]

        流程：
          1. 原始 DreaMS 编码（padding → charge → peak_emb → fourier）
          2. 计算化学偏置（仅当 chem_attn_enabled）
          3. 化学偏置仅注入最后一层 Transformer
          4. 返回编码结果
        """
        # ---- Step 1: Padding 掩码 ----
        padding_mask = spec[:, :, 0] == 0

        # ---- Step 2: 保存原始 m/z ----
        if self.chem_attn_enabled:
            raw_mz = spec[:, :, 0].clone()

        # ---- Step 3: 电荷特征拼接 ----
        if self.charge_feature:
            if charge is None:
                raise ValueError('charge_feature=True 但未提供 charge 参数')
            charge_features = ~padding_mask * charge.unsqueeze(-1)
            spec = torch.cat([spec, charge_features.unsqueeze(-1)], dim=-1)

        # ---- Step 4: 峰值嵌入 + 傅里叶特征 ----
        peak_embs = self.ff_peak(self._DreaMS__normalize_spec(spec))

        if self.d_fourier:
            fourier_features = self.ff_fourier(self.fourier_enc(spec[..., [0]]))
            spec_emb = torch.cat([peak_embs, fourier_features], dim=-1)
        elif self.d_mz_token:
            import dreams.utils.spectra as su
            tokenized_mzs = self.mz_tokenizer(
                su.to_classes(spec[..., [0]], max_val=self.dformat.max_mz,
                              bin_size=self.hot_mz_bin_size,
                              special_vals=[self.mask_val]).squeeze()
            )
            tokenized_mzs = self.ff_mz_token(tokenized_mzs)
            spec_emb = torch.cat([peak_embs, tokenized_mzs], dim=-1)
        else:
            spec_emb = peak_embs

        # ---- Step 5: Graphormer 距离编码 ----
        graphormer_dists = None
        if self.graphormer_mz_diffs:
            if self.d_fourier:
                graphormer_dists = fourier_features.unsqueeze(2) - fourier_features.unsqueeze(1)
            else:
                graphormer_dists = spec_emb[..., 0].unsqueeze(2) - spec_emb[..., 0].unsqueeze(1)
                graphormer_dists = graphormer_dists.unsqueeze(-1)

        # ---- Step 6: [v3] 化学偏置计算 ----
        chem_bias = None
        chem_bias_specific = None
        if self.chem_attn_enabled and self.chem_rule_engine is not None:
            # 计算峰对质量差
            mz_diffs = ChemicalRuleEngine.compute_peak_pair_mz_diffs(raw_mz)

            # 全类别偏置（用于日志/分析，包含 EE/NR）
            chem_bias = self.chem_rule_engine(
                mz_diffs,
                mz_values=raw_mz,
                precursor_mz=raw_mz[:, 0],
                padding_mask=padding_mask
            )

            # 特异性偏置（仅 NL/CF/ISO，用于残差连接）
            # EE 和 NR 是全覆盖规则，不加区分地给所有峰对加分，会稀释特异性信号
            chem_bias_specific = self.chem_rule_engine(
                mz_diffs,
                mz_values=raw_mz,
                precursor_mz=raw_mz[:, 0],
                padding_mask=padding_mask,
                categories=list(self.chem_attn_categories)
            )

            # 缓存分析数据
            self._last_chem_analysis = {
                'chem_bias': chem_bias.detach(),
                'chem_bias_specific': chem_bias_specific.detach(),
                'rule_weights': self.chem_rule_engine.get_rule_weights().detach().clone(),
                'rule_stats': self.chem_rule_engine.get_rule_stats(),
                'mode': self.chem_attn_mode,
                'categories': list(self.chem_attn_categories),
            }

        # ---- Step 7: Transformer 编码器 ----
        if self.vanilla_transformer:
            if self.chem_attn_enabled and self.chem_attn_mode == 'attention':
                raise RuntimeError(
                    "true chemical attention requires the custom DreaMS encoder"
                )
            spec = self.transformer_encoder(spec_emb, src_key_padding_mask=padding_mask)
        else:
            layer_biases = None
            resolved_layer = None
            if (
                self.chem_attn_enabled
                and self.chem_attn_mode == 'attention'
                and chem_bias_specific is not None
            ):
                effective_bias = chem_bias_specific * self.chem_attention_scale
                layer_biases, resolved_layer = route_chemical_bias_to_layer(
                    effective_bias,
                    int(self.transformer_encoder.n_layers),
                    int(self.chem_attn_layer),
                )
                self._last_chem_analysis.update({
                    'target_layer': int(resolved_layer),
                    'effective_scale': self.chem_attention_scale.detach().clone(),
                    'effective_bias': effective_bias.detach(),
                })
            spec = self.transformer_encoder(
                spec_emb, padding_mask, graphormer_dists,
                chem_bias=layer_biases, gate_weights_per_layer=None
            )

        # ---- Step 8: legacy chemical residual control (not attention) ----
        if (
            self.chem_attn_enabled
            and self.chem_attn_mode == 'residual'
            and chem_bias_specific is not None
        ):
            cw0 = chem_bias_specific.squeeze(1)                    # (batch, n, n)
            cw1 = cw0 - cw0.mean(dim=-1, keepdim=True)             # mean-subtract
            cw2 = cw1 * self.chem_residual_scale                    # × learnable scale
            chem_context = torch.bmm(cw2, spec)
            # 诊断：逐级 retain_grad
            for name, t in [('cw2', cw2), ('chem_context', chem_context)]:
                if t.requires_grad:
                    t.retain_grad()
                    setattr(self, f'_diag_{name}', t)
            spec = spec + chem_context

        return spec

    # =========================================================================
    # 冻结/解冻
    # =========================================================================

    def freeze_backbone(self):
        """冻结 DreaMS backbone 的所有参数（chem_aware 模块除外）"""
        count = 0
        for name, param in self.named_parameters():
            if (
                'chem_rule_engine' not in name
                and 'chem_attention_scale' not in name
                and 'chem_residual_scale' not in name
            ):
                param.requires_grad = False
                count += 1
        return count

    def unfreeze_chem_aware(self):
        """仅解冻 chem_aware 模块参数（规则权重 + 残差缩放）"""
        count = 0
        for name, param in self.named_parameters():
            if (
                'chem_rule_engine' in name
                or 'chem_attention_scale' in name
                or 'chem_residual_scale' in name
            ):
                param.requires_grad = True
                count += 1
        return count

    def get_chem_aware_params(self):
        """返回所有 chem_aware 可训练参数（规则权重 + 残差缩放因子）"""
        if self.chem_rule_engine is None:
            return []
        params = list(self.chem_rule_engine.parameters())
        if self.chem_attention_scale is not None:
            params.append(self.chem_attention_scale)
        if self.chem_residual_scale is not None:
            params.append(self.chem_residual_scale)
        return params

    # =========================================================================
    # 优化器
    # =========================================================================

    def configure_optimizers(self):
        """
        [v3] 仅优化化学规则引擎的权重向量（5 个参数）

        backbone 冻结，仅 rule_weights_raw 参与训练。
        """
        chem_params = self.get_chem_aware_params()
        if not chem_params:
            return super().configure_optimizers()

        optimizer = torch.optim.Adam(chem_params, lr=self.lr * 0.1, weight_decay=self.weight_decay)
        return optimizer

    # =========================================================================
    # 训练步骤
    # =========================================================================

    def step(self, data, batch_idx, log_prefix):
        """
        训练/验证步骤 [v3 简化版]

        直接调用父类 step，不做额外损失计算。
        化学规则权重通过 mask prediction 损失的梯度自然优化。
        """
        return super().step(data, batch_idx, log_prefix)

    # =========================================================================
    # 分析接口
    # =========================================================================

    def get_chem_attn_analysis(self) -> Optional[Dict]:
        """
        获取最近一次前向传播的化学感知分析数据

        返回的字典包含：
            chem_bias: (batch, 1, n, n) — 化学偏置矩阵
            rule_weights: (5,) — 各规则维度的当前权重
            rule_stats: Dict — 各规则的命中统计
            target_layer: int — 化学偏置注入的层索引

        返回 None 如果 chem_attn_enabled=False 或尚未执行 forward
        """
        if not self.chem_attn_enabled:
            return None
        return self._last_chem_analysis

    def on_save_checkpoint(self, checkpoint):
        """保存检查点时清理缓存数据"""
        keys_to_remove = [k for k in checkpoint.get('state_dict', {}).keys()
                         if '_last_chem_analysis' in k]
        for k in keys_to_remove:
            del checkpoint['state_dict'][k]
        return checkpoint
