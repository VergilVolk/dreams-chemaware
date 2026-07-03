"""
化学感知 DreaMS (Chemical-Aware DreaMS) — 模块一主模型 [v3 简化版]

核心改动（v2 → v3）：
  1. 移除 LambdaController / StateExtractor / LambdaGenerator（过度工程化）
  2. 移除 HeadGatingNetwork（门控权重未证明有效）
  3. 化学偏置仅注入最后一层 Transformer（避免跨层复合放大）
  4. ChemicalRuleEngine v3：奖励式 + 每维度独立权重

设计理念：
  - DreaMS 前 6 层自由学习（原版 Graphormer 注意力不受干扰）
  - 最后一层接收化学规则"建议"：匹配碎裂规则的峰对获得正向注意力偏置
  - 每条规则维度的权重独立学习，好规则不会被坏规则拖累
  - 不匹配规则的峰对不受惩罚，保持 DreaMS 原有注意力自由

对比开关：
  chem_attn_enabled = False → 行为与原版 DreaMS 完全一致
  chem_attn_enabled = True  → 最后一层注入化学奖励偏置

作者：module1-chem-attn 开发分支
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Dict, List
from dreams.models.dreams.dreams import DreaMS
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine


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

        # ---- 调用父类初始化 ----
        super().__init__(args, spec_preproc)

        # ---- 构建化学规则引擎 v3 ----
        if self.chem_attn_enabled:
            self.chem_rule_engine = ChemicalRuleEngine(
                tolerance=self.chem_attn_tolerance,
                enable_categories=None  # 全部 6 类启用（NL/CF/ISO/NR/EE/HR）
            )
        else:
            self.chem_rule_engine = None

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

        # ---- Step 6: [v3] 化学奖励偏置 — 仅最后一层 ----
        chem_bias_per_layer = None
        if self.chem_attn_enabled and self.chem_rule_engine is not None:
            # 计算峰对质量差
            mz_diffs = ChemicalRuleEngine.compute_peak_pair_mz_diffs(raw_mz)

            # 化学规则引擎 v3：默认全零 + 匹配规则加分
            chem_bias = self.chem_rule_engine(
                mz_diffs,
                mz_values=raw_mz,
                precursor_mz=raw_mz[:, 0] if 'NR' in self.chem_rule_engine.enabled_categories else None,
                padding_mask=padding_mask
            )

            # 仅注入指定层（默认最后一层），其余层为 None
            target_layer = self.chem_attn_layer
            if target_layer < 0:
                target_layer = self.n_layers + target_layer  # -1 → n_layers-1
            chem_bias_per_layer = [None] * self.n_layers
            chem_bias_per_layer[target_layer] = chem_bias

            # 缓存分析数据
            self._last_chem_analysis = {
                'chem_bias': chem_bias.detach(),
                'rule_weights': self.chem_rule_engine.get_rule_weights().detach().clone(),
                'rule_stats': self.chem_rule_engine.get_rule_stats(),
                'target_layer': target_layer,
            }

        # ---- Step 7: Transformer 编码器 ----
        if self.vanilla_transformer:
            spec = self.transformer_encoder(spec_emb, src_key_padding_mask=padding_mask)
        else:
            spec = self.transformer_encoder(
                spec_emb, padding_mask, graphormer_dists,
                chem_bias=chem_bias_per_layer,
                gate_weights_per_layer=None
            )

        return spec

    # =========================================================================
    # 冻结/解冻
    # =========================================================================

    def freeze_backbone(self):
        """冻结 DreaMS backbone 的所有参数（化学规则引擎除外）"""
        count = 0
        for name, param in self.named_parameters():
            if 'chem_rule_engine' not in name:
                param.requires_grad = False
                count += 1
        return count

    def unfreeze_chem_aware(self):
        """仅解冻化学规则引擎参数"""
        count = 0
        for name, param in self.named_parameters():
            if 'chem_rule_engine' in name:
                param.requires_grad = True
                count += 1
        return count

    def get_chem_aware_params(self):
        """返回化学规则引擎的可训练参数"""
        if self.chem_rule_engine is None:
            return []
        return list(self.chem_rule_engine.parameters())

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
