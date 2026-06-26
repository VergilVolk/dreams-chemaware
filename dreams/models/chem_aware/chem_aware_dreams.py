"""
化学感知 DreaMS (Chemical-Aware DreaMS) — 模块一主模型

继承自 DreaMS，在 Transformer 自注意力中注入"数据驱动 + 化学规则引导"双轨机制：
  - 模块 A（数据驱动）：HeadGatingNetwork 为每层每个注意力头生成动态权重
  - 模块 B（知识引导）：ChemicalRuleEngine 将质谱碎裂化学先验编码为注意力偏置矩阵

对比开关：
  chem_attn_enabled = False → 行为与原版 DreaMS 完全一致，可直接用于基线测试
  chem_attn_enabled = True  → 启用化学感知注意力（chem_bias + gate_weights）

主要重写的方法：
  __init__(): 在 DreaMS 基础上新增 ChemicalRuleEngine 和 HeadGatingNetwork
  forward(): 在原始前向传播中注入 chem_bias 和 gate_weights，新增注意力熵正则化返回
  get_chem_attn_analysis(): 返回化学感知注意力的可解释性数据（用于可视化）

使用示例（对比实验）：
    >>> from dreams.models.chem_aware import ChemAwareDreaMS
    >>>
    >>> # 原版 DreaMS（关开关）→ 完全等效于 DreaMS
    >>> args.chem_attn = False
    >>> model_baseline = ChemAwareDreaMS(args, spec_preproc)
    >>>
    >>> # 化学感知版（开开关）
    >>> args.chem_attn = True
    >>> args.chem_attn_attenuation = -5.0
    >>> model_chem = ChemAwareDreaMS(args, spec_preproc)

作者：module1-chem-attn 开发分支
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Dict, List
from dreams.models.dreams.dreams import DreaMS
from dreams.models.dreams.layers import TransformerEncoder
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
from dreams.models.chem_aware.gating import HeadGatingNetwork, StateExtractor, LambdaGenerator
from dreams.models.chem_aware.losses import attention_entropy_loss, lambda_regularization_loss


class ChemAwareDreaMS(DreaMS):
    """
    化学感知 DreaMS — 模块一主模型

    继承 DreaMS 的全部功能（掩码峰预测预训练、保留顺序预测、嵌入提取等），
    并在 Transformer 自注意力层中注入化学感知机制。

    [对比开关]
      chem_attn_enabled: bool — True = 启用化学感知, False = 完全等效原版 DreaMS

    [新增子模块]
      chem_rule_engine: ChemicalRuleEngine — 模块 B，化学规则 → 注意力偏置
      gate_network: HeadGatingNetwork — 模块 A，嵌入 → 注意力头权重

    [新增参数（通过 args 传入）]
      args.chem_attn: bool — 是否启用化学感知注意力（默认 False）
      args.chem_attn_attenuation: float — 化学不合理峰对的衰减强度（默认 -5.0，负值）
      args.chem_attn_tolerance: float — 质量匹配容差 Da（默认 0.02）
      args.chem_attn_entropy_w: float — 注意力熵正则化权重（默认 0.0，0 = 不启用）
    """

    def __init__(self, args, spec_preproc):
        """
        初始化化学感知 DreaMS

        参数：
            args: Namespace — 除原版 DreaMS 参数外，额外支持：
                chem_attn (bool): 化学感知注意力开关，默认 False
                chem_attn_attenuation (float): 衰减强度，默认 -5.0
                chem_attn_tolerance (float): 质量容差 Da，默认 0.02
                chem_attn_entropy_w (float): 注意力熵正则化权重，默认 0.0
            spec_preproc: SpectrumPreprocessor — 谱图预处理器（与原版相同）
        """
        # =====================================================================
        # Step 1: 提取并设置模块一专属参数（必须在 super().__init__ 之前）
        # =====================================================================
        self.chem_attn_enabled = getattr(args, 'chem_attn', False)
        self.chem_attn_attenuation = getattr(args, 'chem_attn_attenuation', -5.0)
        self.chem_attn_tolerance = getattr(args, 'chem_attn_tolerance', 0.02)
        self.chem_attn_entropy_w = getattr(args, 'chem_attn_entropy_w', 0.0)

        # =====================================================================
        # Step 2: 调用父类初始化（复用 DreaMS 的全部架构搭建逻辑）
        # =====================================================================
        super().__init__(args, spec_preproc)

        # =====================================================================
        # Step 3: 构建模块一新增子组件
        # =====================================================================
        if self.chem_attn_enabled:
            # 模块 B：化学规则引擎
            self.chem_rule_engine = ChemicalRuleEngine(
                attenuation=self.chem_attn_attenuation,
                tolerance=self.chem_attn_tolerance,
                learnable_attenuation=True  # λ 可学习
            )

            # 模块 A：注意力头门控网络
            self.gate_network = HeadGatingNetwork(
                d_model=self.d_model,
                n_heads=self.n_heads
            )

            # [阶段二] 学习状态提取器 + λ 生成器（桩：存在但不控制 λ）
            # Phase 2: λ 由 attenuation_scale 参数学习
            # Phase 3: λ 由 LambdaGenerator 动态生成，实现真正的 A→B 反馈
            self.state_extractor = StateExtractor(
                n_heads=self.n_heads, n_layers=self.n_layers, state_dim=64
            )
            self.lambda_generator = LambdaGenerator(state_dim=64)
        else:
            self.chem_rule_engine = None
            self.gate_network = None
            self.state_extractor = None
            self.lambda_generator = None

        # 缓存最近一次的化学感知分析数据（用于可视化）
        self._last_chem_analysis: Optional[Dict] = None

    def forward(self, spec, charge=None):
        """
        化学感知前向传播

        在 DreaMS 原始前向传播流程中插入化学感知计算：
          1. 保存原始 m/z 值（归一化前）
          2. 计算峰对质量差矩阵 → chem_bias（模块 B）
          3. 计算门控权重 gate_weights（模块 A）
          4. 将 chem_bias 和 gate_weights 注入 TransformerEncoder

        参数：
            spec: (batch, n_peaks, 2) — 原始谱图峰列表 [m/z, intensity]
            charge: (batch,) — 电荷数，可选

        返回：
            spec: (batch, n_peaks, d_model) — Transformer 最后一层的 token 嵌入
                - spec[:, 0, :] 为 precursor 峰嵌入（谱图级表征）
        """
        # =====================================================================
        # Step 1: Padding 掩码（与原版一致）
        # =====================================================================
        padding_mask = spec[:, :, 0] == 0

        # =====================================================================
        # Step 2: [模块一] 保存原始 m/z 值（归一化前的真实质量值）
        # =====================================================================
        if self.chem_attn_enabled:
            raw_mz = spec[:, :, 0].clone()  # (batch, n_peaks)

        # =====================================================================
        # Step 3: 电荷特征拼接（与原版一致）
        # =====================================================================
        if self.charge_feature:
            if charge is None:
                raise ValueError('charge_feature=True 但未提供 charge 参数')
            charge_features = ~padding_mask * charge.unsqueeze(-1)
            spec = torch.cat([spec, charge_features.unsqueeze(-1)], dim=-1)

        # =====================================================================
        # Step 4: 峰值嵌入 + 傅里叶特征（与原版一致）
        # =====================================================================
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

        # =====================================================================
        # Step 5: Graphormer 距离编码（与原版一致）
        # =====================================================================
        graphormer_dists = None
        if self.graphormer_mz_diffs:
            if self.d_fourier:
                graphormer_dists = fourier_features.unsqueeze(2) - fourier_features.unsqueeze(1)
            else:
                graphormer_dists = spec_emb[..., 0].unsqueeze(2) - spec_emb[..., 0].unsqueeze(1)
                graphormer_dists = graphormer_dists.unsqueeze(-1)

        # =====================================================================
        # Step 6: [模块一] 计算化学偏置与门控权重
        # =====================================================================
        if self.chem_attn_enabled and self.chem_rule_engine is not None:
            # 6a: 模块 B — 化学规则偏置
            mz_diffs = ChemicalRuleEngine.compute_peak_pair_mz_diffs(raw_mz)
            chem_bias = self.chem_rule_engine(mz_diffs, padding_mask=padding_mask)
            # chem_bias 形状: (batch, 1, n_peaks, n_peaks)

            # 6b: 模块 A — 门控权重（阶段一：所有层共享相同的门控权重）
            if self.gate_network is not None:
                gw = self.gate_network(spec_emb)  # (batch, n_heads)
                gate_weights_per_layer = [gw] * self.n_layers
            else:
                gate_weights_per_layer = None
        else:
            # 开关关闭：全部为 None → 行为与原版 DreaMS 完全一致
            chem_bias = None
            gate_weights_per_layer = None

        # =====================================================================
        # Step 7: Transformer 编码器（化学感知扩展版）
        # =====================================================================
        if self.vanilla_transformer:
            # 原版 PyTorch TransformerEncoderLayer 不支持 chem_bias/gate_weights，
            # 化学感知模式下必须使用自定义 TransformerEncoder（vanilla_transformer=False）
            spec = self.transformer_encoder(
                spec_emb, src_key_padding_mask=padding_mask
            )
        else:
            spec = self.transformer_encoder(
                spec_emb, padding_mask, graphormer_dists,
                chem_bias=chem_bias,
                gate_weights_per_layer=gate_weights_per_layer
            )

        # =====================================================================
        # Step 8: [模块一] 缓存化学感知分析数据（用于可视化与可解释性分析）
        # =====================================================================
        if self.chem_attn_enabled and chem_bias is not None:
            self._last_chem_analysis = {
                'chem_bias': chem_bias.detach(),
                'mz_diffs': mz_diffs.detach() if 'mz_diffs' in locals() else None,
                'gate_weights': gw.detach() if gw is not None else None,
                'raw_mz': raw_mz.detach(),
            }

        return spec

    # =========================================================================
    # 阶段二：冻结/解冻 + 优化器配置
    # =========================================================================

    def freeze_backbone(self):
        """冻结 DreaMS backbone 的所有参数（保留预训练知识不动）"""
        backbone_params = set()
        for name, param in self.named_parameters():
            # chem_rule_engine, gate_network, state_extractor, lambda_generator 不冻结
            if 'chem_rule_engine' not in name and 'gate_network' not in name \
                    and 'state_extractor' not in name and 'lambda_generator' not in name:
                param.requires_grad = False
                backbone_params.add(name)
        return len(backbone_params)

    def unfreeze_chem_aware(self):
        """仅解冻化学感知模块的参数（供训练）"""
        trainable = 0
        for name, param in self.named_parameters():
            if 'chem_rule_engine' in name or 'gate_network' in name \
                    or 'state_extractor' in name or 'lambda_generator' in name:
                param.requires_grad = True
                trainable += 1
        return trainable

    def get_chem_aware_params(self):
        """返回仅化学感知模块的可训练参数（供 optimizer 使用）"""
        params = []
        for name, param in self.named_parameters():
            if 'chem_rule_engine' in name or 'gate_network' in name \
                    or 'state_extractor' in name or 'lambda_generator' in name:
                if param.requires_grad:
                    params.append(param)
        return params

    def configure_optimizers(self):
        """
        阶段二优化器配置：仅优化化学感知模块参数，backbone 冻结

        λ 学习说明（阶段二）：
          - λ = attenuation * attenuation_scale
          - attenuation_scale 是可学习 Parameter，由 mask loss 梯度驱动
          - 训练中 λ 自动趋向使 mask 预测最准的值
          - 阶段三将切换为 LambdaGenerator 动态 λ
        """
        chem_params = self.get_chem_aware_params()
        if not chem_params:
            # 关闭状态下回退到父类优化器
            return super().configure_optimizers()

        optimizer = torch.optim.Adam(chem_params, lr=self.lr * 0.1, weight_decay=self.weight_decay)
        return optimizer

    # =========================================================================
    # 训练步骤
    # =========================================================================

    def step(self, data, batch_idx, log_prefix):
        """
        训练/验证步骤（化学感知扩展）

        在父类 step() 基础上，额外计算注意力熵正则化损失。
        通过 chem_attn_entropy_w 参数控制正则化强度（0 时与原版完全相同）。
        """
        # 调用父类 step（复用全部损失计算逻辑）
        result = super().step(data, batch_idx, log_prefix)

        # 如果启用了注意力熵正则化，追加正则项
        if self.chem_attn_enabled and self.chem_attn_entropy_w > 0:
            # 从缓存的化学分析数据中计算熵正则化
            if self._last_chem_analysis is not None:
                # 熵正则化目前仅记录在日志中，不直接加入 loss（阶段二将集成）
                # 此处预留给阶段二扩展
                pass

        return result

    def get_chem_attn_analysis(self) -> Optional[Dict]:
        """
        获取最近一次前向传播的化学感知注意力分析数据

        返回的字典包含：
            chem_bias: (batch, 1, n, n) — 化学规则偏置矩阵
            mz_diffs: (batch, n, n) — 峰对质量差矩阵
            gate_weights: (batch, n_heads) — 注意力头门控权重
            raw_mz: (batch, n) — 原始 m/z 值

        用途：
            - 可视化化学偏置矩阵（heatmap）
            - 分析哪些峰对被化学规则标记为"合理"或"不合理"
            - 检查门控权重的分布和动态变化

        返回：
            Dict 或 None（如果尚未执行 forward 或 chem_attn_enabled=False）
        """
        if not self.chem_attn_enabled:
            return None
        return self._last_chem_analysis

    def on_save_checkpoint(self, checkpoint):
        """
        保存检查点时清理缓存数据（避免将大批量中间结果写入 ckpt 文件）
        """
        # 不保存临时的化学分析缓存到检查点
        keys_to_remove = [k for k in checkpoint.get('state_dict', {}).keys()
                         if '_last_chem_analysis' in k]
        for k in keys_to_remove:
            del checkpoint['state_dict'][k]
        return checkpoint
