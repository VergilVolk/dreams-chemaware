"""
化学规则引擎 (Chemical Rule Engine) — 模块 B 核心组件 [v2 五维规则版]

功能：
  将多维质谱碎裂化学先验转化为注意力偏置矩阵，注入 MultiheadAttention 的 softmax 前计算。

五维规则（每维可独立开关，支持消融实验）：
  1. 中性丢失匹配     — 峰对质量差 = 已知中性丢失质量 → 不衰减
  2. 特征碎片离子识别 — 峰本身的 m/z = 已知碎片离子 → 该峰与所有峰的关联不衰减
  3. 同位素模式       — 峰对形成 ~2Da 同位素簇（Cl/Br/S 的 M/M+2）→ 不衰减
  4. 氮规则           — 母离子偶质量 → 合理碎片差为偶数，奇质量 → 奇数（违反衰减 50%）
  5. 偶电子规则       — ESI 下稳定碎片为偶电子离子，非偶电子碎裂路径轻度衰减

消融实验用法：
  >>> engine = ChemicalRuleEngine(enable_rules=['neutral_loss'])
  >>> engine = ChemicalRuleEngine(enable_rules=['neutral_loss', 'char_fragment', 'isotope'])
  >>> engine = ChemicalRuleEngine()  # 全部 5 维

设计原则：
  - 化学合理的峰对 → bias ≈ 0（不衰减）
  - 化学不合理的峰对 → bias = -λ（可学习衰减，非硬截断）
  - 五维规则独立贡献，互不覆盖（叠加制）

参考资料：
  - McLafferty, F. W. "Interpretation of Mass Spectra", 4th ed.
  - Kind, T. & Fiehn, O. "Seven Golden Rules for heuristic filtering of molecular formulas" (2007)
  - DreaMS 原论文 Fig.3d

作者：module1-chem-attn 开发分支
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Set

# ==============================================================================
# 化学先验知识库
# ==============================================================================

# --- 维度 1：常见中性丢失（正离子模式 ESI）---
# 数据来源：McLafferty + GNPS 社区注释 + HMDB 碎片规律
NEUTRAL_LOSSES: Dict[str, float] = {
    # 小分子丢失
    'H2O':  18.0106,    # 水
    'NH3':  17.0265,    # 氨
    'CO':   27.9949,    # 一氧化碳
    'CO2':  43.9898,    # 二氧化碳
    'CH2O': 30.0106,    # 甲醛
    'CH3OH': 32.0262,   # 甲醇
    'HCOOH': 46.0055,   # 甲酸
    'CH3COOH': 60.0211, # 乙酸
    'H2S':  33.9877,    # 硫化氢
    'SO2':  63.9619,    # 二氧化硫
    'SO3':  79.9568,    # 三氧化硫
    'HCl':  35.9767,    # 氯化氢
    'HBr':  79.9262,    # 溴化氢
    'HI':   127.9123,   # 碘化氢
    'HCN':  27.0109,    # 氰化氢
    'H3PO4': 97.9769,   # 磷酸
    # 氨基酸/肽段相关
    'HCONH2': 45.0215,  # 甲酰胺
    'CH3CONH2': 59.0371,# 乙酰胺
    'C3H7NO': 73.0528,  # 丙酰胺
    'H2NCN': 42.0218,   # 氰胺
    # 烷基链丢失
    'CH3':  15.0235,    # 甲基
    'C2H5': 29.0391,    # 乙基
    'C2H4': 28.0313,    # 乙烯
    'C3H6': 42.0470,    # 丙烯
    'C4H8': 56.0626,    # 丁烯
    'C6H12': 84.0939,   # 己烯
}

# --- 维度 2：特征碎片离子（正离子模式）---
# 特定结构的"指纹"离子——峰 m/z 落在这些值说明存在对应子结构
CHARACTERISTIC_FRAGMENTS: Dict[str, List[float]] = {
    'tropylium_C7H7+': [91.0542],
    'phenyl_C6H5+': [77.0386],
    'benzyl_C7H7O+': [107.0491],
    'benzoyl_C7H5O+': [105.0335],
    'acetyl_CH3CO+': [43.0184],
    'propionyl_C2H5CO+': [57.0340],
    'butyryl_C3H7CO+': [71.0497],
    'immonium_K': [86.0964],
    'immonium_H': [110.0713],
    'immonium_F': [120.0808],
    'immonium_Y': [136.0757],
    'immonium_W': [159.0917],
    'immonium_R': [129.1135],
    'immonium_M': [104.0528],
    'hexose_oxonium': [163.0601],
    'pentose_oxonium': [133.0495],
    'deoxyhexose_oxonium': [147.0652],
    'disaccharide_oxonium': [325.1130],
    'pyridinium': [80.0495, 94.0651],
    'quinolinium': [130.0651],
    'phosphate_frag': [98.9842],
}

# --- 维度 3：同位素模式 ---
# M/M+2 同位素簇质量差 ~2 Da（Cl, Br, S 特征）
ISOTOPE_PATTERNS: Dict[str, Tuple[float, float]] = {
    'Cl35_Cl37': (1.9970, 1.9980),   # ³⁵Cl/³⁷Cl
    'Br79_Br81': (1.9975, 1.9985),   # ⁷⁹Br/⁸¹Br
    'S32_S34':   (1.9955, 1.9970),   # ³²S/³⁴S
}


# ==============================================================================
# 化学规则引擎 v2
# ==============================================================================

class ChemicalRuleEngine(nn.Module):
    """
    化学规则引擎 v2（模块 B 核心 — 五维化学先验 + 消融开关）

    每类规则可独立开关，支持消融实验。所有规则缓冲预存为 torch buffer，
    随模型移动到 GPU，不参与梯度计算。

    参数：
        attenuation: float — 默认衰减强度（负值），默认 -5.0
        tolerance: float — 质量匹配容差 (Da)，默认 0.02
        learnable_attenuation: bool — λ 是否可学习，默认 True
        enable_rules: List[str] | None — 启用的规则列表，None = 全部启用。
            可选值: 'neutral_loss', 'char_fragment', 'isotope', 'nitrogen_rule', 'even_electron'
    """

    AVAILABLE_RULES: List[str] = [
        'neutral_loss', 'char_fragment', 'isotope', 'nitrogen_rule', 'even_electron'
    ]

    def __init__(
        self,
        attenuation: float = -2.0,
        tolerance: float = 0.02,
        learnable_attenuation: bool = True,
        enable_rules: Optional[List[str]] = None
    ):
        super().__init__()
        self.tolerance = tolerance
        self.learnable_attenuation = learnable_attenuation
        self.attenuation = attenuation

        # ---- 规则开关 ----
        if enable_rules is None:
            enable_rules = list(self.AVAILABLE_RULES)
        self.enabled_rules: Set[str] = set(enable_rules)
        for r in self.enabled_rules:
            if r not in self.AVAILABLE_RULES:
                raise ValueError(f'未知规则 "{r}"，可用: {self.AVAILABLE_RULES}')

        # ---- 可学习衰减因子 λ ----
        if learnable_attenuation:
            self.attenuation_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('attenuation_scale', torch.tensor(1.0))

        # ---- 预计算 buffer（不参与梯度，随模型移动） ----
        # 维度 1: 中性丢失质量列表
        self.register_buffer(
            'neutral_masses',
            torch.tensor(sorted(set(NEUTRAL_LOSSES.values())), dtype=torch.float32)
        )
        # 维度 2: 特征碎片 m/z 列表（去重排序）
        all_frag = []
        for mz_list in CHARACTERISTIC_FRAGMENTS.values():
            all_frag.extend(mz_list)
        self.register_buffer(
            'fragment_mz',
            torch.tensor(sorted(set(all_frag)), dtype=torch.float32)
        )
        # 维度 3: 同位素质量差范围 (N_iso, 2)
        iso_pairs = [(float(lo), float(hi)) for lo, hi in ISOTOPE_PATTERNS.values()]
        self.register_buffer('isotope_ranges', torch.tensor(iso_pairs, dtype=torch.float32))

        # 最近一次前向传播的各规则贡献（用于分析和可视化）
        self._last_contributions: Dict[str, torch.Tensor] = {}

    # =====================================================================
    # 内部方法
    # =====================================================================

    def _effective_attenuation(self):
        """返回有效衰减强度。训练时保留计算图梯度，推理时可取 .item()"""
        return self.attenuation * self.attenuation_scale

    def _has_rule(self, name: str) -> bool:
        return name in self.enabled_rules

    # =====================================================================
    # 前向传播
    # =====================================================================

    def forward(
        self,
        mz_diffs: torch.Tensor,
        mz_values: Optional[torch.Tensor] = None,
        precursor_mz: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算五维化学感知注意力偏置矩阵

        参数：
            mz_diffs: (batch, n, n) — 峰对质量差绝对值矩阵
            mz_values: (batch, n) — 各峰 m/z 值（维度 2/5 需要），可选
            precursor_mz: (batch,) — 母离子 m/z（维度 4 氮规则需要），可选
            padding_mask: (batch, n) — padding 掩码（True=填充位），可选

        返回：
            chem_bias: (batch, 1, n, n) — 化学偏置矩阵，可广播到所有注意力头
        """
        batch, n, _ = mz_diffs.shape
        device = mz_diffs.device
        atten = self._effective_attenuation()  # 0-dim tensor with grad
        self._last_contributions = {}

        # ---- 初始化：所有峰对默认衰减 ----
        # 用 ones * atten 替代 torch.full，保持梯度链
        chem_bias = torch.ones(batch, 1, n, n, device=device, dtype=torch.float32) * atten

        # ---- 基础豁免：对角线（自注）+ precursor 行/列 ----
        # 用非原地叠加替代原地赋值，保持梯度
        base_mask = torch.ones(batch, 1, n, n, device=device, dtype=torch.float32)
        diag_idx = torch.arange(n, device=device)
        base_mask[:, :, diag_idx, diag_idx] = 0.0  # 对角线不衰减
        base_mask[:, :, 0, :] = 0.0                 # precursor 行
        base_mask[:, :, :, 0] = 0.0                 # precursor 列

        # =================================================================
        # 维度 1: 中性丢失匹配
        # =================================================================
        if self._has_rule('neutral_loss'):
            for mass in self.neutral_masses:
                hit = torch.abs(mz_diffs - mass) < self.tolerance
                base_mask[hit.unsqueeze(1)] = 0.0
            self._last_contributions['neutral_loss'] = None  # 贡献体现在 base_mask 中

        # =================================================================
        # 维度 2: 特征碎片离子识别
        # =================================================================
        if self._has_rule('char_fragment') and mz_values is not None:
            for frag_mz in self.fragment_mz:
                is_frag = torch.abs(mz_values - frag_mz) < self.tolerance
                base_mask[is_frag.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, n)] = 0.0
                base_mask[is_frag.unsqueeze(1).unsqueeze(-2).expand(-1, 1, n, -1)] = 0.0
            self._last_contributions['char_fragment'] = None

        # =================================================================
        # 维度 3: 同位素模式
        # =================================================================
        if self._has_rule('isotope'):
            for iso_lo, iso_hi in self.isotope_ranges:
                iso_hit = (mz_diffs >= iso_lo) & (mz_diffs <= iso_hi)
                base_mask[iso_hit.unsqueeze(1)] = 0.0
            self._last_contributions['isotope'] = None

        # ---- 应用基础掩码 ----
        chem_bias = chem_bias * base_mask

        # =================================================================
        # 维度 4: 氮规则（额外衰减 50%，叠加在已有 bias 上）
        # =================================================================
        if self._has_rule('nitrogen_rule') and precursor_mz is not None:
            prec_parity = (precursor_mz.round().long() % 2).view(-1, 1, 1, 1)
            diff_parity = (mz_diffs.round().long() % 2).unsqueeze(1)
            violation = (prec_parity != diff_parity).float()
            # 违反位叠加 50% 衰减
            chem_bias = chem_bias + violation * atten * 0.5
            self._last_contributions['nitrogen_rule'] = None

        # =================================================================
        # 维度 5: 偶电子规则
        # =================================================================
        if self._has_rule('even_electron') and mz_values is not None:
            too_small = ((mz_diffs > 0.1) & (mz_diffs < 1.0)).float()
            chem_bias = chem_bias + too_small.unsqueeze(1) * atten * 0.3
            self._last_contributions['even_electron'] = None

        # ---- 后处理：padding 位清零 ----
        if padding_mask is not None:
            pad_mask = (~padding_mask).float().unsqueeze(1).unsqueeze(-1) * \
                       (~padding_mask).float().unsqueeze(1).unsqueeze(-2)
            # pad_mask 形状: (batch, 1, n, n)，直接乘
            chem_bias = chem_bias * pad_mask

        return chem_bias

    # =====================================================================
    # 工具方法
    # =====================================================================

    @staticmethod
    def compute_peak_pair_mz_diffs(mz_values: torch.Tensor) -> torch.Tensor:
        """从 m/z 值计算峰对质量差矩阵: |mz_i - mz_j|"""
        return torch.abs(mz_values.unsqueeze(-1) - mz_values.unsqueeze(-2))

    def get_matched_rules(self, mz_diff: float) -> List[str]:
        """查询单个质量差匹配了哪些已知规则（用于可解释性分析）"""
        matched = []
        if self._has_rule('neutral_loss'):
            for name, mass in NEUTRAL_LOSSES.items():
                if abs(mz_diff - mass) < self.tolerance:
                    matched.append(f'NL:{name}')
        if self._has_rule('isotope'):
            for name, (lo, hi) in ISOTOPE_PATTERNS.items():
                if lo <= mz_diff <= hi:
                    matched.append(f'ISO:{name}')
        return matched

    def get_enabled_rules_summary(self) -> str:
        """返回当前启用的规则摘要"""
        return ', '.join(sorted(self.enabled_rules))

    def get_rule_contributions(self) -> Dict[str, torch.Tensor]:
        """返回最近一次 forward 的各规则贡献（用于消融可视化）"""
        return self._last_contributions
