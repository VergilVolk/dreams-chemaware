"""
化学规则引擎 (Chemical Rule Engine) — 模块 B [v3 逐规则向量版]

核心改动（v2 → v3）：
  1. 惩罚 → 奖励：默认 bias = 0（不惩罚任何峰对），匹配规则 → 加正向偏置
  2. 标量 λ → 逐规则向量：每条具体规则一个独立可学习参数（~55 条规则 ~55 个权重）
     - H₂O 丢失和 CO 丢失不再被迫共用一个权重
     - 好规则自动涨，坏规则自动降到 0，互不拖累
  3. Softplus 参数化：权重天然非负，梯度处处可导
  4. 配合"仅最后一层注入"策略，消除跨层复合放大

规则清单（~55 条）：
  - 28 条中性丢失（H₂O, NH₃, CO, CO₂, CH₃OH, ...）
  - 22 条特征碎片离子（苯基, tropylium, immoniums, 糖碎片...）
  - 3 条同位素模式（Cl, Br, S 的 M/M+2）
  - 1 条氮规则
  - 1 条偶电子规则

设计原则：
  - 规则库覆盖到的 → 加分（模型可学习该规则是否可靠）
  - 规则库没覆盖到的 → 不扣分（保持 DreaMS 原有注意力自由）
  - 每条规则独立学习 → 训练完打印"哪些规则有用/没用"本身就是有意义的科学发现

参考资料：
  - McLafferty, F. W. "Interpretation of Mass Spectra", 4th ed.
  - Kind, T. & Fiehn, O. "Seven Golden Rules" (2007)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Set
from dataclasses import dataclass


# ==============================================================================
# 化学先验知识库
# ==============================================================================

@dataclass
class ChemRule:
    """单条化学规则"""
    name: str           # 人类可读名称，如 'NL:H2O', 'CF:tropylium'
    category: str       # 'NL' | 'CF' | 'ISO' | 'NR' | 'EE'
    match_type: str     # 'mass_diff' | 'peak_mz' | 'mass_range' | 'parity' | 'mass_diff_range'
    value: float | Tuple[float, float]  # 匹配目标值


def _build_rule_list() -> List[ChemRule]:
    """从知识库构建完整的规则列表"""
    rules = []

    # --- 中性丢失 (mass_diff) ---
    NEUTRAL_LOSSES = {
        'H2O': 18.0106, 'NH3': 17.0265, 'CO': 27.9949,
        'CO2': 43.9898, 'CH2O': 30.0106, 'CH3OH': 32.0262,
        'HCOOH': 46.0055, 'CH3COOH': 60.0211, 'H2S': 33.9877,
        'SO2': 63.9619, 'SO3': 79.9568, 'HCl': 35.9767,
        'HBr': 79.9262, 'HI': 127.9123, 'HCN': 27.0109,
        'H3PO4': 97.9769, 'HCONH2': 45.0215, 'CH3CONH2': 59.0371,
        'C3H7NO': 73.0528, 'H2NCN': 42.0218,
        'CH3': 15.0235, 'C2H5': 29.0391, 'C2H4': 28.0313,
        'C3H6': 42.0470, 'C4H8': 56.0626, 'C6H12': 84.0939,
        'CH3CN': 41.0265, 'C5H8': 68.0626,
    }
    for name, mass in sorted(NEUTRAL_LOSSES.items(), key=lambda x: x[1]):
        rules.append(ChemRule(name=f'NL:{name}', category='NL',
                      match_type='mass_diff', value=float(mass)))

    # --- 特征碎片离子 (peak_mz) ---
    CHAR_FRAGMENTS = {
        'acetyl': [43.0184], 'propionyl': [57.0340], 'butyryl': [71.0497],
        'phenyl': [77.0386], 'pyridinium_lo': [80.0495],
        'immonium_K': [86.0964], 'tropylium': [91.0542],
        'pyridinium_hi': [94.0651], 'phosphate_frag': [98.9842],
        'immonium_M': [104.0528], 'benzoyl': [105.0335],
        'benzyl': [107.0491], 'immonium_H': [110.0713],
        'immonium_F': [120.0808], 'immonium_R': [129.1135],
        'quinolinium': [130.0651], 'pentose_oxonium': [133.0495],
        'immonium_Y': [136.0757], 'deoxyhexose_oxonium': [147.0652],
        'immonium_W': [159.0917], 'hexose_oxonium': [163.0601],
        'disaccharide_oxonium': [325.1130],
    }
    for name, mz_list in sorted(CHAR_FRAGMENTS.items()):
        for mz in mz_list:
            rules.append(ChemRule(name=f'CF:{name}', category='CF',
                          match_type='peak_mz', value=float(mz)))

    # --- 同位素模式 (mass_range) ---
    ISOTOPE_PATTERNS = {
        'Cl35_Cl37': (1.9970, 1.9980),
        'Br79_Br81': (1.9975, 1.9985),
        'S32_S34':   (1.9955, 1.9970),
    }
    for name, (lo, hi) in sorted(ISOTOPE_PATTERNS.items()):
        rules.append(ChemRule(name=f'ISO:{name}', category='ISO',
                      match_type='mass_range', value=(float(lo), float(hi))))

    # --- 氮规则 (parity) ---
    rules.append(ChemRule(name='nitrogen_rule', category='NR',
                  match_type='parity', value=0.0))

    # --- 偶电子规则 (mass_diff_range) ---
    rules.append(ChemRule(name='even_electron', category='EE',
                  match_type='mass_diff_range', value=(0.02, 1.0)))

    return rules


# ==============================================================================
# 化学规则引擎 v3 — 逐规则权重
# ==============================================================================

class ChemicalRuleEngine(nn.Module):
    """
    化学规则引擎 v3 — 奖励式 + 逐规则独立权重

    每条具体规则（H₂O 丢失、CO 丢失、苯基碎片……）各自拥有一个可学习权重。
    权重初始化为 ~0.05（softplus(-3.0)），通过 mask prediction 损失的梯度自然选择。

    参数：
        tolerance: float — 质量匹配容差 (Da)，默认 0.02
        enable_categories: List[str] | None — 启用的规则类别，
            None = 全部启用。可选值: 'NL', 'CF', 'ISO', 'NR', 'EE'
    """

    # 类别名称
    CATEGORY_NAMES = ['NL', 'CF', 'ISO', 'NR', 'EE']

    def __init__(
        self,
        tolerance: float = 0.02,
        enable_categories: Optional[List[str]] = None
    ):
        super().__init__()
        self.tolerance = tolerance

        # ---- 构建规则列表 ----
        all_rules = _build_rule_list()

        # ---- 规则类别开关 ----
        if enable_categories is None:
            enable_categories = list(self.CATEGORY_NAMES)
        self.enabled_categories: Set[str] = set(enable_categories)

        # 过滤出启用的规则
        self.rules: List[ChemRule] = [
            r for r in all_rules if r.category in self.enabled_categories
        ]

        # ---- 按 match_type 分组，便于批量计算 ----
        self._mass_diff_rules: List[Tuple[int, ChemRule]] = []   # (idx, rule)
        self._peak_mz_rules: List[Tuple[int, ChemRule]] = []
        self._mass_range_rules: List[Tuple[int, ChemRule]] = []
        self._parity_rules: List[Tuple[int, ChemRule]] = []
        self._mass_diff_range_rules: List[Tuple[int, ChemRule]] = []

        for idx, rule in enumerate(self.rules):
            if rule.match_type == 'mass_diff':
                self._mass_diff_rules.append((idx, rule))
            elif rule.match_type == 'peak_mz':
                self._peak_mz_rules.append((idx, rule))
            elif rule.match_type == 'mass_range':
                self._mass_range_rules.append((idx, rule))
            elif rule.match_type == 'parity':
                self._parity_rules.append((idx, rule))
            elif rule.match_type == 'mass_diff_range':
                self._mass_diff_range_rules.append((idx, rule))

        n_rules = len(self.rules)

        # ---- 逐规则独立可学习权重（v3 核心） ----
        # softplus(-3.0) ≈ 0.0486 → 初始权重约 0.05
        self.rule_weights_raw = nn.Parameter(torch.full((n_rules,), -3.0))

        # ---- 预计算匹配目标的 buffer ----
        # mass_diff 类：各规则的目标质量值
        if self._mass_diff_rules:
            md_indices, md_rules = zip(*self._mass_diff_rules)
            self.register_buffer('md_indices', torch.tensor(md_indices, dtype=torch.long))
            self.register_buffer('md_targets', torch.tensor(
                [r.value for r in md_rules], dtype=torch.float32))
        else:
            self.register_buffer('md_indices', torch.tensor([], dtype=torch.long))
            self.register_buffer('md_targets', torch.tensor([], dtype=torch.float32))

        # peak_mz 类：各规则的目标 m/z 值
        if self._peak_mz_rules:
            pm_indices, pm_rules = zip(*self._peak_mz_rules)
            self.register_buffer('pm_indices', torch.tensor(pm_indices, dtype=torch.long))
            self.register_buffer('pm_targets', torch.tensor(
                [r.value for r in pm_rules], dtype=torch.float32))
        else:
            self.register_buffer('pm_indices', torch.tensor([], dtype=torch.long))
            self.register_buffer('pm_targets', torch.tensor([], dtype=torch.float32))

        # mass_range 类：各规则的质量范围 (n_iso, 2)
        if self._mass_range_rules:
            mr_indices, mr_rules = zip(*self._mass_range_rules)
            self.register_buffer('mr_indices', torch.tensor(mr_indices, dtype=torch.long))
            self.register_buffer('mr_ranges', torch.tensor(
                [r.value for r in mr_rules], dtype=torch.float32))
        else:
            self.register_buffer('mr_indices', torch.tensor([], dtype=torch.long))
            self.register_buffer('mr_ranges', torch.tensor([], dtype=torch.float32).reshape(0, 2))

        # ---- 缓存 ----
        self._last_stats: Dict[str, float] = {}

    # =========================================================================
    # 工具方法
    # =========================================================================

    def get_rule_weights(self) -> torch.Tensor:
        """返回所有规则的当前有效权重，形状 (n_rules,)"""
        return F.softplus(self.rule_weights_raw)

    def get_rule_weight_dict(self) -> Dict[str, float]:
        """返回 {规则名称: 权重} 字典（用于日志和可视化）"""
        w = self.get_rule_weights()
        return {rule.name: w[i].item() for i, rule in enumerate(self.rules)}

    def get_rule_names(self) -> List[str]:
        """返回所有规则名称列表"""
        return [r.name for r in self.rules]

    def get_rule_weights_by_category(self) -> Dict[str, Dict[str, float]]:
        """返回按类别分组的 {类别: {规则名: 权重}} 字典"""
        w = self.get_rule_weights()
        result: Dict[str, Dict[str, float]] = {}
        for i, rule in enumerate(self.rules):
            if rule.category not in result:
                result[rule.category] = {}
            result[rule.category][rule.name] = w[i].item()
        return result

    def get_enabled_rules_summary(self) -> str:
        """返回当前启用的规则摘要"""
        cats = ', '.join(sorted(self.enabled_categories))
        return f'{len(self.rules)} rules in [{cats}]'

    def get_rule_stats(self) -> Dict[str, float]:
        """返回最近一次 forward 的各规则匹配统计"""
        return self._last_stats

    # =========================================================================
    # 前向传播
    # =========================================================================

    def forward(
        self,
        mz_diffs: torch.Tensor,
        mz_values: Optional[torch.Tensor] = None,
        precursor_mz: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算化学感知注意力偏置矩阵（v3 奖励式 + 逐规则权重）

        参数：
            mz_diffs: (batch, n, n) — 峰对质量差绝对值矩阵
            mz_values: (batch, n) — 各峰 m/z 值（peak_mz 类规则需要），可选
            precursor_mz: (batch,) — 母离子 m/z（parity 类规则需要），可选
            padding_mask: (batch, n) — padding 掩码（True=填充位），可选

        返回：
            chem_bias: (batch, 1, n, n) — 化学偏置矩阵（非负值，默认全 0）
        """
        batch, n, _ = mz_diffs.shape
        device = mz_diffs.device

        # ---- 有效权重（softplus 保持非负） ----
        w = F.softplus(self.rule_weights_raw)  # (n_rules,)

        # ---- 初始化：全零（不惩罚任何峰对） ----
        chem_bias = torch.zeros(batch, 1, n, n, device=device, dtype=torch.float32)
        self._last_stats = {}
        _debug = not hasattr(self, '_debug_done') or self._debug_done < 3

        # =================================================================
        # mass_diff 类规则（中性丢失）：批量检查 |mz_diff - target| < tol
        # =================================================================
        if len(self.md_targets) > 0:
            # mz_diffs: (batch, n, n), md_targets: (n_md,)
            # → match: (batch, n_md, n, n)
            diffs_expanded = mz_diffs.unsqueeze(1)  # (batch, 1, n, n)
            targets = self.md_targets.view(1, -1, 1, 1)  # (1, n_md, 1, 1)
            match_md = (torch.abs(diffs_expanded - targets) < self.tolerance).float()

            # [DEBUG] 统计每条规则的匹配数
            if _debug:
                per_rule_hits = match_md.sum(dim=(0, 2, 3))  # (n_md,)
                matched_rules = []
                for j, (idx, rule) in enumerate(self._mass_diff_rules):
                    nh = per_rule_hits[j].item()
                    if nh > 0:
                        matched_rules.append(f'{rule.name}={int(nh)}')
                if matched_rules:
                    print(f'[chem_rules] mass_diff matches: {matched_rules[:10]}')
                else:
                    print(f'[chem_rules] mass_diff: ZERO matches across all {len(self._mass_diff_rules)} rules')

            # 加权求和: (batch, n_md, n, n) → (batch, 1, n, n)
            w_md = w[self.md_indices]  # (n_md,)
            bias_md = (match_md * w_md.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
            chem_bias = chem_bias + bias_md

            n_hits = match_md.sum().item()
            self._last_stats['mass_diff_hits'] = n_hits / max(1, batch * n * n * len(self.md_targets))

        # =================================================================
        # peak_mz 类规则（特征碎片）：检查 |peak_mz - target| < tol
        # 匹配的峰 → 其所在行和列全部加分
        # =================================================================
        if len(self.pm_targets) > 0 and mz_values is not None:
            # mz_values: (batch, n), pm_targets: (n_pm,)
            # → is_frag: (batch, n_pm, n) — 每个峰是否匹配每个碎片规则
            mz_expanded = mz_values.unsqueeze(1)  # (batch, 1, n)
            pm_t = self.pm_targets.view(1, -1, 1)  # (1, n_pm, 1)
            is_frag = (torch.abs(mz_expanded - pm_t) < self.tolerance).float()  # (batch, n_pm, n)

            # 行 + 列展开: (batch, n_pm, n, n)
            cf_row = is_frag.unsqueeze(-1).expand(-1, -1, -1, n)
            cf_col = is_frag.unsqueeze(-2).expand(-1, -1, n, -1)
            cf_match = (cf_row + cf_col).clamp(0, 1)  # 避免重复计数

            # 加权求和
            w_pm = w[self.pm_indices]  # (n_pm,)
            bias_cf = (cf_match * w_pm.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
            chem_bias = chem_bias + bias_cf

            n_cf_hits = cf_match.sum().item()
            self._last_stats['peak_mz_hits'] = n_cf_hits / max(1, batch * n * n * len(self.pm_targets))

        # =================================================================
        # mass_range 类规则（同位素）：检查 mz_diff ∈ [lo, hi]
        # =================================================================
        if len(self.mr_ranges) > 0:
            # mr_ranges: (n_mr, 2), mz_diffs: (batch, n, n)
            diffs_expanded = mz_diffs.unsqueeze(1)  # (batch, 1, n, n)
            lo = self.mr_ranges[:, 0].view(1, -1, 1, 1)  # (1, n_mr, 1, 1)
            hi = self.mr_ranges[:, 1].view(1, -1, 1, 1)
            match_mr = ((diffs_expanded >= lo) & (diffs_expanded <= hi)).float()

            w_mr = w[self.mr_indices]  # (n_mr,)
            bias_mr = (match_mr * w_mr.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
            chem_bias = chem_bias + bias_mr

            n_iso_hits = match_mr.sum().item()
            self._last_stats['mass_range_hits'] = n_iso_hits / max(1, batch * n * n * len(self.mr_ranges))

        # =================================================================
        # parity 类规则（氮规则）：奇偶一致性 → 加分
        # =================================================================
        if len(self._parity_rules) > 0 and precursor_mz is not None:
            for idx, rule in self._parity_rules:
                prec_parity = (precursor_mz.round().long() % 2).view(-1, 1, 1, 1).float()
                diff_parity = (mz_diffs.round().long() % 2).unsqueeze(1).float()
                consistent = (prec_parity == diff_parity).float()
                chem_bias = chem_bias + consistent * w[idx]
            self._last_stats['parity_consistency'] = consistent.float().mean().item() if 'consistent' in dir() else 0.0

        # =================================================================
        # mass_diff_range 类规则（偶电子）：质量差不在"太小但不为零"范围 → 加分
        # =================================================================
        if len(self._mass_diff_range_rules) > 0:
            for idx, rule in self._mass_diff_range_rules:
                lo, hi = rule.value
                not_too_small = ((mz_diffs > hi) | (mz_diffs < lo)).float().unsqueeze(1)
                chem_bias = chem_bias + not_too_small * w[idx]
            self._last_stats['mass_diff_range_frac'] = not_too_small.float().mean().item() if 'not_too_small' in dir() else 0.0

        # ---- Padding 位清零 ----
        if padding_mask is not None:
            pad_mat = (~padding_mask).float().unsqueeze(1).unsqueeze(-1) * \
                      (~padding_mask).float().unsqueeze(1).unsqueeze(-2)
            chem_bias = chem_bias * pad_mat

        # [DEBUG] 递增调试计数器
        if _debug:
            self._debug_done = getattr(self, '_debug_done', 0) + 1

        return chem_bias

    # =========================================================================
    # 静态工具
    # =========================================================================

    @staticmethod
    def compute_peak_pair_mz_diffs(mz_values: torch.Tensor) -> torch.Tensor:
        """从 m/z 值计算峰对质量差矩阵: |mz_i - mz_j|"""
        return torch.abs(mz_values.unsqueeze(-1) - mz_values.unsqueeze(-2))

    def get_matched_rules(self, mz_diff: float) -> List[str]:
        """查询单个质量差匹配了哪些已知规则（用于可解释性分析）"""
        matched = []
        for idx, rule in enumerate(self.rules):
            if rule.match_type == 'mass_diff':
                if abs(mz_diff - float(rule.value)) < self.tolerance:
                    matched.append(rule.name)
            elif rule.match_type == 'mass_range':
                lo, hi = rule.value
                if lo <= mz_diff <= hi:
                    matched.append(rule.name)
        return matched
