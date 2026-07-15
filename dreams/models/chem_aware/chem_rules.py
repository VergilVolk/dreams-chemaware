"""
化学规则引擎 (Chemical Rule Engine) — 模块 B [v4(0715) 规则库扩展版]

核心改动（v3 → v4(0715)，任务一 P0 立即注入高可信规则）：
  1. 规则库从 127 条扩展至 ~335 条，并入三个高可信规则源：
     - CompMS2Miner 子结构库（Neut.loss → NL，frag → CF）
     - MS-FINDER 9 条氢重排规则（新建 HR 类别）
     - ESI(+) 常见官能团碎裂规则（教科书/经验）
  2. 规则数据外置到 chem_rules_data.json（数据与代码分离，每条带 source 溯源）；
     _build_rule_list() 优先加载 JSON，缺失时回退到内联基线 127 条。
  3. 新增 HR 类别（match_type='hr_shift'）：编码氢重排数 n_H，
     n_H≠0 匹配"相差 |n_H| 个氢"的峰对，n_H=0 匹配近整数质量差（偶电子规范断裂）。

继承自 v3 的设计（不变）：
  - 惩罚 → 奖励：默认 bias = 0，匹配规则 → 加正向偏置
  - 逐规则独立可学习权重（softplus 参数化，天然非负）
  - 规则库覆盖到的 → 加分；没覆盖到的 → 不扣分（保持 DreaMS 注意力自由）
  - 每条规则独立学习 → 训练完打印"哪些规则有用/没用"即是科学发现

规则类别：NL（中性丢失）| CF（特征碎片）| ISO（同位素）| NR（氮规则）
          | EE（偶电子）| HR（氢重排，v4(0715) 新增）

参考资料：
  - McLafferty, F. W. "Interpretation of Mass Spectra", 4th ed.
  - Kind, T. & Fiehn, O. "Seven Golden Rules" (2007)
  - Tsugawa et al. "Hydrogen Rearrangement Rules", Anal. Chem. 2016 (MS-FINDER)
  - compMS2Miner R package (Substructure_masses)
"""

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Set
from dataclasses import dataclass, field

# 氢原子单同位素质量 (Da)，用于 HR 氢重排规则的质量偏移计算
M_H = 1.0078250319

# 规则库 JSON 路径（与本文件同目录）
_RULES_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chem_rules_data.json")


# ==============================================================================
# 化学先验知识库
# ==============================================================================

@dataclass
class ChemRule:
    """单条化学规则"""
    name: str           # 人类可读名称，如 'NL:H2O', 'CF:tropylium', 'HR:P2_pos'
    category: str       # 'NL' | 'CF' | 'ISO' | 'NR' | 'EE' | 'HR'
    match_type: str     # 'mass_diff' | 'peak_mz' | 'mass_range' | 'parity' | 'mass_diff_range' | 'hr_shift'
    value: float | Tuple[float, float]  # 匹配目标值（HR 规则：带符号氢重排数 n_H）
    source: str = ""    # 规则来源（溯源用），如 'CompMS2Miner', 'Tsugawa 2016'
    meta: dict = field(default_factory=dict)  # 额外元信息（formula, mode, desc...）


def _build_rule_list() -> List[ChemRule]:
    """构建完整规则列表。

    优先从同目录 chem_rules_data.json 加载（任务一 P0 产出的扩展规则库，~335 条，
    含来源溯源）；若 JSON 缺失或损坏，则回退到内联基线规则（127 条），
    保证引擎在任何环境下都能实例化。
    """
    rules = _load_rules_from_json()
    if rules:
        return rules
    print(f"[chem_rules] 未找到/无法加载 {os.path.basename(_RULES_JSON)}，回退到内联基线 127 条规则")
    return _build_baseline_rules()


def _load_rules_from_json() -> List[ChemRule]:
    """从 chem_rules_data.json 加载规则；失败返回空列表。"""
    if not os.path.exists(_RULES_JSON):
        return []
    try:
        with open(_RULES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[chem_rules] 加载 {os.path.basename(_RULES_JSON)} 失败: {e}")
        return []

    rules: List[ChemRule] = []
    for r in data.get("rules", []):
        val = r["value"]
        # mass_range 的 value 是 [lo, hi] → 转 tuple；其余为 float
        if isinstance(val, list):
            val = tuple(float(x) for x in val)
        else:
            val = float(val)
        meta = {k: r[k] for k in ("formula", "mode", "atom", "n_H", "desc",
                                  "target_mass_diff", "ref") if k in r}
        rules.append(ChemRule(name=r["name"], category=r["category"],
                              match_type=r["match_type"], value=val,
                              source=r.get("source", ""), meta=meta))
    return rules


def _build_baseline_rules() -> List[ChemRule]:
    """内联基线规则（127 条），作为 JSON 缺失时的回退。"""
    rules = []

    # --- 中性丢失 (mass_diff) ---
    NEUTRAL_LOSSES = {
        # 小分子丢失（原版）
        'H2O': 18.0106, 'NH3': 17.0265, 'CO': 27.9949,
        'CO2': 43.9898, 'CH2O': 30.0106, 'CH3OH': 32.0262,
        'HCOOH': 46.0055, 'CH3COOH': 60.0211, 'H2S': 33.9877,
        'SO2': 63.9619, 'SO3': 79.9568, 'HCl': 35.9767,
        'HBr': 79.9262, 'HI': 127.9123, 'HCN': 27.0109,
        'H3PO4': 97.9769, 'HCONH2': 45.0215, 'CH3CONH2': 59.0371,
        'C3H7NO': 73.0528, 'H2NCN': 42.0218,
        # 烷基链
        'CH3': 15.0235, 'C2H5': 29.0391, 'C2H4': 28.0313,
        'C3H6': 42.0470, 'C4H8': 56.0626, 'C6H12': 84.0939,
        'CH3CN': 41.0265, 'C5H8': 68.0626,
        # === 新增：II 相代谢 ===
        'glucuronide': 176.0321,         # C6H8O6
        'glucuronide+H2O': 194.0427,     # C6H10O7
        'sulfate': 79.9568,              # SO3（同SO3，但独立权重）
        'glutathione': 307.0838,         # C10H17N3O6S
        'cysteinylglycine': 178.0419,    # C5H10N2O3S
        'glycine_conj': 57.0215,         # C2H3NO
        'taurine_conj': 107.0041,        # C2H5NO2S
        'acetyl_conj': 42.0106,          # C2H2O（区别于ketene）
        'methylation': 14.0157,          # CH2
        'dimethylation': 28.0313,        # C2H4（同C2H4，独立权重）
        # === 新增：氨基酸残基丢失（肽键断裂） ===
        'Gly_res': 57.0215,              # 甘氨酸
        'Ala_res': 71.0371,              # 丙氨酸
        'Ser_res': 87.0320,              # 丝氨酸
        'Pro_res': 97.0528,              # 脯氨酸
        'Val_res': 99.0684,              # 缬氨酸
        'Thr_res': 101.0477,             # 苏氨酸
        'Leu_Ile_res': 113.0841,         # 亮氨酸/异亮氨酸
        'Asn_res': 114.0429,             # 天冬酰胺
        'Asp_res': 115.0269,             # 天冬氨酸
        'Gln_res': 128.0586,             # 谷氨酰胺
        'Glu_res': 129.0426,             # 谷氨酸
        'Met_res': 131.0405,             # 甲硫氨酸
        'His_res': 137.0589,             # 组氨酸
        'Phe_res': 147.0684,             # 苯丙氨酸
        'Arg_res': 156.1011,             # 精氨酸
        'Tyr_res': 163.0633,             # 酪氨酸
        'Trp_res': 186.0793,             # 色氨酸
        'Lys_res': 128.0950,             # 赖氨酸
        # === 新增：脂质相关 ===
        'palmitic_acid': 256.2402,       # C16H32O2
        'oleic_acid': 282.2559,          # C18H34O2
        'stearic_acid': 284.2715,        # C18H36O2
        'linoleic_acid': 280.2402,       # C18H32O2
        'phosphocholine': 183.0660,      # C5H15NO4P (head group)
        'phosphoethanolamine': 141.0191, # C2H8NO4P (head group)
        'glycerol': 92.0473,            # C3H8O3
        # === 新增：其他常见丢失 ===
        'HNCO': 43.0058,                # 异氰酸
        'C2H2O': 42.0106,               # 乙烯酮
        'C3H6O': 58.0419,               # 丙酮
        'HNCS': 59.0068,                # 异硫氰酸
        'CH4O2': 48.0211,               # 过氧化氢? no, methanediol
    }
    for name, mass in sorted(NEUTRAL_LOSSES.items(), key=lambda x: x[1]):
        rules.append(ChemRule(name=f'NL:{name}', category='NL',
                      match_type='mass_diff', value=float(mass)))

    # --- 特征碎片离子 (peak_mz) ---
    CHAR_FRAGMENTS = {
        # 原版
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
        # === 新增：更多 immonium 离子 ===
        'immonium_P': [70.0651],
        'immonium_T': [74.0600],
        'immonium_N': [87.0553],
        'immonium_D': [88.0393],
        'immonium_Q': [101.0715],
        'immonium_E': [102.0555],
        'immonium_L_I': [86.0964],   # 同 immonium_K，独立权重
        # === 新增：脂质特征碎片 ===
        'phosphocholine_head': [184.0733],   # C5H15NO4P+
        'acyl_glycerol': [339.2899],          # C21H39O3+
        'cholesterol_skel': [369.3516],       # C27H45+
        # === 新增：糖/糖苷特征 ===
        'hexose_oxonium_2': [145.0495],       # 脱水中性糖
        'GlcA_oxonium': [177.0395],           # 葡萄糖醛酸
        'NeuAc_oxonium': [292.1030],          # 唾液酸
        'N_acetylhexosamine': [204.0866],     # C8H14NO5+
        # === 新增：药物/天然产物 ===
        'dimethylanilinium': [122.0964],      # C8H12N+
        'methylpiperidinium': [98.0964],       # C6H12N+
        'phenethylamine': [121.0650],          # C8H9N+? 不对, 应该是136
        'amphetamine_frag': [91.0542],         # tropylium, 同原版
        'caffeine_frag': [138.0662],           # C6H8N3O+
        # === 新增：核苷 ===
        'ribose_frag': [133.0495],             # 同 pentose
        'adenine_frag': [136.0618],            # C5H4N5+
        'uracil_frag': [113.0346],             # C4H3N2O2+
        # === 新增：常见污染物/加合物 ===
        'phthalate': [149.0233],               # C8H5O3+
        'phthalate_2': [167.0339],             # C8H7O4+
        'PEG_frag': [89.0597],                 # C4H9O2+ (PEG系列)
        'PEG_frag_2': [133.0859],              # C6H13O3+
        'PEG_frag_3': [177.1121],              # C8H17O4+
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
        # 新增
        'Si28_Si29': (0.9990, 1.0000),    # M/M+1 (~1 Da)
        'Si28_Si30': (1.9960, 1.9980),    # M/M+2 (~2 Da)
        'Se80_Se78': (1.9980, 2.0020),    # ⁷⁸Se/⁸⁰Se
        'Se80_Se82': (1.9960, 2.0000),    # ⁸⁰Se/⁸²Se
        'K39_K41':   (1.9980, 2.0020),    # ³⁹K/⁴¹K (加合物)
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

    # 类别名称（v4(0715) 新增 HR — 氢重排规则）
    CATEGORY_NAMES = ['NL', 'CF', 'ISO', 'NR', 'EE', 'HR']

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
        self._hr_rules: List[Tuple[int, ChemRule]] = []          # v4(0715): 氢重排规则

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
            elif rule.match_type == 'hr_shift':
                self._hr_rules.append((idx, rule))

        n_rules = len(self.rules)

        # ---- 逐规则独立可学习权重（v3 核心） ----
        # softplus(0.0) ≈ 0.693 → 初始权重约 0.69（足够影响 softmax 决策）
        # 之前的 -3.0 → 0.05 太小，预训练模型的注意力 logits 完全不变 → grad=0
        self.rule_weights_raw = nn.Parameter(torch.full((n_rules,), 0.0))

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

        # hr_shift 类（v4(0715) 氢重排）：value 存带符号氢重排数 n_H
        #   - hr_targets = |n_H| × M_H（n_H≠0 时匹配"相差 n 个氢"的峰对）
        #   - hr_is_zero = (n_H == 0)（匹配近整数质量差 = 偶电子规范断裂）
        if self._hr_rules:
            hr_indices, hr_rules = zip(*self._hr_rules)
            self.register_buffer('hr_indices', torch.tensor(hr_indices, dtype=torch.long))
            self.register_buffer('hr_targets', torch.tensor(
                [abs(float(r.value)) * M_H for r in hr_rules], dtype=torch.float32))
            self.register_buffer('hr_is_zero', torch.tensor(
                [float(r.value) == 0.0 for r in hr_rules], dtype=torch.bool))
        else:
            self.register_buffer('hr_indices', torch.tensor([], dtype=torch.long))
            self.register_buffer('hr_targets', torch.tensor([], dtype=torch.float32))
            self.register_buffer('hr_is_zero', torch.tensor([], dtype=torch.bool))

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
    # 规则匹配向量 — 用于对比学习（谱图间规则重叠度计算）
    # =========================================================================

    @torch.no_grad()
    def get_rule_match_vectors(
        self,
        mz_diffs: torch.Tensor,
        mz_values: Optional[torch.Tensor] = None,
        precursor_mz: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        categories: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        计算每张谱图中每条规则是否命中（二进制向量），用于谱图间规则重叠度计算。

        参数同 forward()，categories 可筛选规则类别。

        返回：
            match_vecs: (batch, n_rules) — 每张谱图中每条规则是否至少命中一次
        """
        batch, n, _ = mz_diffs.shape
        device = mz_diffs.device

        if categories is not None:
            active_cats = set(categories) & self.enabled_categories
        else:
            active_cats = self.enabled_categories

        match_vecs = torch.zeros(batch, len(self.rules), device=device)

        # --- mass_diff 规则 ---
        if len(self.md_targets) > 0 and 'NL' in active_cats:
            diffs_expanded = mz_diffs.unsqueeze(1)
            targets = self.md_targets.view(1, -1, 1, 1)
            match_md = (torch.abs(diffs_expanded - targets) < self.tolerance).any(dim=-1).any(dim=-1)  # (batch, n_md)
            match_vecs[:, self.md_indices] = match_md.float()

        # --- peak_mz 规则 ---
        if len(self.pm_targets) > 0 and 'CF' in active_cats:
            if mz_values is not None:
                mz_expanded = mz_values.unsqueeze(1)
                pm_t = self.pm_targets.view(1, -1, 1)
                match_pm = (torch.abs(mz_expanded - pm_t) < self.tolerance).any(dim=-1)  # (batch, n_pm)
                match_vecs[:, self.pm_indices] = match_pm.float()

        # --- mass_range 规则 ---
        if len(self.mr_ranges) > 0 and 'ISO' in active_cats:
            if padding_mask is not None:
                valid_mask = (~padding_mask).float().unsqueeze(1).unsqueeze(-1) * \
                             (~padding_mask).float().unsqueeze(1).unsqueeze(-2)
            else:
                valid_mask = torch.ones_like(mz_diffs).unsqueeze(1)
            diffs_expanded = mz_diffs.unsqueeze(1)
            lo = self.mr_ranges[:, 0].view(1, -1, 1, 1)
            hi = self.mr_ranges[:, 1].view(1, -1, 1, 1)
            match_mr = ((diffs_expanded >= lo) & (diffs_expanded <= hi) & (valid_mask > 0)).any(dim=-1).any(dim=-1)
            match_vecs[:, self.mr_indices] = match_mr.float()

        # --- parity 规则 (NR) ---
        if len(self._parity_rules) > 0 and 'NR' in active_cats:
            if precursor_mz is not None:
                prec_parity = (precursor_mz.round().long() % 2).view(-1, 1, 1, 1).float()
                diff_parity = (mz_diffs.round().long() % 2).unsqueeze(1).float()
                consistent = (prec_parity == diff_parity).any(dim=-1).any(dim=-1)  # (batch, 1)
                for j, (idx, _) in enumerate(self._parity_rules):
                    match_vecs[:, idx] = consistent[:, 0].float()

        # --- mass_diff_range 规则 (EE) ---
        if len(self._mass_diff_range_rules) > 0 and 'EE' in active_cats:
            lo_val, hi_val = self._mass_diff_range_rules[0][1].value
            not_too_small = ((mz_diffs > hi_val) | (mz_diffs < lo_val)).any(dim=-1).any(dim=-1)  # (batch,)
            for j, (idx, _) in enumerate(self._mass_diff_range_rules):
                match_vecs[:, idx] = not_too_small.float()

        # --- hr_shift 规则 (HR, v4(0715)) ---
        if len(self.hr_indices) > 0 and 'HR' in active_cats:
            diffs_expanded = mz_diffs.unsqueeze(1)                     # (batch, 1, n, n)
            tgt = self.hr_targets.view(1, -1, 1, 1)
            match_nonzero = (torch.abs(diffs_expanded - tgt) < self.tolerance)
            near_int = ((torch.abs(mz_diffs - mz_diffs.round()) < self.tolerance)
                        & (mz_diffs >= 12.0)).unsqueeze(1)
            is_zero = self.hr_is_zero.view(1, -1, 1, 1)
            match_hr = torch.where(is_zero, near_int.expand_as(match_nonzero),
                                   match_nonzero).any(dim=-1).any(dim=-1)  # (batch, n_hr)
            match_vecs[:, self.hr_indices] = match_hr.float()

        return match_vecs

    @staticmethod
    def compute_rule_overlap(match_vecs_A: torch.Tensor, match_vecs_B: torch.Tensor) -> torch.Tensor:
        """
        计算两张（或多张）谱图之间的规则 Jaccard 重叠度。

        参数：
            match_vecs_A: (n_rules,) 或 (batch, n_rules)
            match_vecs_B: (n_rules,) 或 (batch, n_rules)

        返回：
            overlap: 标量或 (batch,) — Jaccard 相似度 [0, 1]
        """
        intersection = (match_vecs_A * match_vecs_B).sum(dim=-1).float()
        union = ((match_vecs_A + match_vecs_B) > 0).float().sum(dim=-1)
        return intersection / union.clamp(min=1)

    # =========================================================================
    # 前向传播
    # =========================================================================

    def forward(
        self,
        mz_diffs: torch.Tensor,
        mz_values: Optional[torch.Tensor] = None,
        precursor_mz: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        categories: Optional[List[str]] = None,
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

        # ---- 类别过滤：None = 所有启用类别 ----
        if categories is not None:
            active_cats = set(categories) & self.enabled_categories
        else:
            active_cats = self.enabled_categories

        # ---- 初始化：全零（不惩罚任何峰对） ----
        chem_bias = torch.zeros(batch, 1, n, n, device=device, dtype=torch.float32)
        self._last_stats = {}
        _debug = not hasattr(self, '_debug_done') or self._debug_done < 3

        # =================================================================
        # mass_diff 类规则（中性丢失）：批量检查 |mz_diff - target| < tol
        # =================================================================
        if len(self.md_targets) > 0 and 'NL' in active_cats:
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
        if len(self.pm_targets) > 0 and mz_values is not None and 'CF' in active_cats:
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
        if len(self.mr_ranges) > 0 and 'ISO' in active_cats:
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
        if len(self._parity_rules) > 0 and precursor_mz is not None and 'NR' in active_cats:
            for idx, rule in self._parity_rules:
                prec_parity = (precursor_mz.round().long() % 2).view(-1, 1, 1, 1).float()
                diff_parity = (mz_diffs.round().long() % 2).unsqueeze(1).float()
                consistent = (prec_parity == diff_parity).float()
                chem_bias = chem_bias + consistent * w[idx]
            self._last_stats['parity_consistency'] = consistent.float().mean().item() if 'consistent' in dir() else 0.0

        # =================================================================
        # mass_diff_range 类规则（偶电子）：质量差不在"太小但不为零"范围 → 加分
        # =================================================================
        if len(self._mass_diff_range_rules) > 0 and 'EE' in active_cats:
            for idx, rule in self._mass_diff_range_rules:
                lo, hi = rule.value
                not_too_small = ((mz_diffs > hi) | (mz_diffs < lo)).float().unsqueeze(1)
                chem_bias = chem_bias + not_too_small * w[idx]
            self._last_stats['mass_diff_range_frac'] = not_too_small.float().mean().item() if 'not_too_small' in dir() else 0.0

        # =================================================================
        # hr_shift 类规则（v4(0715) 氢重排，MS-FINDER 9 条 HR 规则）
        #   n_H≠0 → 峰对质量差 ≈ |n_H|×M_H（相差 n 个氢的碎片对）→ 加分
        #   n_H=0 → 峰对质量差接近整数（偶电子规范断裂、无净氢重排）→ 加分
        # =================================================================
        if len(self.hr_indices) > 0 and 'HR' in active_cats:
            diffs_expanded = mz_diffs.unsqueeze(1)                     # (batch, 1, n, n)
            # 非零氢重排：|d - |n_H|×M_H| < tol
            tgt = self.hr_targets.view(1, -1, 1, 1)                    # (1, n_hr, 1, 1)
            match_nonzero = (torch.abs(diffs_expanded - tgt) < self.tolerance)
            # 零氢重排：|d - round(d)| < tol 且 d ≥ 12（含碳骨架断裂，
            # nominal ≥ C 原子质量；排除对角线及 ±1H/±2H 纯氢簇的重叠）
            near_int = ((torch.abs(mz_diffs - mz_diffs.round()) < self.tolerance)
                        & (mz_diffs >= 12.0)).unsqueeze(1)             # (batch, 1, n, n)
            is_zero = self.hr_is_zero.view(1, -1, 1, 1)                # (1, n_hr, 1, 1)
            match_hr = torch.where(is_zero, near_int.expand_as(match_nonzero),
                                   match_nonzero).float()

            w_hr = w[self.hr_indices]                                  # (n_hr,)
            bias_hr = (match_hr * w_hr.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
            chem_bias = chem_bias + bias_hr

            n_hr_hits = match_hr.sum().item()
            self._last_stats['hr_shift_hits'] = n_hr_hits / max(1, batch * n * n * len(self.hr_indices))

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
            elif rule.match_type == 'hr_shift':
                n_h = float(rule.value)
                if n_h == 0.0:
                    # 零氢重排：近整数质量差（偶电子规范断裂，含碳骨架 nominal≥12）
                    if mz_diff >= 12.0 and abs(mz_diff - round(mz_diff)) < self.tolerance:
                        matched.append(rule.name)
                else:
                    if abs(mz_diff - abs(n_h) * M_H) < self.tolerance:
                        matched.append(rule.name)
        return matched
