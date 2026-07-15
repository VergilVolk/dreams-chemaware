"""
build_chem_rules.py — 任务一 (P0) 规则库构建脚本
================================================================================
将三个高可信规则源并入现有 127 条基线规则，去重后输出 chem_rules_data.json：

  数据源 1  氢重排规则 (HR)          9 条   Tsugawa et al., Anal. Chem. 2016 (MS-FINDER)
  数据源 2  CompMS2Miner 子结构库   264 条  compMS2Miner R 包 Substructure_masses
  数据源 3  ESI(+) 官能团碎裂规则    补充    教科书级经验碎裂化学 (McLafferty 等)

流程（对应任务文档 1.4 执行步骤）：
  1. 解析 chem_rules.py 内联的 127 条基线规则（NEUTRAL_LOSSES / CHAR_FRAGMENTS / ISOTOPE_PATTERNS）
  2. 提取 CompMS2Miner：Neut.loss==1 → NL，frag==1 → CF
  3. 编码 9 条 HR 规则为新的 HR 类别
  4. 补充 ESI(+) 常见官能团碎裂规则
  5. 与基线 + 新增内部去重（质量容差 DEDUP_TOL）
  6. 合并输出 chem_rules_data.json（每条带 source 溯源字段）

本脚本零第三方依赖（仅标准库 + ast），可在无 torch 环境运行。

用法:  python build_chem_rules.py
"""

import ast
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHEM_RULES_PY = HERE / "chem_rules.py"
COMPMS2_CSV = HERE / "compms2_substructures.csv"
OUT_JSON = HERE / "chem_rules_data.json"

M_H = 1.0078250319          # 氢原子单同位素质量 (Da)
DEDUP_TOL = 0.005           # 去重容差 (Da)：仅剔除数值上真正重复的规则
NL_MASS_RANGE = (9.0, 600.0)    # 中性丢失质量合理范围
CF_MZ_RANGE = (30.0, 700.0)     # 特征碎片 m/z 合理范围


# ==============================================================================
# 步骤 1：解析基线 127 条规则（直接读 chem_rules.py 内联字典，避免 import torch）
# ==============================================================================

def parse_baseline_rules():
    """用 ast 从 chem_rules.py 的 _build_rule_list() 提取内联知识库字典。

    返回 rules 列表，元素为 dict：{name, category, match_type, value, source}。
    """
    tree = ast.parse(CHEM_RULES_PY.read_text(encoding="utf-8"))

    def _get_dict(func_name, var_name):
        # 全模块搜索该变量（兼容内联字典位于 _build_rule_list 或改造后的
        # _build_baseline_rules；变量名唯一，不依赖所在函数名）
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == var_name:
                        return ast.literal_eval(node.value)
        return {}

    neutral = _get_dict("_build_rule_list", "NEUTRAL_LOSSES")
    frags = _get_dict("_build_rule_list", "CHAR_FRAGMENTS")
    isos = _get_dict("_build_rule_list", "ISOTOPE_PATTERNS")

    src = "baseline (chem_rules.py v3, 127 rules)"
    rules = []
    for name, mass in neutral.items():
        rules.append(dict(name=f"NL:{name}", category="NL",
                          match_type="mass_diff", value=float(mass), source=src))
    for name, mz_list in frags.items():
        for mz in mz_list:
            rules.append(dict(name=f"CF:{name}", category="CF",
                              match_type="peak_mz", value=float(mz), source=src))
    for name, (lo, hi) in isos.items():
        rules.append(dict(name=f"ISO:{name}", category="ISO",
                          match_type="mass_range", value=[float(lo), float(hi)], source=src))
    rules.append(dict(name="nitrogen_rule", category="NR",
                      match_type="parity", value=0.0, source=src))
    rules.append(dict(name="even_electron", category="EE",
                      match_type="mass_diff_range", value=[0.02, 1.0], source=src))
    return rules


# ==============================================================================
# 步骤 2：提取 CompMS2Miner 264 条子结构 → NL / CF
# ==============================================================================

def _clean_name(abbrev, name):
    """从缩写/全名生成规则名后缀（保留字母数字，其余转下划线）。"""
    raw = (abbrev or name or "").strip().strip("[]")
    if not raw:
        raw = (name or "").strip()
    slug = re.sub(r"[^0-9A-Za-z]+", "_", raw).strip("_")
    return slug or "unnamed"


def parse_compms2_rules():
    """读 compms2_substructures.csv，按字段分类为 NL 和 CF 规则。

    - Neut.loss == 1  → NL 类别（match_type=mass_diff，值=丢失中性质量）
    - frag == 1       → CF 类别（match_type=peak_mz，值=碎片离子 m/z）
    一个条目可能同时满足两者 → 生成两条独立规则。
    """
    rows = list(csv.DictReader(COMPMS2_CSV.open(encoding="utf-8")))
    rules = []
    n_skip = 0
    seen_slugs = {}  # 避免同名冲突

    for r in rows:
        mass = float(r["monoisotopic_mass"] or 0)
        shift = float(r.get("mass.shift") or 0)
        neut = r["Neut.loss"].strip() == "1"
        frag = r["frag"].strip() == "1"
        pos = r["pos"].strip() == "1"
        neg = r["neg"].strip() == "1"
        slug = _clean_name(r.get("Abbrev_name"), r.get("name"))
        formula = (r.get("formula") or "").strip()
        ref = (r.get("ref") or "").strip()[:120]
        mode = "+".join([m for m, ok in (("pos", pos), ("neg", neg)) if ok]) or "?"

        if mass <= 0:
            n_skip += 1
            continue

        # 中性丢失：优先用 mass.shift（若合理），否则用 monoisotopic_mass
        if neut:
            val = shift if NL_MASS_RANGE[0] <= shift <= NL_MASS_RANGE[1] else mass
            if NL_MASS_RANGE[0] <= val <= NL_MASS_RANGE[1]:
                base = f"cms2_{slug}"
                key = ("NL", base)
                seen_slugs[key] = seen_slugs.get(key, 0) + 1
                nm = base if seen_slugs[key] == 1 else f"{base}_{seen_slugs[key]}"
                rules.append(dict(
                    name=f"NL:{nm}", category="NL", match_type="mass_diff",
                    value=round(val, 4), source="CompMS2Miner (Substructure_masses)",
                    formula=formula, mode=mode, ref=ref))

        # 特征碎片：carrier 是离子 m/z
        if frag:
            if CF_MZ_RANGE[0] <= mass <= CF_MZ_RANGE[1]:
                base = f"cms2_{slug}"
                key = ("CF", base)
                seen_slugs[key] = seen_slugs.get(key, 0) + 1
                nm = base if seen_slugs[key] == 1 else f"{base}_{seen_slugs[key]}"
                rules.append(dict(
                    name=f"CF:{nm}", category="CF", match_type="peak_mz",
                    value=round(mass, 4), source="CompMS2Miner (Substructure_masses)",
                    formula=formula, mode=mode, ref=ref))

    print(f"[CompMS2Miner] 读入 {len(rows)} 条 → 生成 {len(rules)} 条候选规则（跳过 {n_skip} 条无效质量）")
    return rules


# ==============================================================================
# 步骤 3：9 条 HR 氢重排规则（Tsugawa et al., Anal. Chem. 2016, Figure 1a）
# ==============================================================================
# 规则语义：碎裂键断裂时 charged fragment 相对中性结构 M 的氢数偏移 n_H。
#   n_H != 0 → 匹配"质量差 ≈ |n_H|×m_H"的峰对（相差 n 个氢的碎片对）
#   n_H == 0 → 匹配近整数质量差的峰对（偶电子、无净氢重排的规范断裂）
# value 存带符号氢数 n_H；引擎按上述语义匹配。

HR_RULES = [
    # (代号, 模式, 适用原子, n_H, 说明)
    ("P1", "pos", "C(P,S)",   0,  "正模式初次断裂：C键断裂不加氢（偶电子），P/S通常亦然"),
    ("P2", "pos", "N,O",     +2,  "正模式初次断裂：N/O键断裂加两个氢 [M+2H]+"),
    ("P3", "pos", "N,O",     +1,  "正模式后续断裂：加一个氢中和碎片 [M'+aH]+（N/O更常见）"),
    ("P4", "pos", "C,P",     -1,  "正模式后续断裂：丢一个氢成双键/环 [M'+(a-2)H]+（C/P更常见）"),
    ("N1", "neg", "CNOPS",    0,  "负模式初次断裂：所有元素不招募氢 [M-0H]-"),
    ("N2", "neg", "C,P",     -2,  "负模式初次断裂：C键断裂丢两个氢 [M-2H]-（磷酸亦然）"),
    ("N3", "neg", "S",       -1,  "负模式初次断裂：S键均裂产生奇电子radical（磺酸盐）"),
    ("N4", "neg", "N,O",     +1,  "负模式后续断裂：加一个氢中和碎片（N/O更常见）"),
    ("N5", "neg", "C,P,S",   -2,  "负模式后续断裂：丢一个氢 [M'-(b+2)H]-（C/P/S更常见）"),
]


def build_hr_rules():
    rules = []
    for code, mode, atom, n_h, desc in HR_RULES:
        rules.append(dict(
            name=f"HR:{code}_{mode}",
            category="HR",
            match_type="hr_shift",
            value=float(n_h),                # 带符号氢重排数
            source="Tsugawa et al., Anal. Chem. 2016 (MS-FINDER, 9 HR rules)",
            mode=mode, atom=atom, n_H=n_h, desc=desc,
            target_mass_diff=round(abs(n_h) * M_H, 4)))  # |n_H|×m_H，便于查阅
    print(f"[HR] 编码 {len(rules)} 条氢重排规则")
    return rules


# ==============================================================================
# 步骤 4：ESI(+) 常见官能团碎裂规则补充（教科书/经验碎裂化学）
# ==============================================================================
# 说明：任务文档 1.3 所述"基于>1000张ESI谱图、60+碎裂通道"的原始文献未随包提供，
# 故此处以高置信、教科书级 ESI(+) 官能团碎裂规律补充，并如实标注来源。
# 与 CompMS2Miner / 基线重复的条目将在去重阶段自动剔除。

ESI_SRC = "ESI(+) common functional-group fragmentation (McLafferty / empirical)"

ESI_NEUTRAL_LOSSES = {
    # 含氧
    "2xH2O": 36.0211,          # 双水丢失（多元醇/糖）
    "CH3CHO": 44.0262,         # 乙醛（O-乙基）
    "C2H2": 26.0157,           # 乙炔
    "C3H4O2": 72.0211,         # 丙烯酸/丙二醛
    "C2H4O2": 60.0211,         # 乙酸（酯）
    # 含氮
    "CH3NH2": 31.0422,         # 甲胺
    "C2H7N": 45.0578,          # 二甲胺/乙胺
    "C3H9N": 59.0735,          # 三甲胺/丙胺
    "CH5N3": 59.0483,          # 胍
    "NO": 29.9980,             # 亚硝基（硝基化合物）
    "NO2": 45.9929,            # 硝基
    "HNO2": 47.0007,           # 亚硝酸
    "HNO3": 62.9956,           # 硝酸
    "CH3NCO": 57.0215,         # 甲基异氰酸酯
    # 含硫
    "CH3SH": 48.0034,          # 甲硫醇
    "CH2S": 45.9877,           # 硫代甲醛
    "CH3SOH": 64.0139,         # 亚砜相关
    # 含磷
    "HPO3": 79.9663,           # 偏磷酸
    "H4P2O7": 177.9432,        # 焦磷酸
    # 卤素
    "HF": 20.0062,             # 氟化氢
    "CH3Cl": 49.9923,          # 氯甲烷
    # 糖苷键
    "C6H10O5": 162.0528,       # 己糖丢失（葡萄糖/半乳糖）
    "C5H8O4": 132.0423,        # 戊糖丢失
    "C6H8O4": 144.0423,        # 脱氧己糖相关
    # 芳香/杂环
    "C6H6": 78.0470,           # 苯
    "C5H5N": 79.0422,          # 吡啶
    "C7H8": 92.0626,           # 甲苯
}

ESI_CHAR_FRAGMENTS = {
    # 芳香酰基/苯系
    "hydroxytropylium": [107.0491],     # C7H7O+ 同 benzyl 值，独立权重
    "dihydroxybenzoyl": [137.0233],     # C7H5O3+ 二羟基苯甲酰
    "methoxybenzoyl": [135.0441],       # C8H7O2+
    # 黄酮类（A 环 retro-Diels-Alder 碎片）
    "flavone_1_3A": [153.0182],         # C7H5O4+ 1,3A+ (二羟基)
    "flavone_A_ring": [121.0284],       # C7H5O2+
    # 生物碱/含氮杂环
    "indole_frag": [118.0651],          # C8H8N+ 吲哚亚甲基
    # 核碱基
    "cytosine_frag": [112.0505],        # C4H6N3O+
    "guanine_frag": [152.0567],         # C5H6N5O+
}


def build_esi_rules():
    rules = []
    for name, mass in ESI_NEUTRAL_LOSSES.items():
        rules.append(dict(name=f"NL:esi_{name}", category="NL",
                          match_type="mass_diff", value=float(mass),
                          source=ESI_SRC, mode="pos"))
    for name, mz_list in ESI_CHAR_FRAGMENTS.items():
        for mz in mz_list:
            rules.append(dict(name=f"CF:esi_{name}", category="CF",
                              match_type="peak_mz", value=float(mz),
                              source=ESI_SRC, mode="pos"))
    print(f"[ESI] 补充 {len(rules)} 条 ESI(+) 官能团碎裂规则")
    return rules


# ==============================================================================
# 步骤 5：去重（新增规则 vs 基线 + 新增内部）
# ==============================================================================

def dedup(baseline, new_rules):
    """按 (category, match_type) 分组，用质量容差 DEDUP_TOL 剔除重复。

    基线规则全部保留；新增规则若与"已保留规则"数值重复则丢弃。
    HR / NR / EE 等非数值类不参与质量去重（HR 为全新类别）。
    返回 (保留的新增规则, 丢弃记录)。
    """
    # 已存在的数值目标，按类别归组
    existing = {}  # cat -> list[float]
    for r in baseline:
        if r["match_type"] in ("mass_diff", "peak_mz"):
            existing.setdefault(r["category"], []).append(r["value"])

    kept, dropped = [], []
    for r in new_rules:
        if r["match_type"] not in ("mass_diff", "peak_mz"):
            kept.append(r)   # HR 等直接保留
            continue
        vals = existing.setdefault(r["category"], [])
        dup_of = next((v for v in vals if abs(v - r["value"]) < DEDUP_TOL), None)
        if dup_of is not None:
            dropped.append((r["name"], r["value"], round(dup_of, 4)))
        else:
            vals.append(r["value"])
            kept.append(r)
    return kept, dropped


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    print("=" * 72)
    print("任务一 (P0)：构建 chem_rules_data.json")
    print("=" * 72)

    baseline = parse_baseline_rules()
    print(f"[基线] 解析 {len(baseline)} 条现有规则")

    compms2 = parse_compms2_rules()
    hr = build_hr_rules()
    esi = build_esi_rules()

    # 新增规则合并（顺序：CompMS2Miner → ESI → HR）后去重
    new_all = compms2 + esi + hr
    kept, dropped = dedup(baseline, new_all)
    print(f"[去重] 新增候选 {len(new_all)} 条 → 保留 {len(kept)} 条，剔除重复 {len(dropped)} 条")

    all_rules = baseline + kept

    # 统计
    by_cat = {}
    by_src = {}
    for r in all_rules:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
        s = r["source"].split(" (")[0].split(",")[0]
        by_src[s] = by_src.get(s, 0) + 1

    out = {
        "version": "v4(0715)",
        "task": "P0 — 立即注入高可信规则",
        "n_rules": len(all_rules),
        "baseline_count": len(baseline),
        "added_count": len(kept),
        "dedup_removed": len(dropped),
        "dedup_tolerance_Da": DEDUP_TOL,
        "by_category": by_cat,
        "by_source": by_src,
        "sources": [
            "baseline: chem_rules.py v3 (127 rules)",
            "CompMS2Miner: compMS2Miner R package, Substructure_masses",
            "HR: Tsugawa et al., Anal. Chem. 2016 (MS-FINDER, 9 hydrogen rearrangement rules)",
            "ESI: common ESI(+) functional-group fragmentation (McLafferty / empirical)",
        ],
        "rules": all_rules,
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 72)
    print(f"规则总数: {len(baseline)} → {len(all_rules)}  (+{len(kept)})")
    print(f"按类别: {by_cat}")
    print(f"按来源: {by_src}")
    print(f"已写出: {OUT_JSON.name}")
    if dropped[:8]:
        print(f"去重示例(前8): {dropped[:8]}")


if __name__ == "__main__":
    main()
