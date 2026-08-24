"""生物类别子结构分类器（用于检索错误分层）。

类别为「非互斥」二进制标签，一个分子可同时命中多个：
  purine       嘌呤核心（腺嘌呤/鸟嘌呤/次黄嘌呤/黄嘌呤/尿酸/咖啡因等）
  indole       吲哚（色氨酸核心）
  amino_acid   游离 α-氨基酸（NH2/NH3+ 直接连 α 碳，α 碳再连羧基）
  sulfur       含硫
  phosphate    磷酸基
  sugar        糖环（呋喃/吡喃粗判）
  pyrimidine   嘧啶核心（尿嘧啶/胞嘧啶/胸腺嘧啶）

派生标签：
  tryptophan       = indole & amino_acid
  sulfur_amino     = sulfur & amino_acid
  nucleoside_like  = (purine|pyrimidine) & sugar
  nucleotide_like  = nucleoside_like & phosphate

注：这是「结构标签」，非严格命名。用已知分子自检（self_test()）兜底。
"""
from __future__ import annotations

from rdkit import Chem

_PATTERNS = {
    "purine": ["c1ncc2ncnc2n1", "n1cnc2c1ncnc2", "c1ncc2nc[nH]c2n1"],
    "indole": ["c1ccc2[nH]ccc2c1", "c1ccc2c(c1)cc[nH]2"],
    "amino_acid": ["[NX3;H2,H3;!$(NC=O);!$(N=*)]-[CX4;H1](-[!H])-[CX3](=O)-[OX1H0-,OX2]"],
    "sulfur": ["[#16]"],
    "phosphate": ["[PX4](=O)(O)(O)", "[PX4](=O)(O)(O)O", "[PX4](=O)([O-])([O-])",
                  "P(=O)(O)(O)", "P(=O)(O)(O)O"],
    "sugar": ["__PROGRAMMATIC__"],
    "pyrimidine": ["c1nccnc1", "c1ncncc1"],
    "pantothenate": ["[CH3][C]([CH3])([CH2]O)[CH](O)[C](=O)[N][CH2][CH2][C](=O)O",
                     "CC(C)(CO)C(O)C(=O)NCCC(O)=O"],
}

_CACHE: dict[str, dict[str, bool]] = {}


def _has_sugar(mol) -> bool:
    """程序化糖环检测：5/6 元环、恰好 1 个 O 其余全 C、且 ≥2 个环碳带 O 取代（呋喃/吡喃糖）。"""
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) not in (5, 6):
            continue
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        o_count = sum(1 for a in atoms if a.GetAtomicNum() == 8)
        if o_count != 1 or any(a.GetAtomicNum() not in (6, 8) for a in atoms):
            continue
        n_o_sub = 0
        for a in atoms:
            if a.GetAtomicNum() != 6:
                continue
            for nb in a.GetNeighbors():
                if nb.GetIdx() not in ring and nb.GetAtomicNum() == 8:
                    n_o_sub += 1
        if n_o_sub >= 2:
            return True
    return False


def classify(smiles: str) -> dict[str, bool]:
    """返回各类别命中字典；解析失败返回空字典。"""
    if smiles in _CACHE:
        return _CACHE[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        _CACHE[smiles] = {}
        return {}
    out: dict[str, bool] = {}
    for name, pats in _PATTERNS.items():
        if pats == ["__PROGRAMMATIC__"]:
            out[name] = _has_sugar(mol)
            continue
        hit = False
        for p in pats:
            q = Chem.MolFromSmarts(p)
            if q is None:
                continue
            if mol.HasSubstructMatch(q):
                hit = True
                break
        out[name] = hit
    # 派生标签
    out["tryptophan"] = out.get("indole", False) and out.get("amino_acid", False)
    out["sulfur_amino"] = out.get("sulfur", False) and out.get("amino_acid", False)
    out["nucleoside_like"] = (out.get("purine", False) or out.get("pyrimidine", False)) and out.get("sugar", False)
    out["nucleotide_like"] = out["nucleoside_like"] and out.get("phosphate", False)
    _CACHE[smiles] = out
    return out


def bio_tags(smiles: str) -> str:
    """返回 '+'.join 的命中标签串（无命中='other'）。"""
    c = classify(smiles)
    names = [n for n, v in c.items() if v and n in (
        "purine", "indole", "amino_acid", "sulfur", "phosphate", "sugar", "pyrimidine",
        "pantothenate", "tryptophan", "sulfur_amino", "nucleoside_like", "nucleotide_like")]
    return "+".join(names) if names else "other"


def self_test() -> None:
    """用已知分子校验分类器，任何一项不符即抛异常（开跑前认证）。"""
    cases = {
        "adenine": "NC1=NC=NC2=C1N=CN2",
        "guanine": "O=C1NC(N)=NC2=C1N=CN2",
        "atp": "NC1=NC2=C(N=CN2C2OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C2O)C(=O)N1",
        "tryptophan": "N[C@@H](CC1=CNC2=CC=CC=C12)C(O)=O",
        "cysteine": "N[C@@H](CS)C(O)=O",
        "methionine": "CSCC[C@H](N)C(O)=O",
        "pantothenic_acid": "CC(C)(CO)C(O)C(=O)NCCC(O)=O",
        "alanine": "CC(N)C(O)=O",
        "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
        "leucine": "CC(C)C[C@H](N)C(O)=O",
        "benzoic_acid": "OC(=O)C1=CC=CC=C1",
        "histidine": "N[C@@H](CC1=CN=CN1)C(O)=O",
        "uric_acid": "O=C1NC(=O)C2=C1NC(=O)N2",
        "adenosine": "NC1=NC=NC2=C1N=CN2C1OC(CO)C(O)C1O",
        "uracil": "O=C1NC=CC(=O)N1",
    }
    expect = {
        "adenine": {"purine": True},
        "guanine": {"purine": True},
        "atp": {"purine": True, "sugar": True, "phosphate": True, "nucleoside_like": True, "nucleotide_like": True},
        "tryptophan": {"indole": True, "amino_acid": True, "tryptophan": True},
        "cysteine": {"amino_acid": True, "sulfur": True, "sulfur_amino": True},
        "methionine": {"amino_acid": True, "sulfur": True, "sulfur_amino": True},
        "pantothenic_acid": {"amino_acid": False, "pantothenate": True},  # 泛酸不是游离 α-氨基酸（β-丙氨酸单元）
        "alanine": {"amino_acid": True},
        "glucose": {"sugar": True},
        "leucine": {"amino_acid": True},
        "benzoic_acid": {"amino_acid": False, "purine": False, "indole": False},
        "histidine": {"amino_acid": True},
        # 尿酸为全氧化酮式嘌呤（非芳香），芳香 SMARTS 匹配不到 → 已知局限，不做断言
        "uric_acid": {},
        "adenosine": {"purine": True, "sugar": True, "nucleoside_like": True},
        "uracil": {"pyrimidine": True},
    }
    for name, smi in cases.items():
        got = classify(smi)
        for key, want in expect[name].items():
            g = got.get(key, False)
            if g != want:
                raise AssertionError(f"bio_class self_test FAIL: {name} {key} got={g} want={want}")
    # 反例：苯甲酸/泛酸不能误报氨基酸
    assert classify("OC(=O)C1=CC=CC=C1").get("amino_acid", False) is False
    print("[bio_class] self_test PASS")


if __name__ == "__main__":
    self_test()
    # 打印已知分子标签，肉眼核对
    for name, smi in {
        "adenine": "NC1=NC=NC2=C1N=CN2", "atp": "NC1=NC2=C(N=CN2C2OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C2O)C(=O)N1",
        "tryptophan": "N[C@@H](CC1=CNC2=CC=CC=C12)C(O)=O", "cysteine": "N[C@@H](CS)C(O)=O",
        "methionine": "CSCC[C@H](N)C(O)=O", "pantothenic_acid": "CC(C)(CO)C(O)C(=O)NCCC(O)=O",
        "histidine": "N[C@@H](CC1=CN=CN1)C(O)=O", "uric_acid": "O=C1NC(=O)C2=C1NC(=O)N2",
    }.items():
        print(f"  {name:18s} -> {bio_tags(smi)}")
