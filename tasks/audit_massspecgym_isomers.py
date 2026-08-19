"""
Step 0 审计：MassSpecGym train fold 里的同 formula 异构体候选规模。

目的（决定噪声任务数据构建的规模，并判断是否值得跑 MCES）：
  1. train fold 有多少分子（唯一 ik14）、多少 formula 组含 ≥2 个不同 ik14（= 异构体候选）。
  2. 同 formula 异构体对总数（= 负例池上限）。
  3. Morgan Tanimoto 分布（near-isomer 占比，决定 MCES 是否必要）。

只做便宜事（RDKit formula + Morgan fp），不跑 MCES（MCES 昂贵，留到确认候选对之后）。

用法（任意有 rdkit + h5py + numpy 的 conda 环境，CPU 即可）：
  python tasks/audit_massspecgym_isomers.py
  python tasks/audit_massspecgym_isomers.py --max-tani-pairs 2000000 --tanimoto-cap-per-group 300

输出：
  tasks/audit_isomers_summary.json
"""
import argparse
import json
import os
from collections import defaultdict

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HDF5_PATH = os.path.join(ROOT, "data/models/MassSpecGym_MurckoHist_split.hdf5")
OUT_PATH = os.path.join(ROOT, "tasks/audit_isomers_summary.json")


def decode(x):
    return x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=str, default="train")
    ap.add_argument("--max-tani-pairs", type=int, default=2_000_000,
                    help="全局 Tanimoto 对上界（超出则随机抽样）")
    ap.add_argument("--tanimoto-cap-per-group", type=int, default=300,
                    help="每个 formula 组内最多算多少对 Tanimoto")
    args = ap.parse_args()

    print(f"[1] 读取 HDF5: {HDF5_PATH}")
    with h5py.File(HDF5_PATH, "r") as f:
        folds = np.array([decode(x) for x in f["fold"][:]])
        smiles_all = np.array([decode(x) for x in f["smiles"][:]])
        inchi_all = np.array([decode(x) for x in f["INCHIKEY"][:]])
        n_total = len(folds)
    print(f"  全量谱 {n_total} 张")

    mask = folds == args.fold
    smiles_fold = smiles_all[mask]
    inchi_fold = inchi_all[mask]
    print(f"  [{args.fold}] fold 谱 {int(mask.sum())} 张")

    # 唯一 smiles -> 唯一 ik14 -> formula
    print("[2] RDKit 算 formula（按唯一 smiles 去重）...")
    smiles_to_ik = {}
    ik_to_formula = {}
    for smi, ink in zip(smiles_fold, inchi_fold):
        ik14 = ink[:14]
        smiles_to_ik.setdefault(smi, ik14)
        if ik14 not in ik_to_formula:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                ik_to_formula[ik14] = Chem.rdMolDescriptors.CalcMolFormula(mol)

    n_unique_ik = len(ik_to_formula)
    n_unique_smiles = len(smiles_to_ik)
    print(f"  唯一 ik14（有 formula）: {n_unique_ik}；唯一 smiles: {n_unique_smiles}")

    # 同 formula 分组
    print("[3] 按 formula 分组（同 formula 不同 ik14 = 异构体候选）...")
    fm_to_iks = defaultdict(list)
    for ik, fm in ik_to_formula.items():
        fm_to_iks[fm].append(ik)

    n_formula = len(fm_to_iks)
    multi = {fm: iks for fm, iks in fm_to_iks.items() if len(set(iks)) >= 2}
    n_multi = len(multi)
    total_pairs = 0
    group_sizes = []
    for fm, iks in multi.items():
        k = len(set(iks))
        group_sizes.append(k)
        total_pairs += k * (k - 1) // 2
    group_sizes = np.array(sorted(group_sizes)) if group_sizes else np.array([])

    print(f"  formula 总数: {n_formula}")
    print(f"  含 ≥2 个不同 ik14 的 formula 组: {n_multi}")
    print(f"  异构体对总数（负例池上限）: {total_pairs}")
    if len(group_sizes):
        print(f"  组大小: min={group_sizes.min()} max={group_sizes.max()} "
              f"mean={group_sizes.mean():.1f} median={np.median(group_sizes):.0f}")

    # Morgan Tanimoto 分布（仅同 formula 不同 ik14 对，抽样封顶）
    print("[4] Morgan Tanimoto 分布（同 formula 异构体对，抽样）...")
    # 预计算每个 ik 的 fingerprint（缓存）
    ik_to_smi = {ik: None for ik in ik_to_formula}
    # 反向：ik14 -> smiles（取第一个）
    for smi, ik in smiles_to_ik.items():
        if ik in ik_to_smi and ik_to_smi[ik] is None:
            ik_to_smi[ik] = smi

    fp_cache = {}
    for ik in ik_to_formula:
        smi = ik_to_smi.get(ik)
        if smi:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp_cache[ik] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)

    rng = np.random.default_rng(42)
    tani_pairs = []
    n_done = 0
    for fm, iks in multi.items():
        iks = list(set(iks))
        if len(iks) < 2:
            continue
        # 生成组内所有对（不同 ik）
        pair_list = [(iks[i], iks[j]) for i in range(len(iks)) for j in range(i + 1, len(iks))]
        if len(pair_list) > args.tanimoto_cap_per_group:
            idx = rng.choice(len(pair_list), args.tanimoto_cap_per_group, replace=False)
            pair_list = [pair_list[int(i)] for i in idx]
        for a, b in pair_list:
            if a not in fp_cache or b not in fp_cache:
                continue
            t = DataStructs.TanimotoSimilarity(fp_cache[a], fp_cache[b])
            tani_pairs.append(float(t))
            n_done += 1
            if n_done >= args.max_tani_pairs:
                break
        if n_done >= args.max_tani_pairs:
            break

    tani = np.array(tani_pairs) if tani_pairs else np.array([])
    hist = {}
    if len(tani):
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.01]
        labels = ["<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-0.9", "0.9-1.0"]
        counts, _ = np.histogram(tani, bins=bins)
        for lab, c in zip(labels, counts):
            hist[lab] = int(c)
        print(f"  Tanimoto 对 {len(tani)} 对，分布: {hist}")
        print(f"  near-isomer 候选（Tanimoto ≥ 0.8）: {int((tani >= 0.8).sum())} 对 "
              f"({(tani >= 0.8).mean() * 100:.1f}%)")

    summary = {
        "fold": args.fold,
        "n_spectra_fold": int(mask.sum()),
        "n_unique_ik14": n_unique_ik,
        "n_unique_smiles": n_unique_smiles,
        "n_formula": n_formula,
        "n_formula_multi_ik": n_multi,
        "total_isomer_pairs": int(total_pairs),
        "group_size_min": int(group_sizes.min()) if len(group_sizes) else 0,
        "group_size_max": int(group_sizes.max()) if len(group_sizes) else 0,
        "group_size_mean": float(group_sizes.mean()) if len(group_sizes) else 0,
        "group_size_median": float(np.median(group_sizes)) if len(group_sizes) else 0,
        "tanimoto_pairs_computed": int(len(tani)),
        "tanimoto_histogram": hist,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[5] 已保存 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
