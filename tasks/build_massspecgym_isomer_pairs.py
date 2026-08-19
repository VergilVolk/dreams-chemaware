"""
Step 1: 构建 MassSpecGym train fold 的同 formula 异构体对（MCES 分级）。

复用 T1 build_test_cases.py 的 MCES 逻辑（myopic_mces + 阈值 [0,2]/[6,10]/[3,5]），但：
  - 数据源 = MassSpecGym HDF5（不是 annotated01.mgf）。
  - 判据 = MCES（弃用 Tanimoto：Tanimoto 可能把同分子误判成不同分子=灾难性；MCES 量骨架连接差异，同分子差=0）。
  - 同时记录 ik14 -> 谱行索引，供 Step 3 加载真实谱。

用法（任意有 rdkit + h5py + myopic_mces 的 conda 环境）：
  python tasks/build_massspecgym_isomer_pairs.py
  python tasks/build_massspecgym_isomer_pairs.py --workers 8
  python tasks/build_massspecgym_isomer_pairs.py --max-pairs 2000   # 只算前 N 对（测试）

输出：
  tasks/massspecgym_isomers/pairs.json       # 异构体对 + MCES 分级（near/mid/far/uncomputed）
  tasks/massspecgym_isomers/ik_to_rows.json  # ik14 -> HDF5 谱行索引
  tasks/massspecgym_isomers/stats.json       # 统计
"""
import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HDF5_PATH = os.path.join(ROOT, "data/models/MassSpecGym_MurckoHist_split.hdf5")
OUT_DIR = os.path.join(ROOT, "tasks/massspecgym_isomers")

MAX_BONDS_FOR_MCES = 50  # >50 键的分子跳过（MILP 太慢），同 T1


def decode(x):
    return x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)


def _mces_worker(job):
    """模块级 worker：计算一对 SMILES 的 raw MCES（result[1]）。"""
    smi_a, smi_b = job
    try:
        from myopic_mces import MCES
        return MCES(smi_a, smi_b)[1]
    except Exception:
        return None


def main():
    try:
        from myopic_mces import MCES as _MCES  # noqa: F401  # 提前校验，缺库则大声失败
    except ImportError as e:
        raise SystemExit(f"myopic_mces 未安装：{e}\n  服务器需手动复制该包（sbatch DEPENDENCIES 有说明）")

    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=str, default="train")
    ap.add_argument("--workers", type=int, default=8, help="MCES 并行进程数")
    ap.add_argument("--max-pairs", type=int, default=0, help="最多算多少对（0=全部，测试用）")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1. 读 HDF5，取 train fold ----
    print(f"[1] 读取 HDF5: {HDF5_PATH}")
    with h5py.File(HDF5_PATH, "r") as f:
        folds = np.array([decode(x) for x in f["fold"][:]])
        smiles_all = np.array([decode(x) for x in f["smiles"][:]])
        inchi_all = np.array([decode(x) for x in f["INCHIKEY"][:]])
    mask = folds == args.fold
    print(f"  [{args.fold}] fold 谱 {int(mask.sum())} 张")

    # ---- 2. 建 ik14 -> rows / smiles / formula ----
    print("[2] 建 ik14 索引 + RDKit formula...")
    ik_to_rows = defaultdict(list)
    ik_to_smiles = {}
    ik_to_formula = {}
    ik_to_nbonds = {}
    for idx in np.where(mask)[0]:
        ik14 = inchi_all[idx][:14]
        ik_to_rows[ik14].append(int(idx))
        if ik14 not in ik_to_smiles:
            smi = smiles_all[idx]
            ik_to_smiles[ik14] = smi
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                ik_to_formula[ik14] = rdMolDescriptors.CalcMolFormula(mol)
                ik_to_nbonds[ik14] = mol.GetNumBonds()

    n_ik = len(ik_to_smiles)
    print(f"  唯一 ik14: {n_ik}（有 formula: {len(ik_to_formula)}）")

    # ---- 3. 同 formula 分组，枚举异构体对 ----
    print("[3] 同 formula 分组，枚举异构体对...")
    fm_to_iks = defaultdict(set)
    for ik, fm in ik_to_formula.items():
        fm_to_iks[fm].add(ik)

    pairs = []
    for fm, iks in fm_to_iks.items():
        ik_list = sorted(iks)
        if len(ik_list) < 2:
            continue
        for i in range(len(ik_list)):
            for j in range(i + 1, len(ik_list)):
                pairs.append((ik_list[i], ik_list[j], fm))

    print(f"  异构体对总数: {len(pairs)}")
    if args.max_pairs > 0 and len(pairs) > args.max_pairs:
        pairs = pairs[: args.max_pairs]
        print(f"  截断到前 {args.max_pairs} 对（--max-pairs）")

    # ---- 4. 预过滤（键数 > 50 跳过）+ 准备 MCES job ----
    print("[4] 准备 MCES jobs（键数 > 50 跳过）...")
    jobs = []
    skipped_large = 0
    skipped_nosmi = 0
    for a, b, fm in pairs:
        if a not in ik_to_nbonds or b not in ik_to_nbonds:
            skipped_nosmi += 1
            continue
        if ik_to_nbonds[a] > MAX_BONDS_FOR_MCES or ik_to_nbonds[b] > MAX_BONDS_FOR_MCES:
            skipped_large += 1
            continue
        jobs.append((a, b, fm))
    print(f"  可算 MCES 对: {len(jobs)}；跳过大分子 {skipped_large}；无 SMILES {skipped_nosmi}")

    # ---- 5. 并行算 MCES ----
    print(f"[5] 并行算 MCES（{args.workers} workers）...")
    mces_results = []
    if jobs:
        smi_jobs = [(ik_to_smiles[a], ik_to_smiles[b]) for a, b, _ in jobs]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            mces_results = list(ex.map(_mces_worker, smi_jobs, chunksize=32))

    # ---- 6. 分级 + 保存 ----
    print("[6] 分级 + 保存...")
    near, mid, far, uncomputed = [], [], [], []
    for (a, b, fm), mces_raw in zip(jobs, mces_results):
        entry = {
            "ik_a": a, "ik_b": b, "fm": fm,
            "smi_a": ik_to_smiles[a][:120], "smi_b": ik_to_smiles[b][:120],
            "n_bonds_a": ik_to_nbonds.get(a), "n_bonds_b": ik_to_nbonds.get(b),
            "mces_raw": mces_raw,
        }
        if mces_raw is None:
            uncomputed.append(entry)
        elif 0 <= mces_raw <= 2:
            near.append(entry)
        elif 3 <= mces_raw <= 5:
            mid.append(entry)
        elif 6 <= mces_raw <= 10:
            far.append(entry)
        else:
            far.append(entry)  # >10 也归远异构体

    pairs_out = {"near": near, "mid": mid, "far": far, "uncomputed": uncomputed}
    stats = {
        "fold": args.fold,
        "n_unique_ik14": n_ik,
        "n_isomer_pairs": len(pairs),
        "n_mces_computed": len(jobs),
        "skipped_large_mol": skipped_large,
        "n_near": len(near), "n_mid": len(mid), "n_far": len(far),
        "n_uncomputed": len(uncomputed),
    }
    if mces_results:
        vals = [r for r in mces_results if r is not None]
        if vals:
            stats["mces_raw_min"] = min(vals)
            stats["mces_raw_max"] = max(vals)
            stats["mces_raw_mean"] = float(np.mean(vals))
            stats["mces_raw_median"] = float(np.median(vals))

    for fn, data in [("pairs.json", pairs_out), ("stats.json", stats)]:
        with open(os.path.join(OUT_DIR, fn), "w") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "ik_to_rows.json"), "w") as fh:
        json.dump({k: v for k, v in ik_to_rows.items()}, fh, ensure_ascii=False)
    # 完整 SMILES（供预留接口 ② MolFormer/Chemprop 复算结构，及 MCES 复现）
    with open(os.path.join(OUT_DIR, "ik_to_smiles.json"), "w") as fh:
        json.dump(ik_to_smiles, fh, ensure_ascii=False)

    print(f"  near(0-2): {len(near)}  mid(3-5): {len(mid)}  far(6-10+): {len(far)}  uncomputed: {len(uncomputed)}")
    print(f"  已保存 -> {OUT_DIR}")
    print(f"\n=== Step 1 DONE ===")


if __name__ == "__main__":
    main()
