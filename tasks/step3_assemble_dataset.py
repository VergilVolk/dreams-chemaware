"""
Step 3: 组装对比学习数据集 manifest（anchor / pos / neg_list）。

输入（Step 1 产物 + HDF5 元数据）：
  tasks/massspecgym_isomers/pairs.json       # near/mid/far 异构体对（含 mces_raw）
  tasks/massspecgym_isomers/ik_to_rows.json  # ik14 -> train fold 谱行索引
  data/models/MassSpecGym_MurckoHist_split.hdf5  # 每行 adduct / precursor_mz

输出：
  tasks/massspecgym_isomers/dataset_manifest.json
    { "meta": {...统计...}, "train": [anchor entries], "eval": [anchor entries] }
    anchor entry = {anchor_row, ik14, adduct, precursor_mz, neg:[{ik14,row,grade,mces_raw}]}

设计要点（可调）：
  - anchor = 谱行；正例 = 该谱的噪声版（Step 4 现场生成，每 epoch 重新抽）。
  - 负例 = 同分异构体谱，强制同 adduct（同 formula+同 adduct -> 同 precursor m/z，逼模型看碎裂）。
  - 负例优先序 near > mid > far，cap 每个 anchor 最多 --max-neg 个；far 也进池（"一律负例"），
    每条都打 grade 标签，供后续 near-only vs 全量消融。
  - 按异构体图连通分量划分 train/eval，保证 anchor/正例/负例分子零跨 split 重叠（--eval-frac 控制 eval 分子占比）。

用法（本机 conda，CPU，快）：
  python tasks/step3_assemble_dataset.py
  python tasks/step3_assemble_dataset.py --max-neg 8 --eval-frac 0.1 --split-seed 0
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HDF5_PATH = os.path.join(ROOT, "data/models/MassSpecGym_MurckoHist_split.hdf5")
ISOMER_DIR = os.path.join(ROOT, "tasks/massspecgym_isomers")

GRADE_ORDER = {"near": 0, "mid": 1, "far": 2}


def decode(x):
    return x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=str, default="train")
    ap.add_argument("--max-neg", type=int, default=8, help="每个 anchor 最多几个异构体负例")
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=os.path.join(ISOMER_DIR, "dataset_manifest.json"))
    args = ap.parse_args()

    # ---- 1. 读 pairs，建邻接 ik14 -> [(neighbor, grade, mces_raw)] ----
    with open(os.path.join(ISOMER_DIR, "pairs.json")) as fh:
        pairs = json.load(fh)
    adj = defaultdict(list)
    for grade, entries in [("near", pairs["near"]), ("mid", pairs["mid"]), ("far", pairs["far"])]:
        for e in entries:
            mces = e.get("mces_raw", 99)
            adj[e["ik_a"]].append((e["ik_b"], grade, mces))
            adj[e["ik_b"]].append((e["ik_a"], grade, mces))
    print(f"[1] 异构体邻接：{len(adj)} 个 ik14 出现在异构体对里")

    # ---- 2. 读 ik14 -> rows + HDF5 adduct / precursor_mz ----
    with open(os.path.join(ISOMER_DIR, "ik_to_rows.json")) as fh:
        ik_to_rows = json.load(fh)

    with h5py.File(HDF5_PATH, "r") as f:
        folds = np.array([decode(x) for x in f["fold"][:]])
        adducts = np.array([decode(x) for x in f["adduct"][:]])
        pmzs = np.array(f["precursor_mz"][:], dtype=float)
    train_mask = folds == args.fold

    # ik14 -> {adduct: [rows]}（仅 train fold）
    ik_rows_adduct = defaultdict(lambda: defaultdict(list))
    for ik, rows in ik_to_rows.items():
        for r in rows:
            if train_mask[r]:
                ik_rows_adduct[ik][adducts[r]].append(r)

    # ---- 3. anchor 池 = 有异构体 + 有 train 谱行的 ik14 ----
    anchor_iks = [ik for ik in adj if ik in ik_rows_adduct and ik_rows_adduct[ik]]
    print(f"[2] anchor 池（有异构体+有谱）: {len(anchor_iks)} 个分子")

    # ---- 4. 展开为 anchor entries（每谱行一条），负例同 adduct + near 优先 ----
    def neg_key(t):
        _, grade, mces = t
        return (GRADE_ORDER.get(grade, 9), mces if mces is not None else 99)

    def build_entries(ik):
        entries = []
        neighbors = sorted(adj[ik], key=neg_key)
        for adduct, rows in ik_rows_adduct[ik].items():
            for r in rows:
                negs = []
                for nb, grade, mces in neighbors:
                    if nb not in ik_rows_adduct or adduct not in ik_rows_adduct[nb]:
                        continue  # 无同 adduct 谱行 -> 跳过（非同撞脸场景）
                    nrow = ik_rows_adduct[nb][adduct][0]
                    negs.append({"ik14": nb, "row": int(nrow), "grade": grade, "mces_raw": mces})
                    if len(negs) >= args.max_neg:
                        break
                entries.append({
                    "anchor_row": int(r), "ik14": ik, "adduct": adduct,
                    "precursor_mz": float(pmzs[r]), "neg": negs,
                })
        return entries

    all_entries = []
    for ik in anchor_iks:
        all_entries.extend(build_entries(ik))

    # ---- 5. 按异构体图连通分量划分 train/eval（零跨 split 泄漏） ----
    # 旧 bug：只按 anchor ik14 随机分，负例（异构体）会跨 split——eval 锚的负例谱行出现在
    # 训练侧（师兄审计实测：train/eval 分子重叠 3,731、谱行重叠 4,195、eval 98.5% 分子
    # 在训练侧出现过）。修法：A、B 是异构体（有边）就必须同侧 → 对 anchor 分子做 union-find
    # 连通分量，整个分量一起进 train 或 eval，保证任何异构体对都不跨 split。
    parent = {ik: ik for ik in anchor_iks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for ik in anchor_iks:
        for (nb, _grade, _mces) in adj[ik]:
            if nb in parent:      # nb 也是 anchor 分子（有谱）
                union(ik, nb)

    comp = defaultdict(list)
    for ik in anchor_iks:
        comp[find(ik)].append(ik)
    comps = sorted(comp.values(), key=lambda c: (-len(c), min(c)))
    rng = np.random.default_rng(args.split_seed)
    rng.shuffle(comps)

    target_eval = max(1, int(round(len(anchor_iks) * args.eval_frac)))
    eval_iks = set()
    n_eval = 0
    for c in comps:
        if n_eval >= target_eval:
            break
        eval_iks.update(c)
        n_eval += len(c)
    train_entries = [e for e in all_entries if e["ik14"] not in eval_iks]
    eval_entries = [e for e in all_entries if e["ik14"] in eval_iks]
    comp_sizes = sorted((len(c) for c in comps), reverse=True)
    print(f"[5] 连通分量数={len(comps)}，最大分量={comp_sizes[0]}，size>=2 数={sum(1 for s in comp_sizes if s >= 2)}，"
          f"eval 分子={n_eval}（目标≈{target_eval}）")

    # ---- 6. 统计 + 保存 ----
    def neg_grade_dist(entries):
        c = defaultdict(int)
        empty = 0
        for e in entries:
            if not e["neg"]:
                empty += 1
            for n in e["neg"]:
                c[n["grade"]] += 1
        return dict(c), empty

    tr_dist, tr_empty = neg_grade_dist(train_entries)
    ev_dist, ev_empty = neg_grade_dist(eval_entries)

    # 零泄漏核验：分子（anchor+neg 并集）与谱行都不得跨 train/eval
    def all_molecules(entries):
        mols, rows = set(), set()
        for e in entries:
            mols.add(e["ik14"]); rows.add(e["anchor_row"])
            for n in e["neg"]:
                mols.add(n["ik14"]); rows.add(n["row"])
        return mols, rows

    tr_mols, tr_rows = all_molecules(train_entries)
    ev_mols, ev_rows = all_molecules(eval_entries)
    leak_mols = len(tr_mols & ev_mols)
    leak_rows = len(tr_rows & ev_rows)
    anchor_ov = len({e["ik14"] for e in train_entries} & {e["ik14"] for e in eval_entries})

    meta = {
        "fold": args.fold,
        "max_neg": args.max_neg,
        "eval_frac": args.eval_frac,
        "split_seed": args.split_seed,
        "split": "isomer_connected_component",
        "n_anchor_molecules": len(anchor_iks),
        "n_components": len(comps),
        "n_eval_molecules": n_eval,
        "n_train_anchors": len(train_entries),
        "n_eval_anchors": len(eval_entries),
        "anchor_molecule_overlap": anchor_ov,
        "leak_molecule_overlap": leak_mols,
        "leak_spectrum_row_overlap": leak_rows,
        "train_neg_grade_dist": tr_dist,
        "train_empty_neg": tr_empty,
        "eval_neg_grade_dist": ev_dist,
        "eval_empty_neg": ev_empty,
    }
    out = {"meta": meta, "train": train_entries, "eval": eval_entries}
    with open(args.out, "w") as fh:
        json.dump(out, fh, ensure_ascii=False)
    print(f"[3] train anchors: {len(train_entries)}  eval anchors: {len(eval_entries)}")
    print(f"    train 负例分级: {tr_dist}（empty-neg {tr_empty}）")
    print(f"    eval  负例分级: {ev_dist}（empty-neg {ev_empty}）")
    ok = (anchor_ov == 0 and leak_mols == 0 and leak_rows == 0)
    print(f"[零泄漏核验] anchor 分子重叠={anchor_ov}  分子重叠={leak_mols}  谱行重叠={leak_rows}  "
          f"{'[OK] 全为 0' if ok else '[FAIL] 仍有泄漏!'}")
    print(f"[4] 已保存 -> {args.out}")
    print("\n=== Step 3 DONE ===")


if __name__ == "__main__":
    main()
