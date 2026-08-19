"""M5 -- Met/neg PF vs HF 差异分析（样本级 presence/absence + Fisher 精确检验 + BH 校正）。

输入: 注释表 annotations.csv（含 query_file 列，形如 PF_1 / HF_1）。
分组: 从 query_file 前缀提取（PF / HF），QC/Blank 前缀自动排除。
统计单位: 每个样本（query_file）一个点（该化合物是否被 confident top-1 检出），
不再用谱计数做 2x2（避免一张样本出多张谱导致的伪重复）。

Usage (conda dreams_env, CPU):
    python tasks/run_diff.py \
        --annotations data/msv100574/annotation/met_neg/annotations.csv \
        --out data/msv100574/annotation/met_neg/diff.csv \
        --group-a PF --group-b HF
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from annotation.diff import confident_top1, differential  # noqa: E402
from annotation.params import DEFAULT  # noqa: E402


def derive_group(query_file) -> str:
    """PF_1 -> PF, HF_3 -> HF, Blank_7 -> Blank, QC_2 -> QC."""
    return str(query_file).split("_")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--group-a", default="PF")
    parser.add_argument("--group-b", default="HF")
    parser.add_argument("--fdr-pass", action="store_true", help="额外要求 qvalue 通过（需先跑 FDR）")
    args = parser.parse_args()

    hits = pd.read_csv(args.annotations)
    hits["_group"] = hits["query_file"].apply(derive_group)

    # 只保留要比较的两组
    hits = hits[hits["_group"].isin([args.group_a, args.group_b])].copy()
    total_a = int(hits.loc[hits["_group"] == args.group_a, "query_file"].nunique())
    total_b = int(hits.loc[hits["_group"] == args.group_b, "query_file"].nunique())
    print(f"[diff] 注释行 {len(hits)}；样本数 {args.group_a}={total_a}, {args.group_b}={total_b}", flush=True)

    conf = confident_top1(hits, DEFAULT, fdr_pass=args.fdr_pass)
    print(f"[diff] confident top1: {len(conf)} 行（{conf['lib_inchikey'].nunique()} 个化合物）", flush=True)

    res = differential(conf, group_col="_group", group_a=args.group_a, group_b=args.group_b,
                       params=DEFAULT, total_a=total_a, total_b=total_b)

    # 合并化合物名
    names = conf.drop_duplicates("lib_inchikey")[["lib_inchikey", "lib_name"]]
    res = res.merge(names, on="lib_inchikey", how="left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    n_sig = int((res["q_value"] < 0.05).sum()) if len(res) else 0
    print(f"[diff] 化合物 {len(res)} 个，q<0.05 显著 {n_sig} 个", flush=True)
    print(f"[diff] 前 10 名（按 q 值）:", flush=True)
    cols = [c for c in ("lib_name", f"n_samples_{args.group_a}", f"n_samples_{args.group_b}", "odds_ratio", "q_value") if c in res.columns]
    print(res[cols].head(10).to_string(index=False), flush=True)
    print(f"[diff] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
