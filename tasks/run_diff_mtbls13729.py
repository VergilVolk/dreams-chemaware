"""MTBLS13729 分组差异分析（样本级 presence/absence + Fisher 精确检验 + BH 校正）。

与 run_diff.py 逻辑一致，仅 derive_group 改为解析 MTBLS13729 样本名。
样本名形如 P01-Ltu / P01-LN / P11-Rtu / P11-RN / P21-Rmu / P21-RN。
分组维度: tissue(Tumor/Normal), location(Left/Right), histology(Tubular/Mucinous/Normal)。

用法 (conda dreams_env, CPU):
    python tasks/run_diff_mtbls13729.py \
        --annotations data/mtbls13729/annotation/pos_rp/annotations_fdr.csv \
        --out data/mtbls13729/annotation/pos_rp/diff_tumor_vs_normal.csv \
        --group-col tissue --group-a Tumor --group-b Normal

可加 --fdr-pass 让 confident top-1 额外要求 q-value 通过（更严，化合物更少）。
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


def parse_sample(query_file: str) -> dict[str, str]:
    """P01-Ltu -> {tissue:Tumor, location:Left, histology:Tubular}; P01-LN -> Normal."""
    s = str(query_file)
    tag = s.split("-")[-1] if "-" in s else s
    loc = "Left" if tag.startswith("L") else "Right"
    if tag in ("LN", "RN"):
        return {"tissue": "Normal", "location": loc, "histology": "Normal"}
    hist = "Mucinous" if tag == "Rmu" else "Tubular"
    return {"tissue": "Tumor", "location": loc, "histology": hist}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--group-col", default="tissue",
                        choices=["tissue", "location", "histology"],
                        help="按哪个维度分组")
    parser.add_argument("--group-a", required=True)
    parser.add_argument("--group-b", required=True)
    parser.add_argument("--fdr-pass", action="store_true",
                        help="confident top-1 额外要求 fdr_pass（需先跑 FDR）")
    args = parser.parse_args()

    hits = pd.read_csv(args.annotations)
    grp = hits["query_file"].apply(lambda q: parse_sample(q)[args.group_col])
    hits = hits.assign(_group=grp)

    hits = hits[hits["_group"].isin([args.group_a, args.group_b])].copy()
    total_a = int(hits.loc[hits["_group"] == args.group_a, "query_file"].nunique())
    total_b = int(hits.loc[hits["_group"] == args.group_b, "query_file"].nunique())
    print(f"[diff] 注释行 {len(hits)}；样本数 {args.group_a}={total_a}, {args.group_b}={total_b}",
          flush=True)

    conf = confident_top1(hits, DEFAULT, fdr_pass=args.fdr_pass)
    print(f"[diff] confident top1: {len(conf)} 行（{conf['lib_inchikey'].nunique()} 个化合物）",
          flush=True)

    res = differential(conf, group_col="_group", group_a=args.group_a,
                       group_b=args.group_b, params=DEFAULT,
                       total_a=total_a, total_b=total_b)

    names = conf.drop_duplicates("lib_inchikey")[["lib_inchikey", "lib_name"]]
    res = res.merge(names, on="lib_inchikey", how="left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    n_sig = int((res["q_value"] < 0.05).sum()) if len(res) else 0
    print(f"[diff] 化合物 {len(res)} 个，q<0.05 显著 {n_sig} 个", flush=True)
    print(f"[diff] 前 10 名（按 q 值）:", flush=True)
    cols = [c for c in ("lib_name", f"n_samples_{args.group_a}", f"n_samples_{args.group_b}",
                        "odds_ratio", "q_value") if c in res.columns]
    print(res[cols].head(10).to_string(index=False), flush=True)
    print(f"[diff] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
