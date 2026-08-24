"""Step 1 —— 降阈值报告阶梯。

读我们的 annotations.csv（或 annotations_fdr.csv），对 rank==1 命中，报告
cos ∈ {0.5,0.6,0.7,0.8,0.9} × (纯余弦 / +m/z 硬约束) 两档的：
    谱数、唯一化合物数、注释率
这样就能看到把 confident 刻度从 0.7 降到 0.5 能多出多少「代谢物」，而无需重跑嵌入/检索。

注意 m/z 硬约束才是压假阳性的功臣（去掉它 fp_proxy 会从 0 涨到 ~0.7），所以「更多代谢物」
的诚实刻度是「cos≥t & mz_pass」那一列，而不是纯余弦那一列。

用法 (conda dreams_env, 服务器):
    python tasks/annotation_ladder.py \
        --annotations data/mtbls13729/annotation/pos_rp/annotations.csv \
                       data/mtbls13729/annotation/neg_rp/annotations.csv \
                       data/mtbls13729/annotation/pos_hilic/annotations.csv \
                       data/mtbls13729/annotation/neg_hilic/annotations.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def ladder_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    top1 = df[df["rank"] == 1].copy()
    n_query = int(top1["query_idx"].nunique())

    rows = []
    for t in THRESHOLDS:
        cos = top1[top1["cosine"] >= t]
        mz = cos[cos["mz_pass"]]
        rows.append({
            "cos": t,
            "n_spectra_cos": int(len(cos)),
            "n_unique_cos": int(cos["lib_inchikey"].nunique()),
            "n_spectra_mz": int(len(mz)),
            "n_unique_mz": int(mz["lib_inchikey"].nunique()),
            "rate_mz": round(len(mz) / n_query, 4) if n_query else 0.0,
        })
    lad = pd.DataFrame(rows)
    print(f"\n===== {path} =====", flush=True)
    print(f"query 谱数 = {n_query}", flush=True)
    print(lad.to_string(index=False), flush=True)
    return lad


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", nargs="+", type=Path, required=True,
                   help="一个或多个 annotations.csv / annotations_fdr.csv")
    args = p.parse_args()

    for path in args.annotations:
        if not path.exists():
            print(f"[ladder] 跳过（不存在）: {path}", flush=True)
            continue
        ladder_one(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
