"""MTBLS13729 丰度基础差异分析 —— 适配作者"72 个差异代谢物"的做法。

**为什么单独写这个脚本**：作者（J Proteome Res 2026, DOI 10.1021/acs.jproteome.5c01260）
做的是基于 **MS1 峰面积丰度**的 fold-change 差异（报道了 Ltu vs Rtu 共 72 个改变代谢物）。
我们的 DDA 数据只有 **MS2 谱计数**这个半定量替代（diff.py 已声明此 caveat：谱计数
精度低于 MS1 峰面积/DIA）。现有 run_diff_mtbls13729.py 用的是 presence/absence Fisher
（样本级有无），与作者的丰度 fold-change 是两种不同口径。本脚本补齐"丰度口径"，便于
与作者结论直接对照。

**丰度定义**：每个化合物在每个样本的"丰度"= 该样本内 confident top-1 命中该化合物的
MS2 谱数（spectral counting, 标准 DDA 半定量代理）。

**统计**：fold change（log2，加伪计数 0.5）+ Mann-Whitney U 检验（谱计数零膨胀/非正态，
非参数更稳；Mann & Whitney 1947）+ Benjamini-Hochberg FDR（Benjamini & Hochberg 1995）。

**作者比较方案**（5 组，样本名 tag 直接对应）：
    Ltu vs Rtu   左半 vs 右半管状腺癌（部位，作者 72 DEM 就在这组）
    Rmu vs Rtu   右半黏液 vs 管状腺癌（组织学，鞘脂发现）
    Ltu vs LN / Rtu vs RN / Rmu vs RN   各肿瘤亚组 vs 配对正常（通路差异）

用法 (conda dreams_env, CPU):
    python tasks/run_mtbls13729_abundance.py \
        --annotations data/mtbls13729/annotation/pos_rp/annotations_fdr.csv \
        --out-dir data/mtbls13729/annotation/pos_rp/diff_abundance
可加 --fdr-pass 让 confident top-1 额外要求 q-value 通过（更严）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from annotation.diff import confident_top1  # noqa: E402
from annotation.params import DEFAULT  # noqa: E402

# 作者比较方案：tag 对 (组A, 组B)。tag 从样本名末段解析（P01-Ltu -> Ltu）。
AUTHOR_COMPARISONS: list[tuple[str, str]] = [
    ("Ltu", "Rtu"),   # 部位（管状内）
    ("Rmu", "Rtu"),   # 组织学（右半内）
    ("Ltu", "LN"),    # 左半管状肿瘤 vs 配对正常
    ("Rtu", "RN"),    # 右半管状肿瘤 vs 配对正常
    ("Rmu", "RN"),    # 右半黏液肿瘤 vs 配对正常
]

PSEUDO_COUNT = 0.5  # log2 fold-change 的伪计数，避免除零
FC_MIN_REPORT = 1.0  # 汇总表里 log2FC 绝对值的报告阈值（|log2FC|>=1 即 2 倍）


def parse_tag(query_file: str) -> str:
    """P01-Ltu -> Ltu; P21-Rmu -> Rmu; 取最后一个 '-' 之后的部分。"""
    s = str(query_file)
    return s.split("-")[-1] if "-" in s else s


def spectral_count_matrix(conf: pd.DataFrame) -> pd.DataFrame:
    """每样本每化合物的 MS2 谱计数矩阵（index=InChIKey, columns=样本名, 缺失填 0）。"""
    counts = (
        conf.groupby(["lib_inchikey", "query_file"]).size().rename("n").reset_index()
    )
    mat = counts.pivot(index="lib_inchikey", columns="query_file", values="n").fillna(0)
    return mat


def abundance_differential(
    mat: pd.DataFrame,
    samples_a: list[str],
    samples_b: list[str],
    group_a: str,
    group_b: str,
) -> pd.DataFrame:
    """对矩阵做 fold-change + Mann-Whitney U + BH。返回化合物级结果表。"""
    cols_a = [s for s in samples_a if s in mat.columns]
    cols_b = [s for s in samples_b if s in mat.columns]
    rows = []
    for inchikey in mat.index:
        ca = mat.loc[inchikey, cols_a].to_numpy(dtype=float)
        cb = mat.loc[inchikey, cols_b].to_numpy(dtype=float)
        mean_a = ca.mean()
        mean_b = cb.mean()
        log2fc = float(np.log2((mean_a + PSEUDO_COUNT) / (mean_b + PSEUDO_COUNT)))
        # Mann-Whitney U；两组完全相同的退化情形 p=1
        if np.array_equal(ca, cb):
            p = 1.0
        else:
            p = float(stats.mannwhitneyu(ca, cb, alternative="two-sided").pvalue)
        rows.append({
            "lib_inchikey": inchikey,
            f"mean_count_{group_a}": float(mean_a),
            f"mean_count_{group_b}": float(mean_b),
            f"n_present_{group_a}": int((ca > 0).sum()),
            f"n_present_{group_b}": int((cb > 0).sum()),
            "log2FC": log2fc,
            "p_value": p,
        })
    res = pd.DataFrame(rows)
    if len(res):
        p = res["p_value"].to_numpy()
        n = len(p)
        ranked = np.argsort(p)
        bh = p[ranked] * n / (np.arange(n) + 1)
        q_ranked = np.minimum.accumulate(bh[::-1])[::-1]
        qvals = np.empty(n)
        qvals[ranked] = q_ranked
        res["q_value"] = qvals
    return res.sort_values("q_value").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fdr-pass", action="store_true",
                        help="confident top-1 额外要求 fdr_pass")
    parser.add_argument("--ppm-tol", type=float, default=DEFAULT.ppm_tolerance,
                        help="（保留参数，与检索一致，暂不影响谱计数差异）")
    args = parser.parse_args()

    hits = pd.read_csv(args.annotations)
    # 样本名 tag 用于按作者子组切分
    hits = hits.assign(_tag=hits["query_file"].apply(parse_tag))

    # 全部样本（含零 confident 命中的样本）来自完整 query 集合
    all_samples = sorted(hits["query_file"].unique())
    tag_of = {s: parse_tag(s) for s in all_samples}
    print(f"[abundance] 总谱行 {len(hits)}；样本 {len(all_samples)} 个", flush=True)

    conf = confident_top1(hits, DEFAULT, fdr_pass=args.fdr_pass)
    print(f"[abundance] confident top-1: {len(conf)} 行 / "
          f"{conf['lib_inchikey'].nunique()} 个化合物", flush=True)

    mat = spectral_count_matrix(conf)
    print(f"[abundance] 谱计数矩阵: {mat.shape[0]} 化合物 × {mat.shape[1]} 样本",
          flush=True)

    names = conf.drop_duplicates("lib_inchikey")[["lib_inchikey", "lib_name"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for group_a, group_b in AUTHOR_COMPARISONS:
        samples_a = [s for s in all_samples if tag_of[s] == group_a]
        samples_b = [s for s in all_samples if tag_of[s] == group_b]
        if not samples_a or not samples_b:
            print(f"[abundance] 跳过 {group_a} vs {group_b}（样本缺失）", flush=True)
            continue
        res = abundance_differential(mat, samples_a, samples_b, group_a, group_b)
        res = res.merge(names, on="lib_inchikey", how="left")
        out = args.out_dir / f"diff_{group_a}_vs_{group_b}.csv"
        res.to_csv(out, index=False)
        n_sig = int((res["q_value"] < 0.05).sum()) if len(res) else 0
        n_fc = int((res["log2FC"].abs() >= FC_MIN_REPORT).sum()) if len(res) else 0
        print(f"[abundance] {group_a} vs {group_b}: {len(res)} 化合物, "
              f"q<0.05 显著 {n_sig}, |log2FC|>=1 共 {n_fc} -> {out}", flush=True)
        summary_rows.append({
            "comparison": f"{group_a}_vs_{group_b}",
            "n_compounds": len(res),
            "n_sig_q005": n_sig,
            "n_log2fc_ge1": n_fc,
            "n_samples_A": len(samples_a),
            "n_samples_B": len(samples_b),
        })
        # 前 10 名（按 q 值），复现作者报告风格
        cols = [c for c in ("lib_name", "log2FC", "p_value", "q_value",
                            f"n_present_{group_a}", f"n_present_{group_b}")
                if c in res.columns]
        print(res[cols].head(10).to_string(index=False), flush=True)

    pd.DataFrame(summary_rows).to_csv(args.out_dir / "_summary.csv", index=False)
    print(f"[abundance] 汇总 -> {args.out_dir / '_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
