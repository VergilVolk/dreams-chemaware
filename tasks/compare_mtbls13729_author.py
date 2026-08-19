"""对照作者 345 条注释 —— 复现 / 升级 / 新增（核心交付）。

作者（J Proteome Res 2026, DOI 10.1021/acs.jproteome.5c01260）的 maf.tsv 注释只有
HMDB ID + HMDB 名字 + m/z + RT，**没有结构 / 没有谱图打分 / 没有置信度**，相当于
Schymanski L3-L4（DOI 10.1021/es5002105）。我们用 DreaMS 的 confident top-1
（cosine >= 0.7 + m/z 硬约束 + 可选 FDR）逐 m/z 对照，产出三类结论：

1. **复现 / 覆盖**：作者 345 条里，我们在 ±ppm 内也有 confident 命中的比例。
   （"我们的管线能不能重注释到作者报的特征"）
2. **升级**：对复现的条目，我们补上作者没有的 InChIKey + cosine + qvalue。
   若作者名字本身歧义（含 "»" / " or " / " / "），我们命中的**单一** InChIKey
   就是"解异构体歧义"（如 UDCA vs CDCA、2-氨基丁酸异构体）。
3. **新增（暗物质）**：我们 confident 命中、但作者 345 条没有任何 m/z 匹配的
   化合物 —— 即作者"看不见"的信号，攻击暗物质（da Silva 2015, DOI
   10.1073/pnas.1506877112）的候选。

**m/z 对齐口径**：作者 mass_to_charge 与我们的 query_precursor_mz 来自**同一批
mzML**（作者只是换了自己的峰检测），所以按 ppm（默认 10）对齐同一特征。

用法 (conda dreams_env, CPU):
    python tasks/compare_mtbls13729_author.py \
        --maf-dir _mtbls13729_meta \
        --annotations-root data/mtbls13729/annotation \
        --out-dir data/mtbls13729/annotation/author_compare
可加 --fdr-pass 让 confident top-1 额外要求 q-value 通过（更严）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from annotation.diff import confident_top1  # noqa: E402
from annotation.params import DEFAULT  # noqa: E402

# 面板 -> 作者 maf 文件名
PANEL_MAF: dict[str, str] = {
    "pos_rp": "m_MTBLS13729_LC-MS_positive_reverse-phase_metabolite_profiling_v2_maf.tsv",
    "neg_rp": "m_MTBLS13729_LC-MS_negative_reverse-phase_metabolite_profiling_v2_maf.tsv",
    "pos_hilic": "m_MTBLS13729_LC-MS_positive_hilic_metabolite_profiling_v2_maf.tsv",
    "neg_hilic": "m_MTBLS13729_LC-MS_negative_hilic_metabolite_profiling_v2_maf.tsv",
}

PPM_TOL = 10.0  # 特征对齐容差（同一批 mzML，作者峰检测 vs 我们 raw precursor）

# 作者名字里的歧义分隔符（异构体"或"）
AMBIGUITY_MARKERS = ("»", " or ", " / ", "或", "；")


def min_dppm(x: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """对每个 x，返回它与任意 ref 的最小 |x-ref|/ref*1e6（ppm）；refs 空则 +inf。"""
    x = np.asarray(x, dtype=float)
    refs = np.asarray(refs, dtype=float)
    if refs.size == 0:
        return np.full(x.shape, np.inf)
    return (np.abs(x[:, None] - refs[None, :]) / refs[None, :] * 1e6).min(axis=1)


def is_ambiguous(name: str) -> bool:
    """作者名字是否含异构体歧义分隔符。"""
    s = str(name)
    return any(m in s for m in AMBIGUITY_MARKERS)


def load_author_maf(maf_dir: Path) -> pd.DataFrame:
    """读 4 个作者 maf，合并为 (panel, hmdb, name, mz, rt)。"""
    frames = []
    for panel, fname in PANEL_MAF.items():
        p = maf_dir / fname
        if not p.exists():
            print(f"[author] 跳过 {fname}（不存在）", flush=True)
            continue
        df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
        df = df.rename(columns={
            "database_identifier": "hmdb",
            "metabolite_identification": "name",
            "mass_to_charge": "mz",
            "retention_time": "rt",
        })
        df["panel"] = panel
        df["mz"] = pd.to_numeric(df["mz"], errors="coerce")
        df = df[df["mz"].notna() & (df["mz"] > 0)]
        frames.append(df[["panel", "hmdb", "name", "mz", "rt"]])
    return pd.concat(frames, ignore_index=True)


def load_our_confident(annotations_root: Path, fdr_pass: bool) -> pd.DataFrame:
    """读各面板的 DreaMS 注释，取 confident top-1。"""
    frames = []
    for panel in PANEL_MAF:
        p = annotations_root / panel / "annotations_fdr.csv"
        if not p.exists():
            p = annotations_root / panel / "annotations.csv"
        if not p.exists():
            print(f"[ours] 跳过 {panel}（无 annotations 文件）", flush=True)
            continue
        hits = pd.read_csv(p)
        conf = confident_top1(hits, DEFAULT, fdr_pass=fdr_pass)
        conf = conf.copy()
        conf["panel"] = panel
        keep = ["panel", "query_precursor_mz", "lib_inchikey", "lib_name", "cosine"]
        if "qvalue" in conf.columns:
            keep.append("qvalue")
        frames.append(conf[keep])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maf-dir", type=Path, required=True)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ppm-tol", type=float, default=PPM_TOL)
    parser.add_argument("--fdr-pass", action="store_true")
    args = parser.parse_args()

    author = load_author_maf(args.maf_dir)
    ours = load_our_confident(args.annotations_root, args.fdr_pass)
    print(f"[compare] 作者注释 {len(author)} 条；我们的 confident top-1 {len(ours)} 行",
          flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for panel in PANEL_MAF:
        a = author[author["panel"] == panel]
        o = ours[ours["panel"] == panel]
        if a.empty:
            print(f"[compare] {panel}: 无作者注释，跳过", flush=True)
            continue
        print(f"\n[compare] ===== {panel}: 作者 {len(a)} 条，我们 {len(o)} 行 =====",
              flush=True)

        a = a.copy()
        a["ambiguous"] = a["name"].apply(is_ambiguous)
        a["min_dppm"] = min_dppm(a["mz"].to_numpy(), o["query_precursor_mz"].to_numpy())
        a["covered"] = a["min_dppm"] <= args.ppm_tol

        # 对每个作者条目，找我们的命中结构（复现 -> 升级 -> 解歧义）
        assigned_inchikeys, assigned_names, max_cos, n_distinct = [], [], [], []
        for mz in a["mz"]:
            # 布尔行掩码（长度=len(o)）：每个 query precursor 是否落在 mz 的 ±ppm 内
            mask = min_dppm(o["query_precursor_mz"].to_numpy(),
                            np.array([mz])) <= args.ppm_tol
            sub = o[mask]
            if sub.empty:
                assigned_inchikeys.append("")
                assigned_names.append("")
                max_cos.append(np.nan)
                n_distinct.append(0)
            else:
                assigned_inchikeys.append(";".join(sub["lib_inchikey"].dropna().unique()))
                assigned_names.append(";".join(sub["lib_name"].dropna().unique()))
                max_cos.append(float(sub["cosine"].max()))
                n_distinct.append(int(sub["lib_inchikey"].nunique()))
        a["our_inchikey"] = assigned_inchikeys
        a["our_name"] = assigned_names
        a["our_max_cosine"] = max_cos
        a["our_n_structures"] = n_distinct
        a["resolved"] = a["ambiguous"] & a["covered"] & (a["our_n_structures"] == 1)

        a.to_csv(args.out_dir / f"author_compare_{panel}.csv", index=False)

        n_cov = int(a["covered"].sum())
        n_res = int(a["resolved"].sum())
        n_amb = int(a["ambiguous"].sum())

        # 新增（暗物质）：我们 distinct InChIKey 的 median m/z 不在任何作者 m/z 内
        if o.empty:
            novel = pd.DataFrame()
        else:
            g = o.groupby("lib_inchikey").agg(
                median_mz=("query_precursor_mz", "median"),
                lib_name=("lib_name", "first"),
                max_cosine=("cosine", "max"),
                n_spectra=("query_precursor_mz", "size"),
            ).reset_index()
            g["min_dppm_to_author"] = min_dppm(
                g["median_mz"].to_numpy(), a["mz"].to_numpy()
            )
            novel = g[g["min_dppm_to_author"] > args.ppm_tol].sort_values("n_spectra",
                                                                         ascending=False)
            novel.to_csv(args.out_dir / f"novel_{panel}.csv", index=False)

        print(f"  复现(覆盖): {n_cov}/{len(a)} = {n_cov/len(a)*100:.1f}%", flush=True)
        print(f"  作者歧义条目: {n_amb}；其中解歧义(命中单一结构): {n_res}", flush=True)
        print(f"  新增(暗物质候选): {len(novel)} 个 distinct InChIKey", flush=True)

        summary_rows.append({
            "panel": panel,
            "n_author": len(a),
            "n_covered": n_cov,
            "coverage_pct": round(n_cov / len(a) * 100, 1),
            "n_author_ambiguous": n_amb,
            "n_resolved": n_res,
            "n_our_structures": int(o["lib_inchikey"].nunique()) if not o.empty else 0,
            "n_novel": len(novel),
        })

    pd.DataFrame(summary_rows).to_csv(args.out_dir / "_summary.csv", index=False)
    print(f"\n[compare] 汇总 -> {args.out_dir / '_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
