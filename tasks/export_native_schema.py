"""把我们的 annotations.csv 导出成原生 DreaMS API 的 TSV schema。

我们的 retrieve() 产出的 annotations.csv 是长表（每个 query 谱 × 每个 rank 一行）：
    query_idx, query_file, query_scan, query_precursor_mz, query_group,
    rank, cosine, lib_smiles, lib_inchikey, lib_name, lib_precursor_mz, dppm, mz_pass
（若跑过 FDR，会追加 qvalue / fdr_pass）

原生 anton-bushuiev/DreaMS Space 每个输入文件产出的 TSV，其「注释本质」列是：
    Row, scan_number, RT, precursor_mz, ref_precursor_mz, charge, file_name,
    ref_INCHIKEY, ref_name, ref_smiles, topk, DreaMS_similarity, analog_hit
（另有 spectrum / DreaMS_embedding / ref_* 等原始谱与嵌入载荷列，属于另一套模型
  空间的内部表示，这里无法也不需要复现 —— 见 [[dreams-two-model-spaces-must-not-mix]]）

本脚本把我们的列映射到上述 schema，并从 query manifest 回填 RT / charge（retrieve()
没把这两列带进 annotations.csv），最后产出：
    {out_dir}/annotations_native.tsv       合并表（全部 query 谱 × 全部 rank）
    {out_dir}/per_file/{file_stem}.tsv     每个 query 文件一个 TSV（对齐原生按文件布局）
    {out_dir}/manifest.json                文件级汇总（对齐原生 batch 的 manifest）

用法 (conda dreams_env):
    python tasks/export_native_schema.py \
        --annotations data/mtbls13729/annotation/pos_rp/annotations_fdr.csv \
        --query-manifest data/mtbls13729/embeddings/pos_rp/manifest.csv \
        --out-dir data/mtbls13729/annotation_native/pos_rp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 原生 TSV 的「注释本质」列顺序（去掉原始谱/嵌入载荷列）
NATIVE_COLS = [
    "scan_number", "RT", "precursor_mz", "ref_precursor_mz", "charge",
    "file_name", "ref_INCHIKEY", "ref_name", "ref_smiles",
    "topk", "DreaMS_similarity", "analog_hit",
]
# 我们额外附带的置信度列（原生的 topk 通常=1；我们保留 1..topk，dppm/mz_pass 是 m/z 硬约束）
EXTRA_COLS = ["dppm", "mz_pass", "qvalue", "fdr_pass"]


def _load_joined(annotations: Path, query_manifest: Path) -> pd.DataFrame:
    """读 annotations.csv 并按 query_idx（= manifest 行序）回填 RT / charge。"""
    hits = pd.read_csv(annotations)
    man = pd.read_csv(query_manifest)

    required = ["query_idx", "query_file", "query_scan", "query_precursor_mz",
                "rank", "cosine", "lib_smiles", "lib_inchikey", "lib_name",
                "lib_precursor_mz"]
    missing = [c for c in required if c not in hits.columns]
    if missing:
        raise SystemExit(f"[export] annotations 缺列: {missing}")

    man_pos = man.reset_index().rename(columns={"index": "__man_pos"})
    joined = hits.merge(man_pos, left_on="query_idx", right_on="__man_pos",
                        how="left")
    if joined["RT"].isna().all():
        print("[export] WARN: manifest 没有 RT 列或 join 失败，RT 全为空", file=sys.stderr)
    if "charge" not in joined.columns:
        joined["charge"] = pd.NA
    return joined


def _to_native(joined: pd.DataFrame) -> pd.DataFrame:
    """映射列名 -> 原生 schema（长表，rank = topk）。"""
    out = pd.DataFrame()
    out["scan_number"] = joined["query_scan"].astype("Int64")
    out["RT"] = joined["RT"]
    out["precursor_mz"] = joined["query_precursor_mz"].astype(float)
    out["ref_precursor_mz"] = joined["lib_precursor_mz"].astype(float)
    out["charge"] = joined["charge"]
    out["file_name"] = joined["query_file"]
    out["ref_INCHIKEY"] = joined["lib_inchikey"]
    out["ref_name"] = joined["lib_name"]
    out["ref_smiles"] = joined["lib_smiles"]
    out["topk"] = joined["rank"].astype(int)
    out["DreaMS_similarity"] = joined["cosine"].astype(float).round(6)
    # 原生 analog_hit = |precursor_mz - ref_precursor_mz|.round(2).abs() >= 0.01
    out["analog_hit"] = (
        (out["precursor_mz"] - out["ref_precursor_mz"]).round(2).abs() >= 0.01
    )

    for c in EXTRA_COLS:
        if c in joined.columns:
            out[c] = joined[c]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", type=Path, required=True,
                   help="我们的 annotations.csv（跑过 FDR 则传 annotations_fdr.csv）")
    p.add_argument("--query-manifest", type=Path, required=True,
                   help="query 嵌入目录里的 manifest.csv（回填 RT / charge）")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    joined = _load_joined(args.annotations, args.query_manifest)
    native = _to_native(joined)

    out_dir: Path = args.out_dir
    per_file = out_dir / "per_file"
    per_file.mkdir(parents=True, exist_ok=True)

    # 合并表（Row 全局 1..N）
    combined = native.copy()
    combined.insert(0, "Row", range(1, len(combined) + 1))
    combined.to_csv(out_dir / "annotations_native.tsv", sep="\t", index=False)

    # 每个 query 文件一个 TSV（Row 文件内 1..N，对齐原生按文件布局）
    manifest_records = []
    for fname, grp in native.groupby("file_name", sort=False):
        sub = grp.copy()
        sub.insert(0, "Row", range(1, len(sub) + 1))
        stem = Path(str(fname)).stem
        sub.to_csv(per_file / f"{stem}.tsv", sep="\t", index=False)
        top1 = sub[sub["topk"] == 1]
        rec = {"file": fname, "n_rows": int(len(sub)),
               "n_top1": int(len(top1))}
        if "mz_pass" in sub.columns:
            rec["n_top1_mz_pass"] = int(top1["mz_pass"].sum())
        manifest_records.append(rec)

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_files = len(manifest_records)
    n_rows = len(native)
    n_queries = int((native["topk"] == 1).sum())
    print(f"[export] {n_rows} 行（{n_queries} 个 query 谱 × topk），{n_files} 个文件", flush=True)
    print(f"[export] combined -> {out_dir / 'annotations_native.tsv'}", flush=True)
    print(f"[export] per-file  -> {per_file}/", flush=True)
    print(f"[export] manifest  -> {out_dir / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
