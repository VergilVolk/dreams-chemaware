#!/usr/bin/env python
"""Link DreaMS library annotations to uniformly quantified MTBLS13729 MS1 features.

This is the bridge between identification evidence (MS2) and abundance (MS1).
It never uses MS2 spectral counts as abundance and never promotes a library hit
to Level 1 without an authentic standard and retention-time match.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def link_queries_to_targets(
    hits: pd.DataFrame,
    targets: pd.DataFrame,
    ppm: float,
    rt_sec: float,
) -> pd.DataFrame:
    rows = []
    target_by_sample: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    # The same consensus targets apply to every sample. Keeping this dictionary
    # makes the interface explicit and leaves room for sample-specific exclusions.
    target_sorted = targets.sort_values("mz").reset_index(drop=True)
    target_mz = target_sorted["mz"].to_numpy(float)
    for row in hits.itertuples(index=False):
        mz = float(row.query_precursor_mz)
        tol = mz * ppm * 1e-6
        lo = bisect.bisect_left(target_mz, mz - tol)
        hi = bisect.bisect_right(target_mz, mz + tol)
        if lo == hi:
            continue
        candidates = target_sorted.iloc[lo:hi].copy()
        candidates["feature_dppm"] = np.abs(candidates["mz"] - mz) / candidates["mz"] * 1e6
        candidates["feature_drt_sec"] = np.abs(candidates["rt_sec"] - float(row.query_rt_sec))
        candidates = candidates[candidates["feature_drt_sec"] <= rt_sec]
        if candidates.empty:
            continue
        candidates["link_cost"] = (candidates["feature_dppm"] / ppm) ** 2 + (candidates["feature_drt_sec"] / rt_sec) ** 2
        chosen = candidates.sort_values("link_cost").iloc[0]
        record = row._asdict()
        record.update(
            {
                "feature_id": int(chosen.feature_id),
                "feature_mz": float(chosen.mz),
                "feature_rt_sec": float(chosen.rt_sec),
                "feature_dppm": float(chosen.feature_dppm),
                "feature_drt_sec": float(chosen.feature_drt_sec),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def collapse_feature_annotations(linked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if linked.empty:
        return pd.DataFrame(), pd.DataFrame()
    linked = linked.copy()
    linked["ik14"] = linked["lib_inchikey"].fillna("").astype(str).str[:14]
    candidate = (
        linked.groupby(["feature_id", "ik14"], dropna=False)
        .agg(
            lib_inchikey=("lib_inchikey", "first"),
            lib_name=("lib_name", "first"),
            lib_smiles=("lib_smiles", "first"),
            n_support_spectra=("query_idx", "nunique"),
            n_support_samples=("query_file", "nunique"),
            max_cosine=("cosine", "max"),
            median_cosine=("cosine", "median"),
            min_qvalue=("qvalue", "min") if "qvalue" in linked else ("cosine", lambda _: math.nan),
            median_feature_dppm=("feature_dppm", "median"),
            median_feature_drt_sec=("feature_drt_sec", "median"),
        )
        .reset_index()
    )
    candidate["candidate_score"] = candidate["max_cosine"] + 0.02 * np.log1p(candidate["n_support_spectra"]) + 0.02 * np.log1p(candidate["n_support_samples"])
    candidate = candidate.sort_values(["feature_id", "candidate_score", "max_cosine"], ascending=[True, False, False])

    best_rows = []
    for feature_id, group in candidate.groupby("feature_id", sort=False):
        best = group.iloc[0]
        total_spectra = int(linked.loc[linked["feature_id"] == feature_id, "query_idx"].nunique())
        agreement = float(best.n_support_spectra / total_spectra) if total_spectra else 0.0
        distinct = int(group["ik14"].replace("", np.nan).nunique())
        if best.max_cosine >= 0.8 and best.n_support_spectra >= 2 and agreement >= 0.6:
            tier = "Level 2a-supported"
        elif best.max_cosine >= 0.7:
            tier = "Level 2a-single/ambiguous"
        elif best.max_cosine >= 0.5:
            tier = "Level 3-candidate"
        else:
            tier = "unassigned"
        best_rows.append(
            {
                "feature_id": int(feature_id),
                "best_inchikey": best.lib_inchikey,
                "best_ik14": best.ik14,
                "best_name": best.lib_name,
                "best_smiles": best.lib_smiles,
                "max_cosine": float(best.max_cosine),
                "median_cosine": float(best.median_cosine),
                "n_support_spectra": int(best.n_support_spectra),
                "n_support_samples": int(best.n_support_samples),
                "n_linked_ms2": total_spectra,
                "structure_agreement_fraction": agreement,
                "n_distinct_ik14_candidates": distinct,
                "annotation_evidence_tier": tier,
                "min_qvalue_unvalidated": float(best.min_qvalue) if np.isfinite(best.min_qvalue) else math.nan,
                "confidence_note": "No authentic standard/RT confirmation; never Level 1. Decoy q-values are retained for audit but not used as a 1% FDR claim.",
            }
        )
    return candidate, pd.DataFrame(best_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, default=Path("data/mtbls13729/annotation"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/mtbls13729/embeddings"))
    parser.add_argument("--consensus-dir", type=Path, default=Path("data/mtbls13729/ms1_consensus"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/ms1_ms2_link"))
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--rt-sec", type=float, default=20.0)
    parser.add_argument("--min-cosine", type=float, default=0.5)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "complete", "panels": {}}
    for panel in args.panels:
        annotation_path = args.annotation_root / panel / "annotations_fdr.csv"
        if not annotation_path.exists():
            annotation_path = args.annotation_root / panel / "annotations.csv"
        manifest_path = args.embedding_root / panel / "manifest.csv"
        targets_path = args.consensus_dir / f"{panel}__requantification_targets.csv.gz"
        if not annotation_path.exists() or not manifest_path.exists() or not targets_path.exists():
            report["panels"][panel] = {"status": "missing_inputs", "annotations": str(annotation_path), "manifest": str(manifest_path), "targets": str(targets_path)}
            continue
        hits = pd.read_csv(annotation_path)
        manifest = pd.read_csv(manifest_path).reset_index().rename(columns={"index": "query_idx"})
        rt_values = pd.to_numeric(manifest["RT"], errors="coerce")
        # Current converter stores minutes; auto-detect avoids silently multiplying
        # data that have already been exported in seconds.
        manifest["query_rt_sec"] = rt_values * 60.0 if rt_values.quantile(0.99) < 100 else rt_values
        hits = hits.merge(manifest[["query_idx", "query_rt_sec"]], on="query_idx", how="left", validate="many_to_one")
        # Rank is already mass-constrained after the retrieval protocol fix.
        hits = hits[(hits["rank"] == 1) & hits["mz_pass"].astype(bool) & (hits["cosine"] >= args.min_cosine)]
        targets = pd.read_csv(targets_path)
        linked = link_queries_to_targets(hits, targets, args.ppm, args.rt_sec)
        linked.to_csv(out / f"{panel}__linked_ms2.csv.gz", index=False)
        candidates, best = collapse_feature_annotations(linked)
        candidates.to_csv(out / f"{panel}__feature_annotation_candidates.csv.gz", index=False)
        best.to_csv(out / f"{panel}__feature_best_annotations.csv.gz", index=False)
        report["panels"][panel] = {
            "status": "complete",
            "n_mass_constrained_rank1_hits": len(hits),
            "n_linked_hits": len(linked),
            "link_fraction": float(len(linked) / len(hits)) if len(hits) else 0.0,
            "n_annotated_features": int(best["feature_id"].nunique()) if len(best) else 0,
            "n_level2a_supported": int((best["annotation_evidence_tier"] == "Level 2a-supported").sum()) if len(best) else 0,
            "n_level2a_single_or_ambiguous": int((best["annotation_evidence_tier"] == "Level 2a-single/ambiguous").sum()) if len(best) else 0,
            "confidence_boundary": "No Level 1 without authentic standard and RT confirmation; TDA q-values are not treated as calibrated 1% FDR.",
        }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
