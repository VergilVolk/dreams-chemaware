"""Evaluate official DreaMS retrieval on the fixed HNSCC connectivity panel.

The program is deliberately restricted to published molecules with (1) direct
same-study QC-MS2, (2) a unique connectivity truth, and (3) that truth present
in the precursor-mass candidate pool.  It reports every spectrum-level rank;
this is an external retrieval benchmark, not a biological discovery claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ik14(value: object) -> str:
    return str(value).split("-", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=Path("data/external/MTBLS1905/reference/blind_connectivity_panel.tsv"))
    parser.add_argument("--target-matches", type=Path, default=Path("data/external/MTBLS1905/qc_ms2/audit/published_target_qc_ms2_matches.tsv"))
    parser.add_argument("--candidate-map", type=Path, default=Path("data/external/MTBLS1905/reference/hnscc_positive_mass_candidate_map.tsv"))
    parser.add_argument("--query-dir", type=Path, default=Path("data/external/MTBLS1905/qc_ms2/dreams_official_full"))
    parser.add_argument("--reference-dir", type=Path, default=Path("data/external/MTBLS1905/reference/hnscc_positive_reference_dreams"))
    parser.add_argument("--out", type=Path, default=Path("data/external/MTBLS1905/evaluation/official_dreams_blind_retrieval.tsv"))
    args = parser.parse_args()
    panel = pd.read_csv(args.panel, sep="\t")
    panel = panel[panel.panel_status.eq("evaluable")].set_index("metabolite")
    matches = pd.read_csv(args.target_matches, sep="\t")
    matches = matches[matches.metabolite.isin(panel.index)].copy()
    candidate_map = pd.read_csv(args.candidate_map, sep="\t")
    query_manifest = pd.read_csv(args.query_dir / "manifest.csv")
    query_embeddings = np.load(args.query_dir / "official_embeddings.npy", mmap_mode="r")
    ref_manifest = pd.read_csv(args.reference_dir / "manifest.csv")
    ref_embeddings = np.load(args.reference_dir / "embeddings.npy", mmap_mode="r")
    if len(ref_manifest) != len(ref_embeddings) or len(query_manifest) != len(query_embeddings):
        raise RuntimeError("Embedding/manifest rows mismatch")
    query_index = {(r.source_file, r.spectrum_id): i for i, r in query_manifest.iterrows()}
    ref_keys = ref_manifest.inchikey.map(ik14).to_numpy()
    rows: list[dict] = []
    for hit in matches.itertuples(index=False):
        qid = query_index.get((hit.source_file, hit.spectrum_id))
        if qid is None:
            raise RuntimeError(f"Published QC spectrum absent from full query embedding: {hit.spectrum_id}")
        truth = panel.loc[hit.metabolite, "truth_ik14"]
        cand_rows = candidate_map[candidate_map.target_metabolite.eq(hit.metabolite)].library_record_index.unique()
        cand_rows = np.asarray(sorted(cand_rows), dtype=int)
        valid_truth = cand_rows[ref_keys[cand_rows] == truth]
        if not len(valid_truth):
            raise RuntimeError(f"Truth vanished from candidate set for {hit.metabolite}")
        scores = np.asarray(ref_embeddings[cand_rows] @ query_embeddings[qid], dtype=float)
        descending = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(cand_rows), dtype=int)
        ranks[descending] = np.arange(1, len(cand_rows) + 1)
        truth_positions = np.where(np.isin(cand_rows, valid_truth))[0]
        rank = int(ranks[truth_positions].min())
        best_pos = int(descending[0])
        top_row = int(cand_rows[best_pos])
        rows.append({
            "metabolite": hit.metabolite, "query_source_file": hit.source_file, "query_spectrum_id": hit.spectrum_id,
            "candidate_count": int(len(cand_rows)), "truth_reference_count": int(len(valid_truth)),
            "truth_rank": rank, "top1_correct_connectivity": bool(rank == 1),
            "top5_correct_connectivity": bool(rank <= 5), "top1_score": float(scores[best_pos]),
            "truth_best_score": float(scores[truth_positions].max()),
            "top1_name": ref_manifest.iloc[top_row].get("name", ""), "top1_ik14": ref_keys[top_row],
            "truth_ik14": truth,
        })
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    by_target = out.groupby("metabolite", as_index=False).agg(
        queries=("truth_rank", "size"), best_rank=("truth_rank", "min"), mean_rank=("truth_rank", "mean"),
        top1_any=("top1_correct_connectivity", "max"), top5_any=("top5_correct_connectivity", "max"),
    )
    by_target.to_csv(args.out.with_name(args.out.stem + "_by_target.tsv"), sep="\t", index=False)
    report = {
        "scope": "external same-study QC-DDA published-connectivity blind retrieval",
        "n_query_spectra": int(len(out)), "n_targets": int(out.metabolite.nunique()),
        "spectrum_top1_accuracy": float(out.top1_correct_connectivity.mean()),
        "spectrum_top5_accuracy": float(out.top5_correct_connectivity.mean()),
        "target_top1_any_accuracy": float(by_target.top1_any.mean()),
        "target_top5_any_accuracy": float(by_target.top5_any.mean()),
        "warning": "No conclusion about additional annotations or biological mechanism is permitted from this known-target benchmark alone.",
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
