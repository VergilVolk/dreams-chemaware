#!/usr/bin/env python
"""Fixed MetDNA3-style data+knowledge BioAware development evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from annotation.bioaware import BioAwareConfig, fuse_candidates, top1_transition_table  # noqa: E402


def noisy_or(values) -> float:
    x = np.clip(np.asarray(list(values), dtype=float), 0.0, 1.0)
    return float(1.0 - np.prod(1.0 - x)) if len(x) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data/validation/bioaware_metdna3_dreams_cache_v1"))
    parser.add_argument("--embedding", type=Path, default=Path("data/validation/bioaware_metdna3_data_layer_embeddings.npz"))
    parser.add_argument("--scores", type=Path, default=Path("data/validation/bioaware_metdna3_dreams_official_v1/candidate_scores.csv.gz"))
    parser.add_argument("--paths", type=Path, default=Path("data/validation/bioaware_metdna3_development_eval_v1/evidence_paths.csv.gz"))
    parser.add_argument("--splits", type=Path, default=Path("data/validation/bioaware_metdna3_development_v1/identity_splits.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/bioaware_metdna3_two_layer_development_v1"))
    parser.add_argument("--minimum-data-similarity", type=float, default=0.5)
    args = parser.parse_args()
    manifest = pd.read_csv(args.cache_dir / "external_spectra.csv.gz")
    embedding = np.load(args.embedding, allow_pickle=False)["embedding"]
    if len(manifest) != len(embedding):
        raise RuntimeError("data-layer manifest/embedding mismatch")
    scores = pd.read_csv(args.scores)
    paths = pd.read_csv(args.paths)
    splits = pd.read_csv(args.splits)
    query_map = pd.read_csv(args.cache_dir / "queries.csv.gz").set_index("query_id")
    spectrum_position = {key: pos for pos, key in enumerate(manifest["spectrum_key"])}
    identity_positions = {
        identity: group.index.to_numpy(int)
        for identity, group in manifest.reset_index().groupby("truth_ik14", sort=False)
    }
    config = BioAwareConfig()
    transition_frames = []
    reports = []
    detail_frames = []
    for fold in range(10):
        heldout = set(splits[(splits.fold == fold) & (splits.role == "heldout")].ik14)
        seeds = set(splits[(splits.fold == fold) & (splits.role == "seed")].ik14) & set(identity_positions)
        fold_scores = scores[scores.truth_candidate_id.isin(heldout)].copy()
        fold_paths = paths[paths.fold == fold].copy()
        support_rows = []
        for query_id, candidate_group in fold_scores.groupby("query_id", sort=False):
            query_key = str(query_map.loc[query_id, "spectrum_key"])
            q = embedding[spectrum_position[query_key]]
            seed_similarity = {
                identity: float(np.max(embedding[identity_positions[identity]] @ q))
                for identity in seeds
            }
            local_paths = fold_paths[fold_paths.query_id == query_id].copy()
            if len(local_paths):
                local_paths["data_similarity"] = local_paths.seed_compound_id.map(seed_similarity).fillna(-1.0)
                local_paths = local_paths[local_paths.data_similarity >= args.minimum_data_similarity].copy()
                local_paths["two_layer_contribution"] = (
                    local_paths.contribution.astype(float) * local_paths.data_similarity.astype(float)
                )
                complete = local_paths.source_side_complete.astype(bool)
                local_paths["dependency_key"] = np.where(
                    complete,
                    "complete_seed:" + local_paths.seed_compound_id.astype(str),
                    "missing:" + local_paths.missing_source_signature.fillna("").astype(str),
                )
                detail_frames.append(local_paths)
            for candidate_id in candidate_group.candidate_id.astype(str):
                selected = local_paths[local_paths.query_candidate_id.astype(str) == candidate_id]
                grouped = selected.groupby("dependency_key").two_layer_contribution.max() if len(selected) else []
                support_rows.append({
                    "query_id": query_id, "candidate_id": candidate_id,
                    "network_support": noisy_or(grouped), "network_path_count": int(len(selected)),
                })
        attached = fold_scores.merge(pd.DataFrame(support_rows), on=["query_id", "candidate_id"], validate="one_to_one")
        scored, decisions = fuse_candidates(attached, config)
        transitions, result = top1_transition_table(scored, truth_col="truth_candidate_id")
        transitions["fold"] = fold
        transition_frames.append(transitions)
        reports.append({"fold": fold, "seed_identities_with_external_ms2": len(seeds), **result,
                        "queries_with_network_evidence": int(decisions.network_available.sum())})
        print(f"[two-layer {fold}] C/I={result['corrected']}/{result['introduced']}", flush=True)
    transition = pd.concat(transition_frames, ignore_index=True)
    corrected = int(transition.corrected.sum())
    introduced = int(transition.introduced.sum())
    report = {
        "status": "bioaware_metdna3_two_layer_development_complete", "formal": True,
        "protocol": "DreaMS data-layer cosine >=0.5 AND dependency-corrected one-hop Rhea evidence",
        "folds": reports,
        "combined": {
            "instances": int(len(transition)),
            "baseline_recall1": float(transition.baseline_correct.mean()),
            "bioaware_recall1": float(transition.final_correct.mean()),
            "delta_recall1": float((transition.final_correct.astype(int)-transition.baseline_correct.astype(int)).mean()),
            "corrected": corrected, "introduced": introduced,
        },
        "configuration": {**config.to_dict(), "minimum_data_similarity": args.minimum_data_similarity},
        "gates": {"corrected_gt_introduced": corrected > introduced, "risk_net_positive": corrected - 2*introduced > 0},
        "contracts": {"phenotype_blind": True, "P2b": "forbidden", "RP_or_external_test_opened": False,
                      "threshold_source": "MetDNA3 published data-layer threshold; not fitted here"},
        "claim_limit": "HILIC development mechanism audit; not locked validation or SOTA evidence.",
    }
    report["gates"]["pass_to_next_development_gate"] = all(report["gates"].values())
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError(f"fail-closed: {out}")
    transition.to_csv(out / "transitions.csv.gz", index=False, compression="gzip")
    if detail_frames:
        pd.concat(detail_frames, ignore_index=True).to_csv(out / "evidence_paths.csv.gz", index=False, compression="gzip")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

