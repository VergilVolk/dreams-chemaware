#!/usr/bin/env python
"""Fail-closed preflight for BioAware shared-embedding adapter training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import CandidateGraph, sha256_file  # noqa: E402
from train_noise_final_f1_parm import TokenStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data/validation/bioaware_embedding_relation_manifest")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--d0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_d0_manifest")
    parser.add_argument("--f0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_f0_protocol")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_f1_full_tokens")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/bioaware_embedding_adapter_preflight.json")
    args = parser.parse_args()
    required = [
        args.manifest_dir / "report.json", args.manifest_dir / "rows.csv.gz",
        args.manifest_dir / "identity_pairs.csv.gz", args.graph,
        args.d0_dir / "manifest.npz", args.f0_dir / "symmetric_zero_rank.npy",
        args.token_dir / "report.json", args.embedding_cache,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output.exists():
        raise RuntimeError(f"fail-closed: output exists: {args.output}")
    manifest_report = json.loads((args.manifest_dir / "report.json").read_text(encoding="utf-8"))
    if not manifest_report.get("formal") or manifest_report.get("p3_identity_overlap") != 0:
        raise RuntimeError("shared embedding preflight requires formal P3-disjoint manifest")
    graph = CandidateGraph(args.graph)
    store = TokenStore(args.token_dir, args.embedding_cache)
    rows = pd.read_csv(args.manifest_dir / "rows.csv.gz")
    pairs = pd.read_csv(args.manifest_dir / "identity_pairs.csv.gz")
    reachable = set(map(int, store.rows))
    rows = rows[rows["row"].astype(int).isin(reachable)].copy()
    identity_rows = rows.groupby("ik14")["row"].nunique()
    eligible_identity = set(identity_rows[identity_rows >= 2].index.astype(str))
    pair_reachable = pairs[
        pairs["identity_a"].astype(str).isin(eligible_identity)
        & pairs["identity_b"].astype(str).isin(eligible_identity)
    ].copy()
    with np.load(args.d0_dir / "manifest.npz") as body:
        formula_fold = np.asarray(body["formula_fold"], dtype=np.int8)
        baseline_rank = np.asarray(body["baseline_rank"], dtype=np.int16)
    if len(formula_fold) != graph.n_queries:
        raise RuntimeError("D0 formula fold mismatch")
    symmetric_baseline = np.load(args.f0_dir / "symmetric_zero_rank.npy")
    if not np.array_equal(symmetric_baseline, baseline_rank):
        raise RuntimeError("F0 symmetric zero-init ranks differ from frozen D0 baseline")
    relation_count = pair_reachable["relation_type"].value_counts().sort_index().to_dict()
    fold_report = {}
    for fold in range(5):
        train = pair_reachable[
            (pair_reachable["formula_fold_a"] != fold)
            & (pair_reachable["formula_fold_b"] != fold)
        ]
        strict_heldout = pair_reachable[
            (pair_reachable["formula_fold_a"] == fold)
            & (pair_reachable["formula_fold_b"] == fold)
        ]
        fold_report[str(fold)] = {
            "training_pairs": int(len(train)),
            "training_identities": int(len(set(train.identity_a.astype(str)) | set(train.identity_b.astype(str)))),
            "heldout_queries": int(np.sum(formula_fold == fold)),
            "heldout_errors": int(np.sum((formula_fold == fold) & (baseline_rank != 1))),
            "heldout_near": int(np.sum((formula_fold == fold) & graph.query_has_near)),
            "reaction_pairs": int(train.relation_type.astype(str).str.startswith("reaction_").sum()),
            "near_pairs": int(train.relation_type.astype(str).eq("near_isomer").sum()),
            "strict_heldout_relation_pairs": int(len(strict_heldout)),
            "strict_heldout_reaction_pairs": int(
                strict_heldout.relation_type.astype(str).str.startswith("reaction_").sum()
            ),
        }
    report = {
        "status": "bioaware_embedding_adapter_preflight_complete",
        "formal": True,
        "reachable_rows": int(len(rows)),
        "reachable_identities_with_two_spectra": int(len(eligible_identity)),
        "reachable_pairs": int(len(pair_reachable)),
        "relation_counts": {str(k): int(v) for k, v in relation_count.items()},
        "folds": fold_report,
        "contracts": {
            "reaction_neighbour_is_positive": False,
            "same_identity_is_only_retrieval_positive": True,
            "query_reference_encoder_shared": True,
            "P2b": "forbidden",
            "P3": "not opened",
        },
        "gates": {
            "reachable_identities_ge_1000": len(eligible_identity) >= 1000,
            "near_pairs_ge_300": relation_count.get("near_isomer", 0) >= 300,
            "reaction_pairs_ge_50": sum(v for k, v in relation_count.items() if str(k).startswith("reaction_")) >= 50,
            "every_fold_has_100_errors": all(value["heldout_errors"] >= 100 for value in fold_report.values()),
            "every_fold_has_100_near": all(value["heldout_near"] >= 100 for value in fold_report.values()),
            "every_fold_has_10_strict_heldout_reactions": all(
                value["strict_heldout_reaction_pairs"] >= 10
                for value in fold_report.values()
            ),
        },
        "provenance": {
            "manifest_report_sha256": sha256_file(args.manifest_dir / "report.json"),
            "rows_sha256": sha256_file(args.manifest_dir / "rows.csv.gz"),
            "pairs_sha256": sha256_file(args.manifest_dir / "identity_pairs.csv.gz"),
            "graph_sha256": sha256_file(args.graph),
            "f0_symmetric_rank_sha256": sha256_file(args.f0_dir / "symmetric_zero_rank.npy"),
            "embedding_cache_sha256": sha256_file(args.embedding_cache),
        },
        "claim_limit": "Preflight only; no trainable model or performance result.",
    }
    if not all(report["gates"].values()):
        raise RuntimeError(f"BioAware embedding preflight gates failed: {report['gates']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
