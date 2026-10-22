#!/usr/bin/env python3
"""Build phenotype-blind, candidate-specific IBD sample-context evidence.

The query abundance profile is compared only with independent IBD features
that passed the frozen strict library policy for a candidate InChIKey.  The
query itself is never allowed to serve as its own candidate anchor.  All
features are computed on the complementary participants of each outer fold;
diagnosis and held-out effects are not loaded.

This file deliberately exports context *evidence*, not a promoted BioAware
reranker.  A downstream comparison may use only the paired real/wrong-context
edges and must retain the complete query denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_axis(score_bundle: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bundle = np.load(score_bundle, allow_pickle=False)
    query_ptr = np.asarray(bundle["query_ptr"], dtype=np.int64)
    query_ids = np.asarray(bundle["query_ids"]).astype("U")
    candidate_ids = np.asarray(bundle["candidate_ids"]).astype("U")
    if len(query_ptr) != len(query_ids) + 1 or query_ptr[0] != 0:
        raise RuntimeError("invalid candidate query pointer")
    if query_ptr[-1] != len(candidate_ids) or np.any(np.diff(query_ptr) < 0):
        raise RuntimeError("candidate axis length differs from query pointer")
    return query_ptr, query_ids, candidate_ids


def rank_average(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling, without scipy."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def pair_evidence(
    query_values: np.ndarray,
    query_detected: np.ndarray,
    anchor_values: np.ndarray,
    anchor_detected: np.ndarray,
    *,
    minimum_joint: int,
) -> tuple[float, float, float] | None:
    union = query_detected | anchor_detected
    joint = query_detected & anchor_detected
    n_union = int(union.sum())
    n_joint = int(joint.sum())
    if n_union == 0 or n_joint < minimum_joint:
        return None
    jaccard = float(n_joint / n_union)
    q = rank_average(query_values[joint])
    a = rank_average(anchor_values[joint])
    if np.std(q) == 0.0 or np.std(a) == 0.0:
        return None
    rho = float(np.corrcoef(q, a)[0, 1])
    if not np.isfinite(rho):
        return None
    # Signed and bounded.  Negative co-variation remains evidence against an
    # identity-like event rather than being silently clipped away.
    return jaccard, rho, jaccard * rho


def aggregate_anchor_evidence(
    query_index: int,
    anchor_indices: list[int],
    values: np.ndarray,
    detected: np.ndarray,
    training_mask: np.ndarray,
    *,
    minimum_joint: int,
) -> tuple[float, float, float, int] | None:
    rows: list[tuple[float, float, float]] = []
    for anchor_index in anchor_indices:
        result = pair_evidence(
            values[training_mask, query_index],
            detected[training_mask, query_index],
            values[training_mask, anchor_index],
            detected[training_mask, anchor_index],
            minimum_joint=minimum_joint,
        )
        if result is not None:
            rows.append(result)
    if not rows:
        return None
    matrix = np.asarray(rows, dtype=np.float64)
    medians = np.median(matrix, axis=0)
    return float(medians[0]), float(medians[1]), float(medians[2]), len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-dir", required=True, type=Path)
    parser.add_argument("--identity-partition", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-joint", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.minimum_joint < 3:
        raise ValueError("minimum joint detections must be at least three")

    score_path = args.slice_dir / "wse_candidate_scores.npz"
    context_path = args.slice_dir / "phenotype_blind_sample_context.npz"
    wrong_path = args.slice_dir / "degree_preserving_wrong_context.tsv.gz"
    for path in (score_path, context_path, wrong_path, args.identity_partition):
        if not path.is_file():
            raise FileNotFoundError(path)

    query_ptr, query_ids, candidate_ids = candidate_axis(score_path)
    edge_query_ids = np.repeat(query_ids, np.diff(query_ptr))

    context = np.load(context_path, allow_pickle=False)
    sample_ids = np.asarray(context["sample_id"]).astype("U")
    entity_ids = np.asarray(context["entity_id"]).astype("U")
    values = np.asarray(context["robust_clr"], dtype=np.float64)
    detected = np.asarray(context["detected"], dtype=bool)
    sample_folds = np.asarray(context["outer_fold"], dtype=np.int64)
    if values.shape != detected.shape or values.shape != (len(sample_ids), len(entity_ids)):
        raise RuntimeError("sample-context arrays are not aligned")
    if set(query_ids) != set(entity_ids):
        raise RuntimeError("candidate queries and sample-context entities differ")
    entity_index = {entity_id: i for i, entity_id in enumerate(entity_ids)}

    identity = pd.read_csv(args.identity_partition, sep="\t", dtype=str)
    required_identity = {"entity_id", "library_inchikey", "library_supported"}
    if not required_identity.issubset(identity.columns):
        raise RuntimeError("identity partition lacks candidate-anchor fields")
    supported = identity["library_supported"].str.lower().eq("true")
    identity = identity.loc[supported & identity["library_inchikey"].notna()].copy()
    identity = identity.loc[identity["entity_id"].isin(entity_index)].copy()
    anchor_groups = {
        str(candidate_id): sorted(set(group["entity_id"].astype(str)))
        for candidate_id, group in identity.groupby("library_inchikey", sort=True)
    }

    wrong = pd.read_csv(wrong_path, sep="\t", dtype={"outer_fold": int})
    required_wrong = {"outer_fold", "recipient_entity_id", "donor_entity_id", "fixed_point"}
    if not required_wrong.issubset(wrong.columns):
        raise RuntimeError("wrong-context ledger lacks required fields")
    if wrong["fixed_point"].astype(str).str.lower().eq("true").any():
        raise RuntimeError("wrong-context ledger contains fixed points")
    wrong_map = {
        (int(row.outer_fold), str(row.recipient_entity_id)): str(row.donor_entity_id)
        for row in wrong.itertuples(index=False)
    }

    folds = np.asarray(sorted(set(sample_folds.tolist())), dtype=np.int64)
    shape = (len(folds), len(candidate_ids))
    real_jaccard = np.full(shape, np.nan, dtype=np.float32)
    real_spearman = np.full(shape, np.nan, dtype=np.float32)
    real_support = np.full(shape, np.nan, dtype=np.float32)
    wrong_jaccard = np.full(shape, np.nan, dtype=np.float32)
    wrong_spearman = np.full(shape, np.nan, dtype=np.float32)
    wrong_support = np.full(shape, np.nan, dtype=np.float32)
    real_anchor_count = np.zeros(shape, dtype=np.int16)
    wrong_anchor_count = np.zeros(shape, dtype=np.int16)

    for fold_index, fold in enumerate(folds):
        training = sample_folds != fold
        for edge_index, (query_id, candidate_id) in enumerate(zip(edge_query_ids, candidate_ids)):
            anchors = [
                entity_index[anchor]
                for anchor in anchor_groups.get(str(candidate_id), [])
                if anchor != str(query_id)
            ]
            if not anchors:
                continue
            real = aggregate_anchor_evidence(
                entity_index[str(query_id)], anchors, values, detected, training,
                minimum_joint=args.minimum_joint,
            )
            donor_id = wrong_map.get((int(fold), str(query_id)))
            if donor_id is None or donor_id not in entity_index:
                raise RuntimeError(f"missing wrong context for fold={fold}, entity={query_id}")
            wrong_result = aggregate_anchor_evidence(
                entity_index[donor_id], anchors, values, detected, training,
                minimum_joint=args.minimum_joint,
            )
            if real is not None:
                real_jaccard[fold_index, edge_index] = real[0]
                real_spearman[fold_index, edge_index] = real[1]
                real_support[fold_index, edge_index] = real[2]
                real_anchor_count[fold_index, edge_index] = real[3]
            if wrong_result is not None:
                wrong_jaccard[fold_index, edge_index] = wrong_result[0]
                wrong_spearman[fold_index, edge_index] = wrong_result[1]
                wrong_support[fold_index, edge_index] = wrong_result[2]
                wrong_anchor_count[fold_index, edge_index] = wrong_result[3]

    real_available = np.isfinite(real_support)
    wrong_available = np.isfinite(wrong_support)
    paired_available = real_available & wrong_available
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "candidate_context_evidence.npz"
    np.savez_compressed(
        output,
        schema=np.asarray("ibd200_plate3_candidate_context_evidence_v1"),
        query_ptr=query_ptr,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        outer_folds=folds,
        real_detection_jaccard=real_jaccard,
        real_abundance_spearman=real_spearman,
        real_context_support=real_support,
        real_anchor_count=real_anchor_count,
        wrong_detection_jaccard=wrong_jaccard,
        wrong_abundance_spearman=wrong_spearman,
        wrong_context_support=wrong_support,
        wrong_anchor_count=wrong_anchor_count,
        real_context_available=real_available,
        wrong_context_available=wrong_available,
        paired_context_available=paired_available,
    )

    ledger_rows: list[pd.DataFrame] = []
    for fold_index, fold in enumerate(folds):
        ledger_rows.append(pd.DataFrame({
            "outer_fold": int(fold),
            "entity_id": edge_query_ids,
            "candidate_id": candidate_ids,
            "real_context_support": real_support[fold_index],
            "wrong_context_support": wrong_support[fold_index],
            "paired_context_available": paired_available[fold_index],
            "real_anchor_count": real_anchor_count[fold_index],
            "wrong_anchor_count": wrong_anchor_count[fold_index],
        }))
    ledger = pd.concat(ledger_rows, ignore_index=True)
    ledger_path = args.output_dir / "candidate_context_ledger.tsv.gz"
    ledger.to_csv(
        ledger_path, sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    paired_query_counts = []
    for fold_index in range(len(folds)):
        paired_query_counts.append(int(np.unique(edge_query_ids[paired_available[fold_index]]).size))
    report = {
        "schema": "ibd200_plate3_candidate_context_evidence_v1",
        "status": "PHENOTYPE_BLIND_CANDIDATE_SPECIFIC_CONTEXT_EVIDENCE_COMPLETE",
        "data_role": "development_consumed",
        "denominators": {
            "samples": int(len(sample_ids)),
            "query_entities": int(len(query_ids)),
            "candidate_edges": int(len(candidate_ids)),
            "strict_anchor_entities": int(identity["entity_id"].nunique()),
            "strict_anchor_structures": int(len(anchor_groups)),
            "outer_folds": int(len(folds)),
            "real_available_edges_by_fold": [int(row.sum()) for row in real_available],
            "wrong_available_edges_by_fold": [int(row.sum()) for row in wrong_available],
            "paired_available_edges_by_fold": [int(row.sum()) for row in paired_available],
            "paired_available_queries_by_fold": paired_query_counts,
        },
        "method": {
            "training_samples": "participants outside the evaluated outer fold only",
            "candidate_anchor": "independent IBD entity with frozen strict GNPS Gold/Silver library support for candidate InChIKey",
            "self_anchor": "forbidden",
            "features": "median detection Jaccard, joint-detection Spearman correlation, and their signed product",
            "minimum_joint_detections": int(args.minimum_joint),
            "wrong_context": "frozen whole-profile permutation within exact training detection degree",
            "fair_evaluation_subset": "paired_context_available only; all other candidate edges remain explicit context abstentions",
        },
        "leakage_status": (
            "diagnosis, beta, SE, contrast label, held-out effects and retrieval truth were not loaded; "
            "only complementary-fold sample profiles and frozen phenotype-blind library anchors were used"
        ),
        "provenance": {
            "score_bundle_sha256": sha256(score_path),
            "sample_context_sha256": sha256(context_path),
            "wrong_context_sha256": sha256(wrong_path),
            "identity_partition_sha256": sha256(args.identity_partition),
            "evidence_npz_sha256": sha256(output),
            "ledger_sha256": sha256(ledger_path),
            "implementation_sha256": sha256(Path(__file__)),
        },
        "claim_boundary": [
            "This is candidate-specific event evidence, not a validated BioAware performance gain.",
            "Strict library support is an anchor policy, not MSI level-1 identity.",
            "Missing candidate context is abstention and must never be converted to negative evidence.",
            "Promotion requires real context to outperform degree-preserving wrong context on downstream held-out biological answers.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
