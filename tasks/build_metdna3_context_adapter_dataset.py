"""Build the 10-rotation MetDNA3 HILIC context-adapter development set."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TASKS = Path(__file__).resolve().parent
for location in (ROOT, TASKS):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from build_bioaware_context_evidence_tensor import dependency_key, relation_name, RELATION_TYPES  # noqa: E402


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-scores", type=Path, default=Path("data/validation/bioaware_metdna3_dreams_official_v1/candidate_scores.csv.gz"))
    parser.add_argument("--official-embeddings", type=Path, default=Path("data/validation/bioaware_metdna3_dreams_official_v1/embeddings.npz"))
    parser.add_argument("--query-manifest", type=Path, default=Path("data/validation/bioaware_metdna3_dreams_cache_v2/queries.csv.gz"))
    parser.add_argument("--identity-splits", type=Path, default=Path("data/validation/bioaware_metdna3_development_v1/identity_splits.csv.gz"))
    parser.add_argument("--evidence-paths", type=Path, default=Path("data/validation/bioaware_metdna3_development_eval_noop_filtered_20260830/evidence_paths.csv.gz"))
    parser.add_argument("--seed-dir", type=Path, default=Path("data/validation/bioaware_metdna3_context_seed_embeddings_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/bioaware_metdna3_context_adapter_dataset_v1"))
    parser.add_argument("--max-context-edges", type=int, default=16)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    for path in (
        args.candidate_scores, args.official_embeddings, args.query_manifest,
        args.identity_splits, args.evidence_paths, args.seed_dir / "manifest.csv.gz",
        args.seed_dir / "embeddings.npy", args.seed_dir / "report.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    seed_report = json.loads((args.seed_dir / "report.json").read_text(encoding="utf-8"))
    if not seed_report.get("formal") or seed_report.get("contracts", {}).get("P2b") != "forbidden":
        raise RuntimeError("invalid seed embedding contract")

    candidates = pd.read_csv(args.candidate_scores)
    queries = pd.read_csv(args.query_manifest)
    splits = pd.read_csv(args.identity_splits)
    paths = pd.read_csv(args.evidence_paths)
    official = np.load(args.official_embeddings)
    query_embeddings = normalize(official["query_embedding"])
    reference_embeddings = normalize(official["reference_embedding"])
    reference_rows = official["reference_rows"].astype(np.int64)
    reference_position = {int(row): i for i, row in enumerate(reference_rows)}
    if len(queries) != len(query_embeddings) or queries.query_id.duplicated().any():
        raise RuntimeError("query embedding/manifest mismatch")
    query_position = dict(zip(queries.query_id.astype(str), range(len(queries))))

    seed_manifest = pd.read_csv(args.seed_dir / "manifest.csv.gz")
    seed_embeddings = normalize(np.load(args.seed_dir / "embeddings.npy"))
    if len(seed_manifest) != len(seed_embeddings):
        raise RuntimeError("seed manifest/embedding mismatch")
    prototype_ids: list[str] = []
    prototypes: list[np.ndarray] = []
    for identity, group in seed_manifest.groupby(seed_manifest.truth_ik14.astype(str).str[:14], sort=False):
        prototype_ids.append(str(identity))
        prototypes.append(normalize(seed_embeddings[group.index].mean(axis=0, keepdims=True))[0])
    # Some valid Level-1 network seeds do not have a recoverable raw MS2 row in
    # the downloaded development files.  Missing spectrum is an explicit state,
    # not a fabricated prototype and not a reason to delete the reaction edge.
    missing_prototype_id = "__MISSING_SPECTRUM__"
    prototype_ids.append(missing_prototype_id)
    prototypes.append(np.zeros(seed_embeddings.shape[1], dtype=np.float32))
    prototype_index = {value: i for i, value in enumerate(prototype_ids)}

    paths = paths.copy()
    paths["dependency_key"] = paths.apply(dependency_key, axis=1)
    paths["relation_name"] = paths.apply(relation_name, axis=1)
    paths["relation_type"] = paths.relation_name.map(RELATION_TYPES).astype(np.int16)
    paths["path_confidence"] = np.clip(paths.specificity_weighted_contribution.astype(float), 0, 1)
    paths["experimental_support"] = np.clip(paths.seed_score.astype(float), 0, 1)
    paths["reaction_completeness"] = np.sqrt(
        np.clip(paths.source_side_completeness.astype(float), 0, 1)
        * np.clip(paths.target_side_completeness.astype(float), 0, 1)
    )
    paths["conflict"] = np.maximum(
        paths.curated_direction_conflicted.astype(bool).astype(float),
        1.0 - np.clip(paths.candidate_specificity.astype(float), 0, 1),
    )
    paths = paths.sort_values(
        ["fold", "query_id", "candidate_id", "seed_compound_id", "dependency_key", "path_confidence"],
        ascending=[True, True, True, True, True, False], kind="stable",
    ).drop_duplicates(
        ["fold", "query_id", "candidate_id", "seed_compound_id", "dependency_key"], keep="first",
    )

    split_role = {
        (int(row.fold), str(row.ik14)): str(row.role)
        for row in splits.itertuples(index=False)
    }
    candidate_groups = {str(key): group.reset_index(drop=True) for key, group in candidates.groupby("query_id", sort=False)}
    instance_rows = []
    for fold in sorted(splits.fold.unique()):
        for query_id, group in candidate_groups.items():
            truth = str(group.truth_candidate_id.iloc[0])
            if split_role.get((int(fold), truth)) == "heldout":
                instance_rows.append((int(fold), query_id))
    if len(instance_rows) != 819:
        raise RuntimeError(f"expected 819 frozen rotation instances, found {len(instance_rows)}")

    flat_candidate_ids: list[str] = []
    flat_candidates: list[np.ndarray] = []
    flat_seed_indices: list[np.ndarray] = []
    flat_relations: list[np.ndarray] = []
    flat_features: list[np.ndarray] = []
    flat_masks: list[np.ndarray] = []
    instance_query_ids: list[str] = []
    formulas: list[str] = []
    query_identities: list[str] = []
    folds: list[int] = []
    query_vectors: list[np.ndarray] = []
    positive_indices: list[int] = []
    offsets = [0]
    truncated = 0
    missing_seed_prototypes = set()
    leakage_edges = 0
    for fold, query_id in instance_rows:
        group = candidate_groups[query_id]
        truth = str(group.truth_candidate_id.iloc[0])
        positive = np.flatnonzero(group.candidate_id.astype(str).to_numpy() == truth)
        if len(positive) != 1:
            raise RuntimeError(f"{query_id}: expected one positive")
        instance_query_ids.append(query_id)
        formulas.append(str(group.truth_formula.iloc[0]))
        query_identities.append(truth)
        folds.append(fold)
        query_vectors.append(query_embeddings[query_position[query_id]])
        positive_indices.append(int(positive[0]))
        for candidate in group.itertuples(index=False):
            ref_row = int(candidate.best_reference_row)
            if ref_row not in reference_position:
                raise RuntimeError(f"missing best reference row {ref_row}")
            flat_candidate_ids.append(str(candidate.candidate_id))
            flat_candidates.append(reference_embeddings[reference_position[ref_row]])
            local = paths[
                (paths.fold.astype(int) == fold)
                & (paths.query_id.astype(str) == query_id)
                & (paths.candidate_id.astype(str) == str(candidate.candidate_id))
            ].sort_values(
                ["path_confidence", "reaction_completeness"], ascending=False, kind="stable",
            )
            if len(local) > args.max_context_edges:
                truncated += 1
            local = local.head(args.max_context_edges)
            seed_indices = np.zeros(args.max_context_edges, dtype=np.int64)
            relation_types = np.zeros(args.max_context_edges, dtype=np.int64)
            edge_features = np.zeros((args.max_context_edges, 4), dtype=np.float32)
            edge_masks = np.zeros(args.max_context_edges, dtype=bool)
            for position, edge in enumerate(local.itertuples(index=False)):
                seed_identity = str(edge.seed_compound_id)
                if seed_identity == truth or split_role.get((fold, seed_identity)) != "seed":
                    leakage_edges += 1
                    continue
                has_seed_spectrum = seed_identity in prototype_index
                if not has_seed_spectrum:
                    missing_seed_prototypes.add(seed_identity)
                seed_indices[position] = prototype_index[
                    seed_identity if has_seed_spectrum else missing_prototype_id
                ]
                relation_types[position] = int(edge.relation_type)
                edge_features[position] = [
                    float(edge.path_confidence), float(edge.experimental_support) if has_seed_spectrum else 0.0,
                    float(edge.reaction_completeness), float(edge.conflict),
                ]
                edge_masks[position] = True
            flat_seed_indices.append(seed_indices)
            flat_relations.append(relation_types)
            flat_features.append(edge_features)
            flat_masks.append(edge_masks)
        offsets.append(len(flat_candidate_ids))
    if leakage_edges:
        raise RuntimeError(f"found {leakage_edges} truth/non-seed context edges")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "dataset.npz"
    np.savez_compressed(
        output,
        query_ids=np.asarray(instance_query_ids, dtype=str),
        truth_formulas=np.asarray(formulas, dtype=str),
        query_identities=np.asarray(query_identities, dtype=str),
        rotation_folds=np.asarray(folds, dtype=np.int16),
        query_embeddings=np.asarray(query_vectors, dtype=np.float32),
        offsets=np.asarray(offsets, dtype=np.int64),
        positive_indices=np.asarray(positive_indices, dtype=np.int64),
        candidate_ids=np.asarray(flat_candidate_ids, dtype=str),
        candidate_embeddings=np.asarray(flat_candidates, dtype=np.float32),
        seed_prototype_ids=np.asarray(prototype_ids, dtype=str),
        seed_prototypes=np.asarray(prototypes, dtype=np.float32),
        seed_indices=np.asarray(flat_seed_indices, dtype=np.int64),
        relation_types=np.asarray(flat_relations, dtype=np.int64),
        edge_features=np.asarray(flat_features, dtype=np.float32),
        edge_masks=np.asarray(flat_masks, dtype=bool),
    )
    report = {
        "status": "bioaware_metdna3_context_adapter_dataset_complete",
        "formal": True,
        "rotation_instances": len(instance_rows),
        "unique_queries": len(set(instance_query_ids)),
        "query_identities": len(set(query_identities)),
        "truth_formulas": len(set(formulas)),
        "candidate_rows": len(flat_candidate_ids),
        "candidate_rows_with_context": int(np.asarray(flat_masks).any(axis=1).sum()),
        "context_edges": int(np.asarray(flat_masks).sum()),
        "seed_prototypes": len(prototype_ids),
        "seed_identities_without_recoverable_spectrum": len(missing_seed_prototypes),
        "truncated_candidate_contexts": truncated,
        "contracts": {
            "ten_rotation_identity_isolation": True,
            "heldout_truth_absent_from_context_seeds": True,
            "identity_noop_filtered": True,
            "P2b": "forbidden",
            "internal_validation_or_external_test_opened": False,
            "outcomes_used_for_context_construction": False,
            "missing_seed_spectrum": "explicit zero prototype with experimental-support feature set to zero",
        },
        "provenance": {
            "candidate_scores_sha256": sha256(args.candidate_scores),
            "official_embeddings_sha256": sha256(args.official_embeddings),
            "identity_splits_sha256": sha256(args.identity_splits),
            "evidence_paths_sha256": sha256(args.evidence_paths),
            "seed_embeddings_sha256": sha256(args.seed_dir / "embeddings.npy"),
            "dataset_sha256": sha256(output),
        },
        "claim_limit": "Consumed HILIC development dataset; no RP/internal or external performance claim.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
