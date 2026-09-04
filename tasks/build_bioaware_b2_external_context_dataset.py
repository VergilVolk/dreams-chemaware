#!/usr/bin/env python
"""Build the eight-panel external dataset for B2 context-embedding LOSO training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


UNITS = (
    "BV2cell__hilic", "BV2cell__rplc", "Mouse_brain__hilic",
    "Mouse_brain__rplc", "Mouse_liver__hilic", "Mouse_liver__rplc",
    "NIST_plasma__hilic", "NIST_plasma__rplc",
)
STUDY = {
    "BV2cell__hilic": "BV2cell", "BV2cell__rplc": "BV2cell",
    "Mouse_brain__hilic": "Mouse_brain", "Mouse_brain__rplc": "Mouse_brain",
    "Mouse_liver__hilic": "Mouse_liver", "Mouse_liver__rplc": "Mouse_liver",
    "NIST_plasma__hilic": "NIST_plasma", "NIST_plasma__rplc": "NIST_plasma",
}
EVIDENCE_COLUMNS = (
    "known_edge_path_fraction", "known_edge_identity_path_fraction",
    "known_edge_best_bottleneck", "known_edge_median_bottleneck",
    "predicted_edge_path_fraction", "predicted_edge_identity_path_fraction",
    "predicted_edge_best_bottleneck", "predicted_edge_median_bottleneck",
    "smn_path_fraction", "smn_best_bottleneck",
    "rt_score", "rt_pass_fraction",
)
BIOLOGICAL_CONTEXT_COLUMNS = EVIDENCE_COLUMNS[:10]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_v3_v1"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_b2_external_context_dataset_v1"),
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output: {args.output_dir}")

    query_ids: list[str] = []
    source_query_ids: list[str] = []
    units: list[str] = []
    studies: list[str] = []
    formulas: list[str] = []
    truth_ids: list[str] = []
    query_embeddings: list[np.ndarray] = []
    candidate_ids: list[str] = []
    candidate_embeddings: list[np.ndarray] = []
    evidence_rows: list[np.ndarray] = []
    context_masks: list[bool] = []
    positive_indices: list[int] = []
    offsets = [0]
    provenance: dict[str, dict[str, str]] = {}
    spectral_score_deviations: list[float] = []

    for unit in UNITS:
        root = args.root / unit
        ledger_path = root / "ledger" / "candidate_evidence.csv.gz"
        scores_path = root / "scores" / "candidate_scores.csv.gz"
        embeddings_path = root / "scores" / "embeddings.npz"
        queries_path = root / "cache" / "queries.csv.gz"
        for path in (ledger_path, scores_path, embeddings_path, queries_path):
            if not path.exists():
                raise FileNotFoundError(path)
        ledger = pd.read_csv(ledger_path)
        scores = pd.read_csv(scores_path)
        queries = pd.read_csv(queries_path)
        missing = set(EVIDENCE_COLUMNS) - set(ledger.columns)
        if missing:
            raise RuntimeError(f"{unit} ledger lacks {sorted(missing)}")
        if ledger[["query_id", "candidate_id"]].duplicated().any():
            raise RuntimeError(f"{unit} duplicate ledger candidate rows")
        score_keys = scores[["query_id", "candidate_id", "best_reference_row"]]
        merged = ledger.merge(
            score_keys, on=["query_id", "candidate_id"], how="left", validate="one_to_one",
        )
        if merged.best_reference_row.isna().any():
            raise RuntimeError(f"{unit} candidate reference merge is incomplete")

        body = np.load(embeddings_path)
        query_matrix = normalize(body["query_embedding"])
        reference_matrix = normalize(body["reference_embedding"])
        reference_rows = body["reference_rows"].astype(np.int64)
        if len(queries) != len(query_matrix) or queries.query_id.duplicated().any():
            raise RuntimeError(f"{unit} query embedding order mismatch")
        if len(np.unique(reference_rows)) != len(reference_rows):
            raise RuntimeError(f"{unit} reference embedding rows are not unique")
        query_position = dict(zip(queries.query_id.astype(str), range(len(queries)), strict=True))
        reference_position = dict(zip(reference_rows.tolist(), range(len(reference_rows)), strict=True))

        for source_query_id, group in merged.groupby("query_id", sort=False):
            source_query_id = str(source_query_id)
            if source_query_id not in query_position:
                raise RuntimeError(f"{unit} query missing embedding: {source_query_id}")
            truth = str(group.truth_candidate_id.iloc[0])
            positive = np.flatnonzero(group.candidate_id.astype(str).to_numpy() == truth)
            if len(positive) != 1 or len(group) < 2:
                raise RuntimeError(f"{unit} query needs one truth and at least one negative")
            query_ids.append(unit + "|" + source_query_id)
            source_query_ids.append(source_query_id)
            units.append(unit)
            studies.append(STUDY[unit])
            formulas.append(str(group.truth_formula.iloc[0]))
            truth_ids.append(truth)
            query_embeddings.append(query_matrix[query_position[source_query_id]])
            positive_indices.append(int(positive[0]))
            for row in group.itertuples(index=False):
                reference_row = int(row.best_reference_row)
                if reference_row not in reference_position:
                    raise RuntimeError(f"{unit} missing reference embedding row {reference_row}")
                candidate_ids.append(str(row.candidate_id))
                candidate_embedding = reference_matrix[reference_position[reference_row]]
                candidate_embeddings.append(candidate_embedding)
                reproduced_score = float(
                    np.dot(query_matrix[query_position[source_query_id]], candidate_embedding)
                )
                deviation = abs(reproduced_score - float(row.spectral_score))
                spectral_score_deviations.append(deviation)
                if deviation > 2e-5:
                    raise RuntimeError(
                        f"{unit} spectral score reproduction failed for "
                        f"{source_query_id}/{row.candidate_id}: deviation={deviation}"
                    )
                evidence = np.asarray(
                    [float(getattr(row, column)) for column in EVIDENCE_COLUMNS],
                    dtype=np.float32,
                )
                if not np.isfinite(evidence).all():
                    raise RuntimeError(f"{unit} non-finite evidence for {source_query_id}")
                evidence_rows.append(evidence)
                context_masks.append(bool(np.any(np.abs(evidence[:10]) > 1e-8)))
            offsets.append(len(candidate_ids))
        provenance[unit] = {
            "ledger_sha256": sha256(ledger_path),
            "scores_sha256": sha256(scores_path),
            "embeddings_sha256": sha256(embeddings_path),
            "queries_sha256": sha256(queries_path),
        }

    if len(set(query_ids)) != len(query_ids):
        raise RuntimeError("global query IDs overlap")
    output = args.output_dir / "dataset.npz"
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(
        output,
        query_ids=np.asarray(query_ids, dtype=str),
        source_query_ids=np.asarray(source_query_ids, dtype=str),
        unit_ids=np.asarray(units, dtype=str),
        study_ids=np.asarray(studies, dtype=str),
        truth_formulas=np.asarray(formulas, dtype=str),
        truth_candidate_ids=np.asarray(truth_ids, dtype=str),
        query_embeddings=np.asarray(query_embeddings, dtype=np.float32),
        offsets=np.asarray(offsets, dtype=np.int64),
        positive_indices=np.asarray(positive_indices, dtype=np.int16),
        candidate_ids=np.asarray(candidate_ids, dtype=str),
        candidate_embeddings=np.asarray(candidate_embeddings, dtype=np.float32),
        evidence=np.asarray(evidence_rows, dtype=np.float32),
        context_mask=np.asarray(context_masks, dtype=bool),
        evidence_columns=np.asarray(EVIDENCE_COLUMNS, dtype=str),
    )
    masks = np.asarray(context_masks, dtype=bool)
    report = {
        "status": "bioaware_b2_external_context_dataset_complete",
        "formal": True,
        "queries": len(query_ids),
        "candidate_rows": len(candidate_ids),
        "studies": len(set(studies)),
        "units": len(set(units)),
        "formulas": len(set(formulas)),
        "queries_with_biological_context": int(sum(
            masks[offsets[index]:offsets[index + 1]].any()
            for index in range(len(query_ids))
        )),
        "candidate_rows_with_biological_context": int(masks.sum()),
        "official_spectral_score_reproduction": {
            "maximum_absolute_deviation": float(max(spectral_score_deviations, default=0.0)),
            "tolerance": 2e-5,
            "pass": bool(max(spectral_score_deviations, default=0.0) <= 2e-5),
        },
        "evidence_columns": list(EVIDENCE_COLUMNS),
        "context_mask_definition": (
            "at least one nonzero known/predicted reaction-path or structural-network feature; "
            "RT alone cannot activate B2"
        ),
        "contracts": {
            "output_is_candidate_context_embedding_input": True,
            "spectral_score_is_adapter_feature": False,
            "rules_are_adapter_features": False,
            "truth_used_only_as_listwise_label": True,
            "P2b": "forbidden",
            "phenotype": "forbidden",
            "no_context_exact_fallback_required": True,
        },
        "provenance": provenance | {
            "dataset_sha256": sha256(output),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": "Dataset construction only; no embedding or annotation gain.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
