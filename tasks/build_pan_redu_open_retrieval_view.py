#!/usr/bin/env python3
"""Build a truth-blind open-retrieval view from a frozen Pan-ReDU v4 matrix.

This module does not estimate effects, construct chemical edges, inspect
comparison labels, or open cross-study truth.  It only gives every frozen
query--candidate row a deterministic similar/opposite rank for each already
computed method.  All candidates remain in the full ledger, including
unsupported candidates and rows vetoed because the signed transport interval
contains zero.

An empty directional result is an abstention.  It is never interpreted as a
biological no-match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd


MATRIX_SCHEMA = "pan_redu_fixed_capacity_response_transport_matrix_v4"
MATRIX_STATUS = "DEVELOPMENT_PRETRUTH_RESPONSE_TRANSPORT_MATRIX_COMPLETE"
OUTPUT_SCHEMA = "pan_redu_truth_blind_open_retrieval_view_v1"

QUERY_COLUMNS = ("left_panel_id", "left_contrast_id")
TARGET_COLUMNS = ("right_panel_id", "right_contrast_id")
GROUP_COLUMNS = (*QUERY_COLUMNS, "world", "method")


class MethodSpec(NamedTuple):
    score_column: str
    availability_columns: tuple[str, ...]


METHOD_SPECS: dict[str, MethodSpec] = {
    "transport_conservative": MethodSpec(
        "signed_chemical_evidence_conservative_bound", (),
    ),
    "greedy_same_graph": MethodSpec(
        "greedy_signed_chemical_evidence", ("greedy_intrinsic_score_available",),
    ),
    "centroid_same_encoder": MethodSpec(
        "centroid_cosine",
        ("centroid_intrinsic_score_available", "centroid_scope_supported"),
    ),
}

BASE_COLUMNS = {
    "world", *QUERY_COLUMNS, *TARGET_COLUMNS,
    "answerable", "direction_identified",
    "signed_chemical_evidence_lower", "signed_chemical_evidence_upper",
}
METHOD_COLUMNS = {
    column
    for spec in METHOD_SPECS.values()
    for column in (spec.score_column, *spec.availability_columns)
}
INPUT_COLUMNS = sorted(BASE_COLUMNS | METHOD_COLUMNS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_bool(series: pd.Series, label: str) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not values.isin(mapping).all():
        bad = sorted(values.loc[~values.isin(mapping)].unique())[:3]
        raise RuntimeError(f"{label} contains invalid booleans: {bad}")
    return values.map(mapping).astype(bool)


def _candidate_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame[TARGET_COLUMNS[0]], frame[TARGET_COLUMNS[1]]))


def validate_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate IDs and prove that every world uses one fixed candidate set."""

    if missing := (BASE_COLUMNS | METHOD_COLUMNS) - set(raw.columns):
        raise RuntimeError(f"v4 transport matrix lacks columns: {sorted(missing)}")
    matrix = raw[INPUT_COLUMNS].copy()
    for column in ("world", *QUERY_COLUMNS, *TARGET_COLUMNS):
        matrix[column] = matrix[column].astype(str).str.strip()
        if matrix[column].eq("").any():
            raise RuntimeError(f"transport matrix has empty {column}")
    row_key = ["world", *QUERY_COLUMNS, *TARGET_COLUMNS]
    if matrix.duplicated(row_key).any():
        raise RuntimeError("transport matrix contains duplicate query/world/candidate rows")

    matrix["answerable"] = strict_bool(matrix.answerable, "answerable")
    matrix["direction_identified"] = strict_bool(
        matrix.direction_identified, "direction_identified",
    )
    for column in sorted({c for spec in METHOD_SPECS.values() for c in spec.availability_columns}):
        matrix[column] = strict_bool(matrix[column], column)
    numeric = {
        "signed_chemical_evidence_lower", "signed_chemical_evidence_upper",
        *(spec.score_column for spec in METHOD_SPECS.values()),
    }
    for column in numeric:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    finite_bounds = np.isfinite(matrix.signed_chemical_evidence_lower) & np.isfinite(
        matrix.signed_chemical_evidence_upper
    )
    if not finite_bounds.all():
        raise RuntimeError("transport matrix contains non-finite signed interval endpoints")
    if (matrix.signed_chemical_evidence_lower > matrix.signed_chemical_evidence_upper).any():
        raise RuntimeError("transport matrix contains an inverted signed interval")

    expected_worlds: set[str] | None = None
    for query_key, query in matrix.groupby(list(QUERY_COLUMNS), sort=False):
        worlds = set(query.world)
        if expected_worlds is None:
            expected_worlds = worlds
        elif worlds != expected_worlds:
            raise RuntimeError(
                f"world inventory changes across queries; first={sorted(expected_worlds)}, "
                f"query={query_key}, observed={sorted(worlds)}"
            )
        expected_candidates: set[tuple[str, str]] | None = None
        for world, block in query.groupby("world", sort=False):
            observed = _candidate_set(block)
            if expected_candidates is None:
                expected_candidates = observed
            elif observed != expected_candidates:
                missing = sorted(expected_candidates - observed)[:3]
                added = sorted(observed - expected_candidates)[:3]
                raise RuntimeError(
                    f"candidate universe changes across worlds for query={query_key}, "
                    f"world={world}; missing={missing}, added={added}"
                )
    if expected_worlds is None:
        raise RuntimeError("transport matrix is empty")
    return matrix.reset_index(drop=True)


def _ineligibility_reason(frame: pd.DataFrame) -> pd.Series:
    reason = np.full(len(frame), "", dtype=object)
    reason[~frame.answerable.to_numpy(bool)] = "primary_answerability_false"
    mask = (reason == "") & ~frame.direction_identified.to_numpy(bool)
    reason[mask] = "direction_not_identified"
    mask = (reason == "") & ~frame.direction_interval_excludes_zero.to_numpy(bool)
    reason[mask] = "signed_interval_contains_zero"
    mask = (reason == "") & ~frame.method_available.to_numpy(bool)
    reason[mask] = "method_score_unavailable"
    mask = (reason == "") & ~np.isfinite(frame.method_score.to_numpy(float))
    reason[mask] = "method_score_nonfinite"
    mask = (reason == "") & frame.method_score.eq(0).to_numpy(bool)
    reason[mask] = "method_score_zero"
    return pd.Series(reason, index=frame.index, dtype=str)


def build_open_view(matrix_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the complete method-expanded ledger and directional top hits."""

    matrix = validate_matrix(matrix_raw)
    matrix["direction_interval_excludes_zero"] = (
        matrix.signed_chemical_evidence_lower.gt(0)
        | matrix.signed_chemical_evidence_upper.lt(0)
    )
    ledgers: list[pd.DataFrame] = []
    for method, spec in METHOD_SPECS.items():
        ledger = matrix[[
            "world", *QUERY_COLUMNS, *TARGET_COLUMNS,
            "answerable", "direction_identified",
            "signed_chemical_evidence_lower", "signed_chemical_evidence_upper",
            "direction_interval_excludes_zero",
        ]].copy()
        ledger["method"] = method
        ledger["method_score"] = matrix[spec.score_column].to_numpy(float)
        available = np.ones(len(matrix), dtype=bool)
        for column in spec.availability_columns:
            available &= matrix[column].to_numpy(bool)
        ledger["method_available"] = available
        ledger["directional_face_eligible"] = (
            ledger.answerable
            & ledger.direction_identified
            & ledger.direction_interval_excludes_zero
        )
        ledger["ineligibility_reason"] = _ineligibility_reason(ledger)
        ledger["ranking_eligible"] = ledger.ineligibility_reason.eq("")
        ledger["similar_rank"] = pd.Series(pd.NA, index=ledger.index, dtype="Int64")
        ledger["opposite_rank"] = pd.Series(pd.NA, index=ledger.index, dtype="Int64")
        positive = ledger.ranking_eligible & ledger.method_score.gt(0)
        negative = ledger.ranking_eligible & ledger.method_score.lt(0)
        ledger.loc[positive, "similar_rank"] = (
            ledger.loc[positive]
            .groupby([*QUERY_COLUMNS, "world"])["method_score"]
            .rank(ascending=False, method="min").astype("Int64")
        )
        ledger.loc[negative, "opposite_rank"] = (
            ledger.loc[negative]
            .groupby([*QUERY_COLUMNS, "world"])["method_score"]
            .rank(ascending=True, method="min").astype("Int64")
        )
        ledgers.append(ledger)

    full = pd.concat(ledgers, ignore_index=True)
    group = full.groupby(list(GROUP_COLUMNS), sort=False)
    full["candidate_universe_size"] = group.method_score.transform("size").astype(int)
    full["direction_identified_candidate_count"] = group.directional_face_eligible.transform("sum").astype(int)

    top_rows: list[dict[str, object]] = []
    for group_key, candidates in full.groupby(list(GROUP_COLUMNS), sort=True):
        common = dict(zip(GROUP_COLUMNS, group_key))
        directional_count = int(candidates.directional_face_eligible.sum())
        method_available_count = int((
            candidates.answerable
            & candidates.direction_identified
            & candidates.direction_interval_excludes_zero
            & candidates.method_available
            & np.isfinite(candidates.method_score)
        ).sum())
        for direction, rank_column, sign_mask in (
            ("similar", "similar_rank", candidates.method_score.gt(0)),
            ("opposite", "opposite_rank", candidates.method_score.lt(0)),
        ):
            eligible = candidates.loc[candidates.ranking_eligible & sign_mask]
            if eligible.empty:
                if directional_count == 0:
                    status = "ABSTAIN_NO_DIRECTION_IDENTIFIED_CANDIDATE"
                elif method_available_count == 0:
                    status = "ABSTAIN_METHOD_UNAVAILABLE"
                else:
                    status = f"ABSTAIN_NO_SUPPORTED_{direction.upper()}_CANDIDATE"
                top_rows.append({
                    **common, "direction": direction, "status": status,
                    "target_panel_id": "", "target_contrast_id": "",
                    "method_score": np.nan, "rank": pd.NA,
                    "top_tie_count": 0,
                    "candidate_universe_size": len(candidates),
                    "direction_identified_candidate_count": directional_count,
                })
                continue
            top = eligible.loc[eligible[rank_column].eq(1)].sort_values(
                list(TARGET_COLUMNS), kind="stable",
            )
            status = "ANSWERED_UNIQUE_TOP" if len(top) == 1 else "ANSWERED_TIED_TOP"
            for hit in top.itertuples(index=False):
                top_rows.append({
                    **common, "direction": direction, "status": status,
                    "target_panel_id": hit.right_panel_id,
                    "target_contrast_id": hit.right_contrast_id,
                    "method_score": float(hit.method_score), "rank": 1,
                    "top_tie_count": len(top),
                    "candidate_universe_size": len(candidates),
                    "direction_identified_candidate_count": directional_count,
                })
    tops = pd.DataFrame(top_rows)
    if len(full) != len(matrix) * len(METHOD_SPECS):
        raise RuntimeError("method expansion changed the frozen candidate denominator")
    return full, tops


def validate_matrix_report(report: dict[str, object], matrix_path: Path, row_count: int) -> None:
    if report.get("schema") != MATRIX_SCHEMA or report.get("status") != MATRIX_STATUS:
        raise RuntimeError("input report is not a finalized Pan-ReDU v4 pretruth matrix")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("transport_matrix_sha256") != sha256(matrix_path):
        raise RuntimeError("input matrix checksum disagrees with its finalized report")
    denominators = report.get("denominators")
    if not isinstance(denominators, dict) or int(denominators.get("ordered_contrast_scores", -1)) != row_count:
        raise RuntimeError("input matrix row count disagrees with its finalized report")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--matrix-report", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    report_path = args.matrix_report or args.matrix.parent / "report.json"
    header = pd.read_csv(args.matrix, sep="\t", nrows=0)
    if missing := (BASE_COLUMNS | METHOD_COLUMNS) - set(header.columns):
        raise RuntimeError(f"v4 transport matrix lacks columns: {sorted(missing)}")
    # Deliberately load only IDs, answerability, uncertainty endpoints and
    # precomputed method outputs.  Contrast labels and truth are never read.
    matrix = pd.read_csv(args.matrix, sep="\t", usecols=INPUT_COLUMNS)
    matrix_report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_matrix_report(matrix_report, args.matrix, len(matrix))
    full, tops = build_open_view(matrix)

    args.output_dir.mkdir(parents=True)
    ledger_path = args.output_dir / "open_retrieval_full_ranked_ledger.tsv.gz"
    top_path = args.output_dir / "open_retrieval_top_hits.tsv"
    full.to_csv(ledger_path, sep="\t", index=False, compression="gzip")
    tops.to_csv(top_path, sep="\t", index=False)
    abstentions = int(tops.status.astype(str).str.startswith("ABSTAIN").sum())
    output_report = {
        "schema": OUTPUT_SCHEMA,
        "status": "DEVELOPMENT_PRETRUTH_OPEN_RETRIEVAL_VIEW_COMPLETE",
        "truth_opened": False,
        "comparison_labels_read": False,
        "input_columns_read": INPUT_COLUMNS,
        "methods": list(METHOD_SPECS),
        "denominators": {
            "input_candidate_rows": len(matrix),
            "queries": int(matrix[list(QUERY_COLUMNS)].drop_duplicates().shape[0]),
            "worlds": int(matrix.world.nunique()),
            "methods": len(METHOD_SPECS),
            "full_ranked_ledger_rows": len(full),
            "top_hit_or_abstention_rows": len(tops),
            "abstention_rows": abstentions,
        },
        "candidate_universe": (
            "fixed per query across every world; every method retains every input candidate row"
        ),
        "direction_rule": (
            "similar requires a strictly positive method score; opposite requires a strictly negative "
            "method score; both require primary answerability, direction_identified, and a signed "
            "transport interval that strictly excludes zero"
        ),
        "empty_result_semantics": (
            "abstention due to insufficient direction-identified evidence; never biological no-match"
        ),
        "provenance": {
            "matrix_sha256": sha256(args.matrix),
            "matrix_report_sha256": sha256(report_path),
            "full_ranked_ledger_sha256": sha256(ledger_path),
            "top_hits_sha256": sha256(top_path),
        },
        "claim_boundary": [
            "Top hits are truth-blind discovery hypotheses, not validated biological associations.",
            "An abstention is not evidence that no biological relation exists.",
            "Cross-study truth, exact-axis labels and comparison strings are absent from this view.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(output_report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(output_report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
