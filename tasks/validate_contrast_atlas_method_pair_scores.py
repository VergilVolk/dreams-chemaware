#!/usr/bin/env python3
"""Validate a method-specific score package on the frozen Atlas identity axis.

This is a biology-side handoff validator.  It does not train, calibrate or
modify a spectral method.  It makes sure that each method scores the complete
model-independent candidate axis and forms its own block decisions before any
response beta, SE, phenotype or registered biological relation is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_AXIS = Path(
    "data/validation/contrast_atlas_model_independent_identity_axis_v5_20261027"
)
DEFAULT_CONTRACT = Path(__file__).with_name(
    "contrast_atlas_method_pair_score_contract_v1.json"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def nested_get(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        require(isinstance(current, dict) and key in current, f"missing manifest field: {'.'.join(path)}")
        current = current[key]
    return current


def compare_required_values(
    observed: dict[str, Any], expected: dict[str, Any], prefix: tuple[str, ...] = ()
) -> None:
    for key, expected_value in expected.items():
        path = prefix + (key,)
        observed_value = nested_get(observed, path)
        if isinstance(expected_value, dict):
            require(isinstance(observed_value, dict), f"manifest field is not an object: {'.'.join(path)}")
            compare_required_values(observed, expected_value, path)
        else:
            require(
                observed_value == expected_value,
                f"manifest contract mismatch at {'.'.join(path)}: {observed_value!r} != {expected_value!r}",
            )


def parse_bool(series: pd.Series, name: str) -> np.ndarray:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin(mapping)
    require(not bool(invalid.any()), f"{name} contains non-boolean values")
    return normalized.map(mapping).to_numpy(bool)


def required_columns(frame: pd.DataFrame, columns: list[str], table_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    require(not missing, f"{table_name} misses required columns: {missing}")


def reject_truth_columns(
    frame: pd.DataFrame, forbidden_tokens: list[str], table_name: str
) -> None:
    bad = [
        str(column)
        for column in frame.columns
        if any(token.lower() in str(column).lower() for token in forbidden_tokens)
    ]
    require(not bad, f"{table_name} contains truth/response-bearing columns: {bad}")


def raw_score_values(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    available = parse_bool(frame["score_available"], "score_available")
    scores = pd.to_numeric(frame["raw_score"], errors="coerce").to_numpy(float)
    require(bool(np.all(np.isfinite(scores[available]))), "available pair scores must be finite")
    require(bool(np.all(np.isnan(scores[~available]))), "unavailable pair scores must be blank, not filled")
    reasons = frame["unavailable_reason"].fillna("").astype(str).str.strip().to_numpy()
    require(bool(np.all(reasons[available] == "")), "available scores cannot carry an unavailable_reason")
    require(bool(np.all(reasons[~available] != "")), "unavailable scores require an explicit reason")
    return scores, available


def expected_reciprocal_edges(
    pairs: pd.DataFrame,
    decisions: pd.DataFrame,
    scores_by_id: dict[str, float],
) -> dict[tuple[tuple[str, str], tuple[str, str]], dict[str, Any]]:
    selected_ids = set(
        decisions.loc[parse_bool(decisions["answered"], "answered"), "selected_directed_pair_id"]
        .astype(str)
        .tolist()
    )
    pair_by_id = pairs.set_index("directed_pair_id", drop=False)
    selected_keys: dict[tuple[tuple[str, str], tuple[str, str]], str] = {}
    for pair_id in selected_ids:
        require(pair_id in pair_by_id.index, f"selected pair is absent from frozen axis: {pair_id}")
        row = pair_by_id.loc[pair_id]
        source = (str(row["query_study_id"]), str(row["query_entity_id"]))
        target = (str(row["candidate_study_id"]), str(row["candidate_entity_id"]))
        selected_keys[(source, target)] = pair_id

    result: dict[tuple[tuple[str, str], tuple[str, str]], dict[str, Any]] = {}
    for (source, target), forward_id in selected_keys.items():
        reverse_id = selected_keys.get((target, source))
        if reverse_id is None:
            continue
        left, right = sorted((source, target))
        key = (left, right)
        if key in result:
            continue
        left_to_right = selected_keys[(left, right)]
        right_to_left = selected_keys[(right, left)]
        digest = hashlib.sha256(
            ("reciprocal\0" + "\0".join((*left, *right))).encode("utf-8")
        ).hexdigest()
        result[key] = {
            "reciprocal_edge_id": f"reciprocal_{digest}",
            "left_study_id": left[0],
            "left_entity_id": left[1],
            "right_study_id": right[0],
            "right_entity_id": right[1],
            "left_to_right_directed_pair_id": left_to_right,
            "right_to_left_directed_pair_id": right_to_left,
            "left_to_right_raw_score": scores_by_id[left_to_right],
            "right_to_left_raw_score": scores_by_id[right_to_left],
        }
    return result


def validate_package(axis_dir: Path, package_dir: Path, contract_path: Path) -> dict[str, Any]:
    contract = read_json(contract_path)
    axis_report_path = axis_dir / "report.json"
    pair_axis_path = axis_dir / "directed_complete_candidate_pairs.tsv.gz"
    block_axis_path = axis_dir / "query_target_source_blocks.tsv.gz"
    manifest_path = package_dir / "report.json"
    for path in (axis_report_path, pair_axis_path, block_axis_path, manifest_path):
        require(path.is_file(), f"missing required file: {path}")

    axis_expected = contract["axis"]
    require(sha256(axis_report_path) == axis_expected["axis_report_sha256"], "axis report hash mismatch")
    require(sha256(pair_axis_path) == axis_expected["directed_pair_axis_sha256"], "directed pair axis hash mismatch")
    require(sha256(block_axis_path) == axis_expected["query_block_axis_sha256"], "query block axis hash mismatch")
    axis_report = read_json(axis_report_path)
    require(axis_report.get("schema") == axis_expected["schema"], "axis schema mismatch")
    require(axis_report.get("status") == axis_expected["status"], "axis status mismatch")
    axis_denominators = axis_report["denominators"]
    for contract_key, report_key in (
        ("directed_candidate_pairs", "directed_candidate_pairs"),
        ("query_blocks_total", "query_target_source_blocks"),
        ("nonempty_candidate_blocks", "nonempty_candidate_blocks"),
        ("zero_candidate_blocks", "abstain_no_candidate_blocks"),
    ):
        require(
            int(axis_expected[contract_key]) == int(axis_denominators[report_key]),
            f"axis denominator mismatch: {contract_key}",
        )

    manifest = read_json(manifest_path)
    require(manifest.get("schema") == contract["package_schema"], "package schema mismatch")
    require(manifest.get("data_role") == "development_consumed", "package data_role must remain development_consumed")
    compare_required_values(manifest, contract["required_manifest_values"])
    method = manifest.get("method")
    require(isinstance(method, dict), "manifest.method must be an object")
    for field in ("method_id", "method_version", "artifact_sha256", "raw_score_semantics", "higher_is_better", "probability_interpretation", "raw_score_clipped", "raw_score_transformation"):
        require(field in method, f"missing manifest method field: {field}")
    require(isinstance(method["method_id"], str) and method["method_id"].strip(), "method_id is empty")
    require(isinstance(method["method_version"], str) and method["method_version"].strip(), "method_version is empty")
    require(bool(SHA256_RE.fullmatch(str(method["artifact_sha256"]))), "artifact_sha256 must be a lowercase SHA-256")
    require(method["raw_score_semantics"] in {"ranking_score", "similarity_score", "distance_score", "logit"}, "unsupported raw_score_semantics")
    require(isinstance(method["higher_is_better"], bool), "higher_is_better must be boolean")
    require(method["probability_interpretation"] is False, "raw ranking scores cannot be interpreted as probabilities")
    require(method["raw_score_clipped"] is False, "raw scores must not be clipped")
    require(method["raw_score_transformation"] == "none", "raw score transformation must be none")
    decision_manifest = manifest["decision"]
    tie_tolerance = float(decision_manifest.get("tie_tolerance", 0.0))
    require(math.isfinite(tie_tolerance) and tie_tolerance >= 0, "tie_tolerance must be finite and nonnegative")

    manifest_axis = manifest.get("axis")
    require(isinstance(manifest_axis, dict), "manifest.axis must be an object")
    for key in ("axis_report_sha256", "directed_pair_axis_sha256", "query_block_axis_sha256"):
        require(manifest_axis.get(key) == axis_expected[key], f"package axis provenance mismatch: {key}")

    outputs = contract["required_outputs"]
    output_hashes = manifest.get("outputs")
    require(isinstance(output_hashes, dict), "manifest.outputs must be an object")
    output_paths: dict[str, Path] = {}
    for logical_name, filename in outputs.items():
        path = package_dir / filename
        require(path.is_file(), f"missing package output: {filename}")
        require(output_hashes.get(filename) == sha256(path), f"package output hash mismatch: {filename}")
        output_paths[logical_name] = path

    pairs = pd.read_csv(
        pair_axis_path,
        sep="\t",
        usecols=[
            "query_block_id", "query_block_index", "candidate_index_within_block",
            "directed_pair_id", "query_study_id", "query_entity_id",
            "candidate_study_id", "candidate_entity_id",
        ],
        dtype={
            "query_block_id": str, "directed_pair_id": str,
            "query_study_id": str, "query_entity_id": str,
            "candidate_study_id": str, "candidate_entity_id": str,
        },
    )
    blocks = pd.read_csv(
        block_axis_path,
        sep="\t",
        usecols=[
            "query_block_id", "query_block_index", "candidate_pair_offset_start",
            "candidate_pair_offset_end", "eligible_candidates",
        ],
        dtype={"query_block_id": str},
    )
    nonempty = blocks.loc[blocks["eligible_candidates"].gt(0)].reset_index(drop=True)
    require(len(pairs) == int(axis_expected["directed_candidate_pairs"]), "frozen pair row count mismatch")
    require(len(blocks) == int(axis_expected["query_blocks_total"]), "frozen total block count mismatch")
    require(len(nonempty) == int(axis_expected["nonempty_candidate_blocks"]), "frozen nonempty block count mismatch")

    pair_scores = pd.read_csv(output_paths["pair_scores"], sep="\t", low_memory=False)
    required_columns(pair_scores, contract["pair_score_columns"], "method_pair_scores")
    reject_truth_columns(pair_scores, contract["forbidden_output_column_tokens"], "method_pair_scores")
    require(len(pair_scores) == len(pairs), "method_pair_scores must contain every frozen directed pair exactly once")
    for column in ("directed_pair_id", "query_block_id"):
        require(
            np.array_equal(pair_scores[column].astype(str).to_numpy(), pairs[column].astype(str).to_numpy()),
            f"method_pair_scores is not exactly aligned to frozen axis column {column}",
        )
    for column in ("query_block_index", "candidate_index_within_block"):
        observed = pd.to_numeric(pair_scores[column], errors="raise").to_numpy(np.int64)
        expected = pairs[column].to_numpy(np.int64)
        require(np.array_equal(observed, expected), f"method_pair_scores axis mismatch: {column}")
    require(not pair_scores["directed_pair_id"].duplicated().any(), "duplicate directed_pair_id in method scores")
    scores, available = raw_score_values(pair_scores)

    decisions = pd.read_csv(output_paths["block_decisions"], sep="\t", low_memory=False)
    required_columns(decisions, contract["block_decision_columns"], "method_block_decisions")
    reject_truth_columns(decisions, contract["forbidden_output_column_tokens"], "method_block_decisions")
    require(len(decisions) == len(nonempty), "method_block_decisions must contain every nonempty block exactly once")
    require(
        np.array_equal(decisions["query_block_id"].astype(str).to_numpy(), nonempty["query_block_id"].astype(str).to_numpy()),
        "method_block_decisions is not aligned to all frozen nonempty blocks",
    )
    require(
        np.array_equal(pd.to_numeric(decisions["query_block_index"], errors="raise").to_numpy(np.int64), nonempty["query_block_index"].to_numpy(np.int64)),
        "method_block_decisions query_block_index mismatch",
    )
    require(not decisions["query_block_id"].duplicated().any(), "duplicate block decision")
    statuses = decisions["decision_status"].astype(str)
    require(statuses.isin(contract["allowed_decision_statuses"]).all(), "unsupported decision_status")
    answered = parse_bool(decisions["answered"], "answered")

    starts = nonempty["candidate_pair_offset_start"].to_numpy(np.int64)
    stops = nonempty["candidate_pair_offset_end"].to_numpy(np.int64)
    candidate_counts = stops - starts
    observed_counts = pd.to_numeric(decisions["candidate_count"], errors="raise").to_numpy(np.int64)
    scored_counts = pd.to_numeric(decisions["scored_candidate_count"], errors="raise").to_numpy(np.int64)
    require(np.array_equal(observed_counts, candidate_counts), "block candidate_count differs from frozen axis")
    expected_scored = np.asarray([int(np.sum(available[start:stop])) for start, stop in zip(starts, stops, strict=True)], dtype=np.int64)
    require(np.array_equal(scored_counts, expected_scored), "scored_candidate_count differs from pair availability")

    higher_is_better = bool(method["higher_is_better"])
    pair_ids = pairs["directed_pair_id"].astype(str).to_numpy()
    candidate_ids = pairs["candidate_entity_id"].astype(str).to_numpy()
    for row_index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        row = decisions.iloc[row_index]
        block_scores = scores[start:stop]
        block_available = available[start:stop]
        complete = bool(np.all(block_available))
        status = str(row["decision_status"])
        row_answered = bool(answered[row_index])
        selected_pair = "" if pd.isna(row["selected_directed_pair_id"]) else str(row["selected_directed_pair_id"]).strip()
        selected_candidate = "" if pd.isna(row["selected_candidate_entity_id"]) else str(row["selected_candidate_entity_id"]).strip()
        selected_score = float(pd.to_numeric(pd.Series([row["selected_raw_score"]]), errors="coerce").iloc[0])
        tie_count = int(row["top1_tie_count"])
        margin = float(pd.to_numeric(pd.Series([row["top1_minus_top2_raw_margin"]]), errors="coerce").iloc[0])

        if not complete:
            require(not row_answered and status == "abstain_incomplete_block_evidence", "an incomplete block must fail closed")
            require(not selected_pair and not selected_candidate and math.isnan(selected_score), "an incomplete block cannot select a candidate")
            continue

        best = float(np.max(block_scores) if higher_is_better else np.min(block_scores))
        tied = np.flatnonzero(np.abs(block_scores - best) <= tie_tolerance)
        require(tie_count == len(tied), "top1_tie_count differs from raw model scores")
        if len(tied) != 1:
            require(not row_answered and status == "abstain_tied_top1", "a tied top score must abstain")
            require(not selected_pair and not selected_candidate and math.isnan(selected_score), "a tied block cannot select a candidate")
            continue

        winner = int(tied[0])
        expected_pair = pair_ids[start + winner]
        expected_candidate = candidate_ids[start + winner]
        if row_answered:
            require(status == "selected_unique_top1", "answered block must use selected_unique_top1")
            require(selected_pair == expected_pair, "selected pair is not the method's unique block top1")
            require(selected_candidate == expected_candidate, "selected candidate does not match selected pair")
            require(math.isclose(selected_score, best, rel_tol=1e-9, abs_tol=1e-12), "selected_raw_score differs from pair score")
        else:
            require(status == "abstain_method_rule", "complete unique-top1 abstention must be abstain_method_rule")
            require(not selected_pair and not selected_candidate and math.isnan(selected_score), "abstained block cannot select a candidate")

        if stop - start == 1:
            require(math.isnan(margin), "singleton block margin must be blank")
        else:
            ordered = np.sort(block_scores)
            expected_margin = float(ordered[-1] - ordered[-2] if higher_is_better else ordered[1] - ordered[0])
            require(math.isclose(margin, expected_margin, rel_tol=1e-9, abs_tol=1e-12), "top1_minus_top2_raw_margin mismatch")

    reciprocal = pd.read_csv(output_paths["reciprocal_edges"], sep="\t", low_memory=False)
    required_columns(reciprocal, contract["reciprocal_edge_columns"], "reciprocal_edges")
    reject_truth_columns(reciprocal, contract["forbidden_output_column_tokens"], "reciprocal_edges")
    require(not reciprocal["reciprocal_edge_id"].duplicated().any(), "duplicate reciprocal_edge_id")
    scores_by_id = dict(zip(pair_ids, scores, strict=True))
    expected_edges = expected_reciprocal_edges(pairs, decisions, scores_by_id)
    require(len(reciprocal) == len(expected_edges), "reciprocal edge table does not equal reciprocal method decisions")
    observed_edge_keys: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    for row in reciprocal.itertuples(index=False):
        left = (str(row.left_study_id), str(row.left_entity_id))
        right = (str(row.right_study_id), str(row.right_entity_id))
        require(left < right, "reciprocal edge endpoints must use canonical lexical order")
        key = (left, right)
        require(key in expected_edges, "reciprocal edge is not supported by two selected directions")
        require(key not in observed_edge_keys, "duplicate reciprocal endpoint pair")
        observed_edge_keys.add(key)
        expected = expected_edges[key]
        for field in (
            "reciprocal_edge_id", "left_study_id", "left_entity_id", "right_study_id",
            "right_entity_id", "left_to_right_directed_pair_id", "right_to_left_directed_pair_id",
        ):
            require(str(getattr(row, field)) == str(expected[field]), f"reciprocal edge field mismatch: {field}")
        for field in ("left_to_right_raw_score", "right_to_left_raw_score"):
            require(
                math.isclose(float(getattr(row, field)), float(expected[field]), rel_tol=1e-9, abs_tol=1e-12),
                f"reciprocal directional score mismatch: {field}",
            )
    require(observed_edge_keys == set(expected_edges), "reciprocal edge set is incomplete")

    return {
        "schema": "contrast_atlas_method_pair_score_validation_v1",
        "status": "PASS",
        "method_id": method["method_id"],
        "method_version": method["method_version"],
        "axis": {
            "directed_candidate_pairs": len(pairs),
            "query_blocks_total": len(blocks),
            "nonempty_candidate_blocks": len(nonempty),
            "zero_candidate_blocks": len(blocks) - len(nonempty),
        },
        "delivery": {
            "available_pair_scores": int(np.sum(available)),
            "unavailable_pair_scores": int(np.sum(~available)),
            "answered_blocks": int(np.sum(answered)),
            "reciprocal_edges": len(reciprocal),
        },
        "claim_boundary": "schema/integrity pass only; no identity accuracy, response increment or biological validity is established",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-dir", type=Path, default=DEFAULT_AXIS)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    result = validate_package(args.axis_dir, args.package_dir, args.contract)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
