#!/usr/bin/env python3
"""Finalize the complete pre-truth Pan-ReDU partial-response transport matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from contrast_response_transport import parse_bool


IDENTITY_WORLDS = {
    "all", "known_only", "known_on_all_common_face",
    "unknown_involving_on_all_common_face", "measurement_matched_known",
    "measurement_matched_unresolved",
    "measurement_matched_known_common_availability",
    "measurement_matched_unresolved_common_availability", "unresolved_only",
    "unresolved_response_reassigned_left", "unresolved_response_reassigned_right",
    "measurement_matched_unresolved_response_reassigned_left",
    "measurement_matched_unresolved_response_reassigned_right",
}
PAIR_SCHEMA = "pan_redu_panel_pair_fixed_capacity_response_transport_v4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validated_execution_contract(report: dict[str, object], pair_index: int) -> str:
    contract = report.get("execution_contract")
    observed = str(report.get("execution_contract_sha256", ""))
    if not isinstance(contract, dict) or not observed:
        raise RuntimeError(f"pair lacks an execution contract: {pair_index}")
    expected = canonical_sha256(contract)
    if observed != expected:
        raise RuntimeError(f"pair execution-contract checksum drift: {pair_index}")
    if contract.get("pair_output_schema") != PAIR_SCHEMA:
        raise RuntimeError(f"pair execution-contract schema drift: {pair_index}")
    if report.get("parameters") != contract.get("parameters"):
        raise RuntimeError(f"pair parameter/contract drift: {pair_index}")
    return observed


def require_same_execution_contract(
    expected: str | None, observed: str, pair_index: int,
) -> str:
    if expected is not None and observed != expected:
        raise RuntimeError(f"cross-pair execution-contract drift: {pair_index}")
    return observed if expected is None else expected


def register_panel_assets(
    registry: dict[str, dict[str, str]],
    report: dict[str, object],
    expected_pair: pd.Series,
    pair_index: int,
) -> None:
    assets = report.get("panel_assets")
    if not isinstance(assets, dict):
        raise RuntimeError(f"pair lacks panel-asset fingerprints: {pair_index}")
    expected_panels = {
        str(expected_pair.left_panel_id), str(expected_pair.right_panel_id),
    }
    if set(assets) != expected_panels:
        raise RuntimeError(f"pair panel-asset inventory drift: {pair_index}")
    required = {
        "panel_report_sha256", "embeddings_sha256", "effect_report_sha256",
        "effect_schema", "effect_status", "effects_sha256",
        "families_sha256", "measurement_covariates_sha256",
    }
    for panel_id in sorted(expected_panels):
        fingerprint = assets[panel_id]
        if not isinstance(fingerprint, dict) or set(fingerprint) != required:
            raise RuntimeError(f"invalid panel-asset fingerprint: {pair_index}/{panel_id}")
        normalized = {key: str(fingerprint[key]) for key in sorted(required)}
        if not all(normalized.values()):
            raise RuntimeError(f"empty panel-asset fingerprint: {pair_index}/{panel_id}")
        previous = registry.get(panel_id)
        if previous is not None and previous != normalized:
            raise RuntimeError(f"cross-pair panel-asset drift: {pair_index}/{panel_id}")
        registry[panel_id] = normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest", required=True, type=Path)
    parser.add_argument("--pair-results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    pairs = pd.read_csv(args.pair_manifest, sep="\t")
    pair_manifest_sha256 = sha256(args.pair_manifest)
    matrices = []
    pair_rows = []
    execution_contract_sha256: str | None = None
    panel_asset_registry: dict[str, dict[str, str]] = {}
    for pair_index in pairs.pair_index.astype(int):
        expected_pair = pairs.loc[pairs.pair_index.astype(int).eq(pair_index)]
        if len(expected_pair) != 1:
            raise RuntimeError(f"pair manifest index absent or duplicated: {pair_index}")
        expected_pair = expected_pair.iloc[0]
        root = args.pair_results_root / f"pair_{pair_index:04d}"
        report_path = root / "report.json"
        score_path = root / "ordered_contrast_scores.tsv.gz"
        if not report_path.is_file() or not score_path.is_file():
            raise FileNotFoundError(f"incomplete pair result: {pair_index}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema") != PAIR_SCHEMA:
            raise RuntimeError(
                f"pair estimator/schema drift: {pair_index}; "
                f"expected {PAIR_SCHEMA}, observed {report.get('schema')!r}"
            )
        if report.get("status") != "DEVELOPMENT_PRETRUTH_PARTIAL_RESPONSE_TRANSPORT_COMPLETE":
            raise RuntimeError(f"inadmissible pair status: {pair_index}")
        pair_contract_sha256 = validated_execution_contract(report, pair_index)
        execution_contract_sha256 = require_same_execution_contract(
            execution_contract_sha256, pair_contract_sha256, pair_index,
        )
        if int(report["pair"]["pair_index"]) != pair_index:
            raise RuntimeError(f"pair report index mismatch: {pair_index}")
        actual_score_sha256 = sha256(score_path)
        if report.get("provenance", {}).get("scores_sha256") != actual_score_sha256:
            raise RuntimeError(f"pair score checksum drift: {pair_index}")
        if report.get("provenance", {}).get("pair_manifest_sha256") != pair_manifest_sha256:
            raise RuntimeError(f"pair manifest provenance drift: {pair_index}")
        for column in pairs.columns:
            if str(report["pair"].get(column, "")) != str(expected_pair[column]):
                raise RuntimeError(f"pair report manifest-field drift: {pair_index}/{column}")
        register_panel_assets(panel_asset_registry, report, expected_pair, pair_index)
        matrix = pd.read_csv(score_path, sep="\t")
        if not matrix.pair_index.astype(int).eq(pair_index).all():
            raise RuntimeError(f"score pair index mismatch: {pair_index}")
        denominators = report.get("denominators", {})
        reported_rows = int(denominators.get("ordered_contrast_scores", -1))
        expected_rows = (
            2 * int(denominators.get("left_contrasts", -1))
            * int(denominators.get("right_contrasts", -1))
            * int(denominators.get("worlds", -1))
        )
        if len(matrix) != reported_rows or len(matrix) != expected_rows:
            raise RuntimeError(f"pair score row-count drift: {pair_index}")
        expected_worlds = IDENTITY_WORLDS if report.get("identity_partition_available") else {"all"}
        if set(matrix.world.astype(str)) != expected_worlds:
            raise RuntimeError(f"pair world schema drift: {pair_index}")
        if set(matrix.query_side.astype(str)) != {"left", "right"}:
            raise RuntimeError(f"pair query-side schema drift: {pair_index}")
        answerable = parse_bool(matrix.answerable, "answerable")
        chemical_support = parse_bool(matrix.chemical_support, "chemical_support")
        direction_answerable = parse_bool(matrix.direction_answerable, "direction_answerable")
        direction_identified = parse_bool(matrix.direction_identified, "direction_identified")
        if not answerable.equals(chemical_support & direction_answerable):
            raise RuntimeError(f"pair answerability contract drift: {pair_index}")
        if not direction_answerable.equals(direction_identified):
            raise RuntimeError(f"pair direction-identification contract drift: {pair_index}")
        if int(answerable.sum()) != int(denominators.get("answerable_scores", -1)):
            raise RuntimeError(f"pair answerable denominator drift: {pair_index}")
        matrices.append(matrix)
        pair_rows.append({
            "pair_index": pair_index,
            **report["denominators"],
            "report_sha256": sha256(report_path), "scores_sha256": actual_score_sha256,
        })
    combined = pd.concat(matrices, ignore_index=True)
    if combined.duplicated(["world", "left_panel_id", "left_contrast_id", "right_panel_id", "right_contrast_id"]).any():
        raise RuntimeError("combined matrix contains duplicate ordered contrast comparisons")
    args.output_dir.mkdir(parents=True)
    matrix_path = args.output_dir / "ordered_contrast_transport_matrix.tsv.gz"
    ledger_path = args.output_dir / "pair_completion_ledger.tsv"
    combined.to_csv(matrix_path, sep="\t", index=False, compression="gzip")
    pd.DataFrame(pair_rows).to_csv(ledger_path, sep="\t", index=False)
    report = {
        "schema": "pan_redu_fixed_capacity_response_transport_matrix_v4",
        "status": "DEVELOPMENT_PRETRUTH_RESPONSE_TRANSPORT_MATRIX_COMPLETE",
        "denominators": {
            "panel_pairs": len(pairs), "ordered_contrast_scores": len(combined),
            "query_panels": int(combined.left_panel_id.nunique()),
            "query_contrasts": int(combined[["left_panel_id", "left_contrast_id"]].drop_duplicates().shape[0]),
            "worlds": int(combined.world.nunique()),
            "answerable_scores": int(parse_bool(combined.answerable, "answerable").sum()),
            "panels_with_frozen_asset_fingerprints": len(panel_asset_registry),
        },
        "execution_contract_sha256": execution_contract_sha256,
        "panel_asset_fingerprints": panel_asset_registry,
        "leakage_status": "mechanical concatenation only; no comparison stratum, relevance truth or biological outcome opened",
        "provenance": {
            "pair_manifest_sha256": pair_manifest_sha256,
            "transport_matrix_sha256": sha256(matrix_path),
            "completion_ledger_sha256": sha256(ledger_path),
            "panel_asset_fingerprints_sha256": canonical_sha256(panel_asset_registry),
        },
        "claim_boundary": [
            "This complete matrix is still pre-truth development output.",
            "Biological relevance and unknown-information increment require separate frozen adjudication.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
