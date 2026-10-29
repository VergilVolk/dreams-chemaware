from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from validate_contrast_atlas_method_pair_scores import sha256, validate_package


HERE = Path(__file__).resolve().parent
BASE_CONTRACT = json.loads(
    (HERE / "contrast_atlas_method_pair_score_contract_v1.json").read_text(encoding="utf-8")
)


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )


def reciprocal_id() -> str:
    digest = hashlib.sha256(
        "reciprocal\0A\0A::1\0B\0B::1".encode("utf-8")
    ).hexdigest()
    return f"reciprocal_{digest}"


def build_axis(root: Path) -> tuple[Path, Path]:
    axis = root / "axis"
    axis.mkdir()
    pairs = pd.DataFrame(
        {
            "query_block_id": ["block_a_b", "block_b_a"],
            "query_block_index": [0, 1],
            "candidate_index_within_block": [0, 0],
            "directed_pair_id": ["pair_a_b", "pair_b_a"],
            "query_study_id": ["A", "B"],
            "query_entity_id": ["A::1", "B::1"],
            "candidate_study_id": ["B", "A"],
            "candidate_entity_id": ["B::1", "A::1"],
        }
    )
    blocks = pd.DataFrame(
        {
            "query_block_id": ["block_a_b", "block_b_a"],
            "query_block_index": [0, 1],
            "candidate_pair_offset_start": [0, 1],
            "candidate_pair_offset_end": [1, 2],
            "eligible_candidates": [1, 1],
        }
    )
    pair_path = axis / "directed_complete_candidate_pairs.tsv.gz"
    block_path = axis / "query_target_source_blocks.tsv.gz"
    write_tsv(pairs, pair_path)
    write_tsv(blocks, block_path)
    report = {
        "schema": "contrast_atlas_model_independent_identity_axis_v1",
        "status": "DEVELOPMENT_CONSUMED_MODEL_INDEPENDENT_IDENTITY_AXIS_FROZEN",
        "denominators": {
            "directed_candidate_pairs": 2,
            "query_target_source_blocks": 2,
            "nonempty_candidate_blocks": 2,
            "abstain_no_candidate_blocks": 0,
        },
    }
    report_path = axis / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    contract = deepcopy(BASE_CONTRACT)
    contract["axis"].update(
        {
            "axis_report_sha256": sha256(report_path),
            "directed_pair_axis_sha256": sha256(pair_path),
            "query_block_axis_sha256": sha256(block_path),
            "directed_candidate_pairs": 2,
            "query_blocks_total": 2,
            "nonempty_candidate_blocks": 2,
            "zero_candidate_blocks": 0,
        }
    )
    contract_path = root / "contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return axis, contract_path


def package_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = pd.DataFrame(
        {
            "directed_pair_id": ["pair_a_b", "pair_b_a"],
            "query_block_id": ["block_a_b", "block_b_a"],
            "query_block_index": [0, 1],
            "candidate_index_within_block": [0, 0],
            "raw_score": [1.25, -0.40],
            "score_available": [True, True],
            "unavailable_reason": ["", ""],
        }
    )
    decisions = pd.DataFrame(
        {
            "query_block_id": ["block_a_b", "block_b_a"],
            "query_block_index": [0, 1],
            "candidate_count": [1, 1],
            "scored_candidate_count": [1, 1],
            "decision_status": ["selected_unique_top1", "selected_unique_top1"],
            "answered": [True, True],
            "selected_directed_pair_id": ["pair_a_b", "pair_b_a"],
            "selected_candidate_entity_id": ["B::1", "A::1"],
            "selected_raw_score": [1.25, -0.40],
            "top1_tie_count": [1, 1],
            "top1_minus_top2_raw_margin": [np.nan, np.nan],
        }
    )
    reciprocal = pd.DataFrame(
        {
            "reciprocal_edge_id": [reciprocal_id()],
            "left_study_id": ["A"],
            "left_entity_id": ["A::1"],
            "right_study_id": ["B"],
            "right_entity_id": ["B::1"],
            "left_to_right_directed_pair_id": ["pair_a_b"],
            "right_to_left_directed_pair_id": ["pair_b_a"],
            "left_to_right_raw_score": [1.25],
            "right_to_left_raw_score": [-0.40],
        }
    )
    return scores, decisions, reciprocal


def write_package(
    root: Path,
    contract_path: Path,
    scores: pd.DataFrame,
    decisions: pd.DataFrame,
    reciprocal: pd.DataFrame,
    *,
    manifest_patch: dict | None = None,
) -> Path:
    package = root / "package"
    package.mkdir(exist_ok=True)
    paths = {
        "method_pair_scores.tsv.gz": package / "method_pair_scores.tsv.gz",
        "method_block_decisions.tsv.gz": package / "method_block_decisions.tsv.gz",
        "reciprocal_edges.tsv.gz": package / "reciprocal_edges.tsv.gz",
    }
    write_tsv(scores, paths["method_pair_scores.tsv.gz"])
    write_tsv(decisions, paths["method_block_decisions.tsv.gz"])
    write_tsv(reciprocal, paths["reciprocal_edges.tsv.gz"])
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest = {
        "schema": "contrast_atlas_method_pair_score_package_v1",
        "status": "DEVELOPMENT_CONSUMED_METHOD_PAIR_SCORES_COMPLETE",
        "data_role": "development_consumed",
        "axis": {
            "axis_report_sha256": contract["axis"]["axis_report_sha256"],
            "directed_pair_axis_sha256": contract["axis"]["directed_pair_axis_sha256"],
            "query_block_axis_sha256": contract["axis"]["query_block_axis_sha256"],
        },
        "method": {
            "method_id": "toy_method",
            "method_version": "v1",
            "artifact_sha256": "a" * 64,
            "raw_score_semantics": "ranking_score",
            "higher_is_better": True,
            "probability_interpretation": False,
            "raw_score_clipped": False,
            "raw_score_transformation": "none",
        },
        "truth_blinding": {
            "response_beta_read": False,
            "response_se_read": False,
            "phenotype_read": False,
            "registered_relations_read": False,
            "identity_truth_read": False,
        },
        "missingness": {
            "unavailable_pair_policy": "blank_score",
            "incomplete_block_policy": "abstain_fail_closed",
            "external_score_fallback_used": False,
            "wse_fallback_used": False,
        },
        "decision": {
            "selection_scope": "complete_nonempty_candidate_block",
            "block_decisions_are_method_specific": True,
            "reciprocal_rule": "both_directions_selected_unique_top1",
            "reciprocal_edges_preserve_both_directions": True,
            "tie_tolerance": 0.0,
        },
        "outputs": {name: sha256(path) for name, path in paths.items()},
    }
    if manifest_patch:
        for section, values in manifest_patch.items():
            if isinstance(values, dict) and isinstance(manifest.get(section), dict):
                manifest[section].update(values)
            else:
                manifest[section] = values
    (package / "report.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return package


def test_valid_complete_axis_package_preserves_unbounded_directional_scores(tmp_path: Path) -> None:
    axis, contract = build_axis(tmp_path)
    package = write_package(tmp_path, contract, *package_frames())
    result = validate_package(axis, package, contract)
    assert result["status"] == "PASS"
    assert result["delivery"]["reciprocal_edges"] == 1
    assert result["delivery"]["answered_blocks"] == 2


def test_missing_pair_row_is_rejected_even_if_manifest_hash_is_updated(tmp_path: Path) -> None:
    axis, contract = build_axis(tmp_path)
    scores, decisions, reciprocal = package_frames()
    package = write_package(tmp_path, contract, scores.iloc[:1].copy(), decisions, reciprocal)
    with pytest.raises(RuntimeError, match="every frozen directed pair"):
        validate_package(axis, package, contract)


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"truth_blinding": {"phenotype_read": True}}, "truth_blinding.phenotype_read"),
        ({"missingness": {"wse_fallback_used": True}}, "missingness.wse_fallback_used"),
        ({"method": {"probability_interpretation": True}}, "interpreted as probabilities"),
        ({"method": {"raw_score_clipped": True}}, "must not be clipped"),
    ],
)
def test_truth_probability_clipping_and_wse_fallback_are_rejected(
    tmp_path: Path, patch: dict, message: str
) -> None:
    axis, contract = build_axis(tmp_path)
    package = write_package(tmp_path, contract, *package_frames(), manifest_patch=patch)
    with pytest.raises(RuntimeError, match=message):
        validate_package(axis, package, contract)


def test_missing_final_score_passes_only_with_fail_closed_block_and_no_edge(tmp_path: Path) -> None:
    axis, contract = build_axis(tmp_path)
    scores, decisions, reciprocal = package_frames()
    scores.loc[1, ["raw_score", "score_available", "unavailable_reason"]] = [
        np.nan,
        False,
        "required_model_evidence_missing",
    ]
    decisions.loc[1, [
        "scored_candidate_count", "decision_status", "answered",
        "selected_directed_pair_id", "selected_candidate_entity_id", "selected_raw_score",
    ]] = [0, "abstain_incomplete_block_evidence", False, "", "", np.nan]
    reciprocal = reciprocal.iloc[:0].copy()
    package = write_package(tmp_path, contract, scores, decisions, reciprocal)
    result = validate_package(axis, package, contract)
    assert result["status"] == "PASS"
    assert result["delivery"]["unavailable_pair_scores"] == 1
    assert result["delivery"]["reciprocal_edges"] == 0


def test_reciprocal_edge_cannot_drop_or_duplicate_a_direction(tmp_path: Path) -> None:
    axis, contract = build_axis(tmp_path)
    scores, decisions, reciprocal = package_frames()
    reciprocal.loc[0, "right_to_left_directed_pair_id"] = "pair_a_b"
    package = write_package(tmp_path, contract, scores, decisions, reciprocal)
    with pytest.raises(RuntimeError, match="right_to_left_directed_pair_id"):
        validate_package(axis, package, contract)
