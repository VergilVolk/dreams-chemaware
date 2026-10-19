from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_pair_manifest_report(
    pair_path: Path, topology_path: Path, panel_metadata_path: Path, pair_count: int,
) -> None:
    (pair_path.parent / "report.json").write_text(json.dumps({
        "schema": "pan_redu_source_disjoint_partial_transport_pairs_v2_polarity_safe",
        "status": "PRETRUTH_SOURCE_DISJOINT_SAME_POLARITY_PANEL_PAIRS_FROZEN",
        "required_polarity": "positive",
        "denominators": {"source_disjoint_panel_pairs": pair_count},
        "provenance": {
            "panel_pairs_sha256": file_sha256(pair_path),
            "topology_sha256": file_sha256(topology_path),
            "panel_metadata_sha256": file_sha256(panel_metadata_path),
        },
    }), encoding="utf-8")


def write_panel(
    root: Path, panel: str, signs: list[float],
    passes_effect_qc: list[bool] | None = None,
) -> None:
    effects = root / "effects" / panel
    embeddings = root / "embeddings" / panel
    families = root / "families" / panel
    effects.mkdir(parents=True)
    embeddings.mkdir(parents=True)
    families.mkdir(parents=True)
    panel_report_sha256 = f"frozen-panel-report-{panel}"
    effect_path = effects / "contrast_effects.tsv.gz"
    pd.DataFrame({
        "contrast_id": ["C1"] * 3, "anchor_id": ["A", "B", "C"],
        "contrast_field": ["intervention"] * 3,
        "level_a": ["control"] * 3, "level_b": ["treated"] * 3,
        "precursor_mz": [100.0, 101.0, 102.0],
        "beta_b_minus_a": signs, "se": [0.2] * 3, "unit_prevalence": [1.0] * 3,
        "passes_effect_qc": passes_effect_qc or [True] * 3,
    }).to_csv(effect_path, sep="\t", index=False, compression="gzip")
    (effects / "report.json").write_text(json.dumps({
        "schema": "pan_redu_panel_contrast_effects_v1",
        "status": "development_effect_objects_not_cross_study_evidence",
        "panel_id": panel,
        "input_sha256": {"panel_report": panel_report_sha256},
        "output_sha256": {"contrast_effects": file_sha256(effect_path)},
    }), encoding="utf-8")
    vectors = np.eye(3, dtype=np.float32)
    embedding_path = embeddings / "anchor_embeddings.npz"
    np.savez_compressed(embedding_path, anchor_id=np.asarray(["A", "B", "C"]), embeddings=vectors)
    (embeddings / "report.json").write_text(json.dumps({
        "status": "phenotype_blind_frozen_spectral_embeddings",
        "panel_id": panel,
        "provenance": {
            "panel_report_sha256": panel_report_sha256,
            "embeddings_sha256": file_sha256(embedding_path),
        },
    }), encoding="utf-8")
    family_path = families / "ion_family_assignments.tsv.gz"
    pd.DataFrame({
        "entity_id": ["A", "B", "C"], "ion_family_id": ["fA", "fB", "fC"],
        "ion_family_size": [1, 1, 1], "capacity_weight": [1.0, 1.0, 1.0],
    }).to_csv(family_path, sep="\t", index=False, compression="gzip")
    covariate_path = families / "measurement_covariates.tsv.gz"
    pd.DataFrame({
        "entity_id": ["A", "B", "C"], "median_log2_intensity": [10.0, 11.0, 12.0],
        "precursor_mz": [100.0, 101.0, 102.0],
        "detection_rate": [1.0, 1.0, 1.0], "log2_intensity_iqr": [1.0, 1.0, 1.0],
        "ms2_peak_count": [10, 11, 12],
    }).to_csv(covariate_path, sep="\t", index=False, compression="gzip")
    (families / "report.json").write_text(json.dumps({
        "status": "PHENOTYPE_BLIND_PANEL_ION_FAMILIES_FROZEN",
        "panel_id": panel,
        "provenance": {
            "panel_report_sha256": panel_report_sha256,
            "assignments_sha256": file_sha256(family_path),
            "measurement_covariates_sha256": file_sha256(covariate_path),
        },
    }), encoding="utf-8")


def test_fixed_capacity_retains_effect_qc_failures_in_full_panel_universe(
    tmp_path: Path,
) -> None:
    from evaluate_ad_cross_study_response_retrieval import fixed_capacity_axes
    from run_pan_redu_panel_pair_partial_transport import load_panel_responses

    write_panel(tmp_path, "P1", [1.0, -1.0, 1.0], [True, False, True])
    definitions = pd.DataFrame([{
        "contrast_id": "C1", "contrast_field": "intervention",
        "level_a": "control", "level_b": "treated",
    }])
    responses, universe, *_ = load_panel_responses(
        tmp_path / "effects", tmp_path / "families", "P1", definitions, None,
    )
    fixed = fixed_capacity_axes(universe, universe, responses)["C1"].set_index("entity_id")
    assert len(fixed) == 3
    assert np.isclose(fixed.fixed_capacity.sum(), 1.0)
    assert np.allclose(fixed.fixed_capacity.to_numpy(), 1.0 / 3.0)
    assert not bool(fixed.loc["B", "response_available"])
    assert fixed.loc["B", "response_score"] == 0.0


def test_adjusted_axis_effect_schema_is_admitted_but_mixed_status_fails(
    tmp_path: Path,
) -> None:
    from run_pan_redu_panel_pair_partial_transport import load_panel_responses

    write_panel(tmp_path, "P1", [1.0, -1.0, 1.0])
    report_path = tmp_path / "effects" / "P1" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["schema"] = "adjusted_axis_contrast_effects_v1"
    report["status"] = "development_adjusted_effect_object_not_cross_study_evidence"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    definitions = pd.DataFrame([{
        "contrast_id": "C1", "contrast_field": "intervention",
        "level_a": "control", "level_b": "treated",
    }])

    _, _, _, _, _, _, _, assets = load_panel_responses(
        tmp_path / "effects", tmp_path / "families", "P1", definitions, None,
    )
    assert assets["effect_schema"] == "adjusted_axis_contrast_effects_v1"
    assert assets["effect_status"] == (
        "development_adjusted_effect_object_not_cross_study_evidence"
    )
    assert assets["effect_report_sha256"] == file_sha256(report_path)

    report["status"] = "development_effect_objects_not_cross_study_evidence"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="admitted frozen schema/status"):
        load_panel_responses(
            tmp_path / "effects", tmp_path / "families", "P1", definitions, None,
        )


def test_reverse_summary_swaps_every_side_specific_denominator() -> None:
    from run_pan_redu_panel_pair_partial_transport import reverse_summary

    reversed_summary = reverse_summary({
        "left_entities": 3,
        "right_entities": 5,
        "fixed_capacity_left_total": 0.25,
        "fixed_capacity_right_total": 0.75,
        "signed_chemical_evidence": 0.4,
    })
    assert reversed_summary["left_entities"] == 5
    assert reversed_summary["right_entities"] == 3
    assert reversed_summary["fixed_capacity_left_total"] == 0.75
    assert reversed_summary["fixed_capacity_right_total"] == 0.25
    assert reversed_summary["signed_chemical_evidence"] == 0.4


def test_empty_identity_strata_and_unavailable_reassignment_abstain() -> None:
    from run_pan_redu_panel_pair_partial_transport import (
        identity_axes_or_empty,
        measurement_axes_or_empty,
        reassignment_axes_or_empty,
    )

    def axes(statuses: list[str]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        frame = pd.DataFrame({
            "entity_id": [f"E{i}" for i in range(len(statuses))],
            "identity_status": statuses,
            "fixed_capacity": np.full(len(statuses), 1.0 / len(statuses)),
            "response_score": np.linspace(-0.5, 0.5, len(statuses)),
            "response_available": [True] * len(statuses),
            "precursor_mz": np.arange(len(statuses), dtype=float) + 100.0,
            "median_log2_intensity": np.arange(len(statuses), dtype=float),
            "detection_rate": [1.0] * len(statuses),
            "log2_intensity_iqr": [1.0] * len(statuses),
            "ms2_peak_count": [10] * len(statuses),
        })
        return {"C1": frame.copy()}, frame

    all_unresolved, unresolved_universe = axes(["unresolved_entity"] * 3)
    assert identity_axes_or_empty(all_unresolved, KNOWN := {
        "confirmed_known", "library_supported",
    }, renormalize=True)["C1"].empty
    all_known, _ = axes(["library_supported"] * 3)
    assert identity_axes_or_empty(
        all_known, {"unresolved_entity"}, renormalize=True,
    )["C1"].empty

    one_unresolved, one_unresolved_universe = axes([
        "library_supported", "library_supported", "unresolved_entity",
    ])
    reassigned, ledger, available, reason = reassignment_axes_or_empty(
        one_unresolved_universe, one_unresolved,
    )
    assert not available
    assert "observed_1" in reason
    assert reassigned["C1"].empty
    assert ledger.empty

    empty_pairs = pd.DataFrame(columns=[
        "known_entity_id", "unresolved_entity_id", "distance",
    ])
    matched = measurement_axes_or_empty(
        all_unresolved, empty_pairs, member="unresolved",
    )
    assert matched["C1"].empty

    # Three unresolved entities are sufficient for a deterministic derangement.
    _, ledger, available, reason = reassignment_axes_or_empty(
        unresolved_universe, all_unresolved,
    )
    assert available and not reason and len(ledger) == 3


def test_finalizer_rejects_contract_and_reused_panel_asset_drift() -> None:
    from finalize_pan_redu_partial_transport_matrix import (
        canonical_sha256,
        register_panel_assets,
        require_same_execution_contract,
        validated_execution_contract,
    )

    contract = {
        "pair_output_schema": "pan_redu_panel_pair_fixed_capacity_response_transport_v4",
        "parameters": {"minimum_similarity": 0.7},
        "shared_input_sha256": {"identity_partition": "identity-v1"},
        "implementation_sha256": {"runner": "runner-v1"},
    }
    report = {
        "parameters": contract["parameters"],
        "execution_contract": contract,
        "execution_contract_sha256": canonical_sha256(contract),
    }
    first = validated_execution_contract(report, 0)
    assert require_same_execution_contract(None, first, 0) == first
    changed_contract = {**contract, "parameters": {"minimum_similarity": 0.8}}
    changed_report = {
        "parameters": changed_contract["parameters"],
        "execution_contract": changed_contract,
        "execution_contract_sha256": canonical_sha256(changed_contract),
    }
    second = validated_execution_contract(changed_report, 1)
    with pytest.raises(RuntimeError, match="cross-pair execution-contract drift"):
        require_same_execution_contract(first, second, 1)

    required_fingerprint = {
        "panel_report_sha256": "panel-v1",
        "embeddings_sha256": "embedding-v1",
        "effect_report_sha256": "effect-report-v1",
        "effect_schema": "pan_redu_panel_contrast_effects_v1",
        "effect_status": "development_effect_objects_not_cross_study_evidence",
        "effects_sha256": "effects-v1",
        "families_sha256": "families-v1",
        "measurement_covariates_sha256": "covariates-v1",
    }
    registry: dict[str, dict[str, str]] = {}
    pair0 = pd.Series({"left_panel_id": "P1", "right_panel_id": "P2"})
    register_panel_assets(registry, {
        "panel_assets": {"P1": required_fingerprint, "P2": required_fingerprint},
    }, pair0, 0)
    drifted = {**required_fingerprint, "embeddings_sha256": "embedding-v2"}
    pair1 = pd.Series({"left_panel_id": "P1", "right_panel_id": "P3"})
    with pytest.raises(RuntimeError, match="cross-pair panel-asset drift"):
        register_panel_assets(registry, {
            "panel_assets": {"P1": drifted, "P3": required_fingerprint},
        }, pair1, 1)


def test_frozen_pair_manifest_and_topology_lineage_fail_closed(tmp_path: Path) -> None:
    from run_pan_redu_panel_pair_partial_transport import (
        validate_frozen_pair_manifest,
        validate_pair_topology_lineage,
    )

    pair = pd.DataFrame([{
        "pair_index": 0, "left_panel_id": "P1", "right_panel_id": "P2",
        "left_accession": "S1", "right_accession": "S2",
        "left_source_family": "F1", "right_source_family": "F2",
        "left_polarity": "positive", "right_polarity": "positive",
    }])
    pair_path = tmp_path / "pairs.tsv"
    pair.to_csv(pair_path, sep="\t", index=False)
    topology = pd.DataFrame([
        {"panel_id": "P1", "contrast_id": "C1", "dataset_accession": "S1",
         "preliminary_source_family": "F1"},
        {"panel_id": "P2", "contrast_id": "C1", "dataset_accession": "S2",
         "preliminary_source_family": "F2"},
    ])
    topology_path = tmp_path / "topology.tsv"
    topology.to_csv(topology_path, sep="\t", index=False)
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"panel_id": ["P1", "P2"], "polarity": ["positive", "positive"]}).to_csv(
        metadata_path, sep="\t", index=False,
    )
    write_pair_manifest_report(pair_path, topology_path, metadata_path, 1)
    validate_frozen_pair_manifest(pair_path, topology_path, metadata_path, pair)
    topology.loc[0, "dataset_accession"] = "DRIFT"
    topology.to_csv(topology_path, sep="\t", index=False)
    with pytest.raises(RuntimeError, match="pair-manifest provenance drift"):
        validate_frozen_pair_manifest(pair_path, topology_path, metadata_path, pair)

    left = topology.loc[topology.panel_id.eq("P1")].copy()
    right = topology.loc[topology.panel_id.eq("P2")].copy()
    with pytest.raises(RuntimeError, match="pair/topology dataset_accession drift"):
        validate_pair_topology_lineage(pair.iloc[0], left, right)


def test_source_disjoint_pair_runs_entity_level_partial_transport(tmp_path: Path) -> None:
    write_panel(tmp_path, "P1", [1.0, -1.0, 1.0], [True, False, True])
    write_panel(tmp_path, "P2", [1.0, -1.0, 1.0])
    pairs = pd.DataFrame([{
        "pair_index": 0, "left_panel_id": "P1", "right_panel_id": "P2",
        "left_accession": "S1", "right_accession": "S2",
        "left_source_family": "F1", "right_source_family": "F2",
        "left_polarity": "positive", "right_polarity": "positive",
    }])
    pair_path = tmp_path / "pairs.tsv"
    pairs.to_csv(pair_path, sep="\t", index=False)
    topology = pd.DataFrame([
        {"panel_id": panel, "contrast_id": "C1", "dataset_accession": study,
         "preliminary_source_family": family, "contrast_field": "intervention",
         "level_a": "control", "level_b": "treated", "topology_status": "PASS"}
        for panel, study, family in (("P1", "S1", "F1"), ("P2", "S2", "F2"))
    ])
    topology_path = tmp_path / "topology.tsv"
    topology.to_csv(topology_path, sep="\t", index=False)
    panel_metadata_path = tmp_path / "panel_metadata.tsv"
    pd.DataFrame({
        "panel_id": ["P1", "P2"],
        "polarity": [
            "electrospray ionization (positive)",
            "electrospray ionization (positive)",
        ],
    }).to_csv(panel_metadata_path, sep="\t", index=False)
    write_pair_manifest_report(pair_path, topology_path, panel_metadata_path, len(pairs))
    output = tmp_path / "output"
    script = Path(__file__).with_name("run_pan_redu_panel_pair_partial_transport.py")
    subprocess.run([
        sys.executable, str(script), "--pair-manifest", str(pair_path), "--pair-index", "0",
        "--topology", str(topology_path), "--effects-root", str(tmp_path / "effects"),
        "--panel-metadata", str(panel_metadata_path),
        "--embeddings-root", str(tmp_path / "embeddings"), "--families-root", str(tmp_path / "families"),
        "--output-dir", str(output), "--neighbors", "2", "--minimum-similarity", "0.7",
        "--minimum-entities", "2",
    ], check=True, capture_output=True, text=True)
    scores = pd.read_csv(output / "ordered_contrast_scores.tsv.gz", sep="\t")
    assert len(scores) == 2
    assert scores.answerable.all()
    assert (scores.signed_chemical_evidence > 0).all()
    left_query = scores.loc[scores.query_side.eq("left")].iloc[0]
    right_query = scores.loc[scores.query_side.eq("right")].iloc[0]
    assert int(left_query.left_response_available_entities) == 2
    assert int(left_query.right_response_available_entities) == 3
    assert int(right_query.left_response_available_entities) == 3
    assert int(right_query.right_response_available_entities) == 2
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["denominators"]["ordered_contrast_scores"] == 2
    assert report["identity_partition_available"] is False

    pair_root = tmp_path / "pair_results"
    pair_root.mkdir()
    output.rename(pair_root / "pair_0000")
    final_output = tmp_path / "final"
    finalizer = Path(__file__).with_name("finalize_pan_redu_partial_transport_matrix.py")
    subprocess.run([
        sys.executable, str(finalizer), "--pair-manifest", str(pair_path),
        "--pair-results-root", str(pair_root), "--output-dir", str(final_output),
    ], check=True, capture_output=True, text=True)
    final_report = json.loads((final_output / "report.json").read_text(encoding="utf-8"))
    assert final_report["denominators"]["panel_pairs"] == 1
    assert final_report["denominators"]["ordered_contrast_scores"] == 2

    # A status-compatible legacy estimator must never be silently mixed into
    # the fixed-capacity matrix.
    pair_report_path = pair_root / "pair_0000" / "report.json"
    pair_report = json.loads(pair_report_path.read_text(encoding="utf-8"))
    pair_report["schema"] = "legacy_response_adaptive_transport_v2"
    pair_report_path.write_text(json.dumps(pair_report), encoding="utf-8")
    rejected_output = tmp_path / "rejected_legacy_final"
    rejected = subprocess.run([
        sys.executable, str(finalizer), "--pair-manifest", str(pair_path),
        "--pair-results-root", str(pair_root), "--output-dir", str(rejected_output),
    ], capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "pair estimator/schema drift" in rejected.stderr

    identity_dir = tmp_path / "identity_combined"
    identity_dir.mkdir()
    identity_path = identity_dir / "identity_partition.tsv.gz"
    pd.DataFrame([
        {"panel_id": panel, "anchor_id": anchor, "identity_status": status}
        for panel in ("P1", "P2")
        for anchor, status in (("A", "library_supported"), ("B", "unresolved_entity"), ("C", "unresolved_entity"))
    ]).to_csv(identity_path, sep="\t", index=False, compression="gzip")
    (identity_dir / "report.json").write_text(json.dumps({
        "status": "PHENOTYPE_BLIND_STRICT_LIBRARY_PARTITIONS_COMPLETE",
        "provenance": {
            "combined_partition_sha256": file_sha256(identity_path),
            "sources": [
                {"panel_id": panel, "panel_report_sha256": f"frozen-panel-report-{panel}"}
                for panel in ("P1", "P2")
            ],
        },
    }), encoding="utf-8")
    identified_output = tmp_path / "identified_output"
    subprocess.run([
        sys.executable, str(script), "--pair-manifest", str(pair_path), "--pair-index", "0",
        "--topology", str(topology_path), "--effects-root", str(tmp_path / "effects"),
        "--panel-metadata", str(panel_metadata_path),
        "--embeddings-root", str(tmp_path / "embeddings"), "--families-root", str(tmp_path / "families"),
        "--identity-table", str(identity_path), "--output-dir", str(identified_output),
        "--neighbors", "2", "--minimum-similarity", "0.7", "--minimum-entities", "1",
    ], check=True, capture_output=True, text=True)
    identified_scores = pd.read_csv(identified_output / "ordered_contrast_scores.tsv.gz", sep="\t")
    assert set(identified_scores.world) == {
        "all", "known_only", "known_on_all_common_face",
        "unknown_involving_on_all_common_face", "measurement_matched_known",
        "measurement_matched_unresolved",
        "measurement_matched_known_common_availability",
        "measurement_matched_unresolved_common_availability", "unresolved_only",
        "unresolved_response_reassigned_left", "unresolved_response_reassigned_right",
        "measurement_matched_unresolved_response_reassigned_left",
        "measurement_matched_unresolved_response_reassigned_right",
    }
    assert len(identified_scores) == 26
    assert identified_scores.fixed_capacity_left_total.le(1.0 + 1e-12).all()
    assert identified_scores.fixed_capacity_right_total.le(1.0 + 1e-12).all()
    reassigned = identified_scores.loc[
        identified_scores.world.str.contains("response_reassigned")
    ]
    left_reassigned = reassigned.loc[reassigned.world.str.endswith("_left")]
    right_reassigned = reassigned.loc[reassigned.world.str.endswith("_right")]
    assert left_reassigned.reassigned_panel_id.eq(left_reassigned.left_panel_id).all()
    assert right_reassigned.reassigned_panel_id.eq(right_reassigned.right_panel_id).all()
    matched_rows = identified_scores.loc[
        identified_scores.world.str.startswith("measurement_matched_")
    ]
    assert np.allclose(
        matched_rows.unmatched_mass_mean,
        (
            matched_rows.fixed_capacity_left_total
            + matched_rows.fixed_capacity_right_total
        ) / 2.0,
    )
    identified_report = json.loads(
        (identified_output / "report.json").read_text(encoding="utf-8")
    )
    assert identified_report["reassignment_by_side"]["left"]["reassignment_available"]
    assert identified_report["execution_contract_sha256"]
    assert set(identified_report["panel_assets"]) == {"P1", "P2"}
    assert {
        "greedy_signed_chemical_evidence", "greedy_matched_mass",
        "greedy_intrinsic_score_available", "centroid_cosine",
        "centroid_intrinsic_score_available", "centroid_scope_supported",
        "centroid_left_norm", "centroid_right_norm",
    }.issubset(identified_scores.columns)
    all_world = identified_scores.loc[identified_scores.world.eq("all")].sort_values(
        "query_side"
    )
    assert len(all_world) == 2
    assert np.isclose(
        all_world.greedy_signed_chemical_evidence.iloc[0],
        all_world.greedy_signed_chemical_evidence.iloc[1],
    )
    assert np.isclose(all_world.centroid_cosine.iloc[0], all_world.centroid_cosine.iloc[1])
    assert all_world.centroid_scope_supported.all()
    decomposed = identified_scores.loc[
        identified_scores.world.eq("known_on_all_common_face")
    ]
    assert not decomposed.centroid_scope_supported.any()
    assert not decomposed.centroid_intrinsic_score_available.any()

    # One unresolved entity cannot be deranged.  The fixed 13-world matrix must
    # still complete, with affected stress worlds represented as abstentions.
    sparse_identity_dir = tmp_path / "identity_one_unresolved"
    sparse_identity_dir.mkdir()
    sparse_identity_path = sparse_identity_dir / "identity_partition.tsv.gz"
    pd.DataFrame([
        {"panel_id": panel, "anchor_id": anchor, "identity_status": status}
        for panel in ("P1", "P2")
        for anchor, status in (
            ("A", "library_supported"),
            ("B", "library_supported"),
            ("C", "unresolved_entity"),
        )
    ]).to_csv(sparse_identity_path, sep="\t", index=False, compression="gzip")
    (sparse_identity_dir / "report.json").write_text(json.dumps({
        "status": "PHENOTYPE_BLIND_STRICT_LIBRARY_PARTITIONS_COMPLETE",
        "provenance": {
            "combined_partition_sha256": file_sha256(sparse_identity_path),
            "sources": [
                {"panel_id": panel, "panel_report_sha256": f"frozen-panel-report-{panel}"}
                for panel in ("P1", "P2")
            ],
        },
    }), encoding="utf-8")
    sparse_output = tmp_path / "one_unresolved_output"
    subprocess.run([
        sys.executable, str(script), "--pair-manifest", str(pair_path), "--pair-index", "0",
        "--topology", str(topology_path), "--effects-root", str(tmp_path / "effects"),
        "--panel-metadata", str(panel_metadata_path),
        "--embeddings-root", str(tmp_path / "embeddings"), "--families-root", str(tmp_path / "families"),
        "--identity-table", str(sparse_identity_path), "--output-dir", str(sparse_output),
        "--neighbors", "2", "--minimum-similarity", "0.7", "--minimum-entities", "1",
    ], check=True, capture_output=True, text=True)
    sparse_scores = pd.read_csv(sparse_output / "ordered_contrast_scores.tsv.gz", sep="\t")
    assert len(sparse_scores) == 26
    unavailable_stress = sparse_scores.world.str.contains("response_reassigned")
    assert not sparse_scores.loc[unavailable_stress, "answerable"].any()
    sparse_report = json.loads((sparse_output / "report.json").read_text(encoding="utf-8"))
    assert not sparse_report["reassignment_by_side"]["left"]["reassignment_available"]
    assert sparse_report["reassignment_by_side"]["left"]["unavailable_reason"].endswith("observed_1")


def test_panel_ion_family_builder_uses_frozen_unlabelled_panel(tmp_path: Path) -> None:
    panel = tmp_path / "panel"
    panel.mkdir()
    (panel / "report.json").write_text(json.dumps({
        "status": "development_consumed_abundance_matrix_frozen"
    }), encoding="utf-8")
    values = np.vstack([np.arange(1.0, 21.0), np.arange(1.0, 21.0) * 2])
    np.savez_compressed(
        panel / "ms1_peak_height.npz", values=values,
        anchor_id=np.asarray(["M", "M1"]), sample_id=np.asarray([f"S{i}" for i in range(20)]),
    )
    pd.DataFrame({
        "anchor_id": ["M", "M1"], "precursor_mz": [100.0, 101.003354835],
        "rt_seconds": [20.0, 20.2],
    }).to_csv(panel / "anchor_manifest.tsv", sep="\t", index=False)
    (panel / "representative_spectra.mgf").write_text(
        "BEGIN IONS\nFEATURE_ID=M\nPEPMASS=100\n10 1\n20 2\n30 3\nEND IONS\n"
        "BEGIN IONS\nFEATURE_ID=M1\nPEPMASS=101.003354835\n10 1\n20 2\n30 3\nEND IONS\n",
        encoding="utf-8",
    )
    embedding_path = tmp_path / "embeddings.npz"
    np.savez_compressed(
        embedding_path, anchor_id=np.asarray(["M", "M1"]),
        embeddings=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    output = tmp_path / "families"
    script = Path(__file__).with_name("build_pan_redu_panel_ion_families.py")
    subprocess.run([
        sys.executable, str(script), "--panel-object", str(panel),
        "--embeddings", str(embedding_path), "--panel-id", "P1",
        "--output-dir", str(output), "--minimum-codetected", "6",
    ], check=True, capture_output=True, text=True)
    assignments = pd.read_csv(output / "ion_family_assignments.tsv.gz", sep="\t")
    assert assignments.ion_family_id.nunique() == 1
    assert assignments.capacity_weight.tolist() == [0.5, 0.5]
    covariates = pd.read_csv(output / "measurement_covariates.tsv.gz", sep="\t")
    assert covariates.ms2_peak_count.tolist() == [3, 3]


def test_pair_builder_excludes_wrong_polarity_before_any_response_score(tmp_path: Path) -> None:
    topology = pd.DataFrame([
        {
            "panel_id": panel, "dataset_accession": accession,
            "preliminary_source_family": family, "contrast_id": contrast,
            "topology_status": "PASS",
        }
        for panel, accession, family, contrast in (
            ("P1", "S1", "F1", "C1"),
            ("P2", "S2", "F2", "C2"),
            ("N1", "S3", "F3", "C3"),
        )
    ])
    topology_path = tmp_path / "topology.tsv"
    topology.to_csv(topology_path, sep="\t", index=False)
    metadata_path = tmp_path / "panels.tsv"
    pd.DataFrame({
        "panel_id": ["P1", "P2", "N1"],
        "polarity": [
            "electrospray ionization (positive)",
            "electrospray ionization (positive)",
            "electrospray ionization (negative)",
        ],
    }).to_csv(metadata_path, sep="\t", index=False)
    output = tmp_path / "pairs"
    script = Path(__file__).with_name("prepare_pan_redu_partial_transport_pairs.py")
    subprocess.run([
        sys.executable, str(script), "--topology", str(topology_path),
        "--panel-metadata", str(metadata_path), "--required-polarity", "positive",
        "--output-dir", str(output),
    ], check=True, capture_output=True, text=True)
    pairs = pd.read_csv(output / "panel_pairs.tsv", sep="\t")
    assert pairs[["left_panel_id", "right_panel_id"]].values.tolist() == [["P1", "P2"]]
    assert set(pairs.left_polarity) == {"positive"}
    assert set(pairs.right_polarity) == {"positive"}
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["denominators"]["excluded_wrong_polarity_panels"] == 1
    assert report["denominators"]["excluded_wrong_polarity_contrasts"] == 1


def test_identity_finalizer_requires_and_combines_every_panel(tmp_path: Path) -> None:
    jobs = pd.DataFrame({"job_index": [0, 1], "panel_id": ["P1", "P2"]})
    jobs_path = tmp_path / "jobs.tsv"
    jobs.to_csv(jobs_path, sep="\t", index=False)
    root = tmp_path / "partitions"
    for panel in ("P1", "P2"):
        directory = root / panel
        directory.mkdir(parents=True)
        partition_path = directory / "identity_partition.tsv.gz"
        pd.DataFrame({
            "panel_id": [panel], "anchor_id": ["A"],
            "identity_status": ["unresolved_entity"],
        }).to_csv(partition_path, sep="\t", index=False, compression="gzip")
        (directory / "report.json").write_text(json.dumps({
            "status": "PHENOTYPE_BLIND_STRICT_LIBRARY_PARTITION_FROZEN",
            "panel_id": panel,
            "denominators": {"panel_anchors": 1, "library_supported": 0},
            "provenance": {
                "partition_sha256": file_sha256(partition_path),
                "panel_report_sha256": f"panel-object-{panel}",
            },
        }), encoding="utf-8")
    output = tmp_path / "combined"
    finalizer = Path(__file__).with_name("finalize_pan_redu_panel_identity.py")
    subprocess.run([
        sys.executable, str(finalizer), "--panel-jobs", str(jobs_path),
        "--partitions-root", str(root), "--output-dir", str(output),
    ], check=True, capture_output=True, text=True)
    combined = pd.read_csv(output / "identity_partition.tsv.gz", sep="\t")
    assert len(combined) == 2
    assert set(combined.panel_id) == {"P1", "P2"}
