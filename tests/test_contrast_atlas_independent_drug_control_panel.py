from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tasks" / "build_contrast_atlas_independent_drug_control_panel.py"
SPEC = importlib.util.spec_from_file_location("drug_control_panel", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_community_calls_preserve_author_call_and_identifier_mismatch(tmp_path: Path) -> None:
    activity = pd.DataFrame(
        [
            {
                "Drug": label,
                "Sample": sample,
                "Status": "SAMPLE",
                "Per_drug_thresh": 0.2,
                "Type_metab": call,
                "Slope": slope,
            }
            for label in ("OMEPRAZOLE", "TELMISARTAN", "SERTRALINE_HYDROCHLORIDE")
            for sample, call, slope in (
                ("WLS-001", "T", -0.2),
                ("WLS-250", "F", 0.0),
                ("Z-901", None, -0.1),
            )
        ]
    )
    identifiers = pd.DataFrame(
        [
            {
                "id": "ERS1",
                "Sample": "WLS-001",
                "taxId": 408170,
                "scientificName": "human gut metagenome",
            },
            {
                "id": "ERS2",
                "Sample": "WLS-240",
                "taxId": 408170,
                "scientificName": "human gut metagenome",
            },
        ]
    )
    activity_path = tmp_path / "activity.csv"
    identifiers_path = tmp_path / "identifiers.csv"
    activity.to_csv(activity_path, index=False)
    identifiers.to_csv(identifiers_path, index=False)

    calls, mismatch = MODULE.read_community_calls(activity_path, identifiers_path)

    assert len(calls) == 9
    assert int(calls["evaluable"].sum()) == 6
    assert int(calls["author_no_bacteria_adjusted_positive"].sum()) == 3
    assert int(calls["mapped_human_community"].sum()) == 3
    assert mismatch["evaluable_activity_samples_absent_from_identifier_table"] == ["WLS-250"]
    assert mismatch["identifier_samples_absent_from_target_drug_activity_union"] == ["WLS-240"]


def test_summary_keeps_biology_roles_separate_from_consumed_retrieval() -> None:
    mtbls_rows = []
    community_rows = []
    wse_rows = []
    desired_strain_positive = {"omeprazole": 3, "telmisartan": 2, "sertraline": 0}
    desired_control_positive = {"omeprazole": 2, "telmisartan": 0, "sertraline": 0}
    ranks = {
        "omeprazole": ((16, 4), (17, 2)),
        "telmisartan": ((1, 6), (1, 6)),
        "sertraline": ((4, 5), (8, 5)),
    }
    directions = [
        "MSV000094899_to_MSV000093571",
        "MSV000093571_to_MSV000094899",
    ]
    for drug_key in MODULE.DRUG_CONFIG:
        for index in range(4):
            mtbls_rows.append(
                {
                    "drug_key": drug_key,
                    "test_unit_type": "bacterial_strain",
                    "test_unit": f"strain-{index}",
                    "evaluable": True,
                    "replicated_table_rule_positive": (
                        index < desired_strain_positive[drug_key]
                    ),
                    "consumed_pct": float(index + 1),
                }
            )
        for index in range(2):
            mtbls_rows.append(
                {
                    "drug_key": drug_key,
                    "test_unit_type": "abiotic_pH_control",
                    "test_unit": f"control-{index}",
                    "evaluable": True,
                    "replicated_table_rule_positive": (
                        index < desired_control_positive[drug_key]
                    ),
                    "consumed_pct": float(index + 1),
                }
            )
        for index, positive in enumerate((True, False)):
            community_rows.append(
                {
                    "drug_key": drug_key,
                    "test_unit": f"community-{index}",
                    "evaluable": True,
                    "author_no_bacteria_adjusted_positive": positive,
                    "mapped_human_community": index == 0,
                    "identifier_mapping_status": "mapped",
                    "slope": -0.2 if positive else 0.0,
                }
            )
        for direction_index, direction in enumerate(directions):
            known_rank, unresolved_rank = ranks[drug_key][direction_index]
            for world, rank in (
                ("all", unresolved_rank),
                ("known", known_rank),
                ("unresolved", unresolved_rank),
            ):
                wse_rows.append(
                    {
                        "drug_key": drug_key,
                        "direction": direction,
                        "evidence_world": world,
                        "similar_face_rank": rank,
                        "similar_face_query_maxT_p": 0.01,
                        "mutual_edges": 10 if world == "unresolved" else 2,
                    }
                )

    summary = MODULE.build_summary(
        pd.DataFrame(mtbls_rows),
        pd.DataFrame(community_rows),
        pd.DataFrame(wse_rows),
        {
            "evaluable_activity_samples_absent_from_identifier_table": [],
            "identifier_samples_absent_from_target_drug_activity_union": [],
        },
    ).set_index("drug_key")

    assert summary.loc["telmisartan", "biology_role"] == (
        "robust_cross_project_biological_positive_control"
    )
    assert summary.loc["sertraline", "biology_role"] == (
        "context_heterogeneity_control_not_global_negative"
    )
    assert summary.loc[
        "omeprazole",
        "wse_msv000094899_to_msv000093571_descriptive_known_minus_unresolved_similar_face_rank_shift",
    ] == 12
    assert summary.loc[
        "telmisartan",
        "wse_msv000094899_to_msv000093571_descriptive_known_minus_unresolved_similar_face_rank_shift",
    ] == -5


def test_real_sources_reproduce_frozen_three_drug_denominators(tmp_path: Path) -> None:
    source_dir = ROOT / "data" / "reference" / "contrast_atlas_drug_controls_20261031"
    mtbls = source_dir / "mtbls896_processed" / "TableS3_drug_fold_changes.csv"
    activity = source_dir / "activity_summary_table.csv"
    identifiers = source_dir / "community_identifiers.csv"
    ranks = (
        ROOT
        / "data"
        / "validation"
        / "contrast_atlas_complete_identity_wse_open_retrieval_v5_msv000080918_20261031"
        / "registered_relation_ranks.tsv"
    )
    if not all(path.is_file() for path in (mtbls, activity, identifiers, ranks)):
        pytest.skip("Official source files are not installed in this checkout")

    report = MODULE.build_panel(
        mtbls896_fold_changes=mtbls,
        community_activity=activity,
        community_identifiers=identifiers,
        wse_registered_ranks=ranks,
        output_dir=tmp_path / "panel",
    )
    summary = pd.read_csv(tmp_path / "panel" / "drug_control_summary.tsv", sep="\t").set_index(
        "drug_key"
    )

    assert report["status"] == "DRUG_CONTROL_PANEL_COMPLETE"
    assert report["claim_flags"]["unknown_increment_proven"] is False
    assert report["claim_flags"]["full23_increment_proven"] is False
    assert report["overlap_scope"]["three_way_overlap_count"] == 3
    assert report["overlap_scope"]["consumed_wse_exact_drugs"] == 12
    assert (summary.loc["omeprazole", [
        "mtbls896_strains_positive",
        "mtbls896_strains_evaluable",
        "mtbls896_ph_controls_positive",
        "mtbls896_ph_controls_evaluable",
        "community_author_positive",
        "community_evaluable",
    ]].tolist() == [75, 76, 3, 4, 69, 88])
    assert (summary.loc["telmisartan", [
        "mtbls896_strains_positive",
        "mtbls896_strains_evaluable",
        "mtbls896_ph_controls_positive",
        "mtbls896_ph_controls_evaluable",
        "community_author_positive",
        "community_evaluable",
    ]].tolist() == [12, 76, 0, 4, 83, 89])
    assert (summary.loc["sertraline", [
        "mtbls896_strains_positive",
        "mtbls896_strains_evaluable",
        "mtbls896_ph_controls_positive",
        "mtbls896_ph_controls_evaluable",
        "community_author_positive",
        "community_evaluable",
    ]].tolist() == [0, 76, 0, 4, 49, 87])
    assert summary.loc[
        "omeprazole",
        "wse_msv000093571_to_msv000094899_descriptive_known_minus_unresolved_similar_face_rank_shift",
    ] == 15
