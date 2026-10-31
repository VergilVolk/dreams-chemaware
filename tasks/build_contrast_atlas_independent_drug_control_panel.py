#!/usr/bin/env python3
"""Build an independent biology-control panel for three Atlas drug responses.

This script deliberately keeps three evidence roles separate:

* MTBLS896: published single-strain and abiotic pH-control measurements;
* MTBLS6166 / Zenodo 13943903: a later community project with author-supplied,
  no-bacteria-adjusted metabolic calls;
* ChemProp Atlas: already-consumed, same-parent WSE response retrieval.

The independent biology sources adjudicate whether a drug response is a useful
positive, conditional, or context-heterogeneity control.  They do not turn the
consumed WSE retrieval into an external algorithm benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd


SCHEMA = "contrast_atlas_independent_drug_control_panel_v1"
STATUS = "DRUG_CONTROL_PANEL_COMPLETE"
TARGET_STUDIES = frozenset({"MSV000093571", "MSV000094899"})
IDENTITY_TRACK = "same_precursor_10ppm_identity_hypothesis"
EXACT_DRUG_CLASS = "consumed_exact_drug_cross_design"
SALT_TOKENS = (
    " HYDROCHLORIDE",
    " HCL",
    " SULFATE",
    " CITRATE",
    " MALEATE",
    " FUMARATE",
    " TARTRATE",
    " MESYLATE",
    " TROMETHAMINE",
)

DRUG_CONFIG: Mapping[str, Mapping[str, str]] = {
    "omeprazole": {
        "mtbls896_label": "OMEPRAZOLE",
        "community_label": "OMEPRAZOLE",
        "contrast_id": "drug_omeprazole",
        "biology_role": "conditional_no_bacteria_adjusted_community_positive",
        "claim_boundary": (
            "Community activity is independently supported after the authors' "
            "no-bacteria adjustment, but the 2019 pH controls show substantial "
            "abiotic instability; do not call a universal microbiome-specific response."
        ),
    },
    "telmisartan": {
        "mtbls896_label": "TELMISARTAN",
        "community_label": "TELMISARTAN",
        "contrast_id": "drug_telmisartan",
        "biology_role": "robust_cross_project_biological_positive_control",
        "claim_boundary": (
            "Independent strain-selective and community activity support a biological "
            "positive control; this does not establish an unresolved-signal increment."
        ),
    },
    "sertraline": {
        "mtbls896_label": "SERTRALINE",
        "community_label": "SERTRALINE_HYDROCHLORIDE",
        "contrast_id": "drug_sertraline",
        "biology_role": "context_heterogeneity_control_not_global_negative",
        "claim_boundary": (
            "The isolated-strain screen is negative while many communities are positive; "
            "the same chemical identity must not be assigned one global response label."
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_spaces(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def canonical_drug_name(value: object) -> str:
    name = _normalise_spaces(value).upper().replace("_", " ")
    if name.startswith("DRUG "):
        name = name[len("DRUG ") :]
    for token in SALT_TOKENS:
        name = name.replace(token, "")
    name = re.sub(r"[^A-Z0-9]+", "", name)
    return {"PERICYAZINE": "PERICIAZINE"}.get(name, name)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], source: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def read_mtbls896_calls(path: Path) -> pd.DataFrame:
    """Convert the published wide Table S3 into one row per drug and test unit."""
    wide = pd.read_csv(path, sep=";", header=[0, 1])
    if wide.shape[1] < 7:
        raise ValueError("MTBLS896 Table S3 did not parse as the expected wide table")

    drug_column = wide.columns[0]
    threshold_column = wide.columns[1]
    consumed_columns: Dict[str, Tuple[str, str]] = {}
    fdr_columns: Dict[str, Tuple[str, str]] = {}

    for column in wide.columns[2:]:
        metric = _normalise_spaces(column[1])
        if metric.startswith("% consumed ") and not metric.startswith("% consumed STD "):
            organism = _normalise_spaces(metric[len("% consumed ") :])
            consumed_columns[organism] = column
        elif metric.startswith("p(FDR) "):
            organism = _normalise_spaces(metric[len("p(FDR) ") :])
            fdr_columns[organism] = column

    organisms = sorted(set(consumed_columns) & set(fdr_columns))
    if len(organisms) != 80:
        raise ValueError(
            "Expected 80 MTBLS896 test units (76 strains and four pH controls), "
            f"found {len(organisms)}"
        )

    rows: List[Dict[str, Any]] = []
    drug_labels = wide[drug_column].map(_normalise_spaces).str.upper()
    for drug_key, config in DRUG_CONFIG.items():
        expected_label = config["mtbls896_label"]
        selected = wide.loc[drug_labels.eq(expected_label)]
        if len(selected) != 1:
            raise ValueError(
                f"Expected exactly one MTBLS896 row for {expected_label}, found {len(selected)}"
            )
        source_row = selected.iloc[0]
        threshold = float(source_row[threshold_column])
        for organism in organisms:
            consumed = float(source_row[consumed_columns[organism]])
            fdr_p = float(source_row[fdr_columns[organism]])
            specimen_type = (
                "abiotic_pH_control"
                if organism.lower().startswith("control ph ")
                else "bacterial_strain"
            )
            evaluable = math.isfinite(consumed) and math.isfinite(fdr_p)
            positive = bool(evaluable and consumed > threshold and fdr_p <= 0.05)
            rows.append(
                {
                    "source_accession": "MTBLS896",
                    "source_role": "independent_published_single_strain_screen",
                    "drug_key": drug_key,
                    "source_drug_label": expected_label,
                    "test_unit": organism,
                    "test_unit_type": specimen_type,
                    "adaptive_consumption_threshold_pct": threshold,
                    "consumed_pct": consumed,
                    "fdr_p": fdr_p,
                    "evaluable": evaluable,
                    "replicated_table_rule_positive": positive,
                    "positive_rule": (
                        "consumed_pct > TableS3 adaptive_threshold_pct and FDR_p <= 0.05"
                    ),
                }
            )

    result = pd.DataFrame(rows).sort_values(
        ["drug_key", "test_unit_type", "test_unit"], kind="stable"
    )
    counts = result.groupby(["drug_key", "test_unit_type"], sort=True).size()
    for drug_key in DRUG_CONFIG:
        if int(counts.get((drug_key, "bacterial_strain"), 0)) != 76:
            raise ValueError(f"{drug_key} does not have 76 bacterial-strain rows")
        if int(counts.get((drug_key, "abiotic_pH_control"), 0)) != 4:
            raise ValueError(f"{drug_key} does not have four pH-control rows")
    return result.reset_index(drop=True)


def read_community_calls(
    activity_path: Path, identifier_path: Path
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Read author-provided community calls without reinterpreting their threshold."""
    activity = pd.read_csv(activity_path)
    identifiers = pd.read_csv(identifier_path)
    _require_columns(
        activity,
        ["Drug", "Sample", "Status", "Per_drug_thresh", "Type_metab", "Slope"],
        "community activity table",
    )
    _require_columns(
        identifiers,
        ["id", "Sample", "taxId", "scientificName"],
        "community identifier table",
    )
    if identifiers["Sample"].duplicated().any():
        duplicates = sorted(identifiers.loc[identifiers["Sample"].duplicated(), "Sample"].unique())
        raise ValueError(f"Duplicate community identifier rows: {duplicates}")

    label_to_key = {
        config["community_label"]: drug_key for drug_key, config in DRUG_CONFIG.items()
    }
    selected = activity.loc[activity["Drug"].isin(label_to_key)].copy()
    selected["drug_key"] = selected["Drug"].map(label_to_key)
    selected["author_call"] = selected["Type_metab"].astype("string").str.upper()
    selected["evaluable"] = selected["author_call"].isin(["T", "F"])
    selected["author_no_bacteria_adjusted_positive"] = pd.array(
        [
            call == "T" if evaluable else pd.NA
            for call, evaluable in zip(selected["author_call"], selected["evaluable"])
        ],
        dtype="boolean",
    )

    identifier_columns = identifiers.rename(
        columns={
            "id": "repository_identifier",
            "taxId": "tax_id",
            "scientificName": "scientific_name",
        }
    )
    selected = selected.merge(
        identifier_columns,
        on="Sample",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    selected["identifier_mapping_status"] = selected["_merge"].map(
        {"both": "mapped", "left_only": "unmapped_activity_sample", "right_only": "unused"}
    )
    selected = selected.drop(columns=["_merge"])
    selected["mapped_human_community"] = (
        selected["identifier_mapping_status"].eq("mapped")
        & selected["scientific_name"].eq("human gut metagenome")
    )
    selected["source_accession"] = "MTBLS6166"
    selected["source_role"] = (
        "distinct_preprint_project_same_research_lineage_author_no_bacteria_adjusted_calls"
    )
    selected = selected.rename(
        columns={
            "Drug": "source_drug_label",
            "Sample": "test_unit",
            "Status": "status",
            "Per_drug_thresh": "per_drug_threshold",
            "Type_metab": "type_metab",
            "Slope": "slope",
        }
    )
    selected = selected[
        [
            "source_accession",
            "source_role",
            "drug_key",
            "source_drug_label",
            "test_unit",
            "status",
            "per_drug_threshold",
            "type_metab",
            "slope",
            "evaluable",
            "author_no_bacteria_adjusted_positive",
            "identifier_mapping_status",
            "repository_identifier",
            "tax_id",
            "scientific_name",
            "mapped_human_community",
        ]
    ].sort_values(["drug_key", "test_unit"], kind="stable")

    eligible_samples = set(
        selected.loc[selected["evaluable"], "test_unit"].dropna().astype(str)
    )
    identifier_samples = set(identifiers["Sample"].dropna().astype(str))
    mismatch = {
        "evaluable_activity_samples_absent_from_identifier_table": sorted(
            eligible_samples - identifier_samples
        ),
        "identifier_samples_absent_from_target_drug_activity_union": sorted(
            identifier_samples - eligible_samples
        ),
    }
    return selected.reset_index(drop=True), mismatch


def read_consumed_wse_ranks(path: Path) -> pd.DataFrame:
    """Extract the two directed exact-drug queries from the consumed ChemProp pair."""
    ranks = pd.read_csv(path, sep="\t")
    _require_columns(
        ranks,
        [
            "left_study_id",
            "left_contrast_id",
            "right_study_id",
            "right_contrast_id",
            "relation_class",
            "evidence_world",
            "evidence_track",
            "similar_rank",
            "opposite_rank",
            "similar_query_maxT_p",
            "opposite_query_maxT_p",
            "candidate_universe_size",
            "mutual_edges",
            "weighted_evidence_edges",
            "expected_signed_direction",
            "signed_truth_claimed",
            "evidence_boundary",
        ],
        "WSE registered relation ranks",
    )
    contrast_to_key = {
        config["contrast_id"]: drug_key for drug_key, config in DRUG_CONFIG.items()
    }
    selected = ranks.loc[
        ranks["relation_class"].eq(EXACT_DRUG_CLASS)
        & ranks["evidence_track"].eq(IDENTITY_TRACK)
        & ranks["left_study_id"].isin(TARGET_STUDIES)
        & ranks["right_study_id"].isin(TARGET_STUDIES)
        & ranks["left_study_id"].ne(ranks["right_study_id"])
        & ranks["left_contrast_id"].isin(contrast_to_key)
        & ranks["left_contrast_id"].eq(ranks["right_contrast_id"])
        & ranks["evidence_world"].isin(["all", "known", "unresolved"])
    ].copy()
    selected["drug_key"] = selected["left_contrast_id"].map(contrast_to_key)
    selected["direction"] = (
        selected["left_study_id"] + "_to_" + selected["right_study_id"]
    )
    selected["similar_face_rank"] = pd.to_numeric(
        selected["similar_rank"], errors="coerce"
    )
    selected["similar_face_query_maxT_p"] = pd.to_numeric(
        selected["similar_query_maxT_p"], errors="coerce"
    )
    selected["opposite_face_rank"] = pd.to_numeric(
        selected["opposite_rank"], errors="coerce"
    )
    selected["opposite_face_query_maxT_p"] = pd.to_numeric(
        selected["opposite_query_maxT_p"], errors="coerce"
    )
    selected["data_role"] = "development_consumed_same_parent_chemprop"
    selected["claim_boundary"] = (
        "descriptive consumed retrieval only; not independent algorithm validation"
    )
    selected = selected[
        [
            "drug_key",
            "direction",
            "left_study_id",
            "left_contrast_id",
            "right_study_id",
            "right_contrast_id",
            "evidence_world",
            "evidence_track",
            "similar_face_rank",
            "similar_face_query_maxT_p",
            "opposite_face_rank",
            "opposite_face_query_maxT_p",
            "candidate_universe_size",
            "mutual_edges",
            "weighted_evidence_edges",
            "expected_signed_direction",
            "signed_truth_claimed",
            "evidence_boundary",
            "data_role",
            "claim_boundary",
        ]
    ].sort_values(["drug_key", "direction", "evidence_world"], kind="stable")

    expected_rows = len(DRUG_CONFIG) * 2 * 3
    if len(selected) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} consumed exact-drug rows, found {len(selected)}"
        )
    multiplicity = selected.groupby(["drug_key", "direction", "evidence_world"]).size()
    if not multiplicity.eq(1).all():
        raise ValueError("Consumed WSE exact-drug rows are not unique per drug/direction/world")
    return selected.reset_index(drop=True)


def build_summary(
    mtbls_calls: pd.DataFrame,
    community_calls: pd.DataFrame,
    wse_ranks: pd.DataFrame,
    mismatch: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    directions = [
        "MSV000094899_to_MSV000093571",
        "MSV000093571_to_MSV000094899",
    ]
    for drug_key, config in DRUG_CONFIG.items():
        strain = mtbls_calls.loc[
            mtbls_calls["drug_key"].eq(drug_key)
            & mtbls_calls["test_unit_type"].eq("bacterial_strain")
            & mtbls_calls["evaluable"]
        ]
        ph_control = mtbls_calls.loc[
            mtbls_calls["drug_key"].eq(drug_key)
            & mtbls_calls["test_unit_type"].eq("abiotic_pH_control")
            & mtbls_calls["evaluable"]
        ]
        community = community_calls.loc[
            community_calls["drug_key"].eq(drug_key) & community_calls["evaluable"]
        ]
        human = community.loc[community["mapped_human_community"]]
        unmapped = sorted(
            community.loc[
                community["identifier_mapping_status"].eq("unmapped_activity_sample"),
                "test_unit",
            ].astype(str).unique()
        )
        row: Dict[str, Any] = {
            "drug_key": drug_key,
            "biology_role": config["biology_role"],
            "claim_boundary": config["claim_boundary"],
            "mtbls896_strains_evaluable": int(len(strain)),
            "mtbls896_strains_positive": int(
                strain["replicated_table_rule_positive"].sum()
            ),
            "mtbls896_strain_positive_fraction": float(
                strain["replicated_table_rule_positive"].mean()
            ),
            "mtbls896_strain_median_consumed_pct": float(strain["consumed_pct"].median()),
            "mtbls896_strain_min_consumed_pct": float(strain["consumed_pct"].min()),
            "mtbls896_strain_max_consumed_pct": float(strain["consumed_pct"].max()),
            "mtbls896_ph_controls_evaluable": int(len(ph_control)),
            "mtbls896_ph_controls_positive": int(
                ph_control["replicated_table_rule_positive"].sum()
            ),
            "community_rows_total": int(
                community_calls["drug_key"].eq(drug_key).sum()
            ),
            "community_evaluable": int(len(community)),
            "community_author_positive": int(
                community["author_no_bacteria_adjusted_positive"].sum()
            ),
            "community_author_positive_fraction": float(
                community["author_no_bacteria_adjusted_positive"].mean()
            ),
            "community_median_slope": float(community["slope"].median()),
            "community_positive_median_slope": float(
                community.loc[
                    community["author_no_bacteria_adjusted_positive"].fillna(False), "slope"
                ].median()
            ),
            "mapped_human_communities_evaluable": int(len(human)),
            "mapped_human_communities_author_positive": int(
                human["author_no_bacteria_adjusted_positive"].sum()
            ),
            "unmapped_evaluable_activity_samples": ";".join(unmapped),
            "activity_samples_absent_from_identifier_table": ";".join(
                mismatch["evaluable_activity_samples_absent_from_identifier_table"]
            ),
            "identifier_samples_absent_from_target_drug_activity_union": ";".join(
                mismatch["identifier_samples_absent_from_target_drug_activity_union"]
            ),
        }
        for direction in directions:
            direction_rows = wse_ranks.loc[
                wse_ranks["drug_key"].eq(drug_key)
                & wse_ranks["direction"].eq(direction)
            ].set_index("evidence_world")
            prefix = f"wse_{direction.lower()}"
            for world in ("all", "known", "unresolved"):
                row[f"{prefix}_{world}_similar_face_rank"] = int(
                    direction_rows.loc[world, "similar_face_rank"]
                )
                row[f"{prefix}_{world}_similar_face_query_maxT_p"] = float(
                    direction_rows.loc[world, "similar_face_query_maxT_p"]
                )
                row[f"{prefix}_{world}_mutual_edges"] = int(
                    direction_rows.loc[world, "mutual_edges"]
                )
            row[
                f"{prefix}_descriptive_known_minus_unresolved_similar_face_rank_shift"
            ] = int(
                direction_rows.loc["known", "similar_face_rank"]
                - direction_rows.loc["unresolved", "similar_face_rank"]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("drug_key", kind="stable").reset_index(drop=True)


def audit_three_way_overlap(
    mtbls896_fold_changes: Path,
    community_activity: Path,
    wse_registered_ranks: Path,
) -> Dict[str, Any]:
    """Prove exhaustive parent-drug overlap under an explicit salt normalization."""
    mtbls_labels = pd.read_csv(
        mtbls896_fold_changes, sep=";", header=1, usecols=[0]
    ).iloc[:, 0]
    community_labels = pd.read_csv(
        community_activity, usecols=["Drug"]
    )["Drug"].dropna().drop_duplicates()
    community_labels = community_labels.loc[
        ~community_labels.astype(str).str.startswith("IS_")
    ]
    ranks = pd.read_csv(wse_registered_ranks, sep="\t")
    wse_labels = ranks.loc[
        ranks["relation_class"].eq(EXACT_DRUG_CLASS), "left_contrast_id"
    ].dropna().drop_duplicates()
    sets = {
        "mtbls896": {canonical_drug_name(value) for value in mtbls_labels},
        "mtbls6166_processed": {
            canonical_drug_name(value) for value in community_labels
        },
        "consumed_wse_exact_drugs": {
            canonical_drug_name(value) for value in wse_labels
        },
    }
    overlap = sorted(set.intersection(*sets.values()))
    expected = sorted(canonical_drug_name(key) for key in DRUG_CONFIG)
    if overlap != expected:
        raise ValueError(
            f"Unexpected three-way drug overlap: observed {overlap}, expected {expected}"
        )
    return {
        "mtbls896_unique_drugs": len(sets["mtbls896"]),
        "mtbls6166_processed_non_internal_standard_drugs": len(
            sets["mtbls6166_processed"]
        ),
        "consumed_wse_exact_drugs": len(sets["consumed_wse_exact_drugs"]),
        "three_way_overlap_drugs": overlap,
        "three_way_overlap_count": len(overlap),
        "interpretation": (
            "The three reported drugs are the exhaustive canonicalized parent-drug "
            "overlap under the stated salt normalization, not a subset selected from "
            "additional eligible controls. Original source labels remain in detail files."
        ),
    }


def _json_safe(value: Any) -> Any:
    if value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_panel(
    mtbls896_fold_changes: Path,
    community_activity: Path,
    community_identifiers: Path,
    wse_registered_ranks: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    for source in (
        mtbls896_fold_changes,
        community_activity,
        community_identifiers,
        wse_registered_ranks,
    ):
        if not source.is_file():
            raise FileNotFoundError(source)

    mtbls_calls = read_mtbls896_calls(mtbls896_fold_changes)
    community_calls, mismatch = read_community_calls(
        community_activity, community_identifiers
    )
    wse_ranks = read_consumed_wse_ranks(wse_registered_ranks)
    summary = build_summary(mtbls_calls, community_calls, wse_ranks, mismatch)
    overlap_scope = audit_three_way_overlap(
        mtbls896_fold_changes, community_activity, wse_registered_ranks
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "summary": output_dir / "drug_control_summary.tsv",
        "mtbls896_calls": output_dir / "mtbls896_drug_strain_calls.tsv",
        "community_calls": output_dir / "mtbls6166_drug_community_calls.tsv",
        "wse_ranks": output_dir / "wse_consumed_exact_drug_ranks.tsv",
    }
    summary.to_csv(output_paths["summary"], sep="\t", index=False)
    mtbls_calls.to_csv(output_paths["mtbls896_calls"], sep="\t", index=False)
    community_calls.to_csv(output_paths["community_calls"], sep="\t", index=False)
    wse_ranks.to_csv(output_paths["wse_ranks"], sep="\t", index=False)

    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "selection_status": (
            "post_hoc_independent_contextual_adjudication_not_a_sealed_benchmark"
        ),
        "data_roles": {
            "MTBLS896": (
                "independent published 2019 single-strain project with explicit pH controls; "
                "not a sealed prospective test"
            ),
            "MTBLS6166_Zenodo_13943903": (
                "distinct 2026 preprint community project from the same research lineage as "
                "MTBLS896; author Type_metab calls compare communities with no-bacteria "
                "controls. We reuse the processed calls and do not recompute them from raw; "
                "not peer reviewed and not sealed"
            ),
            "chemprop_wse": (
                "development-consumed same-parent cross-design Atlas retrieval; descriptive only"
            ),
        },
        "denominators": {
            "drugs": int(len(summary)),
            "mtbls896_detail_rows": int(len(mtbls_calls)),
            "mtbls896_strains_per_drug": 76,
            "mtbls896_ph_controls_per_drug": 4,
            "community_detail_rows": int(len(community_calls)),
            "community_evaluable_rows": int(community_calls["evaluable"].sum()),
            "consumed_wse_rows": int(len(wse_ranks)),
            "consumed_wse_directions_per_drug": 2,
            "consumed_wse_worlds_per_direction": 3,
        },
        "community_identifier_mismatch": mismatch,
        "overlap_scope": overlap_scope,
        "drug_controls": _json_safe(summary.to_dict(orient="records")),
        "scientific_verdict": {
            "telmisartan": (
                "robust distinct-project biological positive within one research lineage; "
                "known-only WSE has similar-face rank 1 in both consumed directions, so "
                "this case is not evidence of unresolved increment"
            ),
            "omeprazole": (
                "conditional community positive after no-bacteria adjustment; 2019 pH controls "
                "show abiotic instability. Unresolved WSE has a better similar-face rank than "
                "known-only in both consumed directions, but signed truth was unspecified and "
                "this is a post-hoc descriptive case"
            ),
            "sertraline": (
                "context-heterogeneity control: zero positive isolated strains but many positive "
                "communities; do not force one context-free response truth"
            ),
        },
        "claim_flags": {
            "external_biology_support_for_control_roles": True,
            "distinct_project_support": True,
            "cross_lab_independent_support": False,
            "community_call_recomputed_from_raw": False,
            "registered_signed_truth_claimed": False,
            "unresolved_means_novel_molecule": False,
            "opportunity_matched_unknown_increment": False,
            "unknown_increment_proven": False,
            "full23_increment_proven": False,
            "self_algorithm_increment_proven": False,
            "prospective_validation": False,
            "mechanism_proven": False,
        },
        "input_provenance": {
            "mtbls896_fold_changes": {
                "path": str(mtbls896_fold_changes),
                "sha256": sha256_file(mtbls896_fold_changes),
                "source": "https://doi.org/10.6084/m9.figshare.8119058",
            },
            "community_activity": {
                "path": str(community_activity),
                "sha256": sha256_file(community_activity),
                "source": "https://zenodo.org/records/13943903",
            },
            "community_identifiers": {
                "path": str(community_identifiers),
                "sha256": sha256_file(community_identifiers),
                "source": "https://zenodo.org/records/13943903",
            },
            "wse_registered_ranks": {
                "path": str(wse_registered_ranks),
                "sha256": sha256_file(wse_registered_ranks),
                "source": "local consumed Contrast Atlas result",
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
    }

    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_paths["report"] = report_path
    checksums_path = output_dir / "checksums.sha256"
    checksums_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in sorted(output_paths.values(), key=lambda item: item.name)
        ),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtbls896-fold-changes", type=Path, required=True)
    parser.add_argument("--community-activity", type=Path, required=True)
    parser.add_argument("--community-identifiers", type=Path, required=True)
    parser.add_argument("--wse-registered-ranks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_panel(
        mtbls896_fold_changes=args.mtbls896_fold_changes,
        community_activity=args.community_activity,
        community_identifiers=args.community_identifiers,
        wse_registered_ranks=args.wse_registered_ranks,
        output_dir=args.output_dir,
    )
    print(json.dumps(report["scientific_verdict"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
