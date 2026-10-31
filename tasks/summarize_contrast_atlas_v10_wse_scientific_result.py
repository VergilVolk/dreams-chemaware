#!/usr/bin/env python
"""Summarize the v10 WSE Atlas result without promoting hypotheses to truth.

This script does not change candidate admission, spectral scores, response effects,
or retrieval results.  It extracts the global response-specificity diagnostic and
the new MSV000080918 source's traceable evidence into a compact result package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


NEW_STUDY = "MSV000080918"
IGA_CONTRAST = "mup_hfd_10_11m_iga_ko_vs_intact_date_matched"
CD8_CONTRAST = "mup_hfd_10_11m_cd8_ko_vs_intact_date_matched"
AD_STUDY = "MSV000090788"
AD_CONTRAST = "lesion_vs_nonlesion_paired"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def spearman(left: pd.Series, right: pd.Series) -> float:
    return float(left.rank(method="average").corr(right.rank(method="average")))


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values, kind="mergesort")
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def contrast_effects(
    collection: pd.DataFrame, study_id: str, contrast_id: str
) -> pd.DataFrame:
    frame = collection.loc[
        (collection["study_id"] == study_id)
        & (collection["contrast_id"] == contrast_id),
        ["entity_id", "beta", "se", "prevalence"],
    ].copy()
    if not len(frame):
        raise ValueError(f"missing contrast: {study_id} {contrast_id}")
    z = np.divide(
        frame["beta"].to_numpy(float),
        frame["se"].to_numpy(float),
        out=np.zeros(len(frame), dtype=np.float64),
        where=frame["se"].to_numpy(float) > 0,
    )
    p = np.asarray([math.erfc(abs(value) / math.sqrt(2.0)) for value in z])
    frame["normal_approx_q_diagnostic"] = bh_adjust(p)
    frame["response_weight"] = np.divide(
        frame["beta"].to_numpy(float),
        np.sqrt(frame["beta"].to_numpy(float) ** 2 + frame["se"].to_numpy(float) ** 2),
        out=np.zeros(len(frame), dtype=np.float64),
        where=np.isfinite(frame["beta"].to_numpy(float))
        & np.isfinite(frame["se"].to_numpy(float)),
    )
    return frame


def retrieve_call_count(report: dict, world: str, face: str, result_class: str) -> int:
    rows = [
        row
        for row in report["open_query_calls"]
        if row["evidence_world"] == world
        and row["evidence_track"] == "same_precursor_10ppm_identity_hypothesis"
        and row["response_face"] == face
        and row["result_class"] == result_class
    ]
    if len(rows) != 1:
        raise ValueError((world, face, result_class, rows))
    return int(rows[0]["queries"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval-dir",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_complete_identity_wse_open_retrieval_v8_repeat_"
            "msv000080918_20261031"
        ),
    )
    parser.add_argument(
        "--wse-release-dir",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_complete_identity_wse_release_v4_"
            "msv000080918_20261031"
        ),
    )
    parser.add_argument(
        "--axis-dir",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_model_independent_identity_axis_v6_"
            "msv000080918_20261031"
        ),
    )
    parser.add_argument(
        "--collection-dir",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_open_response_collection_v10_frozenbase_"
            "msv000080918_v7_20261031"
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/validation/contrast_atlas_msv000080918_atlas_source_v5_20261031"),
    )
    parser.add_argument(
        "--source-analysis-dir",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_msv000080918_multistratum_response_v1_20261031"
        ),
    )
    parser.add_argument(
        "--common-library-evidence",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_global_biological_adjudication_v1_20261030/"
            "entity_library_evidence.tsv.gz"
        ),
    )
    parser.add_argument(
        "--ad-db-result",
        type=Path,
        default=Path(
            "data/reference/public_fbmn/a36305db40414b6ea9c5b9a5de380bea/db_result.tsv"
        ),
    )
    parser.add_argument(
        "--spectrum-asset-root",
        type=Path,
        default=Path("D:/DreaMS"),
        help="Fallback root for reference assets absent from the worktree.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_v10_wse_scientific_result_v1_20261031"
        ),
    )
    return parser.parse_args()


def resolve_asset(path: Path, fallback_root: Path) -> Path:
    if path.exists():
        return path.resolve()
    fallback = fallback_root / path
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    retrieval_dir = args.retrieval_dir.resolve()
    wse_release_dir = args.wse_release_dir.resolve()
    axis_dir = args.axis_dir.resolve()
    collection_dir = args.collection_dir.resolve()
    source_dir = args.source_dir.resolve()
    source_analysis_dir = args.source_analysis_dir.resolve()
    common_library_path = args.common_library_evidence.resolve()
    ad_db_path = resolve_asset(args.ad_db_result, args.spectrum_asset_root)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    retrieval_report_path = retrieval_dir / "report.json"
    wse_release_report_path = wse_release_dir / "report.json"
    axis_report_path = axis_dir / "report.json"
    retrieval_report = json.loads(retrieval_report_path.read_text(encoding="utf-8"))
    wse_release_report = json.loads(wse_release_report_path.read_text(encoding="utf-8"))
    axis_report = json.loads(axis_report_path.read_text(encoding="utf-8"))

    top_hits = pd.read_csv(retrieval_dir / "open_local_edge_top_hits.tsv", sep="\t")
    candidate_scores = pd.read_csv(
        retrieval_dir / "open_local_edge_candidate_scores.tsv.gz", sep="\t"
    )
    handoff = pd.read_csv(
        retrieval_dir / "frozen_mutual_edge_algorithm_handoff.tsv.gz", sep="\t"
    )
    pairs = pd.read_csv(
        axis_dir / "directed_complete_candidate_pairs.tsv.gz", sep="\t", low_memory=False
    )
    source_response = pd.read_csv(source_dir / "response_evidence.tsv.gz", sep="\t")
    collection = pd.read_csv(collection_dir / "response_entities.tsv.gz", sep="\t", low_memory=False)
    source_candidates = pd.read_csv(
        source_analysis_dir / "feature_cluster_candidates.tsv.gz", sep="\t"
    )
    common_library = pd.read_csv(common_library_path, sep="\t")
    ad_library = pd.read_csv(ad_db_path, sep="\t")

    identity_track = "same_precursor_10ppm_identity_hypothesis"
    new_top_hits = top_hits.loc[
        (top_hits["query_study_id"] == NEW_STUDY)
        & (top_hits["evidence_track"] == identity_track)
    ].copy()
    new_top_hits = new_top_hits.sort_values(
        ["query_contrast_id", "evidence_world", "response_face"], kind="mergesort"
    )

    response_wide = source_response.pivot_table(
        index="entity_id",
        columns="contrast_id",
        values="beta_log2_treatment_minus_control",
        aggfunc="first",
    ).dropna(subset=[IGA_CONTRAST, CD8_CONTRAST])
    iga = response_wide[IGA_CONTRAST]
    cd8 = response_wide[CD8_CONTRAST]
    pearson_beta = float(iga.corr(cd8))
    spearman_beta = spearman(iga, cd8)
    sign_concordance = float((np.sign(iga) == np.sign(cd8)).mean())

    stable = source_response.loc[
        (source_response["q_cluster_cr1_within_contrast"] <= 0.05)
        & source_response["date_loso_direction_consistent"].fillna(False)
    ]
    stable_sets = {
        contrast: set(rows["entity_id"])
        for contrast, rows in stable.groupby("contrast_id", sort=False)
    }
    iga_stable = stable_sets.get(IGA_CONTRAST, set())
    cd8_stable = stable_sets.get(CD8_CONTRAST, set())

    new_directed_pairs = pairs.loc[
        (pairs["query_study_id"] == NEW_STUDY)
        | (pairs["candidate_study_id"] == NEW_STUDY)
    ]
    new_query_entities_with_candidates = set(
        pairs.loc[pairs["query_study_id"] == NEW_STUDY, "query_entity_id"]
    )
    new_mutual = handoff.loc[
        (handoff["left_study_id"] == NEW_STUDY)
        | (handoff["right_study_id"] == NEW_STUDY)
    ]
    new_entities_with_mutual = set(
        new_mutual.loc[new_mutual["left_study_id"] == NEW_STUDY, "left_entity_id"]
    ) | set(
        new_mutual.loc[new_mutual["right_study_id"] == NEW_STUDY, "right_entity_id"]
    )

    iga_ad_score = candidate_scores.loc[
        (candidate_scores["query_study_id"] == NEW_STUDY)
        & (candidate_scores["query_contrast_id"] == IGA_CONTRAST)
        & (candidate_scores["candidate_study_id"] == AD_STUDY)
        & (candidate_scores["candidate_contrast_id"] == AD_CONTRAST)
        & (candidate_scores["evidence_world"] == "all")
        & (candidate_scores["evidence_track"] == identity_track)
    ]
    if len(iga_ad_score) != 1:
        raise ValueError(f"expected one IgA-to-AD candidate score, found {len(iga_ad_score)}")
    iga_ad_score = iga_ad_score.iloc[0]

    low, high = sorted((NEW_STUDY, AD_STUDY))
    edge_block = handoff.loc[
        (handoff["left_study_id"] == low)
        & (handoff["right_study_id"] == high)
        & (handoff["evidence_world"] == "all")
        & handoff["precursor_compatible_10ppm"].astype(bool)
    ].copy()
    if NEW_STUDY == low:
        edge_block = edge_block.rename(
            columns={
                "left_entity_id": "query_entity_id",
                "right_entity_id": "candidate_entity_id",
                "left_precursor_mz": "query_precursor_mz",
                "right_precursor_mz": "candidate_precursor_mz",
            }
        )
    else:
        edge_block = edge_block.rename(
            columns={
                "right_entity_id": "query_entity_id",
                "left_entity_id": "candidate_entity_id",
                "right_precursor_mz": "query_precursor_mz",
                "left_precursor_mz": "candidate_precursor_mz",
            }
        )
    query_effect = contrast_effects(collection, NEW_STUDY, IGA_CONTRAST).rename(
        columns={
            "entity_id": "query_entity_id",
            "beta": "query_beta",
            "se": "query_se",
            "prevalence": "query_prevalence",
            "normal_approx_q_diagnostic": "query_fdr",
            "response_weight": "query_response_weight",
        }
    )
    candidate_effect = contrast_effects(collection, AD_STUDY, AD_CONTRAST).rename(
        columns={
            "entity_id": "candidate_entity_id",
            "beta": "candidate_beta",
            "se": "candidate_se",
            "prevalence": "candidate_prevalence",
            "normal_approx_q_diagnostic": "candidate_fdr",
            "response_weight": "candidate_response_weight",
        }
    )
    iga_ad = edge_block.merge(query_effect, on="query_entity_id", validate="one_to_one")
    iga_ad = iga_ad.merge(candidate_effect, on="candidate_entity_id", validate="one_to_one")
    iga_ad = iga_ad.loc[
        (np.sign(iga_ad["query_beta"]) != 0)
        & (np.sign(iga_ad["query_beta"]) == np.sign(iga_ad["candidate_beta"]))
    ].copy()
    iga_ad["weighted_edge_contribution"] = (
        iga_ad["chemical_similarity"]
        * iga_ad["query_response_weight"].abs()
        * iga_ad["candidate_response_weight"].abs()
    )
    iga_ad = iga_ad.sort_values("weighted_edge_contribution", ascending=False, kind="mergesort").head(10)
    iga_ad.insert(0, "evidence_rank", np.arange(1, len(iga_ad) + 1))
    iga_ad.insert(0, "candidate_query_maxT_p", float(iga_ad_score["similar_query_maxT_p"]))
    iga_ad.insert(0, "candidate_rank", int(iga_ad_score["similar_rank"]))

    native_new = source_candidates.loc[
        source_candidates["library_supported"].fillna(False),
        ["feature_id", "library_id", "ppm_error", "rt_error_seconds", "ms2_peak_count"],
    ].drop_duplicates("feature_id")
    native_new = native_new.rename(
        columns={
            "feature_id": "query_raw_entity_id",
            "library_id": "query_source_native_library_name",
            "ppm_error": "query_source_native_ppm_error",
            "rt_error_seconds": "query_source_native_rt_error_seconds",
            "ms2_peak_count": "query_source_native_ms2_peak_count",
        }
    )
    native_new["query_raw_entity_id"] = (
        native_new["query_raw_entity_id"].astype(str).str.replace(" ", "", regex=False)
    )
    iga_ad["query_raw_entity_id"] = iga_ad["query_entity_id"].str.split("::", n=1).str[-1]
    iga_ad["candidate_raw_entity_id"] = iga_ad["candidate_entity_id"].str.split("::", n=1).str[-1]
    iga_ad = iga_ad.merge(native_new, on="query_raw_entity_id", how="left", validate="many_to_one")

    ad_native = ad_library.rename(
        columns={
            "#Scan#": "candidate_raw_entity_id",
            "Compound_Name": "candidate_source_native_library_name",
            "MQScore": "candidate_source_native_mqscore",
            "SharedPeaks": "candidate_source_native_shared_peaks",
            "LibraryQualityString": "candidate_source_native_library_quality",
            "Adduct": "candidate_source_native_adduct",
        }
    )[
        [
            "candidate_raw_entity_id",
            "candidate_source_native_library_name",
            "candidate_source_native_mqscore",
            "candidate_source_native_shared_peaks",
            "candidate_source_native_library_quality",
            "candidate_source_native_adduct",
        ]
    ]
    ad_native["candidate_raw_entity_id"] = ad_native["candidate_raw_entity_id"].astype(str)
    iga_ad["candidate_raw_entity_id"] = iga_ad["candidate_raw_entity_id"].astype(str)
    iga_ad = iga_ad.merge(ad_native, on="candidate_raw_entity_id", how="left", validate="many_to_one")

    common = common_library.loc[
        common_library["study_id"] == AD_STUDY,
        [
            "entity_id",
            "library_name",
            "wse",
            "matched_peaks",
            "library_quality",
            "identity_status",
            "library_evidence_class",
        ],
    ].rename(
        columns={
            "entity_id": "candidate_entity_id",
            "library_name": "candidate_common_policy_library_name",
            "wse": "candidate_common_policy_library_wse",
            "matched_peaks": "candidate_common_policy_matched_peaks",
            "library_quality": "candidate_common_policy_library_quality",
            "identity_status": "candidate_common_policy_identity_status",
            "library_evidence_class": "candidate_common_policy_evidence_class",
        }
    )
    iga_ad = iga_ad.merge(common, on="candidate_entity_id", how="left", validate="many_to_one")

    interpretation = {
        ("166.0863_71.3910", "790"): "aromatic_amino_acid_isomer_hypothesis_not_confirmed_identity",
        ("205.0978_109.2506", "1167"): "tryptophan_or_isomer_hypothesis_not_confirmed_identity",
        ("188.0713_108.9686", "996"): "likely_tryptophan_ion_family_duplicate_not_independent_entity",
    }
    iga_ad["scientific_interpretation"] = [
        interpretation.get((q, c), "unannotated_identity_hypothesis")
        for q, c in zip(iga_ad["query_raw_entity_id"], iga_ad["candidate_raw_entity_id"])
    ]
    iga_ad["dual_normal_approx_fdr_005_diagnostic"] = (
        (iga_ad["query_fdr"] <= 0.05) & (iga_ad["candidate_fdr"] <= 0.05)
    )

    top_hits_path = out_dir / "new_source_open_top_hits.tsv"
    candidate_path = out_dir / "iga_ko_to_ad_skin_candidate_evidence.tsv"
    atomic_tsv(new_top_hits, top_hits_path)
    atomic_tsv(iga_ad, candidate_path)

    primary_rows = iga_ad.loc[
        iga_ad["scientific_interpretation"].isin(
            [
                "aromatic_amino_acid_isomer_hypothesis_not_confirmed_identity",
                "tryptophan_or_isomer_hypothesis_not_confirmed_identity",
            ]
        )
    ]
    registered_all = next(
        row
        for row in retrieval_report["registered_relation_descriptive_performance"]
        if row["evidence_world"] == "all"
        and row["evidence_track"] == identity_track
    )

    report = {
        "schema": "contrast_atlas_v10_wse_scientific_result_v1",
        "status": "DEVELOPMENT_CONSUMED_GLOBAL_RETRIEVAL_FAIL_RESPONSE_SPECIFICITY",
        "data_role": "development_consumed",
        "scientific_verdict": {
            "global_open_discovery": "FAIL_RESPONSE_SPECIFICITY",
            "reason": (
                "The current response operator and permutation/max-T calibration call a similar "
                "response for 51/53 heterogeneous queries and an opposite response for 49/53; "
                "these calls cannot be treated as validated biological links."
            ),
            "new_source_use": (
                "MSV000080918 is retained as a composition-matched two-response stress test and "
                "candidate generator, not as independent validation."
            ),
            "aromatic_amino_acid_candidate": (
                "Two high-contribution pairs have source-native phenylalanine/tryptophan labels "
                "on both sides, but common-policy searches expose isomer ambiguity; they are "
                "priority identity hypotheses, not confirmed same molecules or mechanisms."
            ),
            "unknown_increment": "NOT_TESTED_AND_NOT_ESTABLISHED",
            "full23_increment": "NOT_TESTED_NO_COMPLETE_AXIS_SCORE_PACKAGE",
        },
        "global_denominators": {
            "study_source_ids": int(axis_report["denominators"]["studies"]),
            "contrasts": int(retrieval_report["denominators"]["study_level_contrasts"]),
            "entities": int(axis_report["denominators"]["entities"]),
            "directed_candidate_pairs": int(axis_report["denominators"]["directed_candidate_pairs"]),
            "wse_answered_with_positive_spectral_evidence": int(
                wse_release_report["denominators"]["answered_with_positive_spectral_evidence"]
            ),
            "world_specific_mutual_edge_rows": int(
                retrieval_report["denominators"]["world_specific_mutual_edge_rows"]
            ),
            "unique_mutual_entity_pairs": int(
                retrieval_report["denominators"]["unique_mutual_entity_pairs"]
            ),
            "source_disjoint_query_candidate_contrasts": int(
                retrieval_report["denominators"]["directed_query_candidate_pairs"]
            ),
        },
        "global_response_specificity_diagnostic": {
            "all_world_similar_response_queries": retrieve_call_count(
                retrieval_report, "all", "similar", "similar_response"
            ),
            "all_world_opposite_response_queries": retrieve_call_count(
                retrieval_report, "all", "opposite", "opposite_response"
            ),
            "query_denominator": int(retrieval_report["denominators"]["study_level_contrasts"]),
            "registered_relation_denominator": int(registered_all["registered_directional_relations"]),
            "registered_best_face_top1": int(registered_all["best_face_top1"]),
            "registered_best_face_top5": int(registered_all["best_face_top5"]),
            "registered_median_best_face_rank": float(registered_all["median_best_face_rank"]),
            "interpretation": (
                "Permutation significance is evidence against the implemented null, not evidence "
                "that a selected cross-study biological relation is correct."
            ),
        },
        "new_source_axis_coverage": {
            "entities": int(source_response["entity_id"].nunique()),
            "directed_candidate_pairs_involving_source": int(len(new_directed_pairs)),
            "source_entities_with_any_outgoing_candidate": int(len(new_query_entities_with_candidates)),
            "canonical_mutual_edges_all_world": int(
                (new_mutual["evidence_world"] == "all").sum()
            ),
            "canonical_mutual_edges_unresolved_world": int(
                (new_mutual["evidence_world"] == "unresolved").sum()
            ),
            "source_entities_with_any_reciprocal_all_or_unresolved_edge": int(
                len(new_entities_with_mutual)
            ),
        },
        "new_source_response_specificity": {
            "shared_entities": int(len(response_wide)),
            "beta_pearson": pearson_beta,
            "beta_spearman": spearman_beta,
            "sign_concordance": sign_concordance,
            "iga_q005_and_loso_stable_entities": int(len(iga_stable)),
            "cd8_q005_and_loso_stable_entities": int(len(cd8_stable)),
            "stable_overlap": int(len(iga_stable & cd8_stable)),
            "stable_union": int(len(iga_stable | cd8_stable)),
            "interpretation": (
                "The two contrasts share chemistry and collection context but are not known "
                "biological negatives; their different top hits are a stress test, not validity proof."
            ),
        },
        "iga_ad_candidate": {
            "candidate_rank": int(iga_ad_score["similar_rank"]),
            "candidate_query_maxT_p": float(iga_ad_score["similar_query_maxT_p"]),
            "mutual_edges": int(iga_ad_score["mutual_edges"]),
            "reported_top_contributing_edges": int(len(iga_ad)),
            "dual_normal_approx_fdr_005_edges_diagnostic": int(
                iga_ad["dual_normal_approx_fdr_005_diagnostic"].sum()
            ),
            "source_native_aromatic_amino_acid_pairs": int(len(primary_rows)),
            "independent_metabolites_claimed": 0,
            "mechanism_claimed": False,
            "required_next_evidence": [
                "common-policy library search for the new source",
                "ion-family and retention-time adjudication",
                "orthogonal identity evidence or authentic standards",
                "response replication in an independent AD or IgA-perturbation dataset",
            ],
        },
        "claim_boundary": [
            "All data and retrieval results are development/consumed.",
            "The 51/53 and 49/53 counts reject the current retrieval result as a discovery engine; they do not prove every selected pair false.",
            "No signed entity-level truth is available for the registered relations.",
            "Source-native annotations are candidate evidence and do not resolve positional/stereochemical isomers.",
            "The 188.071 feature is treated as a likely tryptophan ion-family duplicate, not an independent metabolite.",
            "No unresolved-information increment or in-house algorithm increment is claimed.",
        ],
        "provenance": {
            "implementation": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "retrieval_report": {"path": str(retrieval_report_path), "sha256": sha256(retrieval_report_path)},
            "wse_release_report": {"path": str(wse_release_report_path), "sha256": sha256(wse_release_report_path)},
            "axis_report": {"path": str(axis_report_path), "sha256": sha256(axis_report_path)},
            "response_collection": {"path": str(collection_dir / "response_entities.tsv.gz"), "sha256": sha256(collection_dir / "response_entities.tsv.gz")},
            "source_response": {"path": str(source_dir / "response_evidence.tsv.gz"), "sha256": sha256(source_dir / "response_evidence.tsv.gz")},
            "ad_source_native_library": {"path": str(ad_db_path), "sha256": sha256(ad_db_path)},
            "common_library_evidence": {"path": str(common_library_path), "sha256": sha256(common_library_path)},
        },
        "outputs_sha256": {
            top_hits_path.name: sha256(top_hits_path),
            candidate_path.name: sha256(candidate_path),
        },
    }
    report_path = out_dir / "report.json"
    atomic_json(report, report_path)
    checksums_path = out_dir / "checksums.sha256"
    checksums_path.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in [top_hits_path, candidate_path, report_path]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
