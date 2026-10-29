#!/usr/bin/env python3
"""Summarize parent-project-held-out Contrast Atlas retrievals.

This analysis removes every candidate contrast from the query's operational
parent-project dependence group before ranking and query-level max-T.  It
measures calibrated cross-project discovery yield; it does not estimate
biological accuracy because most candidate relations are unjudged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_contrast_atlas_complete_identity_wse_global import (
    CHEMPROP_STUDIES,
    IDENTITY_TRACK,
    records_without_nan,
    sha256,
)
from summarize_contrast_atlas_wse_edge_opportunity_controls import (
    SOURCE_INDEPENDENT,
    subset_summary,
)


EXPECTED_RESAMPLES = 32
EXPECTED_DIRECTED_CANDIDATES = 1944
EXPECTED_SCORE_ROWS = 17496
EXPECTED_TOP_ROWS = 918
EXPECTED_PROJECTS = 13
EXPECTED_EVALUABLE_RELATIONS = 18
EXPECTED_EXCLUDED_RELATIONS = 24


def load_project_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    required = {"study_id", "parent_project_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"project map must contain {sorted(required)}")
    if frame["study_id"].duplicated().any():
        raise ValueError("duplicate study_id in project map")
    lookup = dict(zip(frame["study_id"], frame["parent_project_id"]))
    if len(set(lookup.values())) != EXPECTED_PROJECTS:
        raise RuntimeError("operational parent-project denominator drift")
    return lookup


def two_face_calls(
    top: pd.DataFrame,
    project_lookup: dict[str, str],
    alpha: float = 0.05,
    minimum_support: int = 3,
    expected_queries: int | None = 51,
) -> pd.DataFrame:
    selected = top.loc[
        top["evidence_world"].eq("unresolved")
        & top["evidence_track"].eq(IDENTITY_TRACK)
    ].copy()
    rows: list[dict[str, object]] = []
    for (study, contrast), block in selected.groupby(
        ["query_study_id", "query_contrast_id"], sort=True
    ):
        if set(block["response_face"]) != {"similar", "opposite"} or len(block) != 2:
            raise RuntimeError(f"two response faces were not preserved for {study}/{contrast}")
        eligible = block.loc[
            pd.to_numeric(block["weighted_evidence_edges"], errors="coerce").ge(
                minimum_support
            )
            & pd.to_numeric(block["enrichment_z"], errors="coerce").gt(0)
        ].copy()
        if not len(eligible):
            result_class = "abstain_insufficient_evidence"
            chosen = block.sort_values(["response_face"]).iloc[0]
            corrected_p = 1.0
        else:
            eligible["face_p"] = pd.to_numeric(
                eligible["query_maxT_p"], errors="coerce"
            )
            chosen = eligible.sort_values(
                ["face_p", "response_face", "candidate_study_id", "candidate_contrast_id"]
            ).iloc[0]
            corrected_p = min(1.0, 2.0 * float(chosen["face_p"]))
            result_class = (
                "calibrated_cross_project_answer"
                if corrected_p <= alpha
                else "no_calibrated_match"
            )
        rows.append({
            "query_study_id": str(study),
            "query_contrast_id": str(contrast),
            "query_parent_project": project_lookup[str(study)],
            "result_class": result_class,
            "selected_face": str(chosen["response_face"]),
            "two_face_bonferroni_p": corrected_p,
            "candidate_study_id": str(chosen["candidate_study_id"]),
            "candidate_contrast_id": str(chosen["candidate_contrast_id"]),
            "candidate_parent_project": project_lookup[str(chosen["candidate_study_id"])],
            "weighted_evidence_edges": int(chosen["weighted_evidence_edges"]),
            "enrichment_z": float(chosen["enrichment_z"]),
        })
    result = pd.DataFrame(rows)
    if expected_queries is not None and len(result) != expected_queries:
        raise RuntimeError(
            f"query denominator drift: {len(result)} != {expected_queries}"
        )
    return result


def project_coverage(calls: pd.DataFrame) -> pd.DataFrame:
    frame = (
        calls.assign(
            answered=calls["result_class"].eq("calibrated_cross_project_answer"),
            abstained=calls["result_class"].eq("abstain_insufficient_evidence"),
        )
        .groupby("query_parent_project", sort=True)
        .agg(
            queries=("query_contrast_id", "size"),
            calibrated_answers=("answered", "sum"),
            insufficient_evidence=("abstained", "sum"),
        )
        .reset_index()
    )
    frame["answer_coverage"] = frame["calibrated_answers"] / frame["queries"]
    return frame


def query_stability(calls: pd.DataFrame) -> pd.DataFrame:
    """Describe sensitivity-selection stability without treating graphs as replicates."""
    required = {
        "replicate", "query_study_id", "query_contrast_id", "query_parent_project",
        "result_class", "candidate_parent_project",
    }
    if not required.issubset(calls.columns):
        raise ValueError(f"calls missing {sorted(required - set(calls.columns))}")
    rows: list[dict[str, object]] = []
    for keys, block in calls.groupby(
        ["query_study_id", "query_contrast_id", "query_parent_project"], sort=True
    ):
        study, contrast, project = keys
        if block["replicate"].nunique() != EXPECTED_RESAMPLES or len(block) != EXPECTED_RESAMPLES:
            raise RuntimeError(f"sensitivity-selection denominator drift for {study}/{contrast}")
        answered = block["result_class"].eq("calibrated_cross_project_answer")
        answered_count = int(answered.sum())
        if answered_count == EXPECTED_RESAMPLES:
            answer_stability_class = "always_answered"
        elif answered_count == 0:
            answer_stability_class = "never_answered"
        else:
            answer_stability_class = "variable_answer"
        project_counts = block["candidate_parent_project"].value_counts()
        rows.append({
            "query_study_id": str(study),
            "query_contrast_id": str(contrast),
            "query_parent_project": str(project),
            "answered_selections": answered_count,
            "answer_fraction": answered_count / EXPECTED_RESAMPLES,
            "answer_stability_class": answer_stability_class,
            "selected_candidate_projects": int(block["candidate_parent_project"].nunique()),
            "modal_candidate_project": str(project_counts.index[0]),
            "modal_candidate_project_selections": int(project_counts.iloc[0]),
            "modal_candidate_project_fraction": float(project_counts.iloc[0] / EXPECTED_RESAMPLES),
            "calibrated_candidate_projects": int(
                block.loc[answered, "candidate_parent_project"].nunique()
            ),
        })
    return pd.DataFrame(rows)


def summarize_project_coverage(project_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for project, block in project_frame.groupby("query_parent_project", sort=True):
        queries = pd.to_numeric(block["queries"], errors="raise")
        if queries.nunique() != 1 or len(block) != EXPECTED_RESAMPLES:
            raise RuntimeError(f"project coverage denominator drift for {project}")
        rows.append({
            "query_parent_project": str(project),
            "queries": int(queries.iloc[0]),
            "minimum_answer_coverage": float(block["answer_coverage"].min()),
            "median_answer_coverage": float(block["answer_coverage"].median()),
            "maximum_answer_coverage": float(block["answer_coverage"].max()),
            "median_calibrated_answers": float(block["calibrated_answers"].median()),
        })
    return pd.DataFrame(rows)


def summarize_candidate_project_hubs(hub_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (face, project), block in hub_frame.groupby(
        ["response_face", "candidate_parent_project"], sort=True
    ):
        if len(block) != EXPECTED_RESAMPLES:
            raise RuntimeError(f"candidate-project hub denominator drift for {face}/{project}")
        rows.append({
            "response_face": str(face),
            "candidate_parent_project": str(project),
            "minimum_top_hits": int(block["top_hits"].min()),
            "median_top_hits": float(block["top_hits"].median()),
            "maximum_top_hits": int(block["top_hits"].max()),
            "median_candidate_exposure_fraction": float(
                block["candidate_exposure_fraction"].median()
            ),
            "median_top_hit_fraction": float(block["top_hit_fraction"].median()),
            "median_top_hit_to_exposure_ratio": float(
                block["top_hit_to_exposure_ratio"].median()
            ),
        })
    return pd.DataFrame(rows)


def face_and_hub_summary(
    top: pd.DataFrame,
    scores: pd.DataFrame,
    project_lookup: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_top = top.loc[
        top["evidence_world"].eq("unresolved")
        & top["evidence_track"].eq(IDENTITY_TRACK)
    ].copy()
    selected_scores = scores.loc[
        scores["evidence_world"].eq("unresolved")
        & scores["evidence_track"].eq(IDENTITY_TRACK)
    ].copy()
    selected_top["candidate_parent_project"] = selected_top["candidate_study_id"].map(
        project_lookup
    )
    selected_scores["candidate_parent_project"] = selected_scores[
        "candidate_study_id"
    ].map(project_lookup)
    project_rows: list[dict[str, object]] = []
    face_rows: list[dict[str, object]] = []
    exposure = selected_scores["candidate_parent_project"].value_counts()
    total_exposure = int(exposure.sum())
    for face, block in selected_top.groupby("response_face", sort=True):
        hits = block["candidate_parent_project"].value_counts()
        for project in sorted(set(exposure.index).union(hits.index)):
            hit_count = int(hits.get(project, 0))
            exposure_count = int(exposure.get(project, 0))
            hit_fraction = hit_count / len(block)
            exposure_fraction = exposure_count / total_exposure
            project_rows.append({
                "response_face": str(face),
                "candidate_parent_project": str(project),
                "top_hits": hit_count,
                "candidate_exposure_rows": exposure_count,
                "top_hit_fraction": hit_fraction,
                "candidate_exposure_fraction": exposure_fraction,
                "top_hit_to_exposure_ratio": (
                    np.nan if exposure_fraction == 0 else hit_fraction / exposure_fraction
                ),
            })
        candidate_chemprop = block["candidate_study_id"].isin(CHEMPROP_STUDIES)
        query_chemprop = block["query_study_id"].isin(CHEMPROP_STUDIES)
        counts = hits.to_numpy(float)
        face_rows.append({
            "response_face": str(face),
            "face_specific_calibrated_calls": int(
                block["result_class"].eq(f"{face}_response").sum()
            ),
            "no_calibrated_match": int(
                block["result_class"].eq("no_calibrated_match").sum()
            ),
            "insufficient_evidence": int(
                block["result_class"].eq("abstain_insufficient_evidence").sum()
            ),
            "candidate_projects_hit": int(len(hits)),
            "largest_candidate_project_hits": int(hits.max()),
            "candidate_project_hhi": float(np.square(counts / counts.sum()).sum()),
            "chemprop_candidate_hits": int(candidate_chemprop.sum()),
            "nonchemprop_to_chemprop_hits": int((~query_chemprop & candidate_chemprop).sum()),
            "chemprop_to_chemprop_hits": int((query_chemprop & candidate_chemprop).sum()),
        })
    return pd.DataFrame(face_rows), pd.DataFrame(project_rows)


def numeric_distribution(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "n": int(len(numeric)),
        "minimum": float(numeric.min()),
        "q25": float(numeric.quantile(0.25)),
        "median": float(numeric.median()),
        "q75": float(numeric.quantile(0.75)),
        "maximum": float(numeric.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-root", required=True, type=Path)
    parser.add_argument("--source-disjoint-root", required=True, type=Path)
    parser.add_argument("--project-map", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--doc", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.doc.exists():
        raise FileExistsError(args.doc)

    project_lookup = load_project_map(args.project_map)
    per_resample: list[dict[str, object]] = []
    project_rows: list[pd.DataFrame] = []
    face_rows: list[pd.DataFrame] = []
    hub_rows: list[pd.DataFrame] = []
    primary_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    combined_calls: list[pd.DataFrame] = []

    for replicate in range(EXPECTED_RESAMPLES):
        directory = args.retrieval_root / f"resample_{replicate:02d}"
        source_directory = args.source_disjoint_root / f"resample_{replicate:02d}"
        paths = {
            "report": directory / "report.json",
            "top": directory / "open_local_edge_top_hits.tsv",
            "scores": directory / "open_local_edge_candidate_scores.tsv.gz",
            "registered": directory / "registered_relation_ranks.tsv",
            "source_top": source_directory / "open_local_edge_top_hits.tsv",
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        denominator = report["denominators"]
        expected = {
            "parent_project_families": EXPECTED_PROJECTS,
            "directed_query_candidate_pairs": EXPECTED_DIRECTED_CANDIDATES,
            "candidate_score_rows": EXPECTED_SCORE_ROWS,
            "top_hit_rows": EXPECTED_TOP_ROWS,
            "registered_directional_relations_in_candidate_universe": EXPECTED_EVALUABLE_RELATIONS,
            "registered_directional_relations_not_evaluable_same_parent_project": EXPECTED_EXCLUDED_RELATIONS,
        }
        for key, value in expected.items():
            if int(denominator[key]) != value:
                raise RuntimeError(f"{key} drift in resample {replicate}: {denominator[key]}")
        if report["method"]["candidate_universe"] != "parent_project_disjoint":
            raise RuntimeError("candidate-universe contract drift")

        top = pd.read_csv(paths["top"], sep="\t", low_memory=False)
        scores = pd.read_csv(paths["scores"], sep="\t", low_memory=False)
        registered = pd.read_csv(paths["registered"], sep="\t", low_memory=False)
        source_top = pd.read_csv(paths["source_top"], sep="\t", low_memory=False)
        if (
            scores["query_parent_project"].eq(scores["candidate_parent_project"]).any()
        ):
            raise RuntimeError("same-parent candidate survived holdout")
        statuses = registered[[
            "relation_direction", "left_study_id", "left_contrast_id",
            "right_study_id", "right_contrast_id", "evaluation_status",
        ]].drop_duplicates()
        if int(statuses["evaluation_status"].eq("evaluable").sum()) != 18:
            raise RuntimeError("evaluable registered denominator drift")
        if int(statuses["evaluation_status"].eq("not_evaluable_same_parent_project").sum()) != 24:
            raise RuntimeError("same-parent exclusion denominator drift")

        calls = two_face_calls(top, project_lookup)
        calls.insert(0, "replicate", replicate)
        combined_calls.append(calls)
        coverage = project_coverage(calls)
        coverage.insert(0, "replicate", replicate)
        project_rows.append(coverage)
        face, hubs = face_and_hub_summary(top, scores, project_lookup)
        face.insert(0, "replicate", replicate)
        hubs.insert(0, "replicate", replicate)
        face_rows.append(face)
        hub_rows.append(hubs)

        source_calls = two_face_calls(source_top, project_lookup)
        nonchem = ~calls["query_study_id"].isin(CHEMPROP_STUDIES)
        compare_columns = [
            "result_class", "selected_face", "two_face_bonferroni_p",
            "candidate_study_id", "candidate_contrast_id",
        ]
        if not calls.loc[nonchem, compare_columns].reset_index(drop=True).equals(
            source_calls.loc[nonchem, compare_columns].reset_index(drop=True)
        ):
            raise RuntimeError("non-ChemProp queries changed under parent-project holdout")
        answered = calls["result_class"].eq("calibrated_cross_project_answer")
        source_answered = source_calls["result_class"].eq(
            "calibrated_cross_project_answer"
        )
        per_resample.append({
            "replicate": replicate,
            "queries": 51,
            "calibrated_answers": int(answered.sum()),
            "answer_coverage": float(answered.mean()),
            "project_macro_answer_coverage": float(coverage["answer_coverage"].mean()),
            "insufficient_evidence": int(
                calls["result_class"].eq("abstain_insufficient_evidence").sum()
            ),
            "no_calibrated_match": int(
                calls["result_class"].eq("no_calibrated_match").sum()
            ),
            "chemprop_query_answers": int(
                (answered & calls["query_study_id"].isin(CHEMPROP_STUDIES)).sum()
            ),
            "nonchemprop_query_answers": int(
                (answered & ~calls["query_study_id"].isin(CHEMPROP_STUDIES)).sum()
            ),
            "source_disjoint_calibrated_answers": int(source_answered.sum()),
            "answers_lost_after_parent_holdout": int(
                (source_answered & ~answered).sum()
            ),
            "answers_gained_after_parent_holdout": int(
                (~source_answered & answered).sum()
            ),
        })
        primary = subset_summary(registered, SOURCE_INDEPENDENT, "unresolved")
        primary.update({"replicate": replicate})
        primary_rows.append(primary)
        provenance_rows.append({
            "replicate": replicate,
            "report_sha256": sha256(paths["report"]),
            "top_sha256": sha256(paths["top"]),
            "scores_sha256": sha256(paths["scores"]),
            "registered_sha256": sha256(paths["registered"]),
        })

    per_resample_frame = pd.DataFrame(per_resample)
    project_frame = pd.concat(project_rows, ignore_index=True)
    face_frame = pd.concat(face_rows, ignore_index=True)
    hub_frame = pd.concat(hub_rows, ignore_index=True)
    primary_frame = pd.DataFrame(primary_rows)
    calls_frame = pd.concat(combined_calls, ignore_index=True)
    provenance_frame = pd.DataFrame(provenance_rows)
    query_stability_frame = query_stability(calls_frame)
    project_coverage_summary_frame = summarize_project_coverage(project_frame)
    candidate_project_hub_summary_frame = summarize_candidate_project_hubs(hub_frame)

    args.output_dir.mkdir(parents=True)
    outputs = {
        "resamples": args.output_dir / "per_resample_summary.tsv",
        "project_coverage": args.output_dir / "per_resample_project_coverage.tsv",
        "faces": args.output_dir / "per_resample_face_summary.tsv",
        "hubs": args.output_dir / "per_resample_candidate_project_hubs.tsv",
        "primary": args.output_dir / "per_resample_primary_relations.tsv",
        "calls": args.output_dir / "combined_query_calls.tsv.gz",
        "provenance": args.output_dir / "retrieval_provenance.tsv",
        "query_stability": args.output_dir / "query_stability.tsv",
        "project_coverage_summary": args.output_dir / "project_coverage_summary.tsv",
        "candidate_project_hub_summary": args.output_dir / "candidate_project_hub_summary.tsv",
    }
    per_resample_frame.to_csv(outputs["resamples"], sep="\t", index=False)
    project_frame.to_csv(outputs["project_coverage"], sep="\t", index=False)
    face_frame.to_csv(outputs["faces"], sep="\t", index=False)
    hub_frame.to_csv(outputs["hubs"], sep="\t", index=False)
    primary_frame.to_csv(outputs["primary"], sep="\t", index=False)
    calls_frame.to_csv(
        outputs["calls"], sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    provenance_frame.to_csv(outputs["provenance"], sep="\t", index=False)
    query_stability_frame.to_csv(outputs["query_stability"], sep="\t", index=False)
    project_coverage_summary_frame.to_csv(
        outputs["project_coverage_summary"], sep="\t", index=False
    )
    candidate_project_hub_summary_frame.to_csv(
        outputs["candidate_project_hub_summary"], sep="\t", index=False
    )

    face_distributions = []
    for face in ("similar", "opposite"):
        block = face_frame.loc[face_frame["response_face"].eq(face)]
        face_distributions.append({
            "response_face": face,
            "calibrated_calls": numeric_distribution(
                block["face_specific_calibrated_calls"]
            ),
            "chemprop_candidate_hits": numeric_distribution(
                block["chemprop_candidate_hits"]
            ),
            "largest_candidate_project_hits": numeric_distribution(
                block["largest_candidate_project_hits"]
            ),
            "candidate_project_hhi": numeric_distribution(block["candidate_project_hhi"]),
        })
    stability_counts = query_stability_frame["answer_stability_class"].value_counts()
    stable_candidate_project_queries = int(
        query_stability_frame["selected_candidate_projects"].eq(1).sum()
    )
    dominant_hubs: list[dict[str, object]] = []
    overrepresented_hubs: list[dict[str, object]] = []
    for face in ("similar", "opposite"):
        block = candidate_project_hub_summary_frame.loc[
            candidate_project_hub_summary_frame["response_face"].eq(face)
        ]
        dominant_hubs.append(block.sort_values(
            ["median_top_hits", "median_top_hit_to_exposure_ratio", "candidate_parent_project"],
            ascending=[False, False, True],
        ).iloc[0].to_dict())
        supported = block.loc[block["median_top_hits"].ge(3)].sort_values(
            ["median_top_hit_to_exposure_ratio", "median_top_hits", "candidate_parent_project"],
            ascending=[False, False, True],
        )
        overrepresented_hubs.append(supported.iloc[0].to_dict())
    report = {
        "schema": "contrast_atlas_parent_project_holdout_summary_v2",
        "status": "DEVELOPMENT_CONSUMED_PARENT_PROJECT_HOLDOUT_SUMMARIZED",
        "data_role": "development_consumed",
        "scale": {
            "sources": 15,
            "operational_parent_project_groups": 13,
            "study_level_contrasts": 51,
            "directed_candidate_contrasts": EXPECTED_DIRECTED_CANDIDATES,
            "candidate_score_rows_per_resample": EXPECTED_SCORE_ROWS,
            "selection_resamples": EXPECTED_RESAMPLES,
            "response_permutations_per_resample": 256,
            "registered_directions_evaluable": EXPECTED_EVALUABLE_RELATIONS,
            "registered_directions_not_evaluable_same_parent": EXPECTED_EXCLUDED_RELATIONS,
        },
        "calibrated_cross_project_discovery_yield": {
            "calibrated_answers": numeric_distribution(
                per_resample_frame["calibrated_answers"]
            ),
            "answer_coverage": numeric_distribution(per_resample_frame["answer_coverage"]),
            "project_macro_answer_coverage": numeric_distribution(
                per_resample_frame["project_macro_answer_coverage"]
            ),
            "chemprop_query_answers": numeric_distribution(
                per_resample_frame["chemprop_query_answers"]
            ),
            "nonchemprop_query_answers": numeric_distribution(
                per_resample_frame["nonchemprop_query_answers"]
            ),
            "source_disjoint_answers": numeric_distribution(
                per_resample_frame["source_disjoint_calibrated_answers"]
            ),
            "answers_lost_after_parent_holdout": numeric_distribution(
                per_resample_frame["answers_lost_after_parent_holdout"]
            ),
            "answers_gained_after_parent_holdout": numeric_distribution(
                per_resample_frame["answers_gained_after_parent_holdout"]
            ),
        },
        "face_specific_and_hub_distributions": face_distributions,
        "sensitivity_selection_stability": {
            "always_answered_queries": int(stability_counts.get("always_answered", 0)),
            "never_answered_queries": int(stability_counts.get("never_answered", 0)),
            "variable_answer_queries": int(stability_counts.get("variable_answer", 0)),
            "queries_selecting_one_candidate_project_in_all_32_graphs": stable_candidate_project_queries,
            "selected_candidate_projects_per_query": numeric_distribution(
                query_stability_frame["selected_candidate_projects"]
            ),
            "dominant_candidate_project_by_face": records_without_nan(
                pd.DataFrame(dominant_hubs)
            ),
            "most_overrepresented_candidate_project_by_face": records_without_nan(
                pd.DataFrame(overrepresented_hubs)
            ),
        },
        "source_independent_primary": {
            "directions": 10,
            "canonical_pairs": 5,
            "biological_components": 2,
            "directional_top5": numeric_distribution(primary_frame["directional_top5"]),
            "directional_two_face_bonferroni_005": numeric_distribution(
                primary_frame["directional_best_face_bonferroni_005"]
            ),
        },
        "decision": {
            "parent_project_holdout_executed": True,
            "cross_project_biological_retrieval_pass": False,
            "unknown_information_increment_pass": False,
            "in_house_algorithm_increment_evaluated": False,
            "reasons": [
                "The 24 exact-drug ChemProp directions are intentionally excluded, not counted as failures.",
                "Only 10 primary registered directions remain, representing five pairs and two biological components.",
                "The remaining 1,926 directed candidates are unjudged rather than negative.",
                "All registered response faces are unspecified; calibrated calls are discovery yield, not signed accuracy.",
                "All inputs are development/consumed and no sealed external biological validation is present.",
            ],
        },
        "claim_boundary": [
            "The 13 groups are operational dependence groups; independence beyond the known shared ChemProp parent was not proven.",
            "Two-face calls use min(1, 2*min(face-wise query max-T p)) and require at least three weighted evidence edges.",
            "No-calibrated-match and insufficient-evidence are not biological no-match labels.",
            "The 32 graphs are highly overlapping matching-order sensitivity selections, not independent replicates or confidence intervals.",
            "This run evaluates the same-precursor identity-hypothesis track only; it does not validate different-mass chemical relations.",
        ],
        "provenance": {
            "project_map_sha256": sha256(args.project_map),
            **{name: sha256(path) for name, path in outputs.items()},
            "implementation_sha256": sha256(Path(__file__)),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    yield_record = report["calibrated_cross_project_discovery_yield"]
    primary = report["source_independent_primary"]
    stability = report["sensitivity_selection_stability"]
    dominant_by_face = {
        row["response_face"]: row for row in stability["dominant_candidate_project_by_face"]
    }
    overrepresented_by_face = {
        row["response_face"]: row
        for row in stability["most_overrepresented_candidate_project_by_face"]
    }
    document = f"""# Parent-project-held-out system-wide Contrast Atlas result

## Scope

This is a full 15-source, 13 operational-dependence-group, 51-contrast analysis. Every query excludes all contrasts from its own parent-project group before ranking and query-level max-T. The candidate universe therefore contains {EXPECTED_DIRECTED_CANDIDATES:,} directed contrasts rather than 2,286; all 342 removed directions are between the three ChemProp accessions.

The 32 graphs are matching-order sensitivity selections and are highly overlapping. Their ranges are not confidence intervals. All inputs are development/consumed.

## Calibrated cross-project discovery yield

- Two-face-Bonferroni calibrated answers among 51 queries: **{int(yield_record['calibrated_answers']['minimum'])}-{int(yield_record['calibrated_answers']['maximum'])}**, median **{yield_record['calibrated_answers']['median']:.1f}**.
- Project-macro answer coverage across 13 operational groups: **{yield_record['project_macro_answer_coverage']['minimum']:.3f}-{yield_record['project_macro_answer_coverage']['maximum']:.3f}**, median **{yield_record['project_macro_answer_coverage']['median']:.3f}**.
- Answers lost after removing same-parent candidates: **{int(yield_record['answers_lost_after_parent_holdout']['minimum'])}-{int(yield_record['answers_lost_after_parent_holdout']['maximum'])}**, median **{yield_record['answers_lost_after_parent_holdout']['median']:.1f}**.
- Answers gained after the smaller candidate universe was recalibrated: **{int(yield_record['answers_gained_after_parent_holdout']['minimum'])}-{int(yield_record['answers_gained_after_parent_holdout']['maximum'])}**, median **{yield_record['answers_gained_after_parent_holdout']['median']:.1f}**. These are not evidence of correctness.

## Stability and project-hub substitution

- **{stability['always_answered_queries']}/51** queries were answered in all 32 overlapping sensitivity graphs, **{stability['never_answered_queries']}/51** were never answered and **{stability['variable_answer_queries']}/51** varied.
- **{stability['queries_selecting_one_candidate_project_in_all_32_graphs']}/51** queries selected the same candidate parent project in every graph; the median number of selected candidate projects per query was **{stability['selected_candidate_projects_per_query']['median']:.1f}**.
- The dominant similar-response candidate project was **{dominant_by_face['similar']['candidate_parent_project']}** (median **{dominant_by_face['similar']['median_top_hits']:.1f}/51** top hits; exposure-adjusted ratio **{dominant_by_face['similar']['median_top_hit_to_exposure_ratio']:.2f}x**).
- The dominant opposite-response candidate project was **{dominant_by_face['opposite']['candidate_parent_project']}** (median **{dominant_by_face['opposite']['median_top_hits']:.1f}/51** top hits; exposure-adjusted ratio **{dominant_by_face['opposite']['median_top_hit_to_exposure_ratio']:.2f}x**).
- After candidate exposure was considered, **{overrepresented_by_face['similar']['candidate_parent_project']}** was the strongest similar-response project hub (median **{overrepresented_by_face['similar']['median_top_hits']:.1f}** hits; **{overrepresented_by_face['similar']['median_top_hit_to_exposure_ratio']:.2f}x** exposure-adjusted), and **{overrepresented_by_face['opposite']['candidate_parent_project']}** was strongest for opposite responses (median **{overrepresented_by_face['opposite']['median_top_hits']:.1f}**; **{overrepresented_by_face['opposite']['median_top_hit_to_exposure_ratio']:.2f}x**).

These stable selections are not truth. They show that removing the shared ChemProp parent project does not produce broad biological specificity; the retrieval can substitute another project-level response hub.

## Registered source-independent relations

The 24 ChemProp exact-drug directions are **not evaluable** by design and are not failures. The remaining primary set contains only 10 directions, five canonical pairs and two biological components (HFD and IBD).

- Unsigned best-face Top-5: **{int(primary['directional_top5']['minimum'])}-{int(primary['directional_top5']['maximum'])}**, median **{primary['directional_top5']['median']:.1f}**.
- Two-face-Bonferroni query-level max-T calls: **{int(primary['directional_two_face_bonferroni_005']['minimum'])}-{int(primary['directional_two_face_bonferroni_005']['maximum'])}**, median **{primary['directional_two_face_bonferroni_005']['median']:.1f}**.

## Decision

The parent-project holdout is complete, but it does not establish biological accuracy: 1,926/1,944 candidate directions are unjudged, every registered response face is unspecified, and no sealed external biology panel exists. The valid endpoint is calibrated cross-project discovery yield, not accuracy, FDR, specificity or mechanism recovery.

This system-level result remains a gate for the same-axis Official DreaMS, Noise V1 and Full23 comparison; those methods have not yet been evaluated here.
"""
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(document, encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
