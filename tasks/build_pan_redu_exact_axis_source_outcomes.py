#!/usr/bin/env python3
"""Build formal exact-axis source-risk outcomes from the Pan-ReDU v4 matrix.

This is deliberately a fail-closed bridge.  A query enters the formal source
gate only through a score-blind, pre-score closed-gold candidate universe of
exact replications and explicit mismatches.  Unregistered matrix relations are
excluded as unadjudicated (never converted to negatives), while unknown rows
inside the formal registry are inadmissible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from contrast_response_transport import parse_bool


WORLDS = ("all", "known_only", "measurement_matched_known")
TRUTH_CLASSES = {"exact_replication", "explicit_mismatch", "unknown"}
ROLES = {"development", "external"}
MATRIX_SCHEMA = "pan_redu_fixed_capacity_response_transport_matrix_v4"
MATRIX_STATUS = "DEVELOPMENT_PRETRUTH_RESPONSE_TRANSPORT_MATRIX_COMPLETE"
RANKING_METHOD_COLUMNS = {
    "transport_conservative": "signed_chemical_evidence_conservative_bound",
    "greedy_same_graph": "greedy_signed_chemical_evidence",
    "centroid_same_encoder": "centroid_cosine",
}
PAIR_KEY = (
    "query_panel_id", "query_contrast_id",
    "target_panel_id", "target_contrast_id",
)
MATRIX_PAIR_KEY = (
    "left_panel_id", "left_contrast_id",
    "right_panel_id", "right_contrast_id",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_bool(series: pd.Series, label: str) -> pd.Series:
    try:
        return parse_bool(series, label)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_matrix_report(report: dict[str, object], matrix_path: Path) -> None:
    if report.get("schema") != MATRIX_SCHEMA:
        raise RuntimeError(f"matrix report is not {MATRIX_SCHEMA}")
    if report.get("status") != MATRIX_STATUS:
        raise RuntimeError("matrix report is not the complete pre-truth v4 artifact")
    expected = report.get("provenance", {}).get("transport_matrix_sha256")  # type: ignore[union-attr]
    if not expected or expected != sha256(matrix_path):
        raise RuntimeError("v4 transport matrix checksum disagrees with its report")


def validate_truth_registry(raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    required = {
        *PAIR_KEY,
        "query_source", "query_source_family", "target_source",
        "target_source_family", "exact_axis_family_id", "axis", "truth_class",
        "target_exact_axis_family_id", "target_axis", "expected_direction",
        "eligible_exact_axis", "evaluation_role", "truth_frozen_before_scores",
        "selection_blind_to_transport_scores",
        "candidate_universe_frozen_before_scores", "registry_complete_for_query",
    }
    if missing := required - set(raw):
        raise RuntimeError(f"formal truth registry lacks columns: {sorted(missing)}")
    truth = raw.copy()
    text_columns = required - {
        "eligible_exact_axis", "truth_frozen_before_scores",
        "selection_blind_to_transport_scores",
        "candidate_universe_frozen_before_scores", "registry_complete_for_query",
    }
    for column in text_columns:
        truth[column] = truth[column].astype(str).str.strip()
        if truth[column].eq("").any():
            raise RuntimeError(f"formal truth registry has empty {column}")
    for column in (
        "eligible_exact_axis", "truth_frozen_before_scores",
        "selection_blind_to_transport_scores",
        "candidate_universe_frozen_before_scores", "registry_complete_for_query",
    ):
        truth[column] = _strict_bool(truth[column], column)
    if truth.empty:
        raise RuntimeError("formal truth registry is empty")
    if truth.duplicated(list(PAIR_KEY)).any():
        raise RuntimeError("formal truth registry contains duplicate ordered pairs")
    if not set(truth.truth_class).issubset(TRUTH_CLASSES):
        raise RuntimeError("formal truth registry contains unsupported truth_class")
    if truth.truth_class.eq("unknown").any():
        raise RuntimeError(
            "unknown/unreviewed truth cannot enter the formal source gate"
        )
    if not truth.eligible_exact_axis.all():
        raise RuntimeError("every formal candidate must be eligible for the exact axis")
    frozen_columns = (
        "truth_frozen_before_scores", "selection_blind_to_transport_scores",
        "candidate_universe_frozen_before_scores", "registry_complete_for_query",
    )
    if not all(bool(truth[column].all()) for column in frozen_columns):
        raise RuntimeError(
            "truth and complete candidate membership must be score blind and frozen before scores"
        )
    roles = set(truth.evaluation_role)
    if len(roles) != 1 or not roles.issubset(ROLES):
        raise RuntimeError("evaluation_role must be one invariant development or external value")
    role = next(iter(roles))

    exact = truth.truth_class.eq("exact_replication")
    mismatch = truth.truth_class.eq("explicit_mismatch")
    if not truth.loc[exact, "expected_direction"].isin({"similar", "opposite"}).all():
        raise RuntimeError("exact replications require a similar or opposite expected direction")
    if not truth.loc[mismatch, "expected_direction"].eq("not_applicable").all():
        raise RuntimeError("explicit mismatches require expected_direction=not_applicable")
    if not (
        truth.loc[exact, "target_exact_axis_family_id"].eq(
            truth.loc[exact, "exact_axis_family_id"]
        )
        & truth.loc[exact, "target_axis"].eq(truth.loc[exact, "axis"])
    ).all():
        raise RuntimeError(
            "exact replication must independently declare the same target axis and family"
        )
    mismatch_same_axis = (
        truth.loc[mismatch, "target_exact_axis_family_id"].eq(
            truth.loc[mismatch, "exact_axis_family_id"]
        )
        & truth.loc[mismatch, "target_axis"].eq(truth.loc[mismatch, "axis"])
    )
    if mismatch_same_axis.any():
        raise RuntimeError("explicit mismatch cannot independently declare the same axis")
    if truth.query_source.eq(truth.target_source).any():
        raise RuntimeError("formal candidate has query/target source overlap")
    if truth.query_source_family.eq(truth.target_source_family).any():
        raise RuntimeError("formal candidate has query/target source-family overlap")
    # The primary source sign-flip is conditional on a frozen reference atlas.
    # A source that also appears as a query would couple query deltas through a
    # shared random node, so reciprocal retrieval belongs in a separate
    # development audit rather than this formal inference table.
    if overlap := set(truth.query_source) & set(truth.target_source):
        raise RuntimeError(
            "global query/reference source-role overlap is inadmissible for formal inference: "
            f"{sorted(overlap)[:3]}"
        )
    if overlap := set(truth.query_source_family) & set(truth.target_source_family):
        raise RuntimeError(
            "global query/reference source-family-role overlap is inadmissible for formal inference: "
            f"{sorted(overlap)[:3]}"
        )

    query_key = ["query_panel_id", "query_contrast_id"]
    invariant = [
        "query_source", "query_source_family", "exact_axis_family_id", "axis",
        "evaluation_role", "registry_complete_for_query",
    ]
    for column in invariant:
        if (truth.groupby(query_key)[column].nunique() != 1).any():
            raise RuntimeError(f"{column} changes within a formal query")
    if truth.groupby(["query_source", "axis"])[query_key].apply(
        lambda frame: len(frame.drop_duplicates())
    ).gt(1).any():
        raise RuntimeError("one query_source/axis maps to multiple formal queries")
    if (truth.groupby("query_source").exact_axis_family_id.nunique() != 1).any():
        raise RuntimeError("one query_source maps to multiple exact_axis_family_id values")
    if (truth.groupby("query_source_family").query_source.nunique() != 1).any():
        raise RuntimeError(
            "one query provenance family maps to multiple nominal query sources"
        )
    for _, group in truth.groupby(query_key, sort=False):
        counts = group.truth_class.value_counts()
        if int(counts.get("exact_replication", 0)) < 1:
            raise RuntimeError("each formal query requires at least one exact replication")
        if int(counts.get("explicit_mismatch", 0)) < 1:
            raise RuntimeError("each formal query requires at least one explicit mismatch")
        directions = set(group.loc[exact.loc[group.index], "expected_direction"])
        if len(directions) != 1:
            raise RuntimeError("exact replications within one query mix expected directions")

    # The same panel/contrast cannot silently acquire a different provenance
    # when it appears in another query's candidate universe.
    for keys, source_column, family_column in (
        (["query_panel_id", "query_contrast_id"], "query_source", "query_source_family"),
        (["target_panel_id", "target_contrast_id"], "target_source", "target_source_family"),
    ):
        if (truth.groupby(keys)[source_column].nunique() != 1).any():
            raise RuntimeError(f"{source_column} changes for one panel/contrast")
        if (truth.groupby(keys)[family_column].nunique() != 1).any():
            raise RuntimeError(f"{family_column} changes for one panel/contrast")
    target_keys = ["target_panel_id", "target_contrast_id"]
    for column in ("target_exact_axis_family_id", "target_axis"):
        if (truth.groupby(target_keys)[column].nunique() != 1).any():
            raise RuntimeError(f"{column} changes for one target panel/contrast")
    if (truth.groupby("target_source_family").target_source.nunique() != 1).any():
        raise RuntimeError(
            "one target provenance family maps to multiple nominal target sources"
        )
    target_view_counts = truth.groupby([
        *query_key, "target_source_family", "target_axis",
    ])[target_keys].apply(lambda frame: len(frame.drop_duplicates()))
    if target_view_counts.gt(1).any():
        raise RuntimeError(
            "one formal query contains duplicate technical views of a target source-axis"
        )
    return truth.sort_values(list(PAIR_KEY), kind="stable").reset_index(drop=True), role


def validate_matrix(
    raw: pd.DataFrame,
    ranking_method: str = "transport_conservative",
) -> pd.DataFrame:
    if ranking_method not in RANKING_METHOD_COLUMNS:
        raise ValueError(f"unsupported ranking_method: {ranking_method}")
    score_column = RANKING_METHOD_COLUMNS[ranking_method]
    required = {
        "world", *MATRIX_PAIR_KEY, "answerable", "direction_identified",
        score_column,
    }
    if missing := required - set(raw):
        raise RuntimeError(f"v4 transport matrix lacks columns: {sorted(missing)}")
    matrix = raw.copy()
    for column in ("world", *MATRIX_PAIR_KEY):
        matrix[column] = matrix[column].astype(str).str.strip()
        if matrix[column].eq("").any():
            raise RuntimeError(f"v4 transport matrix has empty {column}")
    matrix = matrix.loc[matrix.world.isin(WORLDS)].copy()
    if set(matrix.world) != set(WORLDS):
        raise RuntimeError(f"v4 matrix must contain all formal worlds: {list(WORLDS)}")
    if matrix.duplicated(["world", *MATRIX_PAIR_KEY]).any():
        raise RuntimeError("v4 matrix contains duplicate world/query/target rows")
    matrix["answerable"] = _strict_bool(matrix.answerable, "answerable")
    matrix["direction_identified"] = _strict_bool(
        matrix.direction_identified, "direction_identified",
    )
    if (matrix.answerable & ~matrix.direction_identified).any():
        raise RuntimeError("answerable matrix row lacks direction identification")
    matrix[score_column] = pd.to_numeric(matrix[score_column], errors="coerce")
    finite = np.isfinite(matrix[score_column])
    if (matrix.answerable & ~finite).any():
        raise RuntimeError(
            f"answerable matrix row lacks a finite {ranking_method} score"
        )
    if ranking_method == "centroid_same_encoder":
        if "centroid_scope_supported" not in matrix:
            raise RuntimeError("centroid matrix lacks centroid_scope_supported")
        matrix["centroid_scope_supported"] = _strict_bool(
            matrix.centroid_scope_supported, "centroid_scope_supported",
        )
        if not matrix.centroid_scope_supported.all():
            raise RuntimeError(
                "centroid ranking is inadmissible for a world without global-scope support"
            )
    return matrix.reset_index(drop=True)


def build_exact_axis_source_outcomes(
    matrix_raw: pd.DataFrame,
    truth_raw: pd.DataFrame,
    ranking_method: str = "transport_conservative",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    matrix = validate_matrix(matrix_raw, ranking_method=ranking_method)
    score_column = RANKING_METHOD_COLUMNS[ranking_method]
    truth, role = validate_truth_registry(truth_raw)
    reference_atlas_hash = _canonical_hash([
        {
            "target_panel_id": str(row.target_panel_id),
            "target_contrast_id": str(row.target_contrast_id),
            "target_source": str(row.target_source),
            "target_source_family": str(row.target_source_family),
            "target_exact_axis_family_id": str(row.target_exact_axis_family_id),
            "target_axis": str(row.target_axis),
            "truth_class": str(row.truth_class),
            "expected_direction": str(row.expected_direction),
        }
        for row in truth.sort_values(list(PAIR_KEY), kind="stable").itertuples(index=False)
    ])
    reference_atlas_id = f"formal_reference_sha256:{reference_atlas_hash}"
    query_columns = ["query_panel_id", "query_contrast_id"]
    matrix_query_columns = ["left_panel_id", "left_contrast_id"]
    rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    raw_pairs_for_formal_queries: set[tuple[str, str, str, str]] = set()
    excluded_unregistered_pairs: set[tuple[str, str, str, str]] = set()

    for query_key, query_truth in truth.groupby(query_columns, sort=True):
        query_panel, query_contrast = (str(query_key[0]), str(query_key[1]))
        query_matrix = matrix.loc[
            matrix.left_panel_id.eq(query_panel)
            & matrix.left_contrast_id.eq(query_contrast)
        ].copy()
        if query_matrix.empty:
            raise RuntimeError(f"formal query absent from v4 matrix: {query_key}")

        world_sets: dict[str, set[tuple[str, str]]] = {}
        for world in WORLDS:
            world_rows = query_matrix.loc[query_matrix.world.eq(world)]
            world_sets[world] = set(zip(
                world_rows.right_panel_id, world_rows.right_contrast_id,
            ))
        registered = set(zip(
            query_truth.target_panel_id, query_truth.target_contrast_id,
        ))
        for world in WORLDS:
            if missing_matrix := registered - world_sets[world]:
                raise RuntimeError(
                    f"registered formal candidate is absent from world={world} for "
                    f"query {query_key}: {sorted(missing_matrix)[:3]}"
                )
        raw_observed = set().union(*world_sets.values())
        raw_pairs_for_formal_queries.update(
            (query_panel, query_contrast, panel, contrast)
            for panel, contrast in raw_observed
        )
        excluded_unregistered_pairs.update(
            (query_panel, query_contrast, panel, contrast)
            for panel, contrast in raw_observed - registered
        )

        query_source = str(query_truth.query_source.iloc[0])
        query_provenance_family = str(query_truth.query_source_family.iloc[0])
        exact_axis_family = str(query_truth.exact_axis_family_id.iloc[0])
        axis = str(query_truth.axis.iloc[0])
        direction = str(query_truth.loc[
            query_truth.truth_class.eq("exact_replication"), "expected_direction"
        ].iloc[0])
        canonical_members = [
            {"target_panel_id": panel, "target_contrast_id": contrast}
            for panel, contrast in sorted(registered)
        ]
        universe_hash = _canonical_hash({
            "query_panel_id": query_panel,
            "query_contrast_id": query_contrast,
            "axis": axis,
            "members": canonical_members,
        })
        universe_id = f"formal_exact_axis_sha256:{universe_hash}"
        registry_hash = _canonical_hash([
            {
                "target_panel_id": str(row.target_panel_id),
                "target_contrast_id": str(row.target_contrast_id),
                "truth_class": str(row.truth_class),
                "expected_direction": str(row.expected_direction),
            }
            for row in query_truth.sort_values(
                ["target_panel_id", "target_contrast_id"], kind="stable",
            ).itertuples(index=False)
        ])

        truth_for_merge = query_truth[[
            "target_panel_id", "target_contrast_id", "target_source",
            "target_source_family", "truth_class", "expected_direction",
            "target_exact_axis_family_id", "target_axis",
        ]].rename(columns={
            "target_panel_id": "right_panel_id",
            "target_contrast_id": "right_contrast_id",
        })
        for world in WORLDS:
            candidates = query_matrix.loc[
                query_matrix.world.eq(world)
                & pd.MultiIndex.from_frame(query_matrix[[
                    "right_panel_id", "right_contrast_id",
                ]]).isin(registered)
            ].merge(
                truth_for_merge,
                on=["right_panel_id", "right_contrast_id"],
                how="left", validate="one_to_one",
            )
            if candidates.truth_class.isna().any():
                raise RuntimeError("internal completeness failure after truth merge")
            # Freeze the method-independent primary eligibility face before a
            # ranking score is read.  The digest contains only target IDs and
            # the raw primary answerable/direction flags; it deliberately
            # excludes method scores and all truth/adjudication columns.
            shared_primary_answerability_mask_sha256 = _canonical_hash([
                    {
                        "target_panel_id": str(row.right_panel_id),
                        "target_contrast_id": str(row.right_contrast_id),
                        "answerable": bool(row.answerable),
                        "direction_identified": bool(row.direction_identified),
                    }
                    for row in candidates.sort_values(
                        ["right_panel_id", "right_contrast_id"], kind="stable",
                    ).itertuples(index=False)
            ])
            candidates["ranking_score"] = pd.to_numeric(
                candidates[score_column],
                errors="coerce",
            ) * (1.0 if direction == "similar" else -1.0)
            candidates["ranking_eligible"] = (
                candidates.answerable
                & candidates.direction_identified
                & np.isfinite(candidates.ranking_score)
                & candidates.ranking_score.gt(0)
            )
            ranked = candidates.loc[candidates.ranking_eligible].sort_values(
                ["ranking_score", "right_panel_id", "right_contrast_id"],
                ascending=[False, True, True], kind="stable",
            )
            top = ranked.iloc[0] if len(ranked) else None
            top_class = str(top.truth_class) if top is not None else "abstained"
            correct = top_class == "exact_replication"
            wrong = top_class == "explicit_mismatch"
            answered = bool(correct or wrong)
            rows.append({
                "query_source": query_source,
                "source_family": exact_axis_family,
                "query_provenance_family": query_provenance_family,
                "axis": axis,
                "world": world,
                "ranking_method": ranking_method,
                "evaluation_role": role,
                "whole_query_correct": correct,
                "whole_query_wrong": wrong,
                "whole_query_answered": answered,
                "candidate_universe_id": universe_id,
                "candidate_universe_sha256": universe_hash,
                "candidate_universe_size": len(registered),
                "candidate_universe_frozen": True,
                "shared_primary_answerability_mask_sha256": (
                    shared_primary_answerability_mask_sha256
                ),
                "reference_atlas_id": reference_atlas_id,
                "reference_atlas_frozen": True,
                "query_reference_source_disjoint": True,
                "query_panel_id": query_panel,
                "query_contrast_id": query_contrast,
                "expected_direction": direction,
                "answerable_candidates": int(candidates.ranking_eligible.sum()),
                "top_target_panel_id": (
                    str(top.right_panel_id) if top is not None else ""
                ),
                "top_target_contrast_id": (
                    str(top.right_contrast_id) if top is not None else ""
                ),
                "top_truth_class": top_class,
                "top_signed_conservative_score": (
                    float(top.ranking_score)
                    if top is not None and ranking_method == "transport_conservative"
                    else np.nan
                ),
                "top_aligned_ranking_score": (
                    float(top.ranking_score) if top is not None else np.nan
                ),
                "formal_truth_registry_query_sha256": registry_hash,
            })
            for candidate in candidates.itertuples(index=False):
                candidate_rows.append({
                    "query_source": query_source,
                    "source_family": exact_axis_family,
                    "query_provenance_family": query_provenance_family,
                    "axis": axis,
                    "world": world,
                    "ranking_method": ranking_method,
                    "query_panel_id": query_panel,
                    "query_contrast_id": query_contrast,
                    "target_panel_id": str(candidate.right_panel_id),
                    "target_contrast_id": str(candidate.right_contrast_id),
                    "target_source": str(candidate.target_source),
                    "target_source_family": str(candidate.target_source_family),
                    "target_exact_axis_family_id": str(
                        candidate.target_exact_axis_family_id
                    ),
                    "target_axis": str(candidate.target_axis),
                    "truth_class": str(candidate.truth_class),
                    "answerable": bool(candidate.answerable),
                    "direction_identified": bool(candidate.direction_identified),
                    "signed_conservative_score": (
                        float(candidate.ranking_score)
                        if (
                            ranking_method == "transport_conservative"
                            and np.isfinite(candidate.ranking_score)
                        ) else np.nan
                    ),
                    "aligned_ranking_score": (
                        float(candidate.ranking_score)
                        if np.isfinite(candidate.ranking_score) else np.nan
                    ),
                    "ranking_eligible": bool(candidate.ranking_eligible),
                    "candidate_universe_id": universe_id,
                    "shared_primary_answerability_mask_sha256": (
                        shared_primary_answerability_mask_sha256
                    ),
                })

    outcomes = pd.DataFrame(rows).sort_values(
        ["query_source", "axis", "world"], kind="stable",
    ).reset_index(drop=True)
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["query_source", "axis", "world", "target_panel_id", "target_contrast_id"],
        kind="stable",
    ).reset_index(drop=True)
    report: dict[str, object] = {
        "schema": "pan_redu_formal_exact_axis_source_outcomes_v2_multi_method",
        "status": f"FORMAL_EXACT_AXIS_SOURCE_OUTCOMES_COMPLETE_{role.upper()}",
        "evaluation_role": role,
        "worlds": list(WORLDS),
        "ranking_method": ranking_method,
        "ranking_score_column": score_column,
        "denominators": {
            "formal_queries": int(outcomes[["query_source", "axis"]].drop_duplicates().shape[0]),
            "formal_candidate_pairs": int(len(truth)),
            "raw_matrix_candidate_pairs_for_formal_queries": int(
                len(raw_pairs_for_formal_queries)
            ),
            "excluded_as_unadjudicated": int(len(excluded_unregistered_pairs)),
            "outcome_rows": int(len(outcomes)),
            "correct_outcomes": int(outcomes.whole_query_correct.sum()),
            "wrong_outcomes": int(outcomes.whole_query_wrong.sum()),
            "abstained_outcomes": int((~outcomes.whole_query_answered).sum()),
        },
        "ranking_rule": (
            f"rank only candidates admitted by the shared primary answerability mask using "
            f"{score_column}, aligned to the frozen expected direction and required to be "
            "strictly positive; target IDs are deterministic tie-breakers only. Baselines "
            "cannot change the candidate universe or abstention mask"
        ),
        "truth_contract": (
            "the score-blind registry defines a closed-gold universe of exact_replication "
            "and explicit_mismatch candidates; every target independently carries its frozen "
            "canonical axis and family, and unknown rows inside it fail closed"
        ),
        "candidate_universe_contract": (
            "every registered member must occur in all three worlds and the projected set "
            "is hashed without scores; unregistered raw matrix relations are excluded as "
            "unadjudicated and are never negatives"
        ),
        "shared_answerability_contract": (
            "each query/world hashes target IDs plus raw primary answerable and "
            "direction_identified flags before reading a method score; the hash excludes "
            "ranking scores and truth classes and must be identical across ranking methods"
        ),
        "hierarchy_contract": (
            "query_source_family and target_source_family are provenance families used only "
            "for source-disjointness; output source_family is the invariant biological "
            "exact_axis_family_id used for family-equal inference"
        ),
        "reference_atlas_contract": (
            "all primary query sources and provenance families are globally disjoint from "
            "the score-blind frozen target atlas; reciprocal source reuse is development-only"
        ),
        "reference_atlas_id": reference_atlas_id,
        "claim_boundary": (
            "these are formal retrieval outcomes, not evidence that the registry's biological "
            "adjudication is correct; source-risk inference remains a separate step"
        ),
    }
    return outcomes, candidates, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--matrix-report", type=Path, required=True)
    parser.add_argument("--formal-truth-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ranking-method", choices=sorted(RANKING_METHOD_COLUMNS),
        default="transport_conservative",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    matrix_report = json.loads(args.matrix_report.read_text(encoding="utf-8"))
    validate_matrix_report(matrix_report, args.matrix)
    matrix = pd.read_csv(args.matrix, sep="\t", dtype=str)
    truth = pd.read_csv(args.formal_truth_registry, sep="\t", dtype=str)
    outcomes, candidates, report = build_exact_axis_source_outcomes(
        matrix, truth, ranking_method=args.ranking_method,
    )
    args.output_dir.mkdir(parents=True)
    outcome_path = args.output_dir / "exact_axis_source_risk_outcomes.tsv"
    candidate_path = args.output_dir / "formal_candidate_audit.tsv.gz"
    outcomes.to_csv(outcome_path, sep="\t", index=False)
    candidates.to_csv(candidate_path, sep="\t", index=False, compression="gzip")
    report["provenance"] = {
        "matrix_sha256": sha256(args.matrix),
        "matrix_report_sha256": sha256(args.matrix_report),
        "formal_truth_registry_sha256": sha256(args.formal_truth_registry),
        "outcomes_sha256": sha256(outcome_path),
        "candidate_audit_sha256": sha256(candidate_path),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
