#!/usr/bin/env python3
"""Measure method-dependent entity decisions and biological response recovery.

The candidate universe is the complete same-observed-precursor identity axis.
Methods may change ranking or abstain but may not change that universe.  A
method score selects identity hypotheses; the biological score is then formed
only from independently estimated response effects, so differently calibrated
spectral scores are never multiplied into the biological endpoint.

Additional method tables must contain candidate_pair_id, method_name,
method_score and evidence_available.  This allows B and later integrated models
to enter the exact analysis without modifying samples, effects, or candidates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_msv000095331_chemprop_open_retrieval import (
    CHEMPROP_MGF,
    NEW_STUDY,
    TRUE_DRUG_BY_MIXTURE,
    load_candidate_effects,
    load_new_effects,
)
from build_msv000095331_integrated_algorithm_handoff import sha256


STATUS = "MSV000095331_ENTITY_METHOD_INCREMENT_COMPLETE"
REQUIRED_METHOD_COLUMNS = {
    "candidate_pair_id", "method_name", "method_score", "evidence_available"
}


def load_method_scores(pair_table: pd.DataFrame, paths: list[Path]) -> pd.DataFrame:
    baseline = pair_table[["candidate_pair_id", "wse_score", "wse_matched_fragments"]].copy()
    baseline["method_name"] = "weighted_spectral_entropy"
    baseline["method_score"] = baseline["wse_score"].astype(float)
    baseline["evidence_available"] = (
        baseline["wse_score"].gt(0) & baseline["wse_matched_fragments"].gt(0)
    )
    blocks = [baseline[list(REQUIRED_METHOD_COLUMNS)]]
    for path in paths:
        block = pd.read_csv(path, sep="\t", dtype={"candidate_pair_id": str})
        missing = REQUIRED_METHOD_COLUMNS - set(block.columns)
        if missing:
            raise ValueError(f"{path} misses method columns: {sorted(missing)}")
        block = block[list(REQUIRED_METHOD_COLUMNS)].copy()
        block["method_score"] = pd.to_numeric(block["method_score"], errors="coerce")
        if block["evidence_available"].dtype != bool:
            block["evidence_available"] = (
                block["evidence_available"].astype(str).str.lower().map(
                    {"true": True, "false": False, "1": True, "0": False}
                )
            )
        if block["evidence_available"].isna().any():
            raise ValueError(f"{path} has invalid evidence_available values")
        blocks.append(block)
    scores = pd.concat(blocks, ignore_index=True)
    if scores.duplicated(["method_name", "candidate_pair_id"]).any():
        raise ValueError("duplicate method/candidate pair scores")
    unknown = set(scores["candidate_pair_id"]) - set(pair_table["candidate_pair_id"])
    if unknown:
        raise ValueError(f"method tables contain {len(unknown)} out-of-universe pairs")
    return scores


def choose_entities(
    pair_table: pd.DataFrame,
    query_table: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    methods = sorted(scores["method_name"].astype(str).unique())
    pair_keys = pair_table[[
        "candidate_pair_id", "query_id", "candidate_id", "query_study_id",
        "query_raw_entity_id", "candidate_study_id", "candidate_raw_entity_id",
        "query_identity_status", "candidate_identity_status",
        "precursor_ppm_difference",
    ]]
    scored = scores.merge(pair_keys, on="candidate_pair_id", how="left", validate="many_to_one")
    rows: list[dict] = []
    query_base = query_table.copy()
    for method in methods:
        method_scores = scored.loc[scored["method_name"].eq(method)].copy()
        valid = method_scores.loc[
            method_scores["evidence_available"].eq(True)
            & np.isfinite(method_scores["method_score"])
        ].copy()
        selected: dict[tuple[str, str], pd.Series] = {}
        for key, block in valid.groupby(["query_id", "candidate_study_id"], sort=False):
            block = block.sort_values(
                ["method_score", "precursor_ppm_difference", "candidate_id"],
                ascending=[False, True, True], kind="mergesort",
            )
            top = block.iloc[0].copy()
            top["method_score_margin"] = (
                float(top["method_score"] - block.iloc[1]["method_score"])
                if len(block) >= 2 else np.nan
            )
            top["ranked_candidate_count"] = int(len(block))
            selected[(str(key[0]), str(key[1]))] = top
        for query in query_base.itertuples(index=False):
            key = (str(query.query_id), str(query.candidate_study_id))
            top = selected.get(key)
            row = {
                "method_name": method,
                "query_id": str(query.query_id),
                "query_study_id": str(query.query_study_id),
                "query_raw_entity_id": str(query.query_raw_entity_id),
                "candidate_study_id": str(query.candidate_study_id),
                "candidate_count": int(query.candidate_count),
            }
            if int(query.candidate_count) == 0:
                row.update({
                    "decision_status": "no_mass_compatible_candidate",
                    "selected_candidate_pair_id": "", "selected_candidate_id": "",
                    "selected_candidate_raw_entity_id": "", "selected_method_score": np.nan,
                    "selected_score_margin": np.nan, "ranked_candidate_count": 0,
                    "selected_precursor_ppm_difference": np.nan,
                })
            elif top is None:
                row.update({
                    "decision_status": "mass_candidates_but_method_abstained",
                    "selected_candidate_pair_id": "", "selected_candidate_id": "",
                    "selected_candidate_raw_entity_id": "", "selected_method_score": np.nan,
                    "selected_score_margin": np.nan, "ranked_candidate_count": 0,
                    "selected_precursor_ppm_difference": np.nan,
                })
            else:
                row.update({
                    "decision_status": "selected_identity_hypothesis",
                    "selected_candidate_pair_id": str(top["candidate_pair_id"]),
                    "selected_candidate_id": str(top["candidate_id"]),
                    "selected_candidate_raw_entity_id": str(top["candidate_raw_entity_id"]),
                    "selected_method_score": float(top["method_score"]),
                    "selected_score_margin": float(top["method_score_margin"]),
                    "ranked_candidate_count": int(top["ranked_candidate_count"]),
                    "selected_precursor_ppm_difference": float(top["precursor_ppm_difference"]),
                })
            rows.append(row)
    decisions = pd.DataFrame(rows)
    decisions["mutual_top1"] = False
    for method, block in decisions.groupby("method_name", sort=False):
        answered = block["decision_status"].eq("selected_identity_hypothesis")
        selected = {
            (str(row.query_id), str(row.selected_candidate_id))
            for row in block.loc[answered].itertuples(index=False)
        }
        indices = block.index.to_numpy()
        values = [
            bool(
                row.decision_status == "selected_identity_hypothesis"
                and (str(row.selected_candidate_id), str(row.query_id)) in selected
            )
            for row in block.itertuples(index=False)
        ]
        decisions.loc[indices, "mutual_top1"] = values
    return decisions


def method_transitions(decisions: pd.DataFrame) -> pd.DataFrame:
    baseline_name = "weighted_spectral_entropy"
    baseline = decisions.loc[decisions["method_name"].eq(baseline_name)].copy()
    keys = ["query_id", "candidate_study_id"]
    baseline = baseline[keys + [
        "decision_status", "selected_candidate_id", "mutual_top1"
    ]].rename(columns={
        "decision_status": "baseline_status",
        "selected_candidate_id": "baseline_candidate_id",
        "mutual_top1": "baseline_mutual_top1",
    })
    outputs = []
    for method in sorted(set(decisions["method_name"]) - {baseline_name}):
        current = decisions.loc[decisions["method_name"].eq(method)].merge(
            baseline, on=keys, how="left", validate="one_to_one"
        )
        base_answered = current["baseline_status"].eq("selected_identity_hypothesis")
        now_answered = current["decision_status"].eq("selected_identity_hypothesis")
        current["transition_vs_wse"] = np.select(
            [
                base_answered & now_answered
                & current["baseline_candidate_id"].eq(current["selected_candidate_id"]),
                base_answered & now_answered,
                ~base_answered & now_answered,
                base_answered & ~now_answered,
            ],
            ["retained", "changed", "gained", "lost"],
            default="unresolved_both",
        )
        outputs.append(current)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(columns=[
        *keys, "method_name", "transition_vs_wse"
    ])


def response_weights(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    denominator = np.sqrt(table["beta"].to_numpy(float) ** 2 + table["se"].to_numpy(float) ** 2)
    table["response_weight"] = np.divide(
        table["beta"].to_numpy(float), denominator,
        out=np.zeros(len(table), dtype=float), where=denominator > 0,
    )
    return table


def build_response_support(
    decisions: pd.DataFrame,
    new_effects: pd.DataFrame,
    candidate_effects: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = decisions.loc[
        decisions["query_study_id"].eq(NEW_STUDY)
        & decisions["mutual_top1"].eq(True)
    ].copy()
    q = response_weights(new_effects)[[
        "raw_entity_id", "contrast_id", "mixture", "query_stratum",
        "beta", "se", "response_weight", "prevalence",
    ]].rename(columns={
        "raw_entity_id": "query_raw_entity_id", "contrast_id": "query_contrast_id",
        "beta": "query_beta", "se": "query_se",
        "response_weight": "query_response_weight", "prevalence": "query_prevalence",
    })
    c = response_weights(candidate_effects)[[
        "study_id", "raw_entity_id", "contrast_id", "beta", "se",
        "response_weight", "prevalence",
    ]].rename(columns={
        "study_id": "candidate_study_id", "raw_entity_id": "selected_candidate_raw_entity_id",
        "contrast_id": "candidate_contrast_id", "beta": "candidate_beta",
        "se": "candidate_se", "response_weight": "candidate_response_weight",
        "prevalence": "candidate_prevalence",
    })
    support = edges.merge(q, on="query_raw_entity_id", how="inner", validate="many_to_many")
    support = support.merge(
        c,
        on=["candidate_study_id", "selected_candidate_raw_entity_id"],
        how="inner", validate="many_to_many",
    )
    support["response_product"] = (
        support["query_response_weight"] * support["candidate_response_weight"]
    )
    support["direction_relation"] = np.where(
        support["response_product"].gt(0), "same_direction",
        np.where(support["response_product"].lt(0), "opposite_direction", "zero_direction"),
    )
    expected = support["mixture"].map(TRUE_DRUG_BY_MIXTURE)
    support["prespecified_overlap"] = support["candidate_contrast_id"].eq(expected)

    # Fixed full-response denominators retain coverage effects when methods
    # select different numbers of mutual identity hypotheses.
    q_norm = response_weights(new_effects).groupby("contrast_id")["response_weight"].apply(
        lambda value: float(np.sqrt(np.square(value.to_numpy(float)).sum()))
    )
    c_norm = response_weights(candidate_effects).groupby(
        ["study_id", "contrast_id"]
    )["response_weight"].apply(
        lambda value: float(np.sqrt(np.square(value.to_numpy(float)).sum()))
    )
    grouped = support.groupby(
        ["method_name", "query_contrast_id", "candidate_study_id", "candidate_contrast_id"],
        sort=False,
    )
    scores = grouped.agg(
        matched_entity_pairs=("selected_candidate_pair_id", "nunique"),
        response_numerator=("response_product", "sum"),
        same_direction_pairs=("direction_relation", lambda value: int((value == "same_direction").sum())),
        opposite_direction_pairs=("direction_relation", lambda value: int((value == "opposite_direction").sum())),
        prespecified_overlap=("prespecified_overlap", "max"),
    ).reset_index()
    scores["query_response_norm"] = scores["query_contrast_id"].map(q_norm)
    scores["candidate_response_norm"] = [
        c_norm.get((study, contrast), np.nan)
        for study, contrast in zip(scores["candidate_study_id"], scores["candidate_contrast_id"])
    ]
    scores["coverage_sensitive_response_score"] = np.divide(
        scores["response_numerator"],
        scores["query_response_norm"] * scores["candidate_response_norm"],
        out=np.full(len(scores), np.nan),
        where=(scores["query_response_norm"] * scores["candidate_response_norm"]).to_numpy(float) > 0,
    )
    scores["candidate_rank"] = scores.groupby(
        ["method_name", "query_contrast_id"]
    )["coverage_sensitive_response_score"].rank(ascending=False, method="max")
    return support, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity-handoff", type=Path,
        default=Path("data/validation/msv000095331_complete_identity_handoff_v3_20261023"),
    )
    parser.add_argument(
        "--new-response-dir", type=Path,
        default=Path("data/validation/msv000095331_drug_endpoint_interaction_responses_v3_20261023"),
    )
    parser.add_argument(
        "--collection", type=Path,
        default=Path("data/validation/contrast_atlas_open_response_collection_v9_frozenbase_chemprop_20261023"),
    )
    parser.add_argument("--method-score-table", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    pair_path = args.identity_handoff / "complete_identity_candidate_pairs.tsv.gz"
    query_path = args.identity_handoff / "directional_query_registry.tsv.gz"
    pairs = pd.read_csv(pair_path, sep="\t", dtype=str)
    queries = pd.read_csv(query_path, sep="\t", dtype=str)
    numeric_pair = ["wse_score", "wse_matched_fragments", "precursor_ppm_difference"]
    for column in numeric_pair:
        pairs[column] = pd.to_numeric(pairs[column], errors="raise")
    queries["candidate_count"] = pd.to_numeric(queries["candidate_count"], errors="raise")
    method_scores = load_method_scores(pairs, args.method_score_table)
    decisions = choose_entities(pairs, queries, method_scores)
    transitions = method_transitions(decisions)

    new_effects = load_new_effects(args.new_response_dir / "effects.tsv")
    candidate_effects = load_candidate_effects(
        args.collection / "response_entities.tsv.gz", set(CHEMPROP_MGF)
    )
    support, response_scores = build_response_support(decisions, new_effects, candidate_effects)
    registered = response_scores.loc[response_scores["prespecified_overlap"].eq(True)].copy()
    priority_source = support.loc[support["prespecified_overlap"].eq(True)].copy()
    priority_source["absolute_response_product"] = priority_source["response_product"].abs()
    priority = priority_source.groupby([
        "method_name", "selected_candidate_pair_id", "query_id", "selected_candidate_id",
        "candidate_study_id",
    ], as_index=False).agg(
        selected_method_score=("selected_method_score", "first"),
        selected_score_margin=("selected_score_margin", "first"),
        ranked_candidate_count=("ranked_candidate_count", "first"),
        maximum_absolute_response_product=("absolute_response_product", "max"),
        mean_absolute_response_product=("absolute_response_product", "mean"),
        same_direction_contexts=(
            "direction_relation", lambda value: int((value == "same_direction").sum())
        ),
        opposite_direction_contexts=(
            "direction_relation", lambda value: int((value == "opposite_direction").sum())
        ),
        registered_response_rows=("query_contrast_id", "size"),
    )
    priority = priority.sort_values(
        ["maximum_absolute_response_product", "selected_score_margin"],
        ascending=[False, True], kind="mergesort",
    ).head(100)

    args.output_dir.mkdir(parents=True)
    compression = {"method": "gzip", "mtime": 0}
    outputs = {
        "entity_decisions": args.output_dir / "entity_method_decisions.tsv.gz",
        "method_transitions": args.output_dir / "method_transitions_vs_wse.tsv.gz",
        "entity_response_support": args.output_dir / "entity_response_support.tsv.gz",
        "response_retrieval_scores": args.output_dir / "response_retrieval_scores.tsv.gz",
        "prespecified_overlap_results": args.output_dir / "prespecified_overlap_results.tsv",
        "priority_entities": args.output_dir / "priority_entities_for_algorithm_delta.tsv",
    }
    decisions.to_csv(outputs["entity_decisions"], sep="\t", index=False, compression=compression)
    transitions.to_csv(outputs["method_transitions"], sep="\t", index=False, compression=compression)
    support.to_csv(outputs["entity_response_support"], sep="\t", index=False, compression=compression)
    response_scores.to_csv(outputs["response_retrieval_scores"], sep="\t", index=False, compression=compression)
    registered.to_csv(outputs["prespecified_overlap_results"], sep="\t", index=False)
    priority.to_csv(outputs["priority_entities"], sep="\t", index=False)

    method_summary = {}
    for method, block in decisions.groupby("method_name", sort=False):
        forward = block["query_study_id"].eq(NEW_STUDY)
        method_registered = registered.loc[registered["method_name"].eq(method)]
        method_summary[str(method)] = {
            "forward_entity_queries": int(forward.sum()),
            "forward_selected_identity_hypotheses": int(
                (forward & block["decision_status"].eq("selected_identity_hypothesis")).sum()
            ),
            "forward_mutual_identity_hypotheses": int((forward & block["mutual_top1"]).sum()),
            "prespecified_overlap_contexts": int(len(method_registered)),
            "prespecified_overlap_top1_contexts": int(method_registered["candidate_rank"].eq(1).sum()),
            "prespecified_overlap_median_rank": (
                float(method_registered["candidate_rank"].median()) if len(method_registered) else None
            ),
        }
    report = {
        "status": STATUS,
        "dataset": "MSV000095331 plus three ChemProp studies",
        "dataset_role": "development_consumed_not_external_validation",
        "candidate_axis": "complete same-observed-precursor 10-ppm identity hypotheses",
        "methods": sorted(method_summary),
        "method_summary": method_summary,
        "denominators": {
            "new_study_response_entities": int(new_effects["raw_entity_id"].nunique()),
            "new_study_response_contrasts": int(new_effects["contrast_id"].nunique()),
            "candidate_response_contrasts": int(
                candidate_effects[["study_id", "contrast_id"]].drop_duplicates().shape[0]
            ),
            "directional_entity_queries_per_method": int(len(queries)),
            "identity_candidate_pairs": int(len(pairs)),
        },
        "biological_endpoint": {
            "entity_selection": "method Top-1 followed by cross-direction mutual Top-1",
            "response_score": "sum of matched response-weight products divided by fixed full-contrast L2 norms",
            "spectral_score_enters_response_endpoint": False,
            "prespecified_drug_overlap": TRUE_DRUG_BY_MIXTURE,
            "independence_note": "five microbial strata per mixture are repeated contexts, not independent studies",
        },
        "files": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in outputs.items()
        },
        "inputs": {
            "candidate_pairs_sha256": sha256(pair_path),
            "query_registry_sha256": sha256(query_path),
            "additional_method_tables": [
                {"path": str(path), "sha256": sha256(path)} for path in args.method_score_table
            ],
        },
        "claim_boundary": [
            "Matched unresolved entities remain identity hypotheses without standards or orthogonal identity evidence.",
            "Prespecified drug overlaps are positive controls; ranks across repeated microbial strata are not independent discoveries.",
            "The response endpoint retains coverage and abstention costs and does not reward a method for its raw score scale.",
            "A WSE-only run is a baseline, not evidence that the integrated algorithm changed biology.",
            "Priority entities are post-response follow-up targets and must never be used to fit or tune the spectral model.",
        ],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
