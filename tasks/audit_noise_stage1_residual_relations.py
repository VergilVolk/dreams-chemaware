"""Decompose frozen official DreaMS Top-1 residuals by rival relationship.

This is a read-only CPU audit.  It consumes the corrected candidate graph and
the Stage-1 held-per-query ledger; it does not encode spectra, fit a model, or
use outer outcomes to select training examples.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


GRADE_NAME = {-2: "unclassified", -1: "positive", 0: "near", 1: "mid", 2: "far"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--held-per-query", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_held(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("held-per-query ledger is empty")
    required = {
        "query_index", "query_row", "query_ik14", "query_formula", "near",
        "official_rank", "candidate_rank", "corrected", "introduced",
    }
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"held-per-query ledger lacks {sorted(missing)}")
    return rows


def rank_bucket(rank: int) -> str:
    if rank <= 3:
        return str(rank)
    if rank <= 5:
        return "4-5"
    return "6+"


def relation_category(grade: int, same_formula: bool) -> str:
    if grade == 0:
        return "near_subset_top_rival"
    if same_formula:
        return "same_formula_non_near_top_rival"
    if grade == 1:
        return "mces_mid_top_rival"
    if grade == 2:
        return "mces_far_top_rival"
    return "unclassified_top_rival"


def summarize_counter(counter: Counter[str], denominator: int) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "queries": int(value),
            "fraction": float(value / denominator) if denominator else 0.0,
        }
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    }


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if not args.graph.is_file() or not args.held_per_query.is_file():
        raise FileNotFoundError("graph or held-per-query input is missing")

    held = load_held(args.held_per_query)
    with np.load(args.graph, allow_pickle=False) as body:
        graph = {name: body[name] for name in body.files}
    required_graph = {
        "features", "feature_names", "query_ptr", "molecule_ptr",
        "molecule_label", "molecule_ik14", "molecule_formula",
        "molecule_mces_grade", "query_row", "query_ik14", "query_formula",
        "query_has_near",
    }
    missing = required_graph - set(graph)
    if missing:
        raise RuntimeError(f"candidate graph lacks {sorted(missing)}")
    feature_names = list(map(str, graph["feature_names"]))
    if "dreams_similarity" not in feature_names:
        raise RuntimeError("candidate graph lacks official DreaMS similarity")
    score_column = feature_names.index("dreams_similarity")

    query_ptr = np.asarray(graph["query_ptr"], dtype=np.int64)
    molecule_ptr = np.asarray(graph["molecule_ptr"], dtype=np.int64)
    molecule_label = np.asarray(graph["molecule_label"], dtype=np.int8)
    molecule_ik14 = np.asarray(graph["molecule_ik14"]).astype(str)
    molecule_formula = np.asarray(graph["molecule_formula"]).astype(str)
    molecule_grade = np.asarray(graph["molecule_mces_grade"], dtype=np.int8)
    query_row = np.asarray(graph["query_row"], dtype=np.int64)
    query_ik14 = np.asarray(graph["query_ik14"]).astype(str)
    query_formula = np.asarray(graph["query_formula"]).astype(str)
    query_has_near = np.asarray(graph["query_has_near"], dtype=bool)
    pair_scores = np.asarray(graph["features"][:, score_column], dtype=np.float32)
    molecule_scores = np.maximum.reduceat(pair_scores, molecule_ptr[:-1])

    details: list[dict[str, object]] = []
    rank_counts: Counter[str] = Counter()
    candidate_rank_counts: Counter[str] = Counter()
    selection_rank_mismatches = 0
    official_relation: Counter[str] = Counter()
    candidate_residual_official_relation: Counter[str] = Counter()
    corrected_by_relation: Counter[str] = Counter()
    introduced_by_relation: Counter[str] = Counter()
    transition_by_relation: dict[str, Counter[str]] = defaultdict(Counter)

    for source in held:
        query = int(source["query_index"])
        if query < 0 or query + 1 >= len(query_ptr):
            raise RuntimeError(f"query index {query} is outside the graph")
        if (
            int(source["query_row"]) != int(query_row[query])
            or source["query_ik14"] != query_ik14[query]
            or source["query_formula"] != query_formula[query]
        ):
            raise RuntimeError(f"held/graph query metadata disagree at {query}")
        left, right = map(int, query_ptr[query:query + 2])
        labels = molecule_label[left:right]
        if labels[0] != 1 or int(labels.sum()) != 1:
            raise RuntimeError(f"query {query} has a malformed positive molecule")
        scores = molecule_scores[left:right]
        positive_score = float(scores[0])
        negative_local = 1 + int(np.argmax(scores[1:]))
        negative_global = left + negative_local
        negative_score = float(scores[negative_local])
        selection_rank = 1 + int(np.sum(scores[1:] >= scores[0]))
        recorded_official = int(source["official_rank"])
        selection_rank_mismatches += int(selection_rank != recorded_official)
        official_rank = recorded_official
        candidate_rank = int(source["candidate_rank"])
        same_formula = bool(molecule_formula[negative_global] == query_formula[query])
        grade = int(molecule_grade[negative_global])
        relation = relation_category(grade, same_formula)
        corrected = source["corrected"].lower() == "true"
        introduced = source["introduced"].lower() == "true"
        transition = "corrected" if corrected else "introduced" if introduced else "unchanged"
        official_error = official_rank > 1
        candidate_error = candidate_rank > 1

        if official_error:
            rank_counts[rank_bucket(official_rank)] += 1
            official_relation[relation] += 1
        if candidate_error:
            candidate_rank_counts[rank_bucket(candidate_rank)] += 1
            candidate_residual_official_relation[relation] += 1
        if corrected:
            corrected_by_relation[relation] += 1
        if introduced:
            introduced_by_relation[relation] += 1
        transition_by_relation[relation][transition] += 1

        details.append({
            "query_index": query,
            "query_row": int(query_row[query]),
            "query_ik14": query_ik14[query],
            "query_formula": query_formula[query],
            "query_has_any_near_candidate": bool(query_has_near[query]),
            "official_rank": official_rank,
            "stage1_rank": candidate_rank,
            "official_error": official_error,
            "stage1_error": candidate_error,
            "transition": transition,
            "selection_geometry_rank": selection_rank,
            "selection_geometry_positive_score": positive_score,
            "selection_geometry_best_negative_score": negative_score,
            "selection_geometry_positive_minus_best_negative": positive_score - negative_score,
            "selection_geometry_best_negative_ik14": molecule_ik14[negative_global],
            "selection_geometry_best_negative_formula": molecule_formula[negative_global],
            "selection_geometry_best_negative_same_formula": same_formula,
            "selection_geometry_best_negative_mces_grade": grade,
            "selection_geometry_best_negative_mces_name": GRADE_NAME.get(grade, f"grade_{grade}"),
            "selection_geometry_best_negative_relation": relation,
            "candidate_molecules": right - left,
            "positive_spectra": int(molecule_ptr[left + 1] - molecule_ptr[left]),
            "best_negative_spectra": int(
                molecule_ptr[negative_global + 1] - molecule_ptr[negative_global]
            ),
        })

    n = len(details)
    official_errors = sum(int(row["official_error"]) for row in details)
    candidate_errors = sum(int(row["stage1_error"]) for row in details)
    official_near_errors = sum(
        int(row["official_error"] and row["query_has_any_near_candidate"])
        for row in details
    )
    official_top_near_errors = sum(
        int(row["official_error"] and row["selection_geometry_best_negative_mces_grade"] == 0)
        for row in details
    )
    candidate_near_errors = sum(
        int(row["stage1_error"] and row["query_has_any_near_candidate"])
        for row in details
    )
    required_five_pp = int(np.ceil(0.05 * n))
    candidate_rank2 = candidate_rank_counts.get("2", 0)
    candidate_rank3 = candidate_rank_counts.get("3", 0)
    corrected = sum(int(row["transition"] == "corrected") for row in details)
    introduced = sum(int(row["transition"] == "introduced") for row in details)

    report = {
        "status": "NOISE_STAGE1_RESIDUAL_RELATION_AUDIT_COMPLETE",
        "inputs": {
            "graph": str(args.graph),
            "held_per_query": str(args.held_per_query),
        },
        "population": {
            "queries": n,
            "official_errors": official_errors,
            "official_recall1": 1.0 - official_errors / n,
            "stage1_errors": candidate_errors,
            "stage1_recall1": 1.0 - candidate_errors / n,
            "stage1_corrected": corrected,
            "stage1_introduced": introduced,
            "stage1_net_corrections": corrected - introduced,
            "stage1_risk_net_lambda2": corrected - 2 * introduced,
            "selection_geometry_rank_mismatches_vs_reencoded_official": (
                selection_rank_mismatches
            ),
            "selection_geometry_rank_match_fraction": (
                1.0 - selection_rank_mismatches / n
            ),
        },
        "rank_structure": {
            "official_error_rank": summarize_counter(rank_counts, official_errors),
            "stage1_error_rank": summarize_counter(candidate_rank_counts, candidate_errors),
            "official_rank2_plus_rank3_oracle_pp": 100.0 * (
                rank_counts.get("2", 0) + rank_counts.get("3", 0)
            ) / n,
            "stage1_rank2_plus_rank3_oracle_pp": 100.0 * (
                candidate_rank2 + candidate_rank3
            ) / n,
        },
        "near_structure": {
            "official_errors_in_queries_with_any_near_candidate": official_near_errors,
            "fraction_of_official_errors_in_queries_with_any_near_candidate": (
                official_near_errors / official_errors
            ),
            "official_errors_whose_actual_top_rival_is_near": official_top_near_errors,
            "fraction_of_official_errors_whose_actual_top_rival_is_near": (
                official_top_near_errors / official_errors
            ),
            "stage1_errors_in_queries_with_any_near_candidate": candidate_near_errors,
            "fraction_of_stage1_errors_in_queries_with_any_near_candidate": (
                candidate_near_errors / candidate_errors
            ),
        },
        "official_error_selection_geometry_top_rival_relation": summarize_counter(
            official_relation, official_errors,
        ),
        "stage1_residual_by_selection_geometry_top_rival_relation": summarize_counter(
            candidate_residual_official_relation, candidate_errors,
        ),
        "stage1_transitions_by_selection_geometry_top_rival_relation": {
            relation: dict(sorted(counter.items()))
            for relation, counter in sorted(transition_by_relation.items())
        },
        "five_pp_arithmetic": {
            "required_net_corrections": required_five_pp,
            "fraction_of_stage1_errors_required": required_five_pp / candidate_errors,
            "stage1_rank2_errors": candidate_rank2,
            "stage1_rank3_errors": candidate_rank3,
            "stage1_rank2_plus_rank3": candidate_rank2 + candidate_rank3,
            "stage1_rank2_plus_rank3_headroom_pp": 100.0 * (
                candidate_rank2 + candidate_rank3
            ) / n,
            "rank2_plus_rank3_suffices_for_five_pp": (
                candidate_rank2 + candidate_rank3 >= required_five_pp
            ),
        },
        "claim_limit": (
            "Ranks come from the reencoded-official held ledger. Rival relationship strata "
            "come from the older frozen selection geometry and their rank agreement is "
            "reported explicitly. They do not identify annotation errors or prove that a "
            "relation-aware model can realize zero-introduction oracle headroom."
        ),
    }

    args.output_dir.mkdir(parents=True)
    with gzip.open(
        args.output_dir / "per_query.csv.gz", "wt", encoding="utf-8", newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
