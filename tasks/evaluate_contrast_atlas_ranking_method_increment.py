#!/usr/bin/env python3
"""Formal all-world increment of transport over two fixed ranking baselines.

Inputs are three outcome tables produced separately by
``build_pan_redu_exact_axis_source_outcomes.py`` using the fixed ranking methods
``transport_conservative``, ``greedy_same_graph``, and
``centroid_same_encoder``.  This evaluator never sees candidate scores, never
retunes the strictly-positive ranking rule, and never changes a method's
recorded correct/wrong/abstained outcome.

The primary endpoint is utility = correct - 2 * wrong (abstention = 0), with
axes averaged inside query source, query sources averaged inside biological
family, and families weighted equally.  Each primary-minus-baseline comparison
uses a query-source sign flip and the complete gate; both fixed baselines must
pass as an intersection-union test.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from contrast_response_transport import parse_bool
from evaluate_contrast_atlas_source_risk import source_sign_flip


PRIMARY_METHOD = "transport_conservative"
BASELINE_METHODS = ("greedy_same_graph", "centroid_same_encoder")
ALL_WORLD = "all"
VALID_ROLES = {"development", "external"}
KEY_COLUMNS = ("query_source", "axis")
METHOD_OUTCOME_COLUMNS = (
    "whole_query_correct", "whole_query_wrong", "whole_query_answered",
)
ROW_INVARIANTS = (
    "query_source", "source_family", "query_provenance_family", "axis",
    "evaluation_role", "candidate_universe_id", "candidate_universe_sha256",
    "candidate_universe_size", "candidate_universe_frozen",
    "reference_atlas_id", "reference_atlas_frozen",
    "query_reference_source_disjoint", "query_panel_id", "query_contrast_id",
    "expected_direction", "formal_truth_registry_query_sha256",
    "shared_primary_answerability_mask_sha256",
)


@dataclass(frozen=True)
class RankingIncrementEvaluation:
    query_axis_ledger: pd.DataFrame
    source_summary: pd.DataFrame
    family_summary: pd.DataFrame
    leave_one_family_out: pd.DataFrame
    sign_flip_distribution: pd.DataFrame
    report: dict[str, object]


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


def validate_method_outcomes(
    raw: pd.DataFrame,
    expected_method: str,
) -> tuple[pd.DataFrame, str]:
    """Validate one builder output and retain exactly its ``all`` rows."""
    required = {
        "world", "ranking_method", *ROW_INVARIANTS, *METHOD_OUTCOME_COLUMNS,
        "answerable_candidates",
    }
    if missing := required - set(raw):
        raise RuntimeError(
            f"{expected_method} outcome table lacks columns: {sorted(missing)}"
        )
    frame = raw.copy()
    text_columns = {
        "world", "ranking_method", "query_source", "source_family",
        "query_provenance_family", "axis", "evaluation_role",
        "candidate_universe_id", "candidate_universe_sha256",
        "reference_atlas_id", "query_panel_id", "query_contrast_id",
        "expected_direction", "formal_truth_registry_query_sha256",
        "shared_primary_answerability_mask_sha256",
    }
    for column in text_columns:
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise RuntimeError(f"{expected_method} outcome table has empty {column}")
    if set(frame.ranking_method) != {expected_method}:
        raise RuntimeError(
            f"outcome ranking_method is not invariant {expected_method}"
        )
    frame = frame.loc[frame.world.eq(ALL_WORLD)].copy()
    if frame.empty:
        raise RuntimeError(f"{expected_method} outcome table lacks world=all")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError(f"{expected_method} has duplicate query_source/axis all rows")

    for column in (
        *METHOD_OUTCOME_COLUMNS, "candidate_universe_frozen",
        "reference_atlas_frozen", "query_reference_source_disjoint",
    ):
        frame[column] = _strict_bool(frame[column], f"{expected_method}.{column}")
    if (frame.whole_query_correct & frame.whole_query_wrong).any():
        raise RuntimeError(f"{expected_method} marks one query both correct and wrong")
    expected_answered = frame.whole_query_correct | frame.whole_query_wrong
    if not frame.whole_query_answered.equals(expected_answered):
        raise RuntimeError(
            f"{expected_method} answered flag is inconsistent with correct/wrong"
        )
    for column in ("candidate_universe_frozen", "reference_atlas_frozen"):
        if not bool(frame[column].all()):
            raise RuntimeError(f"{expected_method} requires {column}=true")
    if not bool(frame.query_reference_source_disjoint.all()):
        raise RuntimeError(
            f"{expected_method} contains a query source not held out from the reference atlas"
        )

    for column in ("candidate_universe_size", "answerable_candidates"):
        numeric = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        if np.any(~np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
            raise RuntimeError(f"{column} must contain finite integers")
        frame[column] = numeric.astype(int)
    if (frame.candidate_universe_size <= 0).any():
        raise RuntimeError("candidate_universe_size must be positive")
    if (
        (frame.answerable_candidates < 0)
        | (frame.answerable_candidates > frame.candidate_universe_size)
    ).any():
        raise RuntimeError("answerable_candidates lies outside its candidate universe")

    roles = set(frame.evaluation_role)
    if len(roles) != 1 or not roles.issubset(VALID_ROLES):
        raise RuntimeError(
            f"{expected_method} evaluation_role must be one invariant development or external value"
        )
    role = next(iter(roles))
    if frame.reference_atlas_id.nunique() != 1:
        raise RuntimeError(f"{expected_method} mixes frozen reference atlases")
    if not frame.expected_direction.isin({"similar", "opposite"}).all():
        raise RuntimeError(f"{expected_method} has an invalid expected_direction")
    if (frame.groupby("query_source").source_family.nunique() != 1).any():
        raise RuntimeError(f"{expected_method} maps one source to multiple biological families")
    if (frame.groupby("query_source").query_provenance_family.nunique() != 1).any():
        raise RuntimeError(f"{expected_method} maps one source to multiple provenance families")
    if (
        frame.groupby("query_provenance_family").query_source.nunique() != 1
    ).any():
        raise RuntimeError(
            f"{expected_method} maps one provenance family to multiple nominal query sources"
        )
    return frame.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True), role


def align_method_outcomes(
    primary_raw: pd.DataFrame,
    greedy_raw: pd.DataFrame,
    centroid_raw: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Fail closed unless all score-blind formal universes are identical."""
    raw_by_method = {
        PRIMARY_METHOD: primary_raw,
        "greedy_same_graph": greedy_raw,
        "centroid_same_encoder": centroid_raw,
    }
    tables: dict[str, pd.DataFrame] = {}
    roles: set[str] = set()
    for method, raw in raw_by_method.items():
        tables[method], role = validate_method_outcomes(raw, method)
        roles.add(role)
    if len(roles) != 1:
        raise RuntimeError("evaluation_role differs across ranking methods")

    primary = tables[PRIMARY_METHOD]
    primary_keys = pd.MultiIndex.from_frame(primary[list(KEY_COLUMNS)])
    for method in BASELINE_METHODS:
        candidate = tables[method]
        candidate_keys = pd.MultiIndex.from_frame(candidate[list(KEY_COLUMNS)])
        if not primary_keys.equals(candidate_keys):
            missing = primary_keys.difference(candidate_keys)
            extra = candidate_keys.difference(primary_keys)
            raise RuntimeError(
                f"{method} query/axis universe differs from primary: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        for column in ROW_INVARIANTS:
            if not primary[column].equals(candidate[column]):
                raise RuntimeError(
                    f"{column} differs between {PRIMARY_METHOD} and {method}"
                )
    return tables, next(iter(roles))


def build_query_axis_ledger(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    primary = tables[PRIMARY_METHOD]
    ledger = primary[list(ROW_INVARIANTS)].copy()
    for method, frame in tables.items():
        correct = frame.whole_query_correct.astype(bool)
        wrong = frame.whole_query_wrong.astype(bool)
        answered = frame.whole_query_answered.astype(bool)
        ledger[f"correct__{method}"] = correct.to_numpy(bool)
        ledger[f"wrong__{method}"] = wrong.to_numpy(bool)
        ledger[f"answered__{method}"] = answered.to_numpy(bool)
        ledger[f"utility__{method}"] = (
            correct.astype(int) - 2 * wrong.astype(int)
        ).to_numpy(int)
        ledger[f"answerable_candidates__{method}"] = frame.answerable_candidates.to_numpy(int)
    for baseline in BASELINE_METHODS:
        ledger[f"delta_utility__{baseline}"] = (
            ledger[f"utility__{PRIMARY_METHOD}"] - ledger[f"utility__{baseline}"]
        )
        ledger[f"corrected__{baseline}"] = (
            ledger[f"correct__{PRIMARY_METHOD}"] & ~ledger[f"correct__{baseline}"]
        )
        ledger[f"lost_correct__{baseline}"] = (
            ~ledger[f"correct__{PRIMARY_METHOD}"] & ledger[f"correct__{baseline}"]
        )
        ledger[f"wrong_rescued__{baseline}"] = (
            ~ledger[f"wrong__{PRIMARY_METHOD}"] & ledger[f"wrong__{baseline}"]
        )
        ledger[f"introduced_wrong__{baseline}"] = (
            ledger[f"wrong__{PRIMARY_METHOD}"] & ~ledger[f"wrong__{baseline}"]
        )
    return ledger


def hierarchical_summaries(
    ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_pieces: list[pd.DataFrame] = []
    for baseline in BASELINE_METHODS:
        piece = ledger.groupby(
            ["query_source", "source_family", "evaluation_role"], sort=True,
        ).agg(
            axes=("axis", "nunique"),
            mean_primary_utility=(f"utility__{PRIMARY_METHOD}", "mean"),
            mean_baseline_utility=(f"utility__{baseline}", "mean"),
            mean_delta_utility=(f"delta_utility__{baseline}", "mean"),
            corrected=(f"corrected__{baseline}", "sum"),
            lost_correct=(f"lost_correct__{baseline}", "sum"),
            wrong_rescued=(f"wrong_rescued__{baseline}", "sum"),
            introduced_wrong=(f"introduced_wrong__{baseline}", "sum"),
        ).reset_index()
        piece.insert(0, "baseline_method", baseline)
        source_pieces.append(piece)
    source = pd.concat(source_pieces, ignore_index=True)
    family = source.groupby(
        ["baseline_method", "source_family", "evaluation_role"], sort=True,
    ).agg(
        sources=("query_source", "nunique"),
        mean_primary_utility=("mean_primary_utility", "mean"),
        mean_baseline_utility=("mean_baseline_utility", "mean"),
        mean_delta_utility=("mean_delta_utility", "mean"),
        corrected=("corrected", "sum"),
        lost_correct=("lost_correct", "sum"),
        wrong_rescued=("wrong_rescued", "sum"),
        introduced_wrong=("introduced_wrong", "sum"),
    ).reset_index()
    lofo_rows: list[dict[str, object]] = []
    for baseline, comparison in family.groupby("baseline_method", sort=True):
        for omitted in comparison.source_family:
            retained = comparison.loc[comparison.source_family.ne(omitted)]
            lofo_rows.append({
                "baseline_method": baseline,
                "omitted_source_family": omitted,
                "remaining_families": int(len(retained)),
                "family_equal_mean_delta": (
                    float(retained.mean_delta_utility.mean())
                    if len(retained) else np.nan
                ),
            })
    return source, family, pd.DataFrame(lofo_rows)


def evaluate_ranking_method_increment(
    primary_raw: pd.DataFrame,
    greedy_raw: pd.DataFrame,
    centroid_raw: pd.DataFrame,
    *,
    monte_carlo_draws: int = 99999,
    seed: int = 20261019,
    alpha: float = 0.05,
) -> RankingIncrementEvaluation:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    tables, role = align_method_outcomes(primary_raw, greedy_raw, centroid_raw)
    ledger = build_query_axis_ledger(tables)
    source, family, lofo = hierarchical_summaries(ledger)
    family_count = int(ledger.source_family.nunique())
    source_count = int(ledger.query_source.nunique())
    required_positive = math.ceil(0.75 * family_count)
    denominator_gates = {
        "at_least_four_biological_families": family_count >= 4,
        "at_least_eight_held_out_query_sources": source_count >= 8,
    }

    comparisons: dict[str, object] = {}
    distributions: list[pd.DataFrame] = []
    for method_index, baseline in enumerate(BASELINE_METHODS):
        baseline_source = source.loc[source.baseline_method.eq(baseline)].copy()
        baseline_family = family.loc[family.baseline_method.eq(baseline)].copy()
        baseline_lofo = lofo.loc[lofo.baseline_method.eq(baseline)].copy()
        sign_flip, distribution = source_sign_flip(
            baseline_source,
            monte_carlo_draws=monte_carlo_draws,
            seed=seed + method_index,
            alternative="greater",
        )
        distribution.insert(0, "baseline_method", baseline)
        distributions.append(distribution)
        positive_families = int((baseline_family.mean_delta_utility > 0).sum())
        nonharmful_families = int((baseline_family.mean_delta_utility >= 0).sum())
        overall_delta = float(baseline_family.mean_delta_utility.mean())
        lofo_positive = bool(
            len(baseline_lofo)
            and (baseline_lofo.family_equal_mean_delta > 0).all()
        )
        finite_lofo = baseline_lofo.loc[
            np.isfinite(baseline_lofo.family_equal_mean_delta),
            "family_equal_mean_delta",
        ]
        component_gates = {
            "at_least_three_quarters_families_strictly_positive": (
                positive_families >= required_positive
            ),
            "every_family_nonharmful": nonharmful_families == family_count,
            "every_leave_one_family_out_mean_strictly_positive": lofo_positive,
            "family_equal_mean_delta_strictly_positive": overall_delta > 0,
            "source_sign_flip_one_sided_p_le_alpha": (
                float(sign_flip["p_value"]) <= alpha
            ),
        }
        component_pass = all(denominator_gates.values()) and all(component_gates.values())
        comparisons[baseline] = {
            "baseline_method": baseline,
            "family_equal_endpoint": {
                "mean_primary_utility": float(
                    baseline_family.mean_primary_utility.mean()
                ),
                "mean_baseline_utility": float(
                    baseline_family.mean_baseline_utility.mean()
                ),
                "mean_delta_utility": overall_delta,
            },
            "denominators": {
                "positive_families": positive_families,
                "required_positive_families": required_positive,
                "nonharmful_families": nonharmful_families,
            },
            "transitions": {
                name: int(baseline_family[name].sum())
                for name in (
                    "corrected", "lost_correct", "wrong_rescued", "introduced_wrong",
                )
            },
            "source_sign_flip": sign_flip,
            "leave_one_family_out_minimum_delta": (
                float(finite_lofo.min()) if len(finite_lofo) else None
            ),
            "gates": component_gates,
            "pass_component_gate": component_pass,
        }

    iut_pass = all(
        bool(comparisons[baseline]["pass_component_gate"])  # type: ignore[index]
        for baseline in BASELINE_METHODS
    )
    gates = {
        **denominator_gates,
        "transport_beats_greedy_component": bool(
            comparisons["greedy_same_graph"]["pass_component_gate"]  # type: ignore[index]
        ),
        "transport_beats_centroid_component": bool(
            comparisons["centroid_same_encoder"]["pass_component_gate"]  # type: ignore[index]
        ),
        "intersection_union_both_fixed_baselines_pass": iut_pass,
    }
    report: dict[str, object] = {
        "schema": "contrast_atlas_ranking_method_increment_v1",
        "status": f"CONTRAST_ATLAS_RANKING_METHOD_INCREMENT_EVALUATED_{role.upper()}",
        "evaluation_role": role,
        "world": ALL_WORLD,
        "primary_method": PRIMARY_METHOD,
        "fixed_baseline_methods": list(BASELINE_METHODS),
        "risk_utility": "whole_query_correct - 2 * whole_query_wrong; abstention = 0",
        "denominators": {
            "query_source_axes": int(len(ledger)),
            "held_out_query_sources": source_count,
            "biological_families": family_count,
                "axes_per_source_distribution": {
                    str(int(axes)): int(count)
                    for axes, count in ledger.groupby("query_source")["axis"].nunique(
                    ).value_counts().sort_index().items()
                },
        },
        "comparisons": comparisons,
        "multiplicity": {
            "method": "intersection_union",
            "composite_null": (
                "transport_conservative fails to outperform at least one fixed baseline"
            ),
            "rule": (
                "both predeclared one-sided source-level component gates must pass at alpha; "
                "no baseline is selected from observed outcomes"
            ),
            "alpha_per_component": alpha,
            "alpha_splitting_required": False,
        },
        "gates": gates,
        "pass_formal_ranking_method_increment_gate": all(gates.values()),
        "invariant_contract": (
            "all methods contain identical all-world query_source/axis rows, query objects, "
            "biological and provenance families, expected directions, score-blind candidate-"
            "universe IDs/hashes/sizes, formal truth-registry query hashes, frozen reference "
            "atlas, shared primary-answerability-mask hashes, held-out-source attestation "
            "and evaluation role"
        ),
        "abstention_contract": (
            "the evaluator consumes each builder-recorded correct/wrong/abstained outcome "
            "without a score threshold, score access, candidate filtering or answerability "
            "redefinition; answerable_candidates is reported but is not an evaluator gate"
        ),
        "aggregation_contract": (
            "utility is averaged across axes within query source, query sources within "
            "biological family, then equally across biological families"
        ),
        "inference_boundary": (
            "the source sign flip is conditional on the frozen reference atlas and requires "
            "independent held-out query sources plus paired primary-versus-baseline source-"
            "delta sign symmetry under each component null; passing does not validate the "
            "biological adjudication or establish mechanism"
        ),
    }
    return RankingIncrementEvaluation(
        query_axis_ledger=ledger,
        source_summary=source,
        family_summary=family,
        leave_one_family_out=lofo,
        sign_flip_distribution=pd.concat(distributions, ignore_index=True),
        report=report,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport-outcomes", type=Path, required=True)
    parser.add_argument("--greedy-outcomes", type=Path, required=True)
    parser.add_argument("--centroid-outcomes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--monte-carlo-draws", type=int, default=99999)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    paths = {
        PRIMARY_METHOD: args.transport_outcomes,
        "greedy_same_graph": args.greedy_outcomes,
        "centroid_same_encoder": args.centroid_outcomes,
    }
    tables = {
        method: pd.read_csv(path, sep="\t", dtype=str)
        for method, path in paths.items()
    }
    result = evaluate_ranking_method_increment(
        tables[PRIMARY_METHOD], tables["greedy_same_graph"],
        tables["centroid_same_encoder"],
        monte_carlo_draws=args.monte_carlo_draws,
        seed=args.seed,
        alpha=args.alpha,
    )
    args.output_dir.mkdir(parents=True)
    outputs = {
        "query_axis_ledger": args.output_dir / "query_axis_increment.tsv",
        "source_summary": args.output_dir / "source_increment.tsv",
        "family_summary": args.output_dir / "family_increment.tsv",
        "leave_one_family_out": args.output_dir / "leave_one_family_out.tsv",
        "sign_flip_distribution": args.output_dir / "source_sign_flip_distribution.tsv.gz",
    }
    result.query_axis_ledger.to_csv(outputs["query_axis_ledger"], sep="\t", index=False)
    result.source_summary.to_csv(outputs["source_summary"], sep="\t", index=False)
    result.family_summary.to_csv(outputs["family_summary"], sep="\t", index=False)
    result.leave_one_family_out.to_csv(
        outputs["leave_one_family_out"], sep="\t", index=False,
    )
    result.sign_flip_distribution.to_csv(
        outputs["sign_flip_distribution"], sep="\t", index=False,
        compression="gzip",
    )
    report = dict(result.report)
    report["provenance"] = {
        **{f"{method}_outcomes_sha256": sha256(path) for method, path in paths.items()},
        **{f"{name}_sha256": sha256(path) for name, path in outputs.items()},
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
