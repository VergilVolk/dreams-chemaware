#!/usr/bin/env python3
"""Source- and family-equal risk gate for exact-axis Contrast Atlas outcomes.

The input is already reduced to one whole-query outcome per
``query_source x axis x world``.  This module never treats axes or queries as
independent replicates: axes are averaged within source, sources within family,
and families receive equal weight.  Inference uses one sign per independent
query source, with a required one-to-one mapping from nominal source ID to
provenance family.
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


ALL_WORLD = "all"
KNOWN_WORLDS = ("known_only", "measurement_matched_known")
REQUIRED_WORLDS = (ALL_WORLD, *KNOWN_WORLDS)
VALID_ROLES = {"development", "external"}
BOOLEAN_COLUMNS = (
    "whole_query_correct",
    "whole_query_wrong",
    "whole_query_answered",
    "candidate_universe_frozen",
    "reference_atlas_frozen",
    "query_reference_source_disjoint",
)
KEY_COLUMNS = ("query_source", "axis", "world")
TRANSITIONS = ("corrected", "lost_correct", "wrong_rescued", "introduced_wrong")


@dataclass
class SourceRiskEvaluation:
    report: dict[str, object]
    query_ledger: pd.DataFrame
    source_summary: pd.DataFrame
    family_summary: pd.DataFrame
    leave_one_family_out: pd.DataFrame
    sign_flip_distribution: pd.DataFrame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)


def parse_bool(series: pd.Series, label: str) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not values.isin(mapping).all():
        invalid = sorted(set(values) - set(mapping))
        raise ValueError(f"{label} contains invalid booleans: {invalid[:5]}")
    return values.map(mapping).astype(bool)


def validate_outcomes(raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    required = {
        "query_source", "query_provenance_family", "source_family", "axis",
        "world", "evaluation_role",
        "whole_query_correct", "whole_query_wrong", "whole_query_answered",
        "candidate_universe_id", "candidate_universe_size",
        "candidate_universe_frozen",
        "reference_atlas_id", "reference_atlas_frozen",
        "query_reference_source_disjoint",
    }
    if missing := required - set(raw):
        raise ValueError(f"source-risk outcomes lack columns: {sorted(missing)}")
    frame = raw.copy()
    for column in (
        "query_source", "query_provenance_family", "source_family", "axis",
        "world", "evaluation_role",
        "candidate_universe_id",
        "reference_atlas_id",
    ):
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"{column} contains an empty value")
    for column in BOOLEAN_COLUMNS:
        frame[column] = parse_bool(frame[column], column)
    frame["candidate_universe_size"] = pd.to_numeric(
        frame.candidate_universe_size, errors="raise",
    ).astype(int)
    if (frame.candidate_universe_size <= 0).any():
        raise ValueError("candidate_universe_size must be positive")
    if not frame.candidate_universe_frozen.all():
        raise ValueError("every candidate universe must be explicitly frozen")
    if not frame.reference_atlas_frozen.all():
        raise ValueError("the formal reference atlas must be frozen before query scoring")
    if not frame.query_reference_source_disjoint.all():
        raise ValueError("formal source inference requires globally disjoint query/reference roles")
    if frame.reference_atlas_id.nunique() != 1:
        raise ValueError("all formal query outcomes must use one frozen reference atlas")
    roles = set(frame.evaluation_role)
    if len(roles) != 1 or not roles.issubset(VALID_ROLES):
        raise ValueError("evaluation_role must be one invariant value: development or external")
    role = next(iter(roles))
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("duplicate query_source/axis/world outcome")
    if set(frame.world) != set(REQUIRED_WORLDS):
        raise ValueError(f"worlds must be exactly {list(REQUIRED_WORLDS)}")
    grouped_worlds = frame.groupby(["query_source", "axis"], sort=False).world.agg(
        lambda values: frozenset(values)
    )
    if not grouped_worlds.eq(frozenset(REQUIRED_WORLDS)).all():
        raise ValueError("every query_source/axis must contain all three required worlds")
    if (frame.whole_query_correct & frame.whole_query_wrong).any():
        raise ValueError("one whole query cannot be both correct and wrong")
    if not frame.whole_query_answered.equals(
        frame.whole_query_correct | frame.whole_query_wrong
    ):
        raise ValueError("answered must equal correct OR wrong for exact-axis labeled queries")

    source_family_counts = frame.groupby("query_source").source_family.nunique()
    if (source_family_counts != 1).any():
        raise ValueError("one query_source belongs to more than one source_family")
    provenance_counts = frame.groupby("query_source").query_provenance_family.nunique()
    if (provenance_counts != 1).any():
        raise ValueError("one query_source belongs to multiple provenance families")
    nominal_sources_per_provenance = frame.groupby(
        "query_provenance_family"
    ).query_source.nunique()
    if (nominal_sources_per_provenance != 1).any():
        raise ValueError(
            "one query provenance family maps to multiple nominal query sources"
        )
    source_role_counts = frame.groupby("query_source").evaluation_role.nunique()
    if (source_role_counts != 1).any():
        raise ValueError("evaluation_role changes within a query_source")
    source_axis_counts = frame[["query_source", "axis"]].drop_duplicates().groupby(
        "query_source"
    ).size()
    if source_axis_counts.empty or (source_axis_counts < 1).any():
        raise ValueError("every query_source must contribute at least one frozen axis")

    invariant = [
        "query_provenance_family", "source_family", "evaluation_role",
        "candidate_universe_size", "candidate_universe_frozen",
        "reference_atlas_id", "reference_atlas_frozen",
        "query_reference_source_disjoint",
    ]
    for column in invariant:
        counts = frame.groupby(["query_source", "axis"])[column].nunique()
        if (counts != 1).any():
            raise ValueError(f"{column} changes across worlds within a source axis")
    return frame.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True), role


def build_query_oracle(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index(["query_source", "axis", "world"])
    rows: list[dict[str, object]] = []
    for (source, axis), group in frame.groupby(["query_source", "axis"], sort=True):
        family = str(group.source_family.iloc[0])
        role = str(group.evaluation_role.iloc[0])

        def outcome(world: str) -> tuple[bool, bool, bool, int]:
            row = indexed.loc[(source, axis, world)]
            correct = bool(row.whole_query_correct)
            wrong = bool(row.whole_query_wrong)
            answered = bool(row.whole_query_answered)
            return correct, wrong, answered, int(correct) - 2 * int(wrong)

        all_correct, all_wrong, all_answered, all_utility = outcome(ALL_WORLD)
        known = {world: outcome(world) for world in KNOWN_WORLDS}
        oracle_utility = max(value[3] for value in known.values())
        # Utility values uniquely identify correct (+1), abstain (0), and
        # wrong (-2), so ties cannot create an ambiguous transition state.
        oracle_correct = oracle_utility == 1
        oracle_wrong = oracle_utility == -2
        oracle_answered = oracle_correct or oracle_wrong
        row: dict[str, object] = {
            "query_source": source,
            "query_provenance_family": str(
                group.query_provenance_family.iloc[0]
            ),
            "source_family": family,
            "axis": axis,
            "evaluation_role": role,
            "candidate_universe_id": str(group.candidate_universe_id.iloc[0]),
            "candidate_universe_size": int(group.candidate_universe_size.iloc[0]),
            "all_correct": all_correct,
            "all_wrong": all_wrong,
            "all_answered": all_answered,
            "all_utility": all_utility,
            "known_only_utility": known["known_only"][3],
            "measurement_matched_known_utility": known["measurement_matched_known"][3],
            "oracle_known_correct": oracle_correct,
            "oracle_known_wrong": oracle_wrong,
            "oracle_known_answered": oracle_answered,
            "oracle_known_utility": oracle_utility,
            "delta_utility": all_utility - oracle_utility,
            "corrected": bool(all_correct and not oracle_correct),
            "lost_correct": bool(oracle_correct and not all_correct),
            "wrong_rescued": bool(oracle_wrong and not all_wrong),
            "introduced_wrong": bool(all_wrong and not oracle_wrong),
        }
        for world, (known_correct, known_wrong, known_answered, known_utility) in known.items():
            row.update({
                f"delta_utility__{world}": all_utility - known_utility,
                f"corrected__{world}": bool(all_correct and not known_correct),
                f"lost_correct__{world}": bool(known_correct and not all_correct),
                f"wrong_rescued__{world}": bool(known_wrong and not all_wrong),
                f"introduced_wrong__{world}": bool(all_wrong and not known_wrong),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def hierarchical_summaries(
    ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return long-form summaries for each pre-registered known comparator."""
    source_pieces: list[pd.DataFrame] = []
    for comparator in KNOWN_WORLDS:
        aggregation: dict[str, tuple[str, str]] = {
            "axes": ("axis", "nunique"),
            "mean_all_utility": ("all_utility", "mean"),
            "mean_known_utility": (f"{comparator}_utility", "mean"),
            "mean_delta_utility": (f"delta_utility__{comparator}", "mean"),
        }
        for transition in TRANSITIONS:
            aggregation[f"{transition}_rate"] = (f"{transition}__{comparator}", "mean")
            aggregation[transition] = (f"{transition}__{comparator}", "sum")
        piece = ledger.groupby(
            ["query_source", "query_provenance_family", "source_family", "evaluation_role"],
            sort=True,
        ).agg(**aggregation).reset_index()
        piece.insert(0, "comparator_world", comparator)
        source_pieces.append(piece)
    source = pd.concat(source_pieces, ignore_index=True)
    family = source.groupby(
        ["comparator_world", "source_family", "evaluation_role"], sort=True,
    ).agg(
        sources=("query_source", "nunique"),
        mean_all_utility=("mean_all_utility", "mean"),
        mean_known_utility=("mean_known_utility", "mean"),
        mean_delta_utility=("mean_delta_utility", "mean"),
        corrected_rate=("corrected_rate", "mean"),
        lost_correct_rate=("lost_correct_rate", "mean"),
        wrong_rescued_rate=("wrong_rescued_rate", "mean"),
        introduced_wrong_rate=("introduced_wrong_rate", "mean"),
        corrected=("corrected", "sum"),
        lost_correct=("lost_correct", "sum"),
        wrong_rescued=("wrong_rescued", "sum"),
        introduced_wrong=("introduced_wrong", "sum"),
    ).reset_index()
    lofo_rows = []
    for comparator, comparison in family.groupby("comparator_world", sort=True):
        for omitted in comparison.source_family:
            retained = comparison.loc[comparison.source_family.ne(omitted)]
            lofo_rows.append({
                "comparator_world": comparator,
                "omitted_source_family": omitted,
                "remaining_families": int(len(retained)),
                "family_equal_mean_delta": (
                    float(retained.mean_delta_utility.mean())
                    if len(retained) else float("nan")
                ),
            })
    return source, family, pd.DataFrame(lofo_rows)


def source_sign_flip(
    source_summary: pd.DataFrame,
    *,
    monte_carlo_draws: int,
    seed: int,
    alternative: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    if alternative not in {"greater", "two-sided"}:
        raise ValueError("alternative must be greater or two-sided")
    if monte_carlo_draws < 1:
        raise ValueError("monte_carlo_draws must be positive")
    ordered = source_summary.sort_values("query_source", kind="stable").reset_index(drop=True)
    if ordered.query_source.duplicated().any():
        raise ValueError(
            "sign-flip input must contain one row per query_source; "
            "evaluate each pre-registered comparator separately"
        )
    if "comparator_world" in ordered and ordered.comparator_world.nunique() != 1:
        raise ValueError("sign-flip input mixes multiple comparator worlds")
    source_count = len(ordered)
    if source_count == 0:
        raise ValueError("sign-flip requires at least one source")
    family_size = ordered.groupby("source_family").query_source.transform("size").to_numpy(float)
    family_count = int(ordered.source_family.nunique())
    delta = ordered.mean_delta_utility.to_numpy(float)
    weights = 1.0 / (family_count * family_size)
    observed = float(np.dot(weights, delta))

    if source_count <= 20:
        draws = 1 << source_count
        statistics = np.empty(draws, dtype=float)
        bit_positions = np.arange(source_count, dtype=np.uint64)
        chunk = 1 << 15
        for start in range(0, draws, chunk):
            stop = min(start + chunk, draws)
            masks = np.arange(start, stop, dtype=np.uint64)[:, None]
            signs = 2.0 * ((masks >> bit_positions) & 1).astype(float) - 1.0
            statistics[start:stop] = signs @ (weights * delta)
        method = "exact_enumeration"
        denominator = draws
        if alternative == "greater":
            exceedances = int(np.sum(statistics >= observed - 1e-15))
        else:
            exceedances = int(np.sum(np.abs(statistics) >= abs(observed) - 1e-15))
        p_value = exceedances / denominator
        correction = "none_exact_reference_includes_observed_sign_vector"
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(monte_carlo_draws, source_count))
        statistics = signs @ (weights * delta)
        method = "monte_carlo"
        denominator = monte_carlo_draws + 1
        if alternative == "greater":
            exceedances = int(np.sum(statistics >= observed - 1e-15))
        else:
            exceedances = int(np.sum(np.abs(statistics) >= abs(observed) - 1e-15))
        p_value = (1 + exceedances) / denominator
        correction = "plus_one"
    distribution = pd.DataFrame({
        "sign_flip_id": np.arange(len(statistics), dtype=int),
        "family_equal_mean_delta": statistics,
    })
    report = {
        "method": method,
        "alternative": alternative,
        "sources": source_count,
        "source_families": family_count,
        "observed_family_equal_mean_delta": observed,
        "random_sign_vectors": int(len(statistics)),
        "exceedances": exceedances,
        "p_value": float(p_value),
        "denominator": int(denominator),
        "monte_carlo_correction": correction,
        "seed": seed if method == "monte_carlo" else None,
        "null_quantiles": {
            "q025": float(np.quantile(statistics, 0.025)),
            "median": float(np.median(statistics)),
            "q975": float(np.quantile(statistics, 0.975)),
        },
        "resampling_unit": "one shared sign per query_source across every axis",
        "weighting": "axis-within-source, source-within-family, then equal family weight",
    }
    return report, distribution


def evaluate_source_risk(
    raw: pd.DataFrame,
    *,
    monte_carlo_draws: int = 99999,
    seed: int = 20261018,
    alternative: str = "greater",
    alpha: float = 0.05,
) -> SourceRiskEvaluation:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    frame, role = validate_outcomes(raw)
    ledger = build_query_oracle(frame)
    source, family, lofo = hierarchical_summaries(ledger)
    family_count = int(ledger.source_family.nunique())
    source_count = int(ledger.query_source.nunique())
    required_positive = math.ceil(0.75 * family_count)
    denominator_gates = {
        "at_least_four_source_families": family_count >= 4,
        "at_least_eight_independent_query_sources": source_count >= 8,
        "query_sources_are_unique_provenance_families": (
            ledger.query_provenance_family.nunique() == source_count
        ),
    }

    comparisons: dict[str, object] = {}
    distribution_pieces: list[pd.DataFrame] = []
    for comparator in KNOWN_WORLDS:
        comparator_source = source.loc[
            source.comparator_world.eq(comparator)
        ].copy()
        comparator_family = family.loc[
            family.comparator_world.eq(comparator)
        ].copy()
        comparator_lofo = lofo.loc[
            lofo.comparator_world.eq(comparator)
        ].copy()
        sign_flip, comparator_distribution = source_sign_flip(
            comparator_source,
            monte_carlo_draws=monte_carlo_draws,
            seed=seed,
            alternative=alternative,
        )
        comparator_distribution.insert(0, "comparator_world", comparator)
        distribution_pieces.append(comparator_distribution)

        positive_families = int((comparator_family.mean_delta_utility > 0).sum())
        nonharmful_families = int((comparator_family.mean_delta_utility >= 0).sum())
        overall_delta = float(comparator_family.mean_delta_utility.mean())
        lofo_positive = bool(
            len(comparator_lofo)
            and (comparator_lofo.family_equal_mean_delta > 0).all()
        )
        comparison_gates = {
            "at_least_three_quarters_families_strictly_positive": (
                positive_families >= required_positive
            ),
            "every_remaining_family_nonharmful": (
                nonharmful_families == family_count
            ),
            "every_leave_one_family_out_mean_strictly_positive": lofo_positive,
            "family_equal_mean_delta_strictly_positive": overall_delta > 0,
            "source_sign_flip_one_sided_p_le_alpha": (
                alternative == "greater" and float(sign_flip["p_value"]) <= alpha
            ),
        }
        comparisons[comparator] = {
            "comparator_world": comparator,
            "family_equal_endpoint": {
                "mean_all_utility": float(comparator_family.mean_all_utility.mean()),
                "mean_known_utility": float(
                    comparator_family.mean_known_utility.mean()
                ),
                "mean_delta_utility": overall_delta,
            },
            "transitions": {
                "raw_source_axis_counts": {
                    name: int(ledger[f"{name}__{comparator}"].sum())
                    for name in TRANSITIONS
                },
                "family_equal_rates": {
                    name: float(comparator_family[f"{name}_rate"].mean())
                    for name in TRANSITIONS
                },
            },
            "denominators": {
                "positive_families": positive_families,
                "required_positive_families": required_positive,
                "nonharmful_families": nonharmful_families,
            },
            "source_sign_flip": sign_flip,
            "leave_one_family_out_minimum_delta": (
                float(comparator_lofo.family_equal_mean_delta.min())
                if len(comparator_lofo) else None
            ),
            "gates": comparison_gates,
            "pass_comparator_gate": (
                all(denominator_gates.values()) and all(comparison_gates.values())
            ),
        }

    distribution = pd.concat(distribution_pieces, ignore_index=True)

    # The per-query max uses observed truth utilities.  It is intentionally
    # retained only as a conservative descriptive envelope: swapping ``all``
    # with either fixed comparator does not negate all-max(comparator_1,
    # comparator_2), so an oracle-selected delta has no paired sign-flip null.
    oracle_source = ledger.groupby(
        ["query_source", "source_family"], sort=True,
    ).agg(
        mean_all_utility=("all_utility", "mean"),
        mean_oracle_known_utility=("oracle_known_utility", "mean"),
        mean_delta_utility=("delta_utility", "mean"),
        **{
            f"{name}_rate": (name, "mean")
            for name in TRANSITIONS
        },
    ).reset_index()
    oracle_family = oracle_source.groupby("source_family", sort=True).agg(
        mean_all_utility=("mean_all_utility", "mean"),
        mean_oracle_known_utility=("mean_oracle_known_utility", "mean"),
        mean_delta_utility=("mean_delta_utility", "mean"),
        **{
            f"{name}_rate": (f"{name}_rate", "mean")
            for name in TRANSITIONS
        },
    ).reset_index()
    oracle_descriptive = {
        "status": "DESCRIPTIVE_ONLY_NOT_AN_INFERENTIAL_COMPARATOR",
        "definition": (
            "per-source-axis maximum observed utility across known_only and "
            "measurement_matched_known"
        ),
        "family_equal_endpoint": {
            "mean_all_utility": float(oracle_family.mean_all_utility.mean()),
            "mean_oracle_known_utility": float(
                oracle_family.mean_oracle_known_utility.mean()
            ),
            "mean_delta_utility": float(oracle_family.mean_delta_utility.mean()),
        },
        "transitions": {
            "raw_source_axis_counts": {
                name: int(ledger[name].sum()) for name in TRANSITIONS
            },
            "family_equal_rates": {
                name: float(oracle_family[f"{name}_rate"].mean())
                for name in TRANSITIONS
            },
        },
        "inference_prohibited": (
            "oracle selection depends on observed truth utility and destroys the "
            "sign symmetry required by a paired source sign-flip"
        ),
    }

    both_comparators_pass = all(
        bool(comparisons[world]["pass_comparator_gate"])  # type: ignore[index]
        for world in KNOWN_WORLDS
    )
    gates = {
        **denominator_gates,
        "known_only_comparator_gate_pass": bool(
            comparisons["known_only"]["pass_comparator_gate"]  # type: ignore[index]
        ),
        "measurement_matched_known_comparator_gate_pass": bool(
            comparisons["measurement_matched_known"]["pass_comparator_gate"]  # type: ignore[index]
        ),
        "intersection_union_both_comparators_pass": both_comparators_pass,
    }
    report: dict[str, object] = {
        "schema": "contrast_atlas_source_risk_exact_axis_v2",
        "status": f"CONTRAST_ATLAS_SOURCE_RISK_EVALUATED_{role.upper()}",
        "evaluation_role": role,
        "worlds": list(REQUIRED_WORLDS),
        "risk_utility": "whole_query_correct - 2 * whole_query_wrong; abstention = 0",
        "pre_registered_comparators": list(KNOWN_WORLDS),
        "denominators": {
            "query_source_axes": int(len(ledger)),
            "query_sources": source_count,
            "source_families": family_count,
            "axes_per_source_distribution": {
                str(int(axis_count)): int(frequency)
                for axis_count, frequency in ledger.groupby("query_source")["axis"].nunique(
                ).value_counts().sort_index().items()
            },
        },
        "comparisons": comparisons,
        "multiplicity": {
            "method": "intersection_union",
            "composite_null": (
                "all fails to outperform at least one pre-registered known comparator"
            ),
            "rule": (
                "both comparator-specific one-sided source gates must pass at alpha; "
                "no comparator is selected from observed outcomes"
            ),
            "alpha_per_comparison": alpha,
            "familywise_note": (
                "requiring rejection of every component null is an intersection-union "
                "test and does not require alpha splitting"
            ),
        },
        "oracle_envelope_descriptive": oracle_descriptive,
        "gates": gates,
        "pass_source_level_exact_axis_gate": all(gates.values()),
        "candidate_universe_contract": (
            "candidate_universe_id and size are invariant across all three worlds within "
            "each source axis; every row declares candidate_universe_frozen=true"
        ),
        "reference_atlas_contract": (
            "all rows share one score-blind frozen reference_atlas_id and attest that held-out "
            "query sources and provenance families are globally absent from reference roles"
        ),
        "reference_atlas_id": str(frame.reference_atlas_id.iloc[0]),
        "aggregation_contract": (
            "each actual query_source/axis has all three worlds; available pre-registered "
            "axes are averaged equally within query_source, query_sources are averaged "
            "within source_family, and source_families receive equal weight; different "
            "sources need not share an axis vocabulary"
        ),
        "inference_boundary": (
            f"role={role} is copied from the frozen input and is never upgraded by this gate; "
            "each exact enumeration covers all query-source sign assignments for one fixed "
            "comparator conditional on the frozen reference atlas, while randomization validity "
            "still requires independent query sources and paired-label exchangeability "
            "(equivalently query-source-delta sign symmetry) under that component null"
        ),
    }
    return SourceRiskEvaluation(report, ledger, source, family, lofo, distribution)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--monte-carlo-draws", type=int, default=99999)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--alternative", choices=("greater", "two-sided"), default="greater")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    result = evaluate_source_risk(
        read_table(args.outcomes),
        monte_carlo_draws=args.monte_carlo_draws,
        seed=args.seed,
        alternative=args.alternative,
        alpha=args.alpha,
    )
    args.output_dir.mkdir(parents=True)
    result.query_ledger.to_csv(
        args.output_dir / "source_axis_oracle_ledger.tsv", sep="\t", index=False,
    )
    result.source_summary.to_csv(
        args.output_dir / "source_summary.tsv", sep="\t", index=False,
    )
    result.family_summary.to_csv(
        args.output_dir / "family_summary.tsv", sep="\t", index=False,
    )
    result.leave_one_family_out.to_csv(
        args.output_dir / "leave_one_family_out.tsv", sep="\t", index=False,
    )
    result.sign_flip_distribution.to_csv(
        args.output_dir / "source_sign_flip_distribution.tsv.gz",
        sep="\t", index=False, compression="gzip",
    )
    result.report["provenance"] = {"outcomes_sha256": sha256(args.outcomes)}
    (args.output_dir / "report.json").write_text(
        json.dumps(result.report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": result.report["status"],
        "evaluation_role": result.report["evaluation_role"],
        "pass_source_level_exact_axis_gate": result.report[
            "pass_source_level_exact_axis_gate"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
