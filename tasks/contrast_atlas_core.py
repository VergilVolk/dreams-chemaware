#!/usr/bin/env python
"""Minimal, fail-closed Metabolomic Contrast Atlas retrieval core.

This is not a website and does not annotate structures.  It converts frozen
within-study entity effects into calibrated pattern responses, then reports
strict-entity and shared-pattern retrieval separately.  Evidence tables are
required to be phenotype blind; insufficient overlap returns abstention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY_STATES = {"confirmed_known", "library_supported", "unresolved_entity"}
FORBIDDEN_EVIDENCE_COLUMNS = {
    "beta", "se", "effect", "effect_size", "p_value", "phenotype",
    "outcome", "group_label", "case_control", "validation_label",
}


def parse_bool(series: pd.Series, label: str) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not values.isin(mapping).all():
        raise RuntimeError(f"{label} contains invalid booleans")
    return values.map(mapping).astype(bool)


def validate_effects(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "study_id", "source_family", "contrast_id", "entity_id",
        "independent_unit_id", "identity_status", "beta", "se",
        "qc_pass", "detectable", "data_role",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"contrast records missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["qc_pass"] = parse_bool(frame["qc_pass"], "qc_pass")
    frame["detectable"] = parse_bool(frame["detectable"], "detectable")
    frame["beta"] = pd.to_numeric(frame["beta"], errors="raise")
    frame["se"] = pd.to_numeric(frame["se"], errors="raise")
    if (~np.isfinite(frame[["beta", "se"]].to_numpy(float))).any() or (frame["se"] <= 0).any():
        raise RuntimeError("contrast records require finite beta and positive se")
    if not set(frame["identity_status"].astype(str)).issubset(IDENTITY_STATES):
        raise RuntimeError("unrecognized identity_status")
    key = ["study_id", "contrast_id", "entity_id"]
    if frame.duplicated(key).any():
        raise RuntimeError("duplicate entity within a study contrast")
    unit_key = ["study_id", "contrast_id", "independent_unit_id"]
    if frame.loc[frame["qc_pass"] & frame["detectable"]].duplicated(unit_key).any():
        raise RuntimeError("multiple entities reuse one independent unit")
    source_counts = frame.groupby(["study_id", "contrast_id"])["source_family"].nunique()
    if (source_counts != 1).any():
        raise RuntimeError("each study contrast must belong to exactly one source_family")
    return frame.loc[frame["qc_pass"] & frame["detectable"]].copy()


def validate_evidence(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    forbidden = FORBIDDEN_EVIDENCE_COLUMNS & set(frame.columns)
    if forbidden:
        raise RuntimeError(f"{kind} contains phenotype/effect columns: {sorted(forbidden)}")
    frame = frame.copy()
    if "phenotype_blind" not in frame:
        raise RuntimeError(f"{kind} missing phenotype_blind")
    frame["phenotype_blind"] = parse_bool(frame["phenotype_blind"], f"{kind}.phenotype_blind")
    if not frame["phenotype_blind"].all():
        raise RuntimeError(f"{kind} must be entirely phenotype blind")
    return frame


def validate_loadings(frame: pd.DataFrame) -> pd.DataFrame:
    frame = validate_evidence(frame, "pattern loadings")
    required = {"study_id", "contrast_id", "entity_id", "pattern_id", "weight", "calibrated"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"pattern loadings missing columns: {sorted(missing)}")
    frame["calibrated"] = parse_bool(frame["calibrated"], "calibrated")
    frame["weight"] = pd.to_numeric(frame["weight"], errors="raise")
    if ((frame["weight"] <= 0) | (frame["weight"] > 1)).any():
        raise RuntimeError("pattern weights must lie in (0,1]")
    if frame.duplicated(["study_id", "contrast_id", "entity_id", "pattern_id"]).any():
        raise RuntimeError("duplicate pattern loading")
    return frame.loc[frame["calibrated"]].copy()


def validate_links(frame: pd.DataFrame) -> pd.DataFrame:
    frame = validate_evidence(frame, "strict entity links")
    required = {
        "query_study_id", "query_contrast_id", "query_entity_id",
        "target_study_id", "target_contrast_id", "target_entity_id",
        "match_probability", "claim_level",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"strict entity links missing columns: {sorted(missing)}")
    if not frame["claim_level"].eq("strict_entity").all():
        raise RuntimeError("link table may contain only strict_entity claims")
    frame["match_probability"] = pd.to_numeric(frame["match_probability"], errors="raise")
    if ((frame["match_probability"] < 0) | (frame["match_probability"] > 1)).any():
        raise RuntimeError("match_probability must lie in [0,1]")
    keys = [
        "query_study_id", "query_contrast_id", "query_entity_id",
        "target_study_id", "target_contrast_id", "target_entity_id",
    ]
    if frame.duplicated(keys).any():
        raise RuntimeError("duplicate strict entity link")
    query_side = [
        "query_study_id", "query_contrast_id", "query_entity_id",
        "target_study_id", "target_contrast_id",
    ]
    target_side = [
        "query_study_id", "query_contrast_id",
        "target_study_id", "target_contrast_id", "target_entity_id",
    ]
    if frame.duplicated(query_side).any() or frame.duplicated(target_side).any():
        raise RuntimeError("strict entity links must be one-to-one within each contrast pair")
    return frame


def validate_link_endpoints(links: pd.DataFrame, effects: pd.DataFrame) -> None:
    endpoints = set(zip(effects["study_id"], effects["contrast_id"], effects["entity_id"]))
    query_endpoints = set(zip(
        links["query_study_id"], links["query_contrast_id"], links["query_entity_id"],
    ))
    target_endpoints = set(zip(
        links["target_study_id"], links["target_contrast_id"], links["target_entity_id"],
    ))
    missing = (query_endpoints | target_endpoints) - endpoints
    if missing:
        example = sorted(missing)[:3]
        raise RuntimeError(f"strict entity links contain absent or QC-ineligible endpoints: {example}")


def validate_loading_endpoints(loadings: pd.DataFrame, effects: pd.DataFrame) -> None:
    endpoints = set(zip(effects["study_id"], effects["contrast_id"], effects["entity_id"]))
    loading_endpoints = set(zip(loadings["study_id"], loadings["contrast_id"], loadings["entity_id"]))
    missing = loading_endpoints - endpoints
    if missing:
        example = sorted(missing)[:3]
        raise RuntimeError(f"pattern loadings contain absent or QC-ineligible endpoints: {example}")


def subset_world(effects: pd.DataFrame, world: str) -> pd.DataFrame:
    if world == "all":
        return effects
    if world == "known_only":
        return effects.loc[effects["identity_status"].isin({"confirmed_known", "library_supported"})]
    if world == "unresolved_only":
        return effects.loc[effects["identity_status"].eq("unresolved_entity")]
    raise ValueError(world)


PATTERN_RESPONSE_COLUMNS = [
    "study_id", "source_family", "contrast_id", "pattern_id",
    "pattern_beta", "pattern_se", "uncertainty_model",
    "n_independent_units", "n_biological_units", "weight_sum",
]


def validate_unit_effects(frame: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    required = {
        "study_id", "contrast_id", "biological_unit_id", "entity_id", "effect",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"contrast unit effects missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["effect"] = pd.to_numeric(frame["effect"], errors="raise")
    if (~np.isfinite(frame["effect"].to_numpy(float))).any():
        raise RuntimeError("contrast unit effects must be finite")
    key = ["study_id", "contrast_id", "biological_unit_id", "entity_id"]
    if frame.duplicated(key).any():
        raise RuntimeError("duplicate biological-unit entity effect")
    endpoints = set(zip(effects["study_id"], effects["contrast_id"], effects["entity_id"]))
    observed = set(zip(frame["study_id"], frame["contrast_id"], frame["entity_id"]))
    missing_endpoints = observed - endpoints
    if missing_endpoints:
        raise RuntimeError(f"unit effects contain absent or QC-ineligible endpoints: {sorted(missing_endpoints)[:3]}")
    for key_values, group in frame.groupby(["study_id", "contrast_id"], sort=False):
        expected_units: set[str] | None = None
        for _, entity_group in group.groupby("entity_id", sort=False):
            observed_units = set(entity_group["biological_unit_id"].astype(str))
            if expected_units is None:
                expected_units = observed_units
            elif observed_units != expected_units:
                raise RuntimeError(
                    "biological-unit effects must form a complete synchronized rectangle "
                    f"within study contrast {key_values}"
                )
        if expected_units is None or len(expected_units) < 2:
            raise RuntimeError("at least two synchronized biological units are required")
    summary = (
        frame.groupby(["study_id", "contrast_id", "entity_id"], as_index=False)["effect"]
        .mean().rename(columns={"effect": "unit_mean"})
        .merge(
            effects[["study_id", "contrast_id", "entity_id", "beta"]],
            on=["study_id", "contrast_id", "entity_id"], how="right", validate="one_to_one",
        )
    )
    if summary["unit_mean"].isna().any():
        raise RuntimeError("eligible entity lacks biological-unit effects")
    if not np.allclose(summary["unit_mean"], summary["beta"], rtol=1e-6, atol=1e-8):
        raise RuntimeError("biological-unit effect means disagree with contrast beta")
    return frame


def pattern_responses(
    effects: pd.DataFrame,
    loadings: pd.DataFrame,
    unit_effects: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if unit_effects is not None:
        source = effects[["study_id", "source_family", "contrast_id", "entity_id"]]
        joined = (
            unit_effects.merge(
                source, on=["study_id", "contrast_id", "entity_id"],
                how="inner", validate="many_to_one",
            )
            .merge(
                loadings, on=["study_id", "contrast_id", "entity_id"],
                how="inner", validate="many_to_many",
            )
        )
        if joined.empty:
            return pd.DataFrame(rows=[], columns=PATTERN_RESPONSE_COLUMNS)
        unit_rows = []
        unit_key = [
            "study_id", "source_family", "contrast_id", "pattern_id", "biological_unit_id",
        ]
        for key, group in joined.groupby(unit_key, sort=False):
            weight = group["weight"].to_numpy(float)
            unit_rows.append({
                **dict(zip(unit_key, key)),
                "unit_pattern_effect": float(np.sum(weight * group["effect"].to_numpy(float)) / weight.sum()),
                "n_entities": int(group["entity_id"].nunique()),
            })
        unit_frame = pd.DataFrame(unit_rows)
        rows = []
        pattern_key = ["study_id", "source_family", "contrast_id", "pattern_id"]
        for key, group in unit_frame.groupby(pattern_key, sort=False):
            values = group["unit_pattern_effect"].to_numpy(float)
            if len(values) < 2:
                continue
            rows.append({
                **dict(zip(pattern_key, key)),
                "pattern_beta": float(values.mean()),
                "pattern_se": float(values.std(ddof=1) / np.sqrt(len(values))),
                "uncertainty_model": "synchronized_biological_unit_empirical_se",
                "n_independent_units": int(group["n_entities"].max()),
                "n_biological_units": int(len(values)),
                "weight_sum": float(
                    loadings.loc[
                        (loadings["study_id"].eq(key[0]))
                        & (loadings["contrast_id"].eq(key[2]))
                        & (loadings["pattern_id"].eq(key[3])),
                        "weight",
                    ].sum()
                ),
            })
        return pd.DataFrame(rows, columns=PATTERN_RESPONSE_COLUMNS)
    joined = effects.merge(loadings, on=["study_id", "contrast_id", "entity_id"], how="inner")
    rows: list[dict[str, object]] = []
    for key, group in joined.groupby(["study_id", "source_family", "contrast_id", "pattern_id"], sort=False):
        weight = group["weight"].to_numpy(float)
        beta = group["beta"].to_numpy(float)
        se = group["se"].to_numpy(float)
        total = float(weight.sum())
        rows.append({
            "study_id": key[0], "source_family": key[1], "contrast_id": key[2], "pattern_id": key[3],
            "pattern_beta": float(np.sum(weight * beta) / total),
            # Marginal entity SEs do not identify their covariance because the
            # effects are estimated from the same biological samples.  The L1
            # bound corresponds to worst-case positive dependence and is the
            # only fail-closed summary available without synchronized sample-
            # level bootstrap draws.
            "pattern_se": float(np.sum(np.abs(weight) * se) / total),
            "uncertainty_model": "marginal_se_perfect_positive_dependence_upper_bound",
            "n_independent_units": int(group["independent_unit_id"].nunique()),
            "n_biological_units": 0,
            "weight_sum": total,
        })
    return pd.DataFrame(rows, columns=PATTERN_RESPONSE_COLUMNS)


def weighted_cosine(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    denominator = np.sqrt(np.sum(weight * x * x) * np.sum(weight * y * y))
    return float(np.sum(weight * x * y) / denominator) if denominator > 0 else float("nan")


def retrieve(
    effects_raw: pd.DataFrame,
    loadings_raw: pd.DataFrame,
    links_raw: pd.DataFrame,
    query_study: str,
    query_contrast: str,
    min_entities: int = 3,
    min_patterns: int = 2,
    min_coverage: float = 0.5,
    min_match_probability: float = 0.8,
    association_threshold: float = 0.3,
    allow_same_source: bool = False,
    universe_effects_raw: pd.DataFrame | None = None,
    identity_worlds: tuple[str, ...] = ("all", "known_only", "unresolved_only"),
    unit_effects_raw: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effects = validate_effects(effects_raw)
    universe_effects = validate_effects(universe_effects_raw) if universe_effects_raw is not None else effects
    loadings = validate_loadings(loadings_raw)
    links = validate_links(links_raw)
    validate_loading_endpoints(loadings, effects)
    validate_link_endpoints(links, effects)
    unit_effects = validate_unit_effects(unit_effects_raw, effects) if unit_effects_raw is not None else None
    if not identity_worlds or not set(identity_worlds).issubset({"all", "known_only", "unresolved_only"}):
        raise ValueError("identity_worlds must be a nonempty subset of all/known_only/unresolved_only")
    query_all = universe_effects.loc[
        (universe_effects["study_id"] == query_study)
        & (universe_effects["contrast_id"] == query_contrast)
    ]
    if query_all.empty:
        raise RuntimeError("query contrast absent from the frozen candidate universe")
    query_source = str(query_all["source_family"].iloc[0])
    target_keys = universe_effects[["study_id", "source_family", "contrast_id"]].drop_duplicates()
    target_keys = target_keys.loc[~((target_keys["study_id"] == query_study)
                                    & (target_keys["contrast_id"] == query_contrast))]
    if not allow_same_source:
        target_keys = target_keys.loc[target_keys["source_family"] != query_source]

    output: list[dict[str, object]] = []
    pattern_tables: list[pd.DataFrame] = []
    for world in identity_worlds:
        world_effects = subset_world(effects, world)
        patterns = pattern_responses(world_effects, loadings, unit_effects)
        if not patterns.empty:
            patterns["world"] = world
            pattern_tables.append(patterns)
        query_effects = world_effects.loc[(world_effects["study_id"] == query_study)
                                          & (world_effects["contrast_id"] == query_contrast)]
        query_patterns = patterns.loc[(patterns["study_id"] == query_study)
                                      & (patterns["contrast_id"] == query_contrast)] if not patterns.empty else patterns
        for target in target_keys.itertuples(index=False):
            target_effects = world_effects.loc[(world_effects["study_id"] == target.study_id)
                                               & (world_effects["contrast_id"] == target.contrast_id)]
            relevant = links.loc[
                (links["query_study_id"] == query_study)
                & (links["query_contrast_id"] == query_contrast)
                & (links["target_study_id"] == target.study_id)
                & (links["target_contrast_id"] == target.contrast_id)
                & (links["match_probability"] >= min_match_probability)
            ].copy()
            q_lookup = query_effects.set_index("entity_id")
            t_lookup = target_effects.set_index("entity_id")
            relevant = relevant.loc[relevant["query_entity_id"].isin(q_lookup.index)
                                    & relevant["target_entity_id"].isin(t_lookup.index)]
            entity_score = float("nan")
            entity_concordance = float("nan")
            n_entities = int(relevant["query_entity_id"].nunique())
            entity_query_coverage = n_entities / len(query_effects) if len(query_effects) else 0.0
            entity_target_coverage = n_entities / len(target_effects) if len(target_effects) else 0.0
            if len(relevant):
                q = q_lookup.loc[relevant["query_entity_id"], ["beta", "se"]].reset_index(drop=True)
                t = t_lookup.loc[relevant["target_entity_id"], ["beta", "se"]].reset_index(drop=True)
                weight = relevant["match_probability"].to_numpy(float) / (
                    q["se"].to_numpy(float) ** 2 + t["se"].to_numpy(float) ** 2
                )
                entity_score = weighted_cosine(q["beta"].to_numpy(float), t["beta"].to_numpy(float), weight)
                entity_concordance = float(np.average(
                    np.sign(q["beta"].to_numpy(float)) == np.sign(t["beta"].to_numpy(float)), weights=weight,
                ))
            entity_pass = (
                n_entities >= min_entities
                and entity_query_coverage >= min_coverage
                and entity_target_coverage >= min_coverage
            )

            target_patterns = patterns.loc[(patterns["study_id"] == target.study_id)
                                           & (patterns["contrast_id"] == target.contrast_id)] if not patterns.empty else patterns
            common = query_patterns.merge(target_patterns, on="pattern_id", suffixes=("_q", "_t"))
            n_patterns = len(common)
            pattern_query_coverage = n_patterns / len(query_patterns) if len(query_patterns) else 0.0
            pattern_target_coverage = n_patterns / len(target_patterns) if len(target_patterns) else 0.0
            pattern_score = float("nan")
            if n_patterns:
                variance = common["pattern_se_q"].to_numpy(float) ** 2 + common["pattern_se_t"].to_numpy(float) ** 2
                weight = 1.0 / np.maximum(variance, 1e-12)
                pattern_score = weighted_cosine(
                    common["pattern_beta_q"].to_numpy(float), common["pattern_beta_t"].to_numpy(float), weight,
                )
            pattern_pass = (
                n_patterns >= min_patterns
                and pattern_query_coverage >= min_coverage
                and pattern_target_coverage >= min_coverage
            )
            if not pattern_pass:
                association = "abstain_insufficient_comparable_evidence"
            elif pattern_score >= association_threshold:
                association = "similar_response"
            elif pattern_score <= -association_threshold:
                association = "opposite_response"
            else:
                association = "indeterminate_response"
            output.append({
                "world": world,
                "query_study_id": query_study, "query_contrast_id": query_contrast,
                "target_study_id": target.study_id, "target_source_family": target.source_family,
                "target_contrast_id": target.contrast_id,
                "n_query_entities": int(len(query_effects)), "n_strict_entities": n_entities,
                "n_target_entities": int(len(target_effects)),
                "strict_entity_query_coverage": entity_query_coverage,
                "strict_entity_target_coverage": entity_target_coverage,
                "strict_entity_score": entity_score,
                "strict_direction_concordance": entity_concordance,
                "strict_entity_pass": bool(entity_pass),
                "n_query_patterns": int(len(query_patterns)), "n_common_patterns": int(n_patterns),
                "n_target_patterns": int(len(target_patterns)),
                "shared_pattern_query_coverage": pattern_query_coverage,
                "shared_pattern_target_coverage": pattern_target_coverage,
                "shared_pattern_score": pattern_score,
                "shared_pattern_pass": bool(pattern_pass), "study_contrast_association": association,
            })
    empty_pattern_table = pd.DataFrame(columns=[*PATTERN_RESPONSE_COLUMNS, "world"])
    return (
        pd.DataFrame(output),
        pd.concat(pattern_tables, ignore_index=True) if pattern_tables else empty_pattern_table,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrast-records", type=Path, required=True)
    parser.add_argument("--pattern-loadings", type=Path, required=True)
    parser.add_argument("--strict-links", type=Path, required=True)
    parser.add_argument("--query-study", required=True)
    parser.add_argument("--query-contrast", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-same-source", action="store_true")
    parser.add_argument("--min-entities", type=int, default=3)
    parser.add_argument("--min-patterns", type=int, default=2)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--min-match-probability", type=float, default=0.8)
    parser.add_argument("--association-threshold", type=float, default=0.3)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contrast_records = pd.read_csv(args.contrast_records)
    pattern_loadings = pd.read_csv(args.pattern_loadings)
    strict_links = pd.read_csv(args.strict_links)
    config = {
        "min_entities": args.min_entities,
        "min_patterns": args.min_patterns,
        "min_coverage": args.min_coverage,
        "min_match_probability": args.min_match_probability,
        "association_threshold": args.association_threshold,
        "allow_same_source": args.allow_same_source,
    }
    results, patterns = retrieve(
        contrast_records, pattern_loadings, strict_links,
        args.query_study, args.query_contrast, **config,
    )
    args.output_dir.mkdir(parents=True)
    results.to_csv(args.output_dir / "retrieval_results.csv", index=False)
    patterns.to_csv(args.output_dir / "pattern_responses.csv", index=False)
    report = {
        "status": "CONTRAST_ATLAS_RETRIEVAL_COMPLETE",
        "formal": False,
        "analysis_role": "prototype/development",
        "query": {"study_id": args.query_study, "contrast_id": args.query_contrast},
        "configuration": config,
        "input_sha256": {
            "contrast_records": hashlib.sha256(args.contrast_records.read_bytes()).hexdigest(),
            "pattern_loadings": hashlib.sha256(args.pattern_loadings.read_bytes()).hexdigest(),
            "strict_links": hashlib.sha256(args.strict_links.read_bytes()).hexdigest(),
        },
        "input_rows": {
            "contrast_records": int(len(contrast_records)),
            "pattern_loadings": int(len(pattern_loadings)),
            "strict_links": int(len(strict_links)),
        },
        "target_contrasts": int(len(results.loc[results["world"] == "all"])),
        "abstentions_all_world": int(results.loc[results["world"] == "all", "study_contrast_association"].str.startswith("abstain").sum()),
        "claim_limit": "Prototype retrieval only; no biological correctness or unknown increment is established without held-out truth.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
