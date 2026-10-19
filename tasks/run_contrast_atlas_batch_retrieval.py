#!/usr/bin/env python
"""Run the fail-closed Contrast Atlas retrieval core over a frozen query registry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from contrast_atlas_core import retrieve


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_registry(registry: pd.DataFrame) -> pd.DataFrame:
    required = {
        "query_id", "query_study_id", "query_contrast_id", "cluster_id",
        "source_holdout", "instrument_holdout", "data_role",
    }
    missing = required - set(registry.columns)
    if missing:
        raise RuntimeError(f"query registry missing columns: {sorted(missing)}")
    if registry.duplicated(["query_id"]).any():
        raise RuntimeError("duplicate query_id")
    if registry.duplicated(["query_study_id", "query_contrast_id"]).any():
        raise RuntimeError("duplicate study contrast in query registry")
    if registry[["query_id", "cluster_id", "source_holdout", "instrument_holdout"]].isna().any().any():
        raise RuntimeError("query registry contains missing grouping identifiers")
    return registry.copy()


def batch_retrieve(
    effects: pd.DataFrame,
    loadings: pd.DataFrame,
    links: pd.DataFrame,
    registry_raw: pd.DataFrame,
    universe_effects: pd.DataFrame | None = None,
    unit_effects: pd.DataFrame | None = None,
    identity_worlds: tuple[str, ...] = ("all", "known_only", "unresolved_only"),
    **retrieve_config: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = validate_registry(registry_raw)
    results = []
    pattern_tables = []
    for query in registry.itertuples(index=False):
        result, patterns = retrieve(
            effects, loadings, links,
            str(query.query_study_id), str(query.query_contrast_id),
            universe_effects_raw=universe_effects,
            identity_worlds=identity_worlds,
            unit_effects_raw=unit_effects,
            **retrieve_config,
        )
        result.insert(0, "query_id", query.query_id)
        result["cluster_id"] = query.cluster_id
        result["source_holdout"] = query.source_holdout
        result["instrument_holdout"] = query.instrument_holdout
        result["query_data_role"] = query.data_role
        results.append(result)
        pattern_tables.append(patterns)
    combined = pd.concat(results, ignore_index=True)
    key = ["query_id", "world", "target_study_id", "target_contrast_id"]
    if combined.duplicated(key).any():
        raise RuntimeError("duplicate batch retrieval result")
    patterns = pd.concat(pattern_tables, ignore_index=True)
    if not patterns.empty:
        patterns = patterns.drop_duplicates(
            ["world", "study_id", "contrast_id", "pattern_id"], keep="first",
        )
    return combined, patterns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrast-records", type=Path, required=True)
    parser.add_argument("--pattern-loadings", type=Path, required=True)
    parser.add_argument("--strict-links", type=Path, required=True)
    parser.add_argument("--query-registry", type=Path, required=True)
    parser.add_argument("--contrast-unit-effects", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-same-source", action="store_true")
    parser.add_argument("--min-entities", type=int, default=3)
    parser.add_argument("--min-patterns", type=int, default=8)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--min-match-probability", type=float, default=0.8)
    parser.add_argument("--association-threshold", type=float, default=0.3)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    paths = {
        "contrast_records": args.contrast_records,
        "pattern_loadings": args.pattern_loadings,
        "strict_links": args.strict_links,
        "query_registry": args.query_registry,
    }
    if args.contrast_unit_effects is not None:
        paths["contrast_unit_effects"] = args.contrast_unit_effects
    config = {
        "min_entities": args.min_entities,
        "min_patterns": args.min_patterns,
        "min_coverage": args.min_coverage,
        "min_match_probability": args.min_match_probability,
        "association_threshold": args.association_threshold,
        "allow_same_source": args.allow_same_source,
    }
    registry = pd.read_csv(args.query_registry)
    results, patterns = batch_retrieve(
        pd.read_csv(args.contrast_records),
        pd.read_csv(args.pattern_loadings),
        pd.read_csv(args.strict_links),
        registry,
        unit_effects=(pd.read_csv(args.contrast_unit_effects)
                      if args.contrast_unit_effects is not None else None),
        **config,
    )
    args.output_dir.mkdir(parents=True)
    results.to_csv(args.output_dir / "retrieval_results.csv", index=False)
    patterns.to_csv(args.output_dir / "pattern_responses.csv", index=False)
    report = {
        "status": "CONTRAST_ATLAS_BATCH_RETRIEVAL_COMPLETE",
        "formal": False,
        "analysis_role": "development retrieval before truth opening",
        "queries": int(registry["query_id"].nunique()),
        "query_clusters": int(registry["cluster_id"].nunique()),
        "source_holdouts": int(registry["source_holdout"].nunique()),
        "instrument_holdouts": int(registry["instrument_holdout"].nunique()),
        "configuration": config,
        "input_sha256": {key: sha256(path) for key, path in paths.items()},
        "result_rows": int(len(results)),
        "pattern_rows": int(len(patterns)),
        "claim_limit": "Truth-blind batch retrieval only; no accuracy or biological association is established.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
