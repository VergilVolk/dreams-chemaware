#!/usr/bin/env python
"""Run all frozen evidence-control worlds on one invariant candidate universe.

Each external evidence world is evaluated only through the core's ``all``
identity view.  The external world label is then restored.  This prevents the
core's diagnostic identity decomposition from being confused with the four
paired controls used to test unresolved-chemistry increment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from adjudicate_contrast_atlas_unknown_increment import REQUIRED_WORLDS
from run_contrast_atlas_batch_retrieval import batch_retrieve


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_control_retrievals(
    control_root: Path,
    universe_effects: pd.DataFrame,
    registry: pd.DataFrame,
    **retrieve_config: object,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
    result_tables: list[pd.DataFrame] = []
    pattern_tables: list[pd.DataFrame] = []
    counts: dict[str, dict[str, int]] = {}
    unit_effect_presence = {
        world: (control_root / world / "contrast_unit_effects.csv.gz").is_file()
        for world in REQUIRED_WORLDS
    }
    if len(set(unit_effect_presence.values())) != 1:
        raise RuntimeError("contrast unit effects must be present in every control world or none")
    for external_world in REQUIRED_WORLDS:
        world_dir = control_root / external_world
        required = {
            "effects": world_dir / "contrast_records.csv.gz",
            "loadings": world_dir / "pattern_loadings.csv.gz",
            "links": world_dir / "strict_links.csv.gz",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"control world {external_world} is incomplete: {missing}")
        results, patterns = batch_retrieve(
            pd.read_csv(required["effects"]),
            pd.read_csv(required["loadings"]),
            pd.read_csv(required["links"]),
            registry,
            universe_effects=universe_effects,
            unit_effects=(pd.read_csv(world_dir / "contrast_unit_effects.csv.gz")
                          if (world_dir / "contrast_unit_effects.csv.gz").is_file() else None),
            identity_worlds=("all",),
            **retrieve_config,
        )
        if not results["world"].eq("all").all():
            raise RuntimeError("internal diagnostic world leaked into control retrieval")
        results["world"] = external_world
        patterns["world"] = external_world
        result_tables.append(results)
        pattern_tables.append(patterns)
        counts[external_world] = {
            "retrieval_rows": int(len(results)),
            "pattern_rows": int(len(patterns)),
            "abstentions": int(results["study_contrast_association"].str.startswith("abstain").sum()),
        }
    combined = pd.concat(result_tables, ignore_index=True)
    patterns = pd.concat(pattern_tables, ignore_index=True)
    result_key = ["query_id", "world", "target_study_id", "target_contrast_id"]
    if combined.duplicated(result_key).any():
        raise RuntimeError("duplicate candidate after control-world assembly")
    candidate_key = ["target_study_id", "target_contrast_id"]
    for query_id, query_rows in combined.groupby("query_id", sort=False):
        reference = None
        for world in REQUIRED_WORLDS:
            keys = set(map(tuple, query_rows.loc[query_rows["world"].eq(world), candidate_key].to_numpy()))
            if reference is None:
                reference = keys
            elif keys != reference:
                raise RuntimeError(
                    f"candidate graph changes across evidence-control worlds for query {query_id}"
                )
    return combined, patterns, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--universe-contrast-records", type=Path, required=True)
    parser.add_argument("--query-registry", type=Path, required=True)
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
    config = {
        "min_entities": args.min_entities,
        "min_patterns": args.min_patterns,
        "min_coverage": args.min_coverage,
        "min_match_probability": args.min_match_probability,
        "association_threshold": args.association_threshold,
        "allow_same_source": args.allow_same_source,
    }
    registry = pd.read_csv(args.query_registry)
    universe = pd.read_csv(args.universe_contrast_records)
    results, patterns, counts = run_control_retrievals(
        args.control_root, universe, registry, **config,
    )
    args.output_dir.mkdir(parents=True)
    results.to_csv(args.output_dir / "retrieval_results.csv", index=False)
    patterns.to_csv(args.output_dir / "pattern_responses.csv", index=False)
    report = {
        "status": "CONTRAST_ATLAS_CONTROL_RETRIEVAL_COMPLETE",
        "formal": False,
        "analysis_role": "development retrieval before truth opening",
        "candidate_universe_invariant": True,
        "queries": int(registry["query_id"].nunique()),
        "worlds": list(REQUIRED_WORLDS),
        "world_counts": counts,
        "configuration": config,
        "input_sha256": {
            "universe_contrast_records": sha256(args.universe_contrast_records),
            "query_registry": sha256(args.query_registry),
            "control_world_report": sha256(args.control_root / "report.json"),
        },
        "claim_limit": "Truth-blind paired control retrieval only; no unknown increment or biological relevance is established.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
