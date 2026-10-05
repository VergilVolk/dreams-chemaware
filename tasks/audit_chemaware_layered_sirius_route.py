"""Freeze the local evidence boundary before the one-GPU ChemAware run."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rule-corpus", type=Path,
        default=ROOT / "data/validation/chemaware_layered_rule_corpus_v5_20260928",
    )
    parser.add_argument(
        "--static-source", type=Path,
        default=(ROOT / "data/validation"
                 / "chemaware_layered_rule_source_qualified_v9_profiled_collision_20260928"),
    )
    parser.add_argument(
        "--static-capacity", type=Path,
        default=(ROOT / "data/validation"
                 / "chemaware_layered_static_multisource_official_geometry_capacity_v6_profiled_20260928"),
    )
    parser.add_argument(
        "--sirius-panel", type=Path,
        default=(
            ROOT / "data/validation/chemaware_sirius_source_panel_v9_profiled_collision_20260928"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_report(path: Path) -> dict:
    return json.loads((path / "report.json").read_text(encoding="utf-8"))


def row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle, delimiter="\t"))


def build_report(args: argparse.Namespace) -> dict:
    rules = load_report(args.rule_corpus)
    static = load_report(args.static_source)
    capacity = load_report(args.static_capacity)
    panel = load_report(args.sirius_panel)
    source_job = (ROOT / "tasks/run_chemaware_sirius_source.sbatch").read_text(encoding="utf-8")
    native_job = (ROOT / "tasks/run_chemaware_sirius_native.sbatch").read_text(encoding="utf-8")
    integrated_job = (
        ROOT / "tasks/run_chemaware_layered_sirius_10k_native.sbatch"
    ).read_text(encoding="utf-8")
    static_rows = row_count(args.static_source / "candidate_scores.tsv")
    admitted = static["specificity_qualification"]["admitted_families"]
    gates = {
        "rule_corpus_exceeds_legacy_335": int(rules["records"]) > 335,
        "qualified_candidate_relationships_at_least_10000": static_rows >= 10_000,
        "only_admitted_static_rows_exported": static_rows == int(static["candidate_source_rows"]),
        "corrected_query_registry_is_provenanced": "query_registry" in static["provenance"],
        "static_capacity_is_explicit_engineering_standin": (
            capacity["initialization"] == "OFFICIAL_GEOMETRY_ENGINEERING_STANDIN_NOT_PHASEA"
        ),
        "static_increment_not_overclaimed": int(capacity["source_events_added"]) == 0,
        "sirius_panel_all_gates_pass": all(panel["gates"].values()),
        "one_gpu_no_manual_memory": all(
            "#SBATCH --gpus=1" in text and "#SBATCH --mem" not in text
            for text in (source_job, native_job, integrated_job)
        ),
        "native_loss_unchanged": (
            "--triplet-loss-margin 0.1" in native_job
            and "--official-checkpoint \"$PHASEA\"" in native_job
        ),
        "static_and_sirius_sources_both_wired": (
            "qualified_static_source" in native_job and "qualified_source" in native_job
        ),
        "instrument_profiles_and_candidate_tolerance_frozen": (
            "for profile in orbitrap qtof default" in source_job
            and "profile_args=(-p orbitrap --ppm-max 10 --ppm-max-ms2 5)" in source_job
            and "profile_args=(-p qtof --ppm-max 10 --ppm-max-ms2 10)" in source_job
        ),
        "sealed_role3_confirmation_gate": (
            "audit_chemaware_role3_confirmation.py" in native_job
            and "protected_negative_result" in native_job
            and "CHEMAWARE_ROLE3_STOP" in native_job
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"layered SIRIUS preflight failed: {gates}")
    return {
        "status": "CHEMAWARE_LAYERED_SIRIUS_LOCAL_PREFLIGHT_COMPLETE",
        "formal_training_authorized": False,
        "execution_decision": "SIRIUS_RUNTIME_AND_QUALIFICATION_REQUIRED_BEFORE_TRIPLET_TRAINING",
        "rule_corpus": {
            "records": int(rules["records"]),
            "confidence_tiers": rules["confidence_tiers"],
            "sha256": rules["rule_corpus_sha256"],
        },
        "qualified_static_relationships": {
            "rows": static_rows,
            "queries": int(static["queries"]),
            "admitted_families": admitted,
            "rejected_families": sorted(static["unqualified_source_families"]),
            "incremental_triplets_under_official_geometry_standin": int(
                capacity["source_events_added"]
            ),
            "interpretation": (
                "qualified relationship corpus, but no new active triplet beyond Phase-A; "
                "never pad or relax this result"
            ),
        },
        "sirius_panel": {
            "queries": int(panel["queries"]),
            "query_formula_features": int(panel["query_formula_features"]),
            "controlled_query_formula_features": int(panel["controlled_query_formula_features"]),
            "candidate_rows": int(panel["candidate_rows"]),
            "unique_candidate_connectivities": int(panel["unique_candidate_connectivities"]),
            "instrument_profile_query_counts": panel["instrument_profile_query_counts"],
            "instrument_profile_formula_feature_counts": panel[
                "instrument_profile_formula_feature_counts"
            ],
        },
        "frozen_training_gates": {
            "minimum_new_chemical_events": 1000,
            "minimum_new_chemical_queries": 500,
            "minimum_total_native_events": 10000,
            "target_total_native_events": 12000,
            "loss_weights": "all one",
            "initialization": "protected Phase-A +2.1266 pp checkpoint",
        },
        "single_submission": "sbatch tasks/run_chemaware_layered_sirius_10k_native.sbatch",
        "gates": gates,
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_layered_route_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
