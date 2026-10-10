#!/usr/bin/env python
"""Build a layered entity/candidate evidence ledger without fitting a fusion model.

The ledger keeps four different scientific objects separate:

* Noise V1: stable spectrum coordinate and dense retrieval score;
* WSE: independent classical retrieval score and current primary Top-1 baseline;
* frozen P2b-on-Noise: local spectral competition score;
* ChemAware/BioAware: sparse candidate evidence with explicit applicability.

No module score is added to another.  Ground-truth labels are used only in the
separate development-evaluation summary and never in the output-tier rule.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


PANELS = ("identity_disjoint", "formula_disjoint")
METHODS = ("noise_v1", "weighted_spectral_entropy", "p2b_noise_v1_frozen")
SCHEMA = "unified_entity_evidence_ledger_v1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--score-bundle", type=Path, required=True)
    parser.add_argument(
        "--evaluation-report", type=Path,
        help=(
            "Authoritative report for the score bundle. If omitted, the builder looks for "
            "evaluation/report.json beside the bundle and reconciles all required metrics."
        ),
    )
    parser.add_argument(
        "--chemaware-action-cache", type=Path,
        help="Directory containing action_ledger_<panel>.npz; omission means unavailable.",
    )
    parser.add_argument(
        "--entity-manifest", type=Path,
        help=(
            "Optional CSV(.gz), keyed by panel+query_index. Supported fields include "
            "entity_id, qc_pass, reference_status, orthogonal_structure_status, "
            "precursor_mz, retention_time, blank_ratio, qc_rsd, adduct and ion_family."
        ),
    )
    parser.add_argument(
        "--bioaware-events", type=Path,
        help=(
            "Optional candidate event CSV(.gz), keyed by panel+query_index+candidate_id, "
            "with event_id,event_type,event_score,event_quality_pass."
        ),
    )
    parser.add_argument("--panel", action="append", choices=PANELS)
    parser.add_argument("--dataset-id", default="gnps_gold_silver_10ppm")
    parser.add_argument("--dataset-role", default="development_consumed")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _open_csv(path: Path):
    return gzip.open(path, "rt", encoding="utf-8-sig", newline="") if path.suffix == ".gz" \
        else path.open("r", encoding="utf-8-sig", newline="")


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "pass", "passed"}:
        return True
    if normalized in {"0", "false", "no", "n", "fail", "failed"}:
        return False
    raise ValueError(f"invalid Boolean value: {value!r}")


def load_entity_manifest(path: Path | None) -> dict[tuple[str, int], dict[str, str]]:
    if path is None:
        return {}
    with _open_csv(path) as handle:
        rows = list(csv.DictReader(handle))
    required = {"panel", "query_index"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"entity manifest must contain {sorted(required)}")
    output: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["panel"], int(row["query_index"]))
        if key in output:
            raise RuntimeError(f"duplicate entity manifest key: {key}")
        output[key] = row
    return output


def load_bioaware_events(path: Path | None) -> dict[tuple[str, int, str], dict[str, object]]:
    if path is None:
        return {}
    with _open_csv(path) as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "panel", "query_index", "candidate_id", "event_id", "event_type",
        "event_score", "event_quality_pass",
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"BioAware events must contain {sorted(required)}")
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["panel"], int(row["query_index"]), row["candidate_id"])].append(row)
    output: dict[tuple[str, int, str], dict[str, object]] = {}
    for key, group in grouped.items():
        passed = [row for row in group if _as_bool(row["event_quality_pass"])]
        if not passed:
            continue
        scores = [float(row["event_score"]) for row in passed]
        if not np.all(np.isfinite(scores)):
            raise RuntimeError(f"non-finite BioAware event score at {key}")
        output[key] = {
            "event_count": len(passed),
            "event_max_score": max(scores),
            "event_ids": "|".join(sorted({row["event_id"] for row in passed})),
            "event_types": "|".join(sorted({row["event_type"] for row in passed})),
        }
    return output


def load_scores(path: Path, panels: tuple[str, ...]) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as body:
        if "method_names" not in body:
            raise RuntimeError("score bundle has no method_names")
        names = [str(value) for value in body["method_names"]]
        missing = sorted(set(METHODS) - set(names))
        if missing:
            raise RuntimeError(f"score bundle misses required methods: {missing}")
        output = {}
        for panel in panels:
            key = f"scores_{panel}"
            if key not in body:
                raise RuntimeError(f"score bundle misses {key}")
            values = np.asarray(body[key], dtype=np.float32)
            if values.ndim != 2 or values.shape[0] != len(names):
                raise RuntimeError(f"malformed score array: {key} {values.shape}")
            output[panel] = {method: values[names.index(method)].copy() for method in METHODS}
    return output


def molecule_scores(pair_scores: np.ndarray, molecule_ptr: np.ndarray) -> np.ndarray:
    pair_scores = np.asarray(pair_scores, dtype=np.float32)
    if pair_scores.ndim != 1 or len(pair_scores) != int(molecule_ptr[-1]):
        raise RuntimeError("pair scores do not align to molecule_ptr")
    if not np.all(np.isfinite(pair_scores)):
        raise RuntimeError("pair scores contain non-finite values")
    return np.maximum.reduceat(pair_scores, molecule_ptr[:-1])


def ranks_desc(values: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
    # Candidate-ID tie breaking is truth-blind. Panel order cannot be used here
    # because the evaluation positive is deliberately block-first.
    order = np.lexsort((candidate_ids.astype(str), -values))
    ranks = np.empty(len(values), dtype=np.int32)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int32)
    return ranks


def positive_rank_ties_against(values: np.ndarray, labels: np.ndarray) -> int:
    """Match the frozen evaluator: every negative tied with the positive ranks ahead."""
    positive = int(np.flatnonzero(labels)[0])
    negative = np.flatnonzero(~labels)
    return 1 + int(np.count_nonzero(values[negative] >= values[positive]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_benchmark_panels(benchmark: Path, panels: tuple[str, ...]) -> str:
    report_path = benchmark / "report.json"
    if not report_path.is_file():
        return "report_unavailable"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = report.get("provenance", {})
    for panel in panels:
        key = "panel_identity_sha256" if panel == "identity_disjoint" else "panel_formula_sha256"
        declared = provenance.get(key)
        if declared and sha256(benchmark / f"panel_{panel}.npz") != declared:
            raise RuntimeError(f"benchmark panel fingerprint mismatch: {panel}")
    return "validated_when_declared"


def infer_evaluation_report(score_bundle: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit
    candidates = (
        score_bundle.parent / "evaluation/report.json",
        score_bundle.parent.parent / "evaluation/report.json",
    )
    return next((path for path in candidates if path.is_file()), None)


def validate_metric_reconciliation(
    evaluation_report: Path | None,
    panels: tuple[str, ...],
    panel_reports: dict[str, dict[str, object]],
) -> str:
    if evaluation_report is None:
        return "not_checked_no_authoritative_report"
    authority = json.loads(evaluation_report.read_text(encoding="utf-8"))
    for panel in panels:
        absolute = authority.get("panels", {}).get(panel, {}).get("absolute", {})
        for method in METHODS:
            expected = absolute.get(method, {}).get("retrieval")
            if not isinstance(expected, dict):
                raise RuntimeError(f"authoritative evaluation misses {panel}/{method}")
            observed = panel_reports[panel]["module_ablation_same_denominator"][method]
            for metric in ("mrr", "recall@1", "recall@2", "recall@3", "recall@5", "recall@10", "recall@20"):
                if abs(float(observed[metric]) - float(expected[metric])) > 1e-12:
                    raise RuntimeError(
                        f"score/panel alignment failed at {panel}/{method}/{metric}: "
                        f"{observed[metric]} != {expected[metric]}"
                    )
    return f"matched:{evaluation_report}"


def metric_summary(ranks: np.ndarray) -> dict[str, float | int]:
    ranks = np.asarray(ranks, dtype=np.int64)
    return {
        "queries": int(len(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        **{f"recall@{k}": float(np.mean(ranks <= k)) for k in (1, 2, 3, 5, 10, 20)},
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }


def load_chemaware_ledger(cache: Path | None, panel: str, query_count: int) -> dict[str, np.ndarray] | None:
    if cache is None:
        return None
    path = cache / f"action_ledger_{panel}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    required = {
        "query_index", "abstained", "selected_candidate", "deployment_top_candidate",
        "changed_deployment_top1",
    }
    with np.load(path, allow_pickle=False) as body:
        if not required.issubset(body.files):
            raise RuntimeError(f"ChemAware action ledger misses {sorted(required - set(body.files))}")
        output = {key: np.asarray(body[key]).copy() for key in required}
    if any(len(values) != query_count for values in output.values()):
        raise RuntimeError(f"ChemAware action ledger query count mismatch: {panel}")
    if not np.array_equal(output["query_index"], np.arange(query_count)):
        raise RuntimeError(f"ChemAware action ledger query axis mismatch: {panel}")
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty ledger: {path.name}")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_panel(
    panel: str,
    panel_path: Path,
    pair_scores: dict[str, np.ndarray],
    chemaware_cache: Path | None,
    entity_manifest: dict[tuple[str, int], dict[str, str]],
    bio_events: dict[tuple[str, int, str], dict[str, object]],
    dataset_id: str,
    staging: Path,
) -> dict[str, object]:
    with np.load(panel_path, allow_pickle=False) as body:
        required = {
            "query_row", "query_ptr", "molecule_ptr", "candidate_row", "molecule_label",
            "molecule_ik14", "query_ik14", "query_formula", "near_query",
        }
        if not required.issubset(body.files):
            raise RuntimeError(f"panel misses {sorted(required - set(body.files))}: {panel_path}")
        data = {key: np.asarray(body[key]).copy() for key in required}
    qptr = data["query_ptr"].astype(np.int64)
    mptr = data["molecule_ptr"].astype(np.int64)
    query_count = len(data["query_row"])
    molecule_count = len(data["molecule_label"])
    if len(qptr) != query_count + 1 or len(mptr) != molecule_count + 1:
        raise RuntimeError(f"invalid panel pointers: {panel}")
    if int(qptr[-1]) != molecule_count or int(mptr[-1]) != len(data["candidate_row"]):
        raise RuntimeError(f"panel pointer terminal mismatch: {panel}")
    labels = data["molecule_label"].astype(bool)
    for query in range(query_count):
        left, right = int(qptr[query]), int(qptr[query + 1])
        if np.count_nonzero(labels[left:right]) != 1:
            raise RuntimeError(f"query {query} does not have exactly one evaluation positive")

    scores = {method: molecule_scores(pair_scores[method], mptr) for method in METHODS}
    ranks = {method: np.empty(molecule_count, dtype=np.int32) for method in METHODS}
    top_local = {method: np.empty(query_count, dtype=np.int32) for method in METHODS}
    positive_rank = {method: np.empty(query_count, dtype=np.int32) for method in METHODS}
    for query in range(query_count):
        left, right = int(qptr[query]), int(qptr[query + 1])
        for method in METHODS:
            local_ranks = ranks_desc(
                scores[method][left:right], data["molecule_ik14"][left:right],
            )
            ranks[method][left:right] = local_ranks
            top_local[method][query] = int(np.argmin(local_ranks))
            positive_rank[method][query] = positive_rank_ties_against(
                scores[method][left:right], labels[left:right],
            )

    chem = load_chemaware_ledger(chemaware_cache, panel, query_count)
    candidate_rows: list[dict[str, object]] = []
    entity_rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
    reference_statuses: list[str] = []
    bio_entities = 0
    chem_selected = chem_corrected = chem_introduced = chem_changed = 0
    candidate_ids = data["molecule_ik14"].astype(str)
    query_ids = data["query_ik14"].astype(str)
    formulas = data["query_formula"].astype(str)

    for query in range(query_count):
        left, right = int(qptr[query]), int(qptr[query + 1])
        meta = entity_manifest.get((panel, query), {})
        entity_id = meta.get(
            "entity_id", f"{dataset_id}:{panel}:spectrum_row_{int(data['query_row'][query])}",
        )
        qc_pass = _as_bool(meta.get("qc_pass"), default=True)
        reference_status = meta.get("reference_status", "known_library_query")
        reference_statuses.append(reference_status)
        orthogonal = meta.get("orthogonal_structure_status", "unavailable")
        chem_available = chem is not None and not bool(chem["abstained"][query])
        chem_local = int(chem["selected_candidate"][query]) if chem_available else -1
        if chem_available:
            if not 0 <= chem_local < right - left:
                raise RuntimeError(f"ChemAware selected candidate out of range: {panel}/{query}")
            deployment_local = int(chem["deployment_top_candidate"][query])
            expected_deployment = int(np.argmax(scores["noise_v1"][left:right]))
            if deployment_local != expected_deployment:
                raise RuntimeError(
                    f"ChemAware/Noise candidate-axis mismatch: {panel}/{query}"
                )
            chem_selected += 1
            changed = bool(chem["changed_deployment_top1"][query])
            if changed != (chem_local != deployment_local):
                raise RuntimeError(f"ChemAware changed-action contract mismatch: {panel}/{query}")
            chem_changed += int(changed)
            if changed:
                selected_correct = bool(labels[left + chem_local])
                noise_correct = bool(labels[left + deployment_local])
                chem_corrected += int(selected_correct and not noise_correct)
                chem_introduced += int(noise_correct and not selected_correct)

        entity_has_bio = False
        for global_candidate in range(left, right):
            local_candidate = global_candidate - left
            candidate_id = candidate_ids[global_candidate]
            bio = bio_events.get((panel, query, candidate_id))
            entity_has_bio = entity_has_bio or bio is not None
            chem_status = "unavailable_query"
            if chem_available:
                chem_status = (
                    "selected_candidate" if local_candidate == chem_local
                    else "competitor_in_applicable_query"
                )
            row: dict[str, object] = {
                "dataset_id": dataset_id,
                "panel": panel,
                "entity_id": entity_id,
                "query_index": query,
                "query_spectrum_row": int(data["query_row"][query]),
                "candidate_index": local_candidate,
                "candidate_id": candidate_id,
            }
            for method in METHODS:
                row[f"{method}_score"] = float(scores[method][global_candidate])
                row[f"{method}_rank"] = int(ranks[method][global_candidate])
            row.update({
                "chemaware_applicable": chem_available,
                "chemaware_evidence_status": chem_status,
                "chemaware_selected": chem_available and local_candidate == chem_local,
                "bioaware_applicable": bio is not None,
                "bioaware_event_count": 0 if bio is None else bio["event_count"],
                "bioaware_event_max_score": "" if bio is None else bio["event_max_score"],
                "bioaware_event_ids": "" if bio is None else bio["event_ids"],
                "bioaware_event_types": "" if bio is None else bio["event_types"],
            })
            candidate_rows.append(row)
            truth_rows.append({
                "dataset_id": dataset_id,
                "panel": panel,
                "entity_id": entity_id,
                "query_index": query,
                "query_identity": query_ids[query],
                "query_formula": formulas[query],
                "near_query": bool(data["near_query"][query]),
                "candidate_index": local_candidate,
                "candidate_id": candidate_id,
                "evaluation_is_true_candidate": bool(labels[global_candidate]),
            })
        bio_entities += int(entity_has_bio)

        primary_local = int(top_local["weighted_spectral_entropy"][query])
        if not qc_pass:
            output_tier = "insufficient_evidence"
            rejection_reason = "entity_qc_failed"
        elif orthogonal == "authentic_standard_confirmed":
            output_tier = "trusted_structure"
            rejection_reason = ""
        else:
            output_tier = "ranked_structure_hypothesis"
            rejection_reason = "orthogonal_structure_confirmation_unavailable"
        entity_rows.append({
            "dataset_id": dataset_id,
            "panel": panel,
            "entity_id": entity_id,
            "query_index": query,
            "query_spectrum_row": int(data["query_row"][query]),
            "spectral_coordinate_method": "noise_v1",
            "spectral_coordinate_key": f"spectrum_row_{int(data['query_row'][query])}",
            "reference_status": reference_status,
            "qc_pass": qc_pass,
            "precursor_mz": meta.get("precursor_mz", ""),
            "retention_time": meta.get("retention_time", ""),
            "blank_ratio": meta.get("blank_ratio", ""),
            "qc_rsd": meta.get("qc_rsd", ""),
            "adduct": meta.get("adduct", ""),
            "ion_family": meta.get("ion_family", ""),
            "primary_retrieval_method": "weighted_spectral_entropy",
            "primary_candidate_id": candidate_ids[left + primary_local],
            "noise_v1_top_candidate_id": candidate_ids[left + int(top_local["noise_v1"][query])],
            "weighted_spectral_entropy_top_candidate_id": candidate_ids[left + primary_local],
            "p2b_noise_v1_frozen_top_candidate_id": candidate_ids[
                left + int(top_local["p2b_noise_v1_frozen"][query])
            ],
            "chemaware_applicable": chem_available,
            "chemaware_selected_candidate_id": (
                candidate_ids[left + chem_local] if chem_available else ""
            ),
            "bioaware_applicable": entity_has_bio,
            "orthogonal_structure_status": orthogonal,
            "output_tier": output_tier,
            "structure_claim_status": (
                "confirmed" if output_tier == "trusted_structure" else "unconfirmed"
            ),
            "rejection_reason": rejection_reason,
        })

    _write_csv(staging / f"candidate_evidence_{panel}.csv.gz", candidate_rows)
    _write_csv(staging / f"entity_evidence_{panel}.csv.gz", entity_rows)
    _write_csv(staging / f"evaluation_truth_{panel}.csv.gz", truth_rows)

    module_metrics = {
        method: metric_summary(positive_rank[method]) for method in METHODS
    }
    chem_report = {
        "status": "available" if chem is not None else "unavailable",
        "queries": query_count,
        "applicable_queries": chem_selected,
        "applicability": float(chem_selected / query_count),
        "changed_noise_top1": chem_changed,
        "corrected_vs_noise": chem_corrected,
        "introduced_vs_noise": chem_introduced,
        "risk_net_lambda2": chem_corrected - 2 * chem_introduced,
        "deployment": "evidence_only_no_score_override",
    }
    if chem is None:
        chem_report["unavailable_reason"] = "chemaware_action_cache_not_supplied"
    bio_report = {
        "status": "available" if bio_entities else "unavailable",
        "entities_with_quality_passed_events": bio_entities,
        "coverage": float(bio_entities / query_count),
        "deployment": "candidate_event_evidence_only",
    }
    if not bio_entities:
        bio_report["unavailable_reason"] = "no_real_event_context_for_library_queries"

    normalized_reference = [value.strip().lower() for value in reference_statuses]
    known = sum(value in {"known", "known_library_query", "known_reference"}
                for value in normalized_reference)
    unknown = query_count - known
    known_only = {
        "known_entities": known,
        "unknown_entities": unknown,
        "all_entities": query_count,
        "status": (
            "not_evaluable_all_entities_known"
            if unknown == 0 else "partition_materialized_biology_endpoint_required"
        ),
        "claim_limit": (
            "A library benchmark with no unknown experimental entities cannot estimate "
            "known-only versus known-plus-unknown biological increment."
            if unknown == 0 else
            "Known/unknown partitions exist, but an independent biological endpoint and "
            "held-out condition are required before an incremental application claim."
        ),
    }
    return {
        "queries": query_count,
        "candidate_molecules": molecule_count,
        "directed_spectrum_pairs": int(mptr[-1]),
        "module_ablation_same_denominator": module_metrics,
        "chemaware": chem_report,
        "bioaware": bio_report,
        "known_only_control": known_only,
        "output_tier_counts": {
            tier: sum(row["output_tier"] == tier for row in entity_rows)
            for tier in ("trusted_structure", "ranked_structure_hypothesis", "insufficient_evidence")
        },
    }


def main() -> None:
    args = arguments()
    panels = tuple(args.panel or PANELS)
    if len(set(panels)) != len(panels):
        raise ValueError("duplicate --panel")
    if args.output.exists():
        raise FileExistsError(args.output)
    for panel in panels:
        path = args.benchmark / f"panel_{panel}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
    benchmark_validation = validate_benchmark_panels(args.benchmark, panels)
    evaluation_report = infer_evaluation_report(args.score_bundle, args.evaluation_report)
    entity_manifest = load_entity_manifest(args.entity_manifest)
    bio_events = load_bioaware_events(args.bioaware_events)
    scores = load_scores(args.score_bundle, panels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        panel_reports = {}
        for panel in panels:
            panel_reports[panel] = build_panel(
                panel=panel,
                panel_path=args.benchmark / f"panel_{panel}.npz",
                pair_scores=scores[panel],
                chemaware_cache=args.chemaware_action_cache,
                entity_manifest=entity_manifest,
                bio_events=bio_events,
                dataset_id=args.dataset_id,
                staging=staging,
            )
        metric_reconciliation = validate_metric_reconciliation(
            evaluation_report, panels, panel_reports,
        )
        report = {
            "status": "UNIFIED_ENTITY_EVIDENCE_LEDGER_COMPLETE",
            "schema": SCHEMA,
            "dataset_id": args.dataset_id,
            "dataset_role": args.dataset_role,
            "leakage_status": (
                "development_consumed_no_external_claim"
                if "development" in args.dataset_role else "caller_declared"
            ),
            "algorithm": {
                "shared_coordinate": "noise_v1",
                "primary_retrieval_baseline": "weighted_spectral_entropy",
                "local_spectral_competition": "p2b_noise_v1_frozen",
                "chemical_evidence": "chemaware_sparse_applicability_no_score_override",
                "sample_context": "bioaware_real_quality_passed_events_or_unavailable",
                "fusion_model_fitted": False,
                "truth_used_for_output_tier": False,
            },
            "panels": panel_reports,
            "inputs": {
                "benchmark": str(args.benchmark),
                "score_bundle": str(args.score_bundle),
                "chemaware_action_cache": (
                    None if args.chemaware_action_cache is None else str(args.chemaware_action_cache)
                ),
                "entity_manifest": None if args.entity_manifest is None else str(args.entity_manifest),
                "bioaware_events": None if args.bioaware_events is None else str(args.bioaware_events),
            },
            "alignment_validation": {
                "benchmark_panels": benchmark_validation,
                "score_bundle_metrics": metric_reconciliation,
            },
            "claim_limit": (
                "This artifact unifies entity and candidate evidence without claiming that "
                "module agreement is structural identification. Final external performance "
                "requires the once-opened Enveda benchmark; unknown-entity biology requires "
                "an independent event-level application dataset."
            ),
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
