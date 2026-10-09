"""Seal aligned Enveda module scores against a frozen model and panel.

The input score bundle is produced without fitting or selecting the unified
model on Enveda.  This command validates its complete candidate axis against
the frozen benchmark, restricts modules to registry-enabled assets, and binds
all artifacts by SHA-256 for the one-time evaluator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_grand_unified_evidence_model import sha256_file


DATASET_ID = "enveda_180_filtered_20260713"


def exact(label: str, left: np.ndarray, right: np.ndarray) -> None:
    if not np.array_equal(np.asarray(left), np.asarray(right)):
        raise ValueError(f"DATA_LEAKAGE_GUARD: {label} differs from the frozen panel")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-score-bundle", type=Path, required=True)
    parser.add_argument("--panel-npz", type=Path, required=True)
    parser.add_argument("--panel", choices=("identity_disjoint", "formula_disjoint"), required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--benchmark-checksums", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--freeze-report", type=Path, required=True)
    parser.add_argument("--module-registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    required = (
        args.component_score_bundle, args.panel_npz, args.benchmark_report,
        args.benchmark_checksums, args.model, args.freeze_report, args.module_registry,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required frozen artifacts are absent: {missing}")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite sealed bundle: {args.out}")
    if args.out.suffix != ".npz":
        raise ValueError("sealed output must end in .npz")

    freeze = json.loads(args.freeze_report.read_text(encoding="utf-8"))
    benchmark = json.loads(args.benchmark_report.read_text(encoding="utf-8"))
    registry = json.loads(args.module_registry.read_text(encoding="utf-8"))
    model_sha = sha256_file(args.model)
    registry_sha = sha256_file(args.module_registry)
    if freeze.get("freeze_contract_status") != "DEVELOPMENT_MODEL_FROZEN_EXTERNAL_UNOPENED":
        raise ValueError("DATA_LEAKAGE_GUARD: invalid freeze contract status")
    if freeze.get("external_test_opened") is not False:
        raise ValueError("DATA_LEAKAGE_GUARD: external test was already opened")
    if freeze.get("frozen_model_sha256") != model_sha:
        raise ValueError("DATA_LEAKAGE_GUARD: model hash differs from freeze report")
    if freeze.get("module_registry_sha256") != registry_sha:
        raise ValueError("DATA_LEAKAGE_GUARD: registry hash differs from freeze report")
    if benchmark.get("dataset_id") != DATASET_ID or benchmark.get("truth_status") != "frozen_unscored":
        raise ValueError("DATA_LEAKAGE_GUARD: benchmark is not frozen Enveda-180")
    if benchmark.get("performance_scores_opened") is not False:
        raise ValueError("DATA_LEAKAGE_GUARD: benchmark scores were already opened")
    guard = benchmark.get("leakage_guard", {})
    if guard.get("exclusion_policy_complete") is not True:
        raise ValueError("DATA_LEAKAGE_GUARD: incomplete consumed-source exclusions")
    if int(guard.get("selected_consumed_identity_overlap", -1)) != 0:
        raise ValueError("DATA_LEAKAGE_GUARD: consumed identity overlap is non-zero")
    if int(guard.get("selected_consumed_spectrum_overlap", -1)) != 0:
        raise ValueError("DATA_LEAKAGE_GUARD: consumed spectrum overlap is non-zero")

    checksum_rows = {}
    for raw in args.benchmark_checksums.read_text(encoding="utf-8").splitlines():
        value, name = raw.split(maxsplit=1)
        checksum_rows[name.strip()] = value
    expected_panel_name = f"panel_{args.panel}.npz"
    if checksum_rows.get(expected_panel_name) != sha256_file(args.panel_npz):
        raise ValueError("DATA_LEAKAGE_GUARD: panel hash is absent from frozen checksums")
    if benchmark.get("checksums", {}).get(expected_panel_name) != sha256_file(args.panel_npz):
        raise ValueError("DATA_LEAKAGE_GUARD: panel hash differs from benchmark report")

    if args.component_score_bundle.stat().st_mtime_ns < args.freeze_report.stat().st_mtime_ns:
        raise ValueError(
            "DATA_LEAKAGE_GUARD: component score bundle predates the unified freeze report; "
            "regenerate scores only after the unified model is frozen"
        )
    with np.load(args.component_score_bundle, allow_pickle=False) as source:
        source_arrays = {key: source[key] for key in source.files}
    needed = {
        "module_names", "scores", "availability", "query_ptr", "labels",
        "group_ids", "query_ids", "candidate_ids",
    }
    absent = needed - source_arrays.keys()
    if absent:
        raise ValueError(f"component score bundle lacks fields: {sorted(absent)}")
    with np.load(args.panel_npz, allow_pickle=False) as panel:
        panel_arrays = {key: panel[key] for key in panel.files}
    exact("query_ptr", source_arrays["query_ptr"], panel_arrays["query_ptr"])
    exact("labels", source_arrays["labels"], panel_arrays["molecule_label"])
    exact("group_ids", source_arrays["group_ids"].astype("U"), panel_arrays["query_formula"].astype("U"))
    exact("query_ids", source_arrays["query_ids"].astype("U"), panel_arrays["query_ik14"].astype("U"))
    exact("candidate_ids", source_arrays["candidate_ids"].astype("U"), panel_arrays["molecule_ik14"].astype("U"))

    modules = tuple(map(str, np.asarray(source_arrays["module_names"]).tolist()))
    if modules != tuple(map(str, freeze.get("module_names", []))):
        raise ValueError("DATA_LEAKAGE_GUARD: module order differs from frozen model")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    if modules != tuple(map(str, checkpoint.get("module_names", ()))):
        raise ValueError("DATA_LEAKAGE_GUARD: checkpoint module order mismatch")
    statuses = {row["name"]: row["status"] for row in registry.get("modules", [])}
    unqualified = [name for name in modules if statuses.get(name) != "enabled"]
    if unqualified:
        raise ValueError(f"DATA_LEAKAGE_GUARD: final bundle contains unqualified modules: {unqualified}")
    scores = np.asarray(source_arrays["scores"])
    availability = np.asarray(source_arrays["availability"])
    if scores.shape != availability.shape or scores.shape != (len(modules), len(source_arrays["labels"])):
        raise ValueError("DATA_LEAKAGE_GUARD: score/availability/candidate shapes differ")
    if not np.all(np.isfinite(scores[availability > 0])):
        raise ValueError("DATA_LEAKAGE_GUARD: applicable component scores are non-finite")
    if np.any((availability < 0) | (availability > 1)):
        raise ValueError("DATA_LEAKAGE_GUARD: invalid availability")
    if np.any(np.sum(availability > 0, axis=0) == 0):
        raise ValueError("DATA_LEAKAGE_GUARD: candidate without applicable evidence")

    payload = {key: source_arrays[key] for key in needed}
    if "near_query" in source_arrays:
        exact("near_query", source_arrays["near_query"], panel_arrays["near_query"])
        payload["near_query"] = source_arrays["near_query"]
    payload.update({
        "schema": np.asarray("grand_unified_sealed_external_bundle_v1"),
        "dataset_id": np.asarray(DATASET_ID),
        "evaluation_role": np.asarray("sealed_external_test"),
        "truth_status": np.asarray("frozen_unscored"),
        "allow_final_claim": np.asarray(1, dtype=np.int8),
        "leakage_guard_complete": np.asarray(1, dtype=np.int8),
        "model_frozen_before_scoring": np.asarray(1, dtype=np.int8),
        "panel": np.asarray(args.panel),
        "benchmark_report_sha256": np.asarray(sha256_file(args.benchmark_report)),
        "benchmark_checksums_sha256": np.asarray(sha256_file(args.benchmark_checksums)),
        "benchmark_panel_sha256": np.asarray(sha256_file(args.panel_npz)),
        "component_score_bundle_sha256": np.asarray(sha256_file(args.component_score_bundle)),
        "freeze_contract_sha256": np.asarray(sha256_file(args.freeze_report)),
        "module_registry_sha256": np.asarray(registry_sha),
        "model_sha256": np.asarray(model_sha),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(json.dumps({
        "status": "SEALED_EXTERNAL_BUNDLE_READY_UNSCORED",
        "path": str(args.out),
        "sha256": sha256_file(args.out),
        "panel": args.panel,
        "queries": int(len(source_arrays["query_ptr"]) - 1),
        "candidates": int(len(source_arrays["labels"])),
        "modules": list(modules),
    }, indent=2))


if __name__ == "__main__":
    main()
