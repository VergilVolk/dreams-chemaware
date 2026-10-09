"""Open the frozen Enveda benchmark exactly once for a frozen unified model.

This program is deliberately separate from training.  It reserves an immutable
opening ledger before loading labels, verifies the complete hash chain, and
leaves a failed ledger in place if any guard or evaluation step fails.  A
second run with the same ledger path is therefore rejected rather than used
for model or hyperparameter selection.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch

from grand_unified_evidence_core import AllModuleEvidenceModel
from train_grand_unified_evidence_model import (
    assert_sealed_external_test,
    evaluate,
    load_bundle,
    scalar_text,
    sha256_file,
)


DATASET_ID = "enveda_180_filtered_20260713"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def require_equal(label: str, observed: str, expected: str) -> None:
    if observed != expected:
        raise ValueError(
            f"DATA_LEAKAGE_GUARD: {label} hash mismatch; "
            f"bundle={observed!r}, actual={expected!r}"
        )


def validate_freeze_chain(
    bundle: dict,
    freeze_report: dict,
    model_sha256: str,
    freeze_report_sha256: str,
    registry_sha256: str,
    benchmark_report: dict,
    benchmark_report_sha256: str,
    benchmark_checksums_sha256: str,
    benchmark_panel_sha256: str,
    component_score_bundle_sha256: str,
) -> tuple[str, ...]:
    assert_sealed_external_test(bundle, Path("sealed_external_bundle"))
    if freeze_report.get("freeze_contract_status") != "DEVELOPMENT_MODEL_FROZEN_EXTERNAL_UNOPENED":
        raise ValueError("DATA_LEAKAGE_GUARD: model freeze report is not external-unopened")
    if freeze_report.get("external_test_opened") is not False:
        raise ValueError("DATA_LEAKAGE_GUARD: freeze report says external test was opened")
    require_equal("model", scalar_text(bundle, "model_sha256"), model_sha256)
    require_equal("freeze contract", scalar_text(bundle, "freeze_contract_sha256"), freeze_report_sha256)
    require_equal("module registry", scalar_text(bundle, "module_registry_sha256"), registry_sha256)
    require_equal("benchmark report", scalar_text(bundle, "benchmark_report_sha256"), benchmark_report_sha256)
    require_equal("benchmark checksums", scalar_text(bundle, "benchmark_checksums_sha256"), benchmark_checksums_sha256)
    require_equal("benchmark panel", scalar_text(bundle, "benchmark_panel_sha256"), benchmark_panel_sha256)
    require_equal(
        "component score bundle",
        scalar_text(bundle, "component_score_bundle_sha256"),
        component_score_bundle_sha256,
    )
    if freeze_report.get("frozen_model_sha256") != model_sha256:
        raise ValueError("DATA_LEAKAGE_GUARD: model does not match freeze report")
    if freeze_report.get("module_registry_sha256") != registry_sha256:
        raise ValueError("DATA_LEAKAGE_GUARD: registry does not match freeze report")
    if benchmark_report.get("dataset_id") != DATASET_ID:
        raise ValueError("DATA_LEAKAGE_GUARD: benchmark report is not the authorized Enveda-180 asset")
    if benchmark_report.get("evaluation_role") != "sealed_external_test":
        raise ValueError("DATA_LEAKAGE_GUARD: benchmark role is not sealed_external_test")
    if benchmark_report.get("truth_status") != "frozen_unscored":
        raise ValueError("DATA_LEAKAGE_GUARD: benchmark truth is not frozen_unscored")
    if benchmark_report.get("performance_scores_opened") is not False:
        raise ValueError("DATA_LEAKAGE_GUARD: benchmark report says performance was already opened")
    guard = benchmark_report.get("leakage_guard", {})
    if guard.get("exclusion_policy_complete") is not True:
        raise ValueError("DATA_LEAKAGE_GUARD: benchmark exclusion policy is incomplete")
    if int(guard.get("selected_consumed_identity_overlap", -1)) != 0:
        raise ValueError("DATA_LEAKAGE_GUARD: selected Enveda identities overlap consumed sources")
    if int(guard.get("selected_consumed_spectrum_overlap", -1)) != 0:
        raise ValueError("DATA_LEAKAGE_GUARD: selected Enveda spectra overlap consumed sources")
    model_modules = tuple(map(str, freeze_report.get("module_names", [])))
    bundle_modules = tuple(map(str, np.asarray(bundle["module_names"]).tolist()))
    if bundle_modules != model_modules:
        raise ValueError(
            "DATA_LEAKAGE_GUARD: external module order differs from the frozen model; "
            f"bundle={bundle_modules}, model={model_modules}"
        )
    return bundle_modules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--freeze-report", type=Path, required=True)
    parser.add_argument("--test-bundle", type=Path, required=True)
    parser.add_argument("--component-score-bundle", type=Path, required=True)
    parser.add_argument("--module-registry", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--benchmark-checksums", type=Path, required=True)
    parser.add_argument("--benchmark-panel", type=Path, required=True)
    parser.add_argument("--comparison-baseline", required=True)
    parser.add_argument("--opening-ledger", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    required = (
        args.model, args.freeze_report, args.test_bundle, args.component_score_bundle, args.module_registry,
        args.benchmark_report, args.benchmark_checksums, args.benchmark_panel,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required frozen artifacts are absent: {missing}")
    if args.out_report.exists():
        raise FileExistsError(f"external result already exists: {args.out_report}")
    args.opening_ledger.parent.mkdir(parents=True, exist_ok=True)
    reservation = {
        "schema": "unified_external_opening_ledger_v1",
        "status": "OPENING_RESERVED_LABELS_NOT_YET_LOADED",
        "dataset_id": DATASET_ID,
        "reserved_at_utc": utc_now(),
        "model_path": str(args.model),
        "model_sha256": sha256_file(args.model),
        "test_bundle_path": str(args.test_bundle),
        "test_bundle_sha256": sha256_file(args.test_bundle),
    }
    # Exclusive creation is the one-time gate.  Never delete this file to retry.
    with args.opening_ledger.open("x", encoding="utf-8") as handle:
        json.dump(reservation, handle, indent=2)

    try:
        freeze_report = json.loads(args.freeze_report.read_text(encoding="utf-8"))
        benchmark_report = json.loads(args.benchmark_report.read_text(encoding="utf-8"))
        bundle = load_bundle(args.test_bundle)
        component = load_bundle(args.component_score_bundle)
        with np.load(args.benchmark_panel, allow_pickle=False) as panel:
            frozen_panel = {key: panel[key] for key in panel.files}
        for key in (
            "module_names", "scores", "availability", "query_ptr", "labels",
            "group_ids", "query_ids", "candidate_ids",
        ):
            if not np.array_equal(np.asarray(bundle[key]), np.asarray(component[key])):
                raise ValueError(
                    f"DATA_LEAKAGE_GUARD: sealed bundle field {key} differs from component score bundle"
                )
        panel_bindings = {
            "query_ptr": "query_ptr",
            "labels": "molecule_label",
            "group_ids": "query_formula",
            "query_ids": "query_ik14",
            "candidate_ids": "molecule_ik14",
        }
        for bundle_key, panel_key in panel_bindings.items():
            left = np.asarray(bundle[bundle_key])
            right = np.asarray(frozen_panel[panel_key])
            if left.dtype.kind in "OUS" or right.dtype.kind in "OUS":
                left, right = left.astype("U"), right.astype("U")
            if not np.array_equal(left, right):
                raise ValueError(
                    f"DATA_LEAKAGE_GUARD: sealed bundle field {bundle_key} differs from frozen panel"
                )
        modules = validate_freeze_chain(
            bundle=bundle,
            freeze_report=freeze_report,
            model_sha256=reservation["model_sha256"],
            freeze_report_sha256=sha256_file(args.freeze_report),
            registry_sha256=sha256_file(args.module_registry),
            benchmark_report=benchmark_report,
            benchmark_report_sha256=sha256_file(args.benchmark_report),
            benchmark_checksums_sha256=sha256_file(args.benchmark_checksums),
            benchmark_panel_sha256=sha256_file(args.benchmark_panel),
            component_score_bundle_sha256=sha256_file(args.component_score_bundle),
        )
        if args.comparison_baseline not in modules:
            raise ValueError(
                f"predeclared comparison baseline {args.comparison_baseline!r} is absent"
            )
        device = torch.device(args.device)
        checkpoint = torch.load(args.model, map_location=device, weights_only=True)
        checkpoint_modules = tuple(map(str, checkpoint.get("module_names", ())))
        if checkpoint_modules != modules:
            raise ValueError("DATA_LEAKAGE_GUARD: checkpoint module order mismatch")
        model = AllModuleEvidenceModel(modules).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        query_count = len(np.asarray(bundle["query_ptr"])) - 1
        external = evaluate(
            model, bundle, np.arange(query_count, dtype=np.int64), device,
            comparison_baseline=args.comparison_baseline,
        )
        result = {
            "status": "SEALED_EXTERNAL_TEST_OPENED_ONCE_COMPLETE",
            "schema": "grand_unified_enveda_external_result_v1",
            "dataset_id": DATASET_ID,
            "evaluation_role": "one_time_external_test",
            "opened_at_utc": utc_now(),
            "queries": query_count,
            "predeclared_comparison_baseline": args.comparison_baseline,
            "model_sha256": reservation["model_sha256"],
            "freeze_contract_sha256": sha256_file(args.freeze_report),
            "test_bundle_sha256": reservation["test_bundle_sha256"],
            "benchmark_report_sha256": sha256_file(args.benchmark_report),
            "benchmark_checksums_sha256": sha256_file(args.benchmark_checksums),
            "benchmark_panel_sha256": sha256_file(args.benchmark_panel),
            "module_registry_sha256": sha256_file(args.module_registry),
            "metrics": external,
            "selection_boundary": (
                "No weights, seed, hyperparameters, module set or primary baseline may be "
                "changed in response to this result."
            ),
        }
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out_report, result)
        reservation.update({
            "status": "OPENED_ONCE_COMPLETE_NO_RETRY",
            "completed_at_utc": utc_now(),
            "result_path": str(args.out_report),
            "result_sha256": sha256_file(args.out_report),
        })
        write_json(args.opening_ledger, reservation)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        reservation.update({
            "status": "OPENING_FAILED_NO_AUTOMATIC_RETRY",
            "failed_at_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        write_json(args.opening_ledger, reservation)
        raise


if __name__ == "__main__":
    main()
