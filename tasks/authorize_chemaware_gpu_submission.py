"""Fail-closed authorization for bounded ChemAware GPU pilots.

This program is intentionally run on the login/CPU node *before* ``sbatch``.
It writes an immutable authorization ledger only when the route-specific CPU
causal screen and the common data gate passed.  It also caps the requested
GPU exposure; a training program cannot turn a pilot authorization into a
full-scale experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROUTE_PASS = {
    "rule_candidate_residual": ("CLEAN_RULE_OBSERVABILITY_PASS", "pass_to_gpu_training"),
    "crossmodal_structure": ("CHEMAWARE_CROSSMODAL_STRUCTURE_SCREEN_PASS", "pass_to_gpu_pilot"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=tuple(ROUTE_PASS), required=True)
    parser.add_argument("--route-report", type=Path, required=True)
    parser.add_argument("--data-semantics-report", type=Path, required=True)
    parser.add_argument("--rule-admission-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--array-tasks", type=int, required=True)
    parser.add_argument("--gpus-per-task", type=int, default=1)
    parser.add_argument("--hours-per-task", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--max-train-identities", type=int, required=True)
    parser.add_argument("--max-total-gpu-hours", type=float, default=10.0)
    return parser.parse_args()


def require_report(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def authorize(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite authorization: {args.output}")
    semantics = require_report(args.data_semantics_report)
    if (semantics.get("status") != "CHEMAWARE_DATA_SEMANTICS_PASS"
            or semantics.get("development_training_admissible") is not True):
        raise RuntimeError("GPU blocked: data-semantics gate did not pass")
    route_report = require_report(args.route_report)
    expected_status, pass_field = ROUTE_PASS[args.route]
    if route_report.get("status") != expected_status or route_report.get(pass_field) is not True:
        raise RuntimeError(f"GPU blocked: {args.route} CPU causal screen did not pass")
    evidence = {
        "route_report": {"path": str(args.route_report), "sha256": sha256_file(args.route_report)},
        "data_semantics_report": {"path": str(args.data_semantics_report), "sha256": sha256_file(args.data_semantics_report)},
    }
    if args.route == "rule_candidate_residual":
        if args.rule_admission_report is None:
            raise RuntimeError("GPU blocked: empirical rule-library admission is mandatory")
        admission = require_report(args.rule_admission_report)
        if (admission.get("status") != "CHEMAWARE_RULE_LIBRARY_EMPIRICALLY_ADMITTED"
                or admission.get("formal_training_authorized") is not True
                or admission.get("formula_disjoint_confirmation_pass") is not True):
            raise RuntimeError("GPU blocked: empirical rule-library admission did not pass")
        evidence["rule_admission_report"] = {
            "path": str(args.rule_admission_report), "sha256": sha256_file(args.rule_admission_report),
        }
    if args.array_tasks < 1 or args.array_tasks > 5:
        raise RuntimeError("GPU blocked: pilot permits 1-5 matched arms")
    if args.gpus_per_task != 1:
        raise RuntimeError("GPU blocked: exactly one GPU per task is permitted")
    if args.hours_per_task <= 0 or args.hours_per_task > 2:
        raise RuntimeError("GPU blocked: pilot walltime cap is two hours per arm")
    if args.epochs < 1 or args.epochs > 2:
        raise RuntimeError("GPU blocked: pilot cap is two epochs")
    if args.max_train_identities < 1 or args.max_train_identities > 512:
        raise RuntimeError("GPU blocked: pilot cap is 512 train identities")
    exposure = args.array_tasks * args.gpus_per_task * args.hours_per_task
    if exposure > args.max_total_gpu_hours:
        raise RuntimeError("GPU blocked: requested exposure exceeds total GPU-hour cap")
    created = datetime.now(timezone.utc)
    return {
        "status": "CHEMAWARE_GPU_PILOT_AUTHORIZED",
        "release_eligible": False,
        "scope": {
            "route": args.route, "stage": "bounded_pilot_only",
            "array_tasks": args.array_tasks, "max_concurrent_tasks": 1,
            "gpus_per_task": args.gpus_per_task, "hours_per_task": args.hours_per_task,
            "maximum_gpu_hours": exposure, "epochs": args.epochs,
            "max_train_identities": args.max_train_identities,
            "automatic_full_scale_escalation": False,
        },
        "evidence": evidence,
        "created_utc": created.isoformat(),
        "expires_utc": (created + timedelta(hours=24)).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
    }


def main() -> None:
    args = arguments()
    report = authorize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
