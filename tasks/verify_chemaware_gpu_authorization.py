"""Verify a ChemAware GPU authorization and all evidence hashes at job start."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected_route_report: Path | None = None) -> dict:
    body = json.loads(path.read_text(encoding="utf-8"))
    scope = body.get("scope", {})
    if body.get("status") != "CHEMAWARE_GPU_PILOT_AUTHORIZED" or body.get("release_eligible") is not False:
        raise RuntimeError("invalid GPU authorization status")
    expected = {
        "route": "rule_candidate_residual", "stage": "bounded_pilot_only",
        "array_tasks": 5, "max_concurrent_tasks": 1, "gpus_per_task": 1,
        "hours_per_task": 2.0, "maximum_gpu_hours": 10.0,
        "epochs": 2, "max_train_identities": 512,
        "automatic_full_scale_escalation": False,
    }
    if any(scope.get(key) != value for key, value in expected.items()):
        raise RuntimeError("GPU authorization scope mismatch")
    expires = datetime.fromisoformat(body["expires_utc"])
    if expires.tzinfo is None or datetime.now(timezone.utc) >= expires:
        raise RuntimeError("GPU authorization expired")
    authorizer = ROOT / "tasks/authorize_chemaware_gpu_submission.py"
    if body.get("script_sha256") != sha256_file(authorizer):
        raise RuntimeError("GPU authorizer implementation changed after authorization")
    evidence = body.get("evidence", {})
    for name in ("route_report", "data_semantics_report", "rule_admission_report"):
        item = evidence.get(name, {}); evidence_path = Path(item.get("path", ""))
        if not evidence_path.is_file() or item.get("sha256") != sha256_file(evidence_path):
            raise RuntimeError(f"GPU authorization evidence drift: {name}")
    if expected_route_report is not None:
        if Path(evidence["route_report"]["path"]).resolve() != expected_route_report.resolve():
            raise RuntimeError("runtime clean-observability report differs from authorization")
    route = json.loads(Path(evidence["route_report"]["path"]).read_text(encoding="utf-8"))
    semantics = json.loads(Path(evidence["data_semantics_report"]["path"]).read_text(encoding="utf-8"))
    admission = json.loads(Path(evidence["rule_admission_report"]["path"]).read_text(encoding="utf-8"))
    if route.get("status") != "CLEAN_RULE_OBSERVABILITY_PASS" or route.get("pass_to_gpu_training") is not True:
        raise RuntimeError("clean observability is no longer admissible")
    if semantics.get("status") != "CHEMAWARE_DATA_SEMANTICS_PASS" or semantics.get("development_training_admissible") is not True:
        raise RuntimeError("data semantics is no longer admissible")
    if (admission.get("status") != "CHEMAWARE_RULE_LIBRARY_EMPIRICALLY_ADMITTED"
            or admission.get("formal_training_authorized") is not True
            or admission.get("formula_disjoint_confirmation_pass") is not True):
        raise RuntimeError("rule library is no longer empirically admitted")
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-route-report", type=Path, required=True)
    args = parser.parse_args()
    verify(args.authorization, args.expected_route_report)
    print("CHEMAWARE_GPU_AUTHORIZATION_VERIFIED")


if __name__ == "__main__":
    main()
