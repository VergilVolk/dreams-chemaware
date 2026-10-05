"""Tests for the fail-closed ChemAware GPU authorization ledger."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from authorize_chemaware_gpu_submission import authorize
from verify_chemaware_gpu_authorization import verify


def write(path: Path, body: dict) -> None:
    path.write_text(json.dumps(body), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw); route = root / "route.json"; semantics = root / "semantics.json"
        admission = root / "admission.json"; output = root / "authorization.json"
        write(semantics, {"status": "CHEMAWARE_DATA_SEMANTICS_PASS", "development_training_admissible": True})
        write(route, {"status": "CLEAN_RULE_OBSERVABILITY_PASS", "pass_to_gpu_training": True})
        write(admission, {"status": "CHEMAWARE_RULE_LIBRARY_EMPIRICALLY_ADMITTED",
                          "formal_training_authorized": True, "formula_disjoint_confirmation_pass": True})
        args = SimpleNamespace(route="rule_candidate_residual", route_report=route,
            data_semantics_report=semantics, rule_admission_report=admission, output=output,
            array_tasks=5, gpus_per_task=1, hours_per_task=2.0, epochs=2,
            max_train_identities=512, max_total_gpu_hours=10.0)
        authorized = authorize(args)
        assert authorized["status"] == "CHEMAWARE_GPU_PILOT_AUTHORIZED"
        write(output, authorized)
        assert verify(output, route)["status"] == "CHEMAWARE_GPU_PILOT_AUTHORIZED"
        output.unlink()
        args.hours_per_task = 2.1
        try:
            authorize(args)
        except RuntimeError as error:
            assert "walltime" in str(error)
        else:
            raise AssertionError("over-budget pilot was authorized")
        args.hours_per_task = 2.0
        write(route, {"status": "RETROSPECTIVE_DIAGNOSTIC_ONLY", "pass_to_gpu_training": False})
        try:
            authorize(args)
        except RuntimeError as error:
            assert "causal screen" in str(error)
        else:
            raise AssertionError("failed causal screen was authorized")
    print("GPU authorization tests passed")


if __name__ == "__main__":
    main()
