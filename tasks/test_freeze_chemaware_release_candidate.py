"""Tests for the pre-outer immutable release freeze."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from freeze_chemaware_release_candidate import build_freeze, sha256_file


def dump(path: Path, body: dict) -> None:
    path.write_text(json.dumps(body), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw); summary = root / "summary.json"; report = root / "report.json"
        checkpoint = root / "model.pt"; ledger = root / "ledger.npz"; output = root / "freeze.json"
        checkpoint.write_bytes(b"fixed model"); ledger.write_bytes(b"fixed development")
        dump(report, {"status": "ARM_COMPLETE", "optimization": {"teacher_arm": "rule_response", "outer_fold": 4},
                      "preflight": {"contracts": {"outer_fold_evaluated": False}}})
        dump(summary, {"status": "CANDIDATE_RESIDUAL_CAUSAL_DEVELOPMENT_PASS",
                       "selected_for_outer_confirmation": "rule",
                       "provenance": {"arms": {"rule": {"report_sha256": sha256_file(report),
                                                          "checkpoint_sha256": sha256_file(checkpoint)}}}})
        args = SimpleNamespace(causal_summary=summary, candidate_report=report,
            candidate_checkpoint=checkpoint, development_ledgers=[ledger], output=output,
            outer_fold=4, outer_selection_seed=20260934, bootstrap_draws=10000)
        frozen = build_freeze(args)
        assert frozen["status"] == "CHEMAWARE_RELEASE_CANDIDATE_FROZEN"
        dump(summary, {"status": "CANDIDATE_RESIDUAL_CAUSAL_DEVELOPMENT_FAIL"})
        try:
            build_freeze(args)
        except RuntimeError as error:
            assert "causal development" in str(error)
        else:
            raise AssertionError("failed development candidate was frozen")
    print("release freeze tests passed")


if __name__ == "__main__":
    main()
