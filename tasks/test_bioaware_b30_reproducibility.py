#!/usr/bin/env python3
"""Dependency-free checks for the B30 deterministic replay validator."""

from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path

from validate_bioaware_b30_reproducibility import FILES, compare


def write_fixture(directory: Path, suffix: str = "") -> None:
    directory.mkdir(parents=True)
    report = {
        "status": "bioaware_b30_cross_source_sink_veto_complete",
        "protocol": "fixed",
        "strictly_better_action_than_B17": True,
        "frozen_B17_comparator": {"row_risk_net_lambda2": 43},
        "nested_row_oof": {
            "delta_recall1": 0.0593,
            "corrected": 54,
            "introduced": 3,
            "risk_net_lambda2": 48,
        },
        "nested_physical_oof": {"risk_net_lambda2": 44},
        "folds": [{"source": "held"}],
        "gates": {"pass": True},
        "contracts": {"P2b_used": False},
    }
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    for name in FILES:
        with gzip.open(directory / name, "wb") as handle:
            handle.write(("a,b\n1,2\n" + suffix).encode())


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first, second = root / "a", root / "b"
        write_fixture(first)
        write_fixture(second)
        assert compare(first, second)["corrected"] == 54
        with gzip.open(second / FILES[0], "wb") as handle:
            handle.write(b"a,b\n1,3\n")
        try:
            compare(first, second)
        except RuntimeError as error:
            assert "replay mismatch" in str(error)
        else:
            raise AssertionError("mismatched replay was accepted")
    print("[test_bioaware_b30_reproducibility] PASS")


if __name__ == "__main__":
    main()
