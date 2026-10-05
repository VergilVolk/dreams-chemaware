#!/usr/bin/env python
"""Dependency-free unit checks for the B47 ZIP metadata preflight."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
from zipfile import ZipFile


SCRIPT = Path(__file__).with_name("preflight_bioaware_b47_external_bundle.py")
SPEC = importlib.util.spec_from_file_location("bioaware_b47_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    assert MODULE.safe_member("inputs/peak_table.csv")
    assert not MODULE.safe_member("../truth.csv")
    assert not MODULE.safe_member("/absolute/path.csv")
    assert MODULE.provisional_namespace("inputs/sample_info.csv") == "observable_input_like"
    assert (
        MODULE.provisional_namespace("validation/ground_truth.csv")
        == "sealed_truth_or_evaluation_like"
    )
    assert MODULE.provisional_namespace("Scripts/run.R") == "code_like"
    assert MODULE.provisional_namespace("MetDNA3/output.csv") == "algorithm_output_like"
    assert (
        MODULE.provisional_namespace(
            "Supplementary Data 5. External validation of MSMICA results/inputs/sample_info.csv"
        )
        == "observable_input_like"
    )
    assert (
        MODULE.meaningful_member_name(
            "__MACOSX/Supplementary Data 5. External validation of MSMICA results/._file.csv"
        )
        == "._file.csv"
    )

    with tempfile.TemporaryDirectory() as directory:
        archive_path = Path(directory) / "safe.zip"
        with ZipFile(archive_path, "w") as archive:
            archive.writestr("inputs/peak_table.csv", "name,mz,rt\n")
            archive.writestr("validation/ground_truth.csv", "feature,truth\n")
        report = MODULE.inspect_archive(archive_path)
        assert report["members"] == 2
        assert report["unsafe_member_paths"] == []
        assert report["casefold_duplicate_paths"] == []
        assert report["provisional_namespace_counts"] == {
            "observable_input_like": 1,
            "sealed_truth_or_evaluation_like": 1,
        }
    print("[test_bioaware_b47_external_bundle_preflight] PASS")


if __name__ == "__main__":
    main()
