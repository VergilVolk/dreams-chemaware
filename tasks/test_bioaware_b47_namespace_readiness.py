#!/usr/bin/env python
"""Dependency-free checks for the B47 truth-blind namespace audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
from zipfile import ZipFile


SCRIPT = Path(__file__).with_name("audit_bioaware_b47_namespace_readiness.py")
SPEC = importlib.util.spec_from_file_location("bioaware_b47_namespace", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        archive_path = Path(directory) / "fixture.zip"
        with ZipFile(archive_path, "w") as archive:
            archive.writestr("root/input/feature.tsv", "id_number\tmz\trtime\tS1\tS2\n")
            archive.writestr("root/input/query.mgf", "BEGIN IONS\nEND IONS\nBEGIN IONS\nEND IONS\n")
            archive.writestr("root/Validation/truth.csv", "feature,identity\n")
        with ZipFile(archive_path) as archive:
            members = MODULE.normalized_members(archive)
            feature = MODULE.unique_suffix(members, "input/feature.tsv")
            caps = MODULE.header_capabilities(MODULE.read_header(archive, feature))
            mgf = MODULE.unique_suffix(members, "input/query.mgf")
            assert MODULE.count_mgf_spectra(archive, mgf) == 2
        assert caps == {
            "columns": 5,
            "has_mz": True,
            "has_rt": True,
            "mz_rt_combined_column": False,
            "sample_abundance_columns": 2,
            "has_sample_abundance": True,
            "has_sample_identifiers": True,
        }
        composite = MODULE.header_capabilities(["m/z_RT(sec)", "S1", "S2"])
        assert composite["has_mz"] and composite["has_rt"]
        assert composite["mz_rt_combined_column"]
        assert composite["sample_abundance_columns"] == 2
        counts = MODULE.scope_counts(members, "root/")
        assert counts["sealed_truth_or_evaluation_members"] == 1
    print("[test_bioaware_b47_namespace_readiness] PASS")


if __name__ == "__main__":
    main()
