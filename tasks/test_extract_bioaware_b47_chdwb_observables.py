#!/usr/bin/env python
"""Dependency-free checks for exact-member B47 observable extraction."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
from zipfile import ZipFile


SCRIPT = Path(__file__).with_name("extract_bioaware_b47_chdwb_observables.py")
SPEC = importlib.util.spec_from_file_location("bioaware_b47_extract", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive_path = root / "mixed.zip"
        output = root / "observable"
        output.mkdir()
        prefix = "Supplementary Data 2. Head-to-head comparison between metabolite annotation algorithms/"
        with ZipFile(archive_path, "w") as archive:
            archive.writestr(prefix + MODULE.SELECTED["feature_table.csv"], "name,mz,rt,S1\n")
            archive.writestr(prefix + MODULE.SELECTED["sample_information.csv"], "sample,class\n")
            archive.writestr(prefix + MODULE.SELECTED["query_spectra.mgf"], "BEGIN IONS\nEND IONS\n")
            archive.writestr(prefix + "Validation/truth.csv", "feature,truth\n")
        records = MODULE.extract_selected(archive_path, output)
        assert set(records) == set(MODULE.SELECTED)
        assert set(path.name for path in output.iterdir()) == set(MODULE.SELECTED)
        assert not (output / "truth.csv").exists()
        for name, record in records.items():
            assert record["sha256"] == MODULE.file_digest(output / name)
    print("[test_extract_bioaware_b47_chdwb_observables] PASS")


if __name__ == "__main__":
    main()
