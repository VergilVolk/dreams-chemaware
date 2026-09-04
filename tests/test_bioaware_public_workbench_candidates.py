from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_bioaware_public_workbench_candidates",
    ROOT / "tasks" / "audit_bioaware_public_workbench_candidates.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_concatenated_mwtab_blocks_are_all_read() -> None:
    source = ROOT / "data" / "reference" / "bioaware_public_cohort_probe_20260901"
    assert len(MODULE.read_concatenated_json(source / "ST001264__mwtab__json")) == 2
    assert len(MODULE.read_concatenated_json(source / "ST003550__mwtab__json")) == 2
    assert len(MODULE.read_concatenated_json(source / "ST000923__mwtab__json")) == 4


def test_candidates_fail_closed_for_distinct_reasons() -> None:
    source = ROOT / "data" / "reference" / "bioaware_public_cohort_probe_20260901"
    reports = {study: MODULE.audit_study(study, source) for study in MODULE.STUDIES}

    lipid = reports["ST001264"]
    assert lipid["explicit_negative_dda"]
    assert lipid["explicit_negative_ms2"]
    assert not lipid["negative_results"]["structure_or_annotation_column_present"]
    assert not lipid["pass"]

    tracer = reports["ST003550"]
    assert tracer["explicit_negative_dda"]
    assert tracer["explicit_negative_ms2"]
    assert tracer["structure_counts"]["unique_hmdb_ids"] == 14
    assert not tracer["pass"]

    stool = reports["ST000923"]
    assert stool["negative_metabolite_rows"] == 206
    assert not stool["explicit_negative_dda"]
    assert not stool["explicit_negative_ms2"]
    assert stool["structure_counts"]["inchi_key_rows"] == 0
    assert stool["files"]["raw_bytes"] > 400_000_000_000
    assert not stool["pass"]
