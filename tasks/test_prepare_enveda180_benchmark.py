from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from prepare_enveda180_scoreblind_manifest import secondary_spectrum_hash


ROOT = Path(__file__).resolve().parents[1]
PROTON = 1.007276466621


def write_fixture(path: Path) -> None:
    smiles_values = ["CCCO", "CC(O)C", "CCOC"]  # three C3H8O connectivities
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        index = 0
        for molecule_index, smiles in enumerate(smiles_values):
            mol = Chem.MolFromSmiles(smiles)
            key = Chem.MolToInchiKey(mol)
            formula = rdMolDescriptors.CalcMolFormula(mol)
            precursor = Descriptors.ExactMolWt(mol) + PROTON
            for energy in (20, 60):
                index += 1
                handle.write("BEGIN IONS\n")
                handle.write(f"TITLE=fixture_{index:03d}\n")
                handle.write(f"PEPMASS={precursor:.8f}\n")
                handle.write("CHARGE=1+\nADDUCT=[M+H]+\n")
                handle.write(f"COLLISION_ENERGIES={energy}\n")
                handle.write(f"INCHIKEY={key}\nFORMULA={formula}\nSMILES={smiles}\n")
                handle.write("IONMODE=ESI Positive\n")
                for peak in range(12):
                    mz = 20.0 + peak * 3.0 + molecule_index * 0.013
                    intensity = (peak + 1) ** (1.0 + energy / 100.0) + molecule_index
                    handle.write(f"{mz:.6f} {intensity:.6f}\n")
                handle.write("END IONS\n")


def test_scoreblind_enveda_builder_end_to_end(tmp_path):
    source = tmp_path / "fixture.mgf.gz"
    audit = tmp_path / "audit"
    benchmark = tmp_path / "benchmark"
    consumed = tmp_path / "consumed.csv"
    consumed.write_text("ik14,formula,spectrum_hash\nDUMMYIDENTITY1,C2H6,\n", encoding="utf-8")
    registry = tmp_path / "consumed_sources.json"
    registry.write_text(json.dumps({
        "schema": "unified_consumed_source_registry_v1",
        "sources": [{
            "name": "fixture_consumed", "kind": "csv",
            "path": str(consumed), "required": True,
        }],
    }), encoding="utf-8")
    write_fixture(source)
    subprocess.run(
        [
            sys.executable, str(ROOT / "tasks/prepare_enveda180_scoreblind_manifest.py"),
            "--mgf", str(source), "--out", str(audit), "--progress-every", "1000",
            "--exclusion-registry", str(registry),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable, str(ROOT / "tasks/build_enveda180_cross_condition_benchmark.py"),
            "--source-mgf", str(source), "--audit", str(audit), "--out", str(benchmark),
            "--minimum-queries", "1", "--minimum-formula-queries", "1",
        ],
        check=True,
    )
    report = json.loads((benchmark / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "ENVEDA180_CROSS_CONDITION_BENCHMARK_FROZEN"
    assert report["performance_scores_opened"] is False
    assert report["leakage_guard"]["exclusion_policy_complete"] is True
    assert report["leakage_guard"]["selected_consumed_identity_overlap"] == 0
    assert report["leakage_guard"]["selected_consumed_spectrum_overlap"] == 0
    assert report["identity_disjoint"]["queries"] == 3
    assert report["formula_disjoint"]["queries"] == 3
    assert report["identity_open_set"]["queries"] == 3
    assert report["identity_disjoint"]["query_adducts"] == {"[M+H]+": 3}
    assert (
        report["identity_open_set"]["match_queries"]
        + report["identity_open_set"]["no_match_queries"]
        == 3
    )
    with np.load(benchmark / "panel_identity_disjoint.npz", allow_pickle=False) as body:
        assert body["query_adduct"].tolist() == ["[M+H]+"] * 3
        ptr = body["query_ptr"]
        labels = body["molecule_label"]
        for q in range(len(ptr) - 1):
            local = labels[ptr[q]:ptr[q + 1]]
            assert local[0] == 1
            assert int(local.sum()) == 1
    with np.load(benchmark / "panel_identity_open_set.npz", allow_pickle=False) as body:
        ptr = body["query_ptr"]
        labels = body["molecule_label"]
        has_match = body["query_has_match"]
        for q in range(len(ptr) - 1):
            local = labels[ptr[q]:ptr[q + 1]]
            assert int(local.sum()) == int(has_match[q])
            if not has_match[q]:
                assert len(local) >= 2


def test_missing_required_consumed_source_fails_closed(tmp_path):
    source = tmp_path / "fixture.mgf.gz"
    audit = tmp_path / "audit"
    registry = tmp_path / "consumed_sources.json"
    write_fixture(source)
    registry.write_text(json.dumps({
        "schema": "unified_consumed_source_registry_v1",
        "sources": [{
            "name": "missing_required", "kind": "csv",
            "path": str(tmp_path / "missing.csv"), "required": True,
        }],
    }), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "tasks/prepare_enveda180_scoreblind_manifest.py"),
            "--mgf", str(source), "--out", str(audit),
            "--exclusion-registry", str(registry),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "required exclusion source missing" in (result.stdout + result.stderr)


def test_secondary_msnlib_hash_contract_is_not_silently_ignored(tmp_path):
    source = tmp_path / "fixture.mgf.gz"
    audit = tmp_path / "audit"
    consumed = tmp_path / "consumed.csv"
    registry = tmp_path / "consumed_sources.json"
    write_fixture(source)
    peaks = [
        (20.0 + peak * 3.0, (peak + 1) ** 1.2)
        for peak in range(12)
    ]
    secondary = secondary_spectrum_hash(peaks)
    assert len(secondary) == 32
    consumed.write_text(
        f"ik14,formula,spectrum_hash\nDUMMYIDENTITY1,C2H6,{secondary}\n",
        encoding="utf-8",
    )
    registry.write_text(json.dumps({
        "schema": "unified_consumed_source_registry_v1",
        "sources": [{
            "name": "msnlib_style", "kind": "csv",
            "path": str(consumed), "required": True,
        }],
    }), encoding="utf-8")
    subprocess.run(
        [
            sys.executable, str(ROOT / "tasks/prepare_enveda180_scoreblind_manifest.py"),
            "--mgf", str(source), "--out", str(audit),
            "--exclusion-registry", str(registry),
        ],
        check=True,
    )
    frame = pd.read_csv(audit / "eligible_records.csv.gz")
    assert int(frame["consumed_spectrum_overlap"].astype(bool).sum()) == 1
    report = json.loads((audit / "report.json").read_text(encoding="utf-8"))
    assert report["consumed_spectrum_hash_count"]["secondary_blake2b"] == 1


def test_complete_manifest_can_be_incrementally_upgraded(tmp_path):
    source = tmp_path / "fixture.mgf.gz"
    base = tmp_path / "base"
    upgraded = tmp_path / "upgraded"
    old_csv = tmp_path / "old.csv"
    added_csv = tmp_path / "added.csv"
    old_registry = tmp_path / "old_registry.json"
    new_registry = tmp_path / "new_registry.json"
    write_fixture(source)
    old_csv.write_text("ik14,formula,spectrum_hash\nDUMMYIDENTITY1,C2H6,\n", encoding="utf-8")
    old_registry.write_text(json.dumps({
        "schema": "unified_consumed_source_registry_v1",
        "sources": [{"name": "old", "kind": "csv", "path": str(old_csv), "required": True}],
    }), encoding="utf-8")
    subprocess.run([
        sys.executable, str(ROOT / "tasks/prepare_enveda180_scoreblind_manifest.py"),
        "--mgf", str(source), "--out", str(base),
        "--exclusion-registry", str(old_registry),
    ], check=True)
    base_frame = pd.read_csv(base / "eligible_records.csv.gz")
    # A recompressed/copied historical gzip may not retain its old byte hash.
    # The upgrader must then require the stronger row/content/count contract.
    base_report_path = base / "report.json"
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    base_report["manifest_sha256"] = None
    base_report["conflicting_hashes_sha256"] = None
    base_report["source_mgf_sha256"] = None
    base_report_path.write_text(json.dumps(base_report), encoding="utf-8")
    # Simulate the original v1 schema, which predates the MSnLib-compatible
    # secondary hash. The upgrader should recover it by a peak-only MGF pass.
    base_frame.drop(columns=["spectrum_hash_secondary", "consumed_spectrum_overlap"]).to_csv(
        base / "eligible_records.csv.gz", index=False, compression="gzip",
    )
    added_ik = str(base_frame.iloc[0]["ik14"])
    added_csv.write_text(
        f"ik14,formula,spectrum_hash\n{added_ik},DUMMYFORMULA,\n", encoding="utf-8",
    )
    new_registry.write_text(json.dumps({
        "schema": "unified_consumed_source_registry_v1",
        "sources": [
            {"name": "old", "kind": "csv", "path": str(old_csv), "required": True},
            {"name": "added", "kind": "csv", "path": str(added_csv), "required": True},
        ],
    }), encoding="utf-8")
    subprocess.run([
        sys.executable, str(ROOT / "tasks/upgrade_enveda180_scoreblind_manifest.py"),
        "--base-audit", str(base), "--out", str(upgraded),
        "--source-mgf", str(source),
        "--expected-source-bytes", str(source.stat().st_size),
        "--expected-source-md5", __import__("hashlib").md5(source.read_bytes()).hexdigest(),
        "--exclusion-registry", str(new_registry), "--chunk-size", "2",
    ], check=True)
    frame = pd.read_csv(upgraded / "eligible_records.csv.gz")
    report = json.loads((upgraded / "report.json").read_text(encoding="utf-8"))
    assert len(frame) == len(base_frame)
    assert frame.loc[frame["ik14"] == added_ik, "consumed_identity_overlap"].all()
    assert report["incremental_upgrade"]["added_sources"] == ["added"]
    assert report["incremental_upgrade"]["base_manifest_byte_hash_status"] == "not_recorded"
    assert report["incremental_upgrade"]["base_manifest_semantic_contract_pass"] is True
    assert report["incremental_upgrade"]["base_conflicts_byte_hash_status"] == "not_recorded"
    assert report["incremental_upgrade"]["base_conflicts_semantic_contract_pass"] is True
    assert report["incremental_upgrade"]["secondary_spectrum_hashes"].startswith("reconstructed")
    assert report["incremental_upgrade"]["source_mgf_binding"] == "independent_bytes_and_md5_match"
    assert report["incremental_upgrade"]["all_current_exclusion_sources_reloaded"] is True
    assert frame["spectrum_hash_secondary"].str.fullmatch(r"[0-9a-f]{32}").all()
    assert report["exclusion_policy_complete"] is True

