from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tasks" / "preflight_kgmn_oep003284_inputs.py"


def write_panel(root: Path, name: str) -> None:
    panel = root / name
    panel.mkdir(parents=True)
    (panel / "spectra.mgf").write_text("BEGIN IONS\nPEPMASS=100\nEND IONS\n", encoding="utf-8")
    with (panel / "data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "mz", "rt", "s1", "s2"])
        writer.writerow(["M100T10", 100.0, 10.0, 1.0, 2.0])
    with (panel / "sample.info.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample.name", "group"])
        writer.writerow(["s1", "g1"])
        writer.writerow(["s2", "g2"])


def test_ready_both_polarities(tmp_path: Path) -> None:
    inputs = tmp_path / "OEP003284"
    write_panel(inputs, "POS")
    write_panel(inputs, "NEG")
    output = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input-root", str(inputs), "--output", str(output), "--require-ready"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ready"] is True
    assert report["dreams_edge_ready"] is True
    assert report["panels"]["positive"]["status"] == "ready"
    assert report["panels"]["negative"]["status"] == "ready"


def test_author_ready_mzxml_is_not_dreams_edge_ready(tmp_path: Path) -> None:
    inputs = tmp_path / "OEP003284"
    write_panel(inputs, "POS")
    write_panel(inputs, "NEG")
    for name in ("POS", "NEG"):
        mgf = inputs / name / "spectra.mgf"
        mgf.rename(inputs / name / "spectra.mzXML")
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-root",
            str(inputs),
            "--output",
            str(output),
            "--require-ready",
            "--require-dreams-edge",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ready"] is True
    assert report["dreams_edge_ready"] is False


def test_multiple_mzxml_are_ready_with_explicit_author_mapping_bridge(tmp_path: Path) -> None:
    inputs = tmp_path / "OEP003284"
    write_panel(inputs, "POS")
    write_panel(inputs, "NEG")
    for name in ("POS", "NEG"):
        panel = inputs / name
        (panel / "spectra.mgf").unlink()
        (panel / "run_1.mzXML").write_text("raw1", encoding="utf-8")
        (panel / "run_2.mzXML").write_text("raw2", encoding="utf-8")
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-root",
            str(inputs),
            "--output",
            str(output),
            "--require-ready",
            "--require-dreams-edge",
            "--allow-author-mapped-mzxml",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["dreams_edge_ready"] is True
    assert report["panels"]["positive"]["ms2_file_count"] == 2
    assert report["panels"]["positive"]["dreams_edge_mode"] == "metdna2_author_mapped_mzxml_cache"


def test_rejects_truth_bearing_peak_table(tmp_path: Path) -> None:
    inputs = tmp_path / "OEP003284"
    write_panel(inputs, "POS")
    write_panel(inputs, "NEG")
    path = inputs / "POS" / "data.csv"
    path.write_text("name,mz,rt,s1,s2,inchikey\nM100T10,100,10,1,2,LEAK\n", encoding="utf-8")
    output = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input-root", str(inputs), "--output", str(output), "--require-ready"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ready"] is False
    assert any("forbidden annotation columns" in item for item in report["panels"]["positive"]["problems"])


def test_raw_only_is_not_runnable(tmp_path: Path) -> None:
    inputs = tmp_path / "OEP003284"
    panel = inputs / "positive"
    panel.mkdir(parents=True)
    (panel / "sample.raw").write_bytes(b"vendor")
    output = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input-root", str(inputs), "--output", str(output), "--require-ready"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ready"] is False
    assert report["panels"]["positive"]["status"] == "raw_only"
