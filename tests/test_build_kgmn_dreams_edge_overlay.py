import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("tasks/build_kgmn_dreams_edge_overlay.py")
SOURCE = Path("third_party/MetDNA2")


def test_builds_one_provenance_locked_recursive_edge_hook(tmp_path: Path) -> None:
    output = tmp_path / "MetDNA2_overlay"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--source", str(SOURCE), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output / "DREAMS_EDGE_OVERLAY_MANIFEST.json").read_text())
    assert manifest["author_commit"] == "5685ab219269c2f35cd5087655b0470b2da4d93c"
    assert manifest["contracts"]["author_baseline_uses_overlay"] is False
    assert manifest["contracts"]["hook_scope"] == "dynamic recursive feature-feature MS2 edges only"
    patched = (output / "R" / "RecursiveAnnotationMRN.R").read_text(encoding="utf-8")
    assert patched.count('getOption("MetDNA2.recursive_edge_score_hook", NULL)') == 1
    assert patched.count("'recursive_edge_score_hook',") == 1
    assert "PSOCK workers do not inherit getOption() state" in patched
    assert "candidate generation" not in patched.lower()


def test_refuses_to_overwrite_overlay(tmp_path: Path) -> None:
    output = tmp_path / "MetDNA2_overlay"
    output.mkdir()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--source", str(SOURCE), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "refusing to overwrite" in result.stderr
