import subprocess
import sys
from pathlib import Path


SCRIPT = Path("tasks/validate_kgmn_dreams_edge_calibration_manifest.py")
MANIFEST = Path("data/validation/kgmn_dreams_edge_calibration_manifest_20260831")


def test_frozen_local_manifest_passes_all_invariants() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest-dir", str(MANIFEST)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "validation_passed" in result.stdout
