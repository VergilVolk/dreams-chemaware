from pathlib import Path


SBATCH = Path("tasks/run_netid_public_dreams_edge_stage.sbatch")


def test_sbatch_is_single_gpu_without_forbidden_memory_request() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH --mem" not in text
    assert "#SBATCH --output=/data02/run01/scv7tsl/DreaMS/" in text
    assert "#SBATCH --error=/data02/run01/scv7tsl/DreaMS/" in text


def test_sbatch_installs_and_validates_before_scientific_execution() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    fetch = text.index("python -u tasks/fetch_bioaware_netid_external.py")
    audit = text.index("python -u tasks/audit_netid_public_release.py")
    encode = text.index("python -u tasks/encode_netid_mouse_liver_dreams.py")
    signal = text.index("python -u tasks/audit_netid_dreams_edge_signal.py")
    assert fetch < audit < encode < signal
    assert "python -u tasks/diagnose_netid_edge_signal_modalities.py" in text
    assert 'if [ "$EDGE_STATUS" = "netid_dreams_edge_signal_failed" ]' in text
    assert "set -euo pipefail" in text
    assert "importlib.import_module(package)" in text
