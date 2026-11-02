"""Static contract for the frozen TransExION GNPS auxiliary job."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "tasks/run_transexion_gnps_same_axis_1gpu.sbatch"


def main() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    required = (
        "#SBATCH --gpus=1",
        "#SBATCH --cpus-per-task=10",
        "tasks/test_transexion_same_axis_adapter.py",
        "--device cuda --batch-size 16",
        "TransExION (GNPS-trained; auxiliary)",
        "scores_identity_disjoint.npz",
        "scores_formula_disjoint.npz",
        "--reference-method full23_v132",
        "--bootstrap-resamples 10000",
        "--evaluation-role development --truth-status consumed",
    )
    missing = [value for value in required if value not in text]
    assert not missing, missing
    assert "--allow-final-claim" not in text
    assert "http://" not in text and "https://" not in text
    print("PASS: TransExION GNPS same-axis sbatch contracts")


if __name__ == "__main__":
    main()
