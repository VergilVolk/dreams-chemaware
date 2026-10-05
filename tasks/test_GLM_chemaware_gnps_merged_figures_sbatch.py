"""Static launch contracts for the merged-figure sbatch."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "tasks/run_GLM_chemaware_gnps_merged_figures.sbatch"


def test_cluster_rules_respected() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in text
    assert "--partition" not in text
    assert "#SBATCH --mem" not in text and "--mem=" not in text


def test_evaluations_arrive_by_environment_and_fail_closed() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    for name in ("EVAL_NOISE", "EVAL_CHEMAWARE", "EVAL_PUBLIC"):
        assert name in text
        assert f'"${name}"' in text
    assert "GLM_GNPS_FIG_MISSING_ENV" in text
    assert "GLM_GNPS_FIG_MISSING_EVAL" in text
    assert "GLM_GNPS_FIG_MISSING_CURVES" in text
    assert '[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }' in text


def test_three_evaluations_merged_with_frozen_baseline() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert text.count("--evaluation") == 3
    assert "--baseline official_dreams" in text
    assert "plot_GLM_chemaware_gnps_unified_benchmark.py" in text


def test_outputs_are_sealed() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "summary_table.csv" in text
    assert "main_benchmark_figure.png" in text
    assert "main_benchmark_figure.pdf" in text
    assert "report.json" in text
    assert "SHA256SUMS.txt" in text
