"""Static launch contracts for the ChemAware GNPS unified benchmark."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "tasks/run_GLM_chemaware_gnps_unified_benchmark.sbatch"


def test_cluster_rules_respected() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in text
    assert "--partition" not in text
    assert "#SBATCH --mem" not in text and "--mem=" not in text


def test_no_truth_contact_or_selection() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    for forbidden in (
        "select_chemaware_residual_checkpoint",
        "role-contract-dir",
        "formula-role 2",
        "formula-role 3",
        "listwise",
    ):
        assert forbidden not in text
    assert "--baseline-method official_dreams" in text


def test_base_bundle_is_reproducible_noise_recipe() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert text.count("python -u tasks/build_noise_gnps_article_score_bundle.py") == 1
    assert "--workers 16" in text
    assert "--fragment-tolerance 0.02" in text
    assert "--n-highest-peaks 100" in text


def test_chemaware_methods_enter_through_sha_locked_extension() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert text.count("python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py") == 2
    assert text.count("python -u tasks/prepare_official_embedding_checkpoint.py") == 2
    assert '--embedding-method "chemaware_stage1=' in text
    assert '--embedding-method "chemaware_phaseA=' in text
    assert "python -u tasks/extend_gnps_article_score_bundle.py" in text
    assert "base_method_scores.npz" in text
    assert "extended_method_scores.npz" in text


def test_evaluation_targets_directory_report() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "--score-bundle \"$OUT/extended_method_scores.npz\"" in text
    assert "--output \"$OUT/evaluation\"" in text
    assert '"$OUT"/evaluation/report.json' in text
    assert "--bootstrap-resamples 10000" in text
    assert "--bootstrap-seed 20261003" in text


def test_fail_closed_inputs_and_no_overwrite() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "GLM_GNPS_UNIFIED_MISSING_INPUT" in text
    assert '[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }' in text
    for item in (
        "gnps_absolute_baselines_run_2347472",
        "run_2340721_resume_2340524/training/best.ckpt",
        "phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt",
        "ms_entropy-1.5.2",
    ):
        assert item in text
