"""Static launch contracts for the ChemAware GNPS external battery."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "tasks/run_GLM_chemaware_gnps_external_battery.sbatch"


def test_cluster_rules_respected() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in text
    assert "--partition" not in text
    assert "#SBATCH --mem" not in text


def test_battery_is_encode_and_evaluate_only() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert text.count("python -u tasks/prepare_official_embedding_checkpoint.py") == 3
    assert text.count("python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py") == 4
    assert text.count("python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py") == 4
    for forbidden in (
        "train_chemaware", "listwise", "pytest",
    ):
        assert forbidden not in text


def test_official_encoded_directly_others_prepared() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert '--checkpoint "$OFFICIAL"' in text
    assert '--source "$STAGE1"' in text
    assert '--source "$PHASEA"' in text
    assert '--source "$STEP750"' in text
    assert '--source "$OFFICIAL"' not in text


def test_evaluation_pairs_are_the_preregistered_four() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert '"$OUT/chemaware_stage1_vs_official"' in text
    assert '"$OUT/chemaware_phaseA_vs_official"' in text
    assert '"$OUT/chemaware_step750_vs_official"' in text
    assert '"$OUT/chemaware_phaseA_vs_stage1"' in text
    baseline_official = text.count('--baseline-embeddings "$OFFICIAL_EMB"')
    baseline_stage1 = text.count('--baseline-embeddings "$STAGE1_EMB"')
    assert baseline_official == 3
    assert baseline_stage1 == 1


def test_benchmark_inputs_and_no_overwrite() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    for item in (
        "panel_identity_disjoint.npz", "panel_formula_disjoint.npz",
        "manifest.csv.gz", "spectra.mgf",
    ):
        assert item in text
    assert '[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }' in text
    assert '--bootstrap-resamples 10000 --bootstrap-seed 20260930' in text


def test_evaluator_outputs_are_directories_and_shas_target_report_json() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    # The evaluator writes one directory per comparison (report.json plus
    # per-query outcome tables); checksums must target report.json inside.
    assert '"$OUT"/chemaware_*_vs_*/report.json > "$OUT/SHA256SUMS.txt"' in text
    assert "chemaware_*_vs_*.json" not in text


def test_frozen_checkpoint_paths_match_ledger() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert (
        "run_2340721_resume_2340524/training/best.ckpt" in text
    )
    assert (
        "phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt" in text
    )
    assert "run_2346408_resume_run_2346306/training/step-000750.ckpt" in text
