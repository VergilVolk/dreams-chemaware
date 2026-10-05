"""Static launch contracts for the public-model GNPS benchmark."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "tasks/run_GLM_chemaware_gnps_public_model_benchmark.sbatch"


def test_cluster_rules_respected() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in text
    assert "--partition" not in text
    assert "#SBATCH --mem" not in text and "--mem=" not in text


def test_models_are_vendored_and_md5_pinned() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    # Compute nodes have no egress: models must be vendored, never downloaded.
    assert "data/reference/gnps_public_models_v1" in text
    assert "curl" not in text
    for name in (
        "spec2vec_UniqueInchikeys_ratio05_filtered_iter_50.model",
        "spec2vec_UniqueInchikeys_ratio05_filtered_iter_50.model.trainables.syn1neg.npy",
        "spec2vec_UniqueInchikeys_ratio05_filtered_iter_50.model.wv.vectors.npy",
        "ms2deepscore_model.pt",
        "settings.json",
    ):
        assert f'"$MODELS/{name}"' in text
    assert "a7333c19ace863c5a0a04abc89600f49" in text
    assert "0962044c1020073d487ae2e07d195f0b" in text
    assert "cd774bf75d00b1715334a4fbfe2b711a" in text
    assert "d5cbf4694a1c476ae59e0c810f56c320" in text
    assert "5d6007c453692b44cd73941a9a434d84" in text
    assert "md5sum --check --strict" in text


def test_packages_are_version_pinned_for_manylinux2014() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    for pin in (
        "numpy==1.26.4", "scipy==1.12.0", "pandas==2.2.3",
        "scikit-learn==1.5.2", "matplotlib==3.7.2",
        "matchms==0.27.0", "spec2vec==0.8.0",
        "ms2deepscore==2.4.0", "gensim==4.3.3",
        "smart_open==7.1.0", "sparsestack==0.5.0",
        "numba==0.59.1", "llvmlite==0.42.0",
        "tensorboard==2.18.0", "pickydict==0.5.0",
        "protobuf==4.25.8",
    ):
        assert pin in text
    # No unpinned installs; the complete numpy/scipy ABI is node-local.
    assert text.count("--no-deps") >= 1
    # Wheels bundling a top-level tests/ package collide in --target installs;
    # the sbatch must remove the stray directory deterministically.
    assert 'rm -rf -- "$DEPS/tests"' in text
    assert "PUBLIC_MODEL_IMPORT_PREFLIGHT_PASS" in text
    assert "torch.cuda.is_available()" in text
    assert "Word2Vec(" in text  # gensim ABI functional gate
    assert '"numpy": "1.26.4"' in text
    assert '"scipy": "1.12.0"' in text
    assert "scipy.linalg.triu" in text


def test_ms2deepscore_uses_public_embedding_api_only() -> None:
    adapter = (
        ROOT / "tasks/GLM_build_gnps_public_model_pair_scores.py"
    ).read_text(encoding="utf-8")
    assert "scorer.get_embedding_array(" in adapter
    for forbidden in ("tensorize_spectra", "model.encoder", "model_settings"):
        assert forbidden not in adapter


def test_both_methods_scored_then_extended_and_evaluated() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "--method spec2vec_gnps_2019" in text
    assert "--method ms2deepscore_dual_2024" in text
    assert text.count("--pair-score-method") == 2
    assert "python -u tasks/build_noise_gnps_article_score_bundle.py" in text
    assert "--baseline-method official_dreams" in text
    assert "--bootstrap-resamples 10000" in text
    assert "--bootstrap-seed 20261003" in text
    assert "--embedding-batch-size 256" in text


def test_fail_closed_inputs_and_no_overwrite() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "GLM_GNPS_PUBLIC_MISSING_INPUT" in text
    assert '[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }' in text
    assert "below 8 GiB" in text
