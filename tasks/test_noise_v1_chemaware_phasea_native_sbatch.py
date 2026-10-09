from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = (
    ROOT / "tasks/run_noise_v1_chemaware_phasea_native.sbatch"
).read_text(encoding="utf-8")


def test_noise_to_chemaware_reuses_native_phasea_runtime() -> None:
    assert SBATCH.count("#SBATCH --gpus=1") == 1
    assert SBATCH.count("#SBATCH --ntasks=1") == 1
    assert "#SBATCH --mem" not in SBATCH
    assert 'NOISE="data/validation/noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt"' in SBATCH
    assert "build_chemaware_max_boundary_native_triplets.py" in SBATCH
    assert "train_chemaware_specific_replay_native.py" in SBATCH
    assert '--official-checkpoint "$NOISE"' in SBATCH
    for exact_parameter in (
        "--margin 0.1 --negative-references-per-error 2",
        "--chemical-candidates-per-error 2 --dreams-replay-events 1024",
        "--seed 3407 --lr 5e-6 --weight-decay 0",
        "--triplet-loss-margin 0.1 --batch-size 4 --num-workers 0",
        "--max-epochs 3 --max-steps 2000 --checkpoint-mode fixed_steps",
        "--save-every-n-steps 2000 --n-highest-peaks 100 --device cuda",
    ):
        assert exact_parameter in SBATCH
    assert '--embedding-rows "$TOKEN_ROWS"' in SBATCH
    assert '--official-embeddings "$OFFICIAL_EMBEDDINGS"' in SBATCH
    assert "--min-error-event-fraction 0.20" in SBATCH
    assert "--current-geometry-remine" not in SBATCH
    assert "encode_chemaware_checkpoint_manifest_rows.py" not in SBATCH
    assert "select_chemaware_residual_checkpoint.py" not in SBATCH
    assert 'SELECTED="$TRAINING/step-002000.ckpt"' in SBATCH
    assert "--paired-reference noise_v1 --formula-role 2" in SBATCH
    assert "--paired-reference noise_v1 --formula-role 3" in SBATCH


def test_downstream_scores_are_reused_not_reimplemented() -> None:
    assert 'BASE_SCORES="data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz"' in SBATCH
    assert "extend_gnps_article_score_bundle.py" in SBATCH
    assert "evaluate_noise_gnps_article_benchmark.py" in SBATCH
    forbidden_new_implementations = (
        "unified_retrieval.py",
        "train_unified_candidate_set_ranker.py",
        "grand_unified_evidence_core.py",
        "p2b_noise_v1_frozen=" ,
    )
    for forbidden in forbidden_new_implementations:
        assert forbidden not in SBATCH
