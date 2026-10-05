from pathlib import Path


def main() -> None:
    text = Path("tasks/run_chemaware_full_massspecgym_native_gnps.sbatch").read_text(
        encoding="utf-8",
    )
    assert "#SBATCH --gpus=1" in text
    assert "--partition" not in text
    assert "#SBATCH --mem" not in text and "--mem=" not in text
    assert "e1_train_triplet_pool_10ppm.npz" in text
    assert "e1_val_triplet_pool_10ppm.npz" in text
    assert "export_chemaware_full_massspecgym_candidate_registry.py" in text
    assert "build_chemaware_massbank_candidate_source_ledger.py" in text
    assert "build_chemaware_full_massspecgym_native_triplets.py" in text
    assert "train_chemaware_dreams_native.py" in text
    assert "--max-steps 45000" in text
    assert "--checkpoint-mode fixed_steps" in text
    assert '--official-train-pool "$TRAIN_POOL"' in text
    assert '--official-train-pool "$VAL_POOL"' in text
    assert "encode_gnps_gold_silver_10ppm_checkpoint.py" in text
    assert "evaluate_gnps_gold_silver_10ppm_embeddings.py" in text
    assert "panel_identity_disjoint.npz" in text
    assert "panel_formula_disjoint.npz" in text
    assert "evaluate_chemaware_v2_direct_triplet.py" not in text
    assert "select_chemaware_residual_checkpoint.py" not in text
    assert "GLM_train_chemaware_listwise.py" not in text
    print("PASS: ChemAware full-MassSpecGym one-GPU sbatch contracts")


if __name__ == "__main__":
    main()
