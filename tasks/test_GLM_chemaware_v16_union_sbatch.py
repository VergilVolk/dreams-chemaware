"""Static contracts for the GLM V16 pre-registered submission."""
from pathlib import Path


def main() -> None:
    path = Path(__file__).with_name("run_GLM_chemaware_v16_union.sbatch")
    text = path.read_text(encoding="utf-8")
    # Cluster policy: mandatory --gpus, no explicit partition or memory.
    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH --partition" not in text
    assert "#SBATCH --mem" not in text
    # The pre-registered chain, in order.
    assert "GLM_diagnose_chemaware_evidence_class_errors.py" in text
    assert "--evidence-class-diagnostic" in text
    assert "--expected-current-errors 151" in text
    assert "--expected-admitted-winners 19" in text
    # Amended resource gate: 50 -> 40, post-hoc, with the amendment declared
    # in the script itself and the measured values recorded by the builder.
    assert "AMENDED LAUNCH GATE" in text
    assert "--minimum-launch-boundaries 40" in text
    # Amended mining caps (declared): 8/3 were single-source calibrations.
    assert "AMENDED MINING CAPS" in text
    assert "--maximum-events-per-formula 10" in text
    assert "--maximum-events-per-identity 4" in text
    assert "GLM_build_chemaware_v16_dynamic_triplets.py" in text
    assert "test_GLM_build_chemaware_v16_dynamic_triplets.py" in text
    assert "build_chemaware_layered_10k_native_pool.py" in text
    assert "train_chemaware_dreams_native.py" in text
    assert "evaluate_chemaware_v2_direct_triplet.py" in text
    assert "select_chemaware_residual_checkpoint.py" in text
    assert "audit_chemaware_role3_confirmation.py" in text
    # Pre-registered coverage gates fire before any GPU training.
    assert '"$V16_EVENTS" -lt 1000 || "$V16_QUERIES" -lt 500' in text
    # Both frozen ledgers of the union are consumed.
    assert text.count("--source-ledger") == 2
    assert "run_2346306/qualified_source" in text
    assert "chemaware_massbank_candidate_source_all_qualified_v1_20260928" in text
    # No optimizer-state restore: the init artifact is weights-only.
    assert "restore-adam" not in text
    # Initialization is auto-discovered (login nodes cannot run programs, so
    # nothing may depend on manual path checks) from the documented
    # exploratory +2.3797 pp step-750 run, and role-2 selection is paired
    # against that actual base, not weak Phase A.
    assert 'INIT_RUN="data/validation/chemaware_fragment_graph_native/run_2346408_resume_run_2346306"' in text
    assert "find \"$INIT_RUN\" -type f -name 'step-*750.ckpt'" in text
    assert "GLM_V16_INIT_NOT_FOUND" in text
    assert '--checkpoint "v16_base=$INIT_CKPT"' in text
    assert "--paired-reference v16_base" in text
    assert "--base-name v16_base" in text
    # Role-3 protection baseline stays the protected Phase A.
    assert "--protected-baseline-name phaseA_2pp" in text
    # Sealed GNPS Gold/Silver transfer panels: encode + paired evaluation
    # with multiple recall / AUC metrics against Phase A and official.
    assert "gnps_gold_silver_10ppm_benchmark_v1" in text
    assert "encode_gnps_gold_silver_10ppm_checkpoint.py" in text
    assert "evaluate_gnps_gold_silver_10ppm_embeddings.py" in text
    assert "v16_base_vs_phasea" in text
    assert "v16_selected_vs_phasea" in text
    assert "v16_selected_vs_official" in text
    # Production-identical trainer hyperparameters.
    assert "--lr 2e-6" in text
    assert "--max-steps 1000" in text
    assert "--checkpoint-mode fixed_steps" in text
    # Protected artifacts and hash manifests on every terminal branch.
    assert "protected_negative_result" in text
    assert "protected_candidate" in text
    assert text.count("SHA256SUMS") >= 3
    # The contested arm is preserved but never trained in this job.
    assert "--contested-output" in text
    assert "CONTESTED_ARM_PRESERVED" in text
    # Fail-closed collision and missing-input gates.
    assert '[[ ! -e "$OUT" ]]' in text
    assert "exit 2" in text
    # Structural guard: a backslash continuation must never be followed by a
    # comment line -- the comment splices into the command and orphans the
    # next argument block (the run_2347152 --margin failure).  bash -n cannot
    # catch this, so the contract test must.
    lines = text.splitlines()
    for index, line in enumerate(lines[:-1]):
        if line.rstrip().endswith("\\"):
            stripped = lines[index + 1].lstrip()
            assert not stripped.startswith("#"), (
                f"comment inside a continued command at line {index + 2}: "
                f"{lines[index + 1]!r}"
            )
    print("GLM V16 sbatch contracts passed")


if __name__ == "__main__":
    main()
