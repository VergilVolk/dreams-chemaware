"""Static contracts for the two-stage SIRIUS/ChemAware Slurm route."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = (ROOT / "tasks/run_chemaware_sirius_source.sbatch").read_text()
    native = (ROOT / "tasks/run_chemaware_sirius_native.sbatch").read_text()
    integrated = (
        ROOT / "tasks/run_chemaware_layered_sirius_10k_native.sbatch"
    ).read_text()
    assert "#SBATCH --partition=gpu" in source
    assert "#SBATCH --gpus=1" in source
    assert "#SBATCH --mem" not in source
    assert "query_formula_ms" in source
    assert "query_formula_ms_intensity_rank_permuted" in source
    assert "query_formula_ms_mass_shifted" in source
    assert "for profile in orbitrap qtof default" in source
    assert "profile_args=(-p orbitrap --ppm-max 10 --ppm-max-ms2 5)" in source
    assert "profile_args=(-p qtof --ppm-max 10 --ppm-max-ms2 10)" in source
    assert "profile_args=(--ppm-max 10 --ppm-max-ms2 10)" in source
    assert 'formulas "${profile_args[@]}"' in source
    assert '"$SIRIUS_BIN" custom-db --input "$PANEL/candidate_structures.tsv"' in source
    assert "requires SIRIUS major version 6" in source
    assert "fingerprints" in source and "structures --database ChemCand" in source
    assert "import_chemaware_sirius_controlled_source_ledger.py" in source
    assert "qualify_chemaware_candidate_source_ledger.py" in source
    assert "build_chemaware_layered_rule_source_ledger.py" in source
    assert '"$OUT/qualified_static_source"' in source
    assert "#SBATCH --partition=gpu" in native
    assert "#SBATCH --gpus=1" in native
    assert "#SBATCH --mem" not in native
    assert '--official-checkpoint "$PHASEA"' in native
    assert "build_chemaware_max_boundary_native_triplets.py" not in native
    assert "build_chemaware_multisource_native_triplets.py" in native
    assert 'source_args+=(--source-ledger "$SIRIUS_SOURCE")' in native
    assert 'source_args+=(--source-ledger "$STATIC_SOURCE")' in native
    assert "build_chemaware_layered_10k_native_pool.py" in native
    assert "--minimum-chemical-events 1000" in native
    assert "--minimum-chemical-queries 500" in native
    assert 'mkdir -p "$OUT/protected_corpus"' in native
    assert '"$OUT/protected_corpus/train_pool.npz"' in native
    assert '"$OUT/protected_corpus/sirius_candidate_scores.tsv"' in native
    assert '"$OUT/protected_corpus/static_candidate_scores.tsv"' in native
    assert '"$OUT/protected_corpus/source_inputs/query_registry.tsv"' in native
    assert "--paired-reference phaseA_2pp --formula-role 2" in native
    assert "CHEMAWARE_SIRIUS_ROLE2_STOP" in native
    assert "audit_chemaware_role3_confirmation.py" in native
    assert "CHEMAWARE_ROLE3_STOP" in native
    assert '"$OUT/protected_negative_result"' in native
    assert "#SBATCH --partition=gpu" in integrated
    assert "#SBATCH --gpus=1" in integrated
    assert "#SBATCH --mem" not in integrated
    assert "bash tasks/run_chemaware_sirius_source.sbatch" in integrated
    assert "bash tasks/run_chemaware_sirius_native.sbatch" in integrated
    assert 'SIRIUS_RUN="data/validation/chemaware_sirius_source/run_${SLURM_JOB_ID}"' in integrated
    print("PASS: ChemAware SIRIUS pipeline contracts", flush=True)


if __name__ == "__main__":
    main()
