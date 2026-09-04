from pathlib import Path


def test_combined_sbatch_is_single_gpu_resumable_and_fail_closed() -> None:
    text = Path("tasks/run_kgmn_full_external_pipeline.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH --mem" not in text
    assert "#SBATCH --time=72:00:00" in text
    assert "#SBATCH --output=/data02/run01/scv7tsl/DreaMS/kgmn_full_external_%j.out" in text
    assert "#SBATCH --error=/data02/run01/scv7tsl/DreaMS/kgmn_full_external_%j.err" in text
    assert "#SBATCH --output=logs/" not in text
    assert "#SBATCH --error=logs/" not in text
    assert "preflight_kgmn_full_external_pipeline.py" in text
    stages = [
        "run_kgmn_metdna2_200std_baseline.sbatch",
        "run_kgmn_dreams_edge_calibration.sbatch",
        "run_kgmn_oep003284_author_inputs.sbatch",
        "run_kgmn_oep003284_hidden_seed.sbatch",
    ]
    positions = [text.index(stage) for stage in stages]
    assert positions == sorted(positions)
    assert "official_dreams_eligible_for_dynamic_propagation_test" in text
    assert "Partial or invalid" in text
    assert "final_decision.json" in text


def test_manual_dependency_preflight_has_frozen_hashes_and_24_raw_files() -> None:
    text = Path("tasks/preflight_kgmn_full_external_pipeline.py").read_text(encoding="utf-8")
    assert "EXPECTED_SOURCE_COMMIT" in text
    assert "MassSpecGym_MurckoHist_split.hdf5" in text
    assert "official_embedding_slim.pt" in text
    assert "paired_reaction_decoy_triples.csv.gz" in text
    contract = Path("tasks/contracts/kgmn_oep003284_node_files_20260831.csv").read_text(encoding="utf-8").splitlines()
    assert len(contract) == 25
