"""Cheap contracts for the paired one-GPU ChemAware PEFT experiment."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    trainer = (ROOT / "tasks/train_chemaware_crossview_event_peft.py").read_text(
        encoding="utf-8",
    )
    sbatch = (ROOT / "tasks/run_chemaware_crossview_event_peft.sbatch").read_text(
        encoding="utf-8",
    )

    assert 'choices=(*ARMS, "all")' in trainer
    assert 'requested_arms = ARMS if args.arm == "all"' in trainer
    assert '"matched_random"' in trainer
    assert '"zero_contrast", "matched_random", "correct"' in trainer
    assert '"CHEMAWARE_CROSSVIEW_EVENT_PEFT_EARLY_STOP"' in trainer
    assert '"saved_control_gpu_arms_if_failed"' in trainer
    assert "def matched_random_events(" in trainer
    assert "def candidate_reference_views(" in trainer
    assert "def proposed_local_candidate(" in trainer
    assert "def build_retrieval_arrays(" in trainer
    assert "def evaluate(" in trainer
    assert "def retrieval(" in trainer
    assert "def bootstrap(" in trainer
    assert "def paired_rank_comparison(" in trainer
    assert "def numerical_replay_audit(" in trainer
    assert '"numerical_boundary_queries_have_zero_chemical_weight"' in trainer
    assert '"numerical_boundary_queries_excluded_from_paired_evaluation"' in trainer
    assert '"frozen_provenance_exact"' in trainer
    assert "maximum_fraction: float = 0.001" in trainer
    assert "global_embedding_adapter" not in trainer
    assert "from audit_chemaware_" not in trainer
    assert "import audit_chemaware_" not in trainer
    assert "logmeanexp_segments(" in trainer
    assert "virtual_adamw_descent_updates(" in trainer
    assert '"optimizer_action_fraction_p10_at_least_0_10"' in trainer
    assert '"optimizer_attributable_alignment_p10_at_least_0_05"' in trainer
    assert '"virtual_adamw_error_at_most_1e_3"' in trainer
    assert "load_peft_state_dict(model, initial_peft_state)" in trainer
    assert 'active_event = np.flatnonzero(event["active"])' in trainer
    assert '"correction_and_protection_events_trained": True' in trainer
    assert '"matched_random_has_correct_arm_event_dose"' in trainer
    assert '"all_arms_share_one_frozen_prefix_cache": True' in trainer
    assert '"peft_state_by_arm": arm_states' in trainer
    assert 'recoverable_after_node_or_time_failure' in trainer
    assert 'partial.replace(args.output)' in trainer
    # The cache is constructed once, before the paired arm-training loop.
    assert trainer.index("store = FrozenPrefixSpectrumStore(") < trainer.index(
        "for arm_index, arm in enumerate(requested_arms):"
    )

    directives = [line.strip() for line in sbatch.splitlines() if line.startswith("#SBATCH")]
    assert directives.count("#SBATCH --gpus=1") == 1
    assert not any("--mem" in line for line in directives)
    assert not any("--array" in line for line in directives)
    assert "--arm all" in sbatch
    assert "SLURM_ARRAY" not in sbatch
    assert "global_embedding_adapter" not in sbatch
    assert 'echo "Missing required runtime file: $required_file"' in sbatch
    assert "dreams/models/chem_aware/peft_v3.py" in sbatch
    assert "fde14aee280ed13bd0dbc793d5739cdc989bedf377ce0000b5a5663d6ce715ba" in sbatch
    assert "Stale PEFT runtime" in sbatch
    assert 'test -f "$OUT/peft_checkpoints.pt"' in sbatch
    print("PASS: paired ChemAware cross-view event PEFT and one-GPU sbatch contracts")


if __name__ == "__main__":
    main()
