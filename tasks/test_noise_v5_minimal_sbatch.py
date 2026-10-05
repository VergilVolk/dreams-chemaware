#!/usr/bin/env python
"""Contract test for the V5 minimal sbatch."""
from pathlib import Path


def main() -> None:
    text = Path(__file__).with_name(
        "run_noise_v5_minimal_2gpu.sbatch",
    ).read_text(encoding="utf-8")
    # Two GPUs, matched parallel arms, atlas and GNPS-baseline dependencies.
    assert "#SBATCH --gpus=2" in text
    assert 'run_arm targeted 0' in text
    assert 'run_arm control 1' in text
    assert 'run_arm targeted 0 >"$RUN_ROOT/arm_targeted.log" 2>&1 &' in text
    assert 'run_arm control 1 >"$RUN_ROOT/arm_control.log" 2>&1 &' in text
    assert 'wait "$TARGET_PID"' in text
    assert 'wait "$CONTROL_PID"' in text
    assert ': "${ATLAS_RUN:?export ATLAS_RUN=' in text
    assert '"$ATLAS_RUN/atlas/selected_relations.npz"' in text
    assert '"$BASELINE_RUN/gnps/v1_embeddings.npz"' in text
    assert "noise_relation_t1_t3_run_2347055" in text
    # The safety replay and the pre-registered verdict are part of the job.
    assert "score_noise_dev_graph_recall.py" in text
    assert "--output \"$RUN_ROOT/dev_recall_${label}.json\"" in text
    assert "summarize_noise_v5_minimal.py" in text
    assert "--output \"$RUN_ROOT/verdict.json\"" in text
    # Three paired GNPS evaluations with directory outputs.
    assert "--baseline-embeddings \"$BASELINE_RUN/gnps/v1_embeddings.npz\"" in text
    assert "--candidate-embeddings \"$GNPS/targeted_v5_embeddings.npz\"" in text
    assert "--baseline-embeddings \"$GNPS/targeted_v5_embeddings.npz\"" in text
    assert "--candidate-embeddings \"$GNPS/control_v5_embeddings.npz\"" in text
    assert '--output "$GNPS/paired_v1_vs_targeted_v5"' in text
    assert '--output "$GNPS/paired_v1_vs_control_v5"' in text
    assert '--output "$GNPS/paired_targeted_v5_vs_control_v5"' in text
    assert "/report.json\" \\\\" not in text
    # Discipline: no held fold-0 evaluation, no training on GNPS spectra,
    # both contract tests run inside the job, atlas provenance is verified.
    assert "evaluate_noise_dreams_native" not in text
    assert "held fold-0 evaluation runs in" in text
    assert "GNPS spectra never enter training" in text
    assert "python -u tasks/test_noise_v5_minimal.py" in text
    assert "python -u tasks/test_noise_v5_minimal_sbatch.py" in text
    assert "atlas provenance names a different candidate graph" in text
    # Frozen training contract is invoked through the trainer CLI only.
    assert "--selected-relations \"$ATLAS_RUN/atlas/selected_relations.npz\"" in text
    assert '--arm "$label"' in text
    print("[test_noise_v5_minimal_sbatch] PASS")


if __name__ == "__main__":
    main()
