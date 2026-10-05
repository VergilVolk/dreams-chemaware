#!/usr/bin/env python
from pathlib import Path


def main() -> None:
    text = Path(
        __file__
    ).with_name("run_noise_relation_t1_t3_v2_2gpu.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --gpus=2" in text
    assert "#SBATCH --mem" not in text
    assert "--dependency" not in text
    assert "build_noise_relation_t1_t3_corpus_v2.py" in text
    assert "train_noise_relation_t1_t3_v2.py" in text
    assert "score_fold1_noise_relation_checkpoint.py" in text
    assert "summarize_noise_relation_t1_t3_v2_fold1.py" in text
    assert "--arm clean-only" not in text  # arm is passed via variables, never hardcoded
    assert "clean-only 3407" in text and "action-rotation 3407" in text
    assert '"$SELECTED_ARM" 3408' in text
    assert "seed_3407" in text and "seed_3408" in text
    # The v1 trainer stays a compile-time dependency only (the averaged arm
    # is reused from run 2347055, never retrained by this pipeline).
    assert "python -u tasks/train_noise_relation_t1_t3.py" not in text
    assert "AB_V1_RUN:-data/validation/noise_relation_t1_t3_run_2347055" in text
    # Held spend is gated on the predeclared fold-1 selection.
    assert "proceed_to_held" in text
    assert "NO-GO" in text
    assert "exit 0" in text
    # Independent GNPS panel is mandatory for the winner.
    assert "encode_gnps_gold_silver_10ppm_checkpoint.py" in text
    assert "evaluate_gnps_gold_silver_10ppm_embeddings.py" in text
    # The 1.2 GB Stage-1 Lightning checkpoint reaches GNPS encoding only via
    # the proven format-only slim conversion, never directly.  The single
    # permitted raw-checkpoint use is the graph-embedding encoder of Stage A.1,
    # exactly as the cluster-proven v1 sbatch did.
    assert "prepare_official_embedding_checkpoint.py" in text
    assert '--checkpoint "$STAGE1_SLIM"' in text
    assert text.count('--checkpoint "$STAGE1_CHECKPOINT"') == 1
    assert "final_optimizer.pt" in text
    assert "GLM_probe" not in text and "rrf" not in text.lower()
    print("[test_noise_relation_t1_t3_v2_sbatch] PASS")


if __name__ == "__main__":
    main()
