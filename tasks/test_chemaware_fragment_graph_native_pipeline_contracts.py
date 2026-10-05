"""Static contracts for the single-entry fragment-graph native pipeline."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "tasks/run_chemaware_fragment_graph_native.sbatch"


def main() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH --mem" not in text
    assert "--maximum-queries" not in text
    assert "SHARDS=8" in text
    assert 'REUSE_RUN="${1:-}"' in text
    assert '"latest-qualified"' in text
    assert 'SOURCE_LEDGER="$REUSE_RUN/qualified_source"' in text
    assert 'EMBEDDING_DIR="$REUSE_RUN/phasea_embeddings"' in text
    assert "sirius_source_panel" not in text.lower()
    assert 'TRAIN_EVIDENCE="$BASE/evidence/train_triplet_evidence.npz"' in text
    assert '--evidence "$TRAIN_EVIDENCE"' in text
    assert '--manifest "$MANIFEST"' in text

    source = text.index("python -u tasks/build_chemaware_fragment_graph_candidate_source_ledger.py")
    merge = text.index("python -u tasks/merge_chemaware_fragment_graph_source_ledgers.py")
    qualify = text.index("python -u tasks/qualify_chemaware_candidate_source_ledger.py")
    encode = text.index("python -u tasks/encode_chemaware_checkpoint_manifest_rows.py")
    triplets = text.index("python -u tasks/build_chemaware_dynamic_reference_native_triplets.py")
    layered = text.index("python -u tasks/build_chemaware_layered_10k_native_pool.py")
    train = text.index("python -u tasks/train_chemaware_dreams_native.py")
    assert source < merge < qualify < encode < triplets < layered < train

    assert "chemaware_phasea_max_boundary_step2000.ckpt" in text
    assert 'PHASEA_POOL="$PHASEA_RUN/phasea_triplets/train_pool.npz"' in text
    assert 'PHASEA_VAL="$PHASEA_RUN/phasea_triplets/val_pool.npz"' in text
    assert "dynamic_source_event_ledger.tsv" in text
    assert "dynamic_chemical_events_added" in text
    assert "dynamic_chemical_queries" in text
    assert "--matched-control-policy pairwise_dominance" in text
    assert "--maximum-positive-references-per-relation 1" in text
    assert "--maximum-negative-references-per-event 1" in text
    assert "--maximum-events-per-formula 8" in text
    assert "--maximum-events-per-identity 3" in text
    assert '--embedding-report "$EMBEDDING_DIR/report.json"' in text
    assert '--geometry-checkpoint "$PHASEA"' in text
    assert "--minimum-chemical-events 1000" in text
    assert "--minimum-chemical-queries 500" in text
    assert '"$CHEM_QUERIES" -lt 500' in text
    assert 'TARGET_EVENTS=$((PHASEA_EVENTS + CHEM_EVENTS))' in text
    assert '--target-events "$TARGET_EVENTS"' in text

    assert '--official-checkpoint "$PHASEA"' in text
    assert '--restore-adam-from "$PHASEA"' not in text
    assert '--role-contract-dir "$BASE/evidence"' in text
    assert "--triplet-loss-margin 0.1 --batch-size 4 --num-workers 0" in text
    assert "--lr 2e-6 --weight-decay 0" in text
    assert "--max-steps 1000 --checkpoint-mode fixed_steps" in text
    assert "--save-every-n-steps 250" in text
    assert "--paired-reference phaseA_2pp --formula-role 2" in text
    assert "--require-positive-formula-ci" in text
    assert "role 3 remains unopened" in text
    assert "--paired-reference phaseA_2pp --formula-role 3" in text
    assert "audit_chemaware_role3_confirmation.py" in text

    # Prior broad MassBank triplets are not silently reused as new chemistry.
    assert "chemaware_massbank_layered_12k_native" not in text
    assert "chemaware_massbank_candidate_source" not in text
    assert "distill" not in text.lower()
    print("PASS: ChemAware fragment-graph native mainline contracts")


if __name__ == "__main__":
    main()
