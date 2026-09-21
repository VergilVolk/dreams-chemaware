#!/bin/bash
#SBATCH --job-name=chem_native_hc
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/chem_native_hc_%j.out
#SBATCH --error=data/validation/chem_native_hc_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OUT="data/validation/chemaware_high_coverage_native/run_${SLURM_JOB_ID}"
EVIDENCE="$OUT/evidence"
TRIPLETS="$OUT/triplets"
TRAINING="$OUT/training"
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

required_files=(
  tasks/audit_chemaware_orthogonal_rule_residual_policy.py
  tasks/chemaware_numpy_sampling.py
  tasks/chemaware_native_triplet_mining_core.py
  tasks/build_chemaware_action_hard_native_triplets.py
  tasks/test_chemaware_high_coverage_triplet_mining.py
  tasks/test_chemaware_dreams_native_runtime_contracts.py
  tasks/train_chemaware_dreams_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz
  data/validation/chemaware_corrected_manifest_tokens_v1/rows.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/official_embeddings_f32.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/mz_f32.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/intensity_f32.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/valid.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/precursor_mz_f32.npy
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
  dreams/models/chem_aware/chem_rules_data.json
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required runtime file: $required_file" >&2
    exit 2
  }
done

python -u tasks/test_chemaware_high_coverage_triplet_mining.py
python -u tasks/test_chemaware_dreams_native_runtime_contracts.py

# Reconstruct the complete roles-0/1 action-event ledger and the disjoint
# role-2 selection / role-3 confirmation ledgers.  No outer role-4 row is read.
python -u tasks/audit_chemaware_orthogonal_rule_residual_policy.py \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --token-dir data/validation/chemaware_corrected_manifest_tokens_v1 \
  --rule-library dreams/models/chem_aware/chem_rules_data.json \
  --rule-control-variants \
    rule_response_content_permuted \
    rule_response_content_permuted_b \
    rule_response_content_permuted_c \
  --output "$EVIDENCE" \
  --folds 5 --fold-seed 20260905 --sampling-seed 20260905 --seed 20260912 \
  --train-identities 4096 --validation-identities 2048 --max-inner-identities 0 \
  --beta 0 0.025 0.05 0.1 0.2 0.4 0.8 1.6 \
  --global-mass-beta 0.1 --global-rule-beta 0.2 \
  --risk-penalty 2 --min-selected-formulas 50 \
  --residual-dose 0 0.25 0.5 1 2 4 \
  --contrast-representation symmetric_summary \
  --selection-control-mode deployment_safe \
  --top-peaks 32 --kernel-dim 2048 --bin-width 0.02 --grid-offsets 4 \
  --intensity-power 0.5 --mass-shift-da 0.137 --pair-weight 0.25 \
  --multi-bin-widths 0.01 0.02 0.05 \
  --uniform-channel-weight 1 --rule-tolerance 0.02 --rule-channel-weight 1 \
  --max-iter 150 --learning-rate 0.05 --max-leaf-nodes 15 \
  --min-samples-leaf 100 --l2-regularization 1 --bootstrap-draws 10000 \
  --triplet-evidence-only

# Build full-coverage, deduplicated DreaMS triplets.  Chemistry changes only
# hard-negative selection; positives and negatives remain identity supervised.
python -u tasks/build_chemaware_action_hard_native_triplets.py \
  --evidence-dir "$EVIDENCE" \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --output "$TRIPLETS" \
  --action-hard-k 2 --specific-hard-k 1 \
  --min-train-queries 3500 --min-train-events 5000 \
  --min-train-formulas 2000

# Exact official role-3 replay must pass before the first gradient update.
python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$EVIDENCE/confirmation_triplet_evidence.npz" \
  --checkpoint "official=data/e1/official_embedding_slim.pt" \
  --output "$OUT/official_role3_preflight.json" \
  --batch-size 64 --bootstrap-draws 100 --device cuda

# The optimizer, loss, model, preprocessing and dynamic pair sampling are the
# repository's native DreaMS contrastive implementation and official settings.
srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --train-pool "$TRIPLETS/train_pool.npz" \
  --val-pool "$TRIPLETS/val_pool.npz" \
  --official-checkpoint data/e1/official_embedding_slim.pt \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --output "$TRAINING" \
  --seed 3407 --lr 5e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
  --max-epochs 301 --n-highest-peaks 100 --device cuda

python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$EVIDENCE/confirmation_triplet_evidence.npz" \
  --checkpoint "official=data/e1/official_embedding_slim.pt" \
  --checkpoint "chemaware_high_coverage=$TRAINING/best.ckpt" \
  --output "$OUT/shared_embedding_role3_report.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

test -f "$TRIPLETS/report.json"
test -f "$TRAINING/report.json"
test -f "$TRAINING/best.ckpt"
test -f "$OUT/shared_embedding_role3_report.json"
echo "COMPLETE: high-coverage native DreaMS ChemAware continuation -> $OUT"
