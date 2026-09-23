#!/bin/bash
#SBATCH --job-name=chem_pair_native
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_pair_native_%j.out
#SBATCH --error=data/validation/chem_pair_native_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BASE_CHECKPOINT="${BASE_CHECKPOINT:-data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt}"
EVIDENCE="${EVIDENCE:-data/validation/chemaware_high_coverage_native/run_2340524/evidence}"
OLD_POOL="${OLD_POOL:-data/validation/chemaware_high_coverage_native/run_2340524/triplets/train_pool.npz}"
OUT="data/validation/chemaware_pair_expanded_native/run_${SLURM_JOB_ID}"
CACHE="$OUT/stage1_embedding_cache"
BANK="$OUT/pair_expanded_triplets"
CORRECT="$OUT/training_correct"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

required_files=(
  "$BASE_CHECKPOINT"
  "$EVIDENCE/train_triplet_evidence.npz"
  "$EVIDENCE/selection_triplet_evidence.npz"
  "$EVIDENCE/confirmation_triplet_evidence.npz"
  "$OLD_POOL"
  tasks/encode_chemaware_checkpoint_manifest_rows.py
  tasks/build_chemaware_reference_aligned_native_triplets.py
  tasks/audit_chemaware_native_triplet_activity.py
  tasks/validate_chemaware_triplet_activity_gain.py
  tasks/select_chemaware_residual_checkpoint.py
  tasks/test_chemaware_pair_expanded_native_triplets.py
  tasks/train_chemaware_dreams_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required runtime file: $required_file" >&2
    exit 2
  }
done

python -u tasks/test_chemaware_pair_expanded_native_triplets.py

# Freeze current geometry at the confirmed +1.8144 pp checkpoint.
srun --export=ALL --preserve-env python -u tasks/encode_chemaware_checkpoint_manifest_rows.py \
  --checkpoint "$BASE_CHECKPOINT" \
  --evidence-dir "$EVIDENCE" \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --official-checkpoint data/e1/official_embedding_slim.pt \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --output "$CACHE" --batch-size 64 --n-highest-peaks 100 --device cuda

# Expand only unique, nonzero native-hinge spectrum pairs.  Two shared hard
# candidate slots, up to two chemical slots, and hard fallbacks fill a maximum
# of five candidate molecules.  Boundary and hard positive views are distinct.
python -u tasks/build_chemaware_reference_aligned_native_triplets.py \
  --evidence-dir "$EVIDENCE" --embedding-cache "$CACHE" \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --output "$BANK" --margin 0.1 --candidates-per-query 5 \
  --chemical-candidates-per-query 2 \
  --negative-references-per-candidate 2 \
  --explicit-positive-references-per-negative 2 \
  --max-active-spectrum-events-per-query 12 \
  --chemical-hardness-window 0.50 \
  --min-chemical-activation-probability 0.05 \
  --min-train-queries 3500 --min-candidate-events 4500 \
  --min-spectrum-events 8000 --min-train-formulas 2000 \
  --min-chemical-candidates 300 --min-chemical-spectrum-events 1000

# The same stage-1 embedding cache must show a material gain over the old pool
# before any optimizer step is allowed.
python -u tasks/audit_chemaware_native_triplet_activity.py \
  --pool "$OLD_POOL" --embedding-cache "$CACHE" --margin 0.1 \
  --output "$OUT/old_pool_activity_audit.json"
python -u tasks/audit_chemaware_native_triplet_activity.py \
  --pool "$BANK/train_pool.npz" --embedding-cache "$CACHE" --margin 0.1 \
  --output "$OUT/new_pool_activity_audit.json"
python -u tasks/validate_chemaware_triplet_activity_gain.py \
  --old "$OUT/old_pool_activity_audit.json" \
  --new "$OUT/new_pool_activity_audit.json" \
  --min-active-event-ratio 1.25 \
  --min-activation-probability-ratio 1.25 \
  --min-mean-hinge-ratio 1.25 \
  --output "$OUT/triplet_activity_gate.json"

# Keep the successful native DreaMS model, loss, optimizer and hyperparameters.
# Role 2 selects the shortest checkpoint that passes the full metric gate.
srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --train-pool "$BANK/train_pool.npz" --val-pool "$BANK/selection_pool.npz" \
  --official-checkpoint "$BASE_CHECKPOINT" \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --output "$CORRECT" --seed 3407 --lr 5e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
  --max-epochs 20 --max-steps 3000 --checkpoint-mode fixed_steps \
  --save-every-n-steps 500 --n-highest-peaks 100 --device cuda

checkpoint_args=(
  --checkpoint "official=data/e1/official_embedding_slim.pt"
  --checkpoint "stage1_base=$BASE_CHECKPOINT"
)
for checkpoint in "$CORRECT"/step-*.ckpt; do
  [[ -f "$checkpoint" ]] || continue
  stem="$(basename "$checkpoint" .ckpt)"
  checkpoint_args+=(--checkpoint "pair_${stem}=$checkpoint")
done
[[ ${#checkpoint_args[@]} -ge 8 ]] || { echo "Pair-expanded checkpoints are incomplete" >&2; exit 5; }

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$EVIDENCE/selection_triplet_evidence.npz" \
  "${checkpoint_args[@]}" \
  --paired-reference stage1_base --formula-role 2 \
  --output "$OUT/role2_checkpoint_evaluation.json" \
  --batch-size 64 --bootstrap-draws 2000 --device cuda

python -u tasks/select_chemaware_residual_checkpoint.py \
  --evaluation "$OUT/role2_checkpoint_evaluation.json" \
  --base-name stage1_base --output "$OUT/checkpoint_selection.json"
SELECTED="$(tr -d '\r\n' < "$OUT/checkpoint_selection.checkpoint.txt")"
SELECTED_STEP="$(tr -d '\r\n' < "$OUT/checkpoint_selection.step.txt")"

if [[ "$SELECTED_STEP" == "0" ]]; then
  echo "COMPLETE: pair-expanded triplets did not beat frozen stage 1 on role 2"
  exit 0
fi

# Exact-budget semantic-null arms distinguish generic hard-pair continuation
# from the incremental value of correct ChemAware candidate selection.
null_names=(content_permuted content_permuted_b content_permuted_c)
null_checkpoints=()
for null_name in "${null_names[@]}"; do
  null_output="$OUT/training_${null_name}"
  srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py \
    --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
    --train-pool "$BANK/train_pool_${null_name}.npz" \
    --val-pool "$BANK/selection_pool_${null_name}.npz" \
    --official-checkpoint "$BASE_CHECKPOINT" \
    --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
    --output "$null_output" --seed 3407 --lr 5e-6 --weight-decay 0 \
    --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
    --max-epochs 20 --max-steps "$SELECTED_STEP" --checkpoint-mode fixed_steps \
    --save-every-n-steps "$SELECTED_STEP" --n-highest-peaks 100 --device cuda
  null_checkpoints+=(--checkpoint "null_${null_name}=$null_output/last.ckpt")
done

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$EVIDENCE/confirmation_triplet_evidence.npz" \
  --checkpoint "official=data/e1/official_embedding_slim.pt" \
  --checkpoint "stage1_base=$BASE_CHECKPOINT" \
  --checkpoint "pair_expanded_correct=$SELECTED" \
  "${null_checkpoints[@]}" \
  --paired-reference stage1_base --formula-role 3 \
  --output "$OUT/role3_pair_expanded_and_null_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

test -f "$BANK/report.json"
test -f "$OUT/triplet_activity_gate.json"
test -f "$OUT/checkpoint_selection.json"
test -f "$OUT/role3_pair_expanded_and_null_evaluation.json"
echo "COMPLETE: ChemAware pair-expanded native triplets -> $OUT"
