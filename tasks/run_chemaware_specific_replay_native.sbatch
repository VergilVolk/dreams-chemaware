#!/bin/bash
#SBATCH --job-name=chem_spec_replay
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_spec_replay_%j.out
#SBATCH --error=data/validation/chem_spec_replay_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BASE_CHECKPOINT="data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt"
EVIDENCE="data/validation/chemaware_high_coverage_native/run_2340524/evidence"
PAIR_BANK="data/validation/chemaware_pair_expanded_native/run_2342187/pair_expanded_triplets"
OUT="data/validation/chemaware_specific_replay_native/run_${SLURM_JOB_ID}"
BANK="$OUT/specific_replay_triplets"
CORRECT="$OUT/training_correct"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

required_files=(
  "$BASE_CHECKPOINT"
  "$EVIDENCE/selection_triplet_evidence.npz"
  "$EVIDENCE/confirmation_triplet_evidence.npz"
  "$PAIR_BANK/report.json"
  "$PAIR_BANK/train_pool.npz"
  "$PAIR_BANK/selection_pool.npz"
  "$PAIR_BANK/confirmation_pool.npz"
  tasks/build_chemaware_specific_replay_native_triplets.py
  tasks/test_chemaware_specific_replay_native_triplets.py
  tasks/train_chemaware_dreams_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/select_chemaware_residual_checkpoint.py
  data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
)
for null_name in content_permuted content_permuted_b content_permuted_c; do
  for role in train selection confirmation; do
    required_files+=("$PAIR_BANK/${role}_pool_${null_name}.npz")
  done
done
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required specific-replay file: $required_file" >&2
    exit 2
  }
done

python -u tasks/test_chemaware_specific_replay_native_triplets.py

# Freeze one safety boundary per query, then retain only correct-arm chemical
# candidates selected by at most one of the three content-permuted nulls.
python -u tasks/build_chemaware_specific_replay_native_triplets.py \
  --pair-bank "$PAIR_BANK" \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --output "$BANK" --chemical-events-per-query 3 \
  --maximum-null-candidate-agreement 1 \
  --min-train-queries 3500 --min-specific-chemical-events 300 \
  --min-specific-chemical-queries 150 --min-chemical-fraction 0.06 \
  --min-active-chemical-fraction 0.15 \
  --max-null-mean-hinge-gap 0.05 --max-null-mean-activation-gap 0.10

# The model, optimizer, preprocessing and loss remain the native DreaMS stack.
# A smaller LR reflects that this is a residual continuation from the confirmed
# +1.8144 pp stage-1 checkpoint, not a fresh full fine-tune.
srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --train-pool "$BANK/train_pool.npz" --val-pool "$BANK/selection_pool.npz" \
  --official-checkpoint "$BASE_CHECKPOINT" \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --output "$CORRECT" --seed 3407 --lr 3e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
  --max-epochs 20 --max-steps 3000 --checkpoint-mode fixed_steps \
  --save-every-n-steps 500 --n-highest-peaks 100 --device cuda

checkpoint_args=(
  --checkpoint "official=data/e1/official_embedding_slim.pt"
  --checkpoint "stage1_base=$BASE_CHECKPOINT"
)
mapfile -t trained_checkpoints < <(find "$CORRECT" -maxdepth 1 -type f -name 'step-*.ckpt' | sort)
[[ ${#trained_checkpoints[@]} -eq 6 ]] || {
  echo "Expected six specific-replay checkpoints, found ${#trained_checkpoints[@]}" >&2
  exit 5
}
for checkpoint in "${trained_checkpoints[@]}"; do
  stem="$(basename "$checkpoint" .ckpt)"
  checkpoint_args+=(--checkpoint "specific_${stem}=$checkpoint")
done

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$EVIDENCE/selection_triplet_evidence.npz" \
  "${checkpoint_args[@]}" \
  --paired-reference stage1_base --formula-role 2 \
  --output "$OUT/role2_checkpoint_evaluation.json" \
  --batch-size 64 --bootstrap-draws 2000 --device cuda

python -u tasks/select_chemaware_residual_checkpoint.py \
  --evaluation "$OUT/role2_checkpoint_evaluation.json" \
  --base-name stage1_base --require-positive-formula-ci \
  --output "$OUT/checkpoint_selection.json"
SELECTED="$(tr -d '\r\n' < "$OUT/checkpoint_selection.checkpoint.txt")"
SELECTED_STEP="$(tr -d '\r\n' < "$OUT/checkpoint_selection.step.txt")"

if [[ "$SELECTED_STEP" == "0" ]]; then
  echo "COMPLETE: specific chemical replay did not pass the frozen role-2 gate"
  exit 0
fi

# Only a role-2-qualified correct arm earns the three exact-schedule null runs.
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
    --output "$null_output" --seed 3407 --lr 3e-6 --weight-decay 0 \
    --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
    --max-epochs 20 --max-steps "$SELECTED_STEP" --checkpoint-mode fixed_steps \
    --save-every-n-steps "$SELECTED_STEP" --n-highest-peaks 100 --device cuda
  null_checkpoints+=(--checkpoint "null_${null_name}=$null_output/last.ckpt")
done

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$EVIDENCE/confirmation_triplet_evidence.npz" \
  --checkpoint "official=data/e1/official_embedding_slim.pt" \
  --checkpoint "stage1_base=$BASE_CHECKPOINT" \
  --checkpoint "specific_replay_correct=$SELECTED" \
  "${null_checkpoints[@]}" \
  --paired-reference stage1_base --formula-role 3 \
  --output "$OUT/role3_specific_replay_and_null_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

test -f "$BANK/report.json"
test -f "$OUT/checkpoint_selection.json"
test -f "$OUT/role3_specific_replay_and_null_evaluation.json"
echo "COMPLETE: ChemAware specific chemical replay -> $OUT"
