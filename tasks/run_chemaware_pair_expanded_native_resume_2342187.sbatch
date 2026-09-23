#!/bin/bash
#SBATCH --job-name=chem_pair_eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_pair_eval_%j.out
#SBATCH --error=data/validation/chem_pair_eval_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# This recovery entrypoint never repeats the completed 3,000-step correct-arm
# training.  It resumes at the failed role-2 evaluation in the existing run.
OUT="data/validation/chemaware_pair_expanded_native/run_2342187"
BASE_CHECKPOINT="data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt"
EVIDENCE="data/validation/chemaware_high_coverage_native/run_2340524/evidence"
BANK="$OUT/pair_expanded_triplets"
CORRECT="$OUT/training_correct"

required_files=(
  "$BASE_CHECKPOINT"
  "$EVIDENCE/selection_triplet_evidence.npz"
  "$EVIDENCE/confirmation_triplet_evidence.npz"
  "$BANK/train_pool_content_permuted.npz"
  "$BANK/selection_pool_content_permuted.npz"
  "$BANK/train_pool_content_permuted_b.npz"
  "$BANK/selection_pool_content_permuted_b.npz"
  "$BANK/train_pool_content_permuted_c.npz"
  "$BANK/selection_pool_content_permuted_c.npz"
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/chemaware_v2_triplet_eval_core.py
  tasks/select_chemaware_residual_checkpoint.py
  tasks/train_chemaware_dreams_native.py
  data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required recovery file: $required_file" >&2
    exit 2
  }
done

python -u tasks/test_chemaware_v2_direct_triplet_evaluation.py

mapfile -t trained_checkpoints < <(find "$CORRECT" -maxdepth 1 -type f -name 'step-*.ckpt' | sort)
[[ ${#trained_checkpoints[@]} -eq 6 ]] || {
  echo "Expected exactly six completed correct-arm checkpoints, found ${#trained_checkpoints[@]}" >&2
  printf '%s\n' "${trained_checkpoints[@]}" >&2
  exit 4
}

checkpoint_args=(
  --checkpoint "official=data/e1/official_embedding_slim.pt"
  --checkpoint "stage1_base=$BASE_CHECKPOINT"
)
for checkpoint in "${trained_checkpoints[@]}"; do
  stem="$(basename "$checkpoint" .ckpt)"
  checkpoint_args+=(--checkpoint "pair_${stem}=$checkpoint")
done

if [[ ! -f "$OUT/role2_checkpoint_evaluation.json" ]]; then
  srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
    --policy "$EVIDENCE/selection_triplet_evidence.npz" \
    "${checkpoint_args[@]}" \
    --paired-reference stage1_base --formula-role 2 \
    --output "$OUT/role2_checkpoint_evaluation.json" \
    --batch-size 64 --bootstrap-draws 2000 --device cuda
else
  echo "Reusing completed role-2 evaluation: $OUT/role2_checkpoint_evaluation.json"
fi

if [[ ! -f "$OUT/checkpoint_selection.json" ]]; then
  python -u tasks/select_chemaware_residual_checkpoint.py \
    --evaluation "$OUT/role2_checkpoint_evaluation.json" \
    --base-name stage1_base --output "$OUT/checkpoint_selection.json"
else
  echo "Reusing completed checkpoint selection: $OUT/checkpoint_selection.json"
fi

SELECTED="$(tr -d '\r\n' < "$OUT/checkpoint_selection.checkpoint.txt")"
SELECTED_STEP="$(tr -d '\r\n' < "$OUT/checkpoint_selection.step.txt")"
[[ -f "$SELECTED" ]] || { echo "Selected checkpoint is missing: $SELECTED" >&2; exit 5; }

if [[ "$SELECTED_STEP" == "0" ]]; then
  echo "COMPLETE: pair-expanded triplets did not beat frozen stage 1 on role 2"
  exit 0
fi

null_names=(content_permuted content_permuted_b content_permuted_c)
null_checkpoints=()
for null_name in "${null_names[@]}"; do
  null_output="$OUT/training_${null_name}"
  if [[ ! -f "$null_output/last.ckpt" ]]; then
    [[ ! -e "$null_output" ]] || {
      echo "Incomplete null-arm directory requires inspection: $null_output" >&2
      exit 6
    }
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
  else
    echo "Reusing completed null arm: $null_output/last.ckpt"
  fi
  null_checkpoints+=(--checkpoint "null_${null_name}=$null_output/last.ckpt")
done

if [[ ! -f "$OUT/role3_pair_expanded_and_null_evaluation.json" ]]; then
  srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
    --policy "$EVIDENCE/confirmation_triplet_evidence.npz" \
    --checkpoint "official=data/e1/official_embedding_slim.pt" \
    --checkpoint "stage1_base=$BASE_CHECKPOINT" \
    --checkpoint "pair_expanded_correct=$SELECTED" \
    "${null_checkpoints[@]}" \
    --paired-reference stage1_base --formula-role 3 \
    --output "$OUT/role3_pair_expanded_and_null_evaluation.json" \
    --batch-size 64 --bootstrap-draws 5000 --device cuda
else
  echo "Reusing completed role-3 evaluation: $OUT/role3_pair_expanded_and_null_evaluation.json"
fi

test -f "$OUT/role2_checkpoint_evaluation.json"
test -f "$OUT/checkpoint_selection.json"
test -f "$OUT/role3_pair_expanded_and_null_evaluation.json"
echo "COMPLETE: resumed ChemAware pair-expanded run -> $OUT"
