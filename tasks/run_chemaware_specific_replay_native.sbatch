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
SOURCE_RUN="data/validation/chemaware_pair_expanded_native/run_2342187"
CACHE="$SOURCE_RUN/stage1_embedding_cache"
OUT="data/validation/chemaware_specific_replay_native/run_${SLURM_JOB_ID}"
PAIR_BANK="$OUT/candidate_discovery_bank"
BANK="$OUT/specific_replay_triplets"
CORRECT="$OUT/training_correct"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

required_files=(
  "$BASE_CHECKPOINT"
  "$EVIDENCE/selection_triplet_evidence.npz"
  "$EVIDENCE/confirmation_triplet_evidence.npz"
  "$EVIDENCE/train_triplet_evidence.npz"
  "$CACHE/rows.npy"
  "$CACHE/embeddings_f32.npy"
  tasks/build_chemaware_reference_aligned_native_triplets.py
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
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required specific-replay file: $required_file" >&2
    exit 2
  }
done

python -u tasks/test_chemaware_specific_replay_native_triplets.py

# This is a discovery bank, not the optimizer curriculum. Re-open the frozen
# candidate tensor to recover at most two directional chemical candidates;
# generic fallback events below are discarded by the specific-replay builder.
# Keeping two candidates prevents the four arms from manufacturing agreement
# merely by exhausting the same small candidate set.
python -u tasks/build_chemaware_reference_aligned_native_triplets.py \
  --evidence-dir "$EVIDENCE" --embedding-cache "$CACHE" \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --output "$PAIR_BANK" --margin 0.1 --candidates-per-query 8 \
  --chemical-candidates-per-query 2 \
  --negative-references-per-candidate 2 \
  --explicit-positive-references-per-negative 2 \
  --max-active-spectrum-events-per-query 18 \
  --chemical-hardness-window 0.75 \
  --min-chemical-activation-probability 0.05 \
  --min-train-queries 3500 --min-candidate-events 5000 \
  --min-spectrum-events 9000 --min-train-formulas 2000 \
  --min-chemical-candidates 500 --min-chemical-spectrum-events 1400

# Freeze one safety boundary per query, then retain the strictest sufficiently
# covered counterfactual chemical profile (null agreement 0, 1, or 2).
python -u tasks/build_chemaware_specific_replay_native_triplets.py \
  --pair-bank "$PAIR_BANK" \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --output "$BANK" --chemical-events-per-query 4 \
  --maximum-null-candidate-agreement 2 \
  --min-train-queries 3500 --min-specific-chemical-events 300 \
  --min-specific-chemical-queries 150 --min-chemical-fraction 0.06 \
  --min-active-chemical-fraction 0.15 \
  --max-null-mean-hinge-gap 0.05 --max-null-mean-activation-gap 0.10 \
  --max-null-event-hinge-gap 0.25 --max-null-event-activation-gap 0.25

# The model, optimizer, preprocessing and loss remain the native DreaMS stack.
# This is a low-dose residual continuation from the confirmed +1.8144 pp
# checkpoint. The failed pair-expanded run showed that 2,000--3,000 further
# steps can exchange corrections for new errors, so this run is capped below
# one complete pass through the approximately 4.8k-event curriculum.
srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --train-pool "$BANK/train_pool.npz" --val-pool "$BANK/selection_pool.npz" \
  --official-checkpoint "$BASE_CHECKPOINT" \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --output "$CORRECT" --seed 3407 --lr 3e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
  --max-epochs 2 --max-steps 1000 --checkpoint-mode fixed_steps \
  --save-every-n-steps 250 --n-highest-peaks 100 --device cuda

checkpoint_args=(
  --checkpoint "official=data/e1/official_embedding_slim.pt"
  --checkpoint "stage1_base=$BASE_CHECKPOINT"
)
mapfile -t trained_checkpoints < <(find "$CORRECT" -maxdepth 1 -type f -name 'step-*.ckpt' | sort)
[[ ${#trained_checkpoints[@]} -eq 4 ]] || {
  echo "Expected four residual checkpoints, found ${#trained_checkpoints[@]}" >&2
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
