#!/bin/bash
#SBATCH --job-name=chem_maxbound
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_maxbound_%j.out
#SBATCH --error=data/validation/chem_maxbound_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OFFICIAL="data/e1/official_embedding_slim.pt"
STAGE1="data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt"
BASE="data/validation/chemaware_high_coverage_native/run_2340524"
OUT="data/validation/chemaware_max_boundary_native/run_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

required_files=(
  "$OFFICIAL"
  "$STAGE1"
  "$BASE/triplets/report.json"
  "$BASE/triplets/train_pool.npz"
  "$BASE/triplets/val_pool.npz"
  "$BASE/evidence/train_triplet_evidence.npz"
  "$BASE/evidence/selection_triplet_evidence.npz"
  "$BASE/evidence/confirmation_triplet_evidence.npz"
  data/e1/e1_train_triplet_pool_10ppm.npz
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz
  data/validation/chemaware_corrected_manifest_tokens_v1/rows.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/official_embeddings_f32.npy
  dreams/models/pretrained/ssl_model_server.pt
  tasks/build_chemaware_max_boundary_native_triplets.py
  tasks/test_chemaware_max_boundary_native_triplets.py
  tasks/train_chemaware_specific_replay_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/select_chemaware_residual_checkpoint.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required max-boundary file: $required_file" >&2
    exit 2
  }
done

python -u tasks/test_chemaware_max_boundary_native_triplets.py

# The only algorithmic change is the empirical triplet distribution.  The
# downstream model, preprocessing, cosine hinge, Adam optimizer and full
# backbone update are the mature DreaMS native implementation.
python -u tasks/build_chemaware_max_boundary_native_triplets.py \
  --base-bank "$BASE/triplets" \
  --evidence "$BASE/evidence/train_triplet_evidence.npz" \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --embedding-rows data/validation/chemaware_corrected_manifest_tokens_v1/rows.npy \
  --official-embeddings data/validation/chemaware_corrected_manifest_tokens_v1/official_embeddings_f32.npy \
  --dreams-replay-pool data/e1/e1_train_triplet_pool_10ppm.npz \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --output "$OUT/triplets" \
  --margin 0.1 --negative-references-per-error 2 \
  --chemical-candidates-per-error 2 --dreams-replay-events 1024 \
  --seed 3407 --min-error-event-fraction 0.20

srun --export=ALL --preserve-env python -u tasks/train_chemaware_specific_replay_native.py \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --train-pool "$OUT/triplets/train_pool.npz" \
  --val-pool "$OUT/triplets/val_pool.npz" \
  --official-checkpoint "$OFFICIAL" \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --output "$OUT/training" \
  --seed 3407 --lr 5e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
  --max-epochs 4 --max-steps 3000 --checkpoint-mode fixed_steps \
  --save-every-n-steps 500 --n-highest-peaks 100 --device cuda

checkpoint_args=(
  --checkpoint "official=$OFFICIAL"
  --checkpoint "stage1_base=$STAGE1"
)
mapfile -t trained_checkpoints < <(
  find "$OUT/training" -maxdepth 1 -type f -name 'step-*.ckpt' | sort
)
[[ ${#trained_checkpoints[@]} -eq 6 ]] || {
  echo "Expected six max-boundary checkpoints, found ${#trained_checkpoints[@]}" >&2
  exit 5
}
for checkpoint in "${trained_checkpoints[@]}"; do
  checkpoint_args+=(--checkpoint "maxbound_$(basename "$checkpoint" .ckpt)=$checkpoint")
done

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/selection_triplet_evidence.npz" \
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
  for checkpoint in "${trained_checkpoints[@]}"; do
    /bin/rm -f -- "$checkpoint"
  done
  echo "CHEMAWARE_MAX_BOUNDARY_ROLE2_STOP: retrieval-aligned curriculum did not safely beat +1.8144 pp"
  exit 0
fi

SELECTED_REAL="$(realpath "$SELECTED")"
for checkpoint in "${trained_checkpoints[@]}"; do
  checkpoint_real="$(realpath "$checkpoint")"
  if [[ "$checkpoint_real" != "$SELECTED_REAL" ]]; then
    /bin/rm -f -- "$checkpoint"
  fi
done

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/confirmation_triplet_evidence.npz" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "stage1_base=$STAGE1" \
  --checkpoint "max_boundary=$SELECTED_REAL" \
  --paired-reference stage1_base --formula-role 3 \
  --output "$OUT/role3_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

test -f "$OUT/triplets/report.json"
test -f "$OUT/checkpoint_selection.json"
test -f "$OUT/role3_evaluation.json"
echo "COMPLETE: ChemAware max-boundary native embedding -> $OUT"
