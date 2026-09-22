#!/bin/bash
#SBATCH --job-name=chem_resid_s2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_resid_s2_%j.out
#SBATCH --error=data/validation/chem_resid_s2_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BASE_CHECKPOINT="${BASE_CHECKPOINT:-data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt}"
EVIDENCE="${EVIDENCE:-data/validation/chemaware_high_coverage_native/run_2340524/evidence}"
OUT="data/validation/chemaware_residual_native_stage2/run_${SLURM_JOB_ID}"
CACHE="$OUT/stage1_embedding_cache"
BANK030="$OUT/triplets_w030"
BANK050="$OUT/triplets_w050"
CORRECT="$OUT/training_correct"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

required_files=(
  "$BASE_CHECKPOINT"
  "$EVIDENCE/train_triplet_evidence.npz"
  "$EVIDENCE/selection_triplet_evidence.npz"
  "$EVIDENCE/confirmation_triplet_evidence.npz"
  tasks/encode_chemaware_checkpoint_manifest_rows.py
  tasks/build_chemaware_residual_native_triplets.py
  tasks/select_chemaware_residual_triplet_bank.py
  tasks/select_chemaware_residual_checkpoint.py
  tasks/test_chemaware_residual_native_triplets.py
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

python -u tasks/test_chemaware_residual_native_triplets.py

# Cache the spectrum-only embeddings produced by yesterday's +1.8144 pp model.
# No candidate label or rule score is used by this encoder pass.
srun --export=ALL --preserve-env python -u tasks/encode_chemaware_checkpoint_manifest_rows.py \
  --checkpoint "$BASE_CHECKPOINT" \
  --evidence-dir "$EVIDENCE" \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --official-checkpoint data/e1/official_embedding_slim.pt \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --output "$CACHE" --batch-size 64 --n-highest-peaks 100 --device cuda

# Two predeclared hardness windows are built from the same cache.  Role 2 may
# select between them, but role 3 cannot affect the triplet bank.
python -u tasks/build_chemaware_residual_native_triplets.py \
  --evidence-dir "$EVIDENCE" --embedding-cache "$CACHE" \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --output "$BANK030" --margin 0.1 --chemical-hardness-window 0.30 \
  --events-per-query 2 --min-train-queries 3500 --min-train-events 6500 \
  --min-train-formulas 2000 --min-chemical-events 1000

python -u tasks/build_chemaware_residual_native_triplets.py \
  --evidence-dir "$EVIDENCE" --embedding-cache "$CACHE" \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --output "$BANK050" --margin 0.1 --chemical-hardness-window 0.50 \
  --events-per-query 2 --min-train-queries 3500 --min-train-events 6500 \
  --min-train-formulas 2000 --min-chemical-events 1000

python -u tasks/select_chemaware_residual_triplet_bank.py \
  --bank "w030=$BANK030" --bank "w050=$BANK050" \
  --min-relative-violation 0.95 \
  --output "$OUT/triplet_bank_selection.json"
BANK="$(tr -d '\r\n' < "$OUT/triplet_bank_selection.path.txt")"
[[ -f "$BANK/train_pool.npz" ]] || { echo "Selected triplet bank is invalid: $BANK" >&2; exit 4; }

# The model, preprocessing, random reference sampling, cosine triplet loss and
# Adam optimizer are still the repository's native DreaMS implementation.
srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --train-pool "$BANK/train_pool.npz" --val-pool "$BANK/selection_pool.npz" \
  --official-checkpoint "$BASE_CHECKPOINT" \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --output "$CORRECT" --seed 3407 --lr 5e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
  --max-epochs 20 --max-steps 1000 --checkpoint-mode fixed_steps \
  --save-every-n-steps 250 --n-highest-peaks 100 --device cuda

checkpoint_args=(
  --checkpoint "official=data/e1/official_embedding_slim.pt"
  --checkpoint "stage1_base=$BASE_CHECKPOINT"
)
for checkpoint in "$CORRECT"/step-*.ckpt; do
  [[ -f "$checkpoint" ]] || continue
  stem="$(basename "$checkpoint" .ckpt)"
  checkpoint_args+=(--checkpoint "residual_${stem}=$checkpoint")
done
[[ ${#checkpoint_args[@]} -ge 6 ]] || { echo "No residual step checkpoints found" >&2; exit 5; }

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
  echo "COMPLETE: residual stage 2 did not beat the frozen +1.8144 pp stage-1 model on role 2"
  exit 0
fi

# Only after a correct-chemistry checkpoint beats stage 1 do the three matched
# semantic-null arms receive exactly the same optimizer-step budget.
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
  --checkpoint "residual_correct=$SELECTED" \
  "${null_checkpoints[@]}" \
  --paired-reference stage1_base --formula-role 3 \
  --output "$OUT/role3_residual_and_null_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

test -f "$OUT/triplet_bank_selection.json"
test -f "$OUT/checkpoint_selection.json"
test -f "$OUT/role3_residual_and_null_evaluation.json"
echo "COMPLETE: ChemAware residual native-triplet stage 2 -> $OUT"
