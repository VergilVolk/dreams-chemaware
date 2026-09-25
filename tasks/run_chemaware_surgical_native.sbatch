#!/bin/bash
#SBATCH --job-name=chem_surgical
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_surgical_%j.out
#SBATCH --error=data/validation/chem_surgical_%j.err

set -euo pipefail
echo "STOPPED: superseded by tasks/run_chemaware_max_boundary_native.sbatch; no GPU work was authorized" >&2
exit 2
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OFFICIAL="data/e1/official_embedding_slim.pt"
STAGE1_CHECKPOINT="data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt"
BASE_BANK="data/validation/chemaware_high_coverage_native/run_2340524/triplets"
EVIDENCE="data/validation/chemaware_high_coverage_native/run_2340524/evidence"
SPECIFIC_BANK="data/validation/chemaware_specific_replay_native/run_2343546/specific_replay_triplets"
OUT="data/validation/chemaware_surgical_native/run_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

required_files=(
  "$OFFICIAL"
  "$STAGE1_CHECKPOINT"
  "$BASE_BANK/report.json"
  "$BASE_BANK/train_pool.npz"
  "$BASE_BANK/val_pool.npz"
  "$SPECIFIC_BANK/report.json"
  "$EVIDENCE/selection_triplet_evidence.npz"
  "$EVIDENCE/confirmation_triplet_evidence.npz"
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz
  dreams/models/pretrained/ssl_model_server.pt
  tasks/build_chemaware_surgical_native_triplets.py
  tasks/test_chemaware_surgical_native_triplets.py
  tasks/train_chemaware_specific_replay_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/select_chemaware_residual_checkpoint.py
  tasks/summarize_chemaware_surgical_native.py
)
for arm in content_permuted content_permuted_b content_permuted_c; do
  required_files+=(
    "$SPECIFIC_BANK/train_pool_${arm}.npz"
  )
done
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required surgical-run file: $required_file" >&2
    exit 2
  }
done

python -u tasks/test_chemaware_surgical_native_triplets.py
python -u tasks/train_chemaware_specific_replay_native.py --help >/dev/null

# Dose is replacement count, not dataset expansion. Both curricula retain the
# exact event budget and protect every official-hard stage-1 event.
for dose in 1 2; do
  python -u tasks/build_chemaware_surgical_native_triplets.py \
    --base-bank "$BASE_BANK" --specific-bank "$SPECIFIC_BANK" \
    --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
    --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
    --output "$OUT/triplets_dose${dose}" \
    --chemical-events-per-query "$dose" \
    --min-train-replacements 100 --min-train-replacement-queries 50 \
    --min-untouched-fraction 0.95
done

train_native() {
  local train_pool="$1"
  local val_pool="$2"
  local output="$3"
  local maximum_steps="$4"
  local save_steps="$5"
  srun --export=ALL --preserve-env python -u tasks/train_chemaware_specific_replay_native.py \
    --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
    --train-pool "$train_pool" --val-pool "$val_pool" \
    --official-checkpoint data/e1/official_embedding_slim.pt \
    --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
    --output "$output" --seed 3407 --lr 5e-6 --weight-decay 0 \
    --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
    --max-epochs 4 --max-steps "$maximum_steps" --checkpoint-mode fixed_steps \
    --save-every-n-steps "$save_steps" --n-highest-peaks 100 --device cuda
}

# Relearn jointly from the official embedding. Continuing from stage 1 was the
# failed residual intervention: it produced local R@1 motion while damaging
# micro-AUC. Fixed steps make dose and checkpoint selection externally auditable.
for dose in 1 2; do
  train_native \
    "$OUT/triplets_dose${dose}/train_pool.npz" \
    "$OUT/triplets_dose${dose}/val_pool.npz" \
    "$OUT/training_dose${dose}" 3000 1000
done

checkpoint_args=(
  --checkpoint "official=$OFFICIAL"
  --checkpoint "stage1_base=$STAGE1_CHECKPOINT"
)
mapfile -t trained_checkpoints < <(
  find "$OUT" -mindepth 2 -maxdepth 2 -type f \
    -path '*/training_dose*/step-*.ckpt' | sort
)
[[ ${#trained_checkpoints[@]} -eq 6 ]] || {
  echo "Expected six surgical checkpoints, found ${#trained_checkpoints[@]}" >&2
  exit 5
}

CHECKPOINT_LEDGER="$OUT/correct_checkpoint_sha256.tsv"
printf 'sha256\tbytes\tpath\n' > "$CHECKPOINT_LEDGER"
for checkpoint in "${trained_checkpoints[@]}"; do
  checkpoint_sha256="$(sha256sum "$checkpoint" | awk '{print $1}')"
  checkpoint_bytes="$(stat -c '%s' "$checkpoint")"
  [[ "$checkpoint_bytes" -le 1073741824 ]] || {
    echo "Weights-only checkpoint unexpectedly exceeds 1 GiB: $checkpoint" >&2
    exit 7
  }
  printf '%s\t%s\t%s\n' \
    "$checkpoint_sha256" "$checkpoint_bytes" "$checkpoint" >> "$CHECKPOINT_LEDGER"
  if [[ "$checkpoint" == *"/training_dose1/"* ]]; then
    dose_name="dose1"
  elif [[ "$checkpoint" == *"/training_dose2/"* ]]; then
    dose_name="dose2"
  else
    echo "Cannot determine surgical dose for $checkpoint" >&2
    exit 8
  fi
  step_name="$(basename "$checkpoint" .ckpt)"
  checkpoint_args+=(--checkpoint "${dose_name}_${step_name}=$checkpoint")
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
  for checkpoint in "${trained_checkpoints[@]}"; do
    /bin/rm -f -- "$checkpoint"
  done
  echo "CHEMAWARE_SURGICAL_ROLE2_STOP: neither fixed-budget dose improves stage 1 safely"
  exit 0
fi

SELECTED_REAL="$(realpath "$SELECTED")"
if [[ "$SELECTED_REAL" == *"/training_dose1/"* ]]; then
  SELECTED_DOSE=1
elif [[ "$SELECTED_REAL" == *"/training_dose2/"* ]]; then
  SELECTED_DOSE=2
else
  echo "Selected checkpoint is outside the surgical dose arms: $SELECTED_REAL" >&2
  exit 9
fi
for checkpoint in "${trained_checkpoints[@]}"; do
  checkpoint_real="$(realpath "$checkpoint")"
  if [[ "$checkpoint_real" != "$SELECTED_REAL" ]]; then
    /bin/rm -f -- "$checkpoint"
  fi
done

# Only the role-2-qualified dose earns matched semantic-null training.
null_checkpoint_args=()
for null_name in content_permuted content_permuted_b content_permuted_c; do
  null_output="$OUT/training_selected_${null_name}"
  train_native \
    "$OUT/triplets_dose${SELECTED_DOSE}/train_pool_${null_name}.npz" \
    "$OUT/triplets_dose${SELECTED_DOSE}/val_pool_${null_name}.npz" \
    "$null_output" "$SELECTED_STEP" "$SELECTED_STEP"
  mapfile -t produced < <(
    find "$null_output" -maxdepth 1 -type f -name 'step-*.ckpt' | sort
  )
  [[ ${#produced[@]} -eq 1 ]] || {
    echo "Expected one compact null checkpoint for $null_name" >&2
    exit 6
  }
  null_checkpoint_args+=(--checkpoint "null_${null_name}=${produced[0]}")
done

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$EVIDENCE/confirmation_triplet_evidence.npz" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "stage1_base=$STAGE1_CHECKPOINT" \
  --checkpoint "surgical_correct=$SELECTED_REAL" \
  "${null_checkpoint_args[@]}" \
  --paired-reference stage1_base --formula-role 3 \
  --output "$OUT/role3_surgical_and_null_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

python -u tasks/summarize_chemaware_surgical_native.py \
  --evaluation "$OUT/role3_surgical_and_null_evaluation.json" \
  --target-delta-recall1 0.05 \
  --output "$OUT/final_decision.json"

test -f "$OUT/triplets_dose${SELECTED_DOSE}/report.json"
test -f "$OUT/checkpoint_selection.json"
test -f "$OUT/role3_surgical_and_null_evaluation.json"
test -f "$OUT/final_decision.json"
echo "COMPLETE: ChemAware fixed-budget surgical native embedding -> $OUT"
