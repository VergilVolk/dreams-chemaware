#!/bin/bash
#SBATCH --job-name=chem_mb12k
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/chem_mb12k_%j.out
#SBATCH --error=data/validation/chem_mb12k_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
BASE="data/validation/chemaware_high_coverage_native/run_2340524"
PHASEA="data/validation/chemaware_online_role_budget_native/run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt"
CORPUS="data/validation/chemaware_massbank_layered_12k_native_v1_20260928"
OUT="data/validation/chemaware_massbank_layered_native/run_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

required_files=(
  "$OFFICIAL" "$ARCHITECTURE" "$DATA" "$PHASEA"
  "$CORPUS/train_pool.npz" "$CORPUS/val_pool.npz" "$CORPUS/report.json"
  "$CORPUS/chemical_source_event_ledger.tsv" "$CORPUS/corpus_event_manifest.tsv"
  "$BASE/evidence/selection_triplet_evidence.npz"
  "$BASE/evidence/confirmation_triplet_evidence.npz"
  tasks/test_chemaware_massbank_layered_runtime.py
  tasks/train_chemaware_dreams_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/select_chemaware_residual_checkpoint.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing ChemAware MassBank-layered file: $required_file" >&2
    exit 2
  }
done

python -u tasks/test_chemaware_massbank_layered_runtime.py --corpus "$CORPUS"

sha256sum \
  "$CORPUS/train_pool.npz" "$CORPUS/val_pool.npz" "$CORPUS/report.json" \
  "$CORPUS/chemical_source_event_ledger.tsv" "$CORPUS/corpus_event_manifest.tsv" \
  "$PHASEA" > "$OUT/input_sha256sums.txt"
cp "$CORPUS/report.json" "$OUT/frozen_corpus_report.json"

# Continue directly from the protected +2.1266 pp embedding.  The model,
# preprocessor, random native reference sampling, triplet loss and Adam
# optimizer are the repository's mature DreaMS implementations.
srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py \
  --data "$DATA" \
  --train-pool "$CORPUS/train_pool.npz" \
  --val-pool "$CORPUS/val_pool.npz" \
  --official-checkpoint "$PHASEA" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --output "$OUT/training" \
  --seed 3407 --lr 2e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
  --max-epochs 2 --max-steps 1000 --checkpoint-mode fixed_steps \
  --save-every-n-steps 250 --n-highest-peaks 100 --device cuda

checkpoint_args=(
  --checkpoint "official=$OFFICIAL"
  --checkpoint "phaseA_2pp=$PHASEA"
)
mapfile -t trained_checkpoints < <(
  find "$OUT/training" -maxdepth 1 -type f -name 'step-*.ckpt' | sort
)
[[ ${#trained_checkpoints[@]} -eq 4 ]] || {
  echo "Expected four fixed-step checkpoints, found ${#trained_checkpoints[@]}" >&2
  exit 5
}
for checkpoint in "${trained_checkpoints[@]}"; do
  checkpoint_args+=(--checkpoint "massbank12k_$(basename "$checkpoint" .ckpt)=$checkpoint")
done

# Role 2 is the only checkpoint-selection set.  Every candidate is compared
# against Phase A, not against the weaker original model.
srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/selection_triplet_evidence.npz" \
  "${checkpoint_args[@]}" \
  --paired-reference phaseA_2pp --formula-role 2 \
  --output "$OUT/role2_checkpoint_evaluation.json" \
  --batch-size 64 --bootstrap-draws 10000 --device cuda

python -u tasks/select_chemaware_residual_checkpoint.py \
  --evaluation "$OUT/role2_checkpoint_evaluation.json" \
  --base-name phaseA_2pp --require-positive-formula-ci \
  --output "$OUT/checkpoint_selection.json"
SELECTED="$(tr -d '\r\n' < "$OUT/checkpoint_selection.checkpoint.txt")"
SELECTED_STEP="$(tr -d '\r\n' < "$OUT/checkpoint_selection.step.txt")"

if [[ "$SELECTED_STEP" == "0" ]]; then
  echo "CHEMAWARE_MASSBANK12K_ROLE2_STOP: retain protected Phase A; role 3 remains unopened"
  echo "COMPLETE: MassBank-layered role-2 audit -> $OUT"
  exit 0
fi

SELECTED_REAL="$(realpath "$SELECTED")"
srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/confirmation_triplet_evidence.npz" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "phaseA_2pp=$PHASEA" \
  --checkpoint "massbank12k_selected=$SELECTED_REAL" \
  --paired-reference phaseA_2pp --formula-role 3 \
  --output "$OUT/role3_evaluation.json" \
  --batch-size 64 --bootstrap-draws 10000 --device cuda

mkdir -p "$OUT/protected_candidate"
ln "$SELECTED_REAL" "$OUT/protected_candidate/chemaware_massbank12k_selected.ckpt"
cp "$CORPUS/report.json" "$OUT/protected_candidate/corpus_report.json"
cp "$CORPUS/chemical_source_event_ledger.tsv" "$OUT/protected_candidate/"
cp "$OUT/input_sha256sums.txt" "$OUT/protected_candidate/"
cp "$OUT/checkpoint_selection.json" "$OUT/protected_candidate/"
cp "$OUT/role2_checkpoint_evaluation.json" "$OUT/protected_candidate/"
cp "$OUT/role3_evaluation.json" "$OUT/protected_candidate/"
cp "$OUT/training/report.json" "$OUT/protected_candidate/training_report.json"
(
  cd "$OUT/protected_candidate"
  sha256sum -- * > SHA256SUMS
)
echo "COMPLETE: protected ChemAware MassBank-layered candidate -> $OUT/protected_candidate"
