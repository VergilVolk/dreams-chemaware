#!/bin/bash
#SBATCH --job-name=chem_2pp_save
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/chem_2pp_save_%j.out
#SBATCH --error=data/validation/chem_2pp_save_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OFFICIAL="data/e1/official_embedding_slim.pt"
BASE="data/validation/chemaware_high_coverage_native/run_2340524"
SOURCE_RUN="data/validation/chemaware_max_boundary_remine/run_2343962"
SOURCE_CHECKPOINT="$SOURCE_RUN/phase_a/training/step-002000.ckpt"
OUT="data/validation/chemaware_phasea_2pp_protected/run_${SLURM_JOB_ID}"
REBUILD="$OUT/rebuild"
EVALUATION="$OUT/role2_reproduction.json"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

required_files=(
  "$OFFICIAL"
  "$BASE/triplets/report.json"
  "$BASE/triplets/train_pool.npz"
  "$BASE/triplets/val_pool.npz"
  "$BASE/evidence/train_triplet_evidence.npz"
  "$BASE/evidence/selection_triplet_evidence.npz"
  data/e1/e1_train_triplet_pool_10ppm.npz
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz
  data/validation/chemaware_corrected_manifest_tokens_v1/rows.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/official_embeddings_f32.npy
  dreams/models/pretrained/ssl_model_server.pt
  tasks/build_chemaware_max_boundary_native_triplets.py
  tasks/train_chemaware_specific_replay_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/freeze_chemaware_phasea_2pp_artifact.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required +2.1266 pp protection file: $required_file" >&2
    exit 2
  }
done

CHECKPOINT="$SOURCE_CHECKPOINT"
TRIPLET_REPORT="$SOURCE_RUN/phase_a/triplets/report.json"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Original run-2343962 checkpoint was cleaned; rebuilding exact Phase A"
  python -u tasks/build_chemaware_max_boundary_native_triplets.py \
    --base-bank "$BASE/triplets" \
    --evidence "$BASE/evidence/train_triplet_evidence.npz" \
    --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
    --embedding-rows data/validation/chemaware_corrected_manifest_tokens_v1/rows.npy \
    --official-embeddings data/validation/chemaware_corrected_manifest_tokens_v1/official_embeddings_f32.npy \
    --dreams-replay-pool data/e1/e1_train_triplet_pool_10ppm.npz \
    --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
    --output "$REBUILD/triplets" \
    --margin 0.1 --negative-references-per-error 2 \
    --chemical-candidates-per-error 2 --dreams-replay-events 1024 \
    --seed 3407 --min-error-event-fraction 0.20

  srun --export=ALL --preserve-env python -u tasks/train_chemaware_specific_replay_native.py \
    --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
    --train-pool "$REBUILD/triplets/train_pool.npz" \
    --val-pool "$REBUILD/triplets/val_pool.npz" \
    --official-checkpoint "$OFFICIAL" \
    --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
    --output "$REBUILD/training" \
    --seed 3407 --lr 5e-6 --weight-decay 0 \
    --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
    --max-epochs 3 --max-steps 2000 --checkpoint-mode fixed_steps \
    --save-every-n-steps 2000 --n-highest-peaks 100 --device cuda
  CHECKPOINT="$REBUILD/training/step-002000.ckpt"
  TRIPLET_REPORT="$REBUILD/triplets/report.json"
fi
test -f "$CHECKPOINT"
test -f "$TRIPLET_REPORT"

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/selection_triplet_evidence.npz" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "phaseA_step-002000=$CHECKPOINT" \
  --formula-role 2 --output "$EVALUATION" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

python -u tasks/freeze_chemaware_phasea_2pp_artifact.py \
  --checkpoint "$CHECKPOINT" \
  --evaluation "$EVALUATION" \
  --triplet-report "$TRIPLET_REPORT" \
  --output "$OUT/protected_artifact"

test -f "$OUT/protected_artifact/chemaware_phasea_max_boundary_step2000.ckpt"
test -f "$OUT/protected_artifact/artifact_manifest.json"
test -f "$OUT/protected_artifact/SHA256SUMS"
echo "COMPLETE: protected ChemAware +2.1266 pp role-2 embedding -> $OUT/protected_artifact"
