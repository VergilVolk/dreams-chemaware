#!/bin/bash
#SBATCH --job-name=noise_final_d1b
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-14%3
#SBATCH --output=data/validation/noise_final_d1b_%A_%a.out
#SBATCH --error=data/validation/noise_final_d1b_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

required=(
  data/validation/g8r_noise_final_d0_manifest/decision.json
  data/validation/g8r_noise_final_d0_manifest/manifest.npz
  data/validation/g8r_error_atlas_listwise_cache.npz
  data/validation/g8r_p2_official_embeddings.npz
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "Missing D1b dependency: $file" >&2; exit 3; }
done

SEEDS=(20260825 20260826 20260827)
TASK_ID="${SLURM_ARRAY_TASK_ID}"
FOLD=$((TASK_ID % 5))
SEED_INDEX=$((TASK_ID / 5))
SEED="${SEEDS[$SEED_INDEX]}"
OUTPUT_ROOT="data/validation/g8r_noise_final_d1b_adapter"
OUTPUT="$OUTPUT_ROOT/seed_${SEED}/fold_${FOLD}"
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite D1b fold: $OUTPUT" >&2; exit 4; }

echo "=== final noise D1b seed=$SEED fold=$FOLD started: $(date) ==="
python -u tasks/train_noise_final_d1_adapter.py \
  --outer-fold "$FOLD" \
  --seed "$SEED" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda \
  --batch-size 64 \
  --epochs 4 \
  --lambda-safety 8.0 \
  --safety-margin 0.0
echo "=== final noise D1b seed=$SEED fold=$FOLD completed: $(date) ==="

