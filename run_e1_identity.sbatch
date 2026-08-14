#!/bin/bash
#SBATCH --gpus=1
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --output=e1_identity_%A_%a.out
#SBATCH --error=e1_identity_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
TRAIN_POOL="data/e1/e1_train_triplet_pool.npz"
VAL_POOL="data/e1/e1_val_triplet_pool.npz"
BASE="dreams/models/pretrained/ssl_model_server.pt"

for REQUIRED in "$DATA" "$TRAIN_POOL" "$VAL_POOL" "$BASE"; do
    if [[ ! -f "$REQUIRED" ]]; then
        echo "Missing required E1 input: $REQUIRED" >&2
        exit 2
    fi
done

echo "E1 identity-only hard-triplet fine-tuning"
echo "Seed: $SEED"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python -u tasks/train_e1_identity.py \
    --data "$DATA" \
    --train-pool "$TRAIN_POOL" \
    --val-pool "$VAL_POOL" \
    --base-ckpt "$BASE" \
    --seed "$SEED" \
    --epochs 10 \
    --batch-size 16 \
    --grad-accum 2 \
    --lr 5e-6 \
    --margin 0.1 \
    --n-highest-peaks 100 \
    --val-triplets 10000 \
    --patience 3 \
    --device cuda \
    --amp \
    --output-dir data/e1/runs

python -u tasks/eval_e0_baseline.py \
    --data "$DATA" \
    --ckpt "data/e1/runs/seed_${SEED}/best_e1.pt" \
    --fold val \
    --device cuda \
    --batch-size 64 \
    --output-dir "data/validation/e1_identity/seed_${SEED}"

echo "Finished: $(date)"
