#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=g8r_p2_train_%j.out
#SBATCH --error=g8r_p2_train_%j.err

# Training is fail-closed: no model is exported unless nested formula-isolated
# OOF gates protect overall Recall@1, near Recall@1, MRR and error transitions.
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Required by PyTorch deterministic mode for CUDA/cuBLAS matrix operations.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

required=(
  data/validation/g8r_p2_listwise_cache.npz
  data/validation/g8r_p2_listwise_cache.json
  data/validation/g8r_p2_cache_headroom.json
  tasks/g8r_p2_listwise_core.py
  tasks/train_g8r_p2_listwise.py
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "MISSING: $file" >&2; exit 2; }
done
[[ ! -e data/validation/g8r_p2_listwise_model.pt ]] || {
  echo "REFUSE: frozen P2 model already exists" >&2
  exit 3
}

python -u tasks/train_g8r_p2_listwise.py \
  --device cuda \
  --folds 5 \
  --epochs 12 \
  --patience 3 \
  --seeds 20260824 20260825 20260826

echo "=== P2 listwise training completed: $(date) ==="
