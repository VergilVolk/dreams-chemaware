#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=g8r_p2b_fusion_%j.out
#SBATCH --error=g8r_p2b_fusion_%j.err

# CPU-only deterministic P2b diagnostic/export.  It never reads sealed P3.
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

required=(
  data/validation/g8r_p2_listwise_cache.npz
  data/validation/g8r_p2_listwise_cache.json
  tasks/g8r_p2_listwise_core.py
  tasks/g8r_p2_rank_fusion_core.py
  tasks/train_g8r_p2b_rank_fusion.py
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "MISSING: $file" >&2; exit 2; }
done
[[ ! -e data/validation/g8r_p2b_rank_fusion.json ]] || {
  echo "REFUSE: frozen P2b artifact already exists" >&2
  exit 3
}
[[ ! -e data/validation/g8r_p2b_rank_fusion.selection.json ]] || {
  echo "REFUSE: P2b selection report already exists" >&2
  exit 4
}

python -u tasks/train_g8r_p2b_rank_fusion.py \
  --folds 5 \
  --weight-denominator 10 \
  --bootstrap 2000 \
  --seed 20260825

echo "=== P2b rank fusion completed: $(date) ==="
