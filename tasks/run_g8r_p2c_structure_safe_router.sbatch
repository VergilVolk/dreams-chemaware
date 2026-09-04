#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=g8r_p2c_router_%j.out
#SBATCH --error=g8r_p2c_router_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

required=(
  data/validation/g8r_p2_listwise_cache.npz
  data/validation/g8r_p2b_rank_fusion.json
  tasks/g8r_p2_rank_fusion_core.py
  tasks/train_g8r_p2b_rank_fusion.py
  tasks/audit_g8r_p2c_structure_safe_router.py
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "MISSING: $file" >&2; exit 2; }
done
[[ ! -e data/validation/g8r_p2c_structure_safe_router.json ]] || {
  echo "REFUSE: output already exists" >&2
  exit 3
}

python -u tasks/audit_g8r_p2c_structure_safe_router.py --bootstrap 5000 --seed 20260830
echo "=== P2c structure-safe router completed: $(date) ==="
