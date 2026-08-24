#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=g8r_p2_cache_audit_%j.out
#SBATCH --error=g8r_p2_cache_audit_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

for file in \
  data/validation/g8r_p2_listwise_cache.npz \
  data/validation/g8r_p2_listwise_cache.json \
  tasks/g8r_p2_listwise_core.py \
  tasks/audit_g8r_p2_cache_headroom.py; do
  [[ -f "$file" ]] || { echo "MISSING: $file" >&2; exit 2; }
done

python -u tasks/audit_g8r_p2_cache_headroom.py --target-delta 0.04

echo "=== P2 cache/headroom audit completed: $(date) ==="
