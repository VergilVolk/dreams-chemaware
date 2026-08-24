#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=g8r_p3_v3_dry_%j.out
#SBATCH --error=g8r_p3_v3_dry_%j.err

# Mandatory server-side capacity/provenance audit. Writes no manifests and does
# not consume the one-shot P3 lock.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -u tasks/build_g8r_p3_test.py \
  --dry-run \
  --mces-workers "${SLURM_CPUS_PER_TASK:-8}"

echo "=== P3 v3 dry-run completed: $(date) ==="
echo "Only submit run_g8r_p3_lock.sbatch if main=3000 and near-core>=450 passed."
