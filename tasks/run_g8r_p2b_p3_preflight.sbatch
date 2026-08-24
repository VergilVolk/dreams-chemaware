#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=g8r_p2b_p3_preflight_%j.out
#SBATCH --error=g8r_p2b_p3_preflight_%j.err

# Hash/candidate-graph validation only. No model score is computed.
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -u tasks/eval_g8r_p2b_on_sealed_p3.py --preflight-only
echo "=== frozen P2b P3 preflight completed: $(date) ==="
