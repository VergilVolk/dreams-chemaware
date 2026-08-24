#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=g8r_p3_candidate_ambiguity_audit_%j.out
#SBATCH --error=g8r_p3_candidate_ambiguity_audit_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

python -u tasks/audit_g8r_p3_candidate_ambiguity_router.py
echo "=== P3 candidate-ambiguity post-hoc audit completed: $(date) ==="
