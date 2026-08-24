#!/bin/bash
#SBATCH --job-name=g8r_peak_specific
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/g8r_peak_specific_%j.out
#SBATCH --error=data/validation/g8r_peak_specific_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== causal peak specificity started: $(date) ==="
python -u tasks/analyze_g8r_causal_peak_specificity.py \
  --bootstrap 5000 \
  --min-controls 5 \
  --seed 20260824
echo "=== causal peak specificity completed: $(date) ==="
