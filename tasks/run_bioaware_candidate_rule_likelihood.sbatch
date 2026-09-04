#!/bin/bash
#SBATCH --job-name=bioaware_g2rule
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_g2rule_%j.out
#SBATCH --error=bioaware_g2rule_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== BioAware G2 candidate rule likelihood started: $(date) ==="
python -u tasks/audit_bioaware_candidate_rule_likelihood.py
echo "=== BioAware G2 candidate rule likelihood completed: $(date) ==="
