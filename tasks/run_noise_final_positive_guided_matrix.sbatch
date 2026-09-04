#!/bin/bash
#SBATCH --job-name=noise_pos_matrix
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/noise_pos_matrix_%j.out
#SBATCH --error=data/validation/noise_pos_matrix_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUTPUT="data/validation/g8r_noise_final_positive_guided_matrix"
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite positive-guided matrix: $OUTPUT" >&2; exit 4;
}

echo "=== positive-guided full error-space matrix started: $(date) ==="
python -u tasks/test_noise_final_positive_guided_matrix.py
python -u tasks/audit_noise_final_positive_guided_matrix.py \
  --device cuda \
  --batch-size 256 \
  --doses 0.25 0.50 0.75 1.00
echo "=== positive-guided full error-space matrix completed: $(date) ==="
