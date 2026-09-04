#!/bin/bash
#SBATCH --job-name=noise_r2_preflight
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/noise_r2_preflight_%j.out
#SBATCH --error=data/validation/noise_r2_preflight_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

OUTPUT=data/validation/g8r_noise_final_r2_preflight
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite $OUTPUT" >&2; exit 4; }
echo "=== final noise R2 preflight started: $(date) ==="
python -u tasks/test_noise_final_r2_shared_encoder.py
python -u tasks/audit_noise_final_r2_preflight.py --output-dir "$OUTPUT"
python -u tasks/validate_noise_final_r2_preflight.py --output-dir "$OUTPUT"
echo "=== final noise R2 preflight completed: $(date) ==="
