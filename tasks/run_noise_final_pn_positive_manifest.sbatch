#!/bin/bash
#SBATCH --job-name=noise_pn_manifest
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/noise_pn_manifest_%j.out
#SBATCH --error=data/validation/noise_pn_manifest_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

OUTPUT="data/validation/g8r_noise_final_pn_positive_manifest"
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite frozen P-arm manifest: $OUTPUT" >&2; exit 4;
}

echo "=== final noise P/N positive manifest started: $(date) ==="
python -u tasks/test_noise_final_pn_shared_encoder.py
python -u tasks/build_noise_final_pn_positive_manifest.py
echo "=== final noise P/N positive manifest completed: $(date) ==="
