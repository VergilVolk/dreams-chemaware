#!/bin/bash
#SBATCH --job-name=noise_final_e4m0
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=data/validation/noise_final_e4m0_%j.out
#SBATCH --error=data/validation/noise_final_e4m0_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

OUTPUT="data/validation/g8r_noise_final_e4_target_cache"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite frozen E4-M0 output: $OUTPUT" >&2
  exit 1
fi

echo "=== final noise E4-M0 target cache started: $(date) ==="
python -u tasks/test_noise_final_e4_target_cache.py
python -u tasks/build_noise_final_e4_target_cache.py --device cuda --batch-size 128 --output-dir "$OUTPUT"
python -u tasks/validate_noise_final_e4_target_cache.py --output-dir "$OUTPUT"
echo "=== final noise E4-M0 target cache completed: $(date) ==="
