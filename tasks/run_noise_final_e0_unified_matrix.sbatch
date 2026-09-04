#!/bin/bash
#SBATCH --job-name=noise_final_e0
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/noise_final_e0_%j.out
#SBATCH --error=data/validation/noise_final_e0_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

OUTPUT="data/validation/g8r_noise_final_e0_unified_matrix"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite locked E0 output: $OUTPUT" >&2
  exit 1
fi

echo "=== final noise E0 unified evidence ledger started: $(date) ==="
python -u tasks/build_noise_final_e0_unified_matrix.py --output-dir "$OUTPUT"
python -u tasks/validate_noise_final_e0_unified_matrix.py --output-dir "$OUTPUT"
echo "=== final noise E0 unified evidence ledger completed: $(date) ==="
