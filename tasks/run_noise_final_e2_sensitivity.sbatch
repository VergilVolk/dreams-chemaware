#!/bin/bash
#SBATCH --job-name=noise_final_e2_m1b
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/noise_final_e2_m1b_%j.out
#SBATCH --error=data/validation/noise_final_e2_m1b_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

OUTPUT="data/validation/g8r_noise_final_e2_sensitivity"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite frozen E2 sensitivity output: $OUTPUT" >&2
  exit 1
fi

echo "=== final noise E2 sensitivity audit started: $(date) ==="
python -u tasks/test_noise_final_e2_sensitivity.py
python -u tasks/audit_noise_final_e2_sensitivity.py --output-dir "$OUTPUT"
python -u tasks/validate_noise_final_e2_sensitivity.py --output-dir "$OUTPUT"
echo "=== final noise E2 sensitivity audit completed: $(date) ==="
