#!/bin/bash
#SBATCH --job-name=noise_final_e2_m1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/noise_final_e2_m1_%j.out
#SBATCH --error=data/validation/noise_final_e2_m1_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

OUTPUT="data/validation/g8r_noise_final_e2_corrective_scan"
if [ -e "$OUTPUT" ]; then
  echo "Refusing to overwrite $OUTPUT" >&2
  exit 2
fi

echo "=== final noise E2 corrective scan started: $(date) ==="
python -u tasks/test_noise_final_e2_corrective_scan.py
python -u tasks/run_noise_final_e2_corrective_scan.py --device cuda --batch-size 128 --output-dir "$OUTPUT"
python -u tasks/analyze_noise_final_e2_corrective_scan.py --scan-dir "$OUTPUT" --bootstrap-resamples 5000
python -u tasks/validate_noise_final_e2_corrective_scan.py --output-dir "$OUTPUT"
echo "=== final noise E2 corrective scan completed: $(date) ==="
