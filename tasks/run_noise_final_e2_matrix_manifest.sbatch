#!/bin/bash
#SBATCH --job-name=noise_final_e2_m0
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=data/validation/noise_final_e2_m0_%j.out
#SBATCH --error=data/validation/noise_final_e2_m0_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

OUTPUT="data/validation/g8r_noise_final_e2_matrix_manifest"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite frozen E2 manifest: $OUTPUT" >&2
  exit 1
fi

echo "=== final noise E2 matrix manifest started: $(date) ==="
python -u tasks/build_noise_final_e2_matrix_manifest.py --output-dir "$OUTPUT"
python -u tasks/validate_noise_final_e2_matrix_manifest.py --output-dir "$OUTPUT"
echo "=== final noise E2 matrix manifest completed: $(date) ==="
