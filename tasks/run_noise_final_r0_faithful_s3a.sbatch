#!/bin/bash
#SBATCH --job-name=noise_r0_faithful
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/noise_final_r0_faithful_%j.out
#SBATCH --error=data/validation/noise_final_r0_faithful_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

OUTPUT="data/validation/g8r_noise_final_r0_faithful_s3a"
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite $OUTPUT" >&2; exit 4; }

echo "=== noise final R0 faithful S3A manifest started: $(date) ==="
python -u tasks/test_noise_final_r0_faithful_s3a.py
python -u tasks/build_noise_final_r0_faithful_s3a_manifest.py --output-dir "$OUTPUT"
python -u tasks/validate_noise_final_r0_faithful_s3a.py --output-dir "$OUTPUT"
echo "=== noise final R0 faithful S3A manifest completed: $(date) ==="
