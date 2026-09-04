#!/bin/bash
#SBATCH --job-name=noise_e15_audit
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=data/validation/noise_e15_audit_%j.out
#SBATCH --error=data/validation/noise_e15_audit_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUTPUT="data/validation/g8r_noise_final_e15_preflight.json"
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite E15 preflight audit: $OUTPUT" >&2
  exit 3
}

echo "=== E15 implementation-contract audit started: $(date) ==="
python -u tasks/test_noise_final_e15_preflight.py
python -u tasks/test_noise_final_e4a_direct_augmentation.py
python -u tasks/test_noise_final_e14_crossfit_teacher.py
python -u tasks/audit_noise_final_e15_preflight.py --output "$OUTPUT"
echo "=== E15 implementation-contract audit completed: $(date) ==="
