#!/bin/bash
#SBATCH --job-name=noise_pn_uncovered
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/noise_pn_uncovered_%j.out
#SBATCH --error=data/validation/noise_pn_uncovered_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUTPUT="data/validation/g8r_noise_final_pn_uncovered_errors"
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite uncovered-error ledger: $OUTPUT" >&2; exit 4;
}
echo "=== P/N uncovered-error audit started: $(date) ==="
python -u tasks/audit_noise_final_pn_uncovered_errors.py
echo "=== P/N uncovered-error audit completed: $(date) ==="
