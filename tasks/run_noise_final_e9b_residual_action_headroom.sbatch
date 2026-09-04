#!/bin/bash
#SBATCH --job-name=noise_e9b_cap
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=data/validation/noise_e9b_cap_%j.out
#SBATCH --error=data/validation/noise_e9b_cap_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUTPUT="data/validation/g8r_noise_final_e9b_residual_action_headroom_v2"
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite E9-B output: $OUTPUT" >&2; exit 4; }
echo "=== E9-B residual mature-action headroom started: $(date) ==="
python -u tasks/test_noise_final_e9b_residual_action_headroom.py
python -u tasks/audit_noise_final_e9b_residual_action_headroom.py --output-dir "$OUTPUT"
echo "=== E9-B residual mature-action headroom completed: $(date) ==="
