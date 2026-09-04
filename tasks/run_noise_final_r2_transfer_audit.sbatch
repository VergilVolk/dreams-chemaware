#!/bin/bash
#SBATCH --job-name=noise_r2_audit
#SBATCH --partition=gpu
#SBATCH --array=1-2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/noise_r2_audit_%A_%a.out
#SBATCH --error=data/validation/noise_r2_audit_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

BACKBONE_LR=(5e-7 1e-6 2e-6)
HEAD_LR=(2.5e-6 5e-6 1e-5)
I="${SLURM_ARRAY_TASK_ID}"
BLR="${BACKBONE_LR[$I]}"
HLR="${HEAD_LR[$I]}"
TAG="blr_$(printf '%.0e' "$BLR")_hlr_$(printf '%.0e' "$HLR")"
BASE="data/validation/g8r_noise_final_r2_shared_encoder/$TAG/seed_20260827/fold_0"
OUTPUT="data/validation/g8r_noise_final_r2_transfer_audit/$TAG/seed_20260827/fold_0"

[[ -f "$BASE/final_shared_encoder.pt" ]] || { echo "Missing R2 checkpoint: $BASE" >&2; exit 3; }
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite audit: $OUTPUT" >&2; exit 4; }

echo "=== R2 transfer audit $TAG started: $(date) ==="
python -u tasks/audit_noise_final_r2_transfer.py \
  --checkpoint "$BASE/final_shared_encoder.pt" \
  --output-dir "$OUTPUT" --outer-fold 0 --device cuda
python -u tasks/validate_noise_final_r2_transfer_audit.py --output-dir "$OUTPUT"
echo "=== R2 transfer audit $TAG completed: $(date) ==="
