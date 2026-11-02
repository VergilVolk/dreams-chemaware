#!/bin/bash
# Post-score identity increment analysis for the completed Atlas v10 run.
# No model inference, no manual memory request, and no scoring-time truth access.
#SBATCH --job-name=atlas_v10_increment
# The E6 site submit policy requires an explicit GPU request even though this
# post-scoring analysis performs no GPU inference.
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/atlas_v10_increment_%j.out
#SBATCH --error=data/validation/atlas_v10_increment_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

RUN_DIR="${ATLAS_V10_RUN_DIR:-data/validation/contrast_atlas_v10_complete_axis_full23_run_2433291}"
OUT="data/validation/contrast_atlas_v10_identity_increment_run_${SLURM_JOB_ID}"

python -u tasks/analyze_contrast_atlas_v10_identity_increment.py \
  --run-dir "$RUN_DIR" \
  --out-dir "$OUT"

echo "ATLAS_V10_IDENTITY_INCREMENT_COMPLETE: $OUT"
