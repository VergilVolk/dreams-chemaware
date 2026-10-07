#!/bin/bash
#SBATCH --job-name=lcnec_s1d
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/lcnec_s1d_%j.out
#SBATCH --error=data/validation/lcnec_s1d_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

STAGE1_RUN="data/validation/lcnec_global_remodeling_stage1/run_2352292"
OUT="data/validation/lcnec_global_remodeling_stage1d/run_${SLURM_JOB_ID}"
mkdir -p "$OUT"

python -u tasks/analyze_lcnec_remodeling_stage1d.py \
  --stage1-dir "$STAGE1_RUN/analysis" \
  --output-dir "$OUT" \
  --k 3 \
  --bootstrap-draws 1000 \
  --sign-flips 10000 \
  --overwrite

echo "STAGE1D_COMPLETE: $OUT"
