#!/bin/bash
#SBATCH --job-name=lcnec_s1c
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/lcnec_s1c_%j.out
#SBATCH --error=data/validation/lcnec_s1c_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

STAGE1_RUN="data/validation/lcnec_global_remodeling_stage1/run_2352292"
OUT="data/validation/lcnec_global_remodeling_stage1c/run_${SLURM_JOB_ID}"

[[ -d "$STAGE1_RUN/analysis" ]] || { echo "Stage-1 artifacts not found" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "Collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

# Install leidenalg if not present (CPU-only, fast)
pip install -q leidenalg igraph 2>/dev/null || true

python -u tasks/analyze_lcnec_remodeling_stage1c.py \
  --stage1-dir "$STAGE1_RUN/analysis" \
  --output-dir "$OUT" \
  --knn-k 10 \
  --resolution 1.0 \
  --resolution-sensitivity 0.5 \
  --sign-flips 10000 \
  --overwrite

echo "STAGE1C_COMPLETE: $OUT"
