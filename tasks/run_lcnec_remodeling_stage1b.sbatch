#!/bin/bash
# Stage-1b: Re-run decomposition with robust threshold graph (no k parameter).
# Reuses ALL Stage-1 artifacts (embeddings, EIC matrix) from run_2352292.
# Only the graph definition changes: mutual kNN → deterministic cosine threshold.
#SBATCH --job-name=lcnec_s1b
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/lcnec_s1b_%j.out
#SBATCH --error=data/validation/lcnec_s1b_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

STAGE1_RUN="data/validation/lcnec_global_remodeling_stage1/run_2352292"
OUT="data/validation/lcnec_global_remodeling_stage1b/run_${SLURM_JOB_ID}"

[[ -d "$STAGE1_RUN/analysis" ]] || { echo "Stage-1 artifacts not found: $STAGE1_RUN" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "Collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

python -u tasks/analyze_lcnec_remodeling_stage1b.py \
  --stage1-dir "$STAGE1_RUN/analysis" \
  --output-dir "$OUT" \
  --target-coverage 0.80 \
  --theta-sensitivity 0.01 \
  --sign-flips 10000

echo "STAGE1B_COMPLETE: $OUT"
