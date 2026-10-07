#!/bin/bash
#SBATCH --job-name=lcnec_p2v2
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/lcnec_p2v2_%j.out
#SBATCH --error=data/validation/lcnec_p2v2_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

STAGE1_RUN="data/validation/lcnec_global_remodeling_stage1/run_2352292"
OUT="data/validation/lcnec_p2v2/run_${SLURM_JOB_ID}"
mkdir -p "$OUT"

required=(
  "$STAGE1_RUN/query/priority_dark_modules.mgf"
  "$STAGE1_RUN/analysis/primary_decomposition.npz"
  "data/validation/gnps_gold_silver_10ppm_benchmark_v1/manifest.csv.gz"
  "data/validation/gnps_gold_silver_10ppm_benchmark_v1/spectra.mgf"
)
for p in "${required[@]}"; do
  [[ -s "$p" ]] || { echo "P2V2_MISSING: $p" >&2; exit 5; }
done

python -u tasks/GLM_p2v2_coordinate_projection.py \
  --query-mgf "$STAGE1_RUN/query/priority_dark_modules.mgf" \
  --effects-npz "$STAGE1_RUN/analysis/primary_decomposition.npz" \
  --output-dir "$OUT" \
  --ppm-gate 1000 \
  --top-k 5 \
  --overwrite

echo "P2V2_COMPLETE: $OUT"
