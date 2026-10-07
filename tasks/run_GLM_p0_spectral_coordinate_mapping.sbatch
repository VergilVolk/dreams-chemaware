#!/bin/bash
# P0-L is CPU analysis of frozen scores. The cluster submission policy for
# this project requires an explicit GPU request, so one GPU is requested but
# no new embedding or model training is performed.
#SBATCH --job-name=glm_p0_coord
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/glm_p0_coord_%j.out
#SBATCH --error=data/validation/glm_p0_coord_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BENCHMARK="${GNPS_BENCHMARK:-data/validation/gnps_gold_silver_10ppm_benchmark_v1}"
SCORE_BUNDLE="${P0_SCORE_BUNDLE:-data/validation/GLM_gnps_article_benchmark_s2v26/run15/bundle/method_scores.npz}"
IDENTITY_PANEL="${P0_IDENTITY_PANEL:-$BENCHMARK/panel_identity_disjoint.npz}"
if [[ ! -f "$IDENTITY_PANEL" ]]; then
  IDENTITY_PANEL="data/validation/GLM_gnps_identity_panel_reconstruction/panel_identity_disjoint.npz"
fi
OUT="data/validation/GLM_p0_spectral_coordinate_mapping/run_${SLURM_JOB_ID}"

required=(
  "$BENCHMARK/manifest.csv.gz"
  "$BENCHMARK/panel_formula_disjoint.npz"
  "$IDENTITY_PANEL"
  "$SCORE_BUNDLE"
  "tasks/GLM_p0_spectral_coordinate_mapping.py"
  "docs/SPECTRAL_COORDINATE_METABOLOMICS_P0_CONTRACT_20261007.md"
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "P0-L preflight missing: $path" >&2; exit 2; }
done
[[ ! -e "$OUT" ]] || { echo "P0-L output collision: $OUT" >&2; exit 3; }

python -m py_compile tasks/GLM_p0_spectral_coordinate_mapping.py
python -u tasks/GLM_p0_spectral_coordinate_mapping.py \
  --benchmark "$BENCHMARK" \
  --identity-panel "$IDENTITY_PANEL" \
  --score-bundle "$SCORE_BUNDLE" \
  --primary-method noise_v1 \
  --baseline-method official_dreams \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 20261007 \
  --output "$OUT"

find "$OUT" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SHA256SUMS"
echo "COMPLETE: P0-L spectral-coordinate identifiability -> $OUT"
