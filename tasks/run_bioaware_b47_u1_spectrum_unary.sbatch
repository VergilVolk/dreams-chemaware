#!/bin/bash
#SBATCH --job-name=bio_b47_u1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_b47_u1_%j.out
#SBATCH --error=bioaware_b47_u1_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

GRAPH=data/validation/noise_corrected_candidate_graph_v1_20260906
FINAL=data/validation/bioaware_b47_u1_spectrum_unary_dev_20260921_v1
WORK="${FINAL}.partial.${SLURM_JOB_ID}"

echo "=== BioAware B47 U1 spectrum-only unary started: $(date) ==="
required=(
  "$GRAPH/candidate_graph.npz" "$GRAPH/report.json"
  tasks/bioaware_b47_u1_core.py
  tasks/develop_bioaware_b47_u1_spectrum_unary.py
  tasks/test_bioaware_b47_u1_spectrum_unary.py
  tasks/validate_bioaware_b47_u1_spectrum_unary.py
)
for path in "${required[@]}"; do
  test -s "$path" || { echo "Missing required dependency: $path" >&2; exit 2; }
done
test ! -e "$FINAL" || { echo "Refusing to overwrite frozen result: $FINAL" >&2; exit 3; }
test ! -e "$WORK" || { echo "Refusing stale partial: $WORK" >&2; exit 3; }
python -c "import numpy,pandas; print('[BioAware B47 U1 dependencies] PASS')"
python -m py_compile \
  tasks/bioaware_b47_u1_core.py \
  tasks/develop_bioaware_b47_u1_spectrum_unary.py \
  tasks/test_bioaware_b47_u1_spectrum_unary.py \
  tasks/validate_bioaware_b47_u1_spectrum_unary.py
python -u tasks/test_bioaware_b47_u1_spectrum_unary.py
python -u tasks/develop_bioaware_b47_u1_spectrum_unary.py \
  --graph-dir "$GRAPH" --output "$WORK" --preflight-only
test ! -e "$WORK" || { echo "U1 preflight unexpectedly wrote output" >&2; exit 4; }
python -u tasks/develop_bioaware_b47_u1_spectrum_unary.py \
  --graph-dir "$GRAPH" --output "$WORK" \
  --folds 5 --fold-seed 20260921 --bootstrap-resamples 5000
python -u tasks/validate_bioaware_b47_u1_spectrum_unary.py --input "$WORK"
mv "$WORK" "$FINAL"
echo "=== BioAware B47 U1 spectrum-only unary completed: $(date) ==="
echo "Result: $FINAL/report.json"
