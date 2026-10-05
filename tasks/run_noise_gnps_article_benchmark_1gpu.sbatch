#!/bin/bash
#SBATCH --job-name=noise_gnps_paper
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/noise_gnps_article_%j.out
#SBATCH --error=data/validation/noise_gnps_article_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_DEPS="$LOCAL_BASE/noise_gnps_article_deps_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_DEPS"
# The pinned wheel is cp311; fail in seconds with a clear reason if the
# activated interpreter is anything else.
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"pinned ms_entropy wheel requires Python 3.11, got {sys.version.split()[0]}"
    )
PY
python -m pip install --no-deps --target "$LOCAL_DEPS" \
  third_party/noise_article_wheels/ms_entropy-1.5.2-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
export PYTHONPATH="$LOCAL_DEPS:$PYTHONPATH"

BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
BASELINE_RUN="${BASELINE_RUN:-data/validation/gnps_absolute_baselines_run_2347472}"
OFFICIAL="$BASELINE_RUN/gnps/official_embeddings.npz"
NOISE_V1="$BASELINE_RUN/gnps/v1_embeddings.npz"
RUN_ROOT="data/validation/noise_gnps_article_benchmark_run_${SLURM_JOB_ID}"
BUNDLE="$RUN_ROOT/method_scores.npz"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$RUN_ROOT"

python -u tasks/test_gnps_pair_score_benchmark.py
python -u tasks/test_noise_gnps_article_benchmark.py
python -u tasks/test_noise_gnps_article_benchmark_sbatch.py

python -u tasks/build_noise_gnps_article_score_bundle.py \
  --benchmark "$BENCHMARK" \
  --official-embeddings "$OFFICIAL" \
  --noise-v1-embeddings "$NOISE_V1" \
  --output "$BUNDLE" \
  --workers 16 \
  --fragment-tolerance 0.02 \
  --n-highest-peaks 100

python -u tasks/evaluate_noise_gnps_article_benchmark.py \
  --benchmark "$BENCHMARK" \
  --score-bundle "$BUNDLE" \
  --output "$RUN_ROOT/evaluation" \
  --baseline-method official_dreams \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 20261003

python -u tasks/plot_noise_gnps_article_benchmark.py \
  --evaluation "$RUN_ROOT/evaluation" \
  --output "$RUN_ROOT/figures"

echo "=== Noise/P2b GNPS article benchmark: $RUN_ROOT ==="
