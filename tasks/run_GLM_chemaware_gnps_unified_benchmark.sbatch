#!/bin/bash
# ChemAware entry into the unified GNPS article benchmark: build the frozen
# classical+official score bundle once, then append the sealed ChemAware
# encoders (stage-1, Phase-A) through the SHA-locked extension layer and
# evaluate every method on the identical sealed panels.  No checkpoint
# selection, no tuning, no truth contact anywhere before evaluation.
#SBATCH --job-name=chem_gnps_unified
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_gnps_unified_%j.out
#SBATCH --error=data/validation/chem_gnps_unified_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
BASELINE_RUN="data/validation/gnps_absolute_baselines_run_2347472"
OFFICIAL_EMB="$BASELINE_RUN/gnps/official_embeddings.npz"
NOISE_V1_EMB="$BASELINE_RUN/gnps/v1_embeddings.npz"
ARCH="dreams/models/pretrained/ssl_model_server.pt"
STAGE1="data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt"
PHASEA="data/validation/chemaware_online_role_budget_native/run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt"
MS_ENTROPY_WHEEL="third_party/noise_article_wheels/ms_entropy-1.5.2-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl"
OUT="data/validation/GLM_chemaware_gnps_unified_benchmark/run_${SLURM_JOB_ID}"
SCRATCH_BASE="${SLURM_TMPDIR:-/tmp}"
SCRATCH="$SCRATCH_BASE/chem_gnps_unified_${SLURM_JOB_ID}"
LOCAL_DEPS="$SCRATCH_BASE/chem_gnps_unified_deps_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT" "$SCRATCH" "$LOCAL_DEPS"
cleanup() { rm -rf -- "$SCRATCH" "$LOCAL_DEPS"; }
trap cleanup EXIT

required=(
  "$BENCHMARK/spectra.mgf"
  "$BENCHMARK/manifest.csv.gz"
  "$BENCHMARK/panel_identity_disjoint.npz"
  "$BENCHMARK/panel_formula_disjoint.npz"
  "$OFFICIAL_EMB" "$NOISE_V1_EMB" "$ARCH" "$STAGE1" "$PHASEA"
  "$MS_ENTROPY_WHEEL"
  tasks/noise_gnps_article_spectral_scores.py
  tasks/build_noise_gnps_article_score_bundle.py
  tasks/evaluate_noise_gnps_article_benchmark.py
  tasks/extend_gnps_article_score_bundle.py
  tasks/prepare_official_embedding_checkpoint.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "GLM_GNPS_UNIFIED_MISSING_INPUT: $path" >&2; exit 2; }
done

python -m py_compile \
  tasks/noise_gnps_article_spectral_scores.py \
  tasks/build_noise_gnps_article_score_bundle.py \
  tasks/evaluate_noise_gnps_article_benchmark.py \
  tasks/extend_gnps_article_score_bundle.py

python -m pip install --no-deps --quiet --target "$LOCAL_DEPS" "$MS_ENTROPY_WHEEL"
export PYTHONPATH="$LOCAL_DEPS:$PYTHONPATH"

python -u tasks/build_noise_gnps_article_score_bundle.py \
  --benchmark "$BENCHMARK" \
  --official-embeddings "$OFFICIAL_EMB" \
  --noise-v1-embeddings "$NOISE_V1_EMB" \
  --output "$OUT/base_method_scores.npz" \
  --workers 16 \
  --fragment-tolerance 0.02 \
  --n-highest-peaks 100

python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$STAGE1" --output "$SCRATCH/chemaware_stage1_slim.pt"
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$PHASEA" --output "$SCRATCH/chemaware_phaseA_slim.pt"

srun --export=ALL --preserve-env python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$BENCHMARK" --checkpoint "$SCRATCH/chemaware_stage1_slim.pt" \
  --architecture-checkpoint "$ARCH" \
  --output "$SCRATCH/chemaware_stage1_embeddings.npz" \
  --batch-size 64 --label chemaware-stage1
srun --export=ALL --preserve-env python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$BENCHMARK" --checkpoint "$SCRATCH/chemaware_phaseA_slim.pt" \
  --architecture-checkpoint "$ARCH" \
  --output "$SCRATCH/chemaware_phaseA_embeddings.npz" \
  --batch-size 64 --label chemaware-phaseA

python -u tasks/extend_gnps_article_score_bundle.py \
  --benchmark "$BENCHMARK" \
  --base-bundle "$OUT/base_method_scores.npz" \
  --embedding-method "chemaware_stage1=$SCRATCH/chemaware_stage1_embeddings.npz" \
  --embedding-method "chemaware_phaseA=$SCRATCH/chemaware_phaseA_embeddings.npz" \
  --output "$OUT/extended_method_scores.npz"

python -u tasks/evaluate_noise_gnps_article_benchmark.py \
  --benchmark "$BENCHMARK" \
  --score-bundle "$OUT/extended_method_scores.npz" \
  --output "$OUT/evaluation" \
  --baseline-method official_dreams \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 20261003

cp "$SCRATCH"/chemaware_stage1_embeddings.json "$OUT/chemaware_stage1_encoding.json"
cp "$SCRATCH"/chemaware_phaseA_embeddings.json "$OUT/chemaware_phaseA_encoding.json"
sha256sum "$STAGE1" "$PHASEA" \
  "$OUT/base_method_scores.npz" \
  "$OUT/extended_method_scores.npz" \
  "$OUT"/evaluation/report.json > "$OUT/SHA256SUMS.txt"

echo "GLM_GNPS_UNIFIED_BENCHMARK_COMPLETE: $OUT"
