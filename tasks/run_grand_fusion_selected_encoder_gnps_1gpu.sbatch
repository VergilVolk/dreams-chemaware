#!/bin/bash
#SBATCH --job-name=grand_encoder_gnps
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --time=18:00:00
#SBATCH --output=data/validation/grand_encoder_gnps_%j.out
#SBATCH --error=data/validation/grand_encoder_gnps_%j.err

# One-time external confirmation for the single task-vector checkpoint selected
# from the bounded seven-arm MassSpecGym development scan.
set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

: "${SELECTED_CHECKPOINT:?Set SELECTED_CHECKPOINT to exactly one preregistered task-vector arm}"
BENCHMARK="${GNPS_BENCHMARK:-data/validation/gnps_gold_silver_10ppm_benchmark_v1}"
BASE_BUNDLE="${SCORE_BUNDLE:-data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz}"
ARCHITECTURE="${ARCHITECTURE:-dreams/models/pretrained/ssl_model_server.pt}"
RUN_ROOT="data/validation/grand_fusion_selected_encoder_gnps_${SLURM_JOB_ID}"
[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$RUN_ROOT"

required=(
  "$SELECTED_CHECKPOINT" "$BASE_BUNDLE" "$ARCHITECTURE"
  "$BENCHMARK/spectra.mgf" "$BENCHMARK/manifest.csv.gz"
  "$BENCHMARK/panel_identity_disjoint.npz" "$BENCHMARK/panel_formula_disjoint.npz"
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/extend_gnps_article_score_bundle.py
  tasks/extract_pair_score_cache_from_bundle.py
  tasks/evaluate_gnps_gold_silver_10ppm_pair_scores.py
  tasks/adjudicate_grand_fusion_external.py
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "Missing selected-encoder input: $path" >&2; exit 5; }
done

python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$BENCHMARK" --checkpoint "$SELECTED_CHECKPOINT" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --output "$RUN_ROOT/grand_encoder_embeddings.npz" \
  --batch-size 64 --label grand_encoder_selected_once

python -u tasks/extend_gnps_article_score_bundle.py \
  --benchmark "$BENCHMARK" --base-bundle "$BASE_BUNDLE" \
  --embedding-method "grand_encoder=$RUN_ROOT/grand_encoder_embeddings.npz" \
  --output "$RUN_ROOT/method_scores_with_grand_encoder.npz"

for method in grand_encoder weighted_spectral_entropy p2b_noise_v1_frozen noise_v1; do
  python -u tasks/extract_pair_score_cache_from_bundle.py \
    --benchmark "$BENCHMARK" \
    --score-bundle "$RUN_ROOT/method_scores_with_grand_encoder.npz" \
    --method "$method" --output "$RUN_ROOT/${method}_cache"
done

for baseline in weighted_spectral_entropy p2b_noise_v1_frozen noise_v1; do
  python -u tasks/evaluate_gnps_gold_silver_10ppm_pair_scores.py \
    --benchmark "$BENCHMARK" \
    --baseline-score-cache "$RUN_ROOT/${baseline}_cache" \
    --candidate-score-cache "$RUN_ROOT/grand_encoder_cache" \
    --output "$RUN_ROOT/grand_encoder_vs_${baseline}" \
    --bootstrap-resamples 10000 --bootstrap-seed 20261004
done

python -u tasks/adjudicate_grand_fusion_external.py \
  --mode encoder \
  --wse "$RUN_ROOT/grand_encoder_vs_weighted_spectral_entropy/report.json" \
  --p2b "$RUN_ROOT/grand_encoder_vs_p2b_noise_v1_frozen/report.json" \
  --noise "$RUN_ROOT/grand_encoder_vs_noise_v1/report.json" \
  --output "$RUN_ROOT/adjudication.json"

cat >"$RUN_ROOT/CLAIM_BOUNDARY.txt" <<'EOF'
This is the one-time external confirmation of one checkpoint selected without
GNPS outcomes. It must be reported against all three frozen baselines and may
not be used to select another task-vector coefficient.
EOF
echo "Selected encoder GNPS confirmation complete: $RUN_ROOT"
