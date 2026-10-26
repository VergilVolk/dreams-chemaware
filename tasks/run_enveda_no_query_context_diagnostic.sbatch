#!/bin/bash
# One bounded mechanism diagnostic: remove only query-constant context columns
# from the nonlinear Enveda integrator. Reuses every upstream score/cache.
# No encoder, MSG, V2, candidate graph, or benchmark asset is regenerated.
#SBATCH --job-name=e180_no_qctx
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/e180_no_qctx_%j.out
#SBATCH --error=data/validation/e180_no_qctx_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

ENVEDA="${ENVEDA_BENCHMARK:-data/validation/enveda180_cross_condition_benchmark_v3_20261012}"
ENVEDA_LAYER="${ENVEDA_LAYER_RUN:-data/validation/enveda180_layered_scoreblind_run_2372550}"
SOURCE_RUN="${ENVEDA_UNIFIED_SOURCE_RUN:-data/validation/enveda_unified_candidate_integrator_run_2411772}"
REUSE_RUN="${ENVEDA_UNIFIED_REUSE_RUN:-data/validation/enveda_unified_candidate_integrator_run_2411378}"
GNPS="${GNPS_BENCHMARK:-data/validation/gnps_gold_silver_10ppm_benchmark_v1}"
GNPS_BASE="${GNPS_SCORE_BUNDLE:-data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz}"
GNPS_MSG="${NOISE_MSG_GNPS_CACHE:-data/validation/noise_msg_fusion_run_2349961/gnps_fusion_scores}"
GNPS_CHEM="${GNPS_CONTINUOUS_CHEM_DIR:-data/validation/wse_conditional_residual_run_2376648/continuous_chem_evidence}"
B_PREDICTIONS="${B_GNPS_OOF_PREDICTIONS:-data/validation/B_dual_encoder_multireranker_run_2400157/modeling_nested_oof/candidate_predictions.npz}"
OUT="data/validation/enveda_no_query_context_diagnostic_run_${SLURM_JOB_ID}"

ENVEDA_STAGE1="$REUSE_RUN/inputs/enveda_chemaware_stage1_embeddings.npz"
ENVEDA_PHASEA="$REUSE_RUN/inputs/enveda_chemaware_phasea_embeddings.npz"
ENVEDA_FULLMSG="$REUSE_RUN/inputs/enveda_noise_chemaware_fullmsg_embeddings.npz"
ENVEDA_MSG="$REUSE_RUN/inputs/enveda_msg_cache"
ENVEDA_CHEM="$SOURCE_RUN/inputs/enveda_continuous_chem"
GNPS_STAGE1="$SOURCE_RUN/gnps_inputs/chemaware_stage1_embeddings.npz"
GNPS_PHASEA="$SOURCE_RUN/gnps_inputs/chemaware_phasea_embeddings.npz"
GNPS_FULLMSG="$SOURCE_RUN/gnps_inputs/noise_chemaware_fullmsg_embeddings.npz"

required=(
  "$ENVEDA/manifest.csv.gz"
  "$ENVEDA/panel_identity_disjoint.npz" "$ENVEDA/panel_formula_disjoint.npz"
  "$ENVEDA_LAYER/method_scores_unopened.npz"
  "$ENVEDA_STAGE1" "$ENVEDA_PHASEA" "$ENVEDA_FULLMSG"
  "$ENVEDA_MSG/report.json" "$ENVEDA_MSG/pair_scores_identity_disjoint.npy"
  "$ENVEDA_MSG/pair_scores_formula_disjoint.npy"
  "$ENVEDA_CHEM/report.json" "$ENVEDA_CHEM/continuous_chem_identity_disjoint.npz"
  "$ENVEDA_CHEM/continuous_chem_formula_disjoint.npz"
  "$GNPS/manifest.csv.gz"
  "$GNPS/panel_identity_disjoint.npz" "$GNPS/panel_formula_disjoint.npz"
  "$GNPS_BASE" "$GNPS_MSG/report.json"
  "$GNPS_MSG/pair_scores_identity_disjoint.npy" "$GNPS_MSG/pair_scores_formula_disjoint.npy"
  "$GNPS_STAGE1" "$GNPS_PHASEA" "$GNPS_FULLMSG"
  "$GNPS_CHEM/report.json" "$GNPS_CHEM/continuous_chem_identity_disjoint.npz"
  "$GNPS_CHEM/continuous_chem_formula_disjoint.npz" "$B_PREDICTIONS"
  tasks/enveda_unified_candidate_integrator_core.py
  tasks/train_enveda_unified_candidate_integrator.py
  tasks/evaluate_enveda_integrator_on_gnps.py
  tasks/test_enveda_unified_candidate_integrator.py
)
missing=()
for path in "${required[@]}"; do
  [[ -s "$path" ]] || missing+=("$path")
done
if (( ${#missing[@]} > 0 )); then
  printf 'ENVEDA_NO_QUERY_CONTEXT_MISSING: %s\n' "${missing[@]}" >&2
  exit 4
fi
[[ ! -e "$OUT" ]] || { echo "ENVEDA_NO_QUERY_CONTEXT_OUTPUT_EXISTS: $OUT" >&2; exit 5; }
mkdir -p "$OUT"

python -m py_compile \
  tasks/enveda_unified_candidate_integrator_core.py \
  tasks/train_enveda_unified_candidate_integrator.py \
  tasks/evaluate_enveda_integrator_on_gnps.py
python -u tasks/test_enveda_unified_candidate_integrator.py

python -u tasks/train_enveda_unified_candidate_integrator.py \
  --benchmark "$ENVEDA" \
  --score-bundle "$ENVEDA_LAYER/method_scores_unopened.npz" \
  --msg-cache "$ENVEDA_MSG" \
  --stage1-embeddings "$ENVEDA_STAGE1" \
  --phasea-embeddings "$ENVEDA_PHASEA" \
  --fullmsg-embeddings "$ENVEDA_FULLMSG" \
  --chem-dir "$ENVEDA_CHEM" \
  --output "$OUT/modeling" --folds 5 --epochs 20 --batch-queries 256 \
  --hidden 16 --learning-rate 0.001 --weight-decay 0.0001 \
  --anchor-strength 0.01 --bootstrap-resamples 5000 --device cuda \
  --exclude-query-constant-context

python -u tasks/evaluate_enveda_integrator_on_gnps.py \
  --benchmark "$GNPS" --checkpoint "$OUT/modeling/unified_candidate_integrator.pt" \
  --score-bundle "$GNPS_BASE" --msg-cache "$GNPS_MSG" \
  --stage1-embeddings "$GNPS_STAGE1" \
  --phasea-embeddings "$GNPS_PHASEA" \
  --fullmsg-embeddings "$GNPS_FULLMSG" \
  --chem-dir "$GNPS_CHEM" \
  --comparator-score "B_gnps_nested_oof=$B_PREDICTIONS" \
  --output "$OUT/gnps_fixed_evaluation" --device cuda --bootstrap-resamples 5000

sha256sum \
  "$OUT/modeling/unified_candidate_integrator.pt" \
  "$OUT/modeling/candidate_predictions_oof.npz" \
  "$OUT/modeling/report.json" \
  "$OUT/gnps_fixed_evaluation/report.json" \
  > "$OUT/SHA256SUMS.txt"
echo "ENVEDA_NO_QUERY_CONSTANT_CONTEXT_DIAGNOSTIC_COMPLETE: $OUT"
