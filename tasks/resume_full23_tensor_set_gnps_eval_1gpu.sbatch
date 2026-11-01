#!/bin/bash
# Resume only the GNPS evaluation after job 2433126 finished model training.
# Usage: sbatch tasks/resume_full23_tensor_set_gnps_eval_1gpu.sbatch
# Override only if needed: SOURCE_JOB_ID=... sbatch ...
#SBATCH --job-name=full23_eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/full23_tensor_eval_%j.out
#SBATCH --error=data/validation/full23_tensor_eval_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

SOURCE_JOB_ID="${SOURCE_JOB_ID:-2433126}"
OUT="data/validation/full23_tensor_set_readout_run_${SOURCE_JOB_ID}"
GNPS="${GNPS_BENCHMARK:-data/validation/gnps_gold_silver_10ppm_benchmark_v1}"
GNPS_IDENTITY="${GNPS_IDENTITY_PANEL:-$GNPS/panel_identity_disjoint.npz}"
if [[ ! -s "$GNPS_IDENTITY" ]]; then
  GNPS_IDENTITY="data/validation/GLM_gnps_identity_panel_reconstruction/panel_identity_disjoint.npz"
fi
SOURCE_RUN="${ENVEDA_UNIFIED_SOURCE_RUN:-data/validation/enveda_unified_candidate_integrator_run_2411772}"
FULL23="${FULL23_V1_CHECKPOINT:-data/validation/enveda_no_query_context_diagnostic_run_2413878/modeling/unified_candidate_integrator.pt}"
GNPS_BASE="${GNPS_SCORE_BUNDLE:-data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz}"
GNPS_MSG="${NOISE_MSG_GNPS_CACHE:-data/validation/noise_msg_fusion_run_2349961/gnps_fusion_scores}"
GNPS_CHEM="${GNPS_CONTINUOUS_CHEM_DIR:-data/validation/wse_conditional_residual_run_2376648/continuous_chem_evidence}"
GNPS_NOISE="data/validation/gnps_absolute_baselines_run_2347472/gnps/v1_embeddings.npz"
GNPS_STAGE1="$SOURCE_RUN/gnps_inputs/chemaware_stage1_embeddings.npz"
GNPS_PHASEA="$SOURCE_RUN/gnps_inputs/chemaware_phasea_embeddings.npz"
GNPS_FULLMSG="$SOURCE_RUN/gnps_inputs/noise_chemaware_fullmsg_embeddings.npz"
MODEL="$OUT/modeling/full23_tensor_set_readout.pt"
EVAL="$OUT/gnps_development_evaluation"
GNPS_VIEW="$OUT/gnps_benchmark_view"

required=(
  "$MODEL" "$FULL23" "$GNPS/report.json" "$GNPS/manifest.csv.gz"
  "$GNPS_IDENTITY" "$GNPS/panel_formula_disjoint.npz" "$GNPS_BASE"
  "$GNPS_MSG/report.json" "$GNPS_MSG/pair_scores_identity_disjoint.npy" "$GNPS_MSG/pair_scores_formula_disjoint.npy"
  "$GNPS_CHEM/report.json" "$GNPS_CHEM/continuous_chem_identity_disjoint.npz" "$GNPS_CHEM/continuous_chem_formula_disjoint.npz"
  "$GNPS_NOISE" "$GNPS_STAGE1" "$GNPS_PHASEA" "$GNPS_FULLMSG"
)
missing=()
for path in "${required[@]}"; do [[ -s "$path" ]] || missing+=("$path"); done
if (( ${#missing[@]} > 0 )); then
  printf 'FULL23_TENSOR_EVAL_MISSING: %s\n' "${missing[@]}" >&2
  exit 4
fi
[[ ! -e "$EVAL" ]] || { echo "FULL23_TENSOR_EVAL_OUTPUT_EXISTS: $EVAL" >&2; exit 5; }

mkdir -p "$GNPS_VIEW"
ln -sfn "$(readlink -f "$GNPS/report.json")" "$GNPS_VIEW/report.json"
ln -sfn "$(readlink -f "$GNPS/manifest.csv.gz")" "$GNPS_VIEW/manifest.csv.gz"
ln -sfn "$(readlink -f "$GNPS_IDENTITY")" "$GNPS_VIEW/panel_identity_disjoint.npz"
ln -sfn "$(readlink -f "$GNPS/panel_formula_disjoint.npz")" "$GNPS_VIEW/panel_formula_disjoint.npz"

python -u tasks/evaluate_full23_tensor_set_on_gnps.py \
  --benchmark "$GNPS_VIEW" --score-bundle "$GNPS_BASE" --msg-cache "$GNPS_MSG" \
  --noise-embeddings "$GNPS_NOISE" --stage1-embeddings "$GNPS_STAGE1" \
  --phasea-embeddings "$GNPS_PHASEA" --fullmsg-embeddings "$GNPS_FULLMSG" \
  --chem-dir "$GNPS_CHEM" --full23-checkpoint "$FULL23" \
  --readout-checkpoint "$MODEL" --output "$EVAL" --device cuda

sha256sum "$MODEL" "$OUT/modeling/report.json" "$EVAL/report.json" \
  "$EVAL/candidate_predictions_identity_disjoint.npz" \
  "$EVAL/candidate_predictions_formula_disjoint.npz" > "$OUT/SHA256SUMS.txt"
echo "FULL23_TENSOR_SET_GNPS_EVALUATION_RESUMED: $OUT"
