#!/bin/bash
# Full23 v1.2: delayed-compression tensor-set readout over frozen project encoders.
# Reuses every upstream cache from the completed v1.1 run.
#SBATCH --job-name=full23_tensor
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/full23_tensor_%j.out
#SBATCH --error=data/validation/full23_tensor_%j.err

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
FULL23="${FULL23_V1_CHECKPOINT:-data/validation/enveda_no_query_context_diagnostic_run_2413878/modeling/unified_candidate_integrator.pt}"
FULL23_OOF="${FULL23_V1_OOF:-data/validation/enveda_no_query_context_diagnostic_run_2413878/modeling/candidate_predictions_oof.npz}"
GNPS="${GNPS_BENCHMARK:-data/validation/gnps_gold_silver_10ppm_benchmark_v1}"
GNPS_IDENTITY="${GNPS_IDENTITY_PANEL:-$GNPS/panel_identity_disjoint.npz}"
if [[ ! -s "$GNPS_IDENTITY" ]]; then
  GNPS_IDENTITY="data/validation/GLM_gnps_identity_panel_reconstruction/panel_identity_disjoint.npz"
fi
GNPS_BASE="${GNPS_SCORE_BUNDLE:-data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz}"
GNPS_MSG="${NOISE_MSG_GNPS_CACHE:-data/validation/noise_msg_fusion_run_2349961/gnps_fusion_scores}"
GNPS_CHEM="${GNPS_CONTINUOUS_CHEM_DIR:-data/validation/wse_conditional_residual_run_2376648/continuous_chem_evidence}"
OUT="data/validation/full23_tensor_set_readout_run_${SLURM_JOB_ID}"

ENVEDA_NOISE="$ENVEDA_LAYER/noise_v1_embeddings.npz"
ENVEDA_STAGE1="$REUSE_RUN/inputs/enveda_chemaware_stage1_embeddings.npz"
ENVEDA_PHASEA="$REUSE_RUN/inputs/enveda_chemaware_phasea_embeddings.npz"
ENVEDA_FULLMSG="$REUSE_RUN/inputs/enveda_noise_chemaware_fullmsg_embeddings.npz"
ENVEDA_MSG="$REUSE_RUN/inputs/enveda_msg_cache"
ENVEDA_CHEM="$SOURCE_RUN/inputs/enveda_continuous_chem"
GNPS_NOISE="data/validation/gnps_absolute_baselines_run_2347472/gnps/v1_embeddings.npz"
GNPS_STAGE1="$SOURCE_RUN/gnps_inputs/chemaware_stage1_embeddings.npz"
GNPS_PHASEA="$SOURCE_RUN/gnps_inputs/chemaware_phasea_embeddings.npz"
GNPS_FULLMSG="$SOURCE_RUN/gnps_inputs/noise_chemaware_fullmsg_embeddings.npz"

required=(
  "$ENVEDA/manifest.csv.gz" "$ENVEDA/panel_identity_disjoint.npz" "$ENVEDA/panel_formula_disjoint.npz"
  "$ENVEDA_LAYER/method_scores_unopened.npz" "$ENVEDA_NOISE"
  "$ENVEDA_STAGE1" "$ENVEDA_PHASEA" "$ENVEDA_FULLMSG"
  "$ENVEDA_MSG/report.json" "$ENVEDA_MSG/pair_scores_identity_disjoint.npy" "$ENVEDA_MSG/pair_scores_formula_disjoint.npy"
  "$ENVEDA_CHEM/report.json" "$ENVEDA_CHEM/continuous_chem_identity_disjoint.npz" "$ENVEDA_CHEM/continuous_chem_formula_disjoint.npz"
  "$FULL23" "$FULL23_OOF" "$(dirname "$FULL23_OOF")/report.json"
  "$GNPS/manifest.csv.gz" "$GNPS_IDENTITY" "$GNPS/panel_formula_disjoint.npz"
  "$GNPS_BASE" "$GNPS_MSG/report.json" "$GNPS_MSG/pair_scores_identity_disjoint.npy" "$GNPS_MSG/pair_scores_formula_disjoint.npy"
  "$GNPS_NOISE" "$GNPS_STAGE1" "$GNPS_PHASEA" "$GNPS_FULLMSG"
  "$GNPS_CHEM/report.json" "$GNPS_CHEM/continuous_chem_identity_disjoint.npz" "$GNPS_CHEM/continuous_chem_formula_disjoint.npz"
  tasks/full23_multireference_readout_core.py tasks/train_full23_multireference_readout.py
  tasks/full23_tensor_set_readout_core.py tasks/train_full23_tensor_set_readout.py
  tasks/evaluate_full23_tensor_set_on_gnps.py tasks/test_full23_tensor_set_readout.py
)
missing=()
for path in "${required[@]}"; do [[ -s "$path" ]] || missing+=("$path"); done
if (( ${#missing[@]} > 0 )); then
  printf 'FULL23_TENSOR_SET_MISSING: %s\n' "${missing[@]}" >&2
  exit 4
fi
if [[ "${1:-}" == "--preflight" ]]; then
  echo "FULL23_TENSOR_SET_PREFLIGHT_PASS"
  exit 0
fi
[[ ! -e "$OUT" ]] || { echo "FULL23_TENSOR_SET_OUTPUT_EXISTS: $OUT" >&2; exit 5; }

python -m py_compile \
  tasks/full23_tensor_set_readout_core.py \
  tasks/train_full23_tensor_set_readout.py \
  tasks/evaluate_full23_tensor_set_on_gnps.py \
  tasks/test_full23_tensor_set_readout.py
python -u tasks/test_full23_tensor_set_readout.py

python -u tasks/train_full23_tensor_set_readout.py \
  --benchmark "$ENVEDA" --score-bundle "$ENVEDA_LAYER/method_scores_unopened.npz" \
  --msg-cache "$ENVEDA_MSG" --noise-embeddings "$ENVEDA_NOISE" \
  --stage1-embeddings "$ENVEDA_STAGE1" --phasea-embeddings "$ENVEDA_PHASEA" \
  --fullmsg-embeddings "$ENVEDA_FULLMSG" --chem-dir "$ENVEDA_CHEM" \
  --full23-checkpoint "$FULL23" --full23-oof "$FULL23_OOF" \
  --output "$OUT/modeling" --device cuda

# The historical identity panel may live in its reconstruction directory,
# while the evaluator expects both panels beside the shared manifest.  Build a
# read-only view without copying or changing any benchmark artifact.
GNPS_VIEW="$OUT/gnps_benchmark_view"
mkdir -p "$GNPS_VIEW"
ln -s "$(readlink -f "$GNPS/manifest.csv.gz")" "$GNPS_VIEW/manifest.csv.gz"
ln -s "$(readlink -f "$GNPS_IDENTITY")" "$GNPS_VIEW/panel_identity_disjoint.npz"
ln -s "$(readlink -f "$GNPS/panel_formula_disjoint.npz")" "$GNPS_VIEW/panel_formula_disjoint.npz"

python -u tasks/evaluate_full23_tensor_set_on_gnps.py \
  --benchmark "$GNPS_VIEW" --score-bundle "$GNPS_BASE" --msg-cache "$GNPS_MSG" \
  --noise-embeddings "$GNPS_NOISE" --stage1-embeddings "$GNPS_STAGE1" \
  --phasea-embeddings "$GNPS_PHASEA" --fullmsg-embeddings "$GNPS_FULLMSG" \
  --chem-dir "$GNPS_CHEM" --full23-checkpoint "$FULL23" \
  --readout-checkpoint "$OUT/modeling/full23_tensor_set_readout.pt" \
  --output "$OUT/gnps_development_evaluation" --device cuda

sha256sum \
  "$OUT/modeling/full23_tensor_set_readout.pt" "$OUT/modeling/report.json" \
  "$OUT/gnps_development_evaluation/report.json" \
  "$OUT/gnps_development_evaluation/candidate_predictions_identity_disjoint.npz" \
  "$OUT/gnps_development_evaluation/candidate_predictions_formula_disjoint.npz" \
  > "$OUT/SHA256SUMS.txt"
echo "FULL23_TENSOR_SET_READOUT_COMPLETE: $OUT"
