#!/bin/bash
# Direct one-factor successors of selected Full23 v1.3.1 in one cached job.
# Submit with: sbatch tasks/run_full23_v131_direct_repairs_1gpu.sbatch
#SBATCH --job-name=full23_v131_direct
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/full23_v131_direct_%j.out
#SBATCH --error=data/validation/full23_v131_direct_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

ENVEDA="data/validation/enveda180_cross_condition_benchmark_v3_20261012"
ENVEDA_LAYER="data/validation/enveda180_layered_scoreblind_run_2372550"
SOURCE_RUN="data/validation/enveda_unified_candidate_integrator_run_2411772"
REUSE_RUN="data/validation/enveda_unified_candidate_integrator_run_2411378"
FULL23="data/validation/enveda_no_query_context_diagnostic_run_2413878/modeling/unified_candidate_integrator.pt"
FULL23_OOF="data/validation/enveda_no_query_context_diagnostic_run_2413878/modeling/candidate_predictions_oof.npz"
V131_RUN="data/validation/full23_v13_cumulative_repairs_run_2434084"
V131="$V131_RUN/gnps_development_evaluation/selected_model.pt"

GNPS="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
GNPS_ID="$GNPS/panel_identity_disjoint.npz"
if [[ ! -s "$GNPS_ID" ]]; then
  GNPS_ID="data/validation/GLM_gnps_identity_panel_reconstruction/panel_identity_disjoint.npz"
fi
GNPS_BASE="data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz"
GNPS_MSG="data/validation/noise_msg_fusion_run_2349961/gnps_fusion_scores"
GNPS_CHEM="data/validation/wse_conditional_residual_run_2376648/continuous_chem_evidence"
OUT="${FULL23_V131_DIRECT_OUT:-data/validation/full23_v131_direct_repairs_run_${SLURM_JOB_ID}}"

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
  "$ENVEDA_LAYER/method_scores_unopened.npz" "$ENVEDA_NOISE" "$ENVEDA_STAGE1" "$ENVEDA_PHASEA" "$ENVEDA_FULLMSG"
  "$ENVEDA_MSG/report.json" "$ENVEDA_CHEM/report.json" "$FULL23" "$FULL23_OOF" "$V131"
  "$GNPS/report.json" "$GNPS/manifest.csv.gz" "$GNPS_ID" "$GNPS/panel_formula_disjoint.npz"
  "$GNPS_BASE" "$GNPS_MSG/report.json" "$GNPS_CHEM/report.json"
  "$GNPS_NOISE" "$GNPS_STAGE1" "$GNPS_PHASEA" "$GNPS_FULLMSG"
  tasks/full23_tensor_set_readout_core.py
  tasks/train_full23_transfer_safe_tensor_readout.py
  tasks/train_full23_v13_cumulative_repairs.py
  tasks/train_full23_v131_direct_repairs.py
  tasks/evaluate_full23_v131_direct_repairs.py
  tasks/test_full23_tensor_set_readout.py
  tasks/test_full23_v131_direct_repairs.py
)
missing=()
for path in "${required[@]}"; do
  [[ -s "$path" ]] || missing+=("$path")
done
if (( ${#missing[@]} > 0 )); then
  printf 'FULL23_V131_DIRECT_MISSING: %s\n' "${missing[@]}" >&2
  exit 4
fi

python -m py_compile \
  tasks/full23_tensor_set_readout_core.py \
  tasks/train_full23_v131_direct_repairs.py \
  tasks/evaluate_full23_v131_direct_repairs.py
python -u tasks/test_full23_tensor_set_readout.py
python -u tasks/test_full23_v131_direct_repairs.py

mkdir -p "$OUT"
if [[ ! -s "$OUT/modeling/report.json" ]]; then
  python -u tasks/train_full23_v131_direct_repairs.py \
    --benchmark "$ENVEDA" --score-bundle "$ENVEDA_LAYER/method_scores_unopened.npz" \
    --msg-cache "$ENVEDA_MSG" --noise-embeddings "$ENVEDA_NOISE" \
    --stage1-embeddings "$ENVEDA_STAGE1" --phasea-embeddings "$ENVEDA_PHASEA" \
    --fullmsg-embeddings "$ENVEDA_FULLMSG" --chem-dir "$ENVEDA_CHEM" \
    --full23-checkpoint "$FULL23" --full23-oof "$FULL23_OOF" \
    --output "$OUT/modeling" --device cuda
fi

VIEW="$OUT/gnps_benchmark_view"
mkdir -p "$VIEW"
ln -sfn "$(readlink -f "$GNPS/report.json")" "$VIEW/report.json"
ln -sfn "$(readlink -f "$GNPS/manifest.csv.gz")" "$VIEW/manifest.csv.gz"
ln -sfn "$(readlink -f "$GNPS_ID")" "$VIEW/panel_identity_disjoint.npz"
ln -sfn "$(readlink -f "$GNPS/panel_formula_disjoint.npz")" "$VIEW/panel_formula_disjoint.npz"

if [[ ! -s "$OUT/gnps_development_evaluation/report.json" ]]; then
  python -u tasks/evaluate_full23_v131_direct_repairs.py \
    --benchmark "$VIEW" --score-bundle "$GNPS_BASE" --msg-cache "$GNPS_MSG" \
    --noise-embeddings "$GNPS_NOISE" --stage1-embeddings "$GNPS_STAGE1" \
    --phasea-embeddings "$GNPS_PHASEA" --fullmsg-embeddings "$GNPS_FULLMSG" \
    --chem-dir "$GNPS_CHEM" --full23-checkpoint "$FULL23" \
    --v131-checkpoint "$V131" --candidate-dir "$OUT/modeling" \
    --output "$OUT/gnps_development_evaluation" --device cuda
fi

echo "FULL23_V131_DIRECT_REPAIRS_COMPLETE: $OUT"
