#!/bin/bash
# Fixed CPU-reproducible repair of the completed CF2 experiment.  The job
# keeps the released Full23 v1 parent, evidence, folds, seed and budget; only
# the delta optimizer changes from AdamW-rotated MGDA to direct Pareto SGD
# with same-batch two-source backtracking.  One job also evaluates the frozen
# outer-OOF predictions against Full23 v1 on the four consumed panels.
#SBATCH --job-name=full23_pareto
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --output=data/validation/full23_pareto_%j.out
#SBATCH --error=data/validation/full23_pareto_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

ENVEDA_LEDGER="${ENVEDA_FULL23_LEDGER:-data/validation/enveda_no_query_context_diagnostic_run_2413878/modeling/candidate_predictions_oof.npz}"
ENVEDA_CHEM="${ENVEDA_CONTINUOUS_CHEM_DIR:-data/validation/enveda_unified_candidate_integrator_run_2411772/inputs/enveda_continuous_chem}"
PARENT_CHECKPOINT="${FULL23_V1_CHECKPOINT:-deploy/unified_additive_full23_v1/assets/unified_candidate_integrator.pt}"
PARENT_REPORT="${FULL23_V1_REPORT:-deploy/unified_additive_full23_v1/evidence/selection_modeling_report.json}"
GNPS_SCORES="${GNPS_FIXED_SCORE_DIR:-data/validation/enveda_no_query_context_diagnostic_run_2413878/gnps_fixed_evaluation}"
GNPS_AXIS="${GNPS_NESTED_AXIS_LEDGER:-data/validation/B_dual_encoder_multireranker_run_2400157/modeling_nested_oof/candidate_predictions.npz}"
GNPS_CHEM="${GNPS_CONTINUOUS_CHEM_DIR:-data/validation/wse_conditional_residual_run_2376648/continuous_chem_evidence}"
GNPS_IDENTITY="${GNPS_IDENTITY_PANEL:-data/validation/GLM_gnps_identity_panel_reconstruction/panel_identity_disjoint.npz}"
GNPS_FORMULA="${GNPS_FORMULA_PANEL:-data/validation/gnps_gold_silver_10ppm_benchmark_v1/panel_formula_disjoint.npz}"
OUT="data/validation/full23_parent_qrel_cf2_pareto_sgd_run_${SLURM_JOB_ID}"

required_files=(
  "$ENVEDA_LEDGER"
  "$PARENT_CHECKPOINT" "$PARENT_REPORT"
  "$GNPS_AXIS" "$GNPS_IDENTITY" "$GNPS_FORMULA"
  "$GNPS_SCORES/candidate_scores_identity_disjoint.npz"
  "$GNPS_SCORES/candidate_scores_formula_disjoint.npz"
  "$ENVEDA_CHEM/report.json"
  "$GNPS_CHEM/report.json"
  tasks/train_full23_parent_query_relative_mgda_residual.py
  tasks/train_full23_parent_qrel_cf2_mgda_residual.py
  tasks/train_full23_parent_qrel_cf2_pareto_sgd.py
  tasks/evaluate_full23_multisource_additive_endpoints.py
)
missing=()
for path in "${required_files[@]}"; do
  [[ -s "$path" ]] || missing+=("$path")
done
if (( ${#missing[@]} > 0 )); then
  printf 'FULL23_PARETO_SGD_MISSING: %s\n' "${missing[@]}" >&2
  exit 4
fi
[[ ! -e "$OUT" ]] || {
  echo "FULL23_PARETO_SGD_OUTPUT_EXISTS: $OUT" >&2
  exit 5
}

python -m py_compile \
  tasks/train_full23_parent_query_relative_mgda_residual.py \
  tasks/train_full23_parent_qrel_cf2_mgda_residual.py \
  tasks/train_full23_parent_qrel_cf2_pareto_sgd.py \
  tasks/evaluate_full23_multisource_additive_endpoints.py
python -u tasks/test_full23_parent_query_relative_mgda_residual.py
python -u tasks/test_full23_parent_qrel_cf2_mgda_residual.py

python -u tasks/train_full23_parent_qrel_cf2_pareto_sgd.py \
  --enveda-ledger "$ENVEDA_LEDGER" \
  --enveda-chem-dir "$ENVEDA_CHEM" \
  --parent-checkpoint "$PARENT_CHECKPOINT" \
  --parent-report "$PARENT_REPORT" \
  --gnps-score-dir "$GNPS_SCORES" \
  --gnps-axis-ledger "$GNPS_AXIS" \
  --gnps-chem-dir "$GNPS_CHEM" \
  --output "$OUT/modeling" \
  --folds 5 --seed 20261026 --epochs 20 --batch-queries 256 \
  --hidden 16 --learning-rate 0.001 --weight-decay 0.0001 \
  --anchor-strength 0.01 --bootstrap-resamples 5000 --device cpu

python -u tasks/evaluate_full23_multisource_additive_endpoints.py \
  --run-output "$OUT/modeling" \
  --method-name full23_parent_qrel_cf2_pareto_sgd_residual \
  --expected-status FULL23_PARENT_QREL_CF2_PARETO_SGD_RESIDUAL_DEVELOPMENT_COMPLETE \
  --enveda-ledger "$ENVEDA_LEDGER" \
  --gnps-score-dir "$GNPS_SCORES" \
  --gnps-axis-ledger "$GNPS_AXIS" \
  --gnps-identity-panel "$GNPS_IDENTITY" \
  --gnps-formula-panel "$GNPS_FORMULA" \
  --bootstrap-resamples 5000 --seed 20261030 \
  --output "$OUT/endpoint_evaluation"

sha256sum \
  "$OUT/modeling/full23_parent_qrel_cf2_pareto_sgd_residual.pt" \
  "$OUT/modeling/candidate_predictions_outer_oof.npz" \
  "$OUT/modeling/report.json" \
  "$OUT/endpoint_evaluation/report.json" \
  > "$OUT/SHA256SUMS.txt"
echo "FULL23_PARENT_QREL_CF2_PARETO_SGD_SERVER_COMPLETE: $OUT"
