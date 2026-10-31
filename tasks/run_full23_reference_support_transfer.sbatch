#!/bin/bash
# Large-source Full23 reference-support training and fixed GNPS transfer.
# Reuses all upstream caches; no spectrum encoding is performed.
#SBATCH --job-name=full23_refsupport
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --output=data/validation/full23_refsupport_%j.out
#SBATCH --error=data/validation/full23_refsupport_%j.err

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
FULL23_RUN="data/validation/enveda_no_query_context_diagnostic_run_2413878"

GNPS_ID="data/validation/GLM_gnps_identity_panel_reconstruction/panel_identity_disjoint.npz"
GNPS_FORMULA="data/validation/gnps_gold_silver_10ppm_benchmark_v1/panel_formula_disjoint.npz"
GNPS_BASE="data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz"
GNPS_MSG="data/validation/noise_msg_fusion_run_2349961/gnps_fusion_scores"
GNPS_SCORES="$FULL23_RUN/gnps_fixed_evaluation"

ENVEDA_MSG="$REUSE_RUN/inputs/enveda_msg_cache"
ENVEDA_STAGE1="$REUSE_RUN/inputs/enveda_chemaware_stage1_embeddings.npz"
ENVEDA_PHASEA="$REUSE_RUN/inputs/enveda_chemaware_phasea_embeddings.npz"
ENVEDA_FULLMSG="$REUSE_RUN/inputs/enveda_noise_chemaware_fullmsg_embeddings.npz"
GNPS_STAGE1="$SOURCE_RUN/gnps_inputs/chemaware_stage1_embeddings.npz"
GNPS_PHASEA="$SOURCE_RUN/gnps_inputs/chemaware_phasea_embeddings.npz"
GNPS_FULLMSG="$SOURCE_RUN/gnps_inputs/noise_chemaware_fullmsg_embeddings.npz"

OUT="data/validation/full23_reference_support_transfer_run_${SLURM_JOB_ID}"

required=(
  tasks/train_full23_reference_coherence_residual.py
  tasks/train_full23_reference_support_risk_residual.py
  tasks/train_full23_reference_support_transfer.py
  tasks/test_full23_reference_support_transfer.py
  "$ENVEDA/panel_identity_disjoint.npz"
  "$ENVEDA/panel_formula_disjoint.npz"
  "$ENVEDA_LAYER/method_scores_unopened.npz"
  "$ENVEDA_LAYER/noise_v1_embeddings.npz"
  "$ENVEDA_MSG/pair_scores_identity_disjoint.npy"
  "$ENVEDA_MSG/pair_scores_formula_disjoint.npy"
  "$ENVEDA_STAGE1" "$ENVEDA_PHASEA" "$ENVEDA_FULLMSG"
  "$FULL23_RUN/modeling/candidate_predictions_oof.npz"
  "$GNPS_ID" "$GNPS_FORMULA" "$GNPS_BASE"
  "$GNPS_MSG/pair_scores_identity_disjoint.npy"
  "$GNPS_MSG/pair_scores_formula_disjoint.npy"
  data/validation/gnps_absolute_baselines_run_2347472/gnps/v1_embeddings.npz
  "$GNPS_STAGE1" "$GNPS_PHASEA" "$GNPS_FULLMSG"
  "$GNPS_SCORES/candidate_scores_identity_disjoint.npz"
  "$GNPS_SCORES/candidate_scores_formula_disjoint.npz"
)
missing=()
for path in "${required[@]}"; do
  [[ -s "$path" ]] || missing+=("$path")
done
if (( ${#missing[@]} > 0 )); then
  printf 'FULL23_REFERENCE_SUPPORT_MISSING: %s\n' "${missing[@]}" >&2
  exit 4
fi
[[ ! -e "$OUT" ]] || { echo "FULL23_REFERENCE_SUPPORT_OUTPUT_EXISTS: $OUT" >&2; exit 5; }

python -m py_compile \
  tasks/train_full23_reference_coherence_residual.py \
  tasks/train_full23_reference_support_risk_residual.py \
  tasks/train_full23_reference_support_transfer.py \
  tasks/test_full23_reference_support_transfer.py
python -u tasks/test_full23_reference_support_transfer.py

python -u tasks/train_full23_reference_support_transfer.py \
  --enveda-benchmark "$ENVEDA" \
  --enveda-score-bundle "$ENVEDA_LAYER/method_scores_unopened.npz" \
  --enveda-msg-cache "$ENVEDA_MSG" \
  --enveda-embedding "chemaware_stage1=$ENVEDA_STAGE1" \
  --enveda-embedding "chemaware_phasea=$ENVEDA_PHASEA" \
  --enveda-embedding "noise_chemaware_fullmsg=$ENVEDA_FULLMSG" \
  --enveda-full23-oof "$FULL23_RUN/modeling/candidate_predictions_oof.npz" \
  --gnps-identity-panel "$GNPS_ID" \
  --gnps-formula-panel "$GNPS_FORMULA" \
  --gnps-score-bundle "$GNPS_BASE" \
  --gnps-msg-cache "$GNPS_MSG" \
  --gnps-embedding "chemaware_stage1=$GNPS_STAGE1" \
  --gnps-embedding "chemaware_phasea=$GNPS_PHASEA" \
  --gnps-embedding "noise_chemaware_fullmsg=$GNPS_FULLMSG" \
  --gnps-full23-scores "$GNPS_SCORES" \
  --output "$OUT" \
  --epochs 40 --batch-queries 256 \
  --learning-rate 0.015 --weight-decay 0.002 --boundary-l2 0.002 \
  --residual-bound 0.25 --bootstrap-resamples 5000 --seed 20261031

echo "FULL23_REFERENCE_SUPPORT_FIXED_TRANSFER_COMPLETE_NO_EXTERNAL_ACCESS: $OUT"
