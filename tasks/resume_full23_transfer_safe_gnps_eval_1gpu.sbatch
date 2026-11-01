#!/bin/bash
# Resume only the transfer-safe GNPS evaluation; reuse the proven benchmark view.
#SBATCH --job-name=full23_safe_eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/full23_safe_eval_%j.out
#SBATCH --error=data/validation/full23_safe_eval_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

if [[ -n "${SOURCE_RUN_DIR:-}" ]]; then
  OUT="$SOURCE_RUN_DIR"
else
  mapfile -t candidates < <(ls -dt data/validation/full23_transfer_safe_tensor_run_* 2>/dev/null || true)
  OUT=""
  for candidate in "${candidates[@]}"; do
    if [[ -s "$candidate/modeling/no_reference_count.pt" \
       && -s "$candidate/modeling/no_count_query_centered.pt" \
       && ! -e "$candidate/gnps_development_evaluation" ]]; then
      OUT="$candidate"
      break
    fi
  done
fi
[[ -n "$OUT" ]] || { echo "FULL23_TRANSFER_SAFE_RESUME_SOURCE_NOT_FOUND" >&2; exit 4; }

# This exact view already passed the MSG benchmark-provenance check in job 2433126.
VIEW="data/validation/full23_tensor_set_readout_run_2433126/gnps_benchmark_view"
SOURCE_RUN="data/validation/enveda_unified_candidate_integrator_run_2411772"
FULL23="data/validation/enveda_no_query_context_diagnostic_run_2413878/modeling/unified_candidate_integrator.pt"
GNPS_BASE="data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz"
GNPS_MSG="data/validation/noise_msg_fusion_run_2349961/gnps_fusion_scores"
GNPS_CHEM="data/validation/wse_conditional_residual_run_2376648/continuous_chem_evidence"
GNPS_NOISE="data/validation/gnps_absolute_baselines_run_2347472/gnps/v1_embeddings.npz"
GNPS_STAGE1="$SOURCE_RUN/gnps_inputs/chemaware_stage1_embeddings.npz"
GNPS_PHASEA="$SOURCE_RUN/gnps_inputs/chemaware_phasea_embeddings.npz"
GNPS_FULLMSG="$SOURCE_RUN/gnps_inputs/noise_chemaware_fullmsg_embeddings.npz"
EVAL="$OUT/gnps_development_evaluation"

required=(
  "$OUT/modeling/no_reference_count.pt" "$OUT/modeling/no_count_query_centered.pt"
  "$VIEW/report.json" "$VIEW/manifest.csv.gz" "$VIEW/panel_identity_disjoint.npz" "$VIEW/panel_formula_disjoint.npz"
  "$FULL23" "$GNPS_BASE" "$GNPS_MSG/report.json" "$GNPS_CHEM/report.json"
  "$GNPS_NOISE" "$GNPS_STAGE1" "$GNPS_PHASEA" "$GNPS_FULLMSG"
)
missing=(); for path in "${required[@]}"; do [[ -s "$path" ]] || missing+=("$path"); done
if (( ${#missing[@]} > 0 )); then printf 'FULL23_TRANSFER_SAFE_RESUME_MISSING: %s\n' "${missing[@]}" >&2; exit 4; fi
[[ ! -e "$EVAL" ]] || { echo "FULL23_TRANSFER_SAFE_EVAL_OUTPUT_EXISTS: $EVAL" >&2; exit 5; }

python -u tasks/evaluate_full23_transfer_safe_tensor_on_gnps.py \
  --benchmark "$VIEW" --score-bundle "$GNPS_BASE" --msg-cache "$GNPS_MSG" \
  --noise-embeddings "$GNPS_NOISE" --stage1-embeddings "$GNPS_STAGE1" \
  --phasea-embeddings "$GNPS_PHASEA" --fullmsg-embeddings "$GNPS_FULLMSG" \
  --chem-dir "$GNPS_CHEM" --full23-checkpoint "$FULL23" \
  --readout-checkpoint "$OUT/modeling" --output "$EVAL" --device cuda

echo "FULL23_TRANSFER_SAFE_GNPS_EVALUATION_RESUMED: $OUT"
