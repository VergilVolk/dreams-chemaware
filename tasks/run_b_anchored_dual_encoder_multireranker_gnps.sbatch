#!/bin/bash
# One bounded development job: fixed B + two ChemAware encoders + MSG/RRF
# family evidence.  GNPS is consumed development; no external data are read.
#SBATCH --job-name=B_dual_multi
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/B_dual_multi_%j.out
#SBATCH --error=data/validation/B_dual_multi_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

BENCHMARK="${GNPS_BENCHMARK:-data/validation/gnps_gold_silver_10ppm_benchmark_v1}"
IDENTITY_PANEL="${GNPS_IDENTITY_PANEL:-data/validation/GLM_gnps_identity_panel_reconstruction/panel_identity_disjoint.npz}"
FORMULA_PANEL="${GNPS_FORMULA_PANEL:-$BENCHMARK/panel_formula_disjoint.npz}"
BASE_SCORES="${GNPS_SCORE_BUNDLE:-data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz}"
SOURCE_RUN="${UNIFIED_GNPS_SOURCE_RUN:-data/validation/unified_candidate_energy_run_2364137}"
AB_RUN="${WSE_AB_PARENT_RUN:-data/validation/wse_conditional_residual_run_2376648}"
CHEM_CACHE="${CHEMAWARE_CONTINUOUS_CACHE:-$AB_RUN/continuous_chem_evidence}"
ARCH="${DREAMS_ARCHITECTURE_CHECKPOINT:-dreams/models/pretrained/ssl_model_server.pt}"
CHEM_STAGE1="${CHEMAWARE_STAGE1_CHECKPOINT:-data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt}"
CHEM_PHASEA="${CHEMAWARE_PHASEA_CHECKPOINT:-data/validation/chemaware_online_role_budget_native/run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt}"
NOISE_STAGE1_EMB="${NOISE_RRF_STAGE1_EMBEDDINGS:-data/validation/gnps_absolute_baselines_run_2347472/gnps/stage1_embeddings.npz}"
OUT="data/validation/B_dual_encoder_multireranker_run_${SLURM_JOB_ID}"
SCRATCH="${SLURM_TMPDIR:-/tmp}/B_dual_multi_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "B_DUAL_MULTI_OUTPUT_COLLISION: $OUT" >&2; exit 3; }
mkdir -p "$OUT" "$SCRATCH"
cleanup() { rm -rf -- "$SCRATCH"; }
trap cleanup EXIT

# MSG is a frozen parent model trained formula-OOF before GNPS application.
# Ambiguous discovery fails closed; an explicit MSG_FUSION_GNPS overrides it.
if [[ -n "${MSG_FUSION_GNPS:-}" ]]; then
  MSG_CACHE="$MSG_FUSION_GNPS"
else
  msg_hits=()
  for candidate in data/validation/noise_msg_fusion_run_*/gnps_fusion_scores; do
    [[ -d "$candidate" && -s "$candidate/report.json" ]] && msg_hits+=("$candidate")
  done
  if (( ${#msg_hits[@]} != 1 )); then
    echo "B_DUAL_MULTI_MSG_CACHE: expected exactly one reusable cache directory, found ${#msg_hits[@]}" >&2
    printf '  %s\n' "${msg_hits[@]:-}" >&2
    exit 4
  fi
  MSG_CACHE="${msg_hits[0]}"
fi

required=(
  "$BENCHMARK/spectra.mgf" "$BENCHMARK/manifest.csv.gz"
  "$BENCHMARK/panel_identity_disjoint.npz" "$BENCHMARK/panel_formula_disjoint.npz"
  "$IDENTITY_PANEL" "$FORMULA_PANEL" "$BASE_SCORES"
  "$SOURCE_RUN/evidence_identity_disjoint.npz" "$SOURCE_RUN/evidence_formula_disjoint.npz"
  "$CHEM_CACHE/continuous_chem_identity_disjoint.npz"
  "$CHEM_CACHE/continuous_chem_formula_disjoint.npz"
  "$AB_RUN/modeling/report.json" "$AB_RUN/modeling/candidate_predictions.npz"
  "$ARCH" "$CHEM_STAGE1" "$CHEM_PHASEA" "$NOISE_STAGE1_EMB"
  "$MSG_CACHE/report.json"
  tasks/extend_gnps_article_score_bundle.py tasks/build_grand_unified_evidence_bundle.py
  tasks/export_gnps_stage1_rrf_pair_scores.py
  tasks/validate_gnps_panel_candidate_axis.py
  tasks/train_b_anchored_dual_encoder_multireranker.py
  tasks/test_b_anchored_dual_encoder_multireranker.py
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "B_DUAL_MULTI_MISSING: $path" >&2; exit 4; }
done

python -m py_compile \
  tasks/export_gnps_stage1_rrf_pair_scores.py \
  tasks/validate_gnps_panel_candidate_axis.py \
  tasks/train_b_anchored_dual_encoder_multireranker.py
python -u tasks/test_b_anchored_dual_encoder_multireranker.py
python -u tasks/validate_gnps_panel_candidate_axis.py \
  --left "$BENCHMARK/panel_identity_disjoint.npz" --right "$IDENTITY_PANEL"
python -u tasks/validate_gnps_panel_candidate_axis.py \
  --left "$BENCHMARK/panel_formula_disjoint.npz" --right "$FORMULA_PANEL"

# Reuse explicit caches when supplied; otherwise encode each frozen ChemAware
# checkpoint once.  No upstream model is trained in this job.
encode_chem() {
  local name="$1" source="$2" supplied="$3"
  if [[ -n "$supplied" ]]; then
    [[ -s "$supplied" ]] || { echo "B_DUAL_MULTI_MISSING_${name}_CACHE: $supplied" >&2; exit 4; }
    printf '%s' "$supplied"
    return
  fi
  local slim="$SCRATCH/${name}.pt" emb="$SCRATCH/${name}_embeddings.npz"
  python -u tasks/prepare_official_embedding_checkpoint.py --source "$source" --output "$slim" >&2
  python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
    --benchmark "$BENCHMARK" --checkpoint "$slim" --architecture-checkpoint "$ARCH" \
    --output "$emb" --panel-file "identity_disjoint=$IDENTITY_PANEL" \
    --panel-file "formula_disjoint=$FORMULA_PANEL" --batch-size 64 --label "$name" >&2
  printf '%s' "$emb"
}
STAGE1_EMB="$(encode_chem chemaware-stage1 "$CHEM_STAGE1" "${GNPS_CHEMAWARE_STAGE1_EMBEDDINGS:-}")"
PHASEA_EMB="$(encode_chem chemaware-phaseA "$CHEM_PHASEA" "${GNPS_CHEMAWARE_PHASEA_EMBEDDINGS:-}")"

if [[ -n "${NOISE_RRF_GNPS_CACHE:-}" ]]; then
  RRF_CACHE="$NOISE_RRF_GNPS_CACHE"
else
  RRF_CACHE="$OUT/noise_rrf_pair_scores"
  python -u tasks/export_gnps_stage1_rrf_pair_scores.py \
    --benchmark "$BENCHMARK" --stage1-embeddings "$NOISE_STAGE1_EMB" --output "$RRF_CACHE"
fi

python -u tasks/extend_gnps_article_score_bundle.py \
  --benchmark "$BENCHMARK" --base-bundle "$BASE_SCORES" \
  --identity-panel "$IDENTITY_PANEL" --formula-panel "$FORMULA_PANEL" \
  --embedding-method "chemaware_stage1_encoder=$STAGE1_EMB" \
  --embedding-method "chemaware_phase_a_encoder=$PHASEA_EMB" \
  --pair-score-method "noise_msg_fusion_stack_v1=$MSG_CACHE" \
  --pair-score-method "noise_rrf_1_7=$RRF_CACHE" \
  --output "$OUT/extended_method_scores.npz"

for panel in identity_disjoint formula_disjoint; do
  if [[ "$panel" == identity_disjoint ]]; then panel_path="$IDENTITY_PANEL"; else panel_path="$FORMULA_PANEL"; fi
  python -u tasks/build_grand_unified_evidence_bundle.py \
    --panel "$panel" --panel-npz "$panel_path" --base-scores "$OUT/extended_method_scores.npz" \
    --module weighted_spectral_entropy --module noise_v1 --module p2b_noise_v1_frozen \
    --module chemaware_stage1_encoder --module chemaware_phase_a_encoder \
    --module noise_msg_fusion_stack_v1 --module noise_rrf_1_7 \
    --out "$OUT/evidence_${panel}.npz"
done

python -u tasks/train_b_anchored_dual_encoder_multireranker.py \
  --identity-bundle "$OUT/evidence_identity_disjoint.npz" \
  --formula-bundle "$OUT/evidence_formula_disjoint.npz" \
  --core-identity-bundle "$SOURCE_RUN/evidence_identity_disjoint.npz" \
  --core-formula-bundle "$SOURCE_RUN/evidence_formula_disjoint.npz" \
  --identity-chem "$CHEM_CACHE/continuous_chem_identity_disjoint.npz" \
  --formula-chem "$CHEM_CACHE/continuous_chem_formula_disjoint.npz" \
  --parent-report "$AB_RUN/modeling/report.json" \
  --parent-predictions "$AB_RUN/modeling/candidate_predictions.npz" \
  --output "$OUT/modeling" --device cuda

cp "${STAGE1_EMB%.npz}.json" "$OUT/chemaware_stage1_encoding.json"
cp "${PHASEA_EMB%.npz}.json" "$OUT/chemaware_phaseA_encoding.json"
echo "B_DUAL_ENCODER_MULTIRERANKER_GNPS_COMPLETE_NO_EXTERNAL_ACCESS: $OUT"
