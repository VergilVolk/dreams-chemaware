#!/bin/bash
# Real application scoring with the selected unified candidate ranker.
# Identity candidates are admitted by precursor compatibility, never by a
# method score. BioAware joins later on the stable candidate_pair_id axis.
#SBATCH --job-name=app_multi_encoder
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/app_multi_encoder_%j.out
#SBATCH --error=data/validation/app_multi_encoder_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

ARCH="${DREAMS_ARCHITECTURE_CHECKPOINT:-dreams/models/pretrained/ssl_model_server.pt}"
NOISE="${NOISE_V1_CHECKPOINT:-data/validation/noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt}"
STAGE1="${CHEMAWARE_STAGE1_CHECKPOINT:-data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt}"
PHASEA="${CHEMAWARE_PHASEA_CHECKPOINT:-data/validation/chemaware_online_role_budget_native/run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt}"
FULLMSG_CHEM="${NOISE_CHEMAWARE_FULLMSG_CHECKPOINT:-data/validation/noise_chemaware_full_massspecgym_native/run_2401890/checkpoint/noise_chemaware_fullmsg_slim.pt}"
OFFICIAL="${OFFICIAL_DREAMS_CHECKPOINT:-data/e1/official_embedding_slim.pt}"
MSG_FUSION_MODEL="${NOISE_MSG_FUSION_MODEL:-data/validation/noise_msg_fusion_run_2349961/fusion_model}"
if [[ -d "$MSG_FUSION_MODEL" ]]; then
  MSG_MODEL_ARTIFACT="$MSG_FUSION_MODEL/fusion_model.joblib"
  MSG_MODEL_REPORT="$MSG_FUSION_MODEL/report.json"
else
  MSG_MODEL_ARTIFACT="$MSG_FUSION_MODEL"
  MSG_MODEL_REPORT="$(dirname -- "$MSG_FUSION_MODEL")/report.json"
fi
FIXED_ADDITIVE_MODEL="data/validation/enveda_no_query_context_diagnostic_run_2413878/modeling/unified_candidate_integrator.pt"
UNIFIED_MODEL="${UNIFIED_RANKER_MODEL:-$FIXED_ADDITIVE_MODEL}"
IDENTITY_HANDOFF="${IDENTITY_HANDOFF:-data/validation/msv000095331_complete_identity_handoff_v3_20261023}"
REUSE_EMBEDDING_ROOT="${APPLICATION_EMBEDDING_CACHE_ROOT:-data/validation/msv000095331_integrated_multiencoder_identity_run_2404230}"
# Optional semicolon-separated NAME=PATH::COLUMN specifications.  They are
# required only when the selected unified payload declares auxiliary families.
UNIFIED_AUXILIARY_SCORE_SPECS="${UNIFIED_AUXILIARY_SCORE_SPECS:-}"
ENVEDA_INTEGRATOR_ARM="${ENVEDA_INTEGRATOR_ARM:-}"
MGF_NEW="data/reference/msv000095331_drug_microbiome_20261023/consensus.mgf"
MGF_M1="data/reference/chemprop2_m1_20261011/consensus.mgf"
MGF_23="data/reference/chemprop_cross_design_20261011/chemprop_23drug/spectra/specs_ms.mgf"
MGF_45="data/reference/chemprop_cross_design_20261011/chemprop_45drug/spectra/specs_ms.mgf"
OUT="data/validation/msv000095331_integrated_multiencoder_identity_run_${SLURM_JOB_ID}"
SCRATCH="${SLURM_TMPDIR:-/tmp}/app_multi_encoder_${SLURM_JOB_ID}"

NOISE_EMBEDDING="$REUSE_EMBEDDING_ROOT/noise_v1_embeddings.npz"
STAGE1_EMBEDDING="$REUSE_EMBEDDING_ROOT/chemaware_stage1_embeddings.npz"
PHASEA_EMBEDDING="$REUSE_EMBEDDING_ROOT/chemaware_phase_a_embeddings.npz"
FULLMSG_EMBEDDING="$REUSE_EMBEDDING_ROOT/noise_chemaware_fullmsg_embeddings.npz"
REUSE_FROZEN_EMBEDDINGS=false
if [[ -s "$NOISE_EMBEDDING" && -s "$STAGE1_EMBEDDING" \
      && -s "$PHASEA_EMBEDDING" && -s "$FULLMSG_EMBEDDING" ]]; then
  REUSE_FROZEN_EMBEDDINGS=true
fi

required=(
  "$ARCH" "$OFFICIAL" "$UNIFIED_MODEL"
  "$MGF_NEW" "$MGF_M1" "$MGF_23" "$MGF_45"
  "$IDENTITY_HANDOFF/complete_identity_candidate_pairs.tsv.gz"
  "$IDENTITY_HANDOFF/directional_query_registry.tsv.gz"
  "$IDENTITY_HANDOFF/wse_identity_inference_axis.npz"
  tasks/encode_crossstudy_mgfs_checkpoint.py
  tasks/massspecgym_embedding_readout_core.py
  tasks/enveda_unified_candidate_integrator_core.py
  tasks/noise_msg_pair_evidence_core.py
  tasks/noise_gnps_article_spectral_scores.py
  tasks/score_crossstudy_identity_multiencoder.py
  tasks/test_crossstudy_identity_multiencoder.py
)
if [[ "$REUSE_FROZEN_EMBEDDINGS" == true ]]; then
  required+=("$NOISE_EMBEDDING" "$STAGE1_EMBEDDING" "$PHASEA_EMBEDDING" "$FULLMSG_EMBEDDING")
else
  required+=(
    "$NOISE" "$STAGE1" "$PHASEA" "$FULLMSG_CHEM"
    tasks/prepare_official_embedding_checkpoint.py
  )
fi
missing=()
for path in "${required[@]}"; do
  [[ -s "$path" ]] || missing+=("$path")
done
if (( ${#missing[@]} > 0 )); then
  printf 'APP_MULTI_ENCODER_MISSING: %s\n' "${missing[@]}" >&2
  exit 4
fi

model_schema="$(python - "$UNIFIED_MODEL" <<'PY'
import sys, torch
try:
    body=torch.load(sys.argv[1], map_location='cpu', weights_only=False)
except TypeError:
    body=torch.load(sys.argv[1], map_location='cpu')
print(body.get('schema','legacy_massspecgym_embedding_readout'))
PY
)"
if [[ "$model_schema" == "enveda_unified_candidate_integrator_v1" \
      && -z "$ENVEDA_INTEGRATOR_ARM" ]]; then
  # The 23-feature learned Additive is the fixed cross-source deployment
  # candidate.  The checkpoint's nonlinear primary remains available only
  # when explicitly requested through ENVEDA_INTEGRATOR_ARM.
  ENVEDA_INTEGRATOR_ARM="additive_linear_control"
fi
if [[ -n "$ENVEDA_INTEGRATOR_ARM" ]]; then
  python - "$UNIFIED_MODEL" "$ENVEDA_INTEGRATOR_ARM" <<'PY'
import sys, torch
try:
    body = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
except TypeError:
    body = torch.load(sys.argv[1], map_location='cpu')
if body.get('schema') != 'enveda_unified_candidate_integrator_v1':
    raise SystemExit('APP_ENVEDA_ARM_REQUIRES_ENVEDA_INTEGRATOR_CHECKPOINT')
if sys.argv[2] not in body.get('arms', {}):
    raise SystemExit(f'APP_ENVEDA_ARM_NOT_IN_CHECKPOINT: {sys.argv[2]}')
PY
fi
if [[ "$model_schema" == "enveda_unified_candidate_integrator_v1" \
      && ( ! -s "$MSG_MODEL_ARTIFACT" || ! -s "$MSG_MODEL_REPORT" ) ]]; then
  echo "APP_UNIFIED_MSG_MODEL_MISSING: $MSG_FUSION_MODEL" >&2
  exit 4
fi
[[ ! -e "$OUT" ]] || { echo "APP_MULTI_ENCODER_OUTPUT_COLLISION: $OUT" >&2; exit 3; }
mkdir -p "$OUT" "$SCRATCH"
trap 'rm -rf -- "$SCRATCH"' EXIT

python -m py_compile \
  tasks/encode_crossstudy_mgfs_checkpoint.py \
  tasks/score_crossstudy_identity_multiencoder.py
python -u tasks/test_crossstudy_identity_multiencoder.py

study_args=(
  --study-mgf "MSV000095331=$MGF_NEW"
  --study-mgf "MSV000094899=$MGF_M1"
  --study-mgf "MSV000096724=$MGF_23"
  --study-mgf "MSV000093571=$MGF_45"
)

encode_one() {
  local name="$1" checkpoint="$2" output="$3"
  python -u tasks/encode_crossstudy_mgfs_checkpoint.py \
    "${study_args[@]}" --checkpoint "$checkpoint" --raw-checkpoint "$ARCH" \
    --encoder-name "$name" --output "$output" --device cuda --batch-size 128
}

if [[ "$REUSE_FROZEN_EMBEDDINGS" == true ]]; then
  echo "APP_MULTI_ENCODER_REUSING_FROZEN_EMBEDDINGS: $REUSE_EMBEDDING_ROOT"
else
  python -u tasks/prepare_official_embedding_checkpoint.py \
    --source "$STAGE1" --output "$SCRATCH/chemaware_stage1_slim.pt"
  python -u tasks/prepare_official_embedding_checkpoint.py \
    --source "$PHASEA" --output "$SCRATCH/chemaware_phasea_slim.pt"
  NOISE_EMBEDDING="$OUT/noise_v1_embeddings.npz"
  STAGE1_EMBEDDING="$OUT/chemaware_stage1_embeddings.npz"
  PHASEA_EMBEDDING="$OUT/chemaware_phasea_embeddings.npz"
  FULLMSG_EMBEDDING="$OUT/noise_chemaware_fullmsg_embeddings.npz"
  encode_one noise_v1 "$NOISE" "$NOISE_EMBEDDING"
  encode_one chemaware_stage1 "$SCRATCH/chemaware_stage1_slim.pt" "$STAGE1_EMBEDDING"
  encode_one chemaware_phasea "$SCRATCH/chemaware_phasea_slim.pt" "$PHASEA_EMBEDDING"
  encode_one noise_chemaware_fullmsg "$FULLMSG_CHEM" "$FULLMSG_EMBEDDING"
fi
OFFICIAL_EMBEDDING="$OUT/official_dreams_embeddings.npz"
encode_one official_dreams "$OFFICIAL" "$OFFICIAL_EMBEDDING"

score_args=(
  --identity-handoff "$IDENTITY_HANDOFF"
  --embedding "noise_v1=$NOISE_EMBEDDING"
  --embedding "chemaware_stage1=$STAGE1_EMBEDDING"
  --embedding "chemaware_phasea=$PHASEA_EMBEDDING"
  --embedding "noise_chemaware_fullmsg=$FULLMSG_EMBEDDING"
  --baseline-embedding "official_dreams=$OFFICIAL_EMBEDDING"
)
if [[ -s "$MSG_MODEL_ARTIFACT" && -s "$MSG_MODEL_REPORT" ]]; then
  score_args+=(--msg-fusion-model "$MSG_FUSION_MODEL")
fi
if [[ -n "$UNIFIED_AUXILIARY_SCORE_SPECS" ]]; then
  IFS=';' read -r -a auxiliary_specs <<< "$UNIFIED_AUXILIARY_SCORE_SPECS"
  for specification in "${auxiliary_specs[@]}"; do
    [[ -n "$specification" ]] && score_args+=(--auxiliary-score "$specification")
  done
fi
if [[ -n "$ENVEDA_INTEGRATOR_ARM" ]]; then
  score_args+=(--enveda-integrator-arm "$ENVEDA_INTEGRATOR_ARM")
fi

python -u tasks/score_crossstudy_identity_multiencoder.py \
  "${score_args[@]}" \
  --unified-model "$UNIFIED_MODEL" --device cuda \
  --output-dir "$OUT/identity_scoring" --fragment-da 0.02

sha256sum \
  "$NOISE_EMBEDDING" \
  "$STAGE1_EMBEDDING" \
  "$PHASEA_EMBEDDING" \
  "$FULLMSG_EMBEDDING" \
  "$OFFICIAL_EMBEDDING" \
  "$OUT/identity_scoring/identity_candidate_scores.tsv.gz" \
  "$OUT/identity_scoring/method_scores_long.tsv.gz" \
  "$OUT/identity_scoring/directional_query_registry.tsv.gz" \
  "$OUT/identity_scoring/report.json" > "$OUT/SHA256SUMS.txt"
echo "MSV000095331_UNIFIED_CANDIDATE_RANKER_APPLICATION_COMPLETE: $OUT"
