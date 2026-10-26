#!/bin/bash
# Real application scoring with the selected unified candidate ranker.
# Identity candidates are admitted by precursor compatibility, never by a
# method score. BioAware joins later on the stable candidate_pair_id axis.
#SBATCH --job-name=app_multi_encoder
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
UNIFIED_MODEL="${UNIFIED_RANKER_MODEL:-}"
IDENTITY_HANDOFF="${IDENTITY_HANDOFF:-data/validation/msv000095331_complete_identity_handoff_v3_20261023}"
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

if [[ -z "$UNIFIED_MODEL" || ! -f "$UNIFIED_MODEL" ]]; then
  UNIFIED_MODEL="$(python - <<'PY'
import json
from pathlib import Path
new=[]
for report in Path('data/validation').glob(
        'enveda_unified_candidate_integrator_run_*/modeling/report.json'):
    try: body=json.loads(report.read_text())
    except Exception: continue
    model=report.parent/'unified_candidate_integrator.pt'
    if body.get('status') == 'ENVEDA_UNIFIED_CANDIDATE_INTEGRATOR_META_DEVELOPMENT_COMPLETE' \
            and model.is_file():
        new.append(model)
if new:
    new.sort(key=lambda path:(path.stat().st_mtime_ns,str(path)))
    print(new[-1])
    raise SystemExit(0)
matches = []
for report in Path('data/validation').glob('massspecgym_unified_ranker_run_*/modeling/report.json'):
    try:
        body = json.loads(report.read_text())
    except Exception:
        continue
    model = report.parent / 'selected_readout.pt'
    if (body.get('status') == 'MASSSPECGYM_UNIFIED_CANDIDATE_RANKER_COMPLETE'
            and body.get('selection_class') == 'full_embedding_integration'
            and body.get('selection') in {'projected_cosine', 'embedding_pair'}
            and model.is_file()):
        matches.append(model)
if not matches:
    raise SystemExit('APP_UNIFIED_RANKER_MODEL_NOT_FOUND')
matches.sort(key=lambda path: (path.stat().st_mtime_ns, str(path)))
print(matches[-1])
PY
)"
fi

required=(
  "$ARCH" "$NOISE" "$STAGE1" "$PHASEA" "$FULLMSG_CHEM" "$OFFICIAL" "$UNIFIED_MODEL"
  "$MGF_NEW" "$MGF_M1" "$MGF_23" "$MGF_45"
  "$IDENTITY_HANDOFF/complete_identity_candidate_pairs.tsv.gz"
  "$IDENTITY_HANDOFF/directional_query_registry.tsv.gz"
  "$IDENTITY_HANDOFF/wse_identity_inference_axis.npz"
  tasks/prepare_official_embedding_checkpoint.py
  tasks/encode_crossstudy_mgfs_checkpoint.py
  tasks/massspecgym_embedding_readout_core.py
  tasks/enveda_unified_candidate_integrator_core.py
  tasks/noise_msg_pair_evidence_core.py
  tasks/noise_gnps_article_spectral_scores.py
  tasks/score_crossstudy_identity_multiencoder.py
  tasks/test_crossstudy_identity_multiencoder.py
)
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

python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$STAGE1" --output "$SCRATCH/chemaware_stage1_slim.pt"
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$PHASEA" --output "$SCRATCH/chemaware_phasea_slim.pt"

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

encode_one noise_v1 "$NOISE" "$OUT/noise_v1_embeddings.npz"
encode_one chemaware_stage1 "$SCRATCH/chemaware_stage1_slim.pt" "$OUT/chemaware_stage1_embeddings.npz"
encode_one chemaware_phasea "$SCRATCH/chemaware_phasea_slim.pt" "$OUT/chemaware_phasea_embeddings.npz"
encode_one noise_chemaware_fullmsg "$FULLMSG_CHEM" "$OUT/noise_chemaware_fullmsg_embeddings.npz"
encode_one official_dreams "$OFFICIAL" "$OUT/official_dreams_embeddings.npz"

score_args=(
  --identity-handoff "$IDENTITY_HANDOFF"
  --embedding "noise_v1=$OUT/noise_v1_embeddings.npz"
  --embedding "chemaware_stage1=$OUT/chemaware_stage1_embeddings.npz"
  --embedding "chemaware_phasea=$OUT/chemaware_phasea_embeddings.npz"
  --embedding "noise_chemaware_fullmsg=$OUT/noise_chemaware_fullmsg_embeddings.npz"
  --baseline-embedding "official_dreams=$OUT/official_dreams_embeddings.npz"
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
  "$OUT/noise_v1_embeddings.npz" \
  "$OUT/chemaware_stage1_embeddings.npz" \
  "$OUT/chemaware_phasea_embeddings.npz" \
  "$OUT/noise_chemaware_fullmsg_embeddings.npz" \
  "$OUT/official_dreams_embeddings.npz" \
  "$OUT/identity_scoring/identity_candidate_scores.tsv.gz" \
  "$OUT/identity_scoring/method_scores_long.tsv.gz" \
  "$OUT/identity_scoring/directional_query_registry.tsv.gz" \
  "$OUT/identity_scoring/report.json" > "$OUT/SHA256SUMS.txt"
echo "MSV000095331_UNIFIED_CANDIDATE_RANKER_APPLICATION_COMPLETE: $OUT"
