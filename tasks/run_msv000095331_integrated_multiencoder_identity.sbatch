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
UNIFIED_MODEL="${UNIFIED_RANKER_MODEL:-}"
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
  "$ARCH" "$NOISE" "$STAGE1" "$PHASEA" "$FULLMSG_CHEM" "$UNIFIED_MODEL"
  "$MGF_NEW" "$MGF_M1" "$MGF_23" "$MGF_45"
  tasks/prepare_official_embedding_checkpoint.py
  tasks/encode_crossstudy_mgfs_checkpoint.py
  tasks/massspecgym_embedding_readout_core.py
  tasks/noise_gnps_article_spectral_scores.py
  tasks/score_crossstudy_identity_multiencoder.py
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "APP_MULTI_ENCODER_MISSING: $path" >&2; exit 4; }
done
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

python -u tasks/score_crossstudy_identity_multiencoder.py \
  --embedding "noise_v1=$OUT/noise_v1_embeddings.npz" \
  --embedding "chemaware_stage1=$OUT/chemaware_stage1_embeddings.npz" \
  --embedding "chemaware_phasea=$OUT/chemaware_phasea_embeddings.npz" \
  --embedding "noise_chemaware_fullmsg=$OUT/noise_chemaware_fullmsg_embeddings.npz" \
  --unified-model "$UNIFIED_MODEL" --device cuda \
  --output-dir "$OUT/identity_scoring" --precursor-ppm 10 --fragment-da 0.02

sha256sum \
  "$OUT/noise_v1_embeddings.npz" \
  "$OUT/chemaware_stage1_embeddings.npz" \
  "$OUT/chemaware_phasea_embeddings.npz" \
  "$OUT/noise_chemaware_fullmsg_embeddings.npz" \
  "$OUT/identity_scoring/identity_candidate_scores.tsv.gz" \
  "$OUT/identity_scoring/method_scores_long.tsv.gz" \
  "$OUT/identity_scoring/report.json" > "$OUT/SHA256SUMS.txt"
echo "MSV000095331_UNIFIED_CANDIDATE_RANKER_APPLICATION_COMPLETE: $OUT"
