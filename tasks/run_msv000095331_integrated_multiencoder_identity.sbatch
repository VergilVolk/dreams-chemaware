#!/bin/bash
# Real application scoring with independent Noise/Stage-1/Phase-A retrieval
# evidence. Identity candidates are admitted by precursor compatibility, never
# by WSE >= 0.70. Chemical-relation and BioAware tracks remain separate.
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
MGF_NEW="data/reference/msv000095331_drug_microbiome_20261023/consensus.mgf"
MGF_M1="data/reference/chemprop2_m1_20261011/consensus.mgf"
MGF_23="data/reference/chemprop_cross_design_20261011/chemprop_23drug/spectra/specs_ms.mgf"
MGF_45="data/reference/chemprop_cross_design_20261011/chemprop_45drug/spectra/specs_ms.mgf"
OUT="data/validation/msv000095331_integrated_multiencoder_identity_run_${SLURM_JOB_ID}"
SCRATCH="${SLURM_TMPDIR:-/tmp}/app_multi_encoder_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "APP_MULTI_ENCODER_OUTPUT_COLLISION: $OUT" >&2; exit 3; }
mkdir -p "$OUT" "$SCRATCH"
trap 'rm -rf -- "$SCRATCH"' EXIT

required=(
  "$ARCH" "$NOISE" "$STAGE1" "$PHASEA"
  "$MGF_NEW" "$MGF_M1" "$MGF_23" "$MGF_45"
  tasks/prepare_official_embedding_checkpoint.py
  tasks/encode_crossstudy_mgfs_checkpoint.py
  tasks/score_crossstudy_identity_multiencoder.py
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "APP_MULTI_ENCODER_MISSING: $path" >&2; exit 4; }
done

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
encode_one chemaware_phase_a "$SCRATCH/chemaware_phasea_slim.pt" "$OUT/chemaware_phase_a_embeddings.npz"

python -u tasks/score_crossstudy_identity_multiencoder.py \
  --embedding "noise_v1=$OUT/noise_v1_embeddings.npz" \
  --embedding "chemaware_stage1=$OUT/chemaware_stage1_embeddings.npz" \
  --embedding "chemaware_phase_a=$OUT/chemaware_phase_a_embeddings.npz" \
  --output-dir "$OUT/identity_scoring" --precursor-ppm 10 --fragment-da 0.02

sha256sum \
  "$OUT/noise_v1_embeddings.npz" \
  "$OUT/chemaware_stage1_embeddings.npz" \
  "$OUT/chemaware_phase_a_embeddings.npz" \
  "$OUT/identity_scoring/identity_candidate_scores.tsv.gz" \
  "$OUT/identity_scoring/method_scores_long.tsv.gz" \
  "$OUT/identity_scoring/report.json" > "$OUT/SHA256SUMS.txt"
echo "MSV000095331_INTEGRATED_MULTIENCODER_IDENTITY_COMPLETE: $OUT"
