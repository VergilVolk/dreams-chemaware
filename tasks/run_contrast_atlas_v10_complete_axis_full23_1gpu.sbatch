#!/bin/bash
# Score the frozen 16-source/294,236-pair Atlas axis with every Full23 expert.
# Candidate admission is the frozen physical axis; no method can inherit WSE edges.
#SBATCH --job-name=atlas_v10_full23
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/atlas_v10_full23_%j.out
#SBATCH --error=data/validation/atlas_v10_full23_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export PYTHONNOUSERSITE=1

AXIS="${ATLAS_V10_AXIS:-data/validation/contrast_atlas_model_independent_identity_axis_v6_msv000080918_20261031}"
SOURCE_MANIFEST="${ATLAS_V10_SOURCE_MANIFEST:-data/validation/contrast_atlas_v10_identity_axis_inputs_v1_20261031/spectrum_source_manifest.json}"
RELEASE_MANIFEST="deploy/unified_additive_full23_v1/manifest.json"
RUNTIME_EQUIVALENCE_ADDENDUM="deploy/unified_additive_full23_v1/runtime_equivalence_addendum_20261031.json"
MS_ENTROPY_WHEEL="third_party/noise_article_wheels/ms_entropy-1.5.2-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl"
MS_ENTROPY_WHEEL_SHA256="f0a9a83359118f0fdb3260e1a6ba891ba51f76523bbfa0310e065c9ee5b9e7a7"
FULL23="deploy/unified_additive_full23_v1/assets/unified_candidate_integrator.pt"
ARCH="${DREAMS_ARCHITECTURE_CHECKPOINT:-dreams/models/pretrained/ssl_model_server.pt}"
OFFICIAL="${OFFICIAL_DREAMS_CHECKPOINT:-data/e1/official_embedding_slim.pt}"
NOISE="${NOISE_V1_CHECKPOINT:-data/validation/noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt}"
STAGE1="${CHEMAWARE_STAGE1_CHECKPOINT:-data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt}"
PHASEA="${CHEMAWARE_PHASEA_CHECKPOINT:-data/validation/chemaware_online_role_budget_native/run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt}"
FULLMSG="${NOISE_CHEMAWARE_FULLMSG_CHECKPOINT:-data/validation/noise_chemaware_full_massspecgym_native/run_2401890/checkpoint/noise_chemaware_fullmsg_slim.pt}"
MSG_ROOT="${NOISE_MSG_FUSION_MODEL:-data/validation/noise_msg_fusion_run_2349961/fusion_model}"
if [[ -d "$MSG_ROOT" ]]; then
  MSG_MODEL="$MSG_ROOT/fusion_model.joblib"
  MSG_REPORT="$MSG_ROOT/report.json"
else
  MSG_MODEL="$MSG_ROOT"
  MSG_REPORT="$(dirname -- "$MSG_ROOT")/report.json"
fi
CONTRACT="tasks/contrast_atlas_v10_method_pair_score_contract_v1.json"
OUT="data/validation/contrast_atlas_v10_complete_axis_full23_run_${SLURM_JOB_ID}"
SCRATCH="${SLURM_TMPDIR:-/tmp}/atlas_v10_full23_${SLURM_JOB_ID}"
LOCAL_DEPS="${SLURM_TMPDIR:-/tmp}/atlas_v10_full23_deps_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "ATLAS_V10_FULL23_OUTPUT_COLLISION: $OUT" >&2; exit 3; }
mkdir -p "$OUT" "$SCRATCH" "$LOCAL_DEPS"
cleanup() { rm -rf -- "$SCRATCH" "$LOCAL_DEPS"; }
trap cleanup EXIT

# The shared conda environment may contain another ms_entropy release.  Install
# the exact public 1.5.2 wheel into node-local scratch and put it first on
# sys.path.  Both the wheel bytes and the imported distribution are checked
# again by preflight before any GPU work.
[[ -f "$MS_ENTROPY_WHEEL" ]] || { echo "Missing pinned ms_entropy wheel: $MS_ENTROPY_WHEEL" >&2; exit 4; }
echo "$MS_ENTROPY_WHEEL_SHA256  $MS_ENTROPY_WHEEL" | sha256sum --check --strict
python -m pip install --no-deps --target "$LOCAL_DEPS" "$MS_ENTROPY_WHEEL"
export PYTHONPATH="$LOCAL_DEPS:$PYTHONPATH"

# This is the only pre-GPU gate.  Missing/stale bytes fail before checkpoint
# conversion or encoder inference.
python -u tasks/preflight_contrast_atlas_v10_full23.py \
  --axis-dir "$AXIS" \
  --source-manifest "$SOURCE_MANIFEST" \
  --release-manifest "$RELEASE_MANIFEST" \
  --runtime-equivalence-addendum "$RUNTIME_EQUIVALENCE_ADDENDUM" \
  --runtime-wheel "$MS_ENTROPY_WHEEL" \
  --artifact "architecture=$ARCH" \
  --artifact "official_dreams=$OFFICIAL" \
  --artifact "noise_v1=$NOISE" \
  --artifact "chemaware_stage1=$STAGE1" \
  --artifact "chemaware_phasea=$PHASEA" \
  --artifact "noise_chemaware_fullmsg=$FULLMSG" \
  --artifact "noise_msg_model=$MSG_MODEL" \
  --artifact "noise_msg_report=$MSG_REPORT" \
  --artifact "full23_v1=$FULL23" \
  --output "$OUT/preflight.json"

python -m py_compile \
  tasks/preflight_contrast_atlas_v10_full23.py \
  tasks/encode_contrast_atlas_v10_axis_checkpoint.py \
  tasks/score_contrast_atlas_v10_complete_axis_full23.py \
  tasks/package_contrast_atlas_v10_method_scores.py
python -m unittest tasks.test_contrast_atlas_v10_full23_pipeline -v
python -u tasks/test_unified_additive_release.py

python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$STAGE1" --output "$SCRATCH/chemaware_stage1_slim.pt"
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$PHASEA" --output "$SCRATCH/chemaware_phasea_slim.pt"

encode_one() {
  local name="$1" checkpoint="$2" canonical="$3" output="$4"
  python -u tasks/encode_contrast_atlas_v10_axis_checkpoint.py \
    --axis-dir "$AXIS" \
    --source-manifest "$SOURCE_MANIFEST" \
    --checkpoint "$checkpoint" \
    --raw-checkpoint "$ARCH" \
    --canonical-artifact "$canonical" \
    --encoder-name "$name" \
    --output "$output" \
    --device cuda --batch-size 128
}

EMBED="$OUT/embeddings"
mkdir -p "$EMBED"
encode_one official_dreams "$OFFICIAL" "$OFFICIAL" "$EMBED/official_dreams.npz"
encode_one noise_v1 "$NOISE" "$NOISE" "$EMBED/noise_v1.npz"
encode_one chemaware_stage1 "$SCRATCH/chemaware_stage1_slim.pt" "$STAGE1" "$EMBED/chemaware_stage1.npz"
encode_one chemaware_phasea "$SCRATCH/chemaware_phasea_slim.pt" "$PHASEA" "$EMBED/chemaware_phasea.npz"
encode_one noise_chemaware_fullmsg "$FULLMSG" "$FULLMSG" "$EMBED/noise_chemaware_fullmsg.npz"

python -u tasks/score_contrast_atlas_v10_complete_axis_full23.py \
  --axis-dir "$AXIS" \
  --source-manifest "$SOURCE_MANIFEST" \
  --embedding "noise_v1=$EMBED/noise_v1.npz" \
  --embedding "chemaware_stage1=$EMBED/chemaware_stage1.npz" \
  --embedding "chemaware_phasea=$EMBED/chemaware_phasea.npz" \
  --embedding "noise_chemaware_fullmsg=$EMBED/noise_chemaware_fullmsg.npz" \
  --official-embedding "$EMBED/official_dreams.npz" \
  --msg-fusion-model "$MSG_ROOT" \
  --full23-checkpoint "$FULL23" \
  --artifact "weighted_spectral_entropy=tasks/noise_gnps_article_spectral_scores.py" \
  --artifact "official_dreams=$OFFICIAL" \
  --artifact "noise_v1=$NOISE" \
  --artifact "p2b_noise_v1_frozen=tasks/noise_gnps_article_spectral_scores.py" \
  --artifact "noise_msg_fusion_stack_v1=$MSG_MODEL" \
  --artifact "chemaware_stage1=$STAGE1" \
  --artifact "chemaware_phasea=$PHASEA" \
  --artifact "noise_chemaware_fullmsg=$FULLMSG" \
  --artifact "full23_v1=$FULL23" \
  --device cuda --fragment-da 0.02 \
  --output-dir "$OUT/scoring"

python -u tasks/package_contrast_atlas_v10_method_scores.py \
  --axis-dir "$AXIS" \
  --scoring-dir "$OUT/scoring" \
  --contract "$CONTRACT" \
  --output-dir "$OUT/method_packages"

find "$OUT" -type f ! -name SHA256SUMS.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$OUT/SHA256SUMS.txt"
echo "CONTRAST_ATLAS_V10_COMPLETE_AXIS_FULL23_FINISHED: $OUT"
