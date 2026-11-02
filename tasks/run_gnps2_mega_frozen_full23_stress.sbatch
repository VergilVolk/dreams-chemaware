#!/bin/bash
# Frozen-model, known-overlap throughput/robustness stress. No training/tuning.
# Submit: sbatch tasks/run_gnps2_mega_frozen_full23_stress.sbatch
# Local dependency check: bash tasks/run_gnps2_mega_frozen_full23_stress.sbatch --preflight
#SBATCH --job-name=gnps2_mega_frozen
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --output=data/validation/gnps2_mega_frozen_%j.out
#SBATCH --error=data/validation/gnps2_mega_frozen_%j.err

set -euo pipefail
if [[ -f tasks/materialize_gnps2_mega_frozen_scoring_axis.py ]]; then
  :
else
  cd /data02/run01/scv7tsl/DreaMS
fi
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export PYTHONNOUSERSITE=1

PREFLIGHT="data/validation/gnps2_standard_mega_7source_preflight_v2_20261102"
SOURCES="tasks/gnps2_standard_mega_sources_v2.json"
ARCH="${DREAMS_ARCHITECTURE_CHECKPOINT:-dreams/models/pretrained/ssl_model_server.pt}"
OFFICIAL="${OFFICIAL_DREAMS_CHECKPOINT:-data/e1/official_embedding_slim.pt}"
NOISE="${NOISE_V1_CHECKPOINT:-data/validation/noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt}"
STAGE1="${CHEMAWARE_STAGE1_CHECKPOINT:-data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt}"
PHASEA="${CHEMAWARE_PHASEA_CHECKPOINT:-data/validation/chemaware_online_role_budget_native/run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt}"
FULLMSG="${NOISE_CHEMAWARE_FULLMSG_CHECKPOINT:-data/validation/noise_chemaware_full_massspecgym_native/run_2401890/checkpoint/noise_chemaware_fullmsg_slim.pt}"
MSG_ROOT="${NOISE_MSG_FUSION_MODEL:-data/validation/noise_msg_fusion_run_2349961/fusion_model}"
FULL23_PACKAGE="deploy/unified_additive_full23_v132"
MS_ENTROPY_WHEEL="third_party/noise_article_wheels/ms_entropy-1.5.2-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl"

required=(
  "$PREFLIGHT/report.json" "$PREFLIGHT/eligible_inventory.csv.gz"
  "$PREFLIGHT/pre_exclusion_q123_pairwise_verification_indices.npz" "$SOURCES"
  "$ARCH" "$OFFICIAL" "$NOISE" "$STAGE1" "$PHASEA" "$FULLMSG"
  "$MSG_ROOT/fusion_model.joblib" "$MSG_ROOT/report.json"
  "$FULL23_PACKAGE/manifest.json" "$FULL23_PACKAGE/assets/full23_v1_parent.pt"
  "$FULL23_PACKAGE/assets/full23_v131_query_centered_raw_penalty_readout.pt"
  "$MS_ENTROPY_WHEEL"
  tasks/materialize_gnps2_mega_frozen_scoring_axis.py
  tasks/score_gnps2_mega_frozen_full23.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
)
missing=0
for path in "${required[@]}"; do
  [[ -e "$path" ]] || { echo "GNPS2_MEGA_FROZEN_MISSING: $path" >&2; missing=1; }
done
[[ "$missing" == 0 ]] || exit 4

python -u tasks/materialize_gnps2_mega_frozen_scoring_axis.py \
  --preflight-dir "$PREFLIGHT" --source-manifest "$SOURCES" \
  --output-dir /tmp/gnps2_mega_preflight_unused --preflight-only
if [[ "${1:-}" == "--preflight" ]]; then
  echo "GNPS2_MEGA_FROZEN_DEPENDENCY_PREFLIGHT_PASS"
  exit 0
fi

source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

JOB_TOKEN="${SLURM_JOB_ID:-manual}"
OUT="data/validation/gnps2_mega_frozen_full23_stress_run_${JOB_TOKEN}"
BENCHMARK="$OUT/benchmark"
EMBED="$OUT/embeddings"
SCRATCH="${SLURM_TMPDIR:-/tmp}/gnps2_mega_frozen_${JOB_TOKEN}"
LOCAL_DEPS="$SCRATCH/deps"
[[ ! -e "$OUT" ]] || { echo "GNPS2_MEGA_FROZEN_OUTPUT_EXISTS: $OUT" >&2; exit 5; }
mkdir -p "$OUT" "$EMBED" "$SCRATCH" "$LOCAL_DEPS"

python -m pip install --no-deps --target "$LOCAL_DEPS" "$MS_ENTROPY_WHEEL"
export PYTHONPATH="$LOCAL_DEPS:$PYTHONPATH"
python -m py_compile tasks/materialize_gnps2_mega_frozen_scoring_axis.py \
  tasks/score_gnps2_mega_frozen_full23.py

python -u tasks/materialize_gnps2_mega_frozen_scoring_axis.py \
  --preflight-dir "$PREFLIGHT" --source-manifest "$SOURCES" \
  --output-dir "$BENCHMARK" --scope available_exclusion

python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$STAGE1" --output "$SCRATCH/chemaware_stage1_slim.pt"
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$PHASEA" --output "$SCRATCH/chemaware_phasea_slim.pt"

encode_one() {
  local label="$1" checkpoint="$2" output="$3"
  python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
    --benchmark "$BENCHMARK" --checkpoint "$checkpoint" \
    --architecture-checkpoint "$ARCH" --output "$output" --label "$label" \
    --batch-size 128 --expected-selected-rows 0 --expected-records 0 \
    --panels identity_disjoint formula_disjoint pairwise_verification \
    --panel-file "pairwise_verification=$BENCHMARK/panel_pairwise_verification.npz"
}

encode_one official_dreams "$OFFICIAL" "$EMBED/official_dreams.npz"
encode_one noise_v1 "$NOISE" "$EMBED/noise_v1.npz"
encode_one chemaware_stage1 "$SCRATCH/chemaware_stage1_slim.pt" "$EMBED/chemaware_stage1.npz"
encode_one chemaware_phasea "$SCRATCH/chemaware_phasea_slim.pt" "$EMBED/chemaware_phasea.npz"
encode_one noise_chemaware_fullmsg "$FULLMSG" "$EMBED/noise_chemaware_fullmsg.npz"

python -u tasks/score_gnps2_mega_frozen_full23.py \
  --benchmark "$BENCHMARK" \
  --embedding "official_dreams=$EMBED/official_dreams.npz" \
  --embedding "noise_v1=$EMBED/noise_v1.npz" \
  --embedding "chemaware_stage1=$EMBED/chemaware_stage1.npz" \
  --embedding "chemaware_phasea=$EMBED/chemaware_phasea.npz" \
  --embedding "noise_chemaware_fullmsg=$EMBED/noise_chemaware_fullmsg.npz" \
  --msg-fusion-model "$MSG_ROOT" --full23-package "$FULL23_PACKAGE" \
  --device cuda --pairwise-wse --output-dir "$OUT/scoring"

find "$OUT" -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > "$OUT/SHA256SUMS.txt"
echo "GNPS2_MEGA_FROZEN_FULL23_STRESS_COMPLETE: $OUT"
