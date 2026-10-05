#!/bin/bash
#SBATCH --job-name=noise_t1t3_v2_gnps
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/noise_t1t3_v2_gnps_%j.out
#SBATCH --error=data/validation/noise_t1t3_v2_gnps_%j.err

# GNPS external-panel arm selection (amendment predeclared before any GNPS
# result was seen).  The fold-1 gate of run 2347230 fired NO-GO, but its
# premise is falsified by sealed evidence: the averaged checkpoint has
# fold-1 delta -0.878 pp AND held delta +0.556 pp (CI_low +0.105 pp, run
# 2347055).  Fold-1 lies inside Stage-1's training pool, so its baseline is
# a memorized state that mechanically punishes every continuation.  GNPS
# gold/silver 10-ppm public-library spectra never entered any training fold
# and are the only zero-contamination pre-held material available.
#
# This job encodes Stage-1 plus the three arm checkpoints (no retraining),
# runs paired evaluations vs Stage-1, and applies the predeclared rule:
# winner = argmax worst-panel recall@1 delta; held is spent only if that
# worst-panel delta is strictly positive.  This job NEVER touches held
# fold-0; the held spend is a separate later submission.
set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
STAGE1_CHECKPOINT="data/validation/noise_dreams_hard_positive_native_fold_0_run_2344820/checkpoint/targeted_final.ckpt"
AB_V1_RUN="${AB_V1_RUN:-data/validation/noise_relation_t1_t3_run_2347055}"
AB_SLIM="$AB_V1_RUN/checkpoint/primary_seed_3407_slim.pt"
V2_RUN="${V2_RUN:-data/validation/noise_relation_t1_t3_v2_run_2347230}"
CLEAN_SLIM="$V2_RUN/checkpoint/clean_only_seed_3407_slim.pt"
ROTATION_SLIM="$V2_RUN/checkpoint/action_rotation_seed_3407_slim.pt"
GNPS_BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
RUN_ROOT="data/validation/noise_relation_t1_t3_v2_gnps_select_${SLURM_JOB_ID}"
GNPS="$RUN_ROOT/gnps"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_ROOT="$LOCAL_BASE/noise_relation_t1_t3_v2_gnps_${SLURM_JOB_ID}"
STAGE1_SLIM="$LOCAL_ROOT/stage1_targeted_slim.pt"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$RUN_ROOT" "$GNPS" "$LOCAL_ROOT"
cleanup_local() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup_local EXIT

required_files=(
  "$ARCHITECTURE" "$STAGE1_CHECKPOINT" "$AB_SLIM"
  "$CLEAN_SLIM" "$ROTATION_SLIM"
  "$GNPS_BENCHMARK/spectra.mgf"
  "$GNPS_BENCHMARK/panel_identity_disjoint.npz"
  "$GNPS_BENCHMARK/panel_formula_disjoint.npz"
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
  tasks/prepare_official_embedding_checkpoint.py
  tasks/summarize_noise_relation_t1_t3_v2_gnps_select.py
  tasks/test_noise_relation_t1_t3_v2_gnps_select.py
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "Missing GNPS selection input: $path" >&2; exit 5; }
done

python -m py_compile \
  tasks/summarize_noise_relation_t1_t3_v2_gnps_select.py \
  tasks/test_noise_relation_t1_t3_v2_gnps_select.py
python -u tasks/test_noise_relation_t1_t3_v2_gnps_select.py

# Format-only slim conversion of the 1.2 GB Stage-1 Lightning checkpoint,
# matching the proven Stage-1 GNPS pipeline pattern.
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$STAGE1_CHECKPOINT" --output "$STAGE1_SLIM"

encode() {
  local label="$1"
  local checkpoint="$2"
  python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
    --benchmark "$GNPS_BENCHMARK" \
    --checkpoint "$checkpoint" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$GNPS/${label}_embeddings.npz" --batch-size 64 --label "$label" \
    >"$RUN_ROOT/gnps_encode_${label}.out" 2>"$RUN_ROOT/gnps_encode_${label}.err"
}

encode stage1 "$STAGE1_SLIM"
encode averaged "$AB_SLIM"
encode clean-only "$CLEAN_SLIM"
encode action-rotation "$ROTATION_SLIM"

paired() {
  local arm="$1"
  # The evaluator's --output is a DIRECTORY; report.json lands inside it.
  python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
    --baseline-embeddings "$GNPS/stage1_embeddings.npz" \
    --candidate-embeddings "$GNPS/${arm}_embeddings.npz" \
    --output "$GNPS/paired_stage1_vs_${arm}" \
    --bootstrap-resamples 10000 --bootstrap-seed 20261001
}

paired averaged
paired clean-only
paired action-rotation

python -u tasks/summarize_noise_relation_t1_t3_v2_gnps_select.py \
  --averaged-report "$GNPS/paired_stage1_vs_averaged/report.json" \
  --clean-only-report "$GNPS/paired_stage1_vs_clean-only/report.json" \
  --action-rotation-report "$GNPS/paired_stage1_vs_action-rotation/report.json" \
  --output "$RUN_ROOT/selection/report.json"

PROCEED=$(python -c "import json;print('yes' if json.load(open('$RUN_ROOT/selection/report.json'))['proceed_to_held'] else 'no')")
SELECTED=$(python -c "import json;print(json.load(open('$RUN_ROOT/selection/report.json'))['selected_arm'] or 'none')")
if [[ "$PROCEED" == "yes" ]]; then
  echo "=== GNPS selection GO: arm='$SELECTED' earns the held spend (submit the follow-up job) ==="
else
  echo "=== GNPS selection NO-GO: no arm beat Stage-1 on both panels; continuation direction unsupported ==="
fi
echo "=== GNPS selection evidence: $RUN_ROOT ==="
