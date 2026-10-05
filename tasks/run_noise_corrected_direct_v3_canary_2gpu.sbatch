#!/bin/bash
#SBATCH --job-name=noise_v3_canary
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/noise_v3_canary_%j.out
#SBATCH --error=data/validation/noise_v3_canary_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[[ -n "${BASE_RUN:-}" ]] || {
  echo "BASE_RUN must name the completed formal N+P+A4 route directory" >&2
  exit 2
}
case "$BASE_RUN" in
  data/validation/noise_corrected_npa4_formal_fold_0_run_*|\
  data/validation/.noise_corrected_npa4_formal_fold_0_run_*.staging) ;;
  *) echo "Unexpected BASE_RUN path: $BASE_RUN" >&2; exit 2 ;;
esac

ALLOCATED_GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a ALLOCATED_GPUS <<< "$ALLOCATED_GPU_LIST"
if (( ${#ALLOCATED_GPUS[@]} < 2 )); then
  echo "Expected two allocated GPUs, got CUDA_VISIBLE_DEVICES=$ALLOCATED_GPU_LIST" >&2
  exit 3
fi
GPU_ZERO="${ALLOCATED_GPUS[0]}"
GPU_ONE="${ALLOCATED_GPUS[1]}"

FOLD=0
SEED=20260907
GRAPH="data/validation/noise_corrected_candidate_graph_v1_20260906"
EVAL_METADATA="data/validation/chemaware_corrected_candidate_manifest_v1"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
INITIAL="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_${FOLD}/final_shared_encoder.pt"
N_ROUTE="$BASE_RUN/routes/n_route"
P_ROUTE="$BASE_RUN/routes/p_route"
A4_ROUTE="$BASE_RUN/routes/a4_route"

FINAL_ROOT="data/validation/noise_corrected_direct_v3_canary_fold_${FOLD}_run_${SLURM_JOB_ID}"
RUN_ROOT="data/validation/.noise_corrected_direct_v3_canary_fold_${FOLD}_run_${SLURM_JOB_ID}.staging"
LEDGER="$RUN_ROOT/ledger"
PLAN_AUDIT="$RUN_ROOT/plan_audit"
ARMS="$RUN_ROOT/arms"
SUMMARY="$RUN_ROOT/summary"

for REQUIRED in \
  "$GRAPH/candidate_graph.npz" "$GRAPH/report.json" \
  "$EVAL_METADATA/manifest.npz" "$DATA" "$OFFICIAL" "$ARCHITECTURE" \
  "$INITIAL" "$(dirname "$INITIAL")/decision.json" \
  "$N_ROUTE/routed_actions.csv.gz" "$N_ROUTE/action_spectra.npz" "$N_ROUTE/report.json" \
  "$P_ROUTE/routed_actions.csv.gz" "$P_ROUTE/action_spectra.npz" "$P_ROUTE/report.json" \
  "$A4_ROUTE/routed_actions.csv.gz" "$A4_ROUTE/action_spectra.npz" "$A4_ROUTE/report.json"; do
  [[ -f "$REQUIRED" ]] || { echo "Missing frozen canary input: $REQUIRED" >&2; exit 4; }
done
[[ ! -e "$RUN_ROOT" && ! -e "$FINAL_ROOT" ]] || {
  echo "Refusing to overwrite unique atomic canary output" >&2
  exit 5
}
mkdir -p "$ARMS"

python -m py_compile \
  tasks/noise_corrected_action_routing.py \
  tasks/noise_corrected_action_routing_v3.py \
  tasks/noise_corrected_action_panel.py \
  tasks/noise_corrected_direct_v3_core.py \
  tasks/noise_corrected_action_expansion_v4.py \
  tasks/noise_corrected_transfer_objective_v4.py \
  tasks/noise_corrected_update_arbitration_v4.py \
  tasks/noise_corrected_directional_update_v4.py \
  tasks/noise_corrected_shuffled_control_v3.py \
  tasks/noise_corrected_fullgraph_evaluation.py \
  tasks/build_noise_corrected_routed_action_ledger.py \
  tasks/audit_noise_corrected_direct_v3_plan.py \
  tasks/train_noise_corrected_routed_direct.py \
  tasks/summarize_noise_corrected_direct_v3_canary.py
python -u tasks/test_noise_corrected_direct_v3_bundle.py
python -u tasks/test_noise_corrected_direct_v3_core.py
python -u tasks/test_noise_corrected_direct_v3_trainer.py
python -u tasks/test_noise_corrected_action_expansion_v4.py
python -u tasks/test_noise_corrected_transfer_objective_v4.py
python -u tasks/test_noise_corrected_update_arbitration_v4.py
python -u tasks/test_noise_corrected_directional_update_v4.py
python -u tasks/test_noise_corrected_fullgraph_evaluation.py
python -u tasks/test_summarize_noise_corrected_direct_v3_canary.py

echo "=== canary stage 1/3: rebuild and verify frozen full action ledger ==="
python -u tasks/build_noise_corrected_routed_action_ledger.py \
  --graph-dir "$GRAPH" --route-dirs "$N_ROUTE" "$P_ROUTE" "$A4_ROUTE" \
  --maximum-corrective-per-query 16 --maximum-risk-per-query 8 \
  --maximum-robust-per-query 8 --output-dir "$LEDGER"
python -u tasks/audit_noise_corrected_direct_v3_plan.py \
  --graph-dir "$GRAPH" --ledger-dir "$LEDGER" --outer-fold "$FOLD" \
  --formula-fold-seed 20260825 --batch-queries 4 --protective-batch-queries 8 \
  --maximum-protective-microbatches-per-step 4 \
  --maximum-auxiliary-microbatches-per-step 4 \
  --maximum-corrective-recycle-factor 4 \
  --maximum-corrective-per-query 16 --maximum-harmful-per-query 8 \
  --maximum-robust-per-query 8 --margin-transfer-fraction 0.50 \
  --margin-transfer-cap 0.10 --output-dir "$PLAN_AUDIT"

COMMON=(
  --graph-dir "$GRAPH" --source-manifest-dir "$EVAL_METADATA"
  --routed-ledger-dir "$LEDGER" --data "$DATA"
  --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE"
  --initial-student-checkpoint "$INITIAL" --outer-fold "$FOLD"
  --formula-fold-seed 20260825 --corrective-objective-mode v3_direct
  --development --epochs 1 --batch-queries 4 --protective-batch-queries 8
  --maximum-corrective-queries 512 --maximum-risk-queries 512
  --maximum-robust-queries 256 --maximum-clean-queries 4096
  --outer-held-eval-queries 4096
  --positive-references 4 --action-conditioned-positive-references 32
  --action-conditioned-negative-molecules 32 --negative-molecules 8
  --references-per-negative 2 --reference-refresh-every-epochs 0
  --n-highest-peaks 100 --unfreeze-blocks 1
  --backbone-lr 2e-6 --head-lr 1e-5 --weight-decay 1e-4
  --rank-margin 0.05 --temperature 0.10 --topk-negatives 8
  --action-safety-slack 0.005 --v3-harmful-damage-slack 0.005
  --lambda-margin-transfer 1.0 --lambda-action-rank 0.25
  --lambda-action-safety 0.25 --lambda-v3-corrective-consistency 0.25
  --lambda-v3-robust-rank 0.25 --lambda-v3-robust-floor 0.25
  --lambda-v3-harmful-rank 1.0 --lambda-v3-harmful-floor 1.0
  --margin-transfer-fraction 0.50 --margin-transfer-cap 0.10
  --transfer-target-allocation hard_cap --maximum-transfer-cap-factor 2.0
  --target-v3-payload-to-transfer-ratio 0.25
  --target-v3-consistency-to-transfer-ratio 0.25
  --target-v3-robust-to-transfer-ratio 0.10
  --target-v3-harmful-to-risk-ratio 0.25
  --v3-branch-scale-cap 16 --minimum-v3-internal-combination-retention 0.50
  --minimum-v3-active-transfer-fraction 0.05
  --v3-maximum-protective-microbatches-per-step 4
  --v3-maximum-auxiliary-microbatches-per-step 4
  --v3-maximum-corrective-recycle-factor 4 --v3-corrective-recycle-full-dose
  --lambda-clean-continuation 1.0 --lambda-risk-floor 2.0
  --lambda-preserve 5.0 --risk-margin-slack 0.005
  --calibration-batches 32 --target-corrective-to-risk-ratio 1.0
  --corrective-scale-cap 256 --minimum-corrective-scale 0.5
  --target-preclip-gradient-norm 0.05 --minimum-action-retention-p10 0.50
  --minimum-optimizer-action-alignment-p10 0.05
  --minimum-optimizer-action-attributable-fraction-p10 0.10
  --minimum-corrective-direction-retention-p10 0.10
  --optimizer-attribution-steps-per-epoch 16
  --maximum-clip-event-fraction 0.10 --grad-clip 1.0
  --progress-every-steps 25 --eval-batch-size 256
  --bootstrap-resamples 2000 --device cuda --no-amp --seed "$SEED"
)

echo "=== canary stage 2/3: routed and shuffled arms on two GPUs ==="
run_arm() {
  local GPU="$1"
  local ARM="$2"
  export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=8
  python -u tasks/train_noise_corrected_routed_direct.py \
    "${COMMON[@]}" --arm "$ARM" --output-dir "$ARMS/$ARM"
}
run_arm "$GPU_ZERO" routed_direct &
PID_ROUTED=$!
run_arm "$GPU_ONE" shuffled_action_control &
PID_SHUFFLED=$!
STATUS_ROUTED=0
STATUS_SHUFFLED=0
wait "$PID_ROUTED" || STATUS_ROUTED=$?
wait "$PID_SHUFFLED" || STATUS_SHUFFLED=$?
if (( STATUS_ROUTED != 0 || STATUS_SHUFFLED != 0 )); then
  echo "Canary routed/shuffled worker failed: $STATUS_ROUTED/$STATUS_SHUFFLED" >&2
  exit 6
fi

echo "=== canary stage 2b/3: clean control on GPU0 ==="
run_arm "$GPU_ZERO" clean_control

echo "=== canary stage 3/3: bounded full-metric comparison ==="
python -u tasks/summarize_noise_corrected_direct_v3_canary.py \
  --routed-dir "$ARMS/routed_direct" \
  --shuffled-dir "$ARMS/shuffled_action_control" \
  --clean-dir "$ARMS/clean_control" --seed "$SEED" \
  --bootstrap-resamples 2000 --output-dir "$SUMMARY"

mv "$RUN_ROOT" "$FINAL_ROOT"
echo "=== bounded direct-v3 canary preserved at: $FINAL_ROOT ==="
