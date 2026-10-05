#!/bin/bash
#SBATCH --job-name=noise_t1t3_v3
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/noise_t1t3_v3_%j.out
#SBATCH --error=data/validation/noise_t1t3_v3_%j.err

# Query-balanced exact-bridge train-side screen.  This job deliberately stops
# after fold-1.  A positive targeted-vs-control result is required before a
# separate outer-held/GNPS job is allowed.
set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
if (( ${#ALLOCATED_GPUS[@]} != 2 )); then
  echo "Expected exactly two GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
  exit 3
fi

DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
STAGE1="data/validation/noise_dreams_hard_positive_native_fold_0_run_2344820"
WARM="$STAGE1/checkpoint/targeted_final.ckpt"
ACTION_BANK="$STAGE1/triplets/action_spectra.npz"
V2_RUN="${V2_RUN:-data/validation/noise_relation_t1_t3_v2_run_2347230}"
CORPUS="$V2_RUN/corpus"
RUN_ROOT="data/validation/noise_relation_t1_t3_v3_run_${SLURM_JOB_ID}"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_ROOT="$LOCAL_BASE/noise_relation_t1_t3_v3_${SLURM_JOB_ID}"
TARGETED="$LOCAL_ROOT/targeted"
CONTROL="$LOCAL_ROOT/control"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
required_files=(
  "$DATA" "$ARCHITECTURE" "$WARM" "$ACTION_BANK"
  "$CORPUS/report.json" "$CORPUS/train.npz" "$CORPUS/validation.npz"
  "$CORPUS/action_events.npz"
  tasks/noise_relation_t1_t3_core.py
  tasks/noise_relation_t1_t3_v3_core.py
  tasks/train_noise_relation_t1_t3_v3.py
  tasks/summarize_noise_relation_t1_t3_v3_fold1.py
  tasks/test_noise_relation_t1_t3_v3.py
  tasks/test_noise_relation_t1_t3_v3_sbatch.py
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "Missing V3 input: $path" >&2; exit 5; }
done
mkdir -p "$RUN_ROOT" "$LOCAL_ROOT"
cleanup_local() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup_local EXIT

python -m py_compile \
  tasks/noise_relation_t1_t3_v3_core.py \
  tasks/train_noise_relation_t1_t3_v3.py \
  tasks/summarize_noise_relation_t1_t3_v3_fold1.py
python -u tasks/test_noise_relation_t1_t3_v3.py
python -u tasks/test_noise_relation_t1_t3_v3_sbatch.py

run_arm() {
  local gpu="$1"
  local arm="$2"
  local output="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/train_noise_relation_t1_t3_v3.py \
    --data "$DATA" --corpus "$CORPUS" --action-bank "$ACTION_BANK" \
    --warm-start-checkpoint "$WARM" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$output" --seed 3407 --rotation 0 --arm "$arm" \
    --action-fraction 0.5 --lr 5e-6 --weight-decay 0 \
    --triplet-loss-margin 0.1 --listwise-temperature 0.07 \
    --n-highest-peaks 100 --maximum-spectra-per-step 64 \
    --maximum-queries-per-step 4 \
    >"$RUN_ROOT/train_${arm}.out" 2>"$RUN_ROOT/train_${arm}.err"
}

run_arm "${ALLOCATED_GPUS[0]}" targeted "$TARGETED" & PID_TARGETED=$!
run_arm "${ALLOCATED_GPUS[1]}" control "$CONTROL" & PID_CONTROL=$!
set +e
wait "$PID_TARGETED"; STATUS_TARGETED=$?
wait "$PID_CONTROL"; STATUS_CONTROL=$?
set -e
if (( STATUS_TARGETED != 0 || STATUS_CONTROL != 0 )); then
  tail -n 200 "$RUN_ROOT/train_targeted.err" >&2 || true
  tail -n 200 "$RUN_ROOT/train_control.err" >&2 || true
  echo "V3 arm failure: targeted=$STATUS_TARGETED control=$STATUS_CONTROL" >&2
  exit 6
fi

mkdir -p "$RUN_ROOT/training_reports" "$RUN_ROOT/fold1"
cp "$TARGETED/report.json" "$RUN_ROOT/training_reports/targeted_seed_3407.json"
cp "$CONTROL/report.json" "$RUN_ROOT/training_reports/control_seed_3407.json"

python -u tasks/summarize_noise_relation_t1_t3_v3_fold1.py \
  --targeted-report "$TARGETED/report.json" \
  --control-report "$CONTROL/report.json" \
  --output "$RUN_ROOT/fold1/selection.json"

PROCEED=$(python -c "import json;print('yes' if json.load(open('$RUN_ROOT/fold1/selection.json'))['proceed_to_outer_held'] else 'no')")
echo "=== V3 query-balanced exact-bridge fold-1 proceed=${PROCEED} ==="
echo "=== Result: $RUN_ROOT ==="
if [[ "$PROCEED" != "yes" ]]; then
  echo "=== NO-GO: outer held and GNPS remain untouched ==="
else
  # Persist only what a subsequent held/replicate job actually needs.  A
  # NO-GO leaves all large local checkpoints to the EXIT cleanup trap.
  mkdir -p "$RUN_ROOT/checkpoint"
  cp "$TARGETED/final_slim.pt" "$RUN_ROOT/checkpoint/targeted_seed_3407_slim.pt"
  cp "$CONTROL/final_slim.pt" "$RUN_ROOT/checkpoint/control_seed_3407_slim.pt"
  cp "$TARGETED/final_optimizer.pt" "$RUN_ROOT/checkpoint/targeted_seed_3407_optimizer.pt"
  echo "=== GO: retained targeted/control slim checkpoints and targeted Adam state ==="
fi
