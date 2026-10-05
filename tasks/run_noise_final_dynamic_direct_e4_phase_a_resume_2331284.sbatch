#!/bin/bash
#SBATCH --job-name=noise_dd_resume
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/noise_dd_resume_2331284_%j.out
#SBATCH --error=data/validation/noise_dd_resume_2331284_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FOLD=0
SEED=20260904
EPOCHS=4
INITIAL="data/validation/g8r_noise_final_e4a_direct/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_highlr_multifold/seed_20260830/fold_${FOLD}/final_shared_encoder.pt"
RUN_ROOT="data/validation/g8r_noise_final_dynamic_direct_e4_phase_a_final/fold_0_run_2331284"
PREFLIGHT="$RUN_ROOT/preflight"
LEDGER="$RUN_ROOT/ledger"
REPLAY="$RUN_ROOT/replay"
M2="$RUN_ROOT/current_crossfit"
SCHEDULE="$RUN_ROOT/schedule"
ARMS_ROOT="$RUN_ROOT/arms"
SUMMARY="$RUN_ROOT/summary"

echo "=== final conditional direct shared-embedding resume for job 2331284 started: $(date) ==="

# Cheap regression tests run before any model allocation.
python -u tasks/test_noise_final_dynamic_direct_schedule.py
python -u tasks/test_noise_final_dynamic_direct_phase_a.py
python -u tasks/audit_noise_final_dynamic_direct_implementation.py

# Read-only verification of every reused artifact. No expensive replay is repeated.
[[ -f "$INITIAL" ]] || { echo "Missing mature E4 checkpoint: $INITIAL" >&2; exit 3; }
python -u tasks/validate_noise_final_dynamic_direct_preflight.py --output-dir "$PREFLIGHT"
python -u tasks/validate_noise_final_dynamic_direct_ledger.py --output-dir "$LEDGER"
python -u tasks/validate_noise_final_dynamic_direct_replay.py --output-dir "$REPLAY"
python -u tasks/validate_noise_final_dynamic_direct_m2_crossfit.py --output-dir "$M2"

# The failed serializer used an atomic staging directory, so no completed
# schedule should exist. Refuse ambiguous or partially published state.
[[ ! -e "$SCHEDULE" ]] || {
  echo "Refusing to overwrite existing schedule: $SCHEDULE" >&2
  exit 4
}
[[ ! -e "$ARMS_ROOT" ]] || {
  echo "Refusing to mix with existing formal arms: $ARMS_ROOT" >&2
  exit 5
}
[[ ! -e "$SUMMARY" ]] || {
  echo "Refusing to overwrite existing summary: $SUMMARY" >&2
  exit 6
}

python -u tasks/build_noise_final_dynamic_direct_schedule.py \
  --ledger-dir "$M2" --output-dir "$SCHEDULE" --seed "$SEED" \
  --epochs "$EPOCHS" --actions-per-identity-family 1 \
  --action-dose-curriculum 0.35 0.45 0.55 0.55 \
  --maximum-action-dose 0.65
python -u tasks/validate_noise_final_dynamic_direct_schedule.py --output-dir "$SCHEDULE"

# Execute every payload and loss path before the formal training budget.
for ARM in clean_continuation matched_random static_target dynamic_np; do
  python -u tasks/train_noise_final_dynamic_direct_phase_a.py \
    --m2-dir "$M2" --schedule-dir "$SCHEDULE" --preflight-dir "$PREFLIGHT" \
    --clean-checkpoint "$INITIAL" --output-dir "$RUN_ROOT/smoke/$ARM" \
    --arm "$ARM" --outer-fold "$FOLD" --formula-fold-seed 20260825 \
    --seed "$SEED" --epochs 1 --batch-queries 1 --unfreeze-blocks 1 \
    --backbone-lr 2e-6 --head-lr 1e-5 --eval-batch-size 96 \
    --device cuda --no-amp --smoke
done

# Exact paired four-arm Phase-A experiment.
for ARM in clean_continuation matched_random static_target dynamic_np; do
  echo "=== final Phase-A arm=$ARM started: $(date) ==="
  python -u tasks/train_noise_final_dynamic_direct_phase_a.py \
    --m2-dir "$M2" --schedule-dir "$SCHEDULE" --preflight-dir "$PREFLIGHT" \
    --clean-checkpoint "$INITIAL" --output-dir "$ARMS_ROOT/$ARM" \
    --arm "$ARM" --outer-fold "$FOLD" --formula-fold-seed 20260825 \
    --seed "$SEED" --epochs "$EPOCHS" --batch-queries 1 \
    --unfreeze-blocks 1 --backbone-lr 2e-6 --head-lr 1e-5 \
    --weight-decay 1e-4 --temperature 0.10 --margin-floor-slack 0.005 \
    --lambda-clean-rank 1.0 --lambda-action-rank 1.0 \
    --lambda-margin-floor 2.0 --lambda-preserve 5.0 --grad-clip 1.0 \
    --eval-batch-size 96 --device cuda --no-amp
  python -u tasks/validate_noise_final_dynamic_direct_phase_a.py \
    --output-dir "$ARMS_ROOT/$ARM"
  echo "=== final Phase-A arm=$ARM completed: $(date) ==="
done

python -u tasks/summarize_noise_final_dynamic_direct_phase_a.py \
  --root "$ARMS_ROOT" --output-dir "$SUMMARY" \
  --bootstrap-resamples 10000 --seed "$SEED"
python -u tasks/validate_noise_final_dynamic_direct_phase_a.py \
  --output-dir "$SUMMARY" --summary

echo "=== final conditional direct shared-embedding resume for job 2331284 completed: $(date) ==="
