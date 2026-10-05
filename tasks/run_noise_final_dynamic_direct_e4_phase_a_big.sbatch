#!/bin/bash
#SBATCH --job-name=noise_dd_e4_big
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/noise_dd_e4_big_%j.out
#SBATCH --error=data/validation/noise_dd_e4_big_%j.err

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
INITIAL="data/validation/g8r_noise_final_e4a_direct/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_highlr_multifold/seed_20260830/fold_${FOLD}/final_shared_encoder.pt"
RUN_ROOT="data/validation/g8r_noise_final_dynamic_direct_e4_phase_a/fold_${FOLD}_run_${SLURM_JOB_ID}"
PREFLIGHT="$RUN_ROOT/preflight"
LEDGER="$RUN_ROOT/ledger"
REPLAY="$RUN_ROOT/replay"
M2="$RUN_ROOT/current_crossfit"
SCHEDULE="$RUN_ROOT/schedule"
ARMS_ROOT="$RUN_ROOT/arms"
SUMMARY="$RUN_ROOT/summary"

[[ -f "$INITIAL" ]] || { echo "Missing mature fold-matched E4 checkpoint: $INITIAL" >&2; exit 3; }
[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to overwrite unique run: $RUN_ROOT" >&2; exit 4; }

echo "=== dynamic-direct mature-E4 full run started: $(date) ==="
python -u tasks/test_noise_final_dynamic_direct_core.py
python -u tasks/test_noise_final_dynamic_direct_preflight.py
python -u tasks/test_noise_final_dynamic_direct_ledger.py
python -u tasks/test_noise_final_dynamic_direct_replay.py
python -u tasks/test_noise_final_dynamic_direct_m2_crossfit.py
python -u tasks/test_noise_final_dynamic_direct_schedule.py
python -u tasks/test_noise_final_dynamic_direct_phase_a.py
python -u tasks/audit_noise_final_dynamic_direct_implementation.py

# Rebuild and replay every action in the mature E4 geometry. Labels from the
# earlier clean-duplicate geometry are never reused as final training weights.
python -u tasks/audit_noise_final_dynamic_direct_preflight.py \
  --outer-fold "$FOLD" --formula-fold-seed 20260825 \
  --initial-checkpoint "$INITIAL" \
  --initialization-contract mature_e4_current_replay \
  --output-dir "$PREFLIGHT" --hash-large-files
python -u tasks/validate_noise_final_dynamic_direct_preflight.py --output-dir "$PREFLIGHT"

python -u tasks/build_noise_final_dynamic_direct_ledger.py \
  --preflight-dir "$PREFLIGHT" --outer-fold "$FOLD" --seed "$SEED" \
  --output-dir "$LEDGER"
python -u tasks/validate_noise_final_dynamic_direct_ledger.py --output-dir "$LEDGER"

python -u tasks/audit_noise_final_dynamic_direct_replay.py \
  --preflight-dir "$PREFLIGHT" --ledger-dir "$LEDGER" \
  --clean-checkpoint "$INITIAL" --output-dir "$REPLAY" \
  --batch-size 64 --fp32-retry-batch-size 16 --actions-per-chunk 256 \
  --device cuda --amp
python -u tasks/validate_noise_final_dynamic_direct_replay.py --output-dir "$REPLAY"

python -u tasks/build_noise_final_dynamic_direct_m2_crossfit.py \
  --ledger-dir "$LEDGER" --replay-dir "$REPLAY" --outer-fold "$FOLD" \
  --seed "$SEED" --bootstrap-resamples 5000 --output-dir "$M2"
python -u tasks/validate_noise_final_dynamic_direct_m2_crossfit.py --output-dir "$M2"

python -u tasks/build_noise_final_dynamic_direct_schedule.py \
  --ledger-dir "$M2" --output-dir "$SCHEDULE" --seed "$SEED" \
  --actions-per-identity-family 1
python -u tasks/validate_noise_final_dynamic_direct_schedule.py --output-dir "$SCHEDULE"

# Executable GPU smoke gate covers clean, control and target payload paths
# before any four-epoch arm can spend the large training budget.
for ARM in clean_continuation matched_random dynamic_np; do
  python -u tasks/train_noise_final_dynamic_direct_phase_a.py \
    --m2-dir "$M2" --schedule-dir "$SCHEDULE" --preflight-dir "$PREFLIGHT" \
    --clean-checkpoint "$INITIAL" --output-dir "$RUN_ROOT/smoke/$ARM" \
    --arm "$ARM" --outer-fold "$FOLD" --formula-fold-seed 20260825 \
    --seed "$SEED" --epochs 1 --batch-queries 1 --unfreeze-blocks 1 \
    --backbone-lr 2e-6 --head-lr 1e-5 --eval-batch-size 96 \
    --device cuda --no-amp --smoke
done

# Four paired arms, one shared initialization and one identical schedule.
for ARM in clean_continuation matched_random static_target dynamic_np; do
  echo "=== Phase-A arm=$ARM started: $(date) ==="
  python -u tasks/train_noise_final_dynamic_direct_phase_a.py \
    --m2-dir "$M2" --schedule-dir "$SCHEDULE" --preflight-dir "$PREFLIGHT" \
    --clean-checkpoint "$INITIAL" --output-dir "$ARMS_ROOT/$ARM" \
    --arm "$ARM" --outer-fold "$FOLD" --formula-fold-seed 20260825 \
    --seed "$SEED" --epochs 4 --batch-queries 1 \
    --unfreeze-blocks 1 --backbone-lr 2e-6 --head-lr 1e-5 \
    --weight-decay 1e-4 --temperature 0.10 --margin-floor-slack 0.005 \
    --lambda-clean-rank 1.0 --lambda-action-rank 1.0 \
    --lambda-margin-floor 2.0 --lambda-preserve 5.0 --grad-clip 1.0 \
    --eval-batch-size 96 --device cuda --no-amp
  python -u tasks/validate_noise_final_dynamic_direct_phase_a.py \
    --output-dir "$ARMS_ROOT/$ARM"
  echo "=== Phase-A arm=$ARM completed: $(date) ==="
done

python -u tasks/summarize_noise_final_dynamic_direct_phase_a.py \
  --root "$ARMS_ROOT" --output-dir "$SUMMARY" \
  --bootstrap-resamples 10000 --seed "$SEED"
python -u tasks/validate_noise_final_dynamic_direct_phase_a.py \
  --output-dir "$SUMMARY" --summary
echo "=== dynamic-direct mature-E4 full run completed: $(date) ==="
