#!/bin/bash
#SBATCH --job-name=noise_dd_m2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/noise_dd_m2_%j.out
#SBATCH --error=data/validation/noise_dd_m2_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

FOLD=0
SOURCE_JOB="${SOURCE_JOB:?submit with sbatch --export=ALL,SOURCE_JOB=<successful-M1-jobid> tasks/run_noise_final_dynamic_direct_m2.sbatch}"
LEDGER="data/validation/g8r_noise_final_dynamic_direct_ledger/fold_${FOLD}_run_${SOURCE_JOB}"
REPLAY="data/validation/g8r_noise_final_dynamic_direct_replay/fold_${FOLD}_run_${SOURCE_JOB}"
OUTPUT="data/validation/g8r_noise_final_dynamic_direct_m2/fold_${FOLD}_run_${SLURM_JOB_ID}"
SCHEDULE="data/validation/g8r_noise_final_dynamic_direct_schedule/fold_${FOLD}_run_${SLURM_JOB_ID}"

echo "=== dynamic direct M2 current-geometry crossfit started: $(date) ==="
python -u tasks/test_noise_final_dynamic_direct_core.py
python -u tasks/test_noise_final_dynamic_direct_ledger.py
python -u tasks/test_noise_final_dynamic_direct_m2_crossfit.py
python -u tasks/test_noise_final_dynamic_direct_schedule.py
python -u tasks/validate_noise_final_dynamic_direct_ledger.py --output-dir "$LEDGER"
python -u tasks/validate_noise_final_dynamic_direct_replay.py --output-dir "$REPLAY"
python -u tasks/build_noise_final_dynamic_direct_m2_crossfit.py \
  --ledger-dir "$LEDGER" --replay-dir "$REPLAY" --outer-fold "$FOLD" \
  --seed 20260904 --bootstrap-resamples 5000 --output-dir "$OUTPUT"
python -u tasks/validate_noise_final_dynamic_direct_m2_crossfit.py --output-dir "$OUTPUT"
python -u tasks/build_noise_final_dynamic_direct_schedule.py \
  --ledger-dir "$OUTPUT" --output-dir "$SCHEDULE" --seed 20260904 \
  --actions-per-identity-family 1
python -u tasks/validate_noise_final_dynamic_direct_schedule.py --output-dir "$SCHEDULE"
echo "=== dynamic direct M2 current-geometry crossfit completed: $(date) ==="
