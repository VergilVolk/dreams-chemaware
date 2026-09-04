#!/bin/bash
#SBATCH --job-name=noise_e15_m1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/noise_e15_m1_%j.out
#SBATCH --error=data/validation/noise_e15_m1_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

FOLD=4
LEDGER="data/validation/g8r_noise_final_e15_multi_action/fold_${FOLD}"
E14="data/validation/g8r_noise_final_e14_crossfit_teacher/fold_${FOLD}"
OUTPUT="data/validation/g8r_noise_final_e15_replay_calibration/fold_${FOLD}_run_${SLURM_JOB_ID}"
[[ -f "$LEDGER/report.json" ]] || { echo "Missing formal E15-M0 ledger" >&2; exit 3; }
[[ -f "$E14/action_outcomes.npz" ]] || { echo "Missing complete E14 outcomes" >&2; exit 4; }
[[ ! -e "$OUTPUT" ]] || { echo "Unexpected job-specific output collision: $OUTPUT" >&2; exit 5; }

echo "=== E15-M1 source replay and calibration started: $(date) ==="
echo "E15-M1 output: $OUTPUT"
python -u tasks/test_noise_final_e15_replay_calibration.py
python -u tasks/audit_noise_final_e15_replay_calibration.py \
  --ledger-dir "$LEDGER" \
  --e14-dir "$E14" \
  --per-source-kind 16 \
  --output-dir "$OUTPUT"
python -u tasks/validate_noise_final_e15_replay_calibration.py --output-dir "$OUTPUT"
echo "=== E15-M1 source replay and calibration completed: $(date) ==="
