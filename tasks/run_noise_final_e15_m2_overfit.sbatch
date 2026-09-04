#!/bin/bash
#SBATCH --job-name=noise_e15_m2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/noise_e15_m2_%j.out
#SBATCH --error=data/validation/noise_e15_m2_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

FOLD=4
M1_DIR="${M1_DIR:-data/validation/g8r_noise_final_e15_replay_calibration/fold_4_run_2327551}"
E14_DIR="${E14_DIR:-data/validation/g8r_noise_final_e14_crossfit_teacher/fold_4}"
INITIAL="${INITIAL:-data/validation/g8r_noise_final_e4a_direct/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_highlr_multifold/seed_20260830/fold_4/final_shared_encoder.pt}"
RUN_ROOT="data/validation/g8r_noise_final_e15_m2/fold_${FOLD}_run_${SLURM_JOB_ID}"
PANEL="$RUN_ROOT/panel"
RESULT="$RUN_ROOT/result"

[[ -f "$M1_DIR/report.json" ]] || { echo "Missing passing E15-M1 artifact: $M1_DIR" >&2; exit 3; }
[[ -f "$E14_DIR/selected_actions.csv.gz" ]] || { echo "Missing exact E14 actions: $E14_DIR" >&2; exit 4; }
[[ -f "$E14_DIR/risk_controls.csv.gz" ]] || { echo "Missing exact E14 risk controls: $E14_DIR" >&2; exit 5; }
[[ -f "$INITIAL" ]] || { echo "Missing mature shared encoder: $INITIAL" >&2; exit 6; }
[[ ! -e "$RUN_ROOT" ]] || { echo "Unexpected job-specific output collision: $RUN_ROOT" >&2; exit 7; }

echo "=== E15-M2 executable panel + shared-encoder overfit started: $(date) ==="
echo "E15-M2 output: $RUN_ROOT"
python -u tasks/test_noise_final_e15_m2.py
python -u tasks/build_noise_final_e15_m2_executable_panel.py \
  --m1-dir "$M1_DIR" --e14-dir "$E14_DIR" --outer-fold "$FOLD" \
  --output-dir "$PANEL"
python -u tasks/train_noise_final_e15_m2_overfit.py \
  --panel-dir "$PANEL" --initial-student-checkpoint "$INITIAL" \
  --outer-fold "$FOLD" --output-dir "$RESULT"
python -u tasks/validate_noise_final_e15_m2_overfit.py --output-dir "$RESULT"
echo "=== E15-M2 completed: $(date) ==="
