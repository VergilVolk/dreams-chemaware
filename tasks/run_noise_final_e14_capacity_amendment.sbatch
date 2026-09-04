#!/bin/bash
#SBATCH --job-name=noise_e14_cap
#SBATCH --partition=gpu
#SBATCH --array=4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/noise_e14_capacity_%A_%a.out
#SBATCH --error=data/validation/noise_e14_capacity_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

FOLD="${SLURM_ARRAY_TASK_ID}"
OUTPUT_DIR="data/validation/g8r_noise_final_e14_crossfit_teacher/fold_${FOLD}"
AMENDMENT="$OUTPUT_DIR/capacity_amendment.json"

[[ -f "$OUTPUT_DIR/report.json" ]] || {
  echo "Missing completed E14 teacher report: $OUTPUT_DIR/report.json" >&2; exit 3;
}
[[ -f "$OUTPUT_DIR/selected_actions.csv.gz" ]] || {
  echo "Missing E14 selected actions" >&2; exit 4;
}
[[ -f "$OUTPUT_DIR/action_outcomes.npz" ]] || {
  echo "Missing E14 action outcomes" >&2; exit 5;
}
[[ ! -e "$AMENDMENT" ]] || {
  echo "Refusing to overwrite E14 capacity amendment: $AMENDMENT" >&2; exit 6;
}

echo "=== E14 capacity amendment fold=${FOLD} started: $(date) ==="
python -u tasks/test_noise_final_e14_crossfit_teacher.py
python -u tasks/audit_noise_final_e14_capacity_amendment.py \
  --output-dir "$OUTPUT_DIR" \
  --bootstrap-resamples 5000 \
  --seed 20260906
python -u tasks/validate_noise_final_e14_crossfit_teacher.py \
  --output-dir "$OUTPUT_DIR"
echo "=== E14 capacity amendment fold=${FOLD} completed: $(date) ==="
