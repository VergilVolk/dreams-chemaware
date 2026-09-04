#!/bin/bash
#SBATCH --job-name=noise_e15_m0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/noise_e15_m0_%j.out
#SBATCH --error=data/validation/noise_e15_m0_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

FOLD=4
E14="data/validation/g8r_noise_final_e14_crossfit_teacher/fold_${FOLD}"
OUTPUT="data/validation/g8r_noise_final_e15_multi_action/fold_${FOLD}"
[[ -f "data/validation/g8r_noise_final_e15_preflight.json" ]] || {
  echo "Missing formal E15 implementation audit" >&2
  exit 3
}
[[ -f "$E14/action_outcomes.npz" ]] || {
  echo "Missing complete E14 action outcomes: $E14" >&2
  exit 4
}
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite E15 multi-action ledger: $OUTPUT" >&2
  exit 5
}

echo "=== E15-M0 multi-action ledger started: $(date) ==="
python -u tasks/test_noise_final_e15_preflight.py
python -u tasks/test_noise_final_e15_multi_action.py
python -u tasks/audit_noise_final_e15_preflight.py \
  --output "data/validation/g8r_noise_final_e15_preflight_m0_${SLURM_JOB_ID}.json"
python -u tasks/build_noise_final_e15_multi_action_ledger.py \
  --e14-dir "$E14" \
  --outer-fold "$FOLD" \
  --maximum-corrective-actions 4 \
  --maximum-harmful-actions 4 \
  --output-dir "$OUTPUT"
python -u tasks/validate_noise_final_e15_multi_action_ledger.py --output-dir "$OUTPUT"
echo "=== E15-M0 multi-action ledger completed: $(date) ==="
