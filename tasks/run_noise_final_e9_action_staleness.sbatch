#!/bin/bash
#SBATCH --job-name=noise_e9_stale
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=data/validation/noise_e9_stale_%j.out
#SBATCH --error=data/validation/noise_e9_stale_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

STUDENT="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_0/final_shared_encoder.pt"
OUTPUT="data/validation/g8r_noise_final_e9_action_staleness"
[[ -f "$STUDENT" ]] || { echo "Missing mature E8 student: $STUDENT" >&2; exit 3; }
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite E9 output: $OUTPUT" >&2; exit 4; }

echo "=== E9 action-staleness audit started: $(date) ==="
python -u tasks/test_noise_final_e9_action_staleness.py
python -u tasks/audit_noise_final_e9_action_staleness.py \
  --student-checkpoint "$STUDENT" \
  --output-dir "$OUTPUT" \
  --outer-fold 0 \
  --selection-batch-size 32 \
  --encode-batch-size 128 \
  --top-k-negatives 8 \
  --softmax-temperature 0.10 \
  --fragment-tolerance 0.02 \
  --bootstrap-resamples 5000 \
  --seed 20260831 \
  --device cuda \
  --no-amp
python -u tasks/validate_noise_final_e9_action_staleness.py --output-dir "$OUTPUT"
echo "=== E9 action-staleness audit completed: $(date) ==="
