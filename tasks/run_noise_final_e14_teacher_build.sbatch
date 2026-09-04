#!/bin/bash
#SBATCH --job-name=noise_e14_teacher
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --output=data/validation/noise_e14_teacher_%A_%a.out
#SBATCH --error=data/validation/noise_e14_teacher_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

FOLD="${SLURM_ARRAY_TASK_ID}"
MATURE_ROOT="data/validation/g8r_noise_final_e4a_direct/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_highlr_multifold/seed_20260830"
CHECKPOINT="$MATURE_ROOT/fold_${FOLD}/final_shared_encoder.pt"
OUTPUT="data/validation/g8r_noise_final_e14_crossfit_teacher/fold_${FOLD}"

[[ -f "$CHECKPOINT" ]] || { echo "Missing mature fold checkpoint: $CHECKPOINT" >&2; exit 3; }
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite E14 teacher: $OUTPUT" >&2; exit 4; }

echo "=== E14 outer-clean teacher fold=${FOLD} started: $(date) ==="
python -u tasks/test_noise_final_e14_crossfit_teacher.py
python -u tasks/build_noise_final_e14_crossfit_p_teacher.py \
  --student-checkpoint "$CHECKPOINT" \
  --outer-fold "$FOLD" \
  --output-dir "$OUTPUT" \
  --batch-size 128 \
  --device cuda \
  --no-amp
python -u tasks/validate_noise_final_e14_crossfit_teacher.py --output-dir "$OUTPUT"
echo "=== E14 outer-clean teacher fold=${FOLD} completed: $(date) ==="
