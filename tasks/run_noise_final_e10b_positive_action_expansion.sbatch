#!/bin/bash
#SBATCH --job-name=noise_e10b_expand
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/noise_e10b_expand_%j.out
#SBATCH --error=data/validation/noise_e10b_expand_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

STUDENT="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_0/final_shared_encoder.pt"
OUTPUT="data/validation/g8r_noise_final_e10b_positive_action_expansion"
[[ -f "$STUDENT" ]] || { echo "Missing mature E8 student: $STUDENT" >&2; exit 3; }
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite E10-B output: $OUTPUT" >&2; exit 4; }

echo "=== E10-B positive action expansion started: $(date) ==="
python -u tasks/test_noise_final_e10_positive_residual_matrix.py
python -u tasks/audit_noise_final_e10_positive_residual_matrix.py \
  --student-checkpoint "$STUDENT" \
  --output-dir "$OUTPUT" \
  --cell-set expanded \
  --positive-references 3 \
  --fragment-tolerance 0.02 \
  --minimum-reference-prevalence 0.67 \
  --maximum-transferred-peaks 5 \
  --batch-size 128 \
  --bootstrap-resamples 5000 \
  --seed 20260902 \
  --device cuda \
  --no-amp
python -u tasks/validate_noise_final_e10_positive_residual_matrix.py \
  --output-dir "$OUTPUT" --cell-set expanded
echo "=== E10-B positive action expansion completed: $(date) ==="
