#!/bin/bash
#SBATCH --job-name=noise_e12a_reach
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=data/validation/noise_e12a_reach_%j.out
#SBATCH --error=data/validation/noise_e12a_reach_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

STUDENT="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_0/final_shared_encoder.pt"
OUTPUT="data/validation/g8r_noise_final_e12a_residual_reachability"
[[ -f "$STUDENT" ]] || { echo "Missing mature E8 student" >&2; exit 3; }
[[ -f data/validation/g8r_noise_final_e11_reference_diversity/report.json ]] || { echo "Missing E11" >&2; exit 4; }
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite E12-A output" >&2; exit 5; }

echo "=== E12-A residual reachability started: $(date) ==="
python -u tasks/test_noise_final_e12a_residual_reachability.py
python -u tasks/audit_noise_final_e12a_residual_reachability.py \
  --student-checkpoint "$STUDENT" \
  --output-dir "$OUTPUT" \
  --fragment-tolerance 0.02 \
  --batch-size 128 \
  --device cuda \
  --no-amp
python -u tasks/validate_noise_final_e12a_residual_reachability.py --output-dir "$OUTPUT"
echo "=== E12-A residual reachability completed: $(date) ==="
