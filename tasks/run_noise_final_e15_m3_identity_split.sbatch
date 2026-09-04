#!/bin/bash
#SBATCH --job-name=noise_e15_m3_split
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/noise_e15_m3_split_%j.out
#SBATCH --error=data/validation/noise_e15_m3_split_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

M2_RUN="${M2_RUN:-data/validation/g8r_noise_final_e15_m2/fold_4_run_2327814}"
INITIAL="${INITIAL:-data/validation/g8r_noise_final_e4a_direct/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_highlr_multifold/seed_20260830/fold_4/final_shared_encoder.pt}"
OUTPUT="data/validation/g8r_noise_final_e15_m3_identity_split/run_${SLURM_JOB_ID}"

[[ -f "$M2_RUN/result/report.json" ]] || { echo "Missing passing M2 result: $M2_RUN" >&2; exit 3; }
[[ -f "$INITIAL" ]] || { echo "Missing mature E4-A initialization: $INITIAL" >&2; exit 4; }
[[ ! -e "$OUTPUT" ]] || { echo "Unexpected M3 split output collision: $OUTPUT" >&2; exit 5; }

echo "=== E15-M3 identity split started: $(date) ==="
python -u tasks/test_noise_final_e15_multi_action.py
python -u tasks/test_noise_final_e15_m2.py
python -u tasks/test_noise_final_e15_m3_identity_split.py
python -u tasks/build_noise_final_e15_m3_identity_split.py \
  --m2-run-dir "$M2_RUN" --initial-student-checkpoint "$INITIAL" \
  --output-dir "$OUTPUT"
python -u tasks/validate_noise_final_e15_m3_identity_split.py --split-dir "$OUTPUT"
echo "=== E15-M3 identity split completed: $(date) ==="
