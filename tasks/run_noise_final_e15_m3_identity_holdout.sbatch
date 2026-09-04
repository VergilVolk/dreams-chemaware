#!/bin/bash
#SBATCH --job-name=noise_e15_m3_train
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=data/validation/noise_e15_m3_train_%j.out
#SBATCH --error=data/validation/noise_e15_m3_train_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

M2_RUN="${M2_RUN:-data/validation/g8r_noise_final_e15_m2/fold_4_run_2327814}"
INITIAL="${INITIAL:-data/validation/g8r_noise_final_e4a_direct/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_highlr_multifold/seed_20260830/fold_4/final_shared_encoder.pt}"
if [[ -z "${SPLIT_DIR:-}" ]]; then
  SPLIT_DIR="$(find data/validation/g8r_noise_final_e15_m3_identity_split -mindepth 2 -maxdepth 2 -name report.json -printf '%T@ %h\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
fi
OUTPUT="data/validation/g8r_noise_final_e15_m3_identity_holdout/run_${SLURM_JOB_ID}"

[[ -n "$SPLIT_DIR" && -f "$SPLIT_DIR/report.json" ]] || { echo "Missing passing M3 split" >&2; exit 3; }
[[ -f "$M2_RUN/result/report.json" ]] || { echo "Missing M2 result" >&2; exit 4; }
[[ -f "$INITIAL" ]] || { echo "Missing mature E4-A initialization" >&2; exit 5; }
[[ ! -e "$OUTPUT" ]] || { echo "Unexpected M3 output collision: $OUTPUT" >&2; exit 6; }

echo "=== E15-M3 paired identity-held training started: $(date) ==="
echo "M3 split: $SPLIT_DIR"
echo "M3 output: $OUTPUT"
python -u tasks/test_noise_final_e15_multi_action.py
python -u tasks/test_noise_final_e15_m2.py
python -u tasks/test_noise_final_e15_m3_identity_split.py
python -u tasks/test_noise_final_e15_m3_identity_holdout.py
python -u tasks/validate_noise_final_e15_m3_identity_split.py --split-dir "$SPLIT_DIR"
python -u tasks/train_noise_final_e15_m3_identity_holdout.py \
  --split-dir "$SPLIT_DIR" --m2-result-report "$M2_RUN/result/report.json" \
  --initial-student-checkpoint "$INITIAL" --output-dir "$OUTPUT"
python -u tasks/validate_noise_final_e15_m3_identity_holdout.py --output-dir "$OUTPUT"
echo "=== E15-M3 paired identity-held training completed: $(date) ==="
