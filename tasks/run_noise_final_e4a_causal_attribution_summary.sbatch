#!/bin/bash
#SBATCH --job-name=noise_e4a_causal_summary
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=data/validation/noise_e4a_causal_summary_%j.out
#SBATCH --error=data/validation/noise_e4a_causal_summary_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

RUN_SUFFIX=e4a_causal_v1_20260901
SEED=20260828
FOLD=0
RUN_ROOT=data/validation/g8r_noise_final_e4a_causal_attribution
SUMMARY_ROOT=data/validation/g8r_noise_final_e4a_causal_attribution_summary
SUMMARY_OUTPUT="$SUMMARY_ROOT/$RUN_SUFFIX/seed_${SEED}/fold_${FOLD}"

echo "=== E4-A causal attribution recovery summary started: $(date) ==="
python -u tasks/test_noise_final_e4a_causal_attribution.py

if [[ -e "$SUMMARY_OUTPUT" ]]; then
  python -u tasks/validate_noise_final_e4a_causal_attribution.py \
    --output-dir "$SUMMARY_OUTPUT"
  echo "Existing complete paired summary validated without overwrite: $SUMMARY_OUTPUT"
else
  python -u tasks/summarize_noise_final_e4a_causal_attribution.py \
    --run-root "$RUN_ROOT" --run-suffix "$RUN_SUFFIX" \
    --seed "$SEED" --outer-fold "$FOLD" \
    --bootstrap-resamples 5000 --output-dir "$SUMMARY_ROOT"
  python -u tasks/validate_noise_final_e4a_causal_attribution.py \
    --output-dir "$SUMMARY_OUTPUT"
fi
echo "=== E4-A causal attribution recovery summary completed: $(date) ==="
