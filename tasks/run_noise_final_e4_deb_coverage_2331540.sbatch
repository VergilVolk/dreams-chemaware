#!/bin/bash
#SBATCH --job-name=e4deb_cover
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=data/validation/noise_e4_deb_coverage_2331540_%j.out
#SBATCH --error=data/validation/noise_e4_deb_coverage_2331540_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SOURCE_RUN="data/validation/g8r_noise_final_dynamic_direct_e4_phase_a_final/fold_0_run_2331284"
RUN_ROOT="data/validation/g8r_noise_final_e4_deb/fold_0_run_2331540"
OUTPUT="$RUN_ROOT/coverage_audit_${SLURM_JOB_ID}"

echo "=== E4-DEB job 2331540 coverage-only audit started: $(date) ==="
python -m py_compile \
  tasks/audit_noise_final_e4_deb_action_coverage.py \
  tasks/test_noise_final_e4_deb_action_coverage.py
python -u tasks/test_noise_final_e4_deb_action_coverage.py
python -u tasks/audit_noise_final_e4_deb_action_coverage.py \
  --manifest-dir "$RUN_ROOT/manifest" \
  --ledger-dir "$SOURCE_RUN/ledger" \
  --replay-dir "$SOURCE_RUN/replay" \
  --output-dir "$OUTPUT" --advantage-threshold 0.01
echo "Coverage report: $OUTPUT/report.json"
echo "=== E4-DEB job 2331540 coverage-only audit completed: $(date) ==="
