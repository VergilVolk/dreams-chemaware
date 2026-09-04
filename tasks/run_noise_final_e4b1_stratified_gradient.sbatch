#!/bin/bash
#SBATCH --job-name=noise_e4b1_strata
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=data/validation/noise_e4b1_strata_%j.out
#SBATCH --error=data/validation/noise_e4b1_strata_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT=data/validation/g8r_noise_final_e4b1_stratified_gradient
REQUIRED=(
  tasks/audit_noise_final_e4b1_stratified_gradient.py
  tasks/test_noise_final_e4b1_stratified_gradient.py
  tasks/validate_noise_final_e4b1_stratified_gradient.py
  data/validation/g8r_noise_final_e4b0_gradient_attribution/report.json
  data/validation/g8r_noise_final_r0_faithful_s3a/training_actions.csv.gz
  data/validation/g8r_real_error_analysis/query_error_signatures.csv.gz
)
for path in "${REQUIRED[@]}"; do
  [[ -f "$path" ]] || {
    echo "Missing required E4-B1 file: $path" >&2
    exit 2
  }
done

echo "=== E4-B1 stratified no-update gradient audit started: $(date) ==="
python -u tasks/test_noise_final_e4b1_stratified_gradient.py

if [[ -e "$OUTPUT" ]]; then
  python -u tasks/validate_noise_final_e4b1_stratified_gradient.py \
    --output-dir "$OUTPUT"
  echo "Reused already-complete E4-B1 output: $OUTPUT"
else
  python -u tasks/audit_noise_final_e4b1_stratified_gradient.py \
    --output-dir "$OUTPUT" --outer-fold 0 \
    --formulas-per-primary-stratum 32 \
    --positive-spectra 4 --negative-molecules 8 \
    --unfreeze-blocks 1 --rank-margin 0.05 \
    --specificity-margin 0.01 --temperature 0.10 \
    --grad-clip 1.0 --bootstrap-resamples 50000 \
    --seed 20260902 --device cuda
  python -u tasks/validate_noise_final_e4b1_stratified_gradient.py \
    --output-dir "$OUTPUT"
fi

echo "=== E4-B1 stratified no-update gradient audit completed: $(date) ==="
