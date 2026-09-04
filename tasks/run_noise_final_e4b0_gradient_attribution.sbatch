#!/bin/bash
#SBATCH --job-name=noise_e4b0_grad
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/noise_e4b0_grad_%j.out
#SBATCH --error=data/validation/noise_e4b0_grad_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT=data/validation/g8r_noise_final_e4b0_gradient_attribution
REQUIRED=(
  tasks/audit_noise_final_e4b0_gradient_attribution.py
  tasks/test_noise_final_e4b0_gradient_attribution.py
  tasks/validate_noise_final_e4b0_gradient_attribution.py
)
for path in "${REQUIRED[@]}"; do
  [[ -f "$path" ]] || {
    echo "Missing required E4-B0 file: $path" >&2
    exit 2
  }
done

echo "=== E4-B0 no-update gradient attribution started: $(date) ==="
python -u tasks/test_noise_final_e4b0_gradient_attribution.py

if [[ -e "$OUTPUT" ]]; then
  python -u tasks/validate_noise_final_e4b0_gradient_attribution.py \
    --output-dir "$OUTPUT"
  echo "Reused already-complete E4-B0 output: $OUTPUT"
else
  python -u tasks/audit_noise_final_e4b0_gradient_attribution.py \
    --output-dir "$OUTPUT" \
    --outer-fold 0 --panel-formulas 32 \
    --positive-spectra 4 --negative-molecules 8 \
    --unfreeze-blocks 1 --rank-margin 0.05 \
    --specificity-margin 0.01 --temperature 0.10 \
    --grad-clip 1.0 --bootstrap-resamples 5000 \
    --seed 20260901 --device cuda
  python -u tasks/validate_noise_final_e4b0_gradient_attribution.py \
    --output-dir "$OUTPUT"
fi

echo "=== E4-B0 no-update gradient attribution completed: $(date) ==="
