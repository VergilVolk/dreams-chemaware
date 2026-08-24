#!/bin/bash
#SBATCH --job-name=noise_v3_s1a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/noise_v3_s1a_%j.out
#SBATCH --error=data/validation/noise_v3_s1a_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

OUTPUT="data/validation/g8r_noise_v3_s1a_single_peak_matrix"

echo "=== Noise-v3 S1a orthogonal matrix started: $(date) ==="
echo "=== stage 1/4: core unit tests ==="
python -u tasks/test_noise_v3_core.py

echo "=== stage 2/4: fail-closed full-graph preflight ==="
python -u tasks/check_noise_v3_candidate_gradient.py

echo "=== stage 3/4: preregistered 3-selector x 4-dose matrix ==="
python -u tasks/audit_noise_v3_candidate_gradient.py \
  --output-dir "$OUTPUT" \
  --device cuda \
  --gradient-batch-size 32 \
  --encode-batch-size 128 \
  --top-k-negatives 5 \
  --softmax-temperature 0.10 \
  --selection-attenuation 0.50 \
  --selectors candidate_gradient role_confounder role_identity \
  --attenuations 0.25 0.50 0.75 1.00 \
  --control-repeats 3 \
  --bootstrap 2000 \
  --seed 20260824

echo "=== stage 4/4: fail-closed matrix integrity validation ==="
python -u tasks/validate_noise_v3_s1a_matrix.py --output-dir "$OUTPUT"
echo "=== Noise-v3 S1a orthogonal matrix completed: $(date) ==="
