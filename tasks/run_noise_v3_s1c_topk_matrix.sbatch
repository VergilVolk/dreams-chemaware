#!/bin/bash
#SBATCH --job-name=noise_v3_s1c
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/noise_v3_s1c_%j.out
#SBATCH --error=data/validation/noise_v3_s1c_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

OUTPUT="data/validation/g8r_noise_v3_s1c_topk_matrix"
HEADROOM="data/validation/g8r_noise_v3_s1c_action_headroom.json"

echo "=== Noise-v3 S1c expanded action matrix started: $(date) ==="
echo "=== stage 1/5: core unit tests ==="
python -u tasks/test_noise_v3_core.py

echo "=== stage 2/5: full candidate-graph preflight ==="
python -u tasks/check_noise_v3_candidate_gradient.py

echo "=== stage 3/5: top-5 gradient + top-3 confounder matrix ==="
python -u tasks/audit_noise_v3_candidate_gradient.py \
  --output-dir "$OUTPUT" \
  --device cuda \
  --gradient-batch-size 32 \
  --encode-batch-size 128 \
  --top-k-negatives 5 \
  --softmax-temperature 0.10 \
  --selection-attenuation 0.50 \
  --selectors candidate_gradient role_confounder \
  --candidate-gradient-top-k 5 \
  --role-confounder-top-k 3 \
  --attenuations 0.25 0.50 0.75 1.00 \
  --control-repeats 2 \
  --bootstrap 2000 \
  --seed 20260824

echo "=== stage 4/5: fail-closed matrix validation ==="
python -u tasks/validate_noise_v3_s1c_topk_matrix.py --output-dir "$OUTPUT"

echo "=== stage 5/5: no-op-aware expanded oracle ==="
python -u tasks/analyze_noise_v3_s1b_action_headroom.py \
  --matrix-dir "$OUTPUT" \
  --output "$HEADROOM" \
  --status noise_v3_s1c_action_headroom_complete \
  --minimum-headroom-pp 2.0 \
  --minimum-recoverable-queries 600
echo "=== Noise-v3 S1c expanded action matrix completed: $(date) ==="
