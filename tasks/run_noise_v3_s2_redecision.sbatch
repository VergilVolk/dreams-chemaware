#!/bin/bash
#SBATCH --job-name=noise_v3_s2_fix
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/noise_v3_s2_redecision_%j.out
#SBATCH --error=data/validation/noise_v3_s2_redecision_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

OUTPUT="data/validation/g8r_noise_v3_s2_headroom_v2.json"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite $OUTPUT" >&2
  exit 2
fi

echo "=== Noise-v3 S2 corrected redecision started: $(date) ==="
python -u tasks/analyze_noise_v3_s2_headroom.py \
  --s1c-dir data/validation/g8r_noise_v3_s1c_topk_matrix \
  --s2-dir data/validation/g8r_noise_v3_s2_sequential \
  --cache data/validation/g8r_error_atlas_listwise_cache.npz \
  --output "$OUTPUT" \
  --minimum-headroom-pp 2.0 \
  --minimum-recoverable-queries 600 \
  --bootstrap 2000 \
  --seed 20260824
echo "=== Noise-v3 S2 corrected redecision completed: $(date) ==="
