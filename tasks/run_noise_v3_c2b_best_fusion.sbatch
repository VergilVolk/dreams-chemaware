#!/bin/bash
#SBATCH --job-name=noise_c2b_best
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/noise_c2b_best_%j.out
#SBATCH --error=data/validation/noise_c2b_best_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
OUTPUT="data/validation/g8r_noise_v3_c2b_best_fusion_v2.json"
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite $OUTPUT" >&2; exit 2; }
for path in data/validation/g8r_noise_v3_c2b_token_pair_cache.npz data/validation/g8r_noise_v3_c2b_token_pair_cache.json data/validation/g8r_p2b_rank_fusion.json; do
  [[ -f "$path" ]] || { echo "Missing best-fusion dependency: $path" >&2; exit 3; }
done
python -u tasks/test_noise_v3_c2b_best_fusion.py
echo "=== Noise-v3 C2-B current-best fusion started: $(date) ==="
python -u tasks/audit_noise_v3_c2b_best_fusion.py --output "$OUTPUT"
echo "=== Noise-v3 C2-B current-best fusion completed: $(date) ==="
