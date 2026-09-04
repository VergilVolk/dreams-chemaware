#!/bin/bash
#SBATCH --job-name=noise_c2b_pair
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/noise_c2b_pair_%j.out
#SBATCH --error=data/validation/noise_c2b_pair_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

OUTPUT="data/validation/g8r_noise_v3_c2b_token_pair_cache.npz"
[[ ! -e "$OUTPUT" && ! -e "${OUTPUT%.npz}.json" ]] || {
  echo "Refusing to overwrite C2-B0 output" >&2; exit 2;
}
for path in \
  data/validation/g8r_error_atlas_listwise_cache.npz \
  data/validation/g8r_noise_v3_c2b_candidate_tokens/report.json \
  data/validation/g8r_noise_v3_c2b_candidate_tokens/rows.npy
do
  [[ -f "$path" ]] || { echo "Missing C2-B0 dependency: $path" >&2; exit 3; }
done

python -u tasks/test_noise_v3_c2b_token_pair_cache.py
echo "=== Noise-v3 C2-B0 token-pair evidence audit started: $(date) ==="
python -u tasks/build_noise_v3_c2b_token_pair_cache.py --output "$OUTPUT"
python -u tasks/validate_noise_v3_c2b_token_pair_cache.py --cache "$OUTPUT"
echo "=== Noise-v3 C2-B0 token-pair evidence audit completed: $(date) ==="
