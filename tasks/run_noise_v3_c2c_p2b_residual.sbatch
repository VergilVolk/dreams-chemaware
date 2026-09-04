#!/bin/bash
#SBATCH --job-name=noise_c2c_train
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/noise_c2c_train_%j.out
#SBATCH --error=data/validation/noise_c2c_train_%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
CACHE=data/validation/g8r_noise_v3_c2c_p2b_molecule_cache.npz
OUTPUT=data/validation/g8r_noise_v3_c2c_p2b_residual.json
[[ ! -e "$CACHE" && ! -e "${CACHE%.npz}.json" && ! -e "$OUTPUT" ]] || { echo 'Refusing overwrite' >&2;exit 2; }
for f in data/validation/g8r_noise_v3_c2b_token_pair_cache.npz data/validation/g8r_p2b_rank_fusion.json;do [[ -f "$f" ]]||{ echo "Missing $f" >&2;exit 3;};done
echo "=== C2-C P2b-anchored nonlinear residual started: $(date) ==="
python -u tasks/build_noise_v3_c2c_p2b_molecule_cache.py --output "$CACHE"
python -u tasks/train_noise_v3_c2c_p2b_residual.py --cache "$CACHE" --output "$OUTPUT" --device cuda
echo "=== C2-C completed: $(date) ==="
