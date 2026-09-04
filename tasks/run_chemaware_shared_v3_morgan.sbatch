#!/bin/bash
#SBATCH --job-name=chemaware_v3_morgan
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/chemaware_v3_morgan_%j.out
#SBATCH --error=data/validation/chemaware_v3_morgan_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

required=(
  data/validation/g8r_error_atlas_listwise_cache.npz
  data/validation/g8r_chemaware_shared_v2_preflight.json
  data/models/MassSpecGym_MurckoHist_split.hdf5
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "MISSING: $file" >&2; exit 2; }
done

python -u tasks/encode_chemaware_shared_v3_morgan.py \
  --graph data/validation/g8r_error_atlas_listwise_cache.npz \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --preflight data/validation/g8r_chemaware_shared_v2_preflight.json \
  --output-dir data/validation/g8r_chemaware_shared_v3_morgan \
  --radius 2 \
  --dimensions 2048
