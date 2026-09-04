#!/bin/bash
#SBATCH --job-name=chemaware_v2_mol
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --output=data/validation/chemaware_v2_molformer_%j.out
#SBATCH --error=data/validation/chemaware_v2_molformer_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== ChemAware shared-v2 frozen MoLFormer cache started: $(date) ==="
python -u tasks/preflight_chemaware_shared_v2.py
python -u tasks/encode_chemaware_shared_v2_molformer.py --local-files-only
echo "=== ChemAware shared-v2 frozen MoLFormer cache completed: $(date) ==="
