#!/bin/bash
#SBATCH --job-name=bioaware_ext_lock
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/bioaware_ext_lock_%j.out
#SBATCH --error=data/validation/bioaware_ext_lock_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== BioAware external benchmark lock started: $(date) ==="
python -u -m pytest -q tests/test_bioaware_netid_external.py tests/test_bioaware_metdna3_external.py tests/test_bioaware_context.py
python -u tasks/preflight_bioaware_netid_dreams_feasibility.py
python -u tasks/lock_bioaware_metdna3_external_source.py
echo "=== BioAware external benchmark lock completed: $(date) ==="
