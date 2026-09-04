#!/bin/bash
#SBATCH --job-name=bioaware_b2_loso
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/bioaware_b2_loso_%j.out
#SBATCH --error=data/validation/bioaware_b2_loso_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== BioAware B2 leave-study-out context embedding started: $(date) ==="
python -u tasks/test_bioaware_context_adapter.py
python -u tasks/test_bioaware_b2_leave_study_out.py
python -u tasks/build_bioaware_b2_external_context_dataset.py
python -u tasks/train_bioaware_b2_leave_study_out.py --device cuda
echo "=== BioAware B2 leave-study-out context embedding completed: $(date) ==="
