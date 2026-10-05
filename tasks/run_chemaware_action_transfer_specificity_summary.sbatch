#!/bin/bash
#SBATCH --job-name=chem_a2_spec
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=data/validation/chem_a2_spec_%j.out
#SBATCH --error=data/validation/chem_a2_spec_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

trap 'echo "FAILED: ChemAware A2 specificity summary stopped before completion" >&2' ERR

# Contracts run before the read-only summary.  The allocation has exactly one
# GPU as required, although this post-hoc calculation does not load DreaMS.
python -m py_compile \
  tasks/summarize_chemaware_action_transfer_specificity.py \
  tasks/test_summarize_chemaware_action_transfer_specificity.py
python tasks/test_summarize_chemaware_action_transfer_specificity.py

# The summarizer resolves the newest completed run_*/screen24 itself and verifies
# COMPLETE.json plus the frozen action-bank hash.  No job id is hand-written.
python -u tasks/summarize_chemaware_action_transfer_specificity.py \
  --bootstrap-draws 10000 \
  --seed 20260906

echo "COMPLETE: ChemAware A2 specificity summary"
