#!/bin/bash
#SBATCH --job-name=chem_rule_summary
##SBATCH --partition=gpu
##SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/chem_rule_summary_%j.out
#SBATCH --error=data/validation/chem_rule_summary_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

: "${ARMS_JOB:?submit with the eight-arm array job id}"
ROOT="data/validation/chemaware_rule_kernel_direct_phase_a/run_${ARMS_JOB}"
python -u tasks/summarize_chemaware_mass_kernel_direct_arms.py \
  --root "$ROOT" --output-dir "$ROOT/summary" \
  --bootstrap-draws 10000 --seed 20260905
