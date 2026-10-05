#!/bin/bash
#SBATCH --job-name=chem_layer10k
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --output=data/validation/chem_layer10k_%j.out
#SBATCH --error=data/validation/chem_layer10k_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# One allocation, one GPU, no hand-written memory request.  The first stage
# creates and qualifies SIRIUS evidence.  The second stage starts only when at
# least one evidence family passed both matched controls and then constructs
# the preserved 10k+ native corpus and fine-tunes from protected Phase A.
bash tasks/run_chemaware_sirius_source.sbatch
SIRIUS_RUN="data/validation/chemaware_sirius_source/run_${SLURM_JOB_ID}" \
  bash tasks/run_chemaware_sirius_native.sbatch

echo "COMPLETE: integrated ChemAware layered SIRIUS 10k route"
