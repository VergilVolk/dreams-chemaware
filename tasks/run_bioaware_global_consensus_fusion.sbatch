#!/bin/bash
#SBATCH --job-name=bioaware_g3v3
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=bioaware_g3v3_%j.out
#SBATCH --error=bioaware_g3v3_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if [[ ! -f data/validation/bioaware_candidate_evidence_ledger_v1/candidate_evidence.csv.gz ]]; then
  python -u tasks/build_bioaware_candidate_evidence_ledger.py
fi
python -u tasks/validate_bioaware_candidate_evidence_ledger.py
python -u tasks/develop_bioaware_global_consensus_fusion.py
