#!/bin/bash
#SBATCH --job-name=chemaware_v3_g2bpd
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=data/validation/chemaware_v3_g2b_pilot_decision_%j.out
#SBATCH --error=data/validation/chemaware_v3_g2b_pilot_decision_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

echo "=== ChemAware shared-v3 G2b pilot decision started: $(date) ==="
python -u tasks/validate_chemaware_shared_v3_g2_pilot.py \
  --g2-root data/validation/g8r_chemaware_shared_v3_peft_g2b_pilot/frozen_probe_targeted \
  --output data/validation/g8r_chemaware_shared_v3_g2b_pilot_decision.json \
  --chemical-objective frozen_probe_targeted
echo "=== ChemAware shared-v3 G2b pilot decision completed: $(date) ==="
