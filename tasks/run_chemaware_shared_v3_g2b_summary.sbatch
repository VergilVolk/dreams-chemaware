#!/bin/bash
#SBATCH --job-name=chemaware_v3_g2b_sum
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/chemaware_v3_g2b_summary_%j.out
#SBATCH --error=data/validation/chemaware_v3_g2b_summary_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python -u tasks/validate_chemaware_shared_v3_g2_pilot.py \
  --g2-root data/validation/g8r_chemaware_shared_v3_peft_g2b_pilot/frozen_probe_targeted \
  --output data/validation/g8r_chemaware_shared_v3_g2b_pilot_decision.json \
  --chemical-objective frozen_probe_targeted \
  --verify-only
echo "=== ChemAware shared-v3 G2b-PEFT summary started: $(date) ==="
python -u tasks/summarize_chemaware_shared_v3_g2.py \
  --g2-root data/validation/g8r_chemaware_shared_v3_peft_g2b/frozen_probe_targeted \
  --output data/validation/g8r_chemaware_shared_v3_peft_g2b_summary.json \
  --chemical-objective frozen_probe_targeted
echo "=== ChemAware shared-v3 G2b-PEFT summary completed: $(date) ==="
