#!/bin/bash
#SBATCH --job-name=network_repro_audit
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=network_repro_audit_%j.out
#SBATCH --error=network_repro_audit_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

python -u tasks/audit_metabolic_network_framework_reproducibility.py \
  --output data/validation/metabolic_network_framework_reproducibility.json
