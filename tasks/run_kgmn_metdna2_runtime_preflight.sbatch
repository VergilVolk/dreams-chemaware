#!/bin/bash
#SBATCH --job-name=kgmn_r_preflight
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=kgmn_r_preflight_%j.out
#SBATCH --error=kgmn_r_preflight_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

command -v Rscript >/dev/null 2>&1 || { echo "Rscript is unavailable" >&2; exit 2; }
Rscript tasks/preflight_kgmn_metdna2_runtime.R third_party/MetDNA2
