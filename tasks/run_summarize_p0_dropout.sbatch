#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=summarize_p0_dropout_%j.out
#SBATCH --error=summarize_p0_dropout_%j.err

# P0 dropout 3-seed summary (reads the sbatch outputs already on the server).
# No GPU needed.
set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python tasks/summarize_p0_dropout.py
