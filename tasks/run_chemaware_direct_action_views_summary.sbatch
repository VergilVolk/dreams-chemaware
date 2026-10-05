#!/bin/bash
#SBATCH --job-name=chem_action_sum
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=data/validation/chem_action_sum_%j.out
#SBATCH --error=data/validation/chem_action_sum_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

: "${ACTION_VIEW_RUN:?set ACTION_VIEW_RUN to the four-arm run directory}"
OUT="$ACTION_VIEW_RUN/causal_summary.json"
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
python tasks/summarize_chemaware_direct_action_views.py \
  --root "$ACTION_VIEW_RUN" --output "$OUT" --bootstrap-draws 10000
