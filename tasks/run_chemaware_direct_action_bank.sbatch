#!/bin/bash
#SBATCH --job-name=chem_action_bank
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/chem_action_bank_%j.out
#SBATCH --error=data/validation/chem_action_bank_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Frozen-encoder action qualification only: no optimizer and no weight update.
# The program itself keeps formula folds 3 and 4 completely action-unseen.
OUT="data/validation/chemaware_direct_action_bank_v1"
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }

python -u tasks/audit_chemaware_direct_action_bank.py \
  --output "$OUT" --device cuda --batch-size 64 --torch-threads 8 \
  --bootstrap-draws 10000 \
  --discovery-folds 0 1 --confirmation-fold 2 \
  --embedding-evaluation-fold 3 --reserve-fold 4

test -f "$OUT/report.json"
test -f "$OUT/action_bank.npz"
python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("data/validation/chemaware_direct_action_bank_v1/report.json").read_text())
print("action_bank_status=", report["status"])
print("pass_to_direct_action_pilot=", report["pass_to_direct_action_pilot"])
PY
