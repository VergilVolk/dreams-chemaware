#!/bin/bash
#SBATCH --job-name=chem_a2_grad
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=data/validation/chem_a2_grad_%j.out
#SBATCH --error=data/validation/chem_a2_grad_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ACTION_BANK="data/validation/chemaware_direct_action_bank_v1"
RUN_ROOT="data/validation/chemaware_action_transfer_gradients/run_${SLURM_JOB_ID}"
[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 3; }
mkdir -p "$RUN_ROOT"
trap 'echo "FAILED: ChemAware A2 gradient screen stopped before completion" >&2' ERR

# CPU contracts run before the 117M model is constructed.
python -m py_compile \
  tasks/chemaware_action_transfer_core.py \
  tasks/audit_chemaware_action_transfer_gradients.py \
  tasks/test_chemaware_action_transfer_core.py \
  tasks/test_chemaware_action_transfer_audit.py
python tasks/test_chemaware_action_transfer_core.py
python tasks/test_chemaware_action_transfer_audit.py

COMMON=(
  --action-bank "$ACTION_BANK"
  --action-generation-seed 20260935
  --selection-seed 20260906
  --max-actions 24
  --unfreeze-blocks 1
  --backbone-lr 2e-6 --head-lr 1e-5
  --temperature 0.10 --pair-target-margin 0
  --bootstrap-draws 10000 --minimum-positive-fraction 0.60
  --replay-tolerance 5e-4
  --device cuda --no-amp --torch-threads 8
)

# This resolves every ledger, action-setting, seed, row-coverage and output
# contract without loading DreaMS or touching model weights.
python -u tasks/audit_chemaware_action_transfer_gradients.py \
  "${COMMON[@]}" --output "$RUN_ROOT/cpu_preflight" --preflight-only
test -f "$RUN_ROOT/cpu_preflight/preflight.json"

# Cost-bounded formula-stratified mechanism screen.  It performs zero optimizer
# steps and cannot authorize a full embedding run by itself.
python -u tasks/audit_chemaware_action_transfer_gradients.py \
  "${COMMON[@]}" --output "$RUN_ROOT/screen24"
test -f "$RUN_ROOT/screen24/COMPLETE.json"
test -f "$RUN_ROOT/screen24/report.json"
test -f "$RUN_ROOT/screen24/per_action.csv"

python - "$RUN_ROOT/screen24/report.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
print("status=", report["status"])
print("winner=", report["winner"])
print("pass_to_full_gradient_confirmation=", report["pass_to_full_gradient_confirmation"])
print("weights_updated=", report["weights_updated"])
print("performance_pp_claim=", report["interpretation"]["performance_pp_claim"])
PY

echo "COMPLETE: ChemAware A2 gradient screen"
