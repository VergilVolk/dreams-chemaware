#!/bin/bash
#SBATCH --job-name=chem_resid_audit
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/chem_resid_audit_%j.out
#SBATCH --error=data/validation/chem_resid_audit_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ROOT=$(python - <<'PY'
import json
import pathlib

root = pathlib.Path("data/validation/chemaware_iceberg_residual_shared_phase_a")
complete = []
for path in root.glob("run_*/phase_a_summary.json"):
    body = json.loads(path.read_text())
    if body.get("status") == "CHEMAWARE_ICEBERG_RESIDUAL_SHARED_PHASE_A_DEVELOPMENT_CAUSAL_FAIL":
        complete.append(path.parent)
assert complete, "no completed failed residual Phase A was found"
print(max(complete, key=lambda path: path.stat().st_mtime))
PY
)

echo "Auditing completed Phase A: $RUN_ROOT"
python -u tasks/audit_chemaware_iceberg_residual_transfer_mechanism.py \
  --run-root "$RUN_ROOT" --alpha 0.50 --huber 0.02 \
  --n-highest-peaks 100 --eval-batch-size 96 \
  --bootstrap-draws 10000 --seed 20260907 \
  --device cuda --no-amp --torch-threads 8
test -f "$RUN_ROOT/transfer_mechanism_audit.json"
echo "COMPLETE: ChemAware residual transfer mechanism audit"

