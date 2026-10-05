#!/bin/bash
#SBATCH --job-name=chem_b1j
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/chem_b1j_%j.out
#SBATCH --error=data/validation/chem_b1j_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ROOT="data/validation/chemaware_jacobian_intersection_actions/run_${SLURM_JOB_ID}"
[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 3; }
trap 'echo "FAILED: ChemAware Jacobian-intersection audit stopped before completion" >&2' ERR

python -m py_compile \
  tasks/chemaware_boundary_consensus_action_core.py \
  tasks/chemaware_jacobian_intersection_action_core.py \
  tasks/audit_chemaware_jacobian_intersection_actions.py \
  tasks/test_chemaware_jacobian_intersection_action_core.py \
  tasks/test_chemaware_jacobian_intersection_action_contract.py \
  tasks/test_chemaware_jacobian_intersection_runtime.py
python tasks/test_chemaware_boundary_consensus_action_core.py
python tasks/test_chemaware_jacobian_intersection_action_core.py
python tasks/test_chemaware_jacobian_intersection_action_contract.py
python tasks/test_chemaware_jacobian_intersection_runtime.py
python -u tasks/audit_chemaware_jacobian_intersection_actions.py \
  --output "$RUN_ROOT" \
  --device cuda --batch-size 16 --torch-threads 8 \
  --bootstrap-draws 10000 \
  --discovery-folds 0 1 --confirmation-fold 2 \
  --embedding-evaluation-fold 3 --reserve-fold 4 \
  --preflight-only

# Frozen first-order action admissibility only: no optimizer step and no
# perturbed-spectrum encoding.  A pass can authorize only a later nonlinear
# action audit, never embedding fine-tuning by itself.
python -u tasks/audit_chemaware_jacobian_intersection_actions.py \
  --output "$RUN_ROOT" \
  --device cuda --batch-size 16 --torch-threads 8 \
  --bootstrap-draws 10000 \
  --discovery-folds 0 1 --confirmation-fold 2 \
  --embedding-evaluation-fold 3 --reserve-fold 4

test -f "$RUN_ROOT/report.json"
test -f "$RUN_ROOT/screen.npz"
echo "COMPLETE: ChemAware chemistry-by-Jacobian action screen"
