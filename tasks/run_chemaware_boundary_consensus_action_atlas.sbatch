#!/bin/bash
#SBATCH --job-name=chem_b1_atlas
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=data/validation/chem_b1_atlas_%j.out
#SBATCH --error=data/validation/chem_b1_atlas_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ROOT="data/validation/chemaware_boundary_consensus_action_atlas/run_${SLURM_JOB_ID}"
[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 3; }
trap 'echo "FAILED: ChemAware B1 action-atlas audit stopped before completion" >&2' ERR

# Fast contracts and fail-closed asset validation run before DreaMS is loaded.
python -m py_compile \
  tasks/chemaware_boundary_consensus_action_core.py \
  tasks/audit_chemaware_boundary_consensus_action_atlas.py \
  tasks/test_chemaware_boundary_consensus_action_core.py \
  tasks/test_chemaware_boundary_consensus_action_atlas_contract.py
python tasks/test_chemaware_boundary_consensus_action_core.py
python tasks/test_chemaware_boundary_consensus_action_atlas_contract.py
python -u tasks/audit_chemaware_boundary_consensus_action_atlas.py \
  --output "$RUN_ROOT" \
  --device cuda --batch-size 64 --torch-threads 8 \
  --bootstrap-draws 10000 \
  --discovery-folds 0 1 --confirmation-fold 2 \
  --embedding-evaluation-fold 3 --reserve-fold 4 \
  --preflight-only

# Frozen official-DreaMS A1 screen only.  It performs no optimizer step and
# cannot authorize embedding fine-tuning by itself.
python -u tasks/audit_chemaware_boundary_consensus_action_atlas.py \
  --output "$RUN_ROOT" \
  --device cuda --batch-size 64 --torch-threads 8 \
  --bootstrap-draws 10000 \
  --discovery-folds 0 1 --confirmation-fold 2 \
  --embedding-evaluation-fold 3 --reserve-fold 4

test -f "$RUN_ROOT/report.json"
test -f "$RUN_ROOT/screen.npz"
echo "COMPLETE: ChemAware B1 same-boundary action atlas"
