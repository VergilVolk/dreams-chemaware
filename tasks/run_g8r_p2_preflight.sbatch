#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=g8r_p2_preflight_%j.out
#SBATCH --error=g8r_p2_preflight_%j.err

# This job never trains or opens P3 predictions.  It must reproduce the frozen
# RAW-v1 +4.35 pp compatibility result and prove P2/P3 disjointness first.
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

required=(
  data/models/MassSpecGym_MurckoHist_split.hdf5
  tasks/massspecgym_isomers/pairs.json
  data/validation/g8r_p3_test/p3_lock_summary.json
  data/validation/g8r_p3_test/p3_p2_allowed_training_ik14.json
  data/validation/g8r_raw_reranker_v1_artifact.json
  data/validation/g8r_raw_reranker_cache.npz
  data/validation/g8r_raw_reranker_cache_val.npz
  tasks/audit_g8r_p2_preflight.py
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "MISSING: $file" >&2; exit 2; }
done

python -u tasks/audit_g8r_p2_preflight.py

echo "=== P2 preflight completed: $(date) ==="
echo "Do not submit cache/training until g8r_p2_preflight.json says PASS and reproduces +4.35 pp."
