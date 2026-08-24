#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=g8r_p3_v3_lock_%j.out
#SBATCH --error=g8r_p3_v3_lock_%j.err

# P3 v3 is CPU-only. It recomputes MCES labels missing from the train-only
# pairs.json, then seals real-spectrum main/challenge panels and P2 allow-lists.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

required=(
  data/models/MassSpecGym_MurckoHist_split.hdf5
  tasks/massspecgym_isomers/g8r_locked/train.json
  tasks/massspecgym_isomers/g8r_locked/val.json
  tasks/massspecgym_isomers/pairs.json
  data/validation/g8r_raw_reranker_cache.npz
  data/validation/g8r_raw_reranker_cache_val.npz
  data/validation/g8r_final_test/test_a_manifest.json
  data/validation/g8r_final_test/test_b_manifest.json
  data/validation/g8r_final_test/test_c_manifest.json
  data/validation/large_observability_residual_audit/discovery_query_audit.csv
  data/validation/large_observability_residual_audit/confirmation_query_audit.csv
  tasks/build_g8r_p3_test.py
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "MISSING: $file" >&2; exit 2; }
done

[[ ! -e data/validation/g8r_p3_test ]] || {
  echo "REFUSE: sealed output already exists: data/validation/g8r_p3_test" >&2
  exit 3
}

python -u tasks/build_g8r_p3_test.py \
  --mces-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --expected-hdf5-sha256 ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f \
  --expected-pairs-sha256 0461e95b7e98d9d7fa0f03c884b7b8b7996e58bbdafecdb95de31e02c6cb0d9a \
  --output-dir data/validation/g8r_p3_test

echo "=== P3 v3 lock completed: $(date) ==="
echo "Review data/validation/g8r_p3_test/p3_lock_summary.json before P2 development."
