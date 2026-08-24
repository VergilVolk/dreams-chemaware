#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --output=g8r_p2b_p3_%j.out
#SBATCH --error=g8r_p2b_p3_%j.err

# One-shot frozen evaluation. No fitting, tuning, threshold search or overwrite.
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

required=(
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
  data/validation/g8r_p2_listwise_cache.json
  data/validation/g8r_p2b_rank_fusion.json
  data/validation/g8r_p2b_rank_fusion.selection.json
  data/validation/g8r_p3_test/p3_lock_summary.json
  data/validation/g8r_p3_test/p3_reference_library_real.json
  tasks/eval_g8r_p2b_on_sealed_p3.py
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "MISSING: $file" >&2; exit 2; }
done
[[ ! -e data/validation/g8r_p2b_p3_final.json ]] || {
  echo "REFUSE: sealed P3 result already exists" >&2
  exit 3
}
[[ ! -e data/validation/g8r_p2b_p3_final.per_query.csv ]] || {
  echo "REFUSE: sealed P3 per-query result already exists" >&2
  exit 4
}

python -u tasks/eval_g8r_p2b_on_sealed_p3.py \
  --device cuda \
  --batch-size 128 \
  --n-highest-peaks 100 \
  --peak-tolerance 0.02 \
  --bootstrap 5000 \
  --seed 20260827

echo "=== frozen P2b sealed-P3 evaluation completed: $(date) ==="
