#!/bin/bash
#SBATCH --job-name=noise_frontier_g0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/noise_frontier_g0_%j.out
#SBATCH --error=data/validation/noise_frontier_g0_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OUT="data/validation/noise_frontier_g0_20260914"
[[ ! -e "$OUT" ]] || { echo "REFUSE: G0 output already exists: $OUT" >&2; exit 3; }

required=(
  data/validation/g8r_error_atlas_listwise_cache.npz
  data/validation/g8r_p2_official_embeddings.npz
  data/validation/g8r_noise_final_f1_full_tokens/report.json
  data/validation/g8r_noise_final_f1_full_tokens/rows.npy
  data/validation/g8r_noise_final_f1_full_tokens/tokens_f16.npy
  data/models/MassSpecGym_MurckoHist_split.hdf5
  dreams/models/chem_aware/chem_rules_data.json
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "MISSING: $path" >&2; exit 2; }
done

echo "=== frozen frontier G0 started: $(date) ==="
python -m py_compile \
  tasks/noise_frontier_g0_core.py \
  tasks/build_noise_frontier_g0_cache.py \
  tasks/audit_noise_frontier_g0_representation.py \
  tasks/test_noise_frontier_g0.py \
  tasks/validate_noise_frontier_g0.py
python -u tasks/test_noise_frontier_g0.py

python -u tasks/build_noise_frontier_g0_cache.py --output "$OUT/cache.npz"
python -u tasks/audit_noise_frontier_g0_representation.py \
  --cache "$OUT/cache.npz" --output "$OUT/report.json"
python -u tasks/validate_noise_frontier_g0.py --output-dir "$OUT"
echo "=== frozen frontier G0 completed: $(date) ==="
