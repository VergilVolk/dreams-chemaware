#!/bin/bash
#SBATCH --job-name=chemaware_v2_g1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --array=0-14%3
#SBATCH --output=data/validation/chemaware_v2_g1_%A_%a.out
#SBATCH --error=data/validation/chemaware_v2_g1_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

seeds=(17 41 73)
fold=$((SLURM_ARRAY_TASK_ID % 5))
seed=${seeds[$((SLURM_ARRAY_TASK_ID / 5))]}

required=(
  data/validation/g8r_error_atlas_listwise_cache.npz
  data/validation/g8r_chemaware_shared_v2_preflight.json
  data/validation/g8r_chemaware_shared_v2_tokens/report.json
  data/e1/official_embedding_slim.pt
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "MISSING: $file" >&2; exit 2; }
done

echo "=== ChemAware shared-v2 G1 fold=$fold seed=$seed started: $(date) ==="
python -u tasks/train_chemaware_shared_v2.py \
  --outer-fold "$fold" \
  --seed "$seed"
echo "=== ChemAware shared-v2 G1 fold=$fold seed=$seed completed: $(date) ==="
