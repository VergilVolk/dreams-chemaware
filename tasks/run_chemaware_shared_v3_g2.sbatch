#!/bin/bash
#SBATCH --job-name=chemaware_v3_g2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=18:00:00
#SBATCH --array=0-74%3
#SBATCH --output=data/validation/chemaware_v3_g2_%A_%a.out
#SBATCH --error=data/validation/chemaware_v3_g2_%A_%a.err

# Submit only after the fail-closed five-arm G2 pilot decision passes.
set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

arms=(correct identity_permuted random_marginal correct_same_formula_scope same_formula_mismatched)
seeds=(17 41 73)
arm=${arms[$((SLURM_ARRAY_TASK_ID % 5))]}
fold=$(((SLURM_ARRAY_TASK_ID / 5) % 5))
seed=${seeds[$((SLURM_ARRAY_TASK_ID / 25))]}

required=(
  data/validation/g8r_error_atlas_listwise_cache.npz
  data/validation/g8r_chemaware_shared_v2_preflight.json
  data/validation/g8r_chemaware_shared_v2_tokens/report.json
  data/validation/g8r_chemaware_shared_v3_morgan/report.json
  data/validation/g8r_chemaware_shared_v3_peft_g1_summary.json
  data/validation/g8r_chemaware_shared_v3_g2_pilot_decision.json
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "MISSING: $file" >&2; exit 2; }
done

# Re-hash the G1 summary, graph, decisions, predictions, and checkpoints pinned
# by the pilot decision immediately before every full-matrix task starts.
python -u tasks/validate_chemaware_shared_v3_g2_pilot.py --verify-only

echo "=== ChemAware shared-v3 G2 arm=$arm fold=$fold seed=$seed started: $(date) ==="
python -u tasks/train_chemaware_shared_v3_peft.py \
  --outer-fold "$fold" \
  --seed "$seed" \
  --last-blocks 1 \
  --rank 8 \
  --alpha 8 \
  --epochs 5 \
  --output-root data/validation/g8r_chemaware_shared_v3_peft_g2 \
  --molecule-teacher-dir data/validation/g8r_chemaware_shared_v3_morgan \
  --teacher-control "$arm" \
  --chemical-weighting absolute_bounded
echo "=== ChemAware shared-v3 G2 arm=$arm fold=$fold seed=$seed completed: $(date) ==="
