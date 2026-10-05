#!/bin/bash
#SBATCH --job-name=chem_dreams_native
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_dreams_native_%j.out
#SBATCH --error=data/validation/chem_dreams_native_%j.err

set -euo pipefail

echo "BLOCKED: the 95-triplet role-2/3 pool is scientifically underpowered; use the roles-0/1 high-purity action miner before training" >&2
exit 2

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OUT="data/validation/chemaware_dreams_native/run_${SLURM_JOB_ID}"
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

required_files=(
  tasks/build_chemaware_dreams_native_triplets.py
  tasks/test_chemaware_dreams_native_triplets.py
  tasks/test_chemaware_dreams_native_runtime_contracts.py
  tasks/train_chemaware_dreams_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/chemaware_v2_triplet_eval_core.py
  tasks/train_e1_identity.py
  tasks/e1_checkpoint_io.py
  dreams/models/heads/heads.py
  dreams/utils/data.py
  data/validation/chemaware_multinull_deployment_safe_full_20260919/validation_policy.npz
  data/validation/chemaware_multinull_deployment_safe_full_20260919/inner_policy.npz
  data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required runtime file: $required_file" >&2
    exit 2
  }
done

python -u tasks/test_chemaware_dreams_native_triplets.py
python -u tasks/test_chemaware_dreams_native_runtime_contracts.py
python -u tasks/build_chemaware_dreams_native_triplets.py --output "$OUT/triplets"

# Exact official replay must pass before any gradient update.
python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --checkpoint "official=data/e1/official_embedding_slim.pt" \
  --output "$OUT/official_role3_preflight.json" \
  --batch-size 64 --bootstrap-draws 100 --device cuda

python -u tasks/train_chemaware_dreams_native.py \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --train-pool "$OUT/triplets/train_pool.npz" \
  --val-pool "$OUT/triplets/val_pool.npz" \
  --official-checkpoint data/e1/official_embedding_slim.pt \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --output "$OUT/training" \
  --seed 3407 --lr 5e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 4 \
  --max-epochs 301 --n-highest-peaks 100 --device cuda

python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --checkpoint "official=data/e1/official_embedding_slim.pt" \
  --checkpoint "chemaware_dreams_native=$OUT/training/best.ckpt" \
  --output "$OUT/shared_embedding_role3_report.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

test -f "$OUT/triplets/report.json"
test -f "$OUT/training/report.json"
test -f "$OUT/training/best.ckpt"
test -f "$OUT/shared_embedding_role3_report.json"
echo "COMPLETE: native DreaMS ChemAware triplet continuation -> $OUT"
