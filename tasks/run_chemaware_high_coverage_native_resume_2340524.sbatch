#!/bin/bash
#SBATCH --job-name=chem_native_resume
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/chem_native_resume_%j.out
#SBATCH --error=data/validation/chem_native_resume_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

SOURCE="data/validation/chemaware_high_coverage_native/run_2340524"
OUT="data/validation/chemaware_high_coverage_native/run_${SLURM_JOB_ID}_resume_2340524"
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }

required_files=(
  "$SOURCE/triplets/train_pool.npz"
  "$SOURCE/triplets/val_pool.npz"
  "$SOURCE/triplets/report.json"
  "$SOURCE/evidence/confirmation_triplet_evidence.npz"
  "$SOURCE/official_role3_preflight.json"
  tasks/train_chemaware_dreams_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required resume artifact: $required_file" >&2
    exit 2
  }
done
mkdir -p "$OUT"

srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --train-pool "$SOURCE/triplets/train_pool.npz" \
  --val-pool "$SOURCE/triplets/val_pool.npz" \
  --official-checkpoint data/e1/official_embedding_slim.pt \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --output "$OUT/training" \
  --seed 3407 --lr 5e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
  --max-epochs 301 --n-highest-peaks 100 --device cuda

python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$SOURCE/evidence/confirmation_triplet_evidence.npz" \
  --checkpoint "official=data/e1/official_embedding_slim.pt" \
  --checkpoint "chemaware_high_coverage=$OUT/training/best.ckpt" \
  --output "$OUT/shared_embedding_role3_report.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

test -f "$OUT/training/report.json"
test -f "$OUT/training/best.ckpt"
test -f "$OUT/shared_embedding_role3_report.json"
echo "COMPLETE: resumed high-coverage native DreaMS training -> $OUT"
