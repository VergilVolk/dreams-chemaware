#!/bin/bash
#SBATCH --job-name=chem_v2_triplet
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_v2_triplet_%j.out
#SBATCH --error=data/validation/chem_v2_triplet_%j.err

set -euo pipefail

echo "BLOCKED: superseded query-filter triplet logic; action-preserving replay is required" >&2
exit 2

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OUT="data/validation/chemaware_v2_direct_triplet/run_${SLURM_JOB_ID}"
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

required_files=(
  tasks/build_chemaware_v2_direct_triplet_pools.py
  tasks/test_chemaware_v2_direct_triplet_pools.py
  tasks/train_e1_identity.py
  tasks/e1_checkpoint_io.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/chemaware_v2_triplet_eval_core.py
  data/validation/chemaware_multinull_deployment_safe_full_20260919/validation_policy.npz
  data/validation/chemaware_multinull_deployment_safe_full_20260919/inner_policy.npz
  data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz
  data/models/MassSpecGym_MurckoHist_split.hdf5
  dreams/models/pretrained/ssl_model_server.pt
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required runtime file: $required_file" >&2
    exit 2
  }
done

OFFICIAL="data/e1/official_embedding_slim.pt"
if [[ ! -f "$OFFICIAL" ]]; then
  python -u tasks/prepare_official_embedding_checkpoint.py \
    --source dreams/models/pretrained/embedding_model.ckpt \
    --output "$OFFICIAL"
fi

python -u tasks/test_chemaware_v2_direct_triplet_pools.py
python -u tasks/build_chemaware_v2_direct_triplet_pools.py --output "$OUT/pools"

# Fail before either optimization arm if the official encoder cannot exactly
# replay the frozen role-3 candidate ledger.
python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --checkpoint "official=$OFFICIAL" \
  --output "$OUT/official_role3_preflight.json" \
  --batch-size 64 --bootstrap-draws 100 --device cuda

common_train=(
  --data data/models/MassSpecGym_MurckoHist_split.hdf5
  --val-pool "$OUT/pools/role3_development_pool.npz"
  --base-ckpt "$OFFICIAL"
  --architecture-ckpt dreams/models/pretrained/ssl_model_server.pt
  --seed 20260920
  --epochs 10
  --batch-size 16
  --lr 1e-6
  --weight-decay 0
  --margin 0.1
  --n-highest-peaks 100
  --num-workers 4
  --grad-accum 1
  --grad-clip 1.0
  --patience 3
  --val-triplets 93
  --device cuda
  --amp
)

python -u tasks/train_e1_identity.py \
  "${common_train[@]}" \
  --train-pool "$OUT/pools/role2_matched_control_train_pool.npz" \
  --output-dir "$OUT/matched_control"

python -u tasks/train_e1_identity.py \
  "${common_train[@]}" \
  --train-pool "$OUT/pools/role2_chemaware_train_pool.npz" \
  --output-dir "$OUT/chemaware"

CONTROL_CKPT="$OUT/matched_control/seed_20260920/best_e1.pt"
CHEM_CKPT="$OUT/chemaware/seed_20260920/best_e1.pt"
[[ -f "$CONTROL_CKPT" ]] || { echo "Missing control checkpoint" >&2; exit 4; }
[[ -f "$CHEM_CKPT" ]] || { echo "Missing ChemAware checkpoint" >&2; exit 4; }

python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "matched_control=$CONTROL_CKPT" \
  --checkpoint "chemaware_triplet=$CHEM_CKPT" \
  --output "$OUT/shared_embedding_role3_report.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

test -f "$OUT/pools/report.json"
test -f "$OUT/official_role3_preflight.json"
test -f "$OUT/shared_embedding_role3_report.json"
echo "COMPLETE: ChemAware V2 direct DreaMS triplet experiment -> $OUT"
