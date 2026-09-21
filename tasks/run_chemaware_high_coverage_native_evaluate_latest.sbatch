#!/bin/bash
#SBATCH --job-name=chem_native_eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/chem_native_eval_%j.out
#SBATCH --error=data/validation/chem_native_eval_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

LATEST_BEST="$({
  find data/validation/chemaware_high_coverage_native \
    -path '*/training/best.ckpt' -type f -printf '%T@ %p\n'
} | sort -nr | head -n 1 | cut -d' ' -f2-)"
[[ -n "$LATEST_BEST" && -f "$LATEST_BEST" ]] || {
  echo "No completed ChemAware best.ckpt was found" >&2
  exit 2
}
TRAINING_DIR="$(dirname "$LATEST_BEST")"
SOURCE_RUN="$(dirname "$TRAINING_DIR")"
LATEST_LAST="$TRAINING_DIR/last.ckpt"

if [[ -f "$SOURCE_RUN/evidence/confirmation_triplet_evidence.npz" ]]; then
  POLICY="$SOURCE_RUN/evidence/confirmation_triplet_evidence.npz"
else
  POLICY="data/validation/chemaware_high_coverage_native/run_2340524/evidence/confirmation_triplet_evidence.npz"
fi
[[ -f "$POLICY" ]] || {
  echo "Missing frozen role-3 policy: $POLICY" >&2
  exit 2
}

OUTPUT="$SOURCE_RUN/role3_evaluation_after_stop.json"
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite existing evaluation: $OUTPUT" >&2
  exit 3
}

echo "Selected stopped run: $SOURCE_RUN"
echo "Selected best checkpoint: $LATEST_BEST"
if [[ -f "$LATEST_LAST" ]]; then
  echo "Selected last checkpoint: $LATEST_LAST"
else
  echo "No last.ckpt found; evaluating best.ckpt only"
fi
echo "Frozen role-3 policy: $POLICY"

checkpoint_args=(
  --checkpoint "official=data/e1/official_embedding_slim.pt"
  --checkpoint "chemaware_best=$LATEST_BEST"
)
if [[ -f "$LATEST_LAST" ]]; then
  checkpoint_args+=(--checkpoint "chemaware_last=$LATEST_LAST")
fi

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$POLICY" \
  "${checkpoint_args[@]}" \
  --output "$OUTPUT" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

test -f "$OUTPUT"
echo "COMPLETE: stopped-training role-3 evaluation -> $OUTPUT"
