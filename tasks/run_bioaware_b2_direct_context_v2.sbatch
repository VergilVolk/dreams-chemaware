#!/bin/bash
#SBATCH --job-name=bioaware_b2_v2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/bioaware_b2_v2_%j.out
#SBATCH --error=data/validation/bioaware_b2_v2_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ROOT=data/validation/bioaware_b2_direct_context_v2_20260905
WORK="${ROOT}.partial.${SLURM_JOB_ID}"
DATASET_DIR="$WORK/dataset"
SOURCE=data/validation/bioaware_metdna3_external_v3_v1

echo "=== BioAware B2 direct-context v2 started: $(date) ==="
python -u tasks/test_bioaware_context_adapter.py
python -u tasks/test_bioaware_b2_leave_study_out.py
python -u tasks/test_bioaware_b2_direct_context_v2.py

for unit in \
  BV2cell__hilic BV2cell__rplc Mouse_brain__hilic Mouse_brain__rplc \
  Mouse_liver__hilic Mouse_liver__rplc NIST_plasma__hilic NIST_plasma__rplc; do
  for path in \
    "$SOURCE/$unit/ledger/candidate_evidence.csv.gz" \
    "$SOURCE/$unit/ledger/report.json" \
    "$SOURCE/$unit/scores/candidate_scores.csv.gz" \
    "$SOURCE/$unit/scores/embeddings.npz" \
    "$SOURCE/$unit/cache/queries.csv.gz"; do
    test -s "$path" || { echo "Missing dependency: $path" >&2; exit 2; }
  done
done

test ! -e "$ROOT" || { echo "Refusing to overwrite: $ROOT" >&2; exit 3; }
test ! -e "$WORK" || { echo "Refusing to overwrite partial work: $WORK" >&2; exit 3; }
mkdir -p "$WORK"
python -u tasks/build_bioaware_b2_external_context_dataset.py \
  --root "$SOURCE" --output-dir "$DATASET_DIR"

run_cell() {
  local name=$1
  local evidence=$2
  local isolation=$3
  echo "=== cell $name evidence=$evidence isolation=$isolation: $(date) ==="
  python -u tasks/train_bioaware_b2_leave_study_out.py \
    --dataset "$DATASET_DIR/dataset.npz" \
    --output-dir "$WORK/$name" \
    --evidence-mode "$evidence" \
    --training-isolation "$isolation" \
    --formal --device cuda
  python -u tasks/replay_bioaware_b2_leave_study_out.py \
    --dataset "$DATASET_DIR/dataset.npz" --input-dir "$WORK/$name" --device cuda
  python -u tasks/audit_bioaware_b2_flip_mechanisms.py \
    --dataset "$DATASET_DIR/dataset.npz" --input-dir "$WORK/$name" --device cuda
}

# Primary and attribution controls.  The recipe is identical in every cell.
run_cell full_study full study
run_cell reaction_smn_study reaction_smn study
run_cell reaction_study reaction_only study
run_cell smn_study smn_only study
run_cell smn_rt_study smn_rt study
run_cell rt_study rt_only study

# Generalisation sensitivity; this does not replace the fixed primary.  On the
# current data, candidate-clean identity and formula isolation select identical
# training queries, so running both would be a duplicate experiment.
run_cell full_formula full truth_formula

python -u tasks/summarize_bioaware_b2_direct_context_v2.py --root "$WORK"
test -s "$WORK/matrix_report.json" || { echo "Missing final matrix report" >&2; exit 4; }
mv "$WORK" "$ROOT"
echo "=== BioAware B2 direct-context v2 completed: $(date) ==="
