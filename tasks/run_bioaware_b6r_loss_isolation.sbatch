#!/bin/bash
#SBATCH --job-name=bio_b6r_iso
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=bioaware_b6r_%A_%a.out
#SBATCH --error=bioaware_b6r_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

MANIFEST=data/validation/bioaware_b4_direct_shared_embedding_v1_20260905/manifest
TEACHER=data/validation/bioaware_b5_pretraining_gate_v2_20260905_2331643/teacher
TASK_ID="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
ARRAY_JOB="${SLURM_ARRAY_JOB_ID:?missing SLURM_ARRAY_JOB_ID}"

case "$TASK_ID" in
  0) MODE=direct_onehot ;;
  1) MODE=bioaware_soft ;;
  *) echo "Invalid array task: $TASK_ID" >&2; exit 2 ;;
esac

FINAL="data/validation/bioaware_b6r_loss_isolation_v1_20260906_${ARRAY_JOB}_${TASK_ID}"
test ! -e "$FINAL" || { echo "Refusing to overwrite: $FINAL" >&2; exit 3; }

required=(
  "$MANIFEST/manifest.npz"
  "$MANIFEST/report.json"
  "$TEACHER/report.json"
  "$TEACHER/candidate_teacher_scores.csv.gz"
  "$TEACHER/teacher_actions.csv.gz"
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
  annotation/bioaware_supervision.py
  tasks/train_bioaware_b4_direct_shared_embedding.py
  tasks/test_bioaware_b6_candidate_supervision.py
)
for path in "${required[@]}"; do
  test -s "$path" || { echo "Missing dependency: $path" >&2; exit 4; }
done

python -c "import numpy,pandas,torch; assert torch.cuda.is_available(); print('[BioAware B6R dependencies] PASS')"
python -m py_compile \
  annotation/bioaware_supervision.py \
  tasks/train_bioaware_b4_direct_shared_embedding.py \
  tasks/test_bioaware_b6_candidate_supervision.py
python -u tasks/test_bioaware_b6_candidate_supervision.py

echo "=== BioAware B6R mode=$MODE fold=0 started: $(date) ==="
python -u tasks/train_bioaware_b4_direct_shared_embedding.py \
  --manifest-dir "$MANIFEST" \
  --candidate-teacher-dir "$TEACHER" \
  --output "$FINAL" \
  --arm full_no_edge_high_recall \
  --supervision-mode "$MODE" \
  --outer-fold 0 \
  --epochs 2 \
  --steps-per-epoch 32 \
  --batch-queries 4 \
  --references-per-molecule 2 \
  --corrective-fraction 0.5 \
  --unfreeze-blocks 1 \
  --backbone-lr 1e-6 \
  --head-lr 5e-6 \
  --temperature 0.10 \
  --teacher-temperature 1.0 \
  --lambda-listwise 1.0 \
  --lambda-corrective 0.0 \
  --lambda-safety-floor 2.0 \
  --lambda-preserve 5.0 \
  --no-amp \
  --device cuda

test -s "$FINAL/report.json" || { echo "Missing B6R report" >&2; exit 5; }
test -s "$FINAL/held_per_query.csv.gz" || { echo "Missing B6R held ledger" >&2; exit 5; }
test -s "$FINAL/final_trainable_state.pt" || { echo "Missing B6R checkpoint" >&2; exit 5; }
echo "=== BioAware B6R completed: $(date) ==="
echo "Result: $FINAL"
