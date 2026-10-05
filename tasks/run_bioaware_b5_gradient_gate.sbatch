#!/bin/bash
#SBATCH --job-name=bio_b5_grad
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_b5_gradient_%j.out
#SBATCH --error=bioaware_b5_gradient_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

TEACHER=data/validation/bioaware_b5_candidate_teacher_v1_20260905
MANIFEST=data/validation/bioaware_b4_direct_shared_embedding_v1_20260905/manifest/manifest.npz
FINAL=data/validation/bioaware_b5_gradient_gate_v1_20260905
WORK="${FINAL}.partial.${SLURM_JOB_ID}"
test ! -e "$FINAL" || { echo "Refusing to overwrite: $FINAL" >&2; exit 3; }
test ! -e "$WORK" || { echo "Refusing to overwrite: $WORK" >&2; exit 3; }
for path in \
  "$TEACHER/report.json" "$TEACHER/teacher_actions.csv.gz" \
  "$TEACHER/candidate_teacher_scores.csv.gz" "$MANIFEST" \
  data/e1/official_embedding_slim.pt dreams/models/pretrained/ssl_model_server.pt \
  tasks/audit_bioaware_b5_gradient_injection.py \
  tasks/test_bioaware_b5_gradient_injection.py; do
  test -s "$path" || { echo "Missing dependency: $path" >&2; exit 2; }
done

python -c "import numpy,pandas,torch; assert torch.cuda.is_available(); print('[BioAware B5 gradient dependencies] PASS')"
python -m py_compile tasks/audit_bioaware_b5_gradient_injection.py tasks/test_bioaware_b5_gradient_injection.py
python -u tasks/test_bioaware_b5_gradient_injection.py
mkdir -p "$WORK"
for arm in full_bioaware_safe full_no_edge_high_recall; do
  python -u tasks/audit_bioaware_b5_gradient_injection.py \
    --b4-manifest "$MANIFEST" --teacher-dir "$TEACHER" \
    --output-dir "$WORK/$arm" --arm "$arm" --outer-fold 0 \
    --device cuda --batch-size 96
done
mv "$WORK" "$FINAL"
echo "=== BioAware B5 candidate-level gradient gate completed: $(date) ==="
