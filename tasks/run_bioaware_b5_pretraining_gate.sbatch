#!/bin/bash
#SBATCH --job-name=bio_b5_gate
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=bioaware_b5_gate_%j.out
#SBATCH --error=bioaware_b5_gate_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

MANIFEST=data/validation/bioaware_b4_direct_shared_embedding_v1_20260905/manifest/manifest.npz
FEATURES=data/validation/bioaware_metdna3_external_negative_loso_ranker_v4_chemically_filtered/candidate_features.csv.gz
FINAL=data/validation/bioaware_b5_pretraining_gate_v2_20260905_${SLURM_JOB_ID}
WORK="${FINAL}.partial.${SLURM_JOB_ID}"
TEACHER="$WORK/teacher"
test ! -e "$FINAL" || { echo "Refusing to overwrite: $FINAL" >&2; exit 3; }
test ! -e "$WORK" || { echo "Refusing to overwrite: $WORK" >&2; exit 3; }

required=(
  "$MANIFEST" "$FEATURES"
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
  tasks/build_bioaware_b5_candidate_teacher.py
  tasks/audit_bioaware_b5_gradient_injection.py
  tasks/test_bioaware_b5_candidate_teacher.py
  tasks/test_bioaware_b5_gradient_injection.py
)
for path in "${required[@]}"; do
  test -s "$path" || { echo "Missing dependency: $path" >&2; exit 2; }
done

python -c "import numpy,pandas,scipy,sklearn,torch; assert torch.cuda.is_available(); print('[BioAware B5 dependencies] PASS')"
python -m py_compile \
  tasks/build_bioaware_b5_candidate_teacher.py \
  tasks/audit_bioaware_b5_gradient_injection.py \
  tasks/test_bioaware_b5_candidate_teacher.py \
  tasks/test_bioaware_b5_gradient_injection.py
python -u tasks/test_bioaware_b5_candidate_teacher.py
python -u tasks/test_bioaware_b5_gradient_injection.py

mkdir -p "$WORK"
python -u tasks/build_bioaware_b5_candidate_teacher.py \
  --candidate-features "$FEATURES" --b4-manifest "$MANIFEST" \
  --output-dir "$TEACHER"
test -s "$TEACHER/report.json" || { echo "Missing candidate teacher report" >&2; exit 4; }

for arm in full_bioaware_safe full_no_edge_high_recall; do
  python -u tasks/audit_bioaware_b5_gradient_injection.py \
    --b4-manifest "$MANIFEST" --teacher-dir "$TEACHER" \
    --output-dir "$WORK/gradient/$arm" --arm "$arm" --outer-fold 0 \
    --device cuda --batch-size 96
done

test -s "$WORK/gradient/full_bioaware_safe/report.json" || exit 5
test -s "$WORK/gradient/full_no_edge_high_recall/report.json" || exit 5
mv "$WORK" "$FINAL"
echo "=== BioAware B5 candidate-level pretraining gate completed: $(date) ==="
echo "Result: $FINAL"
