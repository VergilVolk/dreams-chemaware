#!/bin/bash
#SBATCH --job-name=bio_b5_teacher
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=bioaware_b5_teacher_%j.out
#SBATCH --error=bioaware_b5_teacher_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

FINAL=data/validation/bioaware_b5_candidate_teacher_v1_20260905
WORK="${FINAL}.partial.${SLURM_JOB_ID}"
test ! -e "$FINAL" || { echo "Refusing to overwrite: $FINAL" >&2; exit 3; }
test ! -e "$WORK" || { echo "Refusing to overwrite: $WORK" >&2; exit 3; }

required=(
  data/validation/bioaware_b4_direct_shared_embedding_v1_20260905/manifest/manifest.npz
  data/validation/bioaware_metdna3_external_negative_loso_ranker_v4_chemically_filtered/candidate_features.csv.gz
  tasks/build_bioaware_b5_candidate_teacher.py
  tasks/test_bioaware_b5_candidate_teacher.py
)
for path in "${required[@]}"; do
  test -s "$path" || { echo "Missing dependency: $path" >&2; exit 2; }
done

python -c "import numpy,pandas,scipy,sklearn; print('[BioAware B5 dependencies] PASS')"
python -m py_compile tasks/build_bioaware_b5_candidate_teacher.py tasks/test_bioaware_b5_candidate_teacher.py
python -u tasks/test_bioaware_b5_candidate_teacher.py
python -u tasks/build_bioaware_b5_candidate_teacher.py --output-dir "$WORK"
test -s "$WORK/report.json" || { echo "Missing B5 teacher report" >&2; exit 4; }
mv "$WORK" "$FINAL"
echo "=== BioAware B5 candidate teacher completed: $(date) ==="
echo "Result: $FINAL/report.json"
