#!/bin/bash
#SBATCH --job-name=a4_action_teacher
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/a4_action_teacher_%j.out
#SBATCH --error=data/validation/a4_action_teacher_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

INPUT="data/validation/g8r_noise_v3_a4_exact_peak_scan"
OUTPUT="data/validation/g8r_noise_v3_a4_action_teacher"
SMOKE="${SLURM_TMPDIR:-/tmp}/g8r_a4_teacher_smoke_${SLURM_JOB_ID}"

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite formal action teacher: $OUTPUT" >&2
  exit 2
fi
for path in "$INPUT/scan_queries.csv.gz" "$INPUT/exact_peak_scan.h5"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing A4 dependency: $path" >&2
    exit 3
  fi
done

echo "=== A4 nonlinear action teacher started: $(date) ==="
python -u tasks/test_noise_v3_a4_action_teacher.py

echo "=== smoke: 200 queries, 2 folds, 1 seed, 1 epoch ==="
python -u tasks/train_noise_v3_a4_nonlinear_action_teacher.py \
  --a4-dir "$INPUT" --output-dir "$SMOKE" --device cuda \
  --folds 2 --seeds 20260825 --epochs 1 --batch-size 4096 --max-queries 200 \
  --bootstrap-resamples 200
python -u tasks/validate_noise_v3_a4_action_teacher.py \
  --output-dir "$SMOKE" --allow-smoke

echo "=== formal: full 4,998-query formula-group OOF ==="
python -u tasks/train_noise_v3_a4_nonlinear_action_teacher.py \
  --a4-dir "$INPUT" --output-dir "$OUTPUT" --device cuda \
  --folds 5 --seeds 20260825 20260826 20260827 --epochs 8 \
  --batch-size 8192 --risk-penalty 2.0 --minimum-new-corrections 80 \
  --bootstrap-resamples 5000
python -u tasks/validate_noise_v3_a4_action_teacher.py --output-dir "$OUTPUT"
echo "=== A4 nonlinear action teacher completed: $(date) ==="
