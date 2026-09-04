#!/bin/bash
#SBATCH --job-name=noise_a4b_b1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/noise_a4b_b1_%j.out
#SBATCH --error=data/validation/noise_a4b_b1_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

OUTPUT="data/validation/g8r_noise_v3_a4b_rescue_adapter"
SMOKE="${SLURM_TMPDIR:-/tmp}/g8r_a4b_b1_smoke_${SLURM_JOB_ID}"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite formal A4-B1 output: $OUTPUT" >&2
  exit 2
fi
for path in \
  data/validation/g8r_error_atlas_listwise_cache.npz \
  data/validation/g8r_p2_official_embeddings.npz \
  data/models/MassSpecGym_MurckoHist_split.hdf5 \
  data/validation/g8r_noise_v3_a4b_positive_evidence/decision.json \
  data/validation/g8r_noise_v3_a4b_positive_evidence/paired_results.csv.gz
do
  if [[ ! -f "$path" ]]; then
    echo "Missing locked A4-B1 dependency: $path" >&2
    exit 3
  fi
done

echo "=== Noise-v3 A4-B1 rescue adapter started: $(date) ==="
python -u tasks/test_noise_v3_a4b_rescue_adapter.py

echo "=== smoke: 300 queries, two formula folds ==="
python -u tasks/train_noise_v3_a4b_rescue_adapter.py \
  --output-dir "$SMOKE" --max-queries 300 --folds 2 \
  --seeds 20260825 --epochs 2 --bootstrap-resamples 200
python -u tasks/validate_noise_v3_a4b_rescue_adapter.py \
  --output-dir "$SMOKE" --allow-smoke

echo "=== formal: 4,998-query formula-group OOF ==="
python -u tasks/train_noise_v3_a4b_rescue_adapter.py \
  --output-dir "$OUTPUT" --folds 5 \
  --seeds 20260825 20260826 20260827 --epochs 20 \
  --bootstrap-resamples 5000 --minimum-net-corrections 80
python -u tasks/validate_noise_v3_a4b_rescue_adapter.py --output-dir "$OUTPUT"
echo "=== Noise-v3 A4-B1 rescue adapter completed: $(date) ==="
