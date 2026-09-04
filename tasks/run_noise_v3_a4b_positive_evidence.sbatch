#!/bin/bash
#SBATCH --job-name=noise_a4b_pos
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/noise_a4b_pos_%j.out
#SBATCH --error=data/validation/noise_a4b_pos_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

OUTPUT="data/validation/g8r_noise_v3_a4b_positive_evidence"
SMOKE="${SLURM_TMPDIR:-/tmp}/g8r_a4b_positive_smoke_${SLURM_JOB_ID}"

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite formal A4-B0 output: $OUTPUT" >&2
  exit 2
fi
for path in \
  data/validation/g8r_error_atlas_listwise_cache.npz \
  data/validation/g8r_p2_official_embeddings.npz \
  data/models/MassSpecGym_MurckoHist_split.hdf5 \
  data/validation/g8r_noise_v3_a4_exact_peak_scan/scan_queries.csv.gz \
  data/validation/g8r_noise_v3_a4_exact_peak_scan/exact_peak_scan.h5
do
  if [[ ! -f "$path" ]]; then
    echo "Missing locked A4-B0 dependency: $path" >&2
    exit 3
  fi
done

echo "=== Noise-v3 A4-B0 positive-evidence diagnostic started: $(date) ==="
python -u tasks/test_noise_v3_a4b_positive_evidence.py

echo "=== smoke: balanced 200-query geometry audit ==="
python -u tasks/diagnose_noise_v3_a4b_positive_evidence.py \
  --output-dir "$SMOKE" --max-queries 200 --bootstrap-resamples 200 \
  --alphas 0.10 0.25 0.50 --primary-alpha 0.25 --random-repeats 3
python -u tasks/validate_noise_v3_a4b_positive_evidence.py \
  --output-dir "$SMOKE" --allow-smoke

echo "=== formal: full 4,998-query A4 panel ==="
python -u tasks/diagnose_noise_v3_a4b_positive_evidence.py \
  --output-dir "$OUTPUT" --bootstrap-resamples 5000 \
  --alphas 0.10 0.25 0.50 --primary-alpha 0.25 --random-repeats 3 \
  --minimum-new-corrections 80
python -u tasks/validate_noise_v3_a4b_positive_evidence.py --output-dir "$OUTPUT"
echo "=== Noise-v3 A4-B0 positive-evidence diagnostic completed: $(date) ==="
