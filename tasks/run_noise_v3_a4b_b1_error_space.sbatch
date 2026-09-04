#!/bin/bash
#SBATCH --job-name=noise_a4b_c0
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/noise_a4b_c0_%j.out
#SBATCH --error=data/validation/noise_a4b_c0_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
OUTPUT="data/validation/g8r_noise_v3_a4b_b1_error_space"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite B1-C0 output: $OUTPUT" >&2
  exit 2
fi
for path in \
  data/validation/g8r_error_atlas_listwise_cache.npz \
  data/validation/g8r_p2_official_embeddings.npz \
  data/models/MassSpecGym_MurckoHist_split.hdf5 \
  data/validation/g8r_chemaware_g0_rule_cache.npz \
  data/validation/g8r_noise_v3_a4b_positive_evidence/decision.json \
  data/validation/g8r_noise_v3_a4b_positive_evidence/paired_results.csv.gz \
  data/validation/g8r_noise_v3_a4b_rescue_adapter/decision.json \
  data/validation/g8r_noise_v3_a4b_rescue_adapter/oof_queries.csv.gz \
  data/validation/g8r_noise_v3_a4b_rescue_adapter/oof_embeddings.npz
do
  [[ -f "$path" ]] || { echo "Missing B1-C0 dependency: $path" >&2; exit 3; }
done
echo "=== Noise-v3 A4-B1 C0 error-space audit started: $(date) ==="
python -u tasks/test_noise_v3_a4b_b1_error_space.py
python -u tasks/audit_noise_v3_a4b_b1_error_space.py --output-dir "$OUTPUT"
python -u tasks/validate_noise_v3_a4b_b1_error_space.py --output-dir "$OUTPUT"
echo "=== Noise-v3 A4-B1 C0 error-space audit completed: $(date) ==="
