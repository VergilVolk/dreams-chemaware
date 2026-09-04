#!/bin/bash
#SBATCH --job-name=noise_c2a_dir
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=data/validation/noise_c2a_dir_%j.out
#SBATCH --error=data/validation/noise_c2a_dir_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
OUTPUT="data/validation/g8r_noise_v3_c2a_token_direction"
SMOKE="${SLURM_TMPDIR:-/tmp}/g8r_noise_c2a_smoke_${SLURM_JOB_ID}"
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite $OUTPUT" >&2; exit 2; }
for path in data/validation/g8r_error_atlas_listwise_cache.npz data/validation/g8r_p2_official_embeddings.npz \
 data/validation/g8r_noise_v3_c1_crossfit_teacher/crossfit_examples.csv.gz \
 data/validation/g8r_noise_v3_c2_peak_tokens/report.json data/validation/g8r_noise_v3_c2_peak_tokens/tokens_f16.npy
do [[ -f "$path" ]] || { echo "Missing C2-A dependency: $path" >&2; exit 3; }; done
echo "=== Noise-v3 C2-A peak-token direction expert started: $(date) ==="
python -u tasks/test_noise_v3_c2a_token_direction.py
python -u tasks/train_noise_v3_c2a_token_direction_expert.py --output-dir "$SMOKE" \
  --max-queries 300 --seeds 20260825 --epochs 2 --bootstrap-resamples 200 --minimum-net-corrections 1
python -u tasks/validate_noise_v3_c2a_token_direction.py --output-dir "$SMOKE" --allow-smoke
python -u tasks/train_noise_v3_c2a_token_direction_expert.py --output-dir "$OUTPUT"
python -u tasks/validate_noise_v3_c2a_token_direction.py --output-dir "$OUTPUT"
echo "=== Noise-v3 C2-A peak-token direction expert completed: $(date) ==="
