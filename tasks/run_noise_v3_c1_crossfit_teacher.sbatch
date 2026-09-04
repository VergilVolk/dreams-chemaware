#!/bin/bash
#SBATCH --job-name=noise_c1_xfit
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=data/validation/noise_c1_xfit_%j.out
#SBATCH --error=data/validation/noise_c1_xfit_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
OUTPUT="data/validation/g8r_noise_v3_c1_crossfit_teacher"
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite $OUTPUT" >&2; exit 2; }
for path in \
  data/validation/g8r_error_atlas_listwise_cache.npz \
  data/validation/g8r_p2_official_embeddings.npz
do
  [[ -f "$path" ]] || { echo "Missing C1 dependency: $path" >&2; exit 3; }
done
echo "=== Noise-v3 C1 support-disjoint teacher expansion started: $(date) ==="
python -u tasks/test_noise_v3_c1_crossfit_teacher.py
python -u tasks/build_noise_v3_c1_crossfit_teacher_space.py --output-dir "$OUTPUT"
python -u tasks/validate_noise_v3_c1_crossfit_teacher.py --output-dir "$OUTPUT"
echo "=== Noise-v3 C1 support-disjoint teacher expansion completed: $(date) ==="
