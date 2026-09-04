#!/bin/bash
#SBATCH --job-name=noise_r1_teacher
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/noise_r1_teacher_%j.out
#SBATCH --error=data/validation/noise_r1_teacher_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== final noise R0 + R1 teacher started: $(date) ==="
python -u tasks/test_noise_final_r0_faithful_s3a.py
python -u tasks/test_noise_final_r1_privileged_teacher.py

R0=data/validation/g8r_noise_final_r0_faithful_s3a
if [ ! -d "$R0" ]; then
  python -u tasks/build_noise_final_r0_faithful_s3a_manifest.py --output-dir "$R0"
fi
python -u tasks/validate_noise_final_r0_faithful_s3a.py --output-dir "$R0"

R1=data/validation/g8r_noise_final_r1_privileged_teacher
if [ -e "$R1" ]; then
  echo "Refusing to overwrite $R1" >&2
  exit 2
fi
python -u tasks/build_noise_final_r1_privileged_teacher.py \
  --r0-dir "$R0" \
  --s3a-dir data/validation/g8r_noise_v3_s3a_extended_matrix \
  --output-dir "$R1"
python -u tasks/validate_noise_final_r1_privileged_teacher.py --output-dir "$R1"
echo "=== final noise R1 teacher completed: $(date) ==="
