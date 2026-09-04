#!/bin/bash
#SBATCH --job-name=noise_e16_replay
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/noise_e16_replay_%j.out
#SBATCH --error=data/validation/noise_e16_replay_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

INITIAL="${INITIAL:-data/validation/g8r_noise_final_e4a_direct/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_highlr_multifold/seed_20260830/fold_4/final_shared_encoder.pt}"
if [[ -z "${SPLIT_DIR:-}" ]]; then
  SPLIT_DIR="$(find data/validation/g8r_noise_final_e15_m3_identity_split -mindepth 2 -maxdepth 2 -name report.json -printf '%T@ %h\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
fi
OUTPUT="data/validation/g8r_noise_final_e16_mature_action_replay/run_${SLURM_JOB_ID}"
[[ -f "$SPLIT_DIR/report.json" ]] || { echo "Missing passing M3 split" >&2; exit 3; }
[[ -f "$INITIAL" ]] || { echo "Missing mature E4-A initialization" >&2; exit 4; }
[[ ! -e "$OUTPUT" ]] || { echo "Unexpected E16 output collision" >&2; exit 5; }
echo "=== E16 mature action replay started: $(date) ==="
python -u tasks/test_noise_final_e16_mature_action_replay.py
python -u tasks/build_noise_final_e16_mature_action_replay.py --split-dir "$SPLIT_DIR" \
  --initial-student-checkpoint "$INITIAL" --output-dir "$OUTPUT"
echo "=== E16 mature action replay completed: $(date) ==="
