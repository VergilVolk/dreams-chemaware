#!/bin/bash
#SBATCH --job-name=noise_final_d0
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/noise_final_d0_%j.out
#SBATCH --error=data/validation/noise_final_d0_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

OUTPUT="data/validation/g8r_noise_final_d0_manifest"
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite immutable D0: $OUTPUT" >&2; exit 2; }
required=(
  data/validation/g8r_error_atlas_listwise_cache.npz
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/validation/g8r_p3_test/p3_lock_summary.json
  data/validation/g8r_p3_test/p3_p2_allowed_training_ik14.json
  data/validation/g8r_noise_v3_c1_crossfit_teacher/crossfit_examples.csv.gz
  data/validation/g8r_noise_v3_c1_crossfit_teacher/decision.json
  data/validation/g8r_noise_v3_a4_action_teacher/oof_selected_actions.csv.gz
  data/validation/g8r_noise_v3_a4_action_teacher/decision.json
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "Missing D0 dependency: $file" >&2; exit 3; }
done

echo "=== final noise finetuning D0 started: $(date) ==="
python -u tasks/build_noise_final_d0_manifest.py
echo "=== final noise finetuning D0 completed: $(date) ==="

