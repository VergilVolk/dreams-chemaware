#!/bin/bash
# Read-only diagnostic: locate any Adam-bearing Phase-A checkpoint on the
# server before deciding between a true native continuation and abandoning
# the continuation framing.  No training, no writes outside SCAN output.
#SBATCH --job-name=chem_adam_scan
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/chem_adam_scan_%j.out
#SBATCH --error=data/validation/chem_adam_scan_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT="data/validation/GLM_chemaware_adam_bearing_scan/run_${SLURM_JOB_ID}"
mkdir -p "$OUT"

mapfile -t CANDIDATES < <(find \
  "data/validation/chemaware_online_role_budget_native/run_2345481" \
  "data/validation/chemaware_max_boundary_remine" \
  "data/validation/chemaware_max_boundary_native" \
  -maxdepth 5 -type f -name '*.ckpt' 2>/dev/null | sort)
echo "GLM_ADAM_SCAN_CANDIDATES: ${#CANDIDATES[@]}"
for path in "${CANDIDATES[@]}"; do echo "  $path"; done

ARGS=()
for path in "${CANDIDATES[@]}"; do ARGS+=(--checkpoint "$path"); done
python -u tasks/GLM_scan_chemaware_adam_bearing_checkpoints.py \
  "${ARGS[@]}" --output "$OUT/scan_report.json"
echo "GLM_ADAM_SCAN_COMPLETE: $OUT/scan_report.json"
