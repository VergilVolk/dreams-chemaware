#!/bin/bash
#SBATCH --job-name=g8r_error_analysis
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/g8r_error_analysis_%j.out
#SBATCH --error=data/validation/g8r_error_analysis_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

required=(
  data/validation/g8r_real_error_atlas/query_summary.csv.gz
  data/validation/g8r_real_error_atlas/candidate_edges.csv.gz
  data/validation/g8r_real_error_atlas/report.json
  tasks/analyze_g8r_real_error_atlas.py
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "MISSING: $file" >&2; exit 2; }
done
[[ ! -e data/validation/g8r_real_error_analysis ]] || {
  echo "REFUSE: analysis output already exists" >&2
  exit 3
}

python -u tasks/analyze_g8r_real_error_atlas.py
echo "=== real error analysis completed: $(date) ==="
